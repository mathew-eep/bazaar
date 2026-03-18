from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import BazaarDataset
from src.model import BazaarTFT
from src.recency_aware import get_recency_weighted_sampler, RegimeShiftDetector


QUANTILES = [0.1, 0.5, 0.9]


def _load_allowed_items(args: argparse.Namespace) -> set[str] | None:
    if not args.item_metrics_filter:
        return None

    metrics_path = Path(args.item_metrics_filter)
    if not metrics_path.exists():
        print(f"Item metrics filter file not found at {metrics_path}. Skipping item filter.")
        return None

    df = pd.read_csv(metrics_path)
    required = {"item_tag", "n_windows", "q10_coverage", "q50_coverage", "q90_coverage", "q50_mae_norm"}
    if not required.issubset(df.columns):
        print(f"Item metrics file {metrics_path} missing required columns. Skipping item filter.")
        return None

    filtered = df.copy()
    filtered = filtered[filtered["n_windows"] >= args.filter_min_windows]

    if args.filter_max_q50_mae_norm is not None:
        filtered = filtered[filtered["q50_mae_norm"] <= args.filter_max_q50_mae_norm]

    eps = float(args.filter_coverage_eps)
    filtered = filtered[
        (filtered["q10_coverage"].between(eps, 1.0 - eps))
        & (filtered["q50_coverage"].between(eps, 1.0 - eps))
        & (filtered["q90_coverage"].between(eps, 1.0 - eps))
    ]

    allowed = set(filtered["item_tag"].astype(str).tolist())
    print(
        f"Item filter active: kept {len(allowed)} items from {len(df)} "
        f"(min_windows={args.filter_min_windows}, max_mae={args.filter_max_q50_mae_norm}, coverage_eps={eps})"
    )
    return allowed


def _apply_item_filter(ds: BazaarDataset, allowed_items: set[str] | None) -> None:
    if not allowed_items:
        return

    before = len(ds.samples)
    ds.item_tags = [tag for tag in ds.item_tags if tag in allowed_items]
    ds.samples = [(tag, anchor) for (tag, anchor) in ds.samples if tag in allowed_items]
    after = len(ds.samples)
    print(f"Filtered dataset[{ds.split}] samples: {before} -> {after}")


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: list[float] = QUANTILES) -> torch.Tensor:
    # pred: (B, H, Q), target: (B, H)
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
    target = torch.nan_to_num(target, nan=0.0, posinf=1e6, neginf=-1e6)
    q = torch.tensor(quantiles, device=pred.device, dtype=pred.dtype).view(1, 1, -1)
    target_expanded = target.unsqueeze(-1).expand_as(pred)
    err = target_expanded - pred
    return torch.max(q * err, (q - 1.0) * err).mean()


def quantile_crossing_penalty(pred: torch.Tensor) -> torch.Tensor:
    # Penalize invalid ordering: q10 <= q50 <= q90
    q10 = pred[..., 0]
    q50 = pred[..., 1]
    q90 = pred[..., 2]
    p1 = torch.relu(q10 - q50)
    p2 = torch.relu(q50 - q90)
    return (p1 + p2).mean()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_dataloaders(args: argparse.Namespace, use_recency_bias: bool = True) -> tuple[DataLoader, list[DataLoader], BazaarDataset]:
    allowed_items = _load_allowed_items(args)

    train_ds = BazaarDataset(
        db_path=args.db,
        split="train",
        items_csv=args.items,
        graph_dir=args.graph,
        lookback=args.lookback,
        horizon=args.horizon,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        norm_stats_path=args.norm_stats,
        auto_compute_norm_stats=True,
        augment_rare_mayor=True,
    )
    _apply_item_filter(train_ds, allowed_items)
    if len(train_ds) == 0:
        raise RuntimeError("Training dataset is empty after item filtering. Relax filter thresholds.")

    # Use recency-weighted sampler if enabled (biases toward recent data)
    if use_recency_bias:
        sampler = get_recency_weighted_sampler(
            train_ds,
            db_path=args.db,
            decay_rate=0.95,
            recent_months=12,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    n_windows = max(1, int(args.walk_forward_val_windows))
    if n_windows == 1:
        train_fracs = [float(args.train_frac)]
    else:
        start_frac = max(0.5, float(args.train_frac) - 0.2)
        train_fracs = np.linspace(start_frac, float(args.train_frac), n_windows).tolist()

    val_loaders: list[DataLoader] = []
    for tf in train_fracs:
        val_ds = BazaarDataset(
            db_path=args.db,
            split="val",
            items_csv=args.items,
            graph_dir=args.graph,
            lookback=args.lookback,
            horizon=args.horizon,
            train_frac=float(tf),
            val_frac=args.val_frac,
            norm_stats_path=args.norm_stats,
            auto_compute_norm_stats=False,
            augment_rare_mayor=False,
        )
        _apply_item_filter(val_ds, allowed_items)
        if len(val_ds) == 0:
            raise RuntimeError("Validation dataset is empty after item filtering. Relax filter thresholds.")
        val_loaders.append(DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0))

    return train_loader, val_loaders, train_ds


def build_model(train_ds: BazaarDataset, args: argparse.Namespace, device: torch.device) -> BazaarTFT:
    sample = train_ds[0]
    n_past = int(sample["past_obs"].shape[-1])
    n_future = int(sample["future_known"].shape[-1])
    n_static = int(sample["static"].shape[-1])

    n_ensemble = getattr(args, "ensemble_size", 1)
    models = []
    for i in range(n_ensemble):
        model = BazaarTFT(
            n_past_features=n_past,
            n_future_features=n_future,
            n_static_features=n_static,
            d_model=args.d_model,
            n_heads=args.n_heads,
            lookback=args.lookback,
            horizon=args.horizon,
            n_quantiles=3,
            dropout=args.dropout,
        ).to(device)
        models.append(model)
    if n_ensemble == 1:
        return models[0]
    return models


def maybe_autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    # CPU path: no autocast context
    class _NullCtx:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    return _NullCtx()


def train_one_epoch(
    model: BazaarTFT,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    crossing_weight: float,
) -> float:
    model.train()
    total = 0.0

    for batch in loader:
        past = batch["past_obs"].to(device)
        future = batch["future_known"].to(device)
        static = batch["static"].to(device)
        target = batch["target"].to(device)

        opt.zero_grad(set_to_none=True)
        with maybe_autocast(device):
            pred = model(past, future, static)
            pbl = pinball_loss(pred, target)
            qcp = quantile_crossing_penalty(pred)
            loss = pbl + crossing_weight * qcp

        if not torch.isfinite(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total += float(loss.detach().cpu())

    return total / max(1, len(loader))


def evaluate(model: BazaarTFT, loader: DataLoader, device: torch.device, crossing_weight: float) -> float:
    model.eval()
    total = 0.0

    with torch.no_grad():
        for batch in loader:
            past = batch["past_obs"].to(device)
            future = batch["future_known"].to(device)
            static = batch["static"].to(device)
            target = batch["target"].to(device)

            with maybe_autocast(device):
                pred = model(past, future, static)
                pbl = pinball_loss(pred, target)
                qcp = quantile_crossing_penalty(pred)
                loss = pbl + crossing_weight * qcp

            if not torch.isfinite(loss):
                continue

            total += float(loss.detach().cpu())

    return total / max(1, len(loader))


def evaluate_walk_forward(
    model: BazaarTFT,
    loaders: list[DataLoader],
    device: torch.device,
    crossing_weight: float,
) -> tuple[float, list[float]]:
    n_mc = getattr(model, "mc_dropout", 5) if hasattr(model, "mc_dropout") else 5
    vals = []
    for ld in loaders:
        mc_vals = []
        for _ in range(n_mc):
            model.train()  # Enable dropout for MC
            mc_vals.append(evaluate(model, ld, device, crossing_weight=crossing_weight))
        vals.append(np.mean(mc_vals))
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return float("inf"), []
    return float(np.mean(vals)), vals


def run_training(args: argparse.Namespace) -> None:
    device = get_device()
    train_loader, val_loaders, train_ds = build_dataloaders(args, use_recency_bias=args.recency_bias)
    model = build_model(train_ds, args, device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=args.lr_plateau_factor,
        patience=args.lr_plateau_patience,
        min_lr=args.lr_min,
    )

    ckpt_dir = Path(args.checkpoints)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"

    best_score = float("inf")
    start_epoch = 0

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        state = torch.load(resume_path, map_location=device)
        model_state = state.get("model_state", state)
        model.load_state_dict(model_state)
        if "optimizer_state" in state:
            opt.load_state_dict(state["optimizer_state"])
        start_epoch = int(state.get("epoch", -1)) + 1
        if "score" in state:
            best_score = float(state["score"])
        elif "val_loss" in state:
            best_score = float(state["val_loss"])
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    regime_detector = RegimeShiftDetector(db_path=args.db, threshold_std=2.0)
    regime_detector.compute_baseline()
    epochs_without_improvement = 0

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, train_loader, opt, device, crossing_weight=args.crossing_weight)
        val_loss, val_parts = evaluate_walk_forward(model, val_loaders, device, crossing_weight=args.crossing_weight)
        val_worst = max(val_parts) if val_parts else val_loss
        score = args.objective_mean_weight * val_loss + args.objective_worst_weight * val_worst
        sched.step(score)

        current_lr = opt.param_groups[0]["lr"]

        if val_parts:
            parts = ",".join(f"{v:.5f}" for v in val_parts)
            print(
                f"Epoch {epoch + 1:03d} | train={train_loss:.5f} | val_mean={val_loss:.5f} "
                f"| val_worst={val_worst:.5f} | score={score:.5f} | lr={current_lr:.2e} "
                f"| val_windows=[{parts}]"
            )
        else:
            print(
                f"Epoch {epoch + 1:03d} | train={train_loss:.5f} | val_mean={val_loss:.5f} "
                f"| val_worst={val_worst:.5f} | score={score:.5f} | lr={current_lr:.2e}"
            )

        if score < (best_score - args.early_stop_min_delta):
            best_score = score
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "val_loss": val_loss,
                    "val_worst": val_worst,
                    "score": best_score,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"  saved checkpoint: {best_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch + 1:03d}. "
                f"No score improvement for {epochs_without_improvement} epochs. "
                f"Best score={best_score:.5f}"
            )
            break
        
        # Detect regime shifts periodically (every 5 epochs)
        if (epoch + 1) % args.regime_check_every == 0:
            shift_info = regime_detector.detect_shift()
            if shift_info["should_retrain"]:
                print(f"  ⚠️  Regime shift detected! median_z={shift_info['median_z_score']:.2f}, "
                      f"pct_shifted={shift_info['pct_items_shifted']:.1%}")
                print(f"  💡 Consider retraining from best checkpoint with --recency-bias")


def run_forward_smoke(args: argparse.Namespace) -> None:
    device = get_device()
    train_loader, _, train_ds = build_dataloaders(args, use_recency_bias=False)
    model = build_model(train_ds, args, device)

    batch = next(iter(train_loader))
    past = batch["past_obs"].to(device)
    future = batch["future_known"].to(device)
    static = batch["static"].to(device)

    with torch.no_grad():
        pred = model(past, future, static)

    print("forward-only smoke test")
    print("  past:", tuple(past.shape))
    print("  future:", tuple(future.shape))
    print("  static:", tuple(static.shape))
    print("  pred:", tuple(pred.shape))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6 training loop for BazaarTFT")
    parser.add_argument("--db", default="bazaar.db")
    parser.add_argument("--items", default="top_items.csv")
    parser.add_argument("--graph", default="graph_output")
    parser.add_argument("--norm-stats", default="data/norm_stats.json")
    parser.add_argument("--checkpoints", default="checkpoints")

    parser.add_argument("--lookback", type=int, default=168)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--crossing-weight", type=float, default=0.02)

    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--walk-forward-val-windows", type=int, default=3)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume/fine-tune from.")
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)

    parser.add_argument("--objective-mean-weight", type=float, default=0.7)
    parser.add_argument("--objective-worst-weight", type=float, default=0.3)
    parser.add_argument("--ensemble-size", type=int, default=1, help="Number of models in ensemble.")
    parser.add_argument("--mc-dropout", type=int, default=5, help="Monte Carlo dropout passes during evaluation.")

    parser.add_argument("--lr-plateau-patience", type=int, default=4)
    parser.add_argument("--lr-plateau-factor", type=float, default=0.5)
    parser.add_argument("--lr-min", type=float, default=1e-6)

    parser.add_argument("--item-metrics-filter", default=None)
    parser.add_argument("--filter-min-windows", type=int, default=20)
    parser.add_argument("--filter-max-q50-mae-norm", type=float, default=0.5)
    parser.add_argument("--filter-coverage-eps", type=float, default=0.02)
    parser.add_argument("--regime-check-every", type=int, default=5)

    parser.add_argument(
        "--recency-bias",
        action="store_true",
        default=True,
        help="Use recency-weighted sampling to bias toward recent data (default: True).",
    )
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Run one forward pass and exit (no training).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.forward_only:
        run_forward_smoke(args)
        return
    run_training(args)


if __name__ == "__main__":
    main()
