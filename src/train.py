from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import BazaarDataset
from src.model import BazaarTFT
from src.recency_aware import get_recency_weighted_sampler, RegimeShiftDetector


QUANTILES = [0.1, 0.5, 0.9]


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
        val_loaders.append(DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0))

    return train_loader, val_loaders, train_ds


def build_model(train_ds: BazaarDataset, args: argparse.Namespace, device: torch.device) -> BazaarTFT:
    sample = train_ds[0]
    n_past = int(sample["past_obs"].shape[-1])
    n_future = int(sample["future_known"].shape[-1])
    n_static = int(sample["static"].shape[-1])

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
    return model


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
    vals = [evaluate(model, ld, device, crossing_weight=crossing_weight) for ld in loaders]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return float("inf"), []
    return float(np.mean(vals)), vals


def run_training(args: argparse.Namespace) -> None:
    device = get_device()
    train_loader, val_loaders, train_ds = build_dataloaders(args, use_recency_bias=args.recency_bias)
    model = build_model(train_ds, args, device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    ckpt_dir = Path(args.checkpoints)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"

    best_val = float("inf")
    regime_detector = RegimeShiftDetector(db_path=args.db, threshold_std=2.0)
    regime_detector.compute_baseline()

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, opt, device, crossing_weight=args.crossing_weight)
        val_loss, val_parts = evaluate_walk_forward(model, val_loaders, device, crossing_weight=args.crossing_weight)
        sched.step()

        if val_parts:
            parts = ",".join(f"{v:.5f}" for v in val_parts)
            print(f"Epoch {epoch + 1:03d} | train={train_loss:.5f} | val_mean={val_loss:.5f} | val_windows=[{parts}]")
        else:
            print(f"Epoch {epoch + 1:03d} | train={train_loss:.5f} | val_mean={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "val_loss": best_val,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"  saved checkpoint: {best_path}")
        
        # Detect regime shifts periodically (every 5 epochs)
        if (epoch + 1) % 5 == 0:
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
