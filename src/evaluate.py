from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import BazaarDataset
from src.model import BazaarTFT


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def build_test_loader(args: argparse.Namespace) -> tuple[DataLoader, BazaarDataset]:
    allowed_items = _load_allowed_items(args)
    ds = BazaarDataset(
        db_path=args.db,
        split="test",
        items_csv=args.items,
        graph_dir=args.graph,
        lookback=args.lookback,
        horizon=args.horizon,
        norm_stats_path=args.norm_stats,
        auto_compute_norm_stats=True,
        augment_rare_mayor=False,
    )
    _apply_item_filter(ds, allowed_items)
    if len(ds) == 0:
        raise RuntimeError("Test dataset is empty after item filtering. Relax filter thresholds.")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    return loader, ds


def build_val_loader(args: argparse.Namespace) -> tuple[DataLoader, BazaarDataset]:
    allowed_items = _load_allowed_items(args)
    ds = BazaarDataset(
        db_path=args.db,
        split="val",
        items_csv=args.items,
        graph_dir=args.graph,
        lookback=args.lookback,
        horizon=args.horizon,
        norm_stats_path=args.norm_stats,
        auto_compute_norm_stats=True,
        augment_rare_mayor=False,
    )
    _apply_item_filter(ds, allowed_items)
    if len(ds) == 0:
        raise RuntimeError("Validation dataset is empty after item filtering. Relax filter thresholds.")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    return loader, ds


def build_model_from_dataset(ds: BazaarDataset, args: argparse.Namespace, device: torch.device) -> BazaarTFT:
    sample = ds[0]
    model = BazaarTFT(
        n_past_features=int(sample["past_obs"].shape[-1]),
        n_future_features=int(sample["future_known"].shape[-1]),
        n_static_features=int(sample["static"].shape[-1]),
        d_model=args.d_model,
        n_heads=args.n_heads,
        lookback=args.lookback,
        horizon=args.horizon,
        n_quantiles=3,
        dropout=args.dropout,
    ).to(device)
    return model


def load_checkpoint_if_available(model: BazaarTFT, checkpoint: str | Path | None, device: torch.device) -> bool:
    if checkpoint is None:
        return False

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found at {ckpt_path}. Running evaluation with random weights.")
        return False

    state = torch.load(ckpt_path, map_location=device)
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    print(f"Loaded checkpoint: {ckpt_path}")
    return True


def predict_test_set(
    model: BazaarTFT,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, np.ndarray]:
    model.eval()

    preds = []
    targets = []
    means = []
    stds = []
    current_sells = []
    item_tags: list[str] = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break

            past = batch["past_obs"].to(device)
            future = batch["future_known"].to(device)
            static = batch["static"].to(device)

            pred = model(past, future, static)

            preds.append(pred.cpu().numpy())
            targets.append(batch["target"].cpu().numpy())
            means.append(batch["target_mean"].cpu().numpy())
            stds.append(batch["target_std"].cpu().numpy())
            current_sells.append(batch["current_sell_raw"].cpu().numpy())
            item_tags.extend(list(batch["item_tag"]))

    if not preds:
        raise RuntimeError("No test batches available for evaluation.")

    return {
        "pred": np.concatenate(preds, axis=0),
        "target": np.concatenate(targets, axis=0),
        "target_mean": np.concatenate(means, axis=0),
        "target_std": np.concatenate(stds, axis=0),
        "current_sell_raw": np.concatenate(current_sells, axis=0),
        "item_tag": np.asarray(item_tags, dtype=object),
    }


def coverage_calibration(pred: np.ndarray, target: np.ndarray, quantiles: list[float]) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    print("Coverage calibration")
    for i, q in enumerate(quantiles):
        coverage = float((target <= pred[..., i]).mean())
        rows.append((q, coverage))
        print(f"  Q{int(q * 100):02d}: target={q:.3f} actual={coverage:.3f}")
    return rows


def fit_quantile_offsets(pred: np.ndarray, target: np.ndarray, quantiles: list[float]) -> np.ndarray:
    """Fit additive quantile offsets on a calibration split.

    For each quantile q, compute offset c_q = Quantile_q(target - pred_q).
    Then calibrated predictions are pred_q + c_q.
    """
    offsets = np.zeros(len(quantiles), dtype=np.float32)
    for i, q in enumerate(quantiles):
        residual = target - pred[..., i]
        offsets[i] = float(np.quantile(residual.reshape(-1), q))
    return offsets


def apply_quantile_offsets(pred: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    calibrated = pred.copy()
    for i in range(calibrated.shape[-1]):
        calibrated[..., i] = calibrated[..., i] + float(offsets[i])
    return calibrated


def denormalize_sell(norm_values: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    # norm_values: (N, H), means/stds: (N,)
    means2 = means.reshape(-1, 1)
    stds2 = np.maximum(stds.reshape(-1, 1), 1e-8)
    return np.expm1(norm_values * stds2 + means2)


def simulated_pnl(
    pred_norm: np.ndarray,
    target_norm: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    current_sell_raw: np.ndarray,
    fee_rate: float = 0.01,
    min_margin: float = 0.05,
) -> dict[str, float]:
    # Use horizon end values for decision.
    p10_end_norm = pred_norm[:, -1, 0]
    target_end_norm = target_norm[:, -1]

    p10_end = denormalize_sell(p10_end_norm.reshape(-1, 1), means, stds)[:, 0]
    actual_end = denormalize_sell(target_end_norm.reshape(-1, 1), means, stds)[:, 0]

    entry = current_sell_raw * (1.0 + fee_rate)
    min_exit = entry * (1.0 + min_margin + fee_rate)

    signal = p10_end > min_exit
    actual_exit = actual_end * (1.0 - fee_rate)

    profit = np.where(signal, actual_exit - entry, 0.0)
    trade_profit = profit[signal]
    trade_entry = entry[signal]

    n_trades = int(signal.sum())
    total_profit = float(profit.sum())
    avg_profit = float(total_profit / n_trades) if n_trades > 0 else 0.0

    win_rate = float((trade_profit > 0).mean()) if n_trades > 0 else 0.0
    median_profit = float(np.median(trade_profit)) if n_trades > 0 else 0.0
    avg_return_pct = float((trade_profit / np.maximum(trade_entry, 1e-8)).mean()) if n_trades > 0 else 0.0
    gross_profit = float(trade_profit[trade_profit > 0].sum()) if n_trades > 0 else 0.0
    gross_loss = float(-trade_profit[trade_profit < 0].sum()) if n_trades > 0 else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 1e-9 else (float("inf") if gross_profit > 0 else 0.0)

    if n_trades > 0:
        equity = np.cumsum(trade_profit)
        peaks = np.maximum.accumulate(equity)
        drawdowns = peaks - equity
        max_drawdown = float(drawdowns.max())
    else:
        max_drawdown = 0.0

    print("Simulated P&L")
    print(f"  trades_taken={n_trades}")
    print(f"  total_profit={total_profit:,.2f}")
    print(f"  avg_profit_per_trade={avg_profit:,.2f}")
    print(f"  win_rate={win_rate:.3f}")
    print(f"  median_profit={median_profit:,.2f}")
    print(f"  avg_return_pct={avg_return_pct:.4f}")
    print(f"  profit_factor={profit_factor:.3f}")
    print(f"  max_drawdown={max_drawdown:,.2f}")

    return {
        "trades_taken": float(n_trades),
        "total_profit": total_profit,
        "avg_profit_per_trade": avg_profit,
        "win_rate": win_rate,
        "median_profit": median_profit,
        "avg_return_pct": avg_return_pct,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
    }


def item_level_metrics(pred: np.ndarray, target: np.ndarray, item_tags: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    uniq = pd.Series(item_tags).dropna().unique().tolist()
    for tag in uniq:
        mask = item_tags == tag
        if not np.any(mask):
            continue
        p = pred[mask]
        t = target[mask]
        cov10 = float((t <= p[..., 0]).mean())
        cov50 = float((t <= p[..., 1]).mean())
        cov90 = float((t <= p[..., 2]).mean())
        mae50 = float(np.abs(t - p[..., 1]).mean())
        rows.append(
            {
                "item_tag": str(tag),
                "n_windows": int(mask.sum()),
                "q10_coverage": cov10,
                "q50_coverage": cov50,
                "q90_coverage": cov90,
                "q50_mae_norm": mae50,
            }
        )
    return pd.DataFrame(rows).sort_values("q50_mae_norm", ascending=False)


def build_anomaly_features(db_path: str | Path) -> pd.DataFrame:
    import sqlite3

    con = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query(
            """
            SELECT item_tag, timestamp, sell, buy, sell_volume
            FROM price_history
            ORDER BY item_tag, timestamp
            """,
            con,
        )
    finally:
        con.close()

    if df.empty:
        raise ValueError("price_history is empty; cannot fit anomaly detector")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "sell", "buy", "sell_volume"]).copy()

    parts = []
    for _, g in df.groupby("item_tag", sort=False):
        g = g.sort_values("timestamp").copy()
        g["price_vel"] = g["sell"].pct_change().abs()
        g["vol_price"] = g["sell_volume"] / g["sell"].clip(lower=1.0)
        g["spread_pct"] = (g["buy"] - g["sell"]) / g["sell"].clip(lower=1.0)
        parts.append(g[["price_vel", "vol_price", "spread_pct"]])

    feats = pd.concat(parts, axis=0, ignore_index=True)
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna()
    return feats


def train_isolation_forest(
    db_path: str | Path,
    contamination: float,
    out_path: str | Path,
) -> None:
    try:
        from sklearn.ensemble import IsolationForest
        import joblib
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn and joblib are required for anomaly training. "
            "Install with: pip install scikit-learn joblib"
        ) from exc

    feats = build_anomaly_features(db_path)
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(feats)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)

    preds = model.predict(feats)
    anomaly_rate = float((preds == -1).mean())

    print("IsolationForest")
    print(f"  trained_rows={len(feats):,}")
    print(f"  anomaly_rate={anomaly_rate:.4f}")
    print(f"  saved_to={out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 7 evaluation + anomaly tooling")
    parser.add_argument("--db", default="bazaar.db")
    parser.add_argument("--items", default="top_items.csv")
    parser.add_argument("--graph", default="graph_output")
    parser.add_argument("--norm-stats", default="data/norm_stats.json")
    parser.add_argument("--checkpoint", default=None)

    parser.add_argument("--lookback", type=int, default=168)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--fee-rate", type=float, default=0.01)
    parser.add_argument("--min-margin", type=float, default=0.05)
    parser.add_argument("--calibrate-quantiles", action="store_true")
    parser.add_argument("--item-metrics-out", default="data/item_metrics.csv")

    parser.add_argument("--contamination", type=float, default=0.02)
    parser.add_argument("--isolation-out", default="checkpoints/isolation_forest.pkl")

    parser.add_argument("--skip-model-eval", action="store_true")
    parser.add_argument("--skip-isolation", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)

    parser.add_argument("--item-metrics-filter", default=None)
    parser.add_argument("--filter-min-windows", type=int, default=20)
    parser.add_argument("--filter-max-q50-mae-norm", type=float, default=0.5)
    parser.add_argument("--filter-coverage-eps", type=float, default=0.02)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()

    if not args.skip_model_eval:
        test_loader, test_ds = build_test_loader(args)
        model = build_model_from_dataset(test_ds, args, device)
        load_checkpoint_if_available(model, args.checkpoint, device)

        outputs = predict_test_set(model, test_loader, device, max_batches=args.max_batches)

        quantiles = [0.1, 0.5, 0.9]
        pred_eval = outputs["pred"]

        if args.calibrate_quantiles:
            val_loader, _ = build_val_loader(args)
            val_outputs = predict_test_set(model, val_loader, device, max_batches=args.max_batches)
            offsets = fit_quantile_offsets(val_outputs["pred"], val_outputs["target"], quantiles)
            pred_eval = apply_quantile_offsets(pred_eval, offsets)
            print("Post-hoc quantile calibration")
            print(f"  offsets={offsets.tolist()}")

        coverage_calibration(pred_eval, outputs["target"], quantiles)
        simulated_pnl(
            pred_eval,
            outputs["target"],
            outputs["target_mean"],
            outputs["target_std"],
            outputs["current_sell_raw"],
            fee_rate=args.fee_rate,
            min_margin=args.min_margin,
        )

        item_metrics = item_level_metrics(pred_eval, outputs["target"], outputs["item_tag"])
        out_path = Path(args.item_metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        item_metrics.to_csv(out_path, index=False)
        print("Item-level metrics")
        print(f"  rows={len(item_metrics):,}")
        print(f"  saved_to={out_path}")

    if not args.skip_isolation:
        train_isolation_forest(args.db, contamination=args.contamination, out_path=args.isolation_out)


if __name__ == "__main__":
    main()
