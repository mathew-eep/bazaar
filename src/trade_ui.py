from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluate import (  # noqa: E402
    apply_quantile_offsets,
    build_model_from_dataset,
    build_test_loader,
    build_val_loader,
    fit_quantile_offsets,
    get_device,
    load_checkpoint_if_available,
    predict_test_set,
)


def _make_args(
    db: str,
    items: str,
    graph: str,
    norm_stats: str,
    checkpoint: str,
    lookback: int,
    horizon: int,
    batch_size: int,
    d_model: int,
    n_heads: int,
    dropout: float,
    max_batches: int | None,
) -> argparse.Namespace:
    return argparse.Namespace(
        db=db,
        items=items,
        graph=graph,
        norm_stats=norm_stats,
        checkpoint=checkpoint,
        lookback=lookback,
        horizon=horizon,
        batch_size=batch_size,
        d_model=d_model,
        n_heads=n_heads,
        dropout=dropout,
        max_batches=max_batches,
        item_metrics_filter=None,
        filter_min_windows=1,
        filter_max_q50_mae_norm=None,
        filter_coverage_eps=0.0,
    )


def _available_checkpoints() -> list[str]:
    ckpt_dir = Path("checkpoints")
    if not ckpt_dir.exists():
        return []
    return sorted(str(p) for p in ckpt_dir.glob("*.pt"))


def _denormalize(norm_values: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    means2 = means.reshape(-1, 1)
    stds2 = np.maximum(stds.reshape(-1, 1), 1e-8)
    return np.expm1(norm_values * stds2 + means2)


def build_trade_candidates(
    args: argparse.Namespace,
    calibrate_quantiles: bool,
    fee_rate: float,
    min_margin: float,
) -> pd.DataFrame:
    device = get_device()
    test_loader, test_ds = build_test_loader(args)
    model = build_model_from_dataset(test_ds, args, device)
    load_checkpoint_if_available(model, args.checkpoint, device)

    outputs = predict_test_set(model, test_loader, device, max_batches=args.max_batches)

    pred_eval = outputs["pred"]
    quantiles = [0.1, 0.5, 0.9]

    if calibrate_quantiles:
        val_loader, _ = build_val_loader(args)
        val_outputs = predict_test_set(model, val_loader, device, max_batches=args.max_batches)
        offsets = fit_quantile_offsets(val_outputs["pred"], val_outputs["target"], quantiles)
        pred_eval = apply_quantile_offsets(pred_eval, offsets)

    p10_end_norm = pred_eval[:, -1, 0]
    p50_end_norm = pred_eval[:, -1, 1]
    p90_end_norm = pred_eval[:, -1, 2]

    p10_end = _denormalize(p10_end_norm.reshape(-1, 1), outputs["target_mean"], outputs["target_std"])[:, 0]
    p50_end = _denormalize(p50_end_norm.reshape(-1, 1), outputs["target_mean"], outputs["target_std"])[:, 0]
    p90_end = _denormalize(p90_end_norm.reshape(-1, 1), outputs["target_mean"], outputs["target_std"])[:, 0]

    entry = outputs["current_sell_raw"] * (1.0 + fee_rate)
    min_exit = entry * (1.0 + min_margin + fee_rate)

    signal = p10_end > min_exit

    expected_pnl_low = (p10_end * (1.0 - fee_rate)) - entry
    expected_pnl_mid = (p50_end * (1.0 - fee_rate)) - entry
    expected_pnl_high = (p90_end * (1.0 - fee_rate)) - entry

    confidence = (p10_end - min_exit) / np.maximum(min_exit, 1e-8)

    df = pd.DataFrame(
        {
            "item_tag": outputs["item_tag"],
            "current_sell": outputs["current_sell_raw"],
            "entry_with_fee": entry,
            "min_exit_for_trade": min_exit,
            "pred_q10_end": p10_end,
            "pred_q50_end": p50_end,
            "pred_q90_end": p90_end,
            "expected_pnl_low": expected_pnl_low,
            "expected_pnl_mid": expected_pnl_mid,
            "expected_pnl_high": expected_pnl_high,
            "expected_return_mid_pct": expected_pnl_mid / np.maximum(entry, 1e-8),
            "confidence": confidence,
            "trade_signal": signal,
        }
    )

    # Keep most recent candidate per item by sorting by confidence and dropping duplicates.
    # This yields one actionable row per item in the dashboard.
    df = df.sort_values(["item_tag", "confidence"], ascending=[True, False]).drop_duplicates(subset=["item_tag"], keep="first")

    return df.sort_values("confidence", ascending=False).reset_index(drop=True)


def main() -> None:
    st.set_page_config(page_title="Bazaar Trade Dashboard", layout="wide")
    st.title("Bazaar Trade Dashboard")
    st.caption("Visualize model-based trade candidates from checkpoint predictions.")

    st.sidebar.header("Model")
    ckpts = _available_checkpoints()
    if not ckpts:
        st.error("No checkpoints found in checkpoints/. Add a .pt checkpoint first.")
        return

    default_ckpt = "checkpoints/best.pt" if "checkpoints/best.pt" in ckpts else ckpts[0]
    checkpoint = st.sidebar.selectbox("Checkpoint", options=ckpts, index=ckpts.index(default_ckpt))

    st.sidebar.header("Data")
    db = st.sidebar.text_input("DB path", "bazaar.db")
    items = st.sidebar.text_input("Items CSV", "top_items.csv")
    graph = st.sidebar.text_input("Graph dir", "graph_output")
    norm_stats = st.sidebar.text_input("Norm stats", "data/norm_stats.json")

    st.sidebar.header("Inference")
    lookback = st.sidebar.number_input("Lookback", min_value=1, value=168)
    horizon = st.sidebar.number_input("Horizon", min_value=1, value=24)
    batch_size = st.sidebar.number_input("Batch size", min_value=1, value=64)
    d_model = st.sidebar.number_input("d_model", min_value=8, value=64, step=8)
    n_heads = st.sidebar.number_input("n_heads", min_value=1, value=4)
    dropout = st.sidebar.slider("Dropout", min_value=0.0, max_value=0.5, value=0.1, step=0.01)
    max_batches_raw = st.sidebar.number_input("Max batches (0 = all)", min_value=0, value=0)
    max_batches = None if max_batches_raw == 0 else int(max_batches_raw)

    st.sidebar.header("Trading")
    fee_rate = st.sidebar.slider("Fee rate", min_value=0.0, max_value=0.1, value=0.01, step=0.001)
    min_margin = st.sidebar.slider("Min margin", min_value=0.0, max_value=0.3, value=0.10, step=0.005)
    calibrate_quantiles = st.sidebar.checkbox("Calibrate quantiles", value=True)

    args = _make_args(
        db=db,
        items=items,
        graph=graph,
        norm_stats=norm_stats,
        checkpoint=checkpoint,
        lookback=int(lookback),
        horizon=int(horizon),
        batch_size=int(batch_size),
        d_model=int(d_model),
        n_heads=int(n_heads),
        dropout=float(dropout),
        max_batches=max_batches,
    )

    if st.button("Generate Trade Candidates", type="primary", use_container_width=True):
        try:
            with st.spinner("Running model and building signals..."):
                cands = build_trade_candidates(
                    args=args,
                    calibrate_quantiles=calibrate_quantiles,
                    fee_rate=float(fee_rate),
                    min_margin=float(min_margin),
                )

            tradable = cands[cands["trade_signal"]].copy()
            blocked = cands[~cands["trade_signal"]].copy()

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Items considered", f"{len(cands):,}")
            k2.metric("Trade signals", f"{len(tradable):,}")
            k3.metric("Mean expected mid return", f"{100.0 * cands['expected_return_mid_pct'].mean():.2f}%")
            k4.metric("Mean confidence", f"{100.0 * cands['confidence'].mean():.2f}%")

            st.subheader("Top trade candidates")
            if len(tradable) == 0:
                st.warning("No items passed the conservative trade rule (q10 > min_exit). Lower min margin or review model.")
            else:
                show_cols = [
                    "item_tag",
                    "current_sell",
                    "pred_q10_end",
                    "pred_q50_end",
                    "pred_q90_end",
                    "min_exit_for_trade",
                    "expected_pnl_low",
                    "expected_pnl_mid",
                    "expected_return_mid_pct",
                    "confidence",
                ]
                st.dataframe(tradable[show_cols].head(30), use_container_width=True)

            st.subheader("Signal distribution")
            chart_df = cands[["item_tag", "confidence", "trade_signal"]].copy()
            chart_df["signal"] = np.where(chart_df["trade_signal"], "TRADE", "SKIP")
            st.bar_chart(chart_df.set_index("item_tag")["confidence"])

            st.subheader("Blocked items (top 20 by confidence)")
            if len(blocked) > 0:
                st.dataframe(blocked.head(20), use_container_width=True)
            else:
                st.info("All items signaled trade under current thresholds.")

            out_path = Path("data/trade_candidates.csv")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cands.to_csv(out_path, index=False)
            st.success(f"Saved candidates to {out_path}")

        except Exception as exc:
            st.error(f"Failed to generate candidates: {exc}")


if __name__ == "__main__":
    main()
