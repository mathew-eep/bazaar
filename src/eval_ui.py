from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluate import (
    apply_quantile_offsets,
    build_model_from_dataset,
    build_test_loader,
    build_val_loader,
    coverage_calibration,
    fit_quantile_offsets,
    get_device,
    item_level_metrics,
    load_checkpoint_if_available,
    predict_test_set,
    simulated_pnl,
    train_isolation_forest,
)


def _make_args(
    db: str,
    items: str,
    graph: str,
    norm_stats: str,
    checkpoint: str | None,
    lookback: int,
    horizon: int,
    batch_size: int,
    d_model: int,
    n_heads: int,
    dropout: float,
    fee_rate: float,
    min_margin: float,
    max_batches: int | None,
    item_metrics_filter: str | None,
    filter_min_windows: int,
    filter_max_q50_mae_norm: float,
    filter_coverage_eps: float,
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
        fee_rate=fee_rate,
        min_margin=min_margin,
        max_batches=max_batches,
        item_metrics_filter=item_metrics_filter,
        filter_min_windows=filter_min_windows,
        filter_max_q50_mae_norm=filter_max_q50_mae_norm,
        filter_coverage_eps=filter_coverage_eps,
    )


def _available_checkpoints() -> list[str]:
    ckpt_dir = Path("checkpoints")
    if not ckpt_dir.exists():
        return []
    return sorted([str(p) for p in ckpt_dir.glob("*.pt")])


def _run_model_eval(args: argparse.Namespace, calibrate_quantiles: bool) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    device = get_device()
    test_loader, test_ds = build_test_loader(args)

    model = build_model_from_dataset(test_ds, args, device)
    load_checkpoint_if_available(model, args.checkpoint, device)

    outputs = predict_test_set(model, test_loader, device, max_batches=args.max_batches)

    quantiles = [0.1, 0.5, 0.9]
    pred_eval = outputs["pred"]

    if calibrate_quantiles:
        val_loader, _ = build_val_loader(args)
        val_outputs = predict_test_set(model, val_loader, device, max_batches=args.max_batches)
        offsets = fit_quantile_offsets(val_outputs["pred"], val_outputs["target"], quantiles)
        pred_eval = apply_quantile_offsets(pred_eval, offsets)

    coverage = coverage_calibration(pred_eval, outputs["target"], quantiles)
    coverage_df = pd.DataFrame(coverage, columns=["target_quantile", "actual_coverage"])

    pnl = simulated_pnl(
        pred_eval,
        outputs["target"],
        outputs["target_mean"],
        outputs["target_std"],
        outputs["current_sell_raw"],
        fee_rate=args.fee_rate,
        min_margin=args.min_margin,
    )

    item_metrics_df = item_level_metrics(pred_eval, outputs["target"], outputs["item_tag"])
    return coverage_df, pnl, item_metrics_df


def main() -> None:
    st.set_page_config(page_title="Bazaar Checkpoint Evaluator", layout="wide")
    st.title("Bazaar Checkpoint Evaluator")

    st.sidebar.header("Inputs")

    db = st.sidebar.text_input("DB path", "bazaar.db")
    items = st.sidebar.text_input("Items CSV", "top_items.csv")
    graph = st.sidebar.text_input("Graph dir", "graph_output")
    norm_stats = st.sidebar.text_input("Norm stats", "data/norm_stats.json")

    ckpts = _available_checkpoints()
    default_ckpt = "checkpoints/3_17-1_31pm.pt" if "checkpoints/3_17-1_31pm.pt" in ckpts else (ckpts[0] if ckpts else "")
    checkpoint = st.sidebar.selectbox("Checkpoint", options=ckpts if ckpts else [""], index=(ckpts.index(default_ckpt) if default_ckpt in ckpts else 0))

    lookback = st.sidebar.number_input("Lookback", min_value=1, value=168)
    horizon = st.sidebar.number_input("Horizon", min_value=1, value=24)
    batch_size = st.sidebar.number_input("Batch size", min_value=1, value=64)
    d_model = st.sidebar.number_input("d_model", min_value=8, value=64, step=8)
    n_heads = st.sidebar.number_input("n_heads", min_value=1, value=4)
    dropout = st.sidebar.slider("Dropout", min_value=0.0, max_value=0.5, value=0.1, step=0.01)

    fee_rate = st.sidebar.slider("Fee rate", min_value=0.0, max_value=0.1, value=0.01, step=0.001)
    min_margin = st.sidebar.slider("Min margin", min_value=0.0, max_value=0.5, value=0.05, step=0.005)

    max_batches_raw = st.sidebar.number_input("Max batches (0 = all)", min_value=0, value=0)
    max_batches = None if max_batches_raw == 0 else int(max_batches_raw)

    calibrate_quantiles = st.sidebar.checkbox("Calibrate quantiles on val split", value=False)
    contamination = st.sidebar.slider("Isolation contamination", min_value=0.001, max_value=0.2, value=0.02, step=0.001)

    enable_item_filter = st.sidebar.checkbox("Enable unstable-item filter", value=False)
    item_metrics_filter = st.sidebar.text_input("Item metrics CSV", "data/item_metrics_ui.csv") if enable_item_filter else None
    filter_min_windows = st.sidebar.number_input("Filter min windows", min_value=1, value=20)
    filter_max_q50_mae_norm = st.sidebar.number_input("Filter max q50 MAE norm", min_value=0.0, value=0.5, step=0.05)
    filter_coverage_eps = st.sidebar.slider("Filter coverage epsilon", min_value=0.0, max_value=0.2, value=0.02, step=0.01)

    args = _make_args(
        db=db,
        items=items,
        graph=graph,
        norm_stats=norm_stats,
        checkpoint=checkpoint if checkpoint else None,
        lookback=int(lookback),
        horizon=int(horizon),
        batch_size=int(batch_size),
        d_model=int(d_model),
        n_heads=int(n_heads),
        dropout=float(dropout),
        fee_rate=float(fee_rate),
        min_margin=float(min_margin),
        max_batches=max_batches,
        item_metrics_filter=item_metrics_filter,
        filter_min_windows=int(filter_min_windows),
        filter_max_q50_mae_norm=float(filter_max_q50_mae_norm),
        filter_coverage_eps=float(filter_coverage_eps),
    )

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Run model evaluation", type="primary", use_container_width=True):
            try:
                with st.spinner("Running evaluation..."):
                    coverage_df, pnl, item_metrics_df = _run_model_eval(args, calibrate_quantiles=calibrate_quantiles)

                st.subheader("Coverage calibration")
                st.dataframe(coverage_df, use_container_width=True)

                st.subheader("Simulated P&L")
                st.json(pnl)

                st.subheader("Worst item metrics (top 30 by q50 MAE)")
                st.dataframe(item_metrics_df.head(30), use_container_width=True)

                out_path = Path("data/item_metrics_ui.csv")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                item_metrics_df.to_csv(out_path, index=False)
                st.success(f"Saved item metrics to {out_path}")
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")

    with col_b:
        if st.button("Train isolation forest", use_container_width=True):
            try:
                out_path = "checkpoints/isolation_forest.pkl"
                with st.spinner("Training isolation forest..."):
                    train_isolation_forest(args.db, contamination=float(contamination), out_path=out_path)
                st.success(f"Saved anomaly model to {out_path}")
            except Exception as exc:
                st.error(f"Isolation training failed: {exc}")

    st.markdown("### CLI equivalents")
    st.code(
        "\n".join(
            [
                "python -m src.evaluate \\",
                f"  --checkpoint {checkpoint if checkpoint else 'checkpoints/3_17-1_31pm.pt'} \\",
                f"  --lookback {int(lookback)} --horizon {int(horizon)} --batch-size {int(batch_size)} \\",
                f"  --fee-rate {float(fee_rate):.3f} --min-margin {float(min_margin):.3f}",
            ]
        ),
        language="bash",
    )


if __name__ == "__main__":
    main()
