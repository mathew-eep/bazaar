from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None

    class Dataset:  # type: ignore
        pass

from .features import (
    PRICE_COLS,
    build_static_context,
    compute_norm_stats,
    load_event_sensitivity_by_item,
    load_game_state_features,
    load_item_tags,
    load_norm_stats,
    load_price_history,
    normalize_price_block,
)


RARE_MAYORS = {"Derpy", "Jerry"}


class BazaarDataset(Dataset):
    def __init__(
        self,
        db_path: str | Path = "bazaar.db",
        split: str = "train",
        items_csv: str | Path = "top_items.csv",
        graph_dir: str | Path = "graph_output",
        lookback: int = 168,
        horizon: int = 24,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        norm_stats_path: str | Path = "data/norm_stats.json",
        auto_compute_norm_stats: bool = True,
        augment_rare_mayor: bool = True,
        augmentation_noise_std: float = 0.02,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")
        if lookback <= 0 or horizon <= 0:
            raise ValueError("lookback and horizon must be positive")

        self.db_path = str(db_path)
        self.split = split
        self.lookback = lookback
        self.horizon = horizon
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.augment_rare_mayor = augment_rare_mayor and split == "train"
        self.augmentation_noise_std = augmentation_noise_std

        self.item_tags = load_item_tags(items_csv)

        graph_dir = Path(graph_dir)
        self.item_index = json.loads((graph_dir / "item_index.json").read_text(encoding="utf-8"))
        self.dependency_matrix = np.load(graph_dir / "dependency_matrix.npy")

        norm_stats_file = Path(norm_stats_path)
        if not norm_stats_file.exists():
            if not auto_compute_norm_stats:
                raise FileNotFoundError(
                    f"Normalization stats not found at {norm_stats_file}. "
                    "Enable auto_compute_norm_stats or generate the file first."
                )
            compute_norm_stats(
                db_path=self.db_path,
                item_tags=self.item_tags,
                train_frac=self.train_frac,
                out_path=norm_stats_file,
            )

        self.norm_stats = load_norm_stats(norm_stats_file)

        self.game_state_features = load_game_state_features(self.db_path)
        self._kf_cols = [c for c in self.game_state_features.columns if c.startswith("kf_")]

        self.history = load_price_history(self.db_path, self.item_tags)
        self.event_sensitivity_by_item = load_event_sensitivity_by_item(self.db_path, self.item_tags)
        self.static_by_item = {
            tag: build_static_context(
                tag,
                self.dependency_matrix,
                self.item_index,
                event_sensitivity=self.event_sensitivity_by_item.get(tag),
            )
            for tag in self.item_tags
        }

        self.samples: list[tuple[str, int]] = []
        self._build_index()

    def _build_index(self) -> None:
        for tag in self.item_tags:
            df = self.history.get(tag)
            if df is None or len(df) < (self.lookback + self.horizon):
                continue

            n = len(df)
            train_cut = int(n * self.train_frac)
            val_cut = int(n * (self.train_frac + self.val_frac))

            split_start = 0
            split_end = n - 1
            if self.split == "train":
                split_start, split_end = 0, max(0, train_cut - 1)
            elif self.split == "val":
                split_start, split_end = train_cut, max(train_cut, val_cut - 1)
            elif self.split == "test":
                split_start, split_end = val_cut, n - 1

            first_anchor = max(self.lookback - 1, split_start)
            last_anchor = split_end - self.horizon
            if last_anchor < first_anchor:
                continue

            for anchor in range(first_anchor, last_anchor + 1):
                self.samples.append((tag, anchor))

    def __len__(self) -> int:
        return len(self.samples)

    def _merge_known_future(self, item_df: pd.DataFrame, start_idx: int, end_idx: int) -> tuple[np.ndarray, list[str]]:
        # Convert price timestamps to hourly buckets for joining game_state.
        future_ts = item_df.iloc[start_idx : end_idx + 1]["timestamp"].dt.floor("h").reset_index(drop=True)

        gs = self.game_state_features[["timestamp", "mayor"] + self._kf_cols].sort_values("timestamp").reset_index(drop=True)
        merged = pd.merge_asof(
            future_ts.to_frame(name="timestamp"),
            gs,
            on="timestamp",
            direction="backward",
        )

        # If earliest rows still have no state, forward fill from first known row.
        merged[self._kf_cols] = merged[self._kf_cols].ffill().bfill().fillna(0.0)
        merged["mayor"] = merged["mayor"].fillna("")

        return merged[self._kf_cols].to_numpy(dtype=np.float32), merged["mayor"].astype(str).tolist()

    def _to_tensor(self, arr: np.ndarray):
        if torch is None:
            return arr
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        tag, anchor = self.samples[idx]
        df = self.history[tag]
        item_stats = self.norm_stats[tag]

        past_start = anchor - self.lookback + 1
        past_end = anchor
        fut_start = anchor + 1
        fut_end = anchor + self.horizon

        past_raw = df.iloc[past_start : past_end + 1][PRICE_COLS].to_numpy(dtype=np.float32)
        target_raw = df.iloc[fut_start : fut_end + 1][["sell"]].to_numpy(dtype=np.float32)

        past_obs = normalize_price_block(past_raw, item_stats, PRICE_COLS)

        target_stats = item_stats["sell"]
        current_sell_raw = float(df.iloc[past_end]["sell"])
        target_log = np.log1p(np.clip(target_raw[:, 0], a_min=0.0, a_max=None))
        target = ((target_log - target_stats["mean"]) / target_stats["std"]).astype(np.float32)

        future_known, future_mayors = self._merge_known_future(df, fut_start, fut_end)
        static_vec = self.static_by_item[tag].astype(np.float32, copy=True)

        if self.augment_rare_mayor and any(m in RARE_MAYORS for m in future_mayors):
            past_obs = past_obs + np.random.normal(0.0, self.augmentation_noise_std, size=past_obs.shape).astype(np.float32)
            target = target + np.random.normal(0.0, self.augmentation_noise_std, size=target.shape).astype(np.float32)

        sample = {
            "item_tag": tag,
            "past_obs": self._to_tensor(past_obs.astype(np.float32)),
            "future_known": self._to_tensor(future_known.astype(np.float32)),
            "static": self._to_tensor(static_vec.astype(np.float32)),
            "target": self._to_tensor(target.astype(np.float32)),
            "target_mean": self._to_tensor(np.array(target_stats["mean"], dtype=np.float32)),
            "target_std": self._to_tensor(np.array(target_stats["std"], dtype=np.float32)),
            "current_sell_raw": self._to_tensor(np.array(current_sell_raw, dtype=np.float32)),
        }
        return sample
