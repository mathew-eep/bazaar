from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def compute_recency_weights(
    dataset,
    db_path: str | Path = "bazaar.db",
    decay_rate: float = 0.95,
    recent_months: int = 12,
) -> np.ndarray:
    """
    Compute weights for dataset samples based on recency.
    Recent samples get higher weight to adapt model to recent market regime.
    
    Args:
        dataset: BazaarDataset instance
        db_path: Path to sqlite database
        decay_rate: Weight decay for older samples (higher = faster decay)
        recent_months: Exponentially decay samples older than this
    
    Returns:
        Array of weights, one per sample in dataset
    """
    conn = sqlite3.connect(str(db_path))
    
    # Get all timestamps for items in dataset
    weights = np.ones(len(dataset))
    
    # Map sample index -> (item_tag, anchor_idx)
    for idx in range(len(dataset)):
        tag, anchor = dataset.samples[idx]
        
        # Get timestamp for this anchor point in the item's history
        df = dataset.history.get(tag)
        if df is None or anchor >= len(df):
            continue
        
        anchor_ts = df.iloc[anchor]["timestamp"]
        
        # Get max timestamp (most recent)
        max_ts = dataset.history[tag]["timestamp"].max()
        
        # Compute days ago
        days_ago = (max_ts - anchor_ts).days
        
        # Exponential decay: recent samples get weight ~1, old samples decay
        # After recent_months, weight ~= decay_rate^(month_multiplier)
        weight = decay_rate ** (days_ago / 30.0 / max(1, recent_months))
        weights[idx] = max(weight, 0.01)  # Floor at 0.01 to keep all samples
    
    # Normalize to sum to 1 for sampler
    weights = weights / weights.sum() * len(weights)
    
    conn.close()
    return weights


def get_recency_weighted_sampler(
    dataset,
    db_path: str | Path = "bazaar.db",
    decay_rate: float = 0.95,
    recent_months: int = 12,
    replacement: bool = True,
) -> WeightedRandomSampler:
    """
    Create a weighted sampler that biases toward recent data.
    """
    weights = compute_recency_weights(dataset, db_path, decay_rate, recent_months)
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=len(dataset),
        replacement=replacement,
    )


class RegimeShiftDetector:
    """
    Detect when market regime changes significantly.
    Used to trigger retraining or adapt model.
    """
    
    def __init__(
        self,
        db_path: str | Path = "bazaar.db",
        window_days: int = 30,
        threshold_std: float = 2.0,
    ):
        self.db_path = str(db_path)
        self.window_days = window_days
        self.threshold_std = threshold_std
        self.baseline_stats = None
    
    def compute_baseline(self) -> dict:
        """Compute baseline price stats from all historical data."""
        conn = sqlite3.connect(self.db_path)
        result = conn.execute("""
            SELECT 
                AVG(sell) as mean_price,
                STDEV(sell) as std_price,
                AVG(CAST(volume AS FLOAT)) as mean_volume
            FROM price_history
        """).fetchone()
        conn.close()
        
        self.baseline_stats = {
            "mean_price": result[0],
            "std_price": result[1],
            "mean_volume": result[2],
        }
        return self.baseline_stats
    
    def detect_shift(self) -> dict:
        """
        Check if recent data shows significant shift from baseline.
        Returns shift info: {item_tag: shift_magnitude, ...}
        """
        if self.baseline_stats is None:
            self.compute_baseline()
        
        conn = sqlite3.connect(self.db_path)
        
        # Get recent data stats (last N days)
        recent_stats = conn.execute(f"""
            SELECT 
                item_tag,
                AVG(sell) as recent_mean,
                STDEV(sell) as recent_std,
                COUNT(*) as count
            FROM price_history
            WHERE timestamp > datetime('now', '-{self.window_days} days')
            GROUP BY item_tag
            HAVING count > 10
        """).fetchall()
        
        conn.close()
        
        shifts = {}
        for tag, recent_mean, recent_std, count in recent_stats:
            if recent_mean is None:
                continue
            
            # Z-score: how many stds away from baseline is recent mean?
            baseline_mean = self.baseline_stats["mean_price"]
            baseline_std = self.baseline_stats["std_price"]
            
            if baseline_std > 0:
                z_score = abs(recent_mean - baseline_mean) / baseline_std
                shifts[tag] = z_score
        
        # Summarize overall shift
        shift_values = list(shifts.values())
        if shift_values:
            median_shift = np.median(shift_values)
            pct_shifted = sum(1 for z in shift_values if z > self.threshold_std) / len(shift_values)
        else:
            median_shift = 0
            pct_shifted = 0
        
        return {
            "items_with_shift": shifts,
            "median_z_score": median_shift,
            "pct_items_shifted": pct_shifted,
            "should_retrain": pct_shifted > 0.2 or median_shift > self.threshold_std,
        }
