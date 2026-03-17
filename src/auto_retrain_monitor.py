#!/usr/bin/env python3
"""
Auto-retraining monitor: periodically checks for regime shifts and retrains model.
Run in tmux or as a background service.

Usage:
    python -m src.auto_retrain_monitor \
        --check-interval=3600 \
        --retrain-threshold=0.2 \
        --log-file=logs/monitor.log
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from src.recency_aware import RegimeShiftDetector


def setup_logger(log_file: str | Path = "logs/monitor.log") -> None:
    """Setup logging."""
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Simple append-mode logger
    global log_path
    log_path = log_file


def log_msg(level: str, msg: str) -> None:
    """Log message with timestamp."""
    ts = datetime.now().isoformat()
    line = f"[{ts}] {level:>8} | {msg}"
    print(line)
    
    if 'log_path' in globals():
        with open(log_path, 'a') as f:
            f.write(line + '\n')


def run_retraining(args: argparse.Namespace) -> bool:
    """
    Trigger retraining with recency bias.
    Returns True if successful, False otherwise.
    """
    log_msg("INFO", "🚀 Starting retraining with recency bias...")
    
    cmd = [
        sys.executable, "-m", "src.train",
        "--recency-bias",
        f"--epochs={args.retrain_epochs}",
        f"--lr={args.retrain_lr}",
        f"--dropout={args.retrain_dropout}",
        f"--train-frac={args.train_frac}",
    ]
    
    log_msg("INFO", f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.retrain_timeout)
        
        if result.returncode == 0:
            log_msg("SUCCESS", "✅ Retraining completed successfully!")
            # Log last few lines of output
            lines = result.stdout.split('\n')[-5:]
            for line in lines:
                if line.strip():
                    log_msg("INFO", f"  {line}")
            return True
        else:
            log_msg("ERROR", f"❌ Retraining failed with exit code {result.returncode}")
            log_msg("ERROR", f"stderr: {result.stderr[-500:]}")  # Last 500 chars
            return False
    except subprocess.TimeoutExpired:
        log_msg("ERROR", f"❌ Retraining timeout after {args.retrain_timeout}s")
        return False
    except Exception as e:
        log_msg("ERROR", f"❌ Retraining exception: {e}")
        return False


def check_regime_shift(args: argparse.Namespace) -> tuple[bool, dict]:
    """
    Check if regime shift detected.
    Returns (should_retrain, shift_info).
    """
    try:
        detector = RegimeShiftDetector(
            db_path=args.db,
            threshold_std=args.shift_threshold_std,
        )
        detector.compute_baseline()
        shift_info = detector.detect_shift()
        
        should_retrain = (
            shift_info["should_retrain"] and
            shift_info["pct_items_shifted"] > args.retrain_threshold
        )
        
        return should_retrain, shift_info
    except Exception as e:
        log_msg("ERROR", f"Failed to check regime shift: {e}")
        return False, {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-retraining monitor: detects regime shifts and retrains model."
    )
    
    # Monitoring config
    parser.add_argument("--db", default="bazaar.db", help="Path to database")
    parser.add_argument("--items", default="top_items.csv", help="Path to items CSV")
    parser.add_argument("--graph", default="graph_output", help="Path to graph output")
    parser.add_argument("--checkpoints", default="checkpoints", help="Path to checkpoints")
    
    # Check interval
    parser.add_argument(
        "--check-interval",
        type=int,
        default=3600,
        help="Seconds between regime shift checks (default: 1 hour)",
    )
    
    # Shift detection threshold
    parser.add_argument(
        "--retrain-threshold",
        type=float,
        default=0.2,
        help="Pct of items that must shift before retraining (default: 0.2 = 20%)",
    )
    parser.add_argument(
        "--shift-threshold-std",
        type=float,
        default=2.0,
        help="Z-score threshold for individual item shift (default: 2.0)",
    )
    
    # Retraining config
    parser.add_argument(
        "--retrain-epochs",
        type=int,
        default=20,
        help="Epochs for retraining (default: 20)",
    )
    parser.add_argument(
        "--retrain-lr",
        type=float,
        default=5e-4,
        help="Learning rate for retraining (default: 5e-4)",
    )
    parser.add_argument(
        "--retrain-dropout",
        type=float,
        default=0.15,
        help="Dropout for retraining (default: 0.15)",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="Train fraction for retraining (default: 0.7)",
    )
    parser.add_argument(
        "--retrain-timeout",
        type=int,
        default=86400,  # 24 hours
        help="Timeout for retraining in seconds (default: 86400 = 24h)",
    )
    
    # Logging
    parser.add_argument(
        "--log-file",
        default="logs/monitor.log",
        help="Path to log file (default: logs/monitor.log)",
    )
    
    # Dry-run mode
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check for shifts but don't actually retrain",
    )
    
    args = parser.parse_args()
    
    setup_logger(args.log_file)
    
    log_msg("INFO", "=" * 80)
    log_msg("INFO", "🎬 Auto-retraining monitor started")
    log_msg("INFO", f"   Check interval: {args.check_interval}s ({args.check_interval/3600:.1f}h)")
    log_msg("INFO", f"   Retrain threshold: {args.retrain_threshold:.1%}")
    log_msg("INFO", f"   Shift threshold: {args.shift_threshold_std} std")
    log_msg("INFO", f"   Dry-run mode: {args.dry_run}")
    log_msg("INFO", "=" * 80)
    
    last_retrain_time = 0
    min_retrain_interval = 3600  # Don't retrain more than once per hour
    
    try:
        while True:
            # Check for regime shift
            should_retrain, shift_info = check_regime_shift(args)
            
            if shift_info:
                median_z = shift_info.get("median_z_score", 0)
                pct_shifted = shift_info.get("pct_items_shifted", 0)
                log_msg(
                    "SHIFT",
                    f"Regime shift detected: median_z={median_z:.2f}, "
                    f"pct_shifted={pct_shifted:.1%}",
                )
                
                # Check retraining cooldown
                time_since_retrain = time.time() - last_retrain_time
                if should_retrain and time_since_retrain > min_retrain_interval:
                    if args.dry_run:
                        log_msg("INFO", "🔍 DRY-RUN: Would trigger retraining now")
                    else:
                        success = run_retraining(args)
                        if success:
                            last_retrain_time = time.time()
                            log_msg("INFO", f"Next retrain available in {min_retrain_interval}s")
                else:
                    if should_retrain:
                        wait_time = min_retrain_interval - time_since_retrain
                        log_msg(
                            "INFO",
                            f"⏳ Retrain cooldown active. Wait {wait_time:.0f}s before next retrain",
                        )
            else:
                log_msg("DEBUG", "✅ No significant regime shift detected")
            
            # Sleep before next check
            log_msg("DEBUG", f"Sleeping {args.check_interval}s...")
            time.sleep(args.check_interval)
    
    except KeyboardInterrupt:
        log_msg("INFO", "🛑 Monitor stopped by user (Ctrl+C)")
    except Exception as e:
        log_msg("ERROR", f"💥 Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
