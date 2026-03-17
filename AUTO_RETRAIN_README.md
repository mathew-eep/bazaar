# Auto-Retraining & Regime Shift Detection

This guide explains the new features added to handle market regime changes and improve model performance on recent data.

## Problem

Your initial training showed:
- **Windows 1-2 (old data)**: 0.025-0.03 loss (good)
- **Window 3 (recent data)**: 0.079 loss (3x worse)

This happens because the model was trained on older, stable market data, but recent markets behave differently. The model overfits to historical patterns and fails on new regimes.

## Solution

### 1. Recency-Weighted Training

**What it does**: Biases training data toward recent samples, so the model learns recent market patterns better.

**How it works**:
- Recent samples get weight ≈ 1.0
- Samples older than 12 months decay exponentially at 95% per month
- Samples from 5 years ago get weight ≈ 0.01 (still included, but downweighted)

**Use it**:
```bash
cd ~/bazaar
source .venv/bin/activate

# Retrain WITH recency bias (fixes window 3 overfitting)
python -m src.train \
  --recency-bias \
  --epochs=30 \
  --lr=5e-4 \
  --dropout=0.15 \
  --train-frac=0.7
```

**Expected improvement**: Window 3 validation loss should drop from 0.079 to ~0.05 or better.

---

### 2. Automatic Regime Shift Detection & Retraining

**What it does**: Continuously monitors for market regime changes and automatically retrains when detected.

**How it works**:
1. Every N hours, computes baseline price statistics
2. Compares recent data (last 30 days) to baseline
3. If >20% of items deviate by >2 standard deviations: **regime shift detected**
4. Automatically triggers retraining with recency weighting
5. Logs everything to `logs/monitor.log`

**Launch the monitor**:

```bash
# Option 1: Use launcher script (recommended)
cd ~/bazaar
bash launch_monitor.sh

# Option 2: Manual tmux
tmux new -s bazaar-monitor -d
tmux send-keys -t bazaar-monitor "cd ~/bazaar && source .venv/bin/activate && python -m src.auto_retrain_monitor" Enter

# Option 3: Direct (foreground)
python -m src.auto_retrain_monitor \
  --check-interval=3600 \
  --retrain-threshold=0.2 \
  --log-file=logs/monitor.log
```

**Monitor logs in real-time**:
```bash
tail -f ~/bazaar/logs/monitor.log
```

**Stop the monitor**:
```bash
tmux kill-session -t bazaar-monitor
```

---

## Configuration

### Recency Weighting Options

```bash
# In train.py, defaults are:
# --recency-bias              # Enable (default: True)
# --train-frac=0.7            # Use 70% for train (30% for validation)
# --lr=5e-4                   # Lower learning rate
# --dropout=0.15              # Higher regularization
```

Adjust `--train-frac` to control train/val balance:
- `0.8` = more training data (less validation, faster)
- `0.7` = balanced (recommended for regime changes)
- `0.5` = more validation (slower, better generalization)

### Regime Shift Detection Options

```bash
# In auto_retrain_monitor, adjustable via CLI:

--check-interval=3600              # Check every 1 hour
--retrain-threshold=0.2            # Retrain if >20% items shifted
--shift-threshold-std=2.0          # Item shift = 2 stds from baseline
--retrain-epochs=20                # Retraining duration
--retrain-timeout=86400            # Max 24h per retrain
--dry-run                          # Test mode (don't actually retrain)
```

**Example: More aggressive retraining**:
```bash
python -m src.auto_retrain_monitor \
  --check-interval=1800 \            # Check every 30 min
  --retrain-threshold=0.15 \         # Retrain if >15% items shifted
  --shift-threshold-std=1.5 \        # Stricter shift detection
  --retrain-epochs=30                # Longer retraining
```

---

## Workflows

### Setup: Initial Retrain + Monitor

```bash
# 1. Stop current training (if running)
# Ctrl+C in tmux if training is live

# 2. Retrain with recency bias
cd ~/bazaar
source .venv/bin/activate
python -m src.train \
  --recency-bias \
  --epochs=50 \
  --train-frac=0.7 \
  --lr=5e-4

# 3. Launch monitor in background
tmux new -s monitor -d "cd ~/bazaar && source .venv/bin/activate && python -m src.auto_retrain_monitor"

# 4. Watch logs
tail -f logs/monitor.log
```

### Daily Operations

```bash
# Check monitor status
tmux ls | grep monitor
tail -n 20 logs/monitor.log

# If monitor crashes, restart
tmux kill-session -t monitor
tmux new -s monitor -d "cd ~/bazaar && source .venv/bin/activate && python -m src.auto_retrain_monitor"

# Check best model
ls -lh checkpoints/best.pt
stat checkpoints/best.pt | grep -i modify
```

### Emergency: Force Retraining

```bash
# Stop current retrain (if running)
pkill -f "python.*src.train"
pkill -f "python.*auto_retrain_monitor"

# Immediate retrain
python -m src.train --recency-bias --epochs=20 --train-frac=0.7

# Restart monitor
python -m src.auto_retrain_monitor --dry-run  # Test first
python -m src.auto_retrain_monitor           # Run
```

---

## Expected Behavior

### First Run (with recency bias)
- Epoch 1: High loss (model re-learning with new sample weights)
- Epoch 5-10: Significant improvement
- Epoch 15-20: Convergence, especially on window 3

Example output:
```
Epoch 001 | train=0.12730 | val_mean=0.07119 | val_windows=[0.06666,0.05535,0.09157]
...
Epoch 020 | train=0.053 | val_mean=0.0415 | val_windows=[0.025,0.020,0.055]  ✅ Much better!
```

### Monitor Running

```
[2026-03-17T10:45:12] SHIFT | Regime shift detected: median_z=2.3, pct_shifted=35%
[2026-03-17T10:45:13] INFO | 🚀 Starting retraining with recency bias...
[2026-03-17T11:02:45] SUCCESS | ✅ Retraining completed successfully!
[2026-03-17T11:02:46] INFO | Next retrain available in 3600s
```

---

## Troubleshooting

### Monitor not detecting shifts
```bash
# Test manually
python3 << 'EOF'
from src.recency_aware import RegimeShiftDetector
detector = RegimeShiftDetector()
detector.compute_baseline()
info = detector.detect_shift()
print(info)
EOF
```

### Retraining taking too long
```bash
# Reduce epochs or check GPU
nvidia-smi
ps aux | grep train

# Increase timeout if needed
python -m src.auto_retrain_monitor --retrain-timeout=172800  # 48h
```

### Monitor crashed
```bash
# Check logs
tail -n 50 logs/monitor.log

# Restart
tmux kill-session -t bazaar-monitor
bash launch_monitor.sh
```

---

## Files Added

- `src/recency_aware.py` - Recency weighting & regime detection
- `src/auto_retrain_monitor.py` - Auto-retraining orchestrator
- `launch_monitor.sh` - Helper to launch monitor in tmux

---

## Next Steps

1. **Run recency-weighted retrain**:
   ```bash
   python -m src.train --recency-bias --epochs=50 --train-frac=0.7
   ```

2. **Compare results** with original (check window 3 loss)

3. **Launch monitor** if results look good:
   ```bash
   bash launch_monitor.sh
   ```

4. **Monitor** for 24-48 hours, tweak thresholds if needed

---

Questions? Check logs or run in `--dry-run` mode first to test.
