# Azure Training Runbook for Bazaar TFT

## Goal
Use Azure Student credits to train the Bazaar TFT model reliably and cheaply.

## Recommended path
1. Start with a single GPU VM (fastest to launch).
2. Use Spot if available to stretch credits.
3. Run short validation training first, then full training.
4. Auto-shutdown and deallocate VM after runs.

## Budget strategy
1. Prefer lower-cost GPU sizes first (for example T4 class instances if available).
2. Keep experiments short early (5 epochs), then scale.
3. Save checkpoints to storage and stop the VM immediately.
4. Use Spot VMs where possible.

## Important model compatibility note
Current training code uses CUDA autocast with BF16. Some lower-cost NVIDIA GPUs may not support BF16 well. Use BF16 when supported, and fallback to FP16 otherwise.

## Step-by-step setup

### 1. Create the VM
1. In Azure Portal, create an Ubuntu GPU VM.
2. Choose a low-cost GPU size available in your region.
3. Enable SSH key login.
4. Enable auto-shutdown.
5. Open SSH (22) in NSG.

### 2. Connect and install base tools
1. SSH into the VM.
2. Install git, Python, and venv tools.
3. Confirm GPU is visible with nvidia-smi.

### 3. Create project environment
1. Create a working directory.
2. Clone or upload this project.
3. Create and activate a Python virtual environment.
4. Install dependencies:
   - torch with CUDA build
   - numpy
   - pandas
   - requests
   - tqdm
   - scikit-learn
   - joblib

### 4. Upload required project data
Ensure these exist on the VM:
1. bazaar.db
2. top_items.csv
3. graph_output folder
4. src folder and scripts

### 5. Rebuild event-derived tables
Run:
python rebuild_market_events.py --db bazaar.db --items top_items.csv

### 6. Run smoke checks
Run:
python phase4_smoke_test.py
python phase56_smoke_test.py
python -m src.train --forward-only

### 7. Start training
Use tmux or screen so sessions survive disconnects.

Short sanity run:
python -m src.train --epochs 5 --batch-size 32 --lr 3e-4 --dropout 0.2 --crossing-weight 0.02

Full run:
python -m src.train --epochs 50 --batch-size 64 --lr 5e-4 --dropout 0.15 --crossing-weight 0.02 --walk-forward-val-windows 3

### 8. Evaluate best checkpoint
Run:
python -m src.evaluate --checkpoint checkpoints/best.pt --skip-isolation --calibrate-quantiles

## How to decide if training is good
1. Validation mean loss trends down over epochs.
2. Walk-forward validation windows are stable (not one good window and others bad).
3. Coverage calibration moves closer to targets:
   - Q10 near 0.10
   - Q50 near 0.50
   - Q90 near 0.90
4. Simulated PnL metrics improve:
   - win_rate
   - avg_return_pct
   - profit_factor
   - max_drawdown

## Cost control checklist
1. Stop and deallocate VM right after run.
2. Keep disk sizes minimal.
3. Archive checkpoints to Blob Storage.
4. Use short hyperparameter trials before long runs.
5. Track runtime per epoch to forecast credit consumption.

## Optional next step
After the VM workflow is stable, migrate to Azure ML jobs for:
1. Better experiment tracking
2. Parameter sweeps
3. Scheduled retraining
