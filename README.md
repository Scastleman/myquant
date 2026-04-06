# myquant

Learning-first SPY quant research project with an eventual transformer track.

## Current Defaults

- Python target: `3.12`
- Data frequency: daily
- Storage: Parquet-first
- Primary target: SPY adjusted-close move from `t` to `t+5`
- Secondary benchmark: SPY adjusted-close move from `t` to `t+1`
- Labeling: `down / flat / up` from train-set quantiles
- Model output: probability distribution over future move buckets
- Transformer note: include a latent state or regime token to capture slower market-state shifts
- Event regimes: VIX absolute daily moves of `10%` and `20%`

## Project Docs

- [BASELINE_PROMPT.md](./BASELINE_PROMPT.md)
- [FIRST_DATA_SPEC.md](./FIRST_DATA_SPEC.md)
- [PREBUILD_CHECKLIST.md](./PREBUILD_CHECKLIST.md)
- [SCALING_PLAN.md](./SCALING_PLAN.md)
- [INTRADAY_STORAGE.md](./INTRADAY_STORAGE.md)

## First Build Goal

Build a reproducible daily dataset pipeline for:

1. downloading the phase-1 ETF and macro-proxy universe,
2. generating leakage-safe features,
3. labeling SPY targets,
4. tagging VIX event days,
5. and training honest baseline models before any transformer work.

## Suggested Setup

Install Python `3.12`, then:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

For GPU training on the RTX `5070 Ti`, replace the default PyTorch wheel with the official CUDA build:

```powershell
pip install --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

## Repo Layout

```text
src/myquant/
configs/
data/
artifacts/
notebooks/
tests/
```

## Next Implementation Step

Build the first raw price snapshot and processed phase-1 dataset using:

```powershell
python -m myquant.data.download
python -m myquant.data.dataset
```

Build the larger grouped multi-target panel dataset using:

```powershell
python -m myquant.data.panel_dataset
```

Build the large current-500 daily universe and derived SPY-breadth datasets using:

```powershell
python -m myquant.data.large_universe_download
python -m myquant.data.large_universe_dataset
```

This large daily panel uses the current S&P 500 membership snapshot plus the ETF/macro context set, overlapping windows with `stride=1`, stock-relative-to-SPY features, and cross-sectional breadth features. It is useful for broad pretraining and breadth research, but it remains survivorship-biased because it does not reconstruct historical index membership.

Train the first baseline suite using:

```powershell
python -m myquant.training.run_baselines
```

Train the first patch-based transformer using:

```powershell
python -m myquant.training.run_transformer
```

For a larger CUDA-backed run with mixed precision, worker prefetch, gradient accumulation, step-level terminal progress, and an auto-opened live dashboard:

```powershell
python -m myquant.training.run_transformer --device cuda --epochs 30 --batch-size 256 --lookback 120 --d-model 256 --num-layers 4 --n-heads 8 --patch-length 10 --patch-stride 5 --num-workers 2 --accumulation-steps 2 --log-every-steps 5
```

For the grouped panel problem, point training at the panel parquet:

```powershell
python -m myquant.training.run_transformer --dataset-path data\processed\panel_dataset.parquet --device cuda --epochs 30 --batch-size 512 --lookback 60 --d-model 256 --num-layers 4 --n-heads 8 --patch-length 10 --patch-stride 5 --num-workers 2 --log-every-steps 25
```

For the large daily stock panel, use the multitask setup with an auxiliary excess-return head. On Windows, the current sweet spot is usually `--num-workers 2`. Use `--loader-cache-mode ram` when you want to keep the loader fully RAM-backed and avoid the disk memmap fallback:

```powershell
python -m myquant.training.run_transformer --dataset-path data\processed\large_universe_panel_dataset.parquet --device cuda --primary-target-column target_label_5d --aux-target-columns target_excess_label_5d --aux-loss-weight 0.35 --epochs 10 --batch-size 1024 --lookback 252 --window-stride 5 --learning-rate 3e-4 --warmup-fraction 0.1 --min-lr-ratio 0.1 --d-model 320 --num-layers 8 --n-heads 8 --regime-dim 96 --patch-length 8 --patch-stride 2 --num-workers 2 --loader-cache-mode ram --log-every-steps 200
```

For a higher-VRAM run that spends memory on longer temporal context and a larger backbone, use:

```powershell
python -m myquant.training.run_transformer --dataset-path data\processed\large_universe_panel_dataset.parquet --device cuda --primary-target-column target_label_5d --aux-target-columns target_excess_label_5d --aux-loss-weight 0.35 --epochs 5 --batch-size 768 --lookback 252 --window-stride 5 --learning-rate 1e-4 --warmup-fraction 0.2 --min-lr-ratio 0.1 --d-model 256 --num-layers 6 --n-heads 8 --regime-dim 64 --patch-length 8 --patch-stride 2 --num-workers 2 --loader-cache-mode ram --log-every-steps 50
```

This RAM-backed configuration keeps the loader off the disk cache path, targets a more moderate VRAM footprint, and still feeds the GPU well enough to reach high utilization on the large daily panel.

During transformer training, the run directory now auto-opens a live dashboard and updates these progress artifacts during the run:
- `training_progress.png`
- `training_history.json`
- `live_status.json`
- `dashboard.html`

The dashboard includes the training curves plus live GPU telemetry such as utilization, board VRAM, power draw, and PyTorch reserved memory. Use `--no-open-dashboard` if you want to skip the automatic pop-up.

Estimate sequence counts for larger multi-asset and intraday scenarios using:

```powershell
python -m myquant.planning.sequence_budget
```

Inspect the intraday bar lake using:

```powershell
python -m myquant.storage.bar_store --summary
```
