# BEV Container Alignment

A computer-vision project for container alignment using BEV perception, covering data preparation, model training, evaluation, and inference utilities.

## Project Summary

This project estimates container alignment offsets (for example, X/Y deviation) from front/rear camera views and outputs quantitative error metrics and visual diagnostics.

## What I Built

- Built an end-to-end BEV training and inference pipeline with PyTorch.
- Implemented data preprocessing and BEV label generation scripts.
- Maintained configuration-driven experiments for fast ablation and reproducibility.
- Added evaluation and visualization outputs (metrics, bias curves, heatmaps) for error analysis.
- Organized repository for public sharing: artifact isolation, ignore rules, and data desensitization.

## Tech Stack

- Python
- PyTorch
- NumPy / Matplotlib
- YAML-based config system

## Repository Structure

```text
configs/      Training and experiment configs
data/         Dataset loader and public data notes
models/       Network modules
scripts/      Data generation and utility scripts
utils/        Training/eval helper functions
train.py      Training entry
test.py       Evaluation entry
eval.py       Evaluation utilities
visualize.py  Visualization entry
```

## Reproducibility

### 1) Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2) Prepare Data

Expected raw layout:

- `data/raw/images/front`
- `data/raw/images/rear`
- `data/annotations/_annotations.coco.json`
- `data/raw/camera_calib.yaml.txt`

Generate processed samples:

```bash
python3 scripts/generate_bev_labels.py
```

### 3) Train

```bash
python3 train.py --config configs/train_config_pairmap18.yaml --work_dir work_dirs/retrain_pairmap18
```

### 4) Test

```bash
python3 test.py --config configs/train_config_pairmap18.yaml --checkpoint checkpoints/best.pth --split test --max_samples 10 --print_samples
```

Outputs are written to `test_results/`, including:

- `metrics.json`
- `bias_curve.png`
- `bev_heatmap.png`

## Results

Latest local evaluation (`test_results/metrics.json`):

| Experiment | Mean Error (mm) | Std (mm) | Success@10mm (%) | Samples | Notes |
|---|---:|---:|---:|---:|---|
| Pairmap18 test run | 30.27 | 2.51 | 0.0 | 3 | Small sample size; for pipeline validation |

## Interview Talking Points

- Why BEV representation is suitable for alignment tasks.
- How dataset construction and calibration quality affect offset error.
- How you diagnosed model bias using curve/heatmap outputs.
- Key engineering decisions that improved reproducibility and maintainability.

## Public Data Notice

This is a desensitized public version.

Removed from version control:

- Real processed samples
- Private annotation/mapping files
- Training artifacts and model checkpoints

Use your own data locally under `data/raw/` and regenerate processed samples.
