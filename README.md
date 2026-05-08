# bev_container_alignment1

BEV container alignment training/inference project.

## 1. Environment

```bash
cd /home/bev_container_alignment1
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Data Preparation

Raw data is expected under:

- `data/raw/images/front`
- `data/raw/images/rear`
- `data/annotations/_annotations.coco.json`
- `data/raw/camera_calib.yaml.txt`

Generate processed samples:

```bash
python3 scripts/generate_bev_labels.py
```

## 3. Train

Example training command:

```bash
python3 train.py --config configs/train_config_pairmap18.yaml --work_dir work_dirs/retrain_pairmap18
```

## 4. Test

Example test command:

```bash
python3 test.py --config configs/train_config_pairmap18.yaml --checkpoint work_dirs/retrain_pairmap18/checkpoints/best.pth --split test --max_samples 10 --print_samples
```

Test outputs are written to `test_results/`, including:

- `metrics.json`
- `bias_curve.png`
- `bev_heatmap.png`

## 5. Notes About Repository Size

This repo ignores large artifacts by default (`data/raw`, `data/processed`, `work_dirs`, `checkpoints`, `*.pth`, etc.).

If you need to share weights/data, use external storage (OSS/S3/Drive) and add links here.

## 6. Public Desensitization Notice

For public sharing, real sample images, labels, and private mapping files are removed from this repository.
Please place your own data under `data/raw/` and regenerate processed samples locally.

## 7. Push to GitHub

```bash
cd /home/bev_container_alignment1
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your_name>/<repo>.git
git push -u origin main
```

If the remote already exists:

```bash
git remote set-url origin https://github.com/<your_name>/<repo>.git
git push -u origin main
```
