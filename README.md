# ECE 228 Project: xBD Damage Assessment Benchmark

This repo implements the project suggested by the proposal feedback: use a real post-disaster dataset, save every intermediate artifact, and compare the original U-Net/ResNet idea with stronger baselines.

## What It Runs

- **Change segmentation:** pre + post image -> pixel damage map with `change_unet`.
- **Segmentation baselines:** `unet`, `fcn_resnet50`, `deeplabv3_resnet50`.
- **Crop-level damage classification:** `resnet18`, `mobilenet_v3_small`, `efficientnet_b0`.
- **Saved outputs:** prepared CSVs, crop images, checkpoints, per-epoch metrics JSON, `history.csv`, curves, test metrics, and prediction visualizations.

## Expected xBD Layout

Put the official xBD/xView2 files anywhere under one root directory. The code searches recursively for:

```text
*_pre_disaster.png
*_post_disaster.png
*_pre_disaster.json
*_post_disaster.json
```

## Prepare Real Data

```powershell
conda run -n recsys-gpu python scripts/prepare_xbd.py --data-root D:\path\to\xbd --out-dir outputs\processed --make-crops
```

This creates:

- `outputs/processed/xbd_pairs.csv`
- `outputs/processed/xbd_damage_crops.csv`
- `outputs/processed/damage_crops/*.png`
- `outputs/processed/prepare_summary.json`

## Run Full Benchmark

```powershell
conda run -n recsys-gpu python scripts/run_benchmark.py --config configs/xbd_benchmark.yaml --epochs 20 --batch-size 8
```

For a quicker single experiment:

```powershell
conda run -n recsys-gpu python scripts/train_segmentation.py --model change_unet --epochs 5
conda run -n recsys-gpu python scripts/train_classification.py --model resnet18 --epochs 5
```

## CUDA Environment

The experiments were run in the `recsys-gpu` conda environment with CUDA PyTorch.

Check the environment:

```powershell
conda run -n recsys-gpu python scripts/check_env.py
```

Recreate a similar environment on another machine:

```powershell
conda env create -f environment-recsys-gpu.yml
```

## Smoke Test Without xBD

```powershell
python scripts/make_smoke_xbd.py --out-dir outputs\smoke_xbd
python scripts/prepare_xbd.py --data-root outputs\smoke_xbd --out-dir outputs\smoke_processed --make-crops
python scripts/train_segmentation.py --pairs-csv outputs\smoke_processed\xbd_pairs.csv --model change_unet --epochs 1 --batch-size 2
python scripts/train_classification.py --crops-csv outputs\smoke_processed\xbd_damage_crops.csv --model resnet18 --epochs 1 --batch-size 4
```

## Report-Friendly Comparison

Use `test_metrics.json` from each run directory for tables:

- `pixel_acc`, `mean_iou`, `mean_dice` for segmentation.
- `accuracy`, `macro_f1`, `weighted_f1` for classification.
- per-class IoU/F1 to discuss class imbalance, especially destroyed/minor-damage.

The strongest comparison story is:

1. Original proposal baseline: U-Net localization + ResNet crop classification.
2. Stronger segmentation baselines: FCN and DeepLabV3.
3. Disaster-specific change baseline: pre/post concatenated Change U-Net.
4. Efficient classifier baselines: MobileNetV3 and EfficientNet-B0.
