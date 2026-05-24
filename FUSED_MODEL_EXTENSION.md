# Segmentation-Guided Classification Extension

This extension adds a new model family without modifying the previous benchmark models or report outputs.

## Architecture

The new model is implemented in `src/models/fused_multitask.py`.

It contains two independent branches that both take the raw paired pre/post image tensor as input:

- `FeatureUNetSegmenter`: a U-Net style segmentation model.
- `ResNetStageClassifier`: a ResNet-like full-image classification model.

The U-Net encoder exposes five feature stages. The classifier has five aligned feature stages with the same spatial scale and channel count. At each scale, the U-Net encoder feature is passed through a learnable `1x1` convolution adapter, then added to the classifier feature at the matching scale:

```text
seg_feature_i -> 1x1 conv -> filtered_seg_feature_i
classifier_feature_i + filtered_seg_feature_i -> next classifier stage
```

This follows the requested design: segmentation features are not copied directly; they are filtered by a `1x1` convolution before being fused into the classifier.

## Training Strategy Choice

The default strategy is `warmup_joint`:

1. Train only the segmentation branch for a short warmup period.
2. Then jointly train the segmentation branch, classification branch, and all `1x1` fusion adapters.

This was chosen because the classifier receives guidance from segmentation features. If both branches are trained from scratch at the same time, the classifier is guided by unstable early segmentation features. A short segmentation warmup makes the guidance path more meaningful before joint optimization starts.

The script also supports:

- `joint`: train all parameters together from the first epoch.
- `alternate`: first update the segmentation branch, then update the classifier and fusion adapters.
- `warmup_alternate`: warm up segmentation, then use alternating updates.

## New Files

- `src/models/fused_multitask.py`
- `src/data/fused_xbd.py`
- `scripts/train_fused_multitask.py`
- `configs/fused_xview2_benchmark.yaml`

## Report-Grade Experiment Run

Command:

```bash
conda run -n recsys-gpu python scripts/train_fused_multitask.py \
  --config configs/fused_xview2_benchmark.yaml \
  --run-name fused_warmup_joint_full800_augmented_report \
  --epochs 8 \
  --warmup-epochs 2 \
  --batch-size 4 \
  --base-channels 16 \
  --class-weighted-loss \
  --label-smoothing 0.05
```

The run is saved locally under:

`outputs/fused_runs/fused_warmup_joint_full800_augmented_report`

## Results

This experiment uses full raw pre/post images for both segmentation and classification. The classification target is an image-level worst-damage label derived from the segmentation mask, so these numbers should not be directly compared with the previous crop-level classifier results.

This report-grade run uses the full reproducible project split:

| Split | No-damage | Minor | Major | Destroyed | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 393 | 48 | 97 | 262 | 800 |
| Val | 88 | 19 | 32 | 61 | 200 |
| Test | 89 | 13 | 31 | 67 | 200 |

Test metrics on the 200-image test split:

| Metric | Value |
| --- | ---: |
| Pixel accuracy | 0.9436 |
| Mean IoU | 0.2144 |
| Mean Dice | 0.2396 |
| Classification accuracy | 0.5000 |
| Classification macro-F1 | 0.3511 |
| Classification weighted-F1 | 0.4878 |

The result confirms that the new architecture runs end-to-end and saves all metrics, checkpoints, curves, and visualizations. The classification score is lower than the crop-level classifier because this experiment predicts an image-level worst-damage label from the full raw pre/post image pair. This is a harder and different task, but it is aligned with the requested design that both branches consume the original image data.
