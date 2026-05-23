# Real xBD/xView2 Experiment Results

## Dataset

- Source: HuggingFace `xn67744/Disaster_Recognition_RemoteSense_EN_CN_JA`, xBD/xView2-derived train/test archives.
- Image-pair subset: {'train': 800, 'val': 200, 'test': 200} (1200 pre/post pairs total).
- Crop-level classification set: {'train': 5208, 'test': 1391, 'val': 1388} (7987 crops).
- Crop class counts: {'no-damage': 3720, 'destroyed': 1542, 'major-damage': 1402, 'minor-damage': 1323}.
- Environment: conda `recsys-gpu`, PyTorch CUDA on RTX 4080 Laptop GPU.
- Image size: 256 for segmentation, 224 for crop classification.
- Strongest classifier inputs use paired pre/post crops; data augmentation is enabled for that run.

## Segmentation Results

| run | pixel_acc | mean_iou | mean_dice | loss |
| --- | --- | --- | --- | --- |
| seg_damage_change_unet | 0.9444 | 0.2436 | 0.2804 | 0.4806 |
| seg_damage_deeplabv3_resnet50 | 0.9493 | 0.2644 | 0.3034 | 0.2803 |
| seg_damage_fcn_resnet50 | 0.9531 | 0.3189 | 0.3957 | 0.1709 |

## Classification Results

| run | accuracy | macro_f1 | weighted_f1 | loss |
| --- | --- | --- | --- | --- |
| cls_convnext_tiny_pair_aug12_ls005 | 0.7110 | 0.6716 | 0.7123 | 0.7509 |
| cls_convnext_tiny_pair_noaug8 | 0.6959 | 0.6519 | 0.6986 | 0.8819 |
| cls_efficientnet_b0 | 0.6485 | 0.6083 | 0.6547 | 1.0007 |
| cls_efficientnet_v2_s_pair_aug10_ls005 | 0.6973 | 0.6594 | 0.7007 | 0.7848 |
| cls_mobilenet_v3_small | 0.6607 | 0.5804 | 0.6374 | 1.1512 |
| cls_resnet18 | 0.6477 | 0.5979 | 0.6464 | 1.0264 |

## Notes For Report

- The benchmark uses real post-disaster satellite imagery and real xBD/xView2 masks, addressing the grading feedback about using a real dataset.
- seg_damage_fcn_resnet50 achieved the strongest segmentation result by mean IoU (0.3189).
- cls_convnext_tiny_pair_aug12_ls005 achieved the strongest crop-level classification result by macro-F1 (0.6716).
- Minor and major damage remain harder than no-damage/destroyed, which is consistent with class ambiguity and imbalance in disaster assessment.
- The current run uses a reproducible subset so all baselines can finish on one workstation; scaling to full xBD/tier3 is a natural extension.