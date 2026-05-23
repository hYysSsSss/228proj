# ECE 228 Final Report Package

This folder contains the LaTeX report and all report-ready experiment artifacts for the xBD/xView2 disaster damage assessment project.

## Main Files

- `main.tex`: complete LaTeX report in IEEE conference style.
- `references.bib`: BibTeX references used by the report.
- `figures/`: benchmark plots and training curves used in the report.
- `tables/`: CSV tables and markdown summary of the benchmark results.
- `metrics/`: per-run test metrics, run configurations, and training histories.
- `configs/`: experiment configuration file.
- `data_metadata/`: processed data metadata and crop CSV files.

## Important Results

- Best segmentation model: FCN-ResNet50.
  - Pixel accuracy: 0.9531
  - Mean IoU: 0.3189
  - Mean Dice: 0.3957

- Best classification model: ConvNeXt-Tiny with paired pre/post crops, augmentation, and label smoothing.
  - Accuracy: 0.7110
  - Macro-F1: 0.6716
  - Weighted-F1: 0.7123

## Dataset Note

The full extracted raw xBD/xView2-derived dataset is not copied into this package because it is larger than 11 GB. The raw data remain in:

`D:/Github_desktop/228proj/data/xview2_real`

The package includes the processed metadata CSV files needed to document the exact subset, splits, and crop-level labels used in the experiments.

## Compile

In Overleaf or a local LaTeX installation, compile `main.tex`. If using BibTeX locally, run:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
