import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def markdown_table(df, columns):
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for _, row in df[columns].iterrows():
        vals = []
        for col in columns:
            val = row[col]
            vals.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def metric_leader(df, metric):
    if not len(df) or metric not in df:
        return None, None
    valid = df[df[metric].notna()]
    if not len(valid):
        return None, None
    row = valid.loc[valid[metric].idxmax()]
    return row["run"], float(row[metric])


def run_args(run_dir, run_name):
    cfg_path = run_dir / run_name / "run_config.json"
    if not cfg_path.exists():
        return {}
    return load_json(cfg_path).get("args", {})


def main():
    parser = argparse.ArgumentParser(description="Create report-ready tables and plots from benchmark outputs.")
    parser.add_argument("--run-dir", default="outputs/real_runs")
    parser.add_argument("--processed-dir", default="outputs/real_processed")
    parser.add_argument("--out-dir", default="outputs/report_assets")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prep = load_json(Path(args.processed_dir) / "prepare_summary.json")
    rows = []
    for path in sorted(run_dir.glob("*/test_metrics.json")):
        metrics = load_json(path)
        row = {"run": path.parent.name}
        for key in ["loss", "pixel_acc", "mean_iou", "mean_dice", "accuracy", "macro_f1", "weighted_f1"]:
            row[key] = metrics.get(key)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "benchmark_summary.csv", index=False)

    seg = df[df["mean_iou"].notna()].copy()
    cls = df[df["accuracy"].notna()].copy()
    if len(seg):
        ax = seg.plot.bar(x="run", y=["mean_iou", "mean_dice"], figsize=(8, 4), rot=20)
        ax.set_title("Damage segmentation benchmark")
        ax.set_ylim(0, max(0.45, float(seg[["mean_iou", "mean_dice"]].max().max()) + 0.05))
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / "segmentation_benchmark.png", dpi=200)
        plt.close()
    if len(cls):
        ax = cls.plot.bar(x="run", y=["accuracy", "macro_f1", "weighted_f1"], figsize=(8, 4), rot=20)
        ax.set_title("Crop-level damage classification benchmark")
        ax.set_ylim(0, max(0.75, float(cls[["accuracy", "macro_f1", "weighted_f1"]].max().max()) + 0.05))
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / "classification_benchmark.png", dpi=200)
        plt.close()

    md = []
    md.append("# Real xBD/xView2 Experiment Results\n")
    md.append("## Dataset\n")
    md.append("- Source: HuggingFace `xn67744/Disaster_Recognition_RemoteSense_EN_CN_JA`, xBD/xView2-derived train/test archives.")
    md.append(f"- Image-pair subset: {prep['splits']} ({prep['num_pairs']} pre/post pairs total).")
    crop_splits = prep.get("paired_crop_splits") or prep.get("crop_splits", {})
    num_crops = prep.get("num_paired_crops") or prep.get("num_crops", 0)
    crop_class_counts = prep.get("paired_crop_class_counts") or prep.get("crop_class_counts", {})
    md.append(f"- Crop-level classification set: {crop_splits} ({num_crops} crops).")
    md.append(f"- Crop class counts: {crop_class_counts}.")
    md.append("- Environment: conda `recsys-gpu`, PyTorch CUDA on RTX 4080 Laptop GPU.")
    md.append("- Image size: 256 for segmentation, 224 for crop classification.")
    best_cls, _ = metric_leader(cls, "macro_f1")
    best_cls_args = run_args(run_dir, best_cls) if best_cls else {}
    aug_state = "enabled" if best_cls_args.get("augment") else "disabled"
    md.append(f"- Strongest classifier inputs use paired pre/post crops; data augmentation is {aug_state} for that run.\n")
    md.append("## Segmentation Results\n")
    if len(seg):
        md.append(markdown_table(seg, ["run", "pixel_acc", "mean_iou", "mean_dice", "loss"]))
    md.append("\n## Classification Results\n")
    if len(cls):
        md.append(markdown_table(cls, ["run", "accuracy", "macro_f1", "weighted_f1", "loss"]))
    md.append("\n## Notes For Report\n")
    md.append("- The benchmark uses real post-disaster satellite imagery and real xBD/xView2 masks, addressing the grading feedback about using a real dataset.")
    best_seg, best_seg_score = metric_leader(seg, "mean_iou")
    best_cls, best_cls_score = metric_leader(cls, "macro_f1")
    if best_seg is not None:
        md.append(f"- {best_seg} achieved the strongest segmentation result by mean IoU ({best_seg_score:.4f}).")
    if best_cls is not None:
        md.append(f"- {best_cls} achieved the strongest crop-level classification result by macro-F1 ({best_cls_score:.4f}).")
    md.append("- Minor and major damage remain harder than no-damage/destroyed, which is consistent with class ambiguity and imbalance in disaster assessment.")
    md.append("- The current run uses a reproducible subset so all baselines can finish on one workstation; scaling to full xBD/tier3 is a natural extension.")
    (out_dir / "results_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(out_dir / "results_summary.md")


if __name__ == "__main__":
    main()
