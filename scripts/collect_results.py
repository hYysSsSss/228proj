import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Collect all test_metrics.json files into one CSV.")
    parser.add_argument("--run-dir", default="outputs/runs")
    parser.add_argument("--out", default="outputs/runs/benchmark_summary.csv")
    args = parser.parse_args()

    rows = []
    for path in Path(args.run_dir).glob("*/test_metrics.json"):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        row = {"run": path.parent.name}
        for key in ["loss", "pixel_acc", "mean_iou", "mean_dice", "accuracy", "macro_f1", "weighted_f1"]:
            if key in metrics:
                row[key] = metrics[key]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("run") if rows else pd.DataFrame()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df)


if __name__ == "__main__":
    main()
