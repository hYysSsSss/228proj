import argparse
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description="Run all configured xBD baselines.")
    parser.add_argument("--config", default="configs/xbd_benchmark.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--skip-classification", action="store_true")
    parser.add_argument("--skip-segmentation", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    extra = []
    if args.epochs is not None:
        extra += ["--epochs", str(args.epochs)]
    if args.batch_size is not None:
        extra += ["--batch-size", str(args.batch_size)]

    if not args.skip_segmentation:
        for model in cfg["benchmarks"]["change_segmentation"] + cfg["benchmarks"]["segmentation"]:
            run([sys.executable, "scripts/train_segmentation.py", "--config", args.config, "--model", model] + extra)
    if not args.skip_classification:
        for model in cfg["benchmarks"]["classification"]:
            run([sys.executable, "scripts/train_classification.py", "--config", args.config, "--model", model] + extra)


if __name__ == "__main__":
    main()
