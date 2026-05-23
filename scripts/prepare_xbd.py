import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.xbd import build_xbd_index, make_damage_crops
from src.utils.common import save_json


def main():
    parser = argparse.ArgumentParser(description="Prepare xBD pair index and damage crop tables.")
    parser.add_argument("--data-root", required=True, help="Root directory containing xBD images and labels.")
    parser.add_argument("--out-dir", default="outputs/processed")
    parser.add_argument("--make-crops", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    pairs = build_xbd_index(args.data_root, out_dir)
    summary = {
        "num_pairs": int(len(pairs)),
        "splits": pairs["split"].value_counts().to_dict(),
        "pairs_csv": str(out_dir / "xbd_pairs.csv"),
    }
    if args.make_crops:
        crops = make_damage_crops(out_dir / "xbd_pairs.csv", out_dir)
        summary["num_crops"] = int(len(crops))
        summary["crop_splits"] = crops["split"].value_counts().to_dict() if len(crops) else {}
        summary["crops_csv"] = str(out_dir / "xbd_damage_crops.csv")
    save_json(summary, out_dir / "prepare_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
