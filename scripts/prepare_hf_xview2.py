import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.xbd import build_hf_xview2_index, make_hf_xview2_crops, make_hf_xview2_paired_crops
from src.utils.common import save_json


def main():
    parser = argparse.ArgumentParser(description="Prepare extracted HuggingFace xView2/xBD data.")
    parser.add_argument("--data-root", default="data/xview2_real")
    parser.add_argument("--out-dir", default="outputs/real_processed")
    parser.add_argument("--train-limit", type=int, default=800)
    parser.add_argument("--val-limit", type=int, default=200)
    parser.add_argument("--test-limit", type=int, default=200)
    parser.add_argument("--make-crops", action="store_true")
    parser.add_argument("--make-paired-crops", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    pairs = build_hf_xview2_index(
        args.data_root,
        out_dir,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
    )
    summary = {
        "num_pairs": int(len(pairs)),
        "splits": pairs["split"].value_counts().to_dict(),
        "pairs_csv": str(out_dir / "hf_xview2_pairs.csv"),
    }
    if args.make_crops:
        crops = make_hf_xview2_crops(out_dir / "hf_xview2_pairs.csv", out_dir)
        summary["num_crops"] = int(len(crops))
        summary["crop_splits"] = crops["split"].value_counts().to_dict() if len(crops) else {}
        summary["crops_csv"] = str(out_dir / "hf_xview2_damage_crops.csv")
        summary["crop_class_counts"] = crops["damage"].value_counts().to_dict() if len(crops) else {}
    if args.make_paired_crops:
        pair_crops = make_hf_xview2_paired_crops(out_dir / "hf_xview2_pairs.csv", out_dir)
        summary["num_paired_crops"] = int(len(pair_crops))
        summary["paired_crop_splits"] = pair_crops["split"].value_counts().to_dict() if len(pair_crops) else {}
        summary["paired_crops_csv"] = str(out_dir / "hf_xview2_damage_pair_crops.csv")
        summary["paired_crop_class_counts"] = pair_crops["damage"].value_counts().to_dict() if len(pair_crops) else {}
    save_json(summary, out_dir / "prepare_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
