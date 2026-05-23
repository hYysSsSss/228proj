import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.xbd import XBDChangeSegmentationDataset
from src.models.segmentation import build_segmentation_model
from src.utils.common import ensure_dir, get_device, save_json, seed_everything
from src.utils.training import evaluate_segmentation, train_segmentation


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate xBD segmentation or change-segmentation baselines.")
    parser.add_argument("--config", default="configs/xbd_benchmark.yaml")
    parser.add_argument("--pairs-csv", default=None)
    parser.add_argument("--model", default="change_unet")
    parser.add_argument("--target", choices=["damage", "building"], default="damage")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed_everything(cfg["seed"])
    device = get_device(cfg["device"])
    data_cfg, train_cfg = cfg["data"], cfg["training"]
    pairs_csv = args.pairs_csv or str(Path(cfg["paths"]["processed_dir"]) / "xbd_pairs.csv")
    run_dir = ensure_dir(Path(cfg["paths"]["run_dir"]) / f"seg_{args.target}_{args.model}")
    save_json({"config": cfg, "args": vars(args)}, run_dir / "run_config.json")

    num_classes = 5 if args.target == "damage" else 2
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = XBDChangeSegmentationDataset(pairs_csv, split, data_cfg["image_size"], target=args.target)
        if split == "train" and len(ds) == 0:
            raise ValueError("Training split is empty. Check xbd_pairs.csv.")
        loaders[split] = DataLoader(
            ds,
            batch_size=args.batch_size or train_cfg["batch_size"],
            shuffle=split == "train",
            num_workers=data_cfg.get("num_workers", 0),
        )

    model = build_segmentation_model(args.model, num_classes=num_classes, in_channels=6, pretrained=args.pretrained).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

    if not args.eval_only:
        train_segmentation(
            model,
            loaders,
            optimizer,
            device,
            run_dir,
            epochs=args.epochs or train_cfg["epochs"],
            num_classes=num_classes,
            amp=train_cfg.get("amp", True),
            save_every_epoch=train_cfg.get("save_every_epoch", True),
        )
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))

    test_metrics = evaluate_segmentation(
        model,
        loaders["test"] if len(loaders["test"].dataset) else loaders["val"],
        device,
        run_dir / "test_predictions",
        num_classes,
        max_visuals=16,
    )
    save_json(test_metrics, run_dir / "test_metrics.json")
    print(test_metrics)


if __name__ == "__main__":
    main()
