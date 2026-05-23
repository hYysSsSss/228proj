import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.xbd import XBDDamageCropDataset
from src.models.classification import build_classifier
from src.utils.common import ensure_dir, get_device, save_json, seed_everything
from src.utils.training import evaluate_classifier, train_classifier


CLASS_NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate xBD crop-level damage classifiers.")
    parser.add_argument("--config", default="configs/xbd_benchmark.yaml")
    parser.add_argument("--crops-csv", default=None)
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--class-weighted-loss", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed_everything(cfg["seed"])
    device = get_device(cfg["device"])
    data_cfg, train_cfg = cfg["data"], cfg["training"]
    crops_csv = args.crops_csv or str(Path(cfg["paths"]["processed_dir"]) / "xbd_damage_crops.csv")
    run_name = args.run_name or f"cls_{args.model}"
    run_dir = ensure_dir(Path(cfg["paths"]["run_dir"]) / run_name)
    save_json({"config": cfg, "args": vars(args)}, run_dir / "run_config.json")

    loaders = {}
    datasets = {}
    for split in ["train", "val", "test"]:
        ds = XBDDamageCropDataset(crops_csv, split, image_size=224, augment=split == "train" and args.augment)
        datasets[split] = ds
        sampler = None
        shuffle = split == "train"
        if split == "train" and args.balanced_sampler:
            labels = ds.df["label"].astype(int).to_numpy()
            counts = torch.bincount(torch.tensor(labels), minlength=len(CLASS_NAMES)).float()
            weights = 1.0 / counts.clamp_min(1)
            sample_weights = weights[torch.tensor(labels)]
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            shuffle = False
        loaders[split] = DataLoader(
            ds,
            batch_size=args.batch_size or train_cfg["batch_size"],
            shuffle=shuffle,
            sampler=sampler,
            num_workers=data_cfg.get("num_workers", 0),
        )

    model = build_classifier(args.model, num_classes=len(CLASS_NAMES), pretrained=args.pretrained).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    class_weights = None
    if args.class_weighted_loss:
        labels = datasets["train"].df["label"].astype(int).to_numpy()
        counts = torch.bincount(torch.tensor(labels), minlength=len(CLASS_NAMES)).float()
        class_weights = counts.sum() / counts.clamp_min(1)
        class_weights = class_weights / class_weights.mean()

    if not args.eval_only:
        train_classifier(
            model,
            loaders,
            optimizer,
            device,
            run_dir,
            epochs=args.epochs or train_cfg["epochs"],
            class_names=CLASS_NAMES,
            amp=train_cfg.get("amp", True),
            save_every_epoch=train_cfg.get("save_every_epoch", True),
            class_weights=class_weights,
            label_smoothing=args.label_smoothing,
        )
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))

    test_loader = loaders["test"] if len(loaders["test"].dataset) else loaders["val"]
    test_metrics = evaluate_classifier(model, test_loader, device, CLASS_NAMES)
    save_json(test_metrics, run_dir / "test_metrics.json")
    print(test_metrics)


if __name__ == "__main__":
    main()
