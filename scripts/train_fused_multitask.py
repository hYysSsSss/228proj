import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.fused_xbd import XBDImageMultiTaskDataset
from src.models.fused_multitask import SegGuidedClassifier
from src.utils.common import ensure_dir, get_device, save_json, seed_everything
from src.utils.metrics import classification_scores, segmentation_confusion_matrix, segmentation_scores, top1
from src.utils.visualize import save_segmentation_grid, save_training_curve


CLASS_NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


def autocast_enabled(device, amp):
    return bool(amp and device.type == "cuda")


def set_requires_grad(module, enabled):
    for param in module.parameters():
        param.requires_grad_(enabled)


def make_loaders(pairs_csv, image_size, batch_size, num_workers, limits):
    loaders = {}
    datasets = {}
    for split in ["train", "val", "test"]:
        ds = XBDImageMultiTaskDataset(pairs_csv, split, image_size=image_size, max_items=limits.get(split))
        datasets[split] = ds
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=True,
        )
    return datasets, loaders


def label_counts(dataset):
    counts = {name: 0 for name in CLASS_NAMES}
    for idx in range(len(dataset)):
        label = int(dataset[idx]["label"])
        counts[CLASS_NAMES[label]] += 1
    return counts


@torch.no_grad()
def evaluate(model, loader, device, run_dir, num_seg_classes=5, max_visuals=0):
    model.eval()
    seg_criterion = torch.nn.CrossEntropyLoss()
    cls_criterion = torch.nn.CrossEntropyLoss()
    seg_loss_total = 0.0
    cls_loss_total = 0.0
    y_true, y_pred = [], []
    cm = None
    visual_count = 0
    for batch in tqdm(loader, desc="fused eval", leave=False):
        x = batch["image"].to(device)
        mask = batch["mask"].to(device)
        label = batch["label"].to(device)
        out = model(x)
        seg_logits = out["seg"]
        cls_logits = out["cls"]
        seg_loss = seg_criterion(seg_logits, mask)
        cls_loss = cls_criterion(cls_logits, label)
        pred_mask = seg_logits.argmax(dim=1)
        pred_label = top1(cls_logits)
        batch_cm = segmentation_confusion_matrix(pred_mask, mask, num_seg_classes)
        cm = batch_cm if cm is None else cm + batch_cm
        y_true.extend(label.cpu().tolist())
        y_pred.extend(pred_label.cpu().tolist())
        seg_loss_total += seg_loss.item() * x.size(0)
        cls_loss_total += cls_loss.item() * x.size(0)
        if visual_count < max_visuals:
            visual_dir = Path(run_dir) / "val_predictions"
            for i in range(min(x.size(0), max_visuals - visual_count)):
                save_segmentation_grid(
                    batch["pre"][i],
                    batch["post"][i],
                    mask[i].cpu(),
                    pred_mask[i].cpu(),
                    visual_dir / f"{batch['id'][i]}.png",
                    title=f"cls target={CLASS_NAMES[int(label[i])]}, pred={CLASS_NAMES[int(pred_label[i])]}",
                )
                visual_count += 1
    seg_scores = segmentation_scores(cm)
    cls_scores = classification_scores(y_true, y_pred, CLASS_NAMES)
    n = max(len(loader.dataset), 1)
    return {
        "seg_loss": seg_loss_total / n,
        "cls_loss": cls_loss_total / n,
        "pixel_acc": seg_scores["pixel_acc"],
        "mean_iou": seg_scores["mean_iou"],
        "mean_dice": seg_scores["mean_dice"],
        "accuracy": cls_scores["accuracy"],
        "macro_f1": cls_scores["macro_f1"],
        "weighted_f1": cls_scores["weighted_f1"],
        "per_class": cls_scores["per_class"],
        "confusion_matrix": cm.tolist(),
    }


def train_warmup_segmentation(model, batch, optimizer, scaler, device, amp):
    set_requires_grad(model.segmenter, True)
    set_requires_grad(model.classifier, False)
    set_requires_grad(model.adapters, False)
    x = batch["image"].to(device)
    mask = batch["mask"].to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=autocast_enabled(device, amp)):
        seg_logits = model.segmenter(x)
        loss = torch.nn.functional.cross_entropy(seg_logits, mask)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item(), 0.0


def train_joint(model, batch, optimizer, scaler, device, amp, cls_weight):
    set_requires_grad(model, True)
    x = batch["image"].to(device)
    mask = batch["mask"].to(device)
    label = batch["label"].to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=autocast_enabled(device, amp)):
        out = model(x)
        seg_loss = torch.nn.functional.cross_entropy(out["seg"], mask)
        cls_loss = torch.nn.functional.cross_entropy(out["cls"], label)
        loss = seg_loss + cls_weight * cls_loss
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return seg_loss.item(), cls_loss.item()


def train_alternate(model, batch, seg_optimizer, cls_optimizer, scaler, device, amp, cls_weight):
    x = batch["image"].to(device)
    mask = batch["mask"].to(device)
    label = batch["label"].to(device)

    set_requires_grad(model.segmenter, True)
    set_requires_grad(model.classifier, False)
    set_requires_grad(model.adapters, False)
    seg_optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=autocast_enabled(device, amp)):
        seg_logits = model.segmenter(x)
        seg_loss = torch.nn.functional.cross_entropy(seg_logits, mask)
    scaler.scale(seg_loss).backward()
    scaler.step(seg_optimizer)
    scaler.update()

    set_requires_grad(model.segmenter, False)
    set_requires_grad(model.classifier, True)
    set_requires_grad(model.adapters, True)
    cls_optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=autocast_enabled(device, amp)):
        out = model(x, detach_seg_for_cls=True)
        cls_loss = torch.nn.functional.cross_entropy(out["cls"], label) * cls_weight
    scaler.scale(cls_loss).backward()
    scaler.step(cls_optimizer)
    scaler.update()
    return seg_loss.item(), cls_loss.item() / cls_weight


def main():
    parser = argparse.ArgumentParser(description="Train segmentation-guided full-image classification.")
    parser.add_argument("--config", default="configs/fused_xview2_benchmark.yaml")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--strategy", choices=["joint", "alternate", "warmup_joint", "warmup_alternate"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--cls-loss-weight", type=float, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed_everything(cfg.get("seed", 228))
    device = get_device(cfg.get("device", "auto"))

    paths = cfg["paths"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    run_name = args.run_name or f"fused_{args.strategy or train_cfg['strategy']}"
    run_dir = ensure_dir(Path(paths["run_dir"]) / run_name)
    save_json({"config": cfg, "args": vars(args)}, run_dir / "run_config.json")

    limits = {
        "train": args.train_limit if args.train_limit is not None else train_cfg.get("train_limit"),
        "val": args.val_limit if args.val_limit is not None else train_cfg.get("val_limit"),
        "test": args.test_limit if args.test_limit is not None else train_cfg.get("test_limit"),
    }
    datasets, loaders = make_loaders(
        str(Path(paths["processed_dir"]) / data_cfg["pairs_csv"]),
        image_size=data_cfg["image_size"],
        batch_size=args.batch_size or train_cfg["batch_size"],
        num_workers=data_cfg.get("num_workers", 0),
        limits=limits,
    )
    save_json({split: label_counts(ds) for split, ds in datasets.items()}, run_dir / "label_counts.json")

    base = args.base_channels or train_cfg["base_channels"]
    channels = tuple(base * (2 ** i) for i in range(5))
    model = SegGuidedClassifier(
        in_channels=6,
        seg_classes=data_cfg["seg_classes"],
        cls_classes=len(CLASS_NAMES),
        channels=channels,
    ).to(device)

    strategy = args.strategy or train_cfg["strategy"]
    epochs = args.epochs or train_cfg["epochs"]
    warmup_epochs = args.warmup_epochs if args.warmup_epochs is not None else train_cfg.get("warmup_epochs", 0)
    cls_weight = args.cls_loss_weight if args.cls_loss_weight is not None else train_cfg.get("cls_loss_weight", 1.0)
    amp = train_cfg.get("amp", True)
    scaler = torch.cuda.amp.GradScaler(enabled=autocast_enabled(device, amp))

    joint_optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    seg_optimizer = torch.optim.AdamW(model.segmenter.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    cls_optimizer = torch.optim.AdamW(
        list(model.classifier.parameters()) + list(model.adapters.parameters()),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    history = {
        "train_seg_loss": [],
        "train_cls_loss": [],
        "val_mean_iou": [],
        "val_macro_f1": [],
        "val_accuracy": [],
    }
    best_score = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_seg_loss = 0.0
        train_cls_loss = 0.0
        for batch in tqdm(loaders["train"], desc=f"fused train {epoch}", leave=False):
            if strategy.startswith("warmup") and epoch <= warmup_epochs:
                seg_loss, cls_loss = train_warmup_segmentation(model, batch, seg_optimizer, scaler, device, amp)
            elif strategy in {"alternate", "warmup_alternate"}:
                seg_loss, cls_loss = train_alternate(model, batch, seg_optimizer, cls_optimizer, scaler, device, amp, cls_weight)
            else:
                seg_loss, cls_loss = train_joint(model, batch, joint_optimizer, scaler, device, amp, cls_weight)
            train_seg_loss += seg_loss * batch["image"].size(0)
            train_cls_loss += cls_loss * batch["image"].size(0)
        n_train = max(len(loaders["train"].dataset), 1)
        val = evaluate(model, loaders["val"], device, run_dir, data_cfg["seg_classes"], max_visuals=4)
        history["train_seg_loss"].append(train_seg_loss / n_train)
        history["train_cls_loss"].append(train_cls_loss / n_train)
        history["val_mean_iou"].append(val["mean_iou"])
        history["val_macro_f1"].append(val["macro_f1"])
        history["val_accuracy"].append(val["accuracy"])
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        save_json(val, run_dir / f"epoch_{epoch:03d}_val_metrics.json")
        score = val["macro_f1"] + val["mean_iou"]
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), run_dir / "best.pt")
        save_training_curve(history, run_dir / "training_curve.png")

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    test = evaluate(model, loaders["test"], device, run_dir, data_cfg["seg_classes"], max_visuals=6)
    save_json(test, run_dir / "test_metrics.json")
    print(test)


if __name__ == "__main__":
    main()
