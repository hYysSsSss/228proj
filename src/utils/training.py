from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .common import ensure_dir, save_json
from .metrics import classification_scores, segmentation_confusion_matrix, segmentation_scores, top1
from .visualize import save_segmentation_grid, save_training_curve
from src.models.segmentation import segmentation_logits


def _autocast_enabled(device, amp):
    return bool(amp and device.type == "cuda")


def train_segmentation(model, loaders, optimizer, device, run_dir, epochs, num_classes, amp=True, save_every_epoch=True):
    run_dir = ensure_dir(run_dir)
    scaler = torch.cuda.amp.GradScaler(enabled=_autocast_enabled(device, amp))
    criterion = torch.nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "val_mean_iou": [], "val_mean_dice": []}
    best_miou = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(loaders["train"], desc=f"seg train {epoch}", leave=False):
            x = batch["image"].to(device)
            y = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=_autocast_enabled(device, amp)):
                loss = criterion(segmentation_logits(model(x)), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * x.size(0)
        train_loss /= max(len(loaders["train"].dataset), 1)
        val = evaluate_segmentation(model, loaders["val"], device, run_dir / "val_predictions", num_classes, max_visuals=4)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_mean_iou"].append(val["mean_iou"])
        history["val_mean_dice"].append(val["mean_dice"])
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        save_json(val, run_dir / f"epoch_{epoch:03d}_val_metrics.json")
        if save_every_epoch:
            torch.save(model.state_dict(), run_dir / f"epoch_{epoch:03d}.pt")
        if val["mean_iou"] > best_miou:
            best_miou = val["mean_iou"]
            torch.save(model.state_dict(), run_dir / "best.pt")
        save_training_curve(history, run_dir / "training_curve.png")
    return history


@torch.no_grad()
def evaluate_segmentation(model, loader, device, visual_dir, num_classes, max_visuals=0):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    cm = None
    visual_count = 0
    for batch in tqdm(loader, desc="seg eval", leave=False):
        x = batch["image"].to(device)
        y = batch["mask"].to(device)
        logits = segmentation_logits(model(x))
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)
        batch_cm = segmentation_confusion_matrix(pred, y, num_classes)
        cm = batch_cm if cm is None else cm + batch_cm
        total_loss += loss.item() * x.size(0)
        if visual_count < max_visuals:
            for i in range(min(x.size(0), max_visuals - visual_count)):
                save_segmentation_grid(
                    batch["pre"][i],
                    batch["post"][i],
                    y[i].cpu(),
                    pred[i].cpu(),
                    Path(visual_dir) / f"{batch['id'][i]}.png",
                )
                visual_count += 1
    scores = segmentation_scores(cm)
    scores["loss"] = total_loss / max(len(loader.dataset), 1)
    scores["confusion_matrix"] = cm.tolist()
    return scores


def train_classifier(model, loaders, optimizer, device, run_dir, epochs, class_names, amp=True, save_every_epoch=True, class_weights=None, label_smoothing=0.0):
    run_dir = ensure_dir(run_dir)
    scaler = torch.cuda.amp.GradScaler(enabled=_autocast_enabled(device, amp))
    weight = class_weights.to(device) if class_weights is not None else None
    criterion = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_accuracy": []}
    best_f1 = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(loaders["train"], desc=f"cls train {epoch}", leave=False):
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=_autocast_enabled(device, amp)):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * x.size(0)
        train_loss /= max(len(loaders["train"].dataset), 1)
        val = evaluate_classifier(model, loaders["val"], device, class_names)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_macro_f1"].append(val["macro_f1"])
        history["val_accuracy"].append(val["accuracy"])
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        save_json(val, run_dir / f"epoch_{epoch:03d}_val_metrics.json")
        if save_every_epoch:
            torch.save(model.state_dict(), run_dir / f"epoch_{epoch:03d}.pt")
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            torch.save(model.state_dict(), run_dir / "best.pt")
        save_training_curve(history, run_dir / "training_curve.png")
    return history


@torch.no_grad()
def evaluate_classifier(model, loader, device, class_names):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    y_true, y_pred = [], []
    total_loss = 0.0
    for batch in tqdm(loader, desc="cls eval", leave=False):
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        logits = model(x)
        loss = criterion(logits, y)
        pred = top1(logits)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        total_loss += loss.item() * x.size(0)
    scores = classification_scores(y_true, y_pred, class_names)
    scores["loss"] = total_loss / max(len(loader.dataset), 1)
    return scores
