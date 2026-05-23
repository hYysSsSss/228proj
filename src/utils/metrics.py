import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def segmentation_confusion_matrix(pred, target, num_classes: int, ignore_index=None):
    pred = pred.detach().cpu().numpy().astype(np.int64).ravel()
    target = target.detach().cpu().numpy().astype(np.int64).ravel()
    if ignore_index is not None:
        keep = target != ignore_index
        pred, target = pred[keep], target[keep]
    keep = (target >= 0) & (target < num_classes)
    cm = np.bincount(
        num_classes * target[keep] + pred[keep],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return cm


def segmentation_scores(cm):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = tp + fp + fn
    iou = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
    dice = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    acc = tp.sum() / max(cm.sum(), 1)
    return {
        "pixel_acc": float(acc),
        "mean_iou": float(np.mean(iou)),
        "mean_dice": float(np.mean(dice)),
        "class_iou": iou.tolist(),
        "class_dice": dice.tolist(),
    }


def classification_scores(y_true, y_pred, class_names):
    labels = list(range(len(class_names)))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            class_names[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in labels
        },
    }


def top1(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=1)
