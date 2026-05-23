from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .common import ensure_dir


XBD_COLORS = np.array(
    [
        [0, 0, 0],
        [0, 180, 0],
        [255, 230, 0],
        [255, 128, 0],
        [220, 0, 0],
    ],
    dtype=np.uint8,
)


def tensor_to_image(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.ndim == 3:
            x = x.permute(1, 2, 0).numpy()
    x = np.asarray(x)
    if x.max() <= 1.0:
        x = x * 255
    return np.clip(x, 0, 255).astype(np.uint8)


def colorize_mask(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask).astype(np.int64)
    mask = np.clip(mask, 0, len(XBD_COLORS) - 1)
    return XBD_COLORS[mask]


def save_segmentation_grid(pre, post, target, pred, path, title=None):
    path = Path(path)
    ensure_dir(path.parent)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    panels = [
        (tensor_to_image(pre), "pre"),
        (tensor_to_image(post), "post"),
        (colorize_mask(target), "target"),
        (colorize_mask(pred), "prediction"),
    ]
    for ax, (img, name) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(name)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_training_curve(history, path):
    path = Path(path)
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, values in history.items():
        if values and all(isinstance(v, (int, float)) for v in values):
            ax.plot(values, label=key)
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
