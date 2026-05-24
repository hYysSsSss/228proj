import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class XBDImageMultiTaskDataset(Dataset):
    """Full-image multi-task dataset for segmentation-guided classification."""

    def __init__(self, pairs_csv, split, image_size=256, max_items=None):
        self.df = pd.read_csv(pairs_csv)
        self.df = self.df[self.df["split"].eq(split)].reset_index(drop=True)
        if max_items is not None:
            self.df = self.df.iloc[:max_items].reset_index(drop=True)
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    @staticmethod
    def image_label_from_mask(mask):
        labels = np.unique(mask)
        damage_labels = labels[(labels >= 1) & (labels <= 4)]
        if len(damage_labels) == 0:
            return 0
        return int(damage_labels.max() - 1)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pre = Image.open(row.pre_image).convert("RGB")
        post = Image.open(row.post_image).convert("RGB")
        mask = Image.open(row.mask_path).convert("L")
        pre = TF.resize(pre, [self.image_size, self.image_size])
        post = TF.resize(post, [self.image_size, self.image_size])
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)
        mask_arr = np.array(mask, dtype=np.int64)
        pre_tensor = TF.to_tensor(pre)
        post_tensor = TF.to_tensor(post)
        return {
            "id": row.id,
            "pre": pre_tensor,
            "post": post_tensor,
            "image": torch.cat([pre_tensor, post_tensor], dim=0),
            "mask": torch.from_numpy(mask_arr),
            "label": torch.tensor(self.image_label_from_mask(mask_arr), dtype=torch.long),
        }
