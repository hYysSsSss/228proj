import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


DAMAGE_TO_ID = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
}


def _polygon_from_wkt_like(wkt):
    if not isinstance(wkt, str):
        return None
    text = wkt.strip()
    if not text.startswith("POLYGON"):
        return None
    text = text.replace("POLYGON", "").strip().lstrip("(").rstrip(")")
    ring = text.split("),(")[0].replace("(", "").replace(")", "")
    coords = []
    for item in ring.split(","):
        xy = item.strip().split()
        if len(xy) >= 2:
            coords.append((float(xy[0]), float(xy[1])))
    return coords if len(coords) >= 3 else None


def read_xbd_label(label_path):
    with Path(label_path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    features = data.get("features", {})
    xy = features.get("xy", features if isinstance(features, list) else [])
    rows = []
    for feat in xy:
        props = feat.get("properties", {})
        poly = feat.get("wkt") or feat.get("polygon") or props.get("wkt")
        coords = _polygon_from_wkt_like(poly)
        if coords is None and isinstance(feat.get("geometry"), dict):
            raw = feat["geometry"].get("coordinates", [])
            if raw and raw[0]:
                coords = [(float(x), float(y)) for x, y in raw[0]]
        subtype = props.get("subtype", "no-damage")
        rows.append({"polygon": coords, "damage": subtype})
    return rows


def rasterize_xbd(label_path, size, include_damage=True):
    rows = read_xbd_label(label_path)
    building = Image.new("L", size, 0)
    damage = Image.new("L", size, 0)
    bdraw = ImageDraw.Draw(building)
    ddraw = ImageDraw.Draw(damage)
    for row in rows:
        poly = row["polygon"]
        if not poly:
            continue
        bdraw.polygon(poly, fill=1)
        if include_damage:
            ddraw.polygon(poly, fill=DAMAGE_TO_ID.get(row["damage"], 1))
    return np.array(building, dtype=np.uint8), np.array(damage, dtype=np.uint8), rows


def find_xbd_pairs(data_root):
    root = Path(data_root)
    label_files = sorted(root.rglob("*_post_disaster.json"))
    pairs = []
    for post_label in label_files:
        stem = post_label.name.replace("_post_disaster.json", "")
        pre_label = post_label.with_name(f"{stem}_pre_disaster.json")
        post_img = next(root.rglob(f"{stem}_post_disaster.png"), None)
        pre_img = next(root.rglob(f"{stem}_pre_disaster.png"), None)
        if pre_img and post_img and pre_label.exists():
            split = "unknown"
            for part in post_img.parts:
                if part.lower() in {"train", "tier1", "tier3"}:
                    split = "train"
                elif part.lower() in {"test", "hold"}:
                    split = "test"
                elif part.lower() in {"val", "valid", "validation"}:
                    split = "val"
            pairs.append(
                {
                    "id": stem,
                    "split": split,
                    "pre_image": str(pre_img),
                    "post_image": str(post_img),
                    "pre_label": str(pre_label),
                    "post_label": str(post_label),
                }
            )
    return pd.DataFrame(pairs)


def build_xbd_index(data_root, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = find_xbd_pairs(data_root)
    if df.empty:
        raise FileNotFoundError(
            f"No xBD pairs found under {data_root}. Expected *_pre_disaster.png, "
            "*_post_disaster.png, and matching JSON labels."
        )
    if (df["split"] == "unknown").all():
        rng = np.random.default_rng(228)
        order = rng.permutation(len(df))
        split = np.full(len(df), "train", dtype=object)
        split[order[int(0.7 * len(df)) : int(0.85 * len(df))]] = "val"
        split[order[int(0.85 * len(df)) :]] = "test"
        df["split"] = split
    else:
        rng = np.random.default_rng(228)
        train_idx = df.index[df["split"].eq("train")].to_numpy()
        rng.shuffle(train_idx)
        if not df["split"].eq("val").any() and len(train_idx) >= 4:
            n_val = max(1, int(0.15 * len(train_idx)))
            df.loc[train_idx[:n_val], "split"] = "val"
            train_idx = train_idx[n_val:]
        if not df["split"].eq("test").any() and len(train_idx) >= 4:
            n_test = max(1, int(0.15 * len(train_idx)))
            df.loc[train_idx[:n_test], "split"] = "test"
    df.to_csv(out_dir / "xbd_pairs.csv", index=False)
    return df


class XBDChangeSegmentationDataset(Dataset):
    def __init__(self, csv_path, split, image_size=256, target="damage"):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"].eq(split)].reset_index(drop=True)
        self.image_size = image_size
        self.target = target

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pre = Image.open(row.pre_image).convert("RGB")
        post = Image.open(row.post_image).convert("RGB")
        if "mask_path" in self.df.columns and isinstance(row.mask_path, str):
            damage = np.array(Image.open(row.mask_path).convert("L"), dtype=np.uint8)
            building = (damage > 0).astype(np.uint8)
        else:
            _, damage, _ = rasterize_xbd(row.post_label, post.size, include_damage=True)
            building, _, _ = rasterize_xbd(row.pre_label, pre.size, include_damage=False)
        mask = damage if self.target == "damage" else building
        mask = Image.fromarray(mask)
        pre = TF.resize(pre, [self.image_size, self.image_size])
        post = TF.resize(post, [self.image_size, self.image_size])
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)
        return {
            "id": row.id,
            "pre": TF.to_tensor(pre),
            "post": TF.to_tensor(post),
            "image": torch.cat([TF.to_tensor(pre), TF.to_tensor(post)], dim=0),
            "mask": torch.from_numpy(np.array(mask, dtype=np.int64)),
        }


class XBDDamageCropDataset(Dataset):
    def __init__(self, crop_csv, split, image_size=224, augment=False):
        self.df = pd.read_csv(crop_csv)
        self.df = self.df[self.df["split"].eq(split)].reset_index(drop=True)
        self.augment = augment
        self.resize = transforms.Resize((image_size, image_size))
        self.color_jitter = transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02)
        self.tf = transforms.Compose(
            [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _paired_hflip(self, img):
        half = img.width // 2
        pre = TF.hflip(img.crop((0, 0, half, img.height)))
        post = TF.hflip(img.crop((half, 0, img.width, img.height)))
        out = Image.new("RGB", img.size)
        out.paste(pre, (0, 0))
        out.paste(post, (half, 0))
        return out

    def _augment(self, img, paired):
        if torch.rand(1).item() < 0.5:
            img = self._paired_hflip(img) if paired and img.width >= 2 else TF.hflip(img)
        if torch.rand(1).item() < 0.5:
            img = TF.vflip(img)
        return self.color_jitter(img)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.crop_path).convert("RGB")
        paired = bool(row.get("paired", False))
        if self.augment:
            img = self._augment(img, paired)
        img = self.resize(img)
        return {"image": self.tf(img), "label": int(row.label), "id": row.crop_id}


def make_damage_crops(pairs_csv, out_dir, min_size=8, padding=4):
    pairs = pd.read_csv(pairs_csv)
    out_dir = Path(out_dir)
    crop_dir = out_dir / "damage_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in pairs.iterrows():
        post = Image.open(row.post_image).convert("RGB")
        anns = read_xbd_label(row.post_label)
        for j, ann in enumerate(anns):
            poly = ann["polygon"]
            label = DAMAGE_TO_ID.get(ann["damage"], 1) - 1
            if not poly:
                continue
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            x0, x1 = max(0, int(min(xs)) - padding), min(post.width, int(max(xs)) + padding)
            y0, y1 = max(0, int(min(ys)) - padding), min(post.height, int(max(ys)) + padding)
            if x1 - x0 < min_size or y1 - y0 < min_size:
                continue
            crop_id = f"{row.id}_{j:04d}"
            crop_path = crop_dir / f"{crop_id}.png"
            post.crop((x0, y0, x1, y1)).save(crop_path)
            rows.append(
                {
                    "crop_id": crop_id,
                    "image_id": row.id,
                    "split": row.split,
                    "crop_path": str(crop_path),
                    "label": label,
                    "damage": ann["damage"],
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "xbd_damage_crops.csv", index=False)
    return df


def build_hf_xview2_index(data_root, out_dir, train_limit=None, val_limit=None, test_limit=None):
    root = Path(data_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for source_split in ["train", "test"]:
        image_dir = root / source_split / "images"
        mask_dir = root / source_split / "masks"
        if not image_dir.exists():
            continue
        post_images = sorted(image_dir.glob("*_post_disaster.png"))
        for post in post_images:
            image_id = post.name.replace("_post_disaster.png", "")
            pre = image_dir / f"{image_id}_pre_disaster.png"
            mask = mask_dir / f"{image_id}_post_disaster_target.png"
            if pre.exists() and mask.exists():
                rows.append(
                    {
                        "id": image_id,
                        "source_split": source_split,
                        "split": source_split,
                        "pre_image": str(pre),
                        "post_image": str(post),
                        "mask_path": str(mask),
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No HF xView2 image/mask pairs found under {data_root}")
    train_df = df[df.source_split.eq("train")].sample(frac=1.0, random_state=228).reset_index(drop=True)
    test_df = df[df.source_split.eq("test")].sample(frac=1.0, random_state=228).reset_index(drop=True)
    n_val = val_limit or max(1, int(0.15 * len(train_df)))
    val_df = train_df.iloc[:n_val].copy()
    val_df["split"] = "val"
    train_df = train_df.iloc[n_val:].copy()
    train_df["split"] = "train"
    test_df["split"] = "test"
    if train_limit:
        train_df = train_df.iloc[:train_limit].copy()
    if test_limit:
        test_df = test_df.iloc[:test_limit].copy()
    keep = pd.concat([train_df, val_df, test_df], ignore_index=True)
    keep.to_csv(out_dir / "hf_xview2_pairs.csv", index=False)
    return keep


def _component_boxes(binary):
    try:
        from scipy import ndimage

        labeled, n = ndimage.label(binary)
        objects = ndimage.find_objects(labeled)
        boxes = []
        for label_id, sl in enumerate(objects, start=1):
            if sl is None:
                continue
            ys, xs = sl
            area = int((labeled[sl] == label_id).sum())
            boxes.append((xs.start, ys.start, xs.stop, ys.stop, area))
        return boxes
    except Exception:
        ys, xs = np.where(binary)
        if len(xs) == 0:
            return []
        return [(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1, int(len(xs)))]


def make_hf_xview2_crops(pairs_csv, out_dir, min_pixels=24, padding=8, max_components_per_class=6):
    pairs = pd.read_csv(pairs_csv)
    out_dir = Path(out_dir)
    crop_dir = out_dir / "damage_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in pairs.iterrows():
        post = Image.open(row.post_image).convert("RGB")
        mask = np.array(Image.open(row.mask_path).convert("L"), dtype=np.uint8)
        for damage_id in [1, 2, 3, 4]:
            boxes = sorted(_component_boxes(mask == damage_id), key=lambda b: b[4], reverse=True)
            for j, (x0, y0, x1, y1, area) in enumerate(boxes[:max_components_per_class]):
                if area < min_pixels:
                    continue
                x0 = max(0, x0 - padding)
                y0 = max(0, y0 - padding)
                x1 = min(post.width, x1 + padding)
                y1 = min(post.height, y1 + padding)
                crop_id = f"{row.id}_c{damage_id}_{j:02d}"
                crop_path = crop_dir / f"{crop_id}.png"
                post.crop((x0, y0, x1, y1)).save(crop_path)
                rows.append(
                    {
                        "crop_id": crop_id,
                        "image_id": row.id,
                        "split": row.split,
                        "crop_path": str(crop_path),
                        "label": damage_id - 1,
                        "damage": ["no-damage", "minor-damage", "major-damage", "destroyed"][damage_id - 1],
                        "pixels": area,
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "hf_xview2_damage_crops.csv", index=False)
    return df


def make_hf_xview2_paired_crops(pairs_csv, out_dir, min_pixels=24, padding=16, max_components_per_class=6):
    pairs = pd.read_csv(pairs_csv)
    out_dir = Path(out_dir)
    crop_dir = out_dir / "damage_pair_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in pairs.iterrows():
        pre = Image.open(row.pre_image).convert("RGB")
        post = Image.open(row.post_image).convert("RGB")
        mask = np.array(Image.open(row.mask_path).convert("L"), dtype=np.uint8)
        for damage_id in [1, 2, 3, 4]:
            boxes = sorted(_component_boxes(mask == damage_id), key=lambda b: b[4], reverse=True)
            for j, (x0, y0, x1, y1, area) in enumerate(boxes[:max_components_per_class]):
                if area < min_pixels:
                    continue
                x0 = max(0, x0 - padding)
                y0 = max(0, y0 - padding)
                x1 = min(post.width, x1 + padding)
                y1 = min(post.height, y1 + padding)
                pre_crop = pre.crop((x0, y0, x1, y1))
                post_crop = post.crop((x0, y0, x1, y1))
                paired = Image.new("RGB", (pre_crop.width + post_crop.width, pre_crop.height))
                paired.paste(pre_crop, (0, 0))
                paired.paste(post_crop, (pre_crop.width, 0))
                crop_id = f"{row.id}_pair_c{damage_id}_{j:02d}"
                crop_path = crop_dir / f"{crop_id}.png"
                paired.save(crop_path)
                rows.append(
                    {
                        "crop_id": crop_id,
                        "image_id": row.id,
                        "split": row.split,
                        "crop_path": str(crop_path),
                        "label": damage_id - 1,
                        "damage": ["no-damage", "minor-damage", "major-damage", "destroyed"][damage_id - 1],
                        "pixels": area,
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        "paired": True,
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "hf_xview2_damage_pair_crops.csv", index=False)
    return df
