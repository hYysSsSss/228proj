import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


DAMAGES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


def polygon_wkt(x0, y0, x1, y1):
    return f"POLYGON (({x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}, {x0} {y0}))"


def make_label(polys):
    return {
        "features": {
            "xy": [
                {"wkt": polygon_wkt(*bbox), "properties": {"subtype": damage}}
                for bbox, damage in polys
            ]
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Create a tiny xBD-like dataset for pipeline smoke tests.")
    parser.add_argument("--out-dir", default="outputs/smoke_xbd")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()

    root = Path(args.out_dir)
    rng = np.random.default_rng(228)
    for split in ["train", "val", "test"]:
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        split = "train" if i < int(args.n * 0.65) else "val" if i < int(args.n * 0.85) else "test"
        image_id = f"smoke_{i:03d}"
        pre = Image.new("RGB", (args.size, args.size), (40, 55, 60))
        post = Image.new("RGB", (args.size, args.size), (42, 55, 62))
        pre_draw = ImageDraw.Draw(pre)
        post_draw = ImageDraw.Draw(post)
        polys = []
        for j in range(3):
            x0 = int(rng.integers(5, args.size - 45))
            y0 = int(rng.integers(5, args.size - 45))
            w = int(rng.integers(14, 34))
            h = int(rng.integers(14, 34))
            damage = DAMAGES[(i + j) % len(DAMAGES)]
            bbox = (x0, y0, x0 + w, y0 + h)
            pre_draw.rectangle(bbox, fill=(150, 150, 145), outline=(230, 230, 220))
            color = {
                "no-damage": (145, 150, 145),
                "minor-damage": (185, 165, 80),
                "major-damage": (200, 95, 50),
                "destroyed": (80, 70, 68),
            }[damage]
            post_draw.rectangle(bbox, fill=color, outline=(235, 235, 225))
            polys.append((bbox, damage))
        pre.save(root / split / "images" / f"{image_id}_pre_disaster.png")
        post.save(root / split / "images" / f"{image_id}_post_disaster.png")
        label = make_label(polys)
        (root / split / "labels" / f"{image_id}_pre_disaster.json").write_text(json.dumps(label), encoding="utf-8")
        (root / split / "labels" / f"{image_id}_post_disaster.json").write_text(json.dumps(label), encoding="utf-8")
    print(root)


if __name__ == "__main__":
    main()
