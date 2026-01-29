#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def decode_rle(segm):
    if segm is None:
        return None
    if isinstance(segm.get("counts"), list):
        return mask_utils.decode(mask_utils.frPyObjects(segm, segm["size"][0], segm["size"][1]))
    return mask_utils.decode(segm)


def to_rgb_gray(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("L"), dtype=np.uint8)
    return np.stack([arr, arr, arr], axis=-1)


def overlay_mask(base_rgb: np.ndarray, mask: np.ndarray, color: tuple, alpha: float) -> np.ndarray:
    if mask is None:
        return base_rgb
    mask_bool = mask.astype(bool)
    if mask_bool.ndim == 3:
        mask_bool = mask_bool[:, :, 0]
    overlay = base_rgb.copy()
    for c in range(3):
        overlay[:, :, c] = np.where(
            mask_bool,
            (1 - alpha) * overlay[:, :, c] + alpha * color[c],
            overlay[:, :, c],
        )
    return overlay.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize COCO segm predictions with mask overlay."
    )
    parser.add_argument("--pred_json", required=True, type=Path, help="COCO prediction json (segm).")
    parser.add_argument("--gt_json", required=True, type=Path, help="COCO GT json (for image paths).")
    parser.add_argument("--img_root", required=True, type=Path, help="Image root folder.")
    parser.add_argument("--out_dir", required=True, type=Path, help="Output directory for PNGs.")
    parser.add_argument("--limit", type=int, default=50, help="Max number of images to render.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay alpha.")
    args = parser.parse_args()

    pred = json.load(open(args.pred_json, "r"))
    gt = json.load(open(args.gt_json, "r"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    img_id_to_info = {img["id"]: img for img in gt["images"]}
    cat_id_to_name = {cat["id"]: cat["name"] for cat in gt["categories"]}

    # Simple fixed colors for up to 10 classes
    palette = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 128, 255),
        (255, 128, 0),
        (128, 0, 255),
        (0, 255, 255),
        (255, 0, 255),
        (128, 128, 0),
        (0, 128, 128),
        (128, 0, 128),
    ]

    # group predictions by image
    by_img = {}
    for ann in pred:
        by_img.setdefault(ann["image_id"], []).append(ann)

    rendered = 0
    for img_id, anns in by_img.items():
        if img_id not in img_id_to_info:
            continue
        img_info = img_id_to_info[img_id]
        img_path = args.img_root / img_info["file_name"]
        if not img_path.exists():
            continue

        img = Image.open(img_path)
        canvas = to_rgb_gray(img)

        # sort by score (high to low) for nicer overlay
        anns = sorted(anns, key=lambda x: x.get("score", 0.0), reverse=True)
        for ann in anns:
            segm = ann.get("segmentation")
            mask = decode_rle(segm)
            cat_id = ann.get("category_id", 0)
            color = palette[(cat_id - 1) % len(palette)]
            canvas = overlay_mask(canvas, mask, color=color, alpha=args.alpha)

        name = Path(img_info["file_name"]).stem
        out_path = args.out_dir / f"{name}_pred.png"
        Image.fromarray(canvas).save(out_path)
        rendered += 1
        if rendered >= args.limit:
            break

    print(f"Rendered {rendered} images to {args.out_dir}")


if __name__ == "__main__":
    main()

