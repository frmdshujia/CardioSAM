#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


COCO_METRICS = [
    "AP",
    "AP_50",
    "AP_75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_maxDets@1",
    "AR_maxDets@10",
    "AR_maxDets@100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def _decode_segmentation(segm, height: int, width: int) -> Optional[np.ndarray]:
    if segm is None:
        return None
    if isinstance(segm, dict):
    if isinstance(segm.get("counts"), list):
            rle = mask_utils.frPyObjects(segm, height, width)
            return mask_utils.decode(rle)
    return mask_utils.decode(segm)
    if isinstance(segm, list):
        rle = mask_utils.frPyObjects(segm, height, width)
        return mask_utils.decode(rle)
    return None


def _collapse_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    if mask.ndim == 3:
        return (mask.sum(axis=2) > 0).astype(np.uint8)
    return mask.astype(np.uint8)


def to_rgb_gray(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("L"), dtype=np.uint8)
    return np.stack([arr, arr, arr], axis=-1)


def overlay_mask(
    base_rgb: np.ndarray, mask: Optional[np.ndarray], color: tuple, alpha: float
) -> np.ndarray:
    if mask is None:
        return base_rgb
    mask_bool = mask.astype(bool)
    overlay = base_rgb.copy()
    for c in range(3):
        overlay[:, :, c] = np.where(
            mask_bool,
            (1 - alpha) * overlay[:, :, c] + alpha * color[c],
            overlay[:, :, c],
        )
    return overlay.astype(np.uint8)


def evaluate_coco(gt_json: Path, pred_json: Path, iou_type: str = "segm") -> Dict[str, float]:
    coco_gt = COCO(str(gt_json))
    coco_dt = coco_gt.loadRes(str(pred_json))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {f"coco_eval_{iou_type}_{COCO_METRICS[i]}": v for i, v in enumerate(coco_eval.stats)}


def render_overlays(
    img_path: Path,
    img_info: Dict,
    anns: List[Dict],
    palette: List[Tuple[int, int, int]],
    alpha: float,
    score_thresh: float,
    max_preds_per_image: Optional[int],
) -> np.ndarray:
    img = Image.open(img_path)
    canvas = to_rgb_gray(img)
    height, width = img_info["height"], img_info["width"]
    anns = sorted(anns, key=lambda x: x.get("score", 0.0), reverse=True)
    rendered = 0
    for ann in anns:
        if ann.get("score", 0.0) < score_thresh:
            continue
        segm = ann.get("segmentation")
        mask = _collapse_mask(_decode_segmentation(segm, height, width))
        cat_id = ann.get("category_id", 0)
        color = palette[(cat_id - 1) % len(palette)]
        canvas = overlay_mask(canvas, mask, color=color, alpha=alpha)
        rendered += 1
        if max_preds_per_image and rendered >= max_preds_per_image:
            break
    return canvas


def render_gt_overlay(
    img_path: Path,
    img_info: Dict,
    gt_anns: List[Dict],
    palette: List[Tuple[int, int, int]],
    alpha: float,
) -> np.ndarray:
    img = Image.open(img_path)
    canvas = to_rgb_gray(img)
    height, width = img_info["height"], img_info["width"]
    for ann in gt_anns:
        segm = ann.get("segmentation")
        mask = _collapse_mask(_decode_segmentation(segm, height, width))
        cat_id = ann.get("category_id", 0)
        color = palette[(cat_id - 1) % len(palette)]
        canvas = overlay_mask(canvas, mask, color=color, alpha=alpha)
    return canvas


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
    parser.add_argument("--gt_alpha", type=float, default=0.35, help="GT overlay alpha.")
    parser.add_argument("--score_thresh", type=float, default=0.0, help="Score threshold.")
    parser.add_argument(
        "--max_preds_per_image", type=int, default=None, help="Max preds per image."
    )
    parser.add_argument("--show_gt", action="store_true", help="Also render GT overlay.")
    parser.add_argument("--eval", action="store_true", help="Run COCO eval and log metrics.")
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity.")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name.")
    parser.add_argument("--wandb_dir", type=Path, default=None, help="Wandb dir.")
    parser.add_argument("--wandb_mode", type=str, default=None, help="Wandb mode (online/offline).")
    parser.add_argument("--wandb_group", type=str, default=None, help="Wandb group.")
    parser.add_argument("--wandb_job_type", type=str, default=None, help="Wandb job type.")
    parser.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated wandb tags.")
    parser.add_argument(
        "--wandb_max_images", type=int, default=50, help="Max images to log to wandb."
    )
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

    gt_by_img = {}
    for ann in gt.get("annotations", []):
        gt_by_img.setdefault(ann["image_id"], []).append(ann)

    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_kwargs = {
            "project": args.wandb_project,
            "name": args.wandb_name,
            "entity": args.wandb_entity,
            "dir": str(args.wandb_dir) if args.wandb_dir else None,
            "mode": args.wandb_mode,
            "group": args.wandb_group,
            "job_type": args.wandb_job_type,
            "tags": [t for t in (args.wandb_tags or "").split(",") if t],
        }
        wandb_run = wandb.init(**{k: v for k, v in wandb_kwargs.items() if v is not None})
        wandb.config.update(
            {
                "pred_json": str(args.pred_json),
                "gt_json": str(args.gt_json),
                "img_root": str(args.img_root),
                "limit": args.limit,
                "alpha": args.alpha,
                "gt_alpha": args.gt_alpha,
                "score_thresh": args.score_thresh,
                "max_preds_per_image": args.max_preds_per_image,
                "show_gt": args.show_gt,
            },
            allow_val_change=True,
        )

    eval_metrics = {}
    if args.eval:
        eval_metrics = evaluate_coco(args.gt_json, args.pred_json, iou_type="segm")
        for k, v in eval_metrics.items():
            print(f"{k}: {v:.4f}")
        if wandb_run:
            wandb_run.log(eval_metrics)

    wandb_images = []
    wandb_rows = []
    rendered = 0
    for img_id, anns in by_img.items():
        if img_id not in img_id_to_info:
            continue
        img_info = img_id_to_info[img_id]
        img_path = args.img_root / img_info["file_name"]
        if not img_path.exists():
            continue

        canvas = render_overlays(
            img_path,
            img_info,
            anns,
            palette,
            alpha=args.alpha,
            score_thresh=args.score_thresh,
            max_preds_per_image=args.max_preds_per_image,
        )
        if args.show_gt:
            gt_canvas = render_gt_overlay(
                img_path,
                img_info,
                gt_by_img.get(img_id, []),
                palette,
                alpha=args.gt_alpha,
            )
            canvas = np.concatenate([canvas, gt_canvas], axis=1)

        name = Path(img_info["file_name"]).stem
        out_path = args.out_dir / f"{name}_pred.png"
        Image.fromarray(canvas).save(out_path)
        if wandb_run and len(wandb_images) < args.wandb_max_images:
            wandb_images.append(wandb.Image(canvas, caption=img_info["file_name"]))
            scores = [a.get("score", 0.0) for a in anns]
            wandb_rows.append(
                [
                    img_id,
                    img_info["file_name"],
                    len(anns),
                    float(np.mean(scores)) if scores else 0.0,
                    float(np.max(scores)) if scores else 0.0,
                ]
            )
        rendered += 1
        if rendered >= args.limit:
            break

    if wandb_run:
        if wandb_images:
            wandb_run.log({"pred_overlays": wandb_images})
        if wandb_rows:
            wandb_run.log(
                {
                    "pred_summary": wandb.Table(
                        columns=["image_id", "file_name", "num_preds", "mean_score", "max_score"],
                        data=wandb_rows,
                    )
                }
            )
        wandb_run.finish()
    print(f"Rendered {rendered} images to {args.out_dir}")


if __name__ == "__main__":
    main()

