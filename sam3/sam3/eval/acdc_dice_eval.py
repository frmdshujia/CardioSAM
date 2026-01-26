# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
#
# pyre-unsafe

import logging
from typing import Dict, List, Optional

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from sam3.train.utils.distributed import is_main_process


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


def _merge_masks(anns: List[Dict], height: int, width: int) -> np.ndarray:
    if not anns:
        return np.zeros((height, width), dtype=np.uint8)
    merged = np.zeros((height, width), dtype=np.uint8)
    for ann in anns:
        mask = _decode_segmentation(ann.get("segmentation"), height, width)
        if mask is None:
            continue
        if mask.ndim == 3:
            mask = (mask.sum(axis=2) > 0).astype(np.uint8)
        merged = np.maximum(merged, mask.astype(np.uint8))
    return merged


class ACDCDiceEvaluator:
    """
    Compute Dice for RV/MYO/LV and mDice using COCO GT + prediction json.
    """

    def __init__(self, gt_path: str, pred_score_thresh: float = 0.0):
        self.gt_path = gt_path
        self.pred_score_thresh = float(pred_score_thresh)

    def evaluate(self, dumped_file):
        if not is_main_process():
            return {}

        coco_gt = COCO(self.gt_path)
        coco_dt = coco_gt.loadRes(str(dumped_file))

        name_to_id = {c["name"].lower(): c["id"] for c in coco_gt.loadCats(coco_gt.getCatIds())}
        target = {
            "rv": "right ventricle",
            "myo": "myocardium",
            "lv": "left ventricle",
        }

        cat_ids = {}
        for short, name in target.items():
            cat_id = name_to_id.get(name.lower())
            if cat_id is None:
                logging.warning(f"ACDC Dice: category '{name}' not found in GT.")
            else:
                cat_ids[short] = cat_id

        img_ids = coco_gt.getImgIds()
        dice_scores: Dict[str, List[float]] = {k: [] for k in cat_ids.keys()}

        for img_id in img_ids:
            img_info = coco_gt.loadImgs([img_id])[0]
            height, width = int(img_info["height"]), int(img_info["width"])
            for short, cat_id in cat_ids.items():
                gt_ann_ids = coco_gt.getAnnIds(imgIds=[img_id], catIds=[cat_id])
                dt_ann_ids = coco_dt.getAnnIds(imgIds=[img_id], catIds=[cat_id])
                gt_anns = coco_gt.loadAnns(gt_ann_ids)
                dt_anns = coco_dt.loadAnns(dt_ann_ids)
                if self.pred_score_thresh > 0:
                    dt_anns = [
                        a for a in dt_anns if float(a.get("score", 0.0)) >= self.pred_score_thresh
                    ]

                gt_mask = _merge_masks(gt_anns, height, width)
                dt_mask = _merge_masks(dt_anns, height, width)

                gt_sum = int(gt_mask.sum())
                dt_sum = int(dt_mask.sum())
                if gt_sum == 0 and dt_sum == 0:
                    continue
                denom = gt_sum + dt_sum
                dice = 0.0 if denom == 0 else (2.0 * float((gt_mask & dt_mask).sum()) / denom)
                dice_scores[short].append(dice)

        outs = {}
        per_class = []
        for short in ["rv", "myo", "lv"]:
            scores = dice_scores.get(short, [])
            value = float(np.mean(scores)) if scores else 0.0
            outs[f"acdc_dice_{short}"] = value
            if scores:
                per_class.append(value)

        outs["acdc_mdice"] = float(np.mean(per_class)) if per_class else 0.0
        return outs


