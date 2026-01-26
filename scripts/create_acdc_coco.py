#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_LABEL_MAP = {
    1: "right ventricle",
    2: "myocardium",
    3: "left ventricle",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create COCO-style annotations from ACDC ED/ES masks."
    )
    parser.add_argument(
        "--input_root",
        required=True,
        help="Root folder containing train/test patient folders (sam3input).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write COCO JSON files.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Ratio of patients reserved for val (default: 0.1).",
    )
    parser.add_argument(
        "--val_patients",
        nargs="*",
        default=None,
        help="Optional explicit val patient IDs (e.g., patient091 patient092).",
    )
    return parser.parse_args()


def rle_encode(mask: np.ndarray):
    # COCO expects RLE over column-major order
    pixels = mask.flatten(order="F").astype(np.uint8)
    counts = []
    prev = 0
    run_len = 0
    for val in pixels:
        if val == prev:
            run_len += 1
        else:
            counts.append(run_len)
            run_len = 1
            prev = val
    counts.append(run_len)
    return {"counts": counts, "size": list(mask.shape)}


def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x_min = int(xs.min())
    y_min = int(ys.min())
    x_max = int(xs.max())
    y_max = int(ys.max())
    return [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]


def list_patients(train_root: Path):
    patients = sorted([p.name for p in train_root.iterdir() if p.is_dir()])
    return patients


def split_patients(patients, val_ratio, val_patients):
    if val_patients:
        val_set = set(val_patients)
        train_set = [p for p in patients if p not in val_set]
        return train_set, list(val_set)
    num_val = max(1, int(round(len(patients) * val_ratio)))
    val_set = patients[-num_val:]
    train_set = patients[: len(patients) - num_val]
    return train_set, val_set


def build_coco(split_root: Path, patient_ids, label_map):
    images = []
    annotations = []
    categories = [{"id": k, "name": v} for k, v in label_map.items()]

    image_id = 1
    ann_id = 1

    for patient_id in patient_ids:
        patient_dir = split_root / patient_id
        for phase in ["ED", "ES"]:
            image_dir = patient_dir / "images" / phase
            mask_dir = patient_dir / "masks" / phase
            if not image_dir.exists() or not mask_dir.exists():
                continue
            for image_path in sorted(image_dir.glob("slice_*.png")):
                mask_path = mask_dir / image_path.name
                if not mask_path.exists():
                    continue
                img = Image.open(image_path)
                width, height = img.size
                rel_path = image_path.relative_to(split_root.parent).as_posix()
                images.append(
                    {
                        "id": image_id,
                        "file_name": rel_path,
                        "width": width,
                        "height": height,
                    }
                )

                mask = np.array(Image.open(mask_path))
                for label_value, label_name in label_map.items():
                    binary = mask == label_value
                    area = int(binary.sum())
                    if area == 0:
                        continue
                    bbox = bbox_from_mask(binary)
                    if bbox is None:
                        continue
                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": label_value,
                            "bbox": bbox,
                            "segmentation": rle_encode(binary),
                            "area": area,
                            "iscrowd": 0,
                            "noun_phrase": label_name,
                        }
                    )
                    ann_id += 1

                image_id += 1

    return {"images": images, "annotations": annotations, "categories": categories}


def main():
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_root = input_root / "train"
    test_root = input_root / "test"
    patients = list_patients(train_root)
    train_patients, val_patients = split_patients(
        patients, args.val_ratio, args.val_patients
    )

    label_map = DEFAULT_LABEL_MAP

    train_coco = build_coco(train_root, train_patients, label_map)
    val_coco = build_coco(train_root, val_patients, label_map)

    with (output_dir / "acdc_train.json").open("w") as f:
        json.dump(train_coco, f, indent=2)
    with (output_dir / "acdc_val.json").open("w") as f:
        json.dump(val_coco, f, indent=2)

    if test_root.exists():
        test_patients = list_patients(test_root)
        test_coco = build_coco(test_root, test_patients, label_map)
        with (output_dir / "acdc_test.json").open("w") as f:
            json.dump(test_coco, f, indent=2)
        print("Test patients:", len(test_patients))
        print("Wrote:", output_dir / "acdc_test.json")

    print("Train patients:", len(train_patients), "Val patients:", len(val_patients))
    print("Wrote:", output_dir / "acdc_train.json")
    print("Wrote:", output_dir / "acdc_val.json")


if __name__ == "__main__":
    main()


