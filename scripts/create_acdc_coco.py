#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


DEFAULT_LABEL_MAP = {
    1: "right ventricle",
    2: "myocardium",
    3: "left ventricle",
}

PATHOLOGIES: List[str] = ["NOR", "MINF", "DCM", "HCM", "RV"]
BASE_CAT_TO_STRUCT = {1: "RV", 2: "MYO", 3: "LV"}


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
        "--val_per_group",
        type=int,
        default=2,
        help="When using group-stratified split, number of val patients per group (default: 2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for patient split (default: 42).",
    )
    parser.add_argument(
        "--acdc_db_root",
        default=None,
        help=(
            "Root folder of ACDC database containing training/testing patient folders "
            "(used to read Info.cfg for Group stratification). "
            "Default: <repo_root>/database"
        ),
    )
    parser.add_argument(
        "--val_patients",
        nargs="*",
        default=None,
        help="Optional explicit val patient IDs (e.g., patient091 patient092).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print train/val split (and group stats) without writing COCO JSON.",
    )
    parser.add_argument(
        "--annotation_version",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help=(
            "Annotation detail level (default: 0). "
            "0: 3 anatomy categories (RV/MYO/LV). "
            "1: Keep 3 categories, but write pathology-aware per-annotation noun_phrase (v1). "
            "2: Keep 3 categories, but write pathology-aware per-annotation noun_phrase (v2). "
            "Versions 1/2 require reading patient disease Group from Info.cfg under --acdc_db_root."
        ),
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


def parse_patient_group_from_info_cfg(info_cfg_path: Path):
    if not info_cfg_path.exists():
        return None
    for line in info_cfg_path.read_text().splitlines():
        if line.strip().lower().startswith("group"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                return parts[1].strip()
    return None


def find_info_cfg(acdc_db_root: Path, patient_id: str):
    # ACDC convention: database/{training,testing}/patientXXX/Info.cfg
    candidates = [
        acdc_db_root / "training" / patient_id / "Info.cfg",
        acdc_db_root / "testing" / patient_id / "Info.cfg",
        acdc_db_root / "train" / patient_id / "Info.cfg",
        acdc_db_root / "test" / patient_id / "Info.cfg",
        acdc_db_root / patient_id / "Info.cfg",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def build_patient_group_map(patients, acdc_db_root: Path):
    group_map = {}
    missing = []
    for pid in patients:
        info_cfg = find_info_cfg(acdc_db_root, pid)
        if info_cfg is None:
            missing.append(pid)
            continue
        group = parse_patient_group_from_info_cfg(info_cfg)
        if group is None:
            missing.append(pid)
            continue
        group_norm = normalize_pathology(group)
        if group_norm not in PATHOLOGIES:
            missing.append(pid)
            continue
        group_map[pid] = group_norm
    return group_map, missing


def normalize_pathology(group_value: str) -> Optional[str]:
    if group_value is None:
        return None
    return group_value.strip().upper()


def make_pathology_aware_noun_phrase(annotation_version: int, *, pathology: str, base_cat_id: int) -> str:
    """
    Keep category_id as anatomy (1/2/3), but enrich per-annotation noun_phrase with
    pathology-aware phrasing (v1/v2).
    """
    if annotation_version not in (1, 2):
        raise ValueError(f"Unsupported annotation_version for noun_phrase: {annotation_version}")
    subj = {
        "NOR": "a normal subject",
        "MINF": "a post-myocardial-infarction patient",
        "DCM": "a dilated cardiomyopathy patient",
        "HCM": "a hypertrophic cardiomyopathy patient",
        "RV": "a patient with right ventricular abnormality",
    }
    if annotation_version == 1:
        struct_phrase = {
            "LV": "left ventricle",
            "RV": "right ventricle",
            "MYO": "outer ring-shaped myocardium surrounding the left ventricle",
        }
    else:
        struct_phrase = {
            "LV": "elliptical left ventricle",
            "RV": "crescent-shaped right ventricle",
            "MYO": "outer ring-shaped myocardium surrounding the left ventricle",
        }
    s = BASE_CAT_TO_STRUCT.get(int(base_cat_id))
    if s is None:
        raise ValueError(f"Unexpected base category id: {base_cat_id}")
    if pathology == "HCM" and s == "MYO":
        return f"{subj[pathology]}'s thickened outer ring-shaped myocardium surrounding the left ventricle"
    return f"{subj[pathology]}'s {struct_phrase[s]}"


def split_patients(
    patients,
    val_ratio,
    val_patients,
    *,
    acdc_db_root: Path,
    val_per_group: int,
    seed: int,
):
    if val_patients:
        val_set = set(val_patients)
        train_set = [p for p in patients if p not in val_set]
        return train_set, list(val_set)

    # Prefer group-stratified split when ACDC Info.cfg is available.
    group_map, missing = build_patient_group_map(patients, acdc_db_root)
    if len(group_map) == len(patients) and val_per_group > 0:
        groups = defaultdict(list)
        for pid in patients:
            groups[group_map[pid]].append(pid)

        rng = random.Random(seed)
        val_list = []
        for gname, plist in sorted(groups.items(), key=lambda kv: kv[0]):
            if len(plist) < val_per_group:
                raise ValueError(
                    f"Group '{gname}' has only {len(plist)} patients, cannot sample {val_per_group} for val."
                )
            plist_sorted = sorted(plist)
            rng.shuffle(plist_sorted)
            picked = sorted(plist_sorted[:val_per_group])
            val_list.extend(picked)

        val_set = set(val_list)
        train_set = [p for p in patients if p not in val_set]
        return train_set, sorted(val_set)

    # Fallback: deterministic ratio split (kept for compatibility)
    if missing:
        print(
            f"[WARN] Could not read Group for {len(missing)}/{len(patients)} patients from '{acdc_db_root}'. "
            f"Falling back to ratio-based split."
        )
    num_val = max(1, int(round(len(patients) * val_ratio)))
    val_set = patients[-num_val:]
    train_set = patients[: len(patients) - num_val]
    return train_set, val_set


def build_pathology_map(patient_ids, acdc_db_root: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    missing = []
    unknown = []
    for pid in patient_ids:
        info_cfg = find_info_cfg(acdc_db_root, pid)
        if info_cfg is None:
            missing.append(pid)
            continue
        group = parse_patient_group_from_info_cfg(info_cfg)
        pathology = normalize_pathology(group) if group is not None else None
        if pathology is None:
            missing.append(pid)
            continue
        if pathology not in PATHOLOGIES:
            unknown.append((pid, pathology))
            continue
        out[pid] = pathology
    if missing:
        raise ValueError(
            "Missing Info.cfg Group for some patients (needed for annotation_version>=1). "
            f"Example missing: {sorted(missing)[:5]} (total {len(missing)}) "
            f"under acdc_db_root='{acdc_db_root}'."
        )
    if unknown:
        sample = ", ".join([f"{pid}:{g}" for pid, g in unknown[:5]])
        raise ValueError(
            "Unknown pathology Group tokens in Info.cfg (needed for annotation_version>=1). "
            f"Example: {sample} (total {len(unknown)}). Expected one of {PATHOLOGIES}."
        )
    return out


def build_coco(
    split_root: Path,
    patient_ids,
    label_map,
    *,
    annotation_version: int,
    acdc_db_root: Path,
):
    images = []
    annotations = []
    need_pathology = annotation_version in (1, 2)
    pathology_map: Dict[str, str] = (
        build_pathology_map(patient_ids, acdc_db_root) if need_pathology else {}
    )
    categories = [{"id": k, "name": v} for k, v in label_map.items()]

    image_id = 1
    ann_id = 1

    for patient_id in patient_ids:
        patient_dir = split_root / patient_id
        pathology = pathology_map.get(patient_id) if need_pathology else None
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
                    if annotation_version == 0:
                        category_id = label_value
                        noun_phrase = label_name
                        base_category_id = None
                    elif annotation_version in (1, 2):
                        if pathology is None:
                            raise ValueError(
                                f"Missing pathology for patient '{patient_id}' "
                                f"(annotation_version={annotation_version})."
                            )
                        category_id = int(label_value)  # keep 3-cat
                        noun_phrase = make_pathology_aware_noun_phrase(
                            annotation_version,
                            pathology=pathology,
                            base_cat_id=int(label_value),
                        )
                        base_category_id = int(label_value)
                    else:
                        raise ValueError(f"Unsupported annotation_version: {annotation_version}")
                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "bbox": bbox,
                            "segmentation": rle_encode(binary),
                            "area": area,
                            "iscrowd": 0,
                            "noun_phrase": noun_phrase,
                            **(
                                {}
                                if annotation_version == 0
                                else {
                                    "pathology": pathology,
                                    "base_category_id": base_category_id,
                                }
                            ),
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

    default_db_root = Path(__file__).resolve().parents[1] / "database"
    acdc_db_root = Path(args.acdc_db_root).resolve() if args.acdc_db_root else default_db_root

    train_root = input_root / "train"
    test_root = input_root / "test"
    patients = list_patients(train_root)
    train_patients, val_patients = split_patients(
        patients,
        args.val_ratio,
        args.val_patients,
        acdc_db_root=acdc_db_root,
        val_per_group=args.val_per_group,
        seed=args.seed,
    )

    label_map = DEFAULT_LABEL_MAP

    # Print split details early for reproducibility (useful with --dry_run)
    try:
        group_map, _missing = build_patient_group_map(patients, acdc_db_root)
        if group_map and len(group_map) == len(patients):
            val_groups = defaultdict(list)
            for pid in val_patients:
                val_groups[group_map[pid]].append(pid)
            print("Val groups:")
            for gname, plist in sorted(val_groups.items(), key=lambda kv: kv[0]):
                print(f"  {gname}: {len(plist)} -> {sorted(plist)}")
    except Exception as e:
        print("[WARN] Could not summarize groups for val split:", str(e))

    print("Train patients:", len(train_patients), "Val patients:", len(val_patients))
    print("Val patient IDs:", sorted(val_patients))

    if args.dry_run:
        return

    suffix = ""
    if args.annotation_version == 1:
        suffix = "_np_en_v1"
    elif args.annotation_version == 2:
        suffix = "_np_en_v2"

    # Write split manifest with the same suffix to avoid cross-version overwrites.
    split_manifest = {
        "input_root": str(input_root),
        "acdc_db_root": str(acdc_db_root),
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "val_per_group": int(args.val_per_group),
        "annotation_version": int(args.annotation_version),
        "train_patients": list(train_patients),
        "val_patients": list(val_patients),
    }
    if test_root.exists():
        split_manifest["test_patients"] = list(list_patients(test_root))
    with (output_dir / f"acdc_split_manifest{suffix}.json").open("w") as f:
        json.dump(split_manifest, f, indent=2)

    train_coco = build_coco(
        train_root,
        train_patients,
        label_map,
        annotation_version=args.annotation_version,
        acdc_db_root=acdc_db_root,
    )
    val_coco = build_coco(
        train_root,
        val_patients,
        label_map,
        annotation_version=args.annotation_version,
        acdc_db_root=acdc_db_root,
    )

    with (output_dir / f"acdc_train{suffix}.json").open("w") as f:
        json.dump(train_coco, f, indent=2)
    with (output_dir / f"acdc_val{suffix}.json").open("w") as f:
        json.dump(val_coco, f, indent=2)

    if test_root.exists():
        test_patients = list_patients(test_root)
        test_coco = build_coco(
            test_root,
            test_patients,
            label_map,
            annotation_version=args.annotation_version,
            acdc_db_root=acdc_db_root,
        )
        with (output_dir / f"acdc_test{suffix}.json").open("w") as f:
            json.dump(test_coco, f, indent=2)
        print("Test patients:", len(test_patients))
        print("Wrote:", output_dir / f"acdc_test{suffix}.json")

    print("Wrote:", output_dir / f"acdc_train{suffix}.json")
    print("Wrote:", output_dir / f"acdc_val{suffix}.json")
    print("Wrote:", output_dir / f"acdc_split_manifest{suffix}.json")


if __name__ == "__main__":
    main()


