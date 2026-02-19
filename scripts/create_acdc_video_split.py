#!/usr/bin/env python3
"""
create_acdc_video_split.py

根据已导出的 ACDC 视频数据（sam3input 格式），生成 train/val 分层抽样清单。

分层抽样策略与 create_acdc_coco.py 完全一致：
  - 从 ACDC 原始数据库 Info.cfg 读取疾病分组（NOR/MINF/DCM/HCM/RV）
  - 每组抽取 val_per_group 个患者作为验证集（seed=42）
  - 输出 acdc_video_split_manifest.json

输出格式示例：
{
  "seed": 42,
  "val_per_group": 2,
  "acdc_db_root": "...",
  "video_root": "...",
  "train": {
    "patients": ["patient001", ...],
    "slices": {
      "patient001": {
        "video_paths": ["train/patient001/videos/slice_00", ...],
        "meta": { "ed_frame": 1, "es_frame": 12, "video_num_frames": 30,
                  "pixel_spacing_mm": [1.56, 1.56], "slice_thickness_mm": 8.0,
                  "ed_frame_export_idx": 0, "es_frame_export_idx": 11 }
      },
      ...
    }
  },
  "val": { ... }
}
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib

PATHOLOGIES: List[str] = ["NOR", "MINF", "DCM", "HCM", "RV"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ACDC video train/val split manifest with stratified sampling."
    )
    parser.add_argument(
        "--video_root",
        required=True,
        help="Root of prepared ACDC video data (e.g. sam3input/), contains train/test subdirs.",
    )
    parser.add_argument(
        "--acdc_db_root",
        default=None,
        help=(
            "Root of original ACDC database (contains training/testing patient dirs with Info.cfg). "
            "Default: <repo_root>/database"
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write acdc_video_split_manifest.json.",
    )
    parser.add_argument(
        "--val_per_group",
        type=int,
        default=2,
        help="Number of val patients per disease group (default: 2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for patient split (default: 42).",
    )
    parser.add_argument(
        "--val_patients",
        nargs="*",
        default=None,
        help="Explicit val patient IDs (overrides stratified split).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print split details without writing manifest.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 分组读取（复用 create_acdc_coco.py 的逻辑）
# ---------------------------------------------------------------------------

def normalize_pathology(group_value: str) -> Optional[str]:
    if group_value is None:
        return None
    return group_value.strip().upper()


def parse_patient_group_from_info_cfg(info_cfg_path: Path) -> Optional[str]:
    if not info_cfg_path.exists():
        return None
    for line in info_cfg_path.read_text().splitlines():
        if line.strip().lower().startswith("group"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                return parts[1].strip()
    return None


def find_info_cfg(acdc_db_root: Path, patient_id: str) -> Optional[Path]:
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


def build_patient_group_map(
    patients: List[str], acdc_db_root: Path
) -> Tuple[Dict[str, str], List[str]]:
    group_map: Dict[str, str] = {}
    missing: List[str] = []
    for pid in patients:
        info_cfg = find_info_cfg(acdc_db_root, pid)
        if info_cfg is None:
            missing.append(pid)
            continue
        group = parse_patient_group_from_info_cfg(info_cfg)
        group_norm = normalize_pathology(group)
        if group_norm not in PATHOLOGIES:
            missing.append(pid)
            continue
        group_map[pid] = group_norm
    return group_map, missing


def split_patients(
    patients: List[str],
    val_per_group: int,
    seed: int,
    acdc_db_root: Path,
    val_patients_override: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    import random

    if val_patients_override:
        val_set = set(val_patients_override)
        train_list = [p for p in patients if p not in val_set]
        return train_list, sorted(val_set)

    group_map, missing = build_patient_group_map(patients, acdc_db_root)

    if len(group_map) == len(patients) and val_per_group > 0:
        groups: Dict[str, List[str]] = defaultdict(list)
        for pid in patients:
            groups[group_map[pid]].append(pid)

        rng = random.Random(seed)
        val_list: List[str] = []
        for gname, plist in sorted(groups.items()):
            if len(plist) < val_per_group:
                raise ValueError(
                    f"Group '{gname}' has only {len(plist)} patients, "
                    f"cannot sample {val_per_group} for val."
                )
            plist_sorted = sorted(plist)
            rng.shuffle(plist_sorted)
            picked = sorted(plist_sorted[:val_per_group])
            val_list.extend(picked)

        val_set = set(val_list)
        train_list = [p for p in patients if p not in val_set]
        return train_list, sorted(val_set)

    # fallback: ratio-based
    if missing:
        print(
            f"[WARN] Could not read Group for {len(missing)}/{len(patients)} patients. "
            "Falling back to ratio-based split (last 20%)."
        )
    num_val = max(1, int(round(len(patients) * 0.1)))
    return patients[: len(patients) - num_val], patients[-num_val:]


# ---------------------------------------------------------------------------
# 读取患者 meta（从已导出的 meta.json 或从数据库 .nii.gz 推断）
# ---------------------------------------------------------------------------

def read_patient_meta(
    patient_video_dir: Path, patient_id: str, acdc_db_root: Optional[Path]
) -> Dict:
    """
    优先读取已导出的 meta.json，缺失 spacing 时从数据库 .nii.gz 补充。
    """
    meta_path = patient_video_dir / "meta.json"
    meta: Dict = {}
    if meta_path.exists():
        with meta_path.open() as f:
            meta = json.load(f)

    # 补充 spacing（若 meta.json 中没有）
    if "pixel_spacing_mm" not in meta and acdc_db_root is not None:
        spacing = _read_spacing_from_db(acdc_db_root, patient_id)
        if spacing:
            meta.update(spacing)

    # 确保关键字段存在
    meta.setdefault("patient_id", patient_id)
    meta.setdefault("video_num_frames", None)
    meta.setdefault("ed_frame", None)
    meta.setdefault("es_frame", None)
    meta.setdefault("ed_frame_export_idx", None)
    meta.setdefault("es_frame_export_idx", None)
    meta.setdefault("pixel_spacing_mm", [1.0, 1.0])
    meta.setdefault("slice_thickness_mm", 1.0)

    # 若 video_num_frames 仍缺失，从磁盘推断
    if meta.get("video_num_frames") is None:
        videos_dir = patient_video_dir / "videos"
        if videos_dir.exists():
            slice_dirs = sorted(videos_dir.iterdir())
            if slice_dirs:
                frames = sorted(slice_dirs[0].glob("frame_*.png"))
                meta["video_num_frames"] = len(frames)

    # 推断 ED/ES 在导出帧序列中的索引
    if meta.get("ed_frame_export_idx") is None and meta.get("ed_frame") is not None:
        if meta.get("frame_index_map"):
            # 新版 meta.json 有 frame_index_map（key=export_idx, val=4D_t_0indexed）
            # ACDC ed_frame 是 1-indexed，4D 时间帧是 0-indexed，故 4D_t = ed_frame - 1
            inv = {v: k for k, v in meta["frame_index_map"].items()}
            meta["ed_frame_export_idx"] = inv.get(meta["ed_frame"] - 1)
            meta["es_frame_export_idx"] = inv.get(meta["es_frame"] - 1)
        else:
            # 旧版 meta.json 无 frame_index_map：ACDC frame 编号 1-indexed，export 0-indexed
            # 4D export 帧序号 = ACDC 原始帧编号 - 1
            meta["ed_frame_export_idx"] = max(0, meta["ed_frame"] - 1)
            meta["es_frame_export_idx"] = max(0, meta["es_frame"] - 1)

    return meta


def _read_spacing_from_db(acdc_db_root: Path, patient_id: str) -> Optional[Dict]:
    """从数据库中读取体素间距。"""
    for subdir in ["training", "testing", "train", "test", ""]:
        p = acdc_db_root / subdir / patient_id if subdir else acdc_db_root / patient_id
        fourd = p / f"{patient_id}_4d.nii.gz"
        if fourd.exists():
            try:
                img = nib.load(str(fourd))
                z = img.header.get_zooms()
                return {
                    "pixel_spacing_mm": [float(z[0]), float(z[1])],
                    "slice_thickness_mm": float(z[2]),
                }
            except Exception:
                pass
        # 尝试第一帧
        for name in p.glob(f"{patient_id}_frame*.nii.gz") if p.exists() else []:
            if "_gt" not in name.name:
                try:
                    img = nib.load(str(name))
                    z = img.header.get_zooms()
                    return {
                        "pixel_spacing_mm": [float(z[0]), float(z[1])],
                        "slice_thickness_mm": float(z[2]),
                    }
                except Exception:
                    pass
                break
    return None


# ---------------------------------------------------------------------------
# 构建清单
# ---------------------------------------------------------------------------

def build_patient_slice_info(
    patient_video_dir: Path,
    patient_id: str,
    split_name: str,
    acdc_db_root: Optional[Path],
) -> Dict:
    """
    收集单个患者的切片路径列表与元数据，返回可嵌入 manifest 的 dict。
    """
    meta = read_patient_meta(patient_video_dir, patient_id, acdc_db_root)

    videos_dir = patient_video_dir / "videos"
    video_paths: List[str] = []
    if videos_dir.exists():
        for slice_dir in sorted(videos_dir.iterdir()):
            if slice_dir.is_dir():
                # 存储相对于 video_root 的路径（split/patient/videos/slice_XX）
                video_paths.append(f"{split_name}/{patient_id}/videos/{slice_dir.name}")

    summary_meta = {
        "ed_frame": meta.get("ed_frame"),
        "es_frame": meta.get("es_frame"),
        "ed_frame_export_idx": meta.get("ed_frame_export_idx"),
        "es_frame_export_idx": meta.get("es_frame_export_idx"),
        "video_num_frames": meta.get("video_num_frames"),
        "pixel_spacing_mm": meta.get("pixel_spacing_mm", [1.0, 1.0]),
        "slice_thickness_mm": meta.get("slice_thickness_mm", 1.0),
    }

    return {
        "video_paths": video_paths,
        "meta": summary_meta,
    }


def build_split_section(
    patients: List[str],
    split_root: Path,
    split_name: str,
    acdc_db_root: Optional[Path],
) -> Dict:
    slices: Dict[str, Dict] = {}
    for pid in patients:
        patient_dir = split_root / pid
        if not patient_dir.exists():
            print(f"[WARN] Patient dir not found: {patient_dir}")
            continue
        slices[pid] = build_patient_slice_info(patient_dir, pid, split_name, acdc_db_root)

    return {
        "patients": patients,
        "slices": slices,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    video_root = Path(args.video_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    default_db_root = Path(__file__).resolve().parents[1] / "database"
    acdc_db_root = (
        Path(args.acdc_db_root).expanduser().resolve()
        if args.acdc_db_root
        else default_db_root
    )

    train_root = video_root / "train"
    if not train_root.exists():
        raise FileNotFoundError(f"train split not found: {train_root}")

    # 收集所有训练患者
    all_patients = sorted(
        [d.name for d in train_root.iterdir() if d.is_dir()]
    )
    print(f"Found {len(all_patients)} training patients under {train_root}")

    # 分层抽样
    train_patients, val_patients = split_patients(
        all_patients,
        val_per_group=args.val_per_group,
        seed=args.seed,
        acdc_db_root=acdc_db_root,
        val_patients_override=args.val_patients if args.val_patients else None,
    )

    # 打印分组详情
    group_map, _ = build_patient_group_map(all_patients, acdc_db_root)
    if group_map:
        val_groups: Dict[str, List[str]] = defaultdict(list)
        for pid in val_patients:
            if pid in group_map:
                val_groups[group_map[pid]].append(pid)
        print("Val groups:")
        for gname, plist in sorted(val_groups.items()):
            print(f"  {gname}: {len(plist)} -> {sorted(plist)}")

    print(f"Train: {len(train_patients)} patients, Val: {len(val_patients)} patients")
    print(f"Val patient IDs: {sorted(val_patients)}")

    # 收集测试集患者（test 目录，若存在）
    test_root = video_root / "test"
    test_patients: List[str] = []
    if test_root.exists():
        test_patients = sorted([d.name for d in test_root.iterdir() if d.is_dir()])
        print(f"Found {len(test_patients)} test patients under {test_root}")

    if args.dry_run:
        return

    # 构建 manifest
    train_section = build_split_section(
        train_patients, train_root, "train", acdc_db_root
    )
    val_section = build_split_section(
        val_patients, train_root, "train", acdc_db_root  # val 患者也在 train split 目录下
    )

    # 统计切片总数
    train_slice_count = sum(len(v["video_paths"]) for v in train_section["slices"].values())
    val_slice_count = sum(len(v["video_paths"]) for v in val_section["slices"].values())

    manifest = {
        "seed": args.seed,
        "val_per_group": args.val_per_group,
        "acdc_db_root": str(acdc_db_root),
        "video_root": str(video_root),
        "train_patient_count": len(train_patients),
        "val_patient_count": len(val_patients),
        "train_slice_count": train_slice_count,
        "val_slice_count": val_slice_count,
        "train": train_section,
        "val": val_section,
    }

    # 加入测试集
    if test_patients:
        test_section = build_split_section(
            test_patients, test_root, "test", acdc_db_root
        )
        test_slice_count = sum(len(v["video_paths"]) for v in test_section["slices"].values())
        manifest["test_patient_count"] = len(test_patients)
        manifest["test_slice_count"] = test_slice_count
        manifest["test"] = test_section
        print(f"Test slices: {test_slice_count}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "acdc_video_split_manifest.json"
    with out_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to: {out_path}")
    print(f"  Train slices: {train_slice_count}, Val slices: {val_slice_count}")


if __name__ == "__main__":
    main()
