#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


FRAME_RE = re.compile(r"frame(\d+)", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare ACDC data for SAM3 (export ED/ES slices and cine videos)."
    )
    parser.add_argument(
        "--acdc_root",
        required=True,
        help="Path to ACDC dataset root (contains training/testing folders).",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output root for prepared images/videos.",
    )
    parser.add_argument(
        "--patients",
        nargs="*",
        default=None,
        help="Optional patient IDs to process (e.g., patient001 patient002).",
    )
    parser.add_argument(
        "--slice_stride",
        type=int,
        default=1,
        help="Stride for selecting slices along z-axis (default: 1).",
    )
    parser.add_argument(
        "--max_slices",
        type=int,
        default=None,
        help="Optional cap on number of slices per volume.",
    )
    parser.add_argument(
        "--percentile_low",
        type=float,
        default=1.0,
        help="Lower percentile for intensity clipping (default: 1).",
    )
    parser.add_argument(
        "--percentile_high",
        type=float,
        default=99.0,
        help="Upper percentile for intensity clipping (default: 99).",
    )
    parser.add_argument(
        "--sample_stride",
        type=int,
        default=50,
        help="Stride for sampling voxels when estimating percentiles (default: 50).",
    )
    parser.add_argument(
        "--skip_videos",
        action="store_true",
        help="Skip exporting cine video slices.",
    )
    parser.add_argument(
        "--skip_ed_es",
        action="store_true",
        help="Skip exporting ED/ES slices.",
    )
    parser.add_argument(
        "--skip_gt",
        action="store_true",
        help="Skip exporting ED/ES GT masks as 2D PNGs.",
    )
    parser.add_argument(
        "--skip_gt_vis",
        action="store_true",
        help="Skip exporting visualization-friendly GT masks.",
    )
    parser.add_argument(
        "--no_4d_for_video",
        action="store_true",
        help="Disable using *_4d.nii.gz for full cine frames.",
    )
    return parser.parse_args()


def find_patient_dirs(acdc_root: Path):
    patient_dirs = []
    for root, _, files in os.walk(acdc_root):
        frame_files = [
            f
            for f in files
            if f.lower().endswith(".nii.gz")
            and "frame" in f.lower()
            and "_gt" not in f.lower()
        ]
        if frame_files:
            patient_dirs.append(Path(root))
    return sorted(patient_dirs)


def parse_info_cfg(patient_dir: Path):
    info_path = None
    for name in ["Info.cfg", "info.cfg"]:
        candidate = patient_dir / name
        if candidate.exists():
            info_path = candidate
            break
    ed_frame = None
    es_frame = None
    if info_path and info_path.exists():
        for line in info_path.read_text().splitlines():
            if ":" not in line:
                continue
            key, value = [token.strip() for token in line.split(":", 1)]
            if key.upper() == "ED":
                ed_frame = int(value)
            elif key.upper() == "ES":
                es_frame = int(value)
    return ed_frame, es_frame


def extract_frame_index(filename: str):
    match = FRAME_RE.search(filename)
    if not match:
        return None
    return int(match.group(1))


def get_split_name(patient_dir: Path):
    lower = str(patient_dir).lower()
    if "training" in lower or "train" in lower:
        return "train"
    if "testing" in lower or "test" in lower:
        return "test"
    return "unspecified"


# 获取指定患者目录下所有原始（非_gt）.nii.gz帧文件，并按frame索引排序后返回
def get_frame_files(patient_dir: Path):
    frame_files = []
    for file in patient_dir.iterdir():
        if not file.name.lower().endswith(".nii.gz"):
            continue
        if "_gt" in file.name.lower():
            continue
        frame_idx = extract_frame_index(file.name)
        if frame_idx is None:
            continue
        frame_files.append((frame_idx, file))
    frame_files.sort(key=lambda x: x[0])
    return frame_files


def estimate_intensity_range(frame_files, percentile_low, percentile_high, sample_stride):
    """
    估算一组帧的像素强度范围（vmin, vmax），通过指定的百分位，在所有帧的像素采样中获得。
    Args:
        frame_files: List[(frame_idx, Path)]，所有帧的路径列表
        percentile_low: 最低强度的百分位（如0.5）
        percentile_high: 最高强度的百分位（如99.5）
        sample_stride: 采样步长（>1表示下采样）
    Returns:
        (vmin, vmax): 浮点型的最小值和最大值
    """
    samples = []
    for _, path in frame_files:
        data = nib.load(str(path)).get_fdata()
        flat = data.ravel()
        if sample_stride > 1:
            flat = flat[::sample_stride]
        samples.append(flat)
    if not samples:
        return 0.0, 1.0
    samples = np.concatenate(samples, axis=0)
    vmin = np.percentile(samples, percentile_low)
    vmax = np.percentile(samples, percentile_high)
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def estimate_intensity_range_4d(fourd_path, percentile_low, percentile_high, sample_stride):
    """
    估算4D体数据整体的一组强度范围（vmin, vmax），通过百分位截断异常值。
    Args:
        fourd_path: 4D nii 文件路径
        percentile_low: 最低强度百分位
        percentile_high: 最高强度百分位
        sample_stride: 采样步长
    Returns:
        (vmin, vmax): 浮点型的最小最大值
    """
    data = nib.load(str(fourd_path)).get_fdata()
    flat = data.ravel()
    if sample_stride > 1:
        flat = flat[::sample_stride]
    vmin = np.percentile(flat, percentile_low)
    vmax = np.percentile(flat, percentile_high)
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def normalize_volume(volume, vmin, vmax):
    """
    归一化3D体数据到0-255的uint8范围，先按vmin/vmax截断，再归一化。
    Args:
        volume: 输入3D体数据（numpy数组）
        vmin: 最小值
        vmax: 最大值
    Returns:
        归一化后的uint8类型体数据
    """
    volume = np.clip(volume, vmin, vmax)
    volume = (volume - vmin) / (vmax - vmin)
    volume = np.clip(volume * 255.0, 0, 255).astype(np.uint8)
    return volume


def slice_indices(z_dim, slice_stride, max_slices):
    """
    获取沿z轴的切片索引，按照步长和最大数量采样。
    Args:
        z_dim: z轴总层数
        slice_stride: 层步长
        max_slices: 最大切片数（可为None）
    Returns:
        切片索引列表
    """
    indices = list(range(0, z_dim, slice_stride))
    if max_slices is not None:
        indices = indices[:max_slices]
    return indices


def save_slice_image(slice_array, out_path: Path):
    """
    保存二维切片为图片（如png）。
    Args:
        slice_array: numpy二维数组
        out_path: 输出图片路径
    """
    Image.fromarray(slice_array).save(out_path)


def save_vis_mask(mask_slice, out_path: Path):
    """
    将mask切片可视化并保存，将标签（0-3）缩放到0-255。
    Args:
        mask_slice: 原标签切片
        out_path: 输出图片路径
    """
    # Scale labels (0-3) into 0-255 for easier visualization
    vis = (mask_slice.astype(np.uint8) * 85).clip(0, 255)
    Image.fromarray(vis).save(out_path)


def load_gt_volume(patient_dir: Path, patient_id: str, frame_idx: int):
    """
    加载指定病人的某一帧的gt（标签）体数据。如果不存在则返回None。
    Args:
        patient_dir: 病人目录
        patient_id: 病人ID
        frame_idx: 帧序号
    Returns:
        gt体数据（numpy数组）或None
    """
    gt_name = f"{patient_id}_frame{frame_idx:02d}_gt.nii.gz"
    gt_path = patient_dir / gt_name
    if not gt_path.exists():
        return None
    return nib.load(str(gt_path)).get_fdata()


def prepare_patient(
    patient_dir: Path,
    output_root: Path,
    patients_filter,
    slice_stride,
    max_slices,
    percentile_low,
    percentile_high,
    sample_stride,
    skip_videos,
    skip_ed_es,
    skip_gt,
    skip_gt_vis,
    use_4d_for_video,
):
    patient_id = patient_dir.name
    if patients_filter and patient_id not in patients_filter:
        return

    frame_files = get_frame_files(patient_dir)
    if not frame_files:
        return

    ed_frame, es_frame = parse_info_cfg(patient_dir)
    split = get_split_name(patient_dir)
    out_patient_root = output_root / split / patient_id
    out_images_root = out_patient_root / "images"
    out_masks_root = out_patient_root / "masks"
    out_masks_vis_root = out_patient_root / "masks_vis"
    out_videos_root = out_patient_root / "videos"
    fourd_path = patient_dir / f"{patient_id}_4d.nii.gz"

    if use_4d_for_video and fourd_path.exists():
        vmin, vmax = estimate_intensity_range_4d(
            fourd_path, percentile_low, percentile_high, sample_stride
        )
    else:
        vmin, vmax = estimate_intensity_range(
            frame_files, percentile_low, percentile_high, sample_stride
        )

    video_frames = []
    z_dim = None
    if not skip_videos and use_4d_for_video and fourd_path.exists():
        volume_4d = nib.load(str(fourd_path)).get_fdata()
        z_dim = volume_4d.shape[2]
        num_frames = volume_4d.shape[3]
        for t in range(num_frames):
            volume_u8 = normalize_volume(volume_4d[..., t], vmin, vmax)
            video_frames.append((t, volume_u8))

    for frame_idx, frame_path in frame_files:
        volume = nib.load(str(frame_path)).get_fdata()
        if z_dim is None:
            z_dim = volume.shape[-1]
        volume_u8 = normalize_volume(volume, vmin, vmax)

        if not skip_ed_es and frame_idx in {ed_frame, es_frame}:
            phase = "ED" if frame_idx == ed_frame else "ES"
            phase_dir = out_images_root / phase
            phase_dir.mkdir(parents=True, exist_ok=True)
            mask_dir = out_masks_root / phase
            mask_dir.mkdir(parents=True, exist_ok=True)
            mask_vis_dir = out_masks_vis_root / phase
            mask_vis_dir.mkdir(parents=True, exist_ok=True)
            gt_volume = None
            if not skip_gt:
                gt_volume = load_gt_volume(patient_dir, patient_id, frame_idx)
            for z in slice_indices(volume_u8.shape[-1], slice_stride, max_slices):
                out_path = phase_dir / f"slice_{z:02d}.png"
                # save_slice_image(volume_u8[..., z], out_path)
                if gt_volume is not None:
                    mask_slice = gt_volume[..., z].astype(np.uint8)
                    # 如果该mask全是0，直接跳过导出
                    if np.all(mask_slice == 0):
                        continue
                    mask_path = mask_dir / f"slice_{z:02d}.png"
                    save_slice_image(mask_slice, mask_path)
                    save_slice_image(volume_u8[..., z], out_path)
                    if not skip_gt_vis:
                        mask_vis_path = mask_vis_dir / f"slice_{z:02d}.png"
                        save_vis_mask(mask_slice, mask_vis_path)

    if not skip_videos and video_frames:
        out_videos_root.mkdir(parents=True, exist_ok=True)
        ordered = sorted(video_frames, key=lambda x: x[0])
        num_frames = len(ordered)
        for z in slice_indices(z_dim, slice_stride, max_slices):
            slice_dir = out_videos_root / f"slice_{z:02d}"
            slice_dir.mkdir(parents=True, exist_ok=True)
            for idx, (frame_idx, volume_u8) in enumerate(ordered):
                out_path = slice_dir / f"frame_{idx:02d}.png"
                save_slice_image(volume_u8[..., z], out_path)

    meta = {
        "patient_id": patient_id,
        "split": split,
        "num_frames": len(frame_files),
        "ed_frame": ed_frame,
        "es_frame": es_frame,
        "slice_stride": slice_stride,
        "max_slices": max_slices,
        "percentile_low": percentile_low,
        "percentile_high": percentile_high,
        "intensity_min": vmin,
        "intensity_max": vmax,
        "exported_gt": not skip_gt,
        "exported_gt_vis": not skip_gt_vis,
        "use_4d_for_video": use_4d_for_video,
    }
    out_patient_root.mkdir(parents=True, exist_ok=True)
    with (out_patient_root / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)


def main():
    args = parse_args()
    acdc_root = Path(args.acdc_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    patient_dirs = find_patient_dirs(acdc_root)
    for patient_dir in patient_dirs:
        prepare_patient(
            patient_dir=patient_dir,
            output_root=output_root,
            patients_filter=set(args.patients) if args.patients else None,
            slice_stride=args.slice_stride,
            max_slices=args.max_slices,
            percentile_low=args.percentile_low,
            percentile_high=args.percentile_high,
            sample_stride=args.sample_stride,
            skip_videos=args.skip_videos,
            skip_ed_es=args.skip_ed_es,
            skip_gt=args.skip_gt,
            skip_gt_vis=args.skip_gt_vis,
            use_4d_for_video=not args.no_4d_for_video,
        )


if __name__ == "__main__":
    main()

