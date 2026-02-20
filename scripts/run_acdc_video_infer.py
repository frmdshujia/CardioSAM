#!/usr/bin/env python3
"""
run_acdc_video_infer.py

使用 SAM3 视频预测器对 ACDC 数据做批量推理。支持两种模式：

1. 使用训练 checkpoint（如 checkpoint_50.pt）：
   权重存储在 {"model": {"backbone.*": ..., "detector.*": ...}} 格式中，
   自动适配为视频模型的 detector.* 前缀。

2. 使用原始 SAM3 权重（如 sam3/sam3.pt）：
   直接加载，适合作为 baseline。

每个 slice 视频推理结果输出至：
  <output_root>/<split>/<patient>/<slice>/
    masks/frame_XXXX.png          # 二值 mask（uint8，目标=255）
    masks_npz/frame_XXXX.npz     # float32 mask（obj_id, score, bbox 均记录）
    frame_results.json            # 逐帧 obj_id/score/bbox 信息
    video_meta.json               # 患者元数据（spacing/ED/ES/帧序等）

使用示例（adapter checkpoint）：
  python scripts/run_acdc_video_infer.py \\
    --manifest outputs/annotations/acdc_video_split_manifest.json \\
    --checkpoint outputs/acdc_adapter_280_bz8_ts350/checkpoints/checkpoint_50.pt \\
    --checkpoint_type adapter \\
    --base_checkpoint sam3/sam3.pt \\
    --output_root outputs/acdc_video_infer_adapter \\
    --split test \\
    --prompt "left ventricle"

  注：adapter 推理始终在 image_size=1008 下运行（与 baseline 一致），保证 tracker
  权重（来自 --base_checkpoint）与 backbone 输出特征尺寸匹配。PromptGenerator 通过
  双线性插值适配 1008px 推理，无需与训练分辨率（280px）一致。

使用示例（baseline）：
  python scripts/run_acdc_video_infer.py \\
    --manifest outputs/annotations/acdc_video_split_manifest.json \\
    --checkpoint sam3/sam3.pt \\
    --checkpoint_type baseline \\
    --output_root outputs/acdc_video_infer_baseline \\
    --split test
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Checkpoint 加载适配
# ---------------------------------------------------------------------------

def load_adapted_checkpoint(checkpoint_path: str, checkpoint_type: str, strict: bool = False):
    """
    加载 checkpoint，返回可直接 load_state_dict 的 state_dict。

    checkpoint_type:
      - "adapter": 训练容器格式 {"model": {"backbone.*": ...}}, 需加 detector. 前缀
      - "baseline": 原始 sam3.pt 格式，直接使用
    """
    import torch
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if checkpoint_type == "adapter":
        # 取出 model 字段
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            model_dict = ckpt["model"]
        else:
            model_dict = ckpt
        # 添加 detector. 前缀（视频模型中 detector 是子模块）
        adapted = {f"detector.{k}": v for k, v in model_dict.items()}
        # 过滤 freqs_cis：RoPE 缓存依赖 image_size，由模型配置重新生成（参考 _load_checkpoint）
        adapted = {k: v for k, v in adapted.items() if "freqs_cis" not in k}
        return adapted

    elif checkpoint_type == "baseline":
        # 直接使用原始格式
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        else:
            sd = ckpt
        # 同样过滤 freqs_cis
        sd = {k: v for k, v in sd.items() if "freqs_cis" not in k}
        return sd

    else:
        raise ValueError(f"Unknown checkpoint_type: {checkpoint_type!r}. Use 'adapter' or 'baseline'.")


# ---------------------------------------------------------------------------
# 视频预测器构建（复用 run_baseline_video.py 的思路）
# ---------------------------------------------------------------------------

def build_predictor(
    checkpoint_path: str,
    checkpoint_type: str,
    device: str,
    image_size: int = 1008,
    base_checkpoint: str = None,
):
    """
    构建 Sam3VideoPredictorMultiGPU，加载权重，返回 predictor。

    adapter 模式加载策略（两阶段）：
      1. 先加载 base_checkpoint（如 sam3/sam3.pt），初始化 tracker 和 backbone
         基础权重（不含 prompt_generator），使 tracker 可以正常工作。
      2. 再叠加 adapter checkpoint（含 prompt_generator + 微调后的 transformer
         / segmentation_head），覆盖 detector 部分权重。

    PromptGenerator 虽在 280px 下训练，但其内部用 _resize_handcrafted 做双线性
    插值，可在 1008px 推理时正常运行（空间频率略有差异，但不影响功能）。

    baseline 模式：直接加载 sam3.pt，无需两阶段处理。
    """
    import torch
    from sam3.model_builder import build_sam3_video_predictor

    adapter_cfg = None
    if checkpoint_type == "adapter":
        adapter_cfg = {
            "enable_adapter": True,
            "tuning_stage": "1234",
            "handcrafted_tune": True,
            "embedding_tune": True,
            "adaptor": "adaptor",
        }

    # 始终在 image_size=1008 下构建模型，保证 tracker 与 backbone 特征尺寸一致
    # （tracker.sam_image_embedding_size = 1008 // 14 = 72）
    effective_image_size = 1008
    if checkpoint_type == "baseline":
        effective_image_size = image_size  # baseline 可自定义

    predictor = build_sam3_video_predictor(
        checkpoint_path=None,
        adapter_cfg=adapter_cfg,
        image_size=effective_image_size,
    )

    print(f"[ckpt] Checkpoint       : {checkpoint_path}")
    print(f"[ckpt] Type             : {checkpoint_type}")
    print(f"[ckpt] Model image_size : {effective_image_size}")
    print(f"[ckpt] Adapter enabled  : {adapter_cfg is not None}")

    # ── 阶段 1：adapter 模式先加载 base checkpoint 初始化 tracker ──────────────
    if checkpoint_type == "adapter" and base_checkpoint:
        print(f"[ckpt] Loading base checkpoint for tracker init: {base_checkpoint}")
        base_ckpt = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
        if "model" in base_ckpt and isinstance(base_ckpt["model"], dict):
            base_ckpt = base_ckpt["model"]
        base_ckpt = {k: v for k, v in base_ckpt.items() if "freqs_cis" not in k}
        miss_b, unexp_b = predictor.model.load_state_dict(base_ckpt, strict=False)
        print(f"[ckpt]   Base  missing={len(miss_b)}, unexpected={len(unexp_b)}")
    elif checkpoint_type == "adapter":
        print("[ckpt] WARNING: no --base_checkpoint provided; tracker will be randomly "
              "initialized and propagation quality will be poor.")

    # ── 阶段 2：加载目标 checkpoint（adapter 或 baseline）──────────────────────
    adapted_dict = load_adapted_checkpoint(checkpoint_path, checkpoint_type)
    missing, unexpected = predictor.model.load_state_dict(adapted_dict, strict=False)
    print(f"[ckpt] Adapter/target  missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print("       (first 20 missing):", missing[:20])
    if unexpected:
        print("       (first 20 unexpected):", unexpected[:20])

    predictor.model.eval()
    return predictor


# ---------------------------------------------------------------------------
# 单个视频推理
# ---------------------------------------------------------------------------

def run_single_video(
    predictor,
    video_dir: Path,
    prompt: str,
    frame_index: int,
    output_dir: Path,
    video_meta: Dict,
    save_npz: bool = False,
    save_overlay: bool = False,
    overlay_alpha: float = 0.4,
):
    """
    对单个切片视频目录执行推理，保存逐帧 mask 与 JSON 结果。
    返回 (成功, 帧数)。
    """
    import torch

    # ---- 启动 session ----
    try:
        response = predictor.handle_request(
            request={"type": "start_session", "resource_path": str(video_dir)}
        )
    except Exception as e:
        print(f"  [ERROR] start_session failed: {e}")
        return False, 0

    session_id = response["session_id"]

    # ---- 添加文本 prompt ----
    try:
        predictor.handle_request(
            request={
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": frame_index,
                "text": prompt,
            }
        )
    except Exception as e:
        import traceback as _tb
        print(f"  [ERROR] add_prompt failed: {type(e).__name__}: {e}")
        print("  [ERROR] Full traceback:")
        _tb.print_exc()
        predictor.handle_request({"type": "close_session", "session_id": session_id})
        return False, 0

    # ---- 向前传播 ----
    outputs_by_frame: Dict[int, Dict] = {}
    try:
        stream = predictor.handle_stream_request(
            request={
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
                "start_frame_index": frame_index,
            }
        )
        for event in stream:
            fi = event["frame_index"]
            outputs_by_frame[fi] = event["outputs"]
    except Exception as e:
        print(f"  [ERROR] propagate_in_video failed: {e}")
        traceback.print_exc()
        predictor.handle_request({"type": "close_session", "session_id": session_id})
        return False, 0

    # ---- 关闭 session ----
    try:
        predictor.handle_request({"type": "close_session", "session_id": session_id})
    except Exception:
        pass

    if not outputs_by_frame:
        print("  [WARN] No outputs produced.")
        return True, 0

    # ---- 保存结果 ----
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    if save_npz:
        npz_dir = output_dir / "masks_npz"
        npz_dir.mkdir(parents=True, exist_ok=True)

    if save_overlay:
        overlay_dir = output_dir / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)

    frame_results = []

    # 如需 overlay，载入原始帧图片
    frame_images: Dict[int, np.ndarray] = {}
    if save_overlay:
        for png in sorted(video_dir.glob("frame_*.png")):
            idx_str = png.stem.replace("frame_", "")
            try:
                fi = int(idx_str)
                frame_images[fi] = np.array(Image.open(png).convert("RGB"))
            except Exception:
                pass

    for frame_idx in sorted(outputs_by_frame.keys()):
        out = outputs_by_frame[frame_idx]
        raw_masks = out.get("out_binary_masks", [])
        obj_ids = out.get("out_obj_ids", list(range(len(raw_masks))))
        scores = out.get("out_scores", [None] * len(raw_masks))

        frame_info = {"frame_idx": frame_idx, "objects": []}
        combined_mask = None  # 合并二值 mask（多对象时按 obj_id 着色）

        for idx, (raw_mask, obj_id, score) in enumerate(
            zip(raw_masks, obj_ids, scores)
        ):
            # 转为 numpy
            if hasattr(raw_mask, "cpu"):
                mask_np = raw_mask.detach().cpu().float().numpy()
            else:
                mask_np = np.array(raw_mask, dtype=np.float32)

            if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                mask_np = mask_np[0]

            binary = (mask_np > 0.5).astype(np.uint8)

            # bbox
            ys, xs = np.where(binary)
            if len(xs) > 0:
                bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                area = int(binary.sum())
            else:
                bbox = None
                area = 0

            score_val = float(score) if score is not None else None
            frame_info["objects"].append(
                {
                    "obj_id": int(obj_id) if obj_id is not None else idx,
                    "score": score_val,
                    "bbox_xyxy": bbox,
                    "area": area,
                }
            )

            # 合并 mask（取第一个对象或按 obj_id 着色）
            if combined_mask is None:
                combined_mask = binary * 255
            else:
                combined_mask = np.maximum(combined_mask, binary * 255)

            if save_npz:
                np.savez_compressed(
                    str(npz_dir / f"frame_{frame_idx:04d}.npz"),
                    mask=mask_np,
                    binary=binary,
                    obj_id=np.array([obj_id]),
                    score=np.array([score_val] if score_val is not None else [float("nan")]),
                )

        # 保存二值 mask PNG
        if combined_mask is not None:
            Image.fromarray(combined_mask.astype(np.uint8)).save(
                masks_dir / f"frame_{frame_idx:04d}.png"
            )

        # overlay
        if save_overlay and frame_idx in frame_images:
            img = frame_images[frame_idx].copy()
            colors = [
                (230, 25, 75),
                (60, 180, 75),
                (255, 225, 25),
                (0, 130, 200),
                (245, 130, 48),
            ]
            for idx2, raw_mask in enumerate(raw_masks):
                if hasattr(raw_mask, "cpu"):
                    m = raw_mask.detach().cpu().float().numpy()
                else:
                    m = np.array(raw_mask, dtype=np.float32)
                if m.ndim == 3 and m.shape[0] == 1:
                    m = m[0]
                b = m > 0.5
                color = np.array(colors[idx2 % len(colors)], dtype=np.float32)
                img[b] = (overlay_alpha * color + (1 - overlay_alpha) * img[b]).astype(np.uint8)
            Image.fromarray(img).save(overlay_dir / f"frame_{frame_idx:04d}_overlay.png")

        frame_results.append(frame_info)

    # 保存逐帧 JSON
    with (output_dir / "frame_results.json").open("w") as f:
        json.dump({"video_dir": str(video_dir), "frames": frame_results}, f, indent=2)

    # 保存视频级元数据 JSON
    with (output_dir / "video_meta.json").open("w") as f:
        json.dump(
            {
                "video_dir": str(video_dir),
                "prompt": prompt,
                "prompt_frame_index": frame_index,
                **video_meta,
            },
            f,
            indent=2,
        )

    return True, len(frame_results)


# ---------------------------------------------------------------------------
# 批量推理
# ---------------------------------------------------------------------------

def run_batch(
    predictor,
    manifest: Dict,
    splits: List[str],
    video_root: Path,
    output_root: Path,
    prompt: str,
    save_npz: bool,
    save_overlay: bool,
    patients_filter: Optional[List[str]],
    resume: bool,
):
    total_success = 0
    total_fail = 0

    for split_name in splits:
        if split_name not in manifest:
            print(f"[WARN] Split '{split_name}' not in manifest. Skipping.")
            continue

        split_data = manifest[split_name]
        slices_info: Dict = split_data.get("slices", {})
        patient_list: List[str] = split_data.get("patients", list(slices_info.keys()))

        if patients_filter:
            patient_list = [p for p in patient_list if p in set(patients_filter)]

        print(f"\n{'='*60}")
        print(f"Split: {split_name}  ({len(patient_list)} patients)")
        print(f"{'='*60}")

        for patient_id in patient_list:
            if patient_id not in slices_info:
                print(f"  [WARN] {patient_id}: no slice info in manifest.")
                continue

            pinfo = slices_info[patient_id]
            video_paths: List[str] = pinfo.get("video_paths", [])
            pmeta: Dict = pinfo.get("meta", {})

            if not video_paths:
                print(f"  [WARN] {patient_id}: no video_paths in manifest.")
                continue

            print(f"\n  Patient: {patient_id}  ({len(video_paths)} slices)")

            # ED帧作为初始 prompt 帧（若有则使用，否则用第 0 帧）
            ed_export_idx = pmeta.get("ed_frame_export_idx")
            prompt_frame = int(ed_export_idx) if ed_export_idx is not None else 0

            for rel_path in video_paths:
                video_dir = video_root / rel_path
                if not video_dir.exists():
                    print(f"    [WARN] Video dir not found: {video_dir}")
                    continue

                slice_name = video_dir.name  # e.g. slice_00
                out_dir = output_root / split_name / patient_id / slice_name

                if resume and (out_dir / "frame_results.json").exists():
                    print(f"    [{slice_name}] Already done, skipping (--resume).")
                    total_success += 1
                    continue

                out_dir.mkdir(parents=True, exist_ok=True)

                t0 = time.time()
                ok, n_frames = run_single_video(
                    predictor=predictor,
                    video_dir=video_dir,
                    prompt=prompt,
                    frame_index=prompt_frame,
                    output_dir=out_dir,
                    video_meta={
                        "patient_id": patient_id,
                        "split": split_name,
                        "slice": slice_name,
                        **pmeta,
                    },
                    save_npz=save_npz,
                    save_overlay=save_overlay,
                )
                elapsed = time.time() - t0

                if ok:
                    total_success += 1
                    print(f"    [{slice_name}] OK  {n_frames} frames  {elapsed:.1f}s")
                else:
                    total_fail += 1
                    print(f"    [{slice_name}] FAILED  {elapsed:.1f}s")

    print(f"\n{'='*60}")
    print(f"Done. Success: {total_success}, Failed: {total_fail}")
    return total_success, total_fail


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch ACDC video inference with SAM3 (checkpoint adaptation).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to acdc_video_split_manifest.json produced by create_acdc_video_split.py.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint file (checkpoint_50.pt or sam3.pt).",
    )
    parser.add_argument(
        "--checkpoint_type",
        choices=["adapter", "baseline"],
        required=True,
        help=(
            "'adapter': training checkpoint container (keys like backbone.*), "
            "auto-prefixed with detector.*. "
            "'baseline': original sam3.pt direct format."
        ),
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Root directory for inference results.",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        default=["val"],
        choices=["train", "val", "test", "all"],
        help="Which split(s) to run. Use 'all' to run train+val+test. Default: val.",
    )
    parser.add_argument(
        "--prompt",
        default="left ventricle",
        help="Text prompt for SAM3 (default: 'left ventricle').",
    )
    parser.add_argument(
        "--patients",
        nargs="*",
        default=None,
        help="Optional patient filter (e.g. patient001 patient002).",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=1008,
        help=(
            "Inference image size (default: 1008). For adapter type this is ignored "
            "and always set to 1008 to keep tracker compatible with sam3.pt weights."
        ),
    )
    parser.add_argument(
        "--base_checkpoint",
        default=None,
        help=(
            "Path to baseline sam3.pt, used as Phase-1 weight initialization for "
            "adapter inference (initializes tracker + backbone before loading adapter). "
            "Recommended: sam3/sam3.pt. If omitted, tracker will be randomly initialized."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device (default: cuda).",
    )
    parser.add_argument(
        "--save_npz",
        action="store_true",
        help="Also save float32 mask as .npz per frame.",
    )
    parser.add_argument(
        "--save_overlay",
        action="store_true",
        help="Also save overlay visualization per frame.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip slices that already have frame_results.json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    with manifest_path.open() as f:
        manifest = json.load(f)

    video_root = Path(manifest.get("video_root", manifest_path.parent)).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # 解析 split 参数
    splits = args.split
    if "all" in splits:
        splits = ["train", "val", "test"]

    print(f"Manifest   : {manifest_path}")
    print(f"Video root : {video_root}")
    print(f"Output root: {output_root}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Type       : {args.checkpoint_type}")
    print(f"Splits     : {splits}")
    print(f"Prompt     : {args.prompt!r}")
    print()

    # 构建预测器（加载 checkpoint）
    predictor = build_predictor(
        args.checkpoint,
        args.checkpoint_type,
        args.device,
        image_size=args.image_size,
        base_checkpoint=args.base_checkpoint,
    )

    # 保存推理配置到输出目录
    config = {
        "manifest": str(manifest_path),
        "checkpoint": args.checkpoint,
        "checkpoint_type": args.checkpoint_type,
        "splits": splits,
        "prompt": args.prompt,
        "device": args.device,
        "save_npz": args.save_npz,
        "save_overlay": args.save_overlay,
    }
    with (output_root / "infer_config.json").open("w") as f:
        json.dump(config, f, indent=2)

    # 执行批量推理
    run_batch(
        predictor=predictor,
        manifest=manifest,
        splits=splits,
        video_root=video_root,
        output_root=output_root,
        prompt=args.prompt,
        save_npz=args.save_npz,
        save_overlay=args.save_overlay,
        patients_filter=args.patients,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
