#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAM3 video baseline on a frame folder or mp4.")
    parser.add_argument(
        "--video_path",
        required=True,
        help="Path to a folder of frames or an mp4 video file.",
    )
    parser.add_argument(
        "--prompt",
        default="left ventricle",
        help="Text prompt for SAM3.",
    )
    parser.add_argument(
        "--frame_index",
        type=int,
        default=0,
        help="Frame index for the initial prompt.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save overlay frames.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=20,
        help="Max frames to save overlays for (from propagated results).",
    )
    return parser.parse_args()


def overlay_masks(frame_array, outputs, alpha=0.4):
    image = Image.fromarray(frame_array).convert("RGB")
    img = np.array(image)
    colors = [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
    ]
    masks = outputs.get("out_binary_masks", [])
    for idx, mask in enumerate(masks):
        color = np.array(colors[idx % len(colors)], dtype=np.uint8)
        if hasattr(mask, "cpu"):
            mask = mask.detach().cpu().numpy()
        mask = mask > 0.5
        img[mask] = (alpha * color + (1 - alpha) * img[mask]).astype(np.uint8)
    return Image.fromarray(img)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    from sam3.model_builder import build_sam3_video_predictor
    from sam3.model.io_utils import load_resource_as_video_frames

    video_predictor = build_sam3_video_predictor()

    response = video_predictor.handle_request(
        request={"type": "start_session", "resource_path": args.video_path}
    )
    session_id = response["session_id"]

    video_predictor.handle_request(
        request={
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": args.frame_index,
            "text": args.prompt,
        }
    )

    frame_limit = args.max_frames if args.max_frames is not None else None
    outputs_by_frame = {}
    stream = video_predictor.handle_stream_request(
        request={
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": "forward",
            "start_frame_index": args.frame_index,
            "max_frame_num_to_track": frame_limit,
        }
    )
    for event in stream:
        outputs_by_frame[event["frame_index"]] = event["outputs"]
        if frame_limit is not None and len(outputs_by_frame) >= frame_limit:
            break

    image_size = video_predictor.model.image_size
    frames, _, _ = load_resource_as_video_frames(
        resource_path=args.video_path,
        image_size=image_size,
        offload_video_to_cpu=True,
        img_mean=(0.5, 0.5, 0.5),
        img_std=(0.5, 0.5, 0.5),
    )
    mean = torch.tensor([0.5, 0.5, 0.5])[:, None, None]
    std = torch.tensor([0.5, 0.5, 0.5])[:, None, None]

    for frame_idx in sorted(outputs_by_frame.keys()):
        frame = frames[frame_idx].cpu().float()
        frame = (frame * std + mean).clamp(0, 1)
        frame = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        overlay = overlay_masks(frame, outputs_by_frame[frame_idx])
        out_path = output_dir / f"frame_{frame_idx:03d}_overlay.png"
        overlay.save(out_path)


if __name__ == "__main__":
    main()

