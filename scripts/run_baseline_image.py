#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAM3 image baseline on PNG/JPG.")
    parser.add_argument(
        "--image_path",
        help="Path to a single image (PNG/JPG). If set, --image_dir is ignored.",
    )
    parser.add_argument(
        "--image_dir",
        help="Directory containing images for batch inference.",
    )
    parser.add_argument(
        "--prompt",
        default="left ventricle",
        help="Text prompt for SAM3.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save overlays.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=5,
        help="Max images to process when using --image_dir.",
    )
    return parser.parse_args()


def load_image_paths(image_path, image_dir, max_images):
    if image_path:
        return [Path(image_path)]
    if not image_dir:
        raise ValueError("Please provide either --image_path or --image_dir.")
    image_dir = Path(image_dir)
    exts = {".png", ".jpg", ".jpeg"}
    images = [p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in exts]
    if max_images is not None:
        images = images[: max_images]
    return images


def overlay_masks(image, masks, alpha=0.4):
    image = image.convert("RGB")
    img = np.array(image)
    colors = [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
    ]
    for idx, mask in enumerate(masks):
        color = np.array(colors[idx % len(colors)], dtype=np.uint8)
        mask = mask.squeeze()
        if hasattr(mask, "cpu"):
            mask = mask.detach().cpu().numpy()
        mask = mask > 0.5
        img[mask] = (alpha * color + (1 - alpha) * img[mask]).astype(np.uint8)
    return Image.fromarray(img)


def main():
    args = parse_args()
    image_paths = load_image_paths(args.image_path, args.image_dir, args.max_images)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    for image_path in image_paths:
        image = Image.open(image_path)
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=args.prompt)
        masks = output["masks"]
        overlay = overlay_masks(image, masks)
        out_path = output_dir / f"{image_path.stem}_overlay.png"
        overlay.save(out_path)


if __name__ == "__main__":
    main()

