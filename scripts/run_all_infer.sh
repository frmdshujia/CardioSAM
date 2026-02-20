#!/bin/bash
# Run ACDC video inference for all three anatomical structures (LV, RV, myocardium)
# in both adapter and baseline modes.
# Usage: bash scripts/run_all_infer.sh
# Optional: SAM3_USE_CPU_NMS=1 bash scripts/run_all_infer.sh  (if Triton fails on remote)

cd "$(dirname "$0")/.."

MANIFEST="outputs/annotations/acdc_video_split_manifest.json"
CKPT="outputs/acdc_adapter_280_bz8_ts350/checkpoints/checkpoint_50.pt"
BASE_CKPT="sam3/sam3.pt"

echo "=== Adapter mode (3 structures) ==="
for item in "left ventricle:outputs/acdc_video_infer_adapter_lv" "right ventricle:outputs/acdc_video_infer_adapter_rv" "myocardium:outputs/acdc_video_infer_adapter_myo"; do
  prompt="${item%%:*}"
  out="${item##*:}"
  echo "[Adapter] $prompt -> $out"
  python scripts/run_acdc_video_infer.py \
    --manifest "$MANIFEST" \
    --checkpoint "$CKPT" \
    --checkpoint_type adapter \
    --base_checkpoint "$BASE_CKPT" \
    --output_root "$out" \
    --split test \
    --prompt "$prompt" \
    --save_overlay
done

echo "=== Baseline mode (3 structures) ==="
for item in "left ventricle:outputs/acdc_video_infer_baseline_lv" "right ventricle:outputs/acdc_video_infer_baseline_rv" "myocardium:outputs/acdc_video_infer_baseline_myo"; do
  prompt="${item%%:*}"
  out="${item##*:}"
  echo "[Baseline] $prompt -> $out"
  python scripts/run_acdc_video_infer.py \
    --manifest "$MANIFEST" \
    --checkpoint "$BASE_CKPT" \
    --checkpoint_type baseline \
    --output_root "$out" \
    --split test \
    --prompt "$prompt" \
    --save_overlay
done

echo "=== All inference tasks completed ==="
