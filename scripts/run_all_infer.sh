#!/usr/bin/env bash
# run_all_infer.sh
# 依次对 adapter (checkpoint_50.pt) 和 baseline (sam3.pt) 两个权重
# 分别推理 LV / MYO / RV 三种结构（共 6 个任务）。
# 用法：bash scripts/run_all_infer.sh [test|val|train|all]
# 默认 split=test
set -eu
SPLIT="${1:-test}"
PYTHON=/data/home/shujia/miniconda3/envs/sam3/bin/python
SCRIPT=scripts/run_acdc_video_infer.py
MANIFEST=outputs/annotations/acdc_video_split_manifest.json
CKPT_ADAPTER=outputs/acdc_adapter_280_bz8_ts350/checkpoints/checkpoint_50.pt
CKPT_BASELINE=sam3/sam3.pt
LOG_DIR=outputs/infer_logs
mkdir -p "$LOG_DIR"

cd "$(dirname "$0")/.."   # 确保在项目根目录执行

echo "============================================================"
echo " ACDC video inference — split: $SPLIT"
echo " Start: $(date)"
echo "============================================================"

run_one() {
    local label="$1"
    local ckpt="$2"
    local ckpt_type="$3"
    local prompt="$4"
    local out_dir="$5"
    local log_file="$LOG_DIR/${label}.log"

    echo ""
    echo "------------------------------------------------------------"
    echo "[$label]  prompt='$prompt'  ckpt_type=$ckpt_type"
    echo "  output : $out_dir"
    echo "  log    : $log_file"
    echo "  start  : $(date)"
    echo "------------------------------------------------------------"

    "$PYTHON" "$SCRIPT" \
        --manifest    "$MANIFEST" \
        --checkpoint  "$ckpt" \
        --checkpoint_type "$ckpt_type" \
        --output_root "$out_dir" \
        --split       $SPLIT \
        --prompt      "$prompt" \
        --save_overlay \
        --resume \
        2>&1 | tee "$log_file"

    echo "  finish : $(date)"
}

# ── Adapter checkpoint ────────────────────────────────────────────
run_one "adapter_lv"  "$CKPT_ADAPTER" adapter "left ventricle" \
        outputs/acdc_video_infer_adapter_lv

run_one "adapter_myo" "$CKPT_ADAPTER" adapter "myocardium" \
        outputs/acdc_video_infer_adapter_myo

run_one "adapter_rv"  "$CKPT_ADAPTER" adapter "right ventricle" \
        outputs/acdc_video_infer_adapter_rv

# ── Baseline (sam3.pt) ────────────────────────────────────────────
run_one "baseline_lv"  "$CKPT_BASELINE" baseline "left ventricle" \
        outputs/acdc_video_infer_baseline_lv

run_one "baseline_myo" "$CKPT_BASELINE" baseline "myocardium" \
        outputs/acdc_video_infer_baseline_myo

run_one "baseline_rv"  "$CKPT_BASELINE" baseline "right ventricle" \
        outputs/acdc_video_infer_baseline_rv

echo ""
echo "============================================================"
echo " ALL DONE — $(date)"
echo "============================================================"
