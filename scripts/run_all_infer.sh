#!/bin/bash
# 批量推理脚本：adapter(280px) + no_adapter(280px) + baseline
# 用法：
#   bash scripts/run_all_infer.sh                # 跑 adapter_280px + no_adapter_280px + baseline
#   bash scripts/run_all_infer.sh adapter_280px  # 只跑 280px adapter
#   bash scripts/run_all_infer.sh no_adapter_280px # 只跑 280px 直接微调（无adapter）
#   bash scripts/run_all_infer.sh baseline       # 只跑 baseline

cd "$(dirname "$0")/.."

MANIFEST="outputs/annotations/acdc_video_split_manifest.json"
CKPT="outputs/acdc_adapter_280_bz8_ts350/checkpoints/checkpoint_50.pt"
NO_ADAPTER_CKPT="outputs/acdc_no_adapter_280_bz8_ts350/checkpoints/checkpoint_50.pt"
BASE_CKPT="sam3/sam3.pt"
LOG_DIR="outputs/infer_logs"

mkdir -p "$LOG_DIR"

# RUN_WHAT: 决定本次运行哪些任务（默认全跑）
RUN_WHAT="${1:-all}"

run_one() {
    local mode="$1"       # adapter_280px | no_adapter_280px | baseline
    local prompt="$2"
    local out="$3"
    local log="$4"

    echo ""
    echo "========================================"
    echo "  Mode   : $mode"
    echo "  Prompt : $prompt"
    echo "  Output : $out"
    echo "  Log    : $log"
    echo "========================================"

    if [ "$mode" = "adapter_280px" ]; then
        python scripts/run_acdc_video_infer.py \
            --manifest "$MANIFEST" \
            --checkpoint "$CKPT" \
            --checkpoint_type adapter \
            --base_checkpoint "$BASE_CKPT" \
            --output_root "$out" \
            --split test \
            --prompt "$prompt" \
            --image_size 280 \
            --save_overlay \
            2>&1 | tee "$log"
    elif [ "$mode" = "no_adapter_280px" ]; then
        python scripts/run_acdc_video_infer.py \
            --manifest "$MANIFEST" \
            --checkpoint "$NO_ADAPTER_CKPT" \
            --checkpoint_type no_adapter \
            --base_checkpoint "$BASE_CKPT" \
            --output_root "$out" \
            --split test \
            --prompt "$prompt" \
            --image_size 280 \
            --save_overlay \
            2>&1 | tee "$log"
    else
        python scripts/run_acdc_video_infer.py \
            --manifest "$MANIFEST" \
            --checkpoint "$BASE_CKPT" \
            --checkpoint_type baseline \
            --output_root "$out" \
            --split test \
            --prompt "$prompt" \
            --save_overlay \
            2>&1 | tee "$log"
    fi

    echo "[DONE] $mode / $prompt"
}

if [ "$RUN_WHAT" = "adapter_280px" ] || [ "$RUN_WHAT" = "all" ]; then
    echo "=== Adapter 280px mode ==="
    run_one adapter_280px "left ventricle"  outputs/acdc_video_infer_adapter_lv_280px  "$LOG_DIR/adapter_lv_280px.log"
    run_one adapter_280px "right ventricle" outputs/acdc_video_infer_adapter_rv_280px  "$LOG_DIR/adapter_rv_280px.log"
    run_one adapter_280px "myocardium"      outputs/acdc_video_infer_adapter_myo_280px "$LOG_DIR/adapter_myo_280px.log"
fi

if [ "$RUN_WHAT" = "no_adapter_280px" ] || [ "$RUN_WHAT" = "all" ]; then
    echo ""
    echo "=== No-Adapter 280px mode (direct fine-tune) ==="
    run_one no_adapter_280px "left ventricle"  outputs/acdc_video_infer_no_adapter_lv_280px  "$LOG_DIR/no_adapter_lv_280px.log"
    run_one no_adapter_280px "right ventricle" outputs/acdc_video_infer_no_adapter_rv_280px  "$LOG_DIR/no_adapter_rv_280px.log"
    run_one no_adapter_280px "myocardium"      outputs/acdc_video_infer_no_adapter_myo_280px "$LOG_DIR/no_adapter_myo_280px.log"
fi

if [ "$RUN_WHAT" = "baseline" ] || [ "$RUN_WHAT" = "all" ]; then
    echo ""
    echo "=== Baseline mode ==="
    run_one baseline "left ventricle"  outputs/acdc_video_infer_baseline_lv  "$LOG_DIR/baseline_lv.log"
    run_one baseline "right ventricle" outputs/acdc_video_infer_baseline_rv  "$LOG_DIR/baseline_rv.log"
    run_one baseline "myocardium"      outputs/acdc_video_infer_baseline_myo "$LOG_DIR/baseline_myo.log"
fi

echo ""
echo "=== Tasks completed for: $RUN_WHAT ==="
