#!/bin/bash
# 帧间去重消融实验：interframe / pred_2d_union / pred_2d_seq
# 各跑 50 samples，THRESHOLD=0，GPU 0
set -eux

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_PATH="/mnt/nfs/data/pretrained_models/Qwen3.6-27B"
export model_path="$MODEL_PATH"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export LMUData="${ROOT_DIR}/eval/LMUData"
export pred_root="${ROOT_DIR}/eval/outputs/pixelprune_doc_qwen36"

export VLMEVALKIT_USE_MODELSCOPE=1
export PIXELPRUNE_ENABLED=true
export PIXELPRUNE_METRIC=max
export PIXELPRUNE_THRESHOLD=0.0
export VERBOSE=true
export PIXELPRUNE_VERBOSE=true
export DEDUP_LOG_FILE="enabled"
export PIXELPRUNE_LOG_FILE="enabled"

mkdir -p "$LMUData"
cd "${ROOT_DIR}/eval"

for METHOD in interframe pred_2d_union pred_2d_seq; do
    echo "========================================"
    echo "Running method: $METHOD  threshold=0  samples=50"
    echo "Start: $(date)"
    echo "========================================"

    PIXELPRUNE_METHOD=$METHOD \
    CUDA_VISIBLE_DEVICES=0 \
    python run.py \
        --data Video-MME_64frame \
        --model Qwen3.5-HF \
        --work-dir "$pred_root" \
        --judge exact_matching \
        --max-samples 50

    echo "Done: $METHOD at $(date)"
done

echo "========================================"
echo "All 3 experiments complete: $(date)"
echo "========================================"
