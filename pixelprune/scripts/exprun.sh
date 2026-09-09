#!/bin/bash
# Run baseline ViT-time-ratio experiments for Qwen3.5/Qwen3.6 models.
#
# Usage:
#   bash scripts/exprun.sh [GPU_IDS]
#
# Example:
#   bash scripts/exprun.sh "0,1,2,3"

set -eux

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
GPU_IDS="${1:-$(seq -s, 0 $((NUM_GPUS - 1)))}"

IMAGE_DATASETS="DocVQA_VAL InfoVQA_VAL TextVQA_VAL"
VIDEO_DATASET="Video-MME_64frame"
OUTPUT_ROOT="${ROOT_DIR}/eval/outputs/vit_ratio"

run_dense_model() {
    local model_name="$1"
    local model_path="$2"
    local output_dir="${OUTPUT_ROOT}/${model_name}"

    bash "${SCRIPT_DIR}/eval_baseline_qwen35.sh" \
        "$IMAGE_DATASETS" "$GPU_IDS" "$model_path" "$output_dir"

    VLMEVALKIT_USE_MODELSCOPE=1 \
    bash "${SCRIPT_DIR}/eval_baseline_qwen35.sh" \
        "$VIDEO_DATASET" "$GPU_IDS" "$model_path" "$output_dir"
}

run_moe_model() {
    local model_name="$1"
    local model_path="$2"
    local output_dir="${OUTPUT_ROOT}/${model_name}"

    bash "${SCRIPT_DIR}/eval_baseline_qwen35_moe.sh" \
        "$IMAGE_DATASETS" "$GPU_IDS" "$model_path" "$output_dir"

    VLMEVALKIT_USE_MODELSCOPE=1 \
    bash "${SCRIPT_DIR}/eval_baseline_qwen35_moe.sh" \
        "$VIDEO_DATASET" "$GPU_IDS" "$model_path" "$output_dir"
}

run_dense_model \
    "qwen35_9b" \
    "/mnt/nfs/data/pretrained_models/Qwen3.5-9B/"

run_dense_model \
    "qwen36_27b" \
    "/mnt/nfs/data/pretrained_models/Qwen3.6-27B/"

run_moe_model \
    "qwen36_35b_a3b" \
    "/mnt/nfs/data/pretrained_models/Qwen3.6-35B-A3B/"

echo "========================================"
echo "All ViT time ratio experiments complete: $(date)"
echo "Outputs: $OUTPUT_ROOT"
echo "========================================"
