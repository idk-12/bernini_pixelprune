#!/bin/bash
# Baseline Evaluation on Document Understanding Benchmarks (Qwen3.5, no PixelPrune)
#
# NOTE: Qwen3.5 requires transformers>=5.2.0 (different from Qwen3-VL which uses 4.57.6)
#
# Usage:
#   bash scripts/eval_baseline_qwen35.sh [DATASETS] [GPU_IDS] [MODEL_PATH] [PRED_ROOT]
#
# Examples:
#   bash scripts/eval_baseline_qwen35.sh
#   bash scripts/eval_baseline_qwen35.sh "DocVQA_VAL ChartQA_TEST"
#   bash scripts/eval_baseline_qwen35.sh "" "0,1,2,3"
#   bash scripts/eval_baseline_qwen35.sh "DocVQA_VAL" "0,1" \
#       "/mnt/nfs/data/pretrained_models/Qwen3.5-9B/" \
#       "/data/lijie/PixelPrune/eval/outputs/vit_ratio/qwen35_9b"

set -eux

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_PATH="${3:-/mnt/nfs/data/pretrained_models/Qwen3.6-27B}"
DATASETS="${1:-DocVQA_VAL AI2D_TEST ChartQA_TEST InfoVQA_VAL OCRBench MMLongBench_DOC olmOCRBench}"
# Default: use all available GPUs
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
GPU_IDS="${2:-$(seq -s, 0 $((NUM_GPUS - 1)))}"

# Baseline: PixelPrune disabled
export PIXELPRUNE_ENABLED=false

# Timing: per-sample e2e time + TTFT (needs VERBOSE=true)
export VERBOSE=true

# Enable e2e and vit timing logs (actual paths are set per-dataset in eval/run.py)
export DEDUP_LOG_FILE="enabled"
export PIXELPRUNE_LOG_FILE="enabled"

# Pass model path to VLMEvalKit config via env var
export model_path="$MODEL_PATH"

# PYTHONPATH: pixelprune package + VLMEvalKit (base vlmeval package used by eval/run.py)
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# --- eval/run.py is a patched VLMEvalKit entry point ---
VLMEVALKIT_DIR="${ROOT_DIR}/eval"

# Data & output paths
export LMUData="${VLMEVALKIT_DIR}/LMUData"
mkdir -p "$LMUData"
export pred_root="${4:-${VLMEVALKIT_DIR}/outputs/full_baseline_qwen36}"

echo "========================================"
echo "Baseline: Document Understanding Eval (Qwen3.5)"
echo "Model:      $MODEL_PATH"
echo "Datasets:   $DATASETS"
echo "GPUs:       $GPU_IDS"
echo "VLMEvalKit: $VLMEVALKIT_DIR"
echo "Method:     No pruning (baseline)"
echo "Timing log: per-dataset subdirectory"
echo "Thinking:   disabled (enable_thinking=False)"
echo "========================================"

cd "${VLMEVALKIT_DIR}"

# Convert space-separated datasets to array for proper argument passing
read -ra DATASET_ARRAY <<< "$DATASETS"

# Detect GPU count from GPU_IDS
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
GPU_COUNT=${#GPU_ARRAY[@]}

if [ "$GPU_COUNT" -eq 1 ]; then
    CUDA_VISIBLE_DEVICES=$GPU_IDS \
    python run.py \
        --data "${DATASET_ARRAY[@]}" \
        --model Qwen3.5-HF \
        --work-dir "$pred_root" \
        # --judge exact_matching （mmmu，videomme）
else
    CUDA_VISIBLE_DEVICES=$GPU_IDS \
    torchrun --nproc_per_node="$GPU_COUNT" run.py \
        --data "${DATASET_ARRAY[@]}" \
        --model Qwen3.5-HF \
        --work-dir "$pred_root"
fi

echo "========================================"
echo "Evaluation complete: $(date)"
echo "========================================"
