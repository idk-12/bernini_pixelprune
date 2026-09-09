#!/bin/bash
# Baseline Evaluation on Document Understanding Benchmarks (Qwen3.5-MoE, no PixelPrune)
#
# NOTE: Qwen3.5 requires transformers>=5.2.0 (different from Qwen3-VL which uses 4.57.6)
#
# Usage:
#   bash scripts/eval_baseline_qwen35_moe.sh [DATASETS] [GPU_IDS] [MODEL_PATH] [PRED_ROOT]
#
# Examples:
#   bash scripts/eval_baseline_qwen35_moe.sh
#   bash scripts/eval_baseline_qwen35_moe.sh "DocVQA_VAL ChartQA_TEST"
#   bash scripts/eval_baseline_qwen35_moe.sh "" "0,1,2,3"
#   bash scripts/eval_baseline_qwen35_moe.sh "DocVQA_VAL" "0,1,2,3" \
#       "/mnt/nfs/data/pretrained_models/Qwen3.6-35B-A3B/" \
#       "/data/lijie/PixelPrune/eval/outputs/vit_ratio/qwen36_35b_a3b"

set -eux

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_PATH="${3:-/home/xilab_program/model/Qwen3.6-35B-A3B/}"
DATASETS="${1:-DocVQA_VAL AI2D_TEST ChartQA_TEST InfoVQA_VAL OCRBench MMLongBench_DOC olmOCRBench}"
# Default: use all available GPUs
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
GPU_IDS="${2:-$(seq -s, 0 $((NUM_GPUS - 1)))}"

# Baseline: PixelPrune disabled
export PIXELPRUNE_ENABLED=false
export MODEL_TYPE=qwen3_5_moe

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
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${LMUData}/modelscope}"
export MODELSCOPE_CACHE_HOME="${MODELSCOPE_CACHE_HOME:-$MODELSCOPE_CACHE}"
export MODELSCOPE_CACHE_DIR="${MODELSCOPE_CACHE_DIR:-$MODELSCOPE_CACHE}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${LMUData}/.cache}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
mkdir -p "$MODELSCOPE_CACHE" "$HF_HUB_CACHE"
export pred_root="${4:-${VLMEVALKIT_DIR}/outputs/full_baseline_qwen36_moe}"

echo "========================================"
echo "Baseline: Document Understanding Eval (Qwen3.6-MoE)"
echo "Model:      $MODEL_PATH"
echo "Model type: $MODEL_TYPE"
echo "Datasets:   $DATASETS"
echo "GPUs:       $GPU_IDS"
echo "MP size:    ${MP_SIZE:-1}"
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
if [ -z "${MP_SIZE:-}" ]; then
    if [ "$GPU_COUNT" -gt 1 ]; then
        MP_SIZE=2
    else
        MP_SIZE=1
    fi
fi
if [ "$MP_SIZE" -lt 1 ]; then
    echo "MP_SIZE must be >= 1, got $MP_SIZE" >&2
    exit 1
fi
if [ $((GPU_COUNT % MP_SIZE)) -ne 0 ]; then
    echo "GPU count ($GPU_COUNT) must be divisible by MP_SIZE ($MP_SIZE)" >&2
    exit 1
fi
DP_SIZE=$((GPU_COUNT / MP_SIZE))

if [ "$MP_SIZE" -gt 1 ]; then
    export QWEN_NPU_DEVICE_MAP="${QWEN_NPU_DEVICE_MAP:-auto}"
    export QWEN_NPU_MAX_MEMORY="${QWEN_NPU_MAX_MEMORY:-58GiB}"
    export DIST_BACKEND="${DIST_BACKEND:-hccl}"
    echo "Model/data parallel: MP_SIZE=$MP_SIZE, DP_SIZE=$DP_SIZE, QWEN_NPU_MAX_MEMORY=$QWEN_NPU_MAX_MEMORY"
fi

if [ "$GPU_COUNT" -eq 1 ]; then
    CUDA_VISIBLE_DEVICES=$GPU_IDS \
    ASCEND_VISIBLE_DEVICES=$GPU_IDS \
    ASCEND_RT_VISIBLE_DEVICES=$GPU_IDS \
    python run.py \
        --data "${DATASET_ARRAY[@]}" \
        --model Qwen3.5-HF \
        --work-dir "$pred_root" \
        --judge exact_matching 
        #（mmmu，videomme）
else
    CUDA_VISIBLE_DEVICES=$GPU_IDS \
    ASCEND_VISIBLE_DEVICES=$GPU_IDS \
    ASCEND_RT_VISIBLE_DEVICES=$GPU_IDS \
    python -m torch.distributed.run --nproc_per_node="$DP_SIZE" run.py \
        --data "${DATASET_ARRAY[@]}" \
        --model Qwen3.5-HF \
        --work-dir "$pred_root" \
        --judge exact_matching 
fi

echo "========================================"
echo "Evaluation complete: $(date)"
echo "========================================"
