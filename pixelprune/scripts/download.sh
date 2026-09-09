export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export ASCEND_RT_VISIBLE_DEVICES=0
export ASCEND_VISIBLE_DEVICES=0

export MODELSCOPE_CACHE=/home/xilab_program/PixelPrune/eval/LMUData/modelscope
export MODELSCOPE_CACHE_HOME=/home/xilab_program/PixelPrune/eval/LMUData/modelscope
export XDG_CACHE_HOME=/home/xilab_program/PixelPrune/eval/LMUData/.cache
export MODELSCOPE_CACHE_DIR=/home/xilab_program/PixelPrune/eval/LMUData/modelscope
export QWEN_NPU_MAX_MEMORY=58GiB

VLMEVALKIT_USE_MODELSCOPE=1 bash scripts/eval_baseline_qwen35_moe.sh "LongVideoBench_64frame" "0"
VLMEVALKIT_USE_MODELSCOPE=1 bash scripts/eval_baseline_qwen35_moe.sh "MLVU_64frame" "0"