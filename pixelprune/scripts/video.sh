export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export QWEN_NPU_MAX_MEMORY=58GiB

# VLMEVALKIT_USE_MODELSCOPE=1 bash scripts/eval_baseline_qwen35_moe.sh "Video-MME_64frame" "0,1,2,3,4,5,6,7"
# VLMEVALKIT_USE_MODELSCOPE=1 bash scripts/eval_doc_qwen35_moe.sh "Video-MME_64frame" "0,1,2,3,4,5,6,7"
# VLMEVALKIT_USE_MODELSCOPE=1 PIXELPRUNE_THRESHOLD=0.05 PIXELPRUNE_ANCHORED=false bash scripts/eval_doc_qwen35_moe.sh "Video-MME_64frame" "0,1,2,3,4,5,6,7"
# VLMEVALKIT_USE_MODELSCOPE=1 PIXELPRUNE_METHOD=pred_2d_seq PIXELPRUNE_THRESHOLD=0.05 PIXELPRUNE_ANCHORED=false bash scripts/eval_doc_qwen35_moe.sh "Video-MME_64frame" "0,1,2,3,4,5,6,7"


# VLMEVALKIT_USE_MODELSCOPE=1 bash scripts/eval_baseline_qwen35_moe.sh "LongVideoBench_64frame" "0,1,2,3,4,5,6,7"
# VLMEVALKIT_USE_MODELSCOPE=1 bash scripts/eval_doc_qwen35_moe.sh "LongVideoBench_64frame" "0,1,2,3,4,5,6,7"
# VLMEVALKIT_USE_MODELSCOPE=1 PIXELPRUNE_THRESHOLD=0.05 PIXELPRUNE_ANCHORED=false bash scripts/eval_doc_qwen35_moe.sh "LongVideoBench_64frame" "0,1,2,3,4,5,6,7"
# PIXELPRUNE_METHOD=pred_2d_seq VLMEVALKIT_USE_MODELSCOPE=1 PIXELPRUNE_THRESHOLD=0.05 PIXELPRUNE_ANCHORED=false bash scripts/eval_doc_qwen35_moe.sh "LongVideoBench_64frame" "0,1,2,3,4,5,6,7"


VLMEVALKIT_USE_MODELSCOPE=1 bash scripts/eval_baseline_qwen35_moe.sh "MLVU_MCQ_64frame" "0,1,2,3,4,5,6,7"
# VLMEVALKIT_USE_MODELSCOPE=1 bash scripts/eval_doc_qwen35_moe.sh "MLVU_MCQ_64frame" "0,1,2,3,4,5,6,7"
# VLMEVALKIT_USE_MODELSCOPE=1 PIXELPRUNE_THRESHOLD=0.05 PIXELPRUNE_ANCHORED=false bash scripts/eval_doc_qwen35_moe.sh "MLVU_MCQ_64frame" "0,1,2,3,4,5,6,7"
PIXELPRUNE_METHOD=pred_2d_seq VLMEVALKIT_USE_MODELSCOPE=1 PIXELPRUNE_THRESHOLD=0.05 PIXELPRUNE_ANCHORED=false bash scripts/eval_doc_qwen35_moe.sh "MLVU_MCQ_64frame" "0,1,2,3,4,5,6,7"

