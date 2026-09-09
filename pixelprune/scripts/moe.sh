# VLMEVALKIT_USE_MODELSCOPE=1 bash scripts/eval_baseline_qwen35_moe.sh "DocVQA_VAL"

# bash scripts/eval_baseline_qwen35_moe.sh "TextVQA_VAL"

# bash scripts/eval_baseline_qwen35_moe.sh "InfoVQA_VAL"


# bash scripts/eval_doc_qwen35_moe.sh "DocVQA_VAL"

# bash scripts/eval_doc_qwen35_moe.sh "TextVQA_VAL"

# bash scripts/eval_doc_qwen35_moe.sh "InfoVQA_VAL"
VLMEVALKIT_USE_MODELSCOPE=1 bash scripts/eval_baseline_qwen35_moe.sh "Video-MME_64frame"

VLMEVALKIT_USE_MODELSCOPE=1 bash scripts/eval_doc_qwen35_moe.sh "Video-MME_64frame"

VLMEVALKIT_USE_MODELSCOPE=1 PIXELPRUNE_THRESHOLD=0.05 PIXELPRUNE_ANCHORED=false bash scripts/eval_doc_qwen35_moe.sh "Video-MME_64frame"

VLMEVALKIT_USE_MODELSCOPE=1 PIXELPRUNE_METHOD=pred_2d_seq PIXELPRUNE_THRESHOLD=0.05 PIXELPRUNE_ANCHORED=false bash scripts/eval_doc_qwen35_moe.sh "Video-MME_64frame"



# PIXELPRUNE_PATCH_IMPL=profile bash scripts/eval_doc_qwen35_moe.sh "TextVQA_VAL"

# PIXELPRUNE_PATCH_IMPL=profile PIXELPRUNE_THRESHOLD=0.15 PIXELPRUNE_ANCHORED=false bash scripts/eval_doc_qwen35_moe.sh "TextVQA_VAL"

# PIXELPRUNE_PATCH_IMPL=profile PIXELPRUNE_THRESHOLD=0.25 PIXELPRUNE_ANCHORED=false bash scripts/eval_doc_qwen35_moe.sh "TextVQA_VAL"

# PIXELPRUNE_PATCH_IMPL=profile PIXELPRUNE_THRESHOLD=0.3 PIXELPRUNE_ANCHORED=false bash scripts/eval_doc_qwen35_moe.sh "TextVQA_VAL"

# bash scripts/eval_doc_qwen35_moe.sh "TextVQA_VAL"

# VLMEVALKIT_USE_MODELSCOPE=1 PIXELPRUNE_THRESHOLD=0.05 PIXELPRUNE_ANCHORED=false bash scripts/eval_doc_qwen35_moe.sh "Video-MME_64frame"