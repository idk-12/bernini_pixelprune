PIXELPRUNE_THRESHOLD=0.1 PIXELPRUNE_ANCHORED=false bash scripts/eval_doc_qwen35.sh "TextVQA_VAL"
bash scripts/eval_doc_qwen35.sh "MMMU_DEV_VAL"
bash scripts/eval_baseline_qwen35.sh "MMMU_DEV_VAL"
PIXELPRUNE_THRESHOLD=0.1  bash scripts/eval_doc_qwen35.sh "InfoVQA_VAL"
export VLMEVALKIT_USE_MODELSCOPE=1 bash scripts/eval_baseline_qwen35.sh "Video-MME_64frame"
export VLMEVALKIT_USE_MODELSCOPE=1 PIXELPRUNE_THRESHOLD=0.05 PIXELPRUNE_ANCHORED=false bash scripts/eval_doc_qwen35.sh "Video-MME_64frame"
PIXELPRUNE_METHOD=pred_2d_seq VLMEVALKIT_USE_MODELSCOPE=1 bash scripts/eval_doc_qwen35.sh "Video-MME_64frame"



#post vit
export VLMEVALKIT_USE_MODELSCOPE=1 PIXELPRUNE_PATCH_IMPL=postvit PIXELPRUNE_SELECT_STAGE=post_vit_merged PIXELPRUNE_THRESHOLD=0.5 PIXELPRUNE_ANCHORED=false \
  bash scripts/eval_doc_qwen35.sh "InfoVQA_VAL"

PIXELPRUNE_PATCH_IMPL=postvit PIXELPRUNE_SELECT_STAGE=post_vit_patch \
  bash scripts/eval_doc_qwen35.sh "InfoVQA_VAL"
