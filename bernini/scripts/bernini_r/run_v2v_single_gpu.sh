#!/usr/bin/env bash
# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -euo pipefail

CASE_PATH=${CASE_PATH:-assets/testcases/v2v/v2v_case1.json}
BERNINI_R_CONFIG=${BERNINI_R_CONFIG:-/data/lijie/models/Bernini-R-1.3B-Diffusers}
OUTPUT_DIR=${OUTPUT_DIR:-/data/lijie/Bernini/outputs/v2v}
NUM_FRAMES=${NUM_FRAMES:-81}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-40}
HEIGHT=${HEIGHT:-480}
WIDTH=${WIDTH:-848}

TASK_TYPE=$(jq -r '.task_type' "$CASE_PATH")
PROMPT=$(jq -r '.prompt' "$CASE_PATH")
VIDEO=$(jq -r '.video' "$CASE_PATH")
CASE_NAME=$(basename "$CASE_PATH" .json)
OUTPUT_PATH="$OUTPUT_DIR/${CASE_NAME}_out.mp4"

mkdir -p "$OUTPUT_DIR"

python infer_single_gpu.py \
    --config "$BERNINI_R_CONFIG" \
    --task_type "$TASK_TYPE" \
    --prompt "$PROMPT" \
    --video "$VIDEO" \
    --output "$OUTPUT_PATH" \
    --num_frames "$NUM_FRAMES" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --max_image_size 848 \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --flow_shift 5.0 \
    --seed 42 \
    --fps 16 \
    --omega_txt 4 \
    --omega_tgt 0.5 \
    --omega_img 1.25 \
    --omega_vid 1.25 \
    --omega_scale 0.75 \
    --vit_denoising_step 5 \
    --vit_txt_cfg 1.2 \
    --vit_img_cfg 1.0 \
    --guidance_mode v2v_apg
