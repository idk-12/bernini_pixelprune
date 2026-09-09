#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

CONDA_BIN=${CONDA_BIN:-/root/anaconda3/bin/conda}
RAINFUSION_ENV=${RAINFUSION_ENV:-/home/xilab_program/envs/rainfusion}
BERNINI_CONFIG=${BERNINI_CONFIG:-/home/xilab_program/model/Bernini-Diffusers}
PIXELPRUNE_ROOT=${PIXELPRUNE_ROOT:-/home/xilab_program/PixelPrune}
OPENVE_ROOT=${OPENVE_ROOT:-/home/xilab_program/datasets/OpenVE}
OPENVE_CSV=${OPENVE_CSV:-$OPENVE_ROOT/benchmark_videos.csv}
RUN_TS=${RUN_TS:-$(date +%Y_%-m_%-d_%H_%M)}
OUTPUT_ROOT=${OUTPUT_ROOT:-benchmarks/openve/rainfusion/$RUN_TS}
if [[ "$OUTPUT_ROOT" != /* ]]; then
    OUTPUT_ROOT="$ROOT_DIR/$OUTPUT_ROOT"
fi
PYTHON_BIN=${PYTHON_BIN:-$RAINFUSION_ENV/bin/python}
MAX_ITEMS=${MAX_ITEMS:-0}
START_INDEX=${START_INDEX:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
CPU_THREADS_PER_RANK=${CPU_THREADS_PER_RANK:-16}
export OMP_NUM_THREADS="$CPU_THREADS_PER_RANK"
export MKL_NUM_THREADS="$CPU_THREADS_PER_RANK"
ULYSSES=${ULYSSES:-2}
RAINFUSION_START_STEP=${RAINFUSION_START_STEP:-0}
RAINFUSION_SPARSITY=${RAINFUSION_SPARSITY:-0.7}
VIT_MAX_PIXELS=${VIT_MAX_PIXELS:-401408}
VIT_NUM_FRAMES=${VIT_NUM_FRAMES:-64}
RUN_GROUPS="pixelprune"
export PIXELPRUNE_ROOT

if [[ "$NPROC_PER_NODE" -lt 1 || "$ULYSSES" -lt 1 || $((NPROC_PER_NODE % ULYSSES)) -ne 0 ]]; then
    echo "NPROC_PER_NODE must be positive and divisible by ULYSSES." >&2
    exit 2
fi
"$PYTHON_BIN" -c \
    "from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2 import rain_fusion_attention" \
    >/dev/null || {
    echo "RainFusion v3 is unavailable in environment '$RAINFUSION_ENV'." >&2
    exit 2
}

if [[ ! -f "$OPENVE_CSV" ]]; then
    echo "Missing OpenVE csv: $OPENVE_CSV" >&2
    exit 2
fi

"$PYTHON_BIN" -c \
    "from bernini.pixelprune_utils import _load_selector; _load_selector()" >/dev/null || {
    echo "PixelPrune is not installed in environment '$RAINFUSION_ENV'." >&2
    echo "Install it or set PIXELPRUNE_ROOT=/path/to/PixelPrune." >&2
    exit 2
}

mkdir -p "$OUTPUT_ROOT"

prepare_inputs() {
    local group=$1
    local group_dir="$OUTPUT_ROOT/$group"
    local inputs_path="$group_dir/inputs.json"
    local result_csv="$group_dir/new_result.csv"
    mkdir -p "$group_dir/results"

    OPENVE_ROOT="$OPENVE_ROOT" \
    OPENVE_CSV="$OPENVE_CSV" \
    GROUP_DIR="$group_dir" \
    INPUTS_PATH="$inputs_path" \
    RESULT_CSV="$result_csv" \
    MAX_ITEMS="$MAX_ITEMS" \
    START_INDEX="$START_INDEX" \
    "$PYTHON_BIN" - <<'PY'
import csv
import json
import os
from pathlib import Path

openve_root = Path(os.environ["OPENVE_ROOT"]).resolve()
openve_csv = Path(os.environ["OPENVE_CSV"]).resolve()
group_dir = Path(os.environ["GROUP_DIR"]).resolve()
inputs_path = Path(os.environ["INPUTS_PATH"]).resolve()
result_csv = Path(os.environ["RESULT_CSV"]).resolve()
max_items = int(os.environ["MAX_ITEMS"])
start_index = int(os.environ["START_INDEX"])

def resolve_video(path_value: str) -> Path:
    candidate = Path(path_value)
    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.append(openve_root / candidate)
        candidates.append(openve_root / "videos" / candidate.name)
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"missing source video: {path_value}")

with openve_csv.open(newline="", encoding="utf-8-sig") as file:
    rows = list(csv.DictReader(file))

if start_index:
    rows = rows[start_index:]
if max_items > 0:
    rows = rows[:max_items]

tasks = []
out_rows = []
for offset, row in enumerate(rows):
    global_index = start_index + offset
    source_path = resolve_video(row["original_video"])
    sample_id = f"{global_index:04d}_{Path(row['original_video']).stem}"
    output_path = (group_dir / "results" / f"{sample_id}.mp4").resolve()

    tasks.append(
        {
            "task_type": "v2v",
            "prompt": row["prompt"],
            "video": str(source_path),
            "output": str(output_path),
        }
    )

    out_row = dict(row)
    out_row["original_video"] = str(source_path)
    out_row["edited_result_path"] = str(output_path)
    out_rows.append(out_row)

inputs_path.write_text(
    json.dumps(tasks, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
with result_csv.open("w", newline="", encoding="utf-8") as file:
    fieldnames = list(out_rows[0].keys()) if out_rows else [
        "edited_type",
        "prompt",
        "original_video",
        "edited_result_path",
    ]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)

print(f"Prepared {len(tasks)} OpenVE tasks: {inputs_path}")
print(f"OpenVE result csv: {result_csv}")
PY
}

run_group() {
    local group=$1
    local group_dir="$OUTPUT_ROOT/$group"
    local inputs_path="$group_dir/inputs.json"
    local metrics_path="$group_dir/metrics.jsonl"
    local log_path="$group_dir/run.log"

    local pixelprune_flag=--no-pixelprune
    if [[ "$group" == "pixelprune" ]]; then
        pixelprune_flag=--pixelprune
    fi

    prepare_inputs "$group"

    "$CONDA_BIN" run --no-capture-output -p "$RAINFUSION_ENV" torchrun --standalone \
        --nproc-per-node "$NPROC_PER_NODE" infer_multi_gpu.py \
        --config "$BERNINI_CONFIG" \
        --ulysses "$ULYSSES" \
        --dit_attention_backend rainfusion \
        --rainfusion_start_step "$RAINFUSION_START_STEP" \
        --rainfusion_sparsity "$RAINFUSION_SPARSITY" \
        --num_frames 81 \
        --max_image_size 848 \
        --num_inference_steps 4 \
        --flow_shift 5.0 \
        --height 480 \
        --width 848 \
        --seed 42 \
        --fps 16 \
        --omega_txt 4 \
        --omega_tgt 0.5 \
        --omega_img 1.25 \
        --omega_vid 1.25 \
        --omega_scale 0.75 \
        --vit_denoising_step 5 \
        --vit_max_pixels "$VIT_MAX_PIXELS" \
        --vit_num_frames "$VIT_NUM_FRAMES" \
        --vit_txt_cfg 1.2 \
        --vit_img_cfg 1.0 \
        --guidance_mode vae_txt_vit_wapg \
        --pixelprune_profile \
        --pixelprune_threshold 0.7 \
        --pixelprune_method pred_2d_seq \
        --pixelprune_metrics_output "$metrics_path" \
        "$pixelprune_flag" \
        --no-pixelprune_warmup \
        --inputs "$inputs_path" \
        >"$log_path" 2>&1
}

for group in $RUN_GROUPS; do
    if [[ "$group" != "baseline" && "$group" != "pixelprune" ]]; then
        echo "Unknown group '$group'; use baseline and/or pixelprune." >&2
        exit 2
    fi
    run_group "$group"
done

echo "OpenVE artifacts: $OUTPUT_ROOT"
echo "PixelPrune result csv: $OUTPUT_ROOT/pixelprune/new_result.csv"
echo "Use root_path=/ for OpenVE evaluation because the csv stores absolute paths."
