#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"

CONDA_BIN=${CONDA_BIN:-/root/anaconda3/bin/conda}
RAINFUSION_ENV=${RAINFUSION_ENV:-/home/xilab_program/envs/rainfusion}
BERNINI_CONFIG=${BERNINI_CONFIG:-/home/xilab_program/model/Bernini-Diffusers}
PIXELPRUNE_ROOT=${PIXELPRUNE_ROOT:-/home/xilab_program/PixelPrune}
EDITVERSE_DATA_ROOT=${EDITVERSE_DATA_ROOT:-/home/xilab_program/datasets/EditVerse/EditVerseBench}
EDITVERSE_TEST_JSON=${EDITVERSE_TEST_JSON:-$EDITVERSE_DATA_ROOT/EditVerseBench.json}
RUN_TS=${RUN_TS:-$(date +%Y_%-m_%-d_%H_%M)}
OUTPUT_ROOT=${OUTPUT_ROOT:-benchmarks/editverse/rainfusion/$RUN_TS}
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
COPY_SOURCE=${COPY_SOURCE:-false}
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

if [[ ! -f "$EDITVERSE_TEST_JSON" ]]; then
    echo "Missing EditVerse json: $EDITVERSE_TEST_JSON" >&2
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
    local eval_json="$group_dir/eval_test.json"
    mkdir -p "$group_dir/results"

    EDITVERSE_DATA_ROOT="$EDITVERSE_DATA_ROOT" \
    EDITVERSE_TEST_JSON="$EDITVERSE_TEST_JSON" \
    GROUP_DIR="$group_dir" \
    INPUTS_PATH="$inputs_path" \
    EVAL_JSON="$eval_json" \
    MAX_ITEMS="$MAX_ITEMS" \
    START_INDEX="$START_INDEX" \
    COPY_SOURCE="$COPY_SOURCE" \
    "$PYTHON_BIN" - <<'PY'
import json
import os
import shutil
from pathlib import Path

data_root = Path(os.environ["EDITVERSE_DATA_ROOT"]).resolve()
test_json = Path(os.environ["EDITVERSE_TEST_JSON"]).resolve()
group_dir = Path(os.environ["GROUP_DIR"]).resolve()
inputs_path = Path(os.environ["INPUTS_PATH"]).resolve()
eval_json = Path(os.environ["EVAL_JSON"]).resolve()
max_items = int(os.environ["MAX_ITEMS"])
start_index = int(os.environ["START_INDEX"])
copy_source = os.environ["COPY_SOURCE"].lower() == "true"

with test_json.open(encoding="utf-8") as file:
    entries = json.load(file)

items = sorted(entries.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0])
if start_index:
    items = items[start_index:]
if max_items > 0:
    items = items[:max_items]

tasks = []
manifest = {}
selected_entries = {}
for item_id, item in items:
    selected_entries[item_id] = item
    source_rel = item["<video1>"]
    source_path = Path(source_rel)
    if not source_path.is_absolute():
        source_path = data_root / source_path
    source_path = source_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"missing source video for item {item_id}: {source_path}")

    sample_dir = group_dir / "results" / item_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    source_link = sample_dir / "video1.mp4"
    if source_link.exists() or source_link.is_symlink():
        source_link.unlink()
    if copy_source:
        shutil.copy2(source_path, source_link)
    else:
        source_link.symlink_to(source_path)

    output_path = sample_dir / "generate.mp4"
    prompt = item["<text>"].replace("<video1>", "").strip()
    tasks.append(
        {
            "task_type": "v2v",
            "prompt": prompt,
            "video": str(source_path),
            "output": str(output_path),
        }
    )
    manifest[item_id] = {
        "source": str(source_path),
        "output": str(output_path),
        "prompt": prompt,
    }

inputs_path.write_text(
    json.dumps(tasks, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
(group_dir / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
eval_json.write_text(
    json.dumps(selected_entries, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(f"Prepared {len(tasks)} EditVerse tasks: {inputs_path}")
print(f"EditVerse eval json: {eval_json}")
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

echo "EditVerse artifacts: $OUTPUT_ROOT"
echo "PixelPrune eval dir: $OUTPUT_ROOT/pixelprune/results"
echo "PixelPrune eval json: $OUTPUT_ROOT/pixelprune/eval_test.json"
