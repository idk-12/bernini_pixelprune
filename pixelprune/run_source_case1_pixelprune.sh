#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/data/lijie/PixelPrune"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/tmp}"
VIDEO_PATH="${VIDEO_PATH:-/data/lijie/Bernini/assets/testcases/v2v/source_case1.mp4}"
MODEL_PATH="${MODEL_PATH:-/mnt/nfs/data/pretrained_models/Qwen3.6-35B-A3B/}"
GPU_IDS="${GPU_IDS:-0}"
THRESHOLDS="${THRESHOLDS:-0 0.5}"
NFRAME="${NFRAME:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
PROMPT="${PROMPT:-Describe this video briefly.}"

mkdir -p "${OUT_DIR}"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export VIDEO_PATH MODEL_PATH NFRAME MAX_NEW_TOKENS PROMPT

echo "Root:       ${ROOT_DIR}"
echo "Video:      ${VIDEO_PATH}"
echo "Model:      ${MODEL_PATH}"
echo "Output:     ${OUT_DIR}"
echo "GPU IDs:    ${GPU_IDS}"
echo "Thresholds: ${THRESHOLDS}"
echo

for threshold in ${THRESHOLDS}; do
  safe_threshold="${threshold//./p}"
  prefix="${OUT_DIR}/source_case1_th${safe_threshold}"

  echo "===== PIXELPRUNE_THRESHOLD=${threshold} ====="
  echo "stdout log: ${prefix}.stdout.log"

  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
  PIXELPRUNE_ENABLED=true \
  PIXELPRUNE_THRESHOLD="${threshold}" \
  PIXELPRUNE_METHOD=pred_2d_seq \
  PIXELPRUNE_METRIC=max \
  PIXELPRUNE_ANCHORED=false \
  PIXELPRUNE_VERBOSE=true \
  PIXELPRUNE_LOG_FILE="${prefix}.jsonl" \
  DEDUP_LOG_FILE="${prefix}.dedup.jsonl" \
  VERBOSE=true \
  MODEL_TYPE=qwen3_5_moe \
  python - <<'PY' 2>&1 | tee "${prefix}.stdout.log"
import json
import os
import time

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

from pixelprune import apply_pixelprune


def fix_qwen35_fa2_position_ids():
    import importlib

    decoder_specs = [
        ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5DecoderLayer"),
        ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", "Qwen3_5MoeDecoderLayer"),
    ]

    def make_patched_forward(orig_forward):
        def patched_forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                            position_ids=None, past_key_values=None, cache_position=None, **kwargs):
            if self.layer_type == "full_attention":
                position_ids = None
            return orig_forward(
                self,
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        return patched_forward

    patched = []
    for module_name, class_name in decoder_specs:
        try:
            module = importlib.import_module(module_name)
            decoder_cls = getattr(module, class_name)
        except (ImportError, AttributeError):
            continue
        if getattr(decoder_cls, "__pixelprune_fa2_patched__", False):
            continue
        decoder_cls.forward = make_patched_forward(decoder_cls.forward)
        decoder_cls.__pixelprune_fa2_patched__ = True
        patched.append(class_name)
    if patched:
        print(f"[Fix] Patched Qwen3.5 FA2 position_ids leak for {patched}")


def ensure_video_url(path):
    if path.startswith(("http://", "https://", "file://", "data:video")):
        return path
    if os.path.exists(path):
        return "file://" + path
    raise ValueError(f"Invalid video path: {path}")


model_path = os.environ["MODEL_PATH"]
video_path = ensure_video_url(os.environ["VIDEO_PATH"])
nframe = int(os.environ["NFRAME"])
max_new_tokens = int(os.environ["MAX_NEW_TOKENS"])
prompt = os.environ["PROMPT"]
model_type = os.environ.get("MODEL_TYPE", "qwen3_5_moe")

apply_pixelprune(model=model_type)
fix_qwen35_fa2_position_ids()

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
model.eval()

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, "nframes": nframe},
            {"type": "text", "text": prompt},
        ],
    }
]

try:
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
except TypeError:
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

images, videos, video_kwargs = process_vision_info(
    messages,
    image_patch_size=16,
    return_video_kwargs=True,
    return_video_metadata=True,
)

video_metadata = None
if videos is not None:
    videos, video_metadata = zip(*videos)
    videos, video_metadata = list(videos), list(video_metadata)

inputs = processor(
    text=text,
    images=images,
    videos=videos,
    video_metadata=video_metadata,
    do_resize=False,
    return_tensors="pt",
    **(video_kwargs or {}),
)
inputs.pop("mm_token_type_ids", None)

device = model.device
model_dtype = getattr(model, "dtype", None)
for key, value in list(inputs.items()):
    if torch.is_tensor(value):
        value = value.to(device)
        if model_dtype is not None and value.is_floating_point():
            value = value.to(model_dtype)
        inputs[key] = value

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
elapsed = time.perf_counter() - start

trimmed = generated[:, inputs["input_ids"].shape[-1]:]
response = processor.batch_decode(
    trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]

print("----- MODEL RESPONSE -----")
print(response)
print("----- PROFILE -----")
print(f"Input tokens: {inputs['input_ids'].shape[-1]}")
print(f"Generated tokens: {trimmed.shape[-1]}")
print(f"Total time: {elapsed:.3f}s")
if torch.cuda.is_available():
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

dedup_log = os.environ.get("DEDUP_LOG_FILE")
if dedup_log:
    base, ext = os.path.splitext(dedup_log)
    e2e_path = f"{base}.e2e.rank0{ext or '.jsonl'}"
    with open(e2e_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "video": os.environ["VIDEO_PATH"],
            "model": model_path,
            "threshold": float(os.environ["PIXELPRUNE_THRESHOLD"]),
            "nframe": nframe,
            "input_tokens": int(inputs["input_ids"].shape[-1]),
            "generated_tokens": int(trimmed.shape[-1]),
            "total_time_s": elapsed,
            "response": response,
        }, ensure_ascii=False) + "\n")
PY

  echo
done

echo "Done. Result files:"
ls -lh "${OUT_DIR}"/source_case1_th* 2>/dev/null || true
