# Qwen3.5 MoE Changes Made Today

This file records only today's changes outside these newly created files:

- `pixelprune/patches/qwen3_5_moe_hf_video.py`
- `scripts/eval_baseline_qwen35_moe.sh`
- `scripts/eval_doc_qwen35_moe.sh`

## `pixelprune/__init__.py`

- Added `qwen3_5_moe` to the public `apply_pixelprune()` usage documentation.
- Updated the `apply_pixelprune()` docstring to say Qwen3.5-MoE is supported.

## `pixelprune/patches/__init__.py`

- Added `qwen3_5_moe` as a supported `model` key.
- Added HF routing for `qwen3_5_moe` to `qwen3_5_moe_hf_video.py`.
- Added vLLM routing for `qwen3_5_moe` through the existing `qwen3_5_vllm.py`.
- Updated the invalid-model error message to include `qwen3_5_moe`.

## `eval/vlmeval/vlm/qwen3_vl/model.py`

- Updated `_resolve_patch_backend()` so `MODEL_TYPE=qwen3_5_moe` selects the MoE PixelPrune patch.
- Added automatic MoE routing when the model path is a Qwen3.5/Qwen3.6 path with an MoE active-parameter suffix such as `-A3B`.
- Extended the Qwen3.5 FlashAttention-2 position-id workaround to also patch `Qwen3_5MoeDecoderLayer`.
- Updated stats-only PixelPrune imports so MoE models read helper functions from `qwen3_5_moe_hf_video.py`.
- Updated VIT/FLOPs cache reads so MoE models read `_last_vit_flops` from `qwen3_5_moe_hf_video.py`.

## `quickstart.py`

- Changed the example patch call to:

```python
apply_pixelprune(model="qwen3_5_moe")
```

- Kept the example model path as:

```python
/mnt/nfs/data/pretrained_models/Qwen3.5-35B-A3B/
```
