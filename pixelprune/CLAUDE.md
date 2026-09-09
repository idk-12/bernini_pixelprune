# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PixelPrune compresses visual tokens **before** the ViT encoder via 2D predictive coding — training-free by default, with optional fine-tuning. It supports Qwen3-VL and Qwen3.5 on both HuggingFace and vLLM backends.

## Conda Environments

- `vllm`: video eval and Qwen3.5 eval scripts (`eval_baseline_qwen35.sh`, `eval_doc_qwen35.sh`, etc.)


Tested with `transformers==4.57.6` and `vllm==0.18.0`. Qwen3.5 + HuggingFace requires `transformers>=5.2.0`.

## eval/ vs VLMEvalKit package

`eval/vlmeval/vlm/` and `eval/run.py` override VLMEvalKit's model/prompt logic (loaded first via PYTHONPATH + script directory).

However, `eval/vlmeval/dataset/` has **no `__init__.py`**, so dataset classes (`VideoMME`, `build_dataset`, etc.) are NOT overridden and come from the editable install at `/data/lijie/VLMEvalKit/vlmeval/` (registered in the `vllm` conda env).

Consequence: to find or modify dataset-level logic (frame extraction, dataset splits, `build_dataset`), look in `/data/lijie/VLMEvalKit/vlmeval/dataset/`, not `eval/`.

## Environment Variables

All runtime behavior is controlled via environment variables — no config files needed for inference:

| Variable | Default | Description |
|---|---|---|
| `PIXELPRUNE_ENABLED` | `false` | Enable/disable PixelPrune |
| `PIXELPRUNE_METHOD` | `pred_2d` | Scan strategy: `pred_2d`, `raster`, `serpentine`, `random`, `conncomp` |
| `PIXELPRUNE_METRIC` | `max` | Distance metric: `mae`, `rmse`, `max`, `exact` |
| `PIXELPRUNE_THRESHOLD` | `0.0` | Similarity threshold τ (higher = more aggressive pruning) |
| `PIXELPRUNE_VERBOSE` | `false` | Print per-image pruning stats |
| `PIXELPRUNE_VIT_LAYER` | — | Which ViT layer to apply pruning at |
| `PIXELPRUNE_LOG_FILE` | — | Path prefix for JSONL dedup/vit/e2e log files |

## Evaluation

```bash
# Document understanding (Table 1 in paper)

bash scripts/eval_doc_qwen35.sh "MMMU_DEV_VAL"  # subset

# Full baseline (no PixelPrune)
bash scripts/eval_baseline_qwen35.sh

# GUI understanding (Table 2, requires fine-tuned checkpoint)
bash scripts/eval_gui.sh Qwen/Qwen3-VL-2B-Instruct

# Qwen3.5 document eval
bash scripts/eval_doc_qwen35.sh <model_path>
```

## Training

```bash
cd training
export PIXELPRUNE_ENABLED=true
export PIXELPRUNE_THRESHOLD=0.0
deepspeed --num_gpus 8 train.py \
    --deepspeed_config_path    configs/deepspeed_config.json \
    --training_config_path     configs/training_config.json \
    --task_info_config_path    configs/task_info.json \
    --task_weight_config_path  configs/task_weight.json
```

Training data: JSONL in ShareGPT format. Set `kd_config.alpha > 0` in `training_config.json` to enable knowledge distillation (self-distillation if `ref_model_path` is unset).

## Architecture

### Core Flow

```
apply_pixelprune(model, backend)
    → patches/<model>_<backend>.py::apply_patches()   # monkey-patches model forward
        → core.py::compute_merged_keep_indices()      # unified dispatch
            → methods/<method>.py::select()           # computes keep indices
                → dedup.py                            # distance + consecutive dedup
        → merged_indices_to_patch_indices()           # expand to patch level
        → filters pixel_values before ViT encoder
```

### Key Modules

**`pixelprune/core.py`** — `compute_merged_keep_indices()` reads env vars, creates a selector via `get_selector()`, and returns per-image keep indices. `merged_indices_to_patch_indices()` expands merged-token indices to patch-level indices (block_size = spatial_merge_size²).

**`pixelprune/methods/`** — All selectors inherit `BasePatchSelector` and implement `select(pixel_values, image_grid_thw, spatial_merge_size)`. The registry (`_REGISTRY`) maps names to classes; `register_method` and `get_selector` are the public API. `pred_2d` (LOCO-I 2D predictive coding) is the default and recommended method.

**`pixelprune/dedup.py`** — Distance metrics (MAE/RMSE/max/exact) and consecutive deduplication for packed variable-length sequences. All metrics operate on pixel values normalized to [0,1].

**`pixelprune/patches/`** — Four monkey-patch modules (`qwen3_vl_hf`, `qwen3_vl_vllm`, `qwen3_5_hf`, `qwen3_5_vllm`). Each overrides the relevant model forward method to intercept `pixel_values`, compute prune indices, and slice before ViT. Must be applied **before** `from_pretrained`.

**`eval/`** — Bundled and modified copy of VLMEvalKit. Qwen3-VL model definitions live in `eval/vlmeval/vlm/`. **All evaluation code lives here — do not look in `/data/lijie/VLMEvalKit` for evaluation logic.**

**`training/`** — DeepSpeed multi-GPU training. `train.py` is the entry point; `utils.py` handles model loading, fused kernels (Liger), and checkpoint saving. `data/dataset.py` implements sequence packing with varlen flash attention (`cu_seqlens`).

### Adding a New Selector

```python
from pixelprune.methods import register_method, BasePatchSelector

@register_method
class MySelector(BasePatchSelector):
    name = "my_method"

    def select(self, pixel_values, image_grid_thw, spatial_merge_size=2):
        merged_pv, merged_lengths = self._prepare_merged(
            pixel_values, image_grid_thw, spatial_merge_size
        )
        # return List[torch.Tensor] of kept merged-token indices per image
        ...
```

Then use it via `PIXELPRUNE_METHOD=my_method`.
