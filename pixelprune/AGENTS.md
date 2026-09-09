# Repository Guidelines

## Project Structure & Module Organization

`pixelprune/` contains the installable package. Core pruning entry points live in `pixelprune/core.py`, selection algorithms in `pixelprune/methods/`, and HuggingFace/vLLM monkey patches in `pixelprune/patches/`. `training/` contains DeepSpeed training code, data loaders, and JSON configs. `eval/` is a modified VLMEvalKit tree for model and prompt overrides; see the dataset caveat below. Root-level `scripts/` contains evaluation and analysis utilities. `assets/` stores README/demo images.

## Conda Environments

Use the `vllm` conda environment for video and Qwen3.5 evaluation scripts, including `eval_baseline_qwen35.sh` and `eval_doc_qwen35.sh`. Tested versions are `transformers==4.57.6` and `vllm==0.18.0`; Qwen3.5 with HuggingFace requires `transformers>=5.2.0`.

## Evaluation Package Notes

`eval/vlmeval/vlm/` and `eval/run.py` override VLMEvalKit model and prompt logic because `PYTHONPATH` and script-directory precedence load them first.

`eval/vlmeval/dataset/` has no `__init__.py`, so dataset classes and helpers such as `VideoMME` and `build_dataset` come from the editable VLMEvalKit install in the `vllm` environment: `/data/lijie/VLMEvalKit/vlmeval/`. For frame extraction, splits, or `build_dataset`, inspect and modify `/data/lijie/VLMEvalKit/vlmeval/dataset/`, not `eval/`.

## Coding Style & Naming Conventions

Use Python 3.9+ and 4-space indentation. Keep public functions typed where practical. Use `snake_case` for functions, variables, modules, and environment variables such as `PIXELPRUNE_ENABLED`. Selector classes should inherit `BasePatchSelector`, define a stable `name`, and register through the existing method registry. Preserve concise docstrings for non-obvious tensor shapes and patching behavior.

## Testing Guidelines

There is no dedicated test suite in this snapshot. For package changes, run a small import/smoke check such as `python -c "from pixelprune import apply_pixelprune"`. For pruning logic, validate tensor shapes with a minimal synthetic input. For training or eval changes, run the narrowest relevant script before larger GPU jobs.

## Commit & Pull Request Guidelines

Recent history uses short summaries such as `update training code` and `support vllm`; keep subjects concise and scope-specific. In pull requests, include motivation, changed paths, commands run, and benchmark or timing impact. Link issues when available. Avoid committing checkpoints, downloaded datasets, `eval/outputs/`, or machine-specific absolute paths.

## Configuration Notes

PixelPrune behavior is controlled by environment variables including `PIXELPRUNE_ENABLED`, `PIXELPRUNE_THRESHOLD`, `PIXELPRUNE_METHOD`, `PIXELPRUNE_METRIC`, and `PIXELPRUNE_VERBOSE`. Document new variables in `README.md` and ensure defaults keep pruning disabled unless explicitly enabled.
