# Bernini PixelPrune 使用说明

本文说明 Bernini 中 PixelPrune 的作用范围、运行环境、正式评测脚本、输出文件和性能指标口径。当前目录保留四个正式入口：

| 数据集 | Baseline | PixelPrune |
| --- | --- | --- |
| EditVerse | `run_editverse_baseline.sh` | `run_editverse_pixelprune.sh` |
| OpenVE | `run_openve_baseline.sh` | `run_openve_pixelprune.sh` |

脚本面向 Linux + Ascend NPU 环境，使用 `torchrun` 启动数据并行和 Ulysses 序列并行。四个脚本同时启用 RainFusion DiT attention；baseline 与 PixelPrune 的差别仅为 `--no-pixelprune` 和 `--pixelprune`，因此应使用相同配置成对运行。

## 1. 算法作用范围

PixelPrune 只处理 full Bernini 中送入 Qwen2.5-VL planner 的 source image/video，不处理 Bernini-R，也不直接裁剪 VAE latent、target visual-output query 或 Wan DiT token。

```text
source image/video
  -> Qwen processor：dense raw patches
  -> PixelPrune selector：选择 source merged tokens
  -> Qwen2.5-VL ViT：在 patch_embed 前执行 sparse forward
  -> 压缩 Qwen planner 的 source visual-token 序列
  -> planner + connector
  -> 固定长度 renderer conditioning
  -> Wan DiT + VAE decode
```

每个保留的 merged token 都对应完整的 `spatial_merge_size²` raw-patch group。ViT 会根据保留项重新构造 window/full-attention 的 `cu_seqlens`，并保留 token 在原始 dense grid 中的 RoPE 位置。裁剪后的 ViT 和 planner 序列不会补回原长度；进入 DiT 前，renderer conditioning 仍会按 Bernini 配置补齐或截断到 512。

因此主要预期收益位于：

- source Qwen ViT；
- Qwen MLLM planner；
- 两者对应的显存占用。

PixelPrune 不会直接缩短 DiT conditioning 的最终固定长度，通常不应预期 `dit_diffusion_latency_ms` 随 visual-token reduction 等比例下降。

## 2. 环境准备

### 2.1 必需组件

运行正式脚本前需要准备：

1. 可用的 Ascend NPU、`torch_npu` 和 HCCL 环境；
2. 已安装 Bernini 依赖的 Python/Conda 环境；
3. Bernini-Diffusers 模型目录；
4. PixelPrune 源码或已安装的 `pixelprune` Python package；
5. RainFusion v3 依赖 `mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2`；
6. EditVerse 或 OpenVE 数据集。

PixelPrune 可以安装到运行环境，也可以通过 `PIXELPRUNE_ROOT` 指向源码 checkout。正式脚本会在启动前检查 PixelPrune selector 和 RainFusion v3 是否可导入。

### 2.2 通用环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CONDA_BIN` | `/root/anaconda3/bin/conda` | Conda 可执行文件 |
| `RAINFUSION_ENV` | `/home/xilab_program/envs/rainfusion` | 运行环境目录，脚本使用 `conda run -p` |
| `BERNINI_CONFIG` | `/home/xilab_program/model/Bernini-Diffusers` | full Bernini 模型目录 |
| `PIXELPRUNE_ROOT` | `/home/xilab_program/PixelPrune` | PixelPrune checkout |
| `OUTPUT_ROOT` | 按数据集和时间生成 | 本轮产物根目录 |
| `RUN_TS` | 当前分钟 | 默认输出目录中的运行标识 |
| `MAX_ITEMS` | `0` | 最多运行多少条，`0` 表示全部 |
| `START_INDEX` | `0` | 从排序后的第几条开始 |
| `NPROC_PER_NODE` | `8` | 本机进程/NPU 数 |
| `ULYSSES` | `2` | 每个 Ulysses group 的进程数 |
| `CPU_THREADS_PER_RANK` | `16` | 每个 rank 的 OMP/MKL 线程数 |
| `RAINFUSION_START_STEP` | `0` | RainFusion 开始生效的 DiT step |
| `RAINFUSION_SPARSITY` | `0.7` | RainFusion DiT attention 稀疏率 |
| `VIT_MAX_PIXELS` | `401408` | Qwen ViT 每个输入的最大像素预算 |
| `VIT_NUM_FRAMES` | `64` | 视频送入 Qwen ViT 的固定采样帧数，必须为正偶数 |

必须满足：

```text
NPROC_PER_NODE > 0
ULYSSES > 0
NPROC_PER_NODE % ULYSSES == 0
```

当前正式脚本固定使用：

```text
PixelPrune method    = pred_2d_seq
PixelPrune threshold = 0.7
DiT backend          = rainfusion
Diffusion steps      = 4
Output frames        = 81
```

需要修改这些算法参数时，应同时修改对应的 baseline 和 PixelPrune 脚本，保证对照实验只有 PixelPrune 开关不同。

## 3. 快速开始

四个文件当前可通过 `bash` 直接执行，不依赖 executable bit。

### 3.1 EditVerse

EditVerse JSON 应为以样本 ID 为 key 的对象，每条至少包含：

```json
{
  "0": {
    "<video1>": "relative/or/absolute/source.mp4",
    "<text>": "edit instruction containing optional <video1> placeholder"
  }
}
```

设置路径并运行一组小规模 smoke test：

```bash
export CONDA_BIN=/root/anaconda3/bin/conda
export RAINFUSION_ENV=/home/xilab_program/envs/rainfusion
export BERNINI_CONFIG=/home/xilab_program/model/Bernini-Diffusers
export PIXELPRUNE_ROOT=/home/xilab_program/PixelPrune
export EDITVERSE_DATA_ROOT=/home/xilab_program/datasets/EditVerse/EditVerseBench
export EDITVERSE_TEST_JSON="$EDITVERSE_DATA_ROOT/EditVerseBench.json"

export OUTPUT_ROOT="$PWD/benchmarks/editverse/rainfusion/pixelprune_check"
export MAX_ITEMS=4
export START_INDEX=0
export NPROC_PER_NODE=8
export ULYSSES=2

bash scripts/pixelprune/run_editverse_baseline.sh
bash scripts/pixelprune/run_editverse_pixelprune.sh
```

两个脚本应使用相同的 `OUTPUT_ROOT`。否则各自默认生成的 `RUN_TS` 可能不同，不利于配对比较。

EditVerse 额外支持：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `EDITVERSE_DATA_ROOT` | `/home/xilab_program/datasets/EditVerse/EditVerseBench` | 数据集根目录 |
| `EDITVERSE_TEST_JSON` | `$EDITVERSE_DATA_ROOT/EditVerseBench.json` | benchmark JSON |
| `COPY_SOURCE` | `false` | `false` 时在结果目录建立 source symlink，`true` 时复制原视频 |

### 3.2 OpenVE

OpenVE 输入 CSV 至少需要以下列：

```text
original_video,prompt
```

`original_video` 可以是绝对路径、相对 `OPENVE_ROOT` 的路径，或位于 `$OPENVE_ROOT/videos/` 下的文件名。

```bash
export CONDA_BIN=/root/anaconda3/bin/conda
export RAINFUSION_ENV=/home/xilab_program/envs/rainfusion
export BERNINI_CONFIG=/home/xilab_program/model/Bernini-Diffusers
export PIXELPRUNE_ROOT=/home/xilab_program/PixelPrune
export OPENVE_ROOT=/home/xilab_program/datasets/OpenVE
export OPENVE_CSV="$OPENVE_ROOT/benchmark_videos.csv"

export OUTPUT_ROOT="$PWD/benchmarks/openve/rainfusion/pixelprune_check"
export MAX_ITEMS=4
export START_INDEX=0
export NPROC_PER_NODE=8
export ULYSSES=2

bash scripts/pixelprune/run_openve_baseline.sh
bash scripts/pixelprune/run_openve_pixelprune.sh
```

脚本生成的 `new_result.csv` 会把 `original_video` 和 `edited_result_path` 写成绝对路径。交给 OpenVE evaluator 时应使用 `root_path=/`。

## 4. 并行方式

任务首先按 Ulysses group 做数据并行切分：

```text
data-parallel group 数 = NPROC_PER_NODE / ULYSSES
```

以默认配置 `NPROC_PER_NODE=8, ULYSSES=2` 为例，共有 4 个 data-parallel group，每组 2 个 rank 协同处理一个样本。每个 group 只有 `ulysses_rank == 0` 的进程执行 VAE decode 和输出写盘，其余 rank 在 DiT 完成后返回。

调小测试规模时，`MAX_ITEMS` 最好不小于 data-parallel group 数，以免部分 group 没有任务。显存不足时可减少 `VIT_MAX_PIXELS`、`VIT_NUM_FRAMES` 或 `NPROC_PER_NODE`，但 baseline 与 PixelPrune 必须保持一致。

## 5. 输出目录

### 5.1 EditVerse

```text
OUTPUT_ROOT/
  baseline/
    inputs.json
    manifest.json
    eval_test.json
    run.log
    metrics.rank<N>.jsonl
    results/<sample_id>/video1.mp4
    results/<sample_id>/generate.mp4
  pixelprune/
    ...同上
```

`video1.mp4` 默认是指向原视频的 symlink；使用 `COPY_SOURCE=true` 时为复制文件。

### 5.2 OpenVE

```text
OUTPUT_ROOT/
  baseline/
    inputs.json
    new_result.csv
    run.log
    metrics.rank<N>.jsonl
    results/<sample_id>.mp4
  pixelprune/
    ...同上
```

虽然命令行传入的是 `metrics.jsonl`，profiler 会自动按 rank 写成 `metrics.rank<N>.jsonl`，避免多进程并发写同一文件。

## 6. 性能指标

每行 JSONL 对应某个 rank 的一次样本执行。常用字段如下：

| 字段 | 含义 |
| --- | --- |
| `pixelprune_enabled` | 本次是否开启 PixelPrune |
| `pixelprune_method` / `pixelprune_threshold` | selector 配置 |
| `visual_input_tokens_before/after` | source visual merged-token 裁剪前后数量 |
| `visual_input_token_reduction_ratio` | source visual-token 裁剪率 |
| `mllm_sequence_length_before/after` | Qwen planner 序列裁剪前后长度 |
| `pixelprune_selector_latency_ms` | selector 外层路径耗时 |
| `pixelprune_index_conversion_latency_ms` | keep index 到 sparse ViT layout 的转换耗时 |
| `pixelprune_sequence_rebuild_latency_ms` | planner 序列压缩和 attention mask 重建耗时 |
| `pixelprune_overhead_ms` | 上述三个 PixelPrune overhead 字段之和 |
| `source_vit_latency_ms` | source 的 `mllm_model.visual(...)` forward |
| `target_vit_latency_ms` | target-output 占位视觉项的 Qwen ViT forward |
| `planner_latency_ms` | 完整 `sample_vit_embed`，包含 Qwen planner、connector 和 vit decoder |
| `vit_decoder_latency_ms` | planner 内生成目标 visual embedding 的 decoder，不是 source ViT |
| `dit_diffusion_latency_ms` | 完整 Wan DiT diffusion sampling |
| `vae_encode_latency_ms` / `vae_decode_latency_ms` | VAE 编码/解码 |
| `e2e_excluding_io_ms` | pipeline 调用开始到 VAE decode 完成；只排除输出编码/写盘 |
| `e2e_including_io_ms` | 上述时间再加输出编码和写盘 |
| `output_written` | 该 rank 是否执行 VAE decode 和写盘 |
| `gpu_peak_allocated_bytes/reserved_bytes` | 本次 pipeline 调用内的设备峰值显存 |

### 6.1 推荐比较方法

1. baseline 与 PixelPrune 使用相同数据顺序、seed、模型、并行度和 `OUTPUT_ROOT`；
2. 按 sample 和对应 rank 配对，不要把不同样本或不同 Ulysses rank 随机混合；
3. 比较 ViT 时使用顶层 `source_vit_latency_ms`；
4. 比较完整用户延迟时，只使用 `output_written=true` 的 rank；
5. 比较分布式计算阶段时，对同一个 Ulysses group 取最大 rank 时间更接近关键路径；
6. 先排除 warmup，再报告 median/p90，而不是只报告一次运行。

建议至少报告：

```text
token reduction
ViT speedup       = baseline source_vit / PixelPrune source_vit
planner speedup   = baseline planner / PixelPrune planner
DiT change        = baseline DiT 与 PixelPrune DiT 的差异
end-to-end speedup
peak memory change
```

### 6.2 指标解释注意事项

- `e2e_excluding_io_ms` 只排除输出文件编码/写盘，仍包含输入视频读取、解码和抽帧；
- 非输出 Ulysses rank 的 E2E 在 DiT 后结束，不包含 VAE decode，不能与 `output_written=true` 的 rank 混合；
- `source_vit`、selector 和 VAE encode 已包含在 `preprocess_total_latency_ms` 中，planner 子项已包含在 `planner_latency_ms` 中，所有阶段又包含在 E2E 中，不能把所有字段直接相加；
- `visual_items[*].vit_latency_ms` 是同组 ViT forward 时间，多 source item 时会重复，不能按 item 求和；
- `pixelprune_selector_latency_ms` 包含 selector 周边的切分、数据搬运、校验和同步，不代表纯 selector kernel；
- 当前 `pixelprune_sequence_rebuild_latency_ms` 只汇总 conditional planner 分支的一份 rebuild 时间，因此 `pixelprune_overhead_ms` 会低估三分支的总 rebuild 开销；
- `--pixelprune_profile` 会在子阶段使用设备 event 和同步。它适合 baseline/PixelPrune 同口径比较，但测得的 E2E 不是完全无插桩的生产延迟；
- 当前 `--pixelprune_warmup` 仅作为记录标签，pipeline 不会自动额外执行一次 warmup。

## 7. 自定义调用

不使用正式数据集脚本时，可直接在 full Bernini 的 `infer_single_gpu.py` 或 `infer_multi_gpu.py` 命令中加入：

```bash
--pixelprune \
--pixelprune_profile \
--pixelprune_threshold 0.7 \
--pixelprune_method pred_2d_seq \
--pixelprune_metrics_output /path/to/metrics.jsonl
```

公平 baseline 使用相同命令，将 `--pixelprune` 替换为：

```bash
--no-pixelprune
```

同时保留 `--pixelprune_profile`，保证两组走相同的分组和计时路径。PixelPrune 只支持 `model_type="bernini"`；对 renderer-only Bernini-R 启用 PixelPrune 会报错。没有 source visual input 的纯 t2v 任务不会产生 source-token 裁剪收益。

## 8. 常见问题

### PixelPrune is not installed

确认 package 安装在 `RAINFUSION_ENV` 中，或设置：

```bash
export PIXELPRUNE_ROOT=/absolute/path/to/PixelPrune
```

然后验证：

```bash
"$RAINFUSION_ENV/bin/python" -c \
  "from bernini.pixelprune_utils import _load_selector; _load_selector()"
```

### RainFusion v3 is unavailable

正式脚本固定使用 RainFusion DiT backend。确认运行环境能够导入：

```bash
"$RAINFUSION_ENV/bin/python" -c \
  "from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2 import rain_fusion_attention"
```

### NPROC_PER_NODE must be divisible by ULYSSES

调整两者，使 `NPROC_PER_NODE % ULYSSES == 0`。例如：

```bash
export NPROC_PER_NODE=8
export ULYSSES=2
```

### EditVerse 无法创建 source symlink

在不支持 symlink 的文件系统上使用：

```bash
export COPY_SOURCE=true
```

### Baseline 与 PixelPrune 结果无法配对

确认两次运行使用相同的 `OUTPUT_ROOT`、`MAX_ITEMS`、`START_INDEX`、并行度和数据文件，并避免向已有 metrics JSONL 反复追加不同配置的运行。

## 9. 关键实现位置

| 功能 | 文件 |
| --- | --- |
| CLI 配置和开关 | `bernini/cli.py` |
| selector、metadata、计时器 | `bernini/pixelprune_utils.py` |
| source/target 分组与端到端 profiling | `bernini/pipeline.py` |
| sparse ViT adapter | `bernini/data_utils.py` |
| Qwen2.5-VL sparse forward | `bernini/models/modeling_qwen2_5_vl.py` |
| planner token 序列压缩 | `bernini/data/bernini_process.py` |
| DiT sampling | `bernini/models/wan_diffusion.py` |
