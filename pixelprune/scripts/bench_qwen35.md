# Qwen3.5 Benchmark: 精度 + 速度对比方案

对比 Baseline（无 PixelPrune）和 PixelPrune（pred_2d）在同一数据集上的精度与推理时间。

---

## 环境

- **Conda 环境**：`vlmeval`（含 pandas、transformers 5.7、torch 2.8）
- **模型路径**：`/mnt/nfs/data/pretrained_models/Qwen3.5-35B-A3B`
- **VLMEvalKit**：`/data/lijie/VLMEvalKit`（作为 vlmeval 基础包）

---

## 快速开始（一键对比）

```bash
conda activate vlmeval
cd /data/lijie/PixelPrune

# 单卡，跑 DocVQA_VAL
bash scripts/bench_qwen35.sh DocVQA_VAL 0

# 多卡
bash scripts/bench_qwen35.sh DocVQA_VAL 0,1,2,3

# 换数据集
bash scripts/bench_qwen35.sh ChartQA_TEST 0
```

脚本依次执行：
1. **Baseline**（`PIXELPRUNE_ENABLED=false`）→ 计时 → 写精度结果
2. **PixelPrune**（`pred_2d, τ=0`）→ 计时 → 打印每张图压缩率 → 写精度结果
3. 打印 wall time speedup 和所有输出路径

---

## 单独跑某一侧

### 仅跑 Baseline

```bash
conda activate vlmeval
cd /data/lijie/PixelPrune
bash scripts/eval_full_baseline_qwen35.sh \
    /mnt/nfs/data/pretrained_models/Qwen3.5-35B-A3B \
    DocVQA_VAL \
    0
```

输出目录：`eval/outputs/full_baseline_qwen35/`
时间日志：`eval/outputs/full_baseline_qwen35/logs/e2e.jsonl`

### 仅跑 PixelPrune

```bash
conda activate vlmeval
cd /data/lijie/PixelPrune
bash scripts/eval_doc_qwen35.sh \
    /mnt/nfs/data/pretrained_models/Qwen3.5-35B-A3B \
    DocVQA_VAL \
    0
```

输出目录：`eval/outputs/pixelprune_doc_qwen35/`
时间日志：`eval/outputs/pixelprune_doc_qwen35/logs/e2e.jsonl`
VIT 日志：`eval/outputs/pixelprune_doc_qwen35/logs/vit.jsonl`

---

## 分析结果

```bash
python scripts/analyze_timing.py \
    --baseline-dir eval/outputs/full_baseline_qwen35 \
    --pp-dir       eval/outputs/pixelprune_doc_qwen35 \
    --dataset      DocVQA_VAL
```

输出内容：

| 指标 | 说明 |
|---|---|
| 每样本推理时间（mean/median/total） | 来自 `e2e.jsonl` |
| TTFT（Time To First Token） | 来自 `e2e.jsonl` |
| 输入 token 数减少比例 | baseline vs PixelPrune |
| 每张图压缩率（retain ratio） | 来自 `vit.jsonl` |
| 精度结果文件路径 | VLMEvalKit 输出目录 |

---

## 日志文件说明

每个日志文件由 `DEDUP_LOG_FILE` / `PIXELPRUNE_LOG_FILE` 写出，每行一条 JSON。

### `e2e.jsonl`（每次推理一条）

| 字段 | 说明 |
|---|---|
| `total_time_s` | 单次推理总时间（秒） |
| `ttft_ms` | Time To First Token（毫秒） |
| `decoding_time_s` | 解码阶段时间 |
| `decoding_speed_tok_per_s` | 解码速度（token/s） |
| `num_input_tokens` | 输入 token 数（含视觉 token） |
| `num_generated_tokens` | 生成 token 数 |
| `vram_before_gb / vram_after_gb` | 显存变化 |
| `sample_idx` | 数据集内的样本编号 |

### `vit.jsonl`（PixelPrune 专用，每次推理一条）

| 字段 | 说明 |
|---|---|
| `retain_ratios` | 每张图保留的 merged token 比例 |
| `org_merged_lens` | 原始 merged token 数（per image） |
| `new_merged_lens` | 压缩后 merged token 数（per image） |
| `llm_prune_len` | LLM 输入减少的 token 数 |
| `org_input_len / new_input_len` | 压缩前/后总输入长度 |
| `selector_latency_ms` | 计算 keep indices 的耗时 |

---

## 关键特性确认

| 特性 | 实现方式 |
|---|---|
| **Thinking 关闭** | `apply_chat_template(..., enable_thinking=False)`，代码已有 |
| **每样本推理时间** | `VERBOSE=true` 开启 TextIteratorStreamer 计时 |
| **时间写文件** | `DEDUP_LOG_FILE=<path>` 写 `{path}.e2e.rank0.jsonl` |
| **每张图压缩率** | `PIXELPRUNE_VERBOSE=true` 打印 + `PIXELPRUNE_LOG_FILE` 写文件 |
| **本地模型加载** | `model_path=<local_path>` 通过 env var 传入，直接 `from_pretrained` |

---

## 文件结构

```
scripts/
├── bench_qwen35.sh              # 一键 baseline + PixelPrune 对比脚本
├── eval_doc_qwen35.sh           # 单独跑 PixelPrune（含 timing + verbose）
├── eval_full_baseline_qwen35.sh # 单独跑 Baseline（含 timing）
├── analyze_timing.py            # 解析 JSONL 输出对比表格
└── bench_qwen35.md              # 本文档

eval/
├── run.py                       # VLMEvalKit run.py + PixelPrune Qwen3VLChat 注入
└── outputs/
    ├── full_baseline_qwen35/
    │   ├── logs/e2e.jsonl       # baseline 时间日志
    │   └── Qwen3.5-HF/         # VLMEvalKit 精度结果
    └── pixelprune_doc_qwen35/
        ├── logs/e2e.jsonl       # PixelPrune 时间日志
        ├── logs/vit.jsonl       # 压缩率日志
        └── Qwen3.5-HF/         # VLMEvalKit 精度结果
```

---

## 注意事项

- 运行前必须激活 `vlmeval` conda 环境（有 pandas、transformers 5.7）
- `eval/run.py` 依赖 `/data/lijie/VLMEvalKit` 作为 vlmeval 基础包，路径硬编码在文件头部
- 时间日志在**每次推理后追加**写入，断点重跑不会清空旧数据，分析前注意清理或重命名旧日志
- 多卡用 `torchrun`，日志文件后缀为 `rank0.jsonl`、`rank1.jsonl` 等，`analyze_timing.py` 会自动合并
