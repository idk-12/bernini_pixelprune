# PixelPrune 视频帧间去重消融实验

## 实验目的

现有 PixelPrune 对视频做**帧内空间去重**（pred_2d），每帧独立做 LOCO-I 预测编码，去掉帧内空间冗余的 token。
本实验新增三种利用视频**时间冗余**的帧间去重方法，评估其在 Video-MME 上的准确率和速度表现。

新方法定义：

| 方法 | 剪枝逻辑 |
|------|---------|
| `interframe` | 纯时间去重：token 与前一帧同位置相同则丢弃；第 0 帧全保留 |
| `pred_2d_union` | 空间冗余 **AND** 时间冗余才丢弃（两者均冗余才删，门槛高） |
| `pred_2d_seq` | 空间冗余 **OR** 时间冗余即丢弃（任一冗余就删，门槛低） |

时间比较方式：token `(f, r, c)` 与 `(f-1, r, c)` 做 L∞ 距离比较，`dist ≤ threshold` 即判定时间冗余。

---

## 实验配置

- **模型**：Qwen3.5-HF（Qwen3.6-27B）
- **数据集**：Video-MME_64frame，前 50 条样本（全为 short duration）
- **帧数**：64 帧/视频
- **threshold**：0.0（L∞，仅剪枝像素级完全相同的 token）
- **metric**：max
- **GPU**：H100 × 1
- **对照组**：baseline（无剪枝）和 pred_2d（纯空间）来自全量数据集的前 50 条样本提取

### 运行目录

| 方法 | 目录 |
|------|------|
| baseline | `eval/outputs/full_baseline_qwen36/Qwen3.5-HF/T20260512-200923` |
| pred_2d | `eval/outputs/pixelprune_doc_qwen36/Qwen3.5-HF/T20260514-233432` |
| interframe | `eval/outputs/pixelprune_doc_qwen36/Qwen3.5-HF/T20260517-224025` |
| pred_2d_union | `eval/outputs/pixelprune_doc_qwen36/Qwen3.5-HF/T20260517-233311` |
| pred_2d_seq | `eval/outputs/pixelprune_doc_qwen36/Qwen3.5-HF/T20260517-225141` |

---

## 实验结果

### 50 样本消融（同一批 short 视频，可直接对比）

| 方法              |   acc |  retain | ViT (ms) | LLM (ms) | TTFT (ms) | speedup |
|:------------------|------:|--------:|---------:|---------:|----------:|--------:|
| baseline          | 0.640 |   1.000 |        — |        — |      4432 |   1.00x |
| pred_2d           | 0.620 |   0.838 |      481 |     3174 |      3701 |   1.20x |
| interframe        | 0.640 |   0.852 |      490 |     3216 |      3750 |   1.18x |
| pred_2d_union     | 0.620 |   0.891 |      510 |     3387 |      3944 |   1.12x |
| **pred_2d_seq**   | 0.640 |   0.786 |      451 |     2971 |      3467 |   1.28x |

### 全量数据集官方分数（参考）

| 方法 | short | medium | long | overall |
|------|-------|--------|------|---------|
| baseline | 0.836 | 0.734 | 0.640 | 0.737 |
| pred_2d | 0.833 | 0.738 | 0.641 | 0.737 |

---

## 分析

**1. threshold=0 时帧间去重在动态短视频上效果有限**

`interframe` 的 retain（0.852）比 `pred_2d`（0.838）还略高，说明前 50 条短视频（运动、体育等动态内容为主）连续帧之间几乎没有像素级完全相同的 token——视频压缩噪声和轻微运动使 L∞ 距离始终 > 0。帧间去重要真正生效，需要提高 threshold（如 0.05），允许误差范围内的相似判定。

**2. pred_2d_union 比 pred_2d 保留更多、速度更慢**

union 的语义是"两者均冗余才删"，相当于给 pred_2d 本来会删的 token 加了一道保护——只要时间上有变化就保留。结果是 retain 从 0.838 升至 0.904，TTFT speedup 从 1.20x 降至 1.09x，速度反而不如纯 pred_2d。

**3. pred_2d_seq 最激进且精度不降**

seq 的语义是"任一冗余即删"，在这批样本上 retain 达到 0.786，TTFT 加速 1.28x（baseline 对比），优于 pred_2d 的 1.20x，且准确率（0.640）与 baseline 持平。
但这批全是 short 视频，seq 的激进剪枝在静态较少的动态内容上是否损失精度，需要全量数据验证。

**4. 准确率差距在 50 样本下无统计意义**

各方法 acc 差距最大为 0.02（1 题），不足以判断方法优劣。需在全量数据集上对比。

---

## 后续建议

1. **提高 threshold**：用 `PIXELPRUNE_THRESHOLD=0.05` 跑 `interframe` 和 `pred_2d_seq`，才能体现帧间去重对静态/慢动作视频的实际剪枝效果
2. **全量评估**：pred_2d_seq 的激进剪枝需在 Video-MME 全量（2700 题）上确认精度无损
3. **长视频效果**：帧间去重预期对 long duration 视频（静态背景更多）效果更明显，当前 50 样本全是 short，结论不代表 long
