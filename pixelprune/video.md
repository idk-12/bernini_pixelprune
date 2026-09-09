# PixelPrune 视频支持说明

## 问题背景

原始 patch (`qwen3_5_hf.py`) 只对图像输入（`pixel_values` + `image_grid_thw`）触发 PixelPrune，
视频输入（`pixel_values_videos` + `video_grid_thw`）完全绕过剪枝逻辑，导致：

- Video-MME 等视频数据集跑出来的结果与 baseline 完全相同
- 不生成 `.vit.jsonl` 日志文件

## 修复方案

**不修改原文件**，复制一份新文件并修改，原文件保留备用：

```
pixelprune/patches/qwen3_5_hf.py        # 原文件，只处理图像（保留不动）
pixelprune/patches/qwen3_5_hf_video.py  # 新文件，同时支持图像和视频剪枝
```

`pixelprune/patches/__init__.py` 中 `qwen3_5` HF 路由已更新为指向新文件：

```python
from .qwen3_5_hf_video import apply_patches as _apply
```

---

## 改动详情

### 1. `_adjust_inputs_for_dedup` — 新增 `token_id` 参数

```python
# 修改前
def _adjust_inputs_for_dedup(..., merged_indices=None):
    image_token_id = self.config.image_token_id

# 修改后
def _adjust_inputs_for_dedup(..., merged_indices=None, token_id=None):
    image_token_id = token_id if token_id is not None else self.config.image_token_id
```

**原因**：该函数通过在 `input_ids` 中查找 `vision_start_token_id` + `image_token_id` 的模式来定位视觉 token 位置，然后按剪枝后的长度裁剪序列。视频 token 的结构相同，但用的是 `video_token_id`。参数化后图像/视频均可复用。

---

### 2. `_model_forward` — 支持视频剪枝路径

新增辅助函数 `_apply_keep_mask`，提取原有的 position_ids / attention_mask 裁剪逻辑（图像、视频均复用）：

```python
def _apply_keep_mask(keep_mask, position_ids, attention_mask):
    # 按 keep_mask 裁剪 position_ids / attention_mask，左侧 pad 对齐
    ...
```

`_model_forward` 的改动：

| 改动点 | 修改前 | 修改后 |
|--------|--------|--------|
| 函数签名 | `keep_indices=None` | `keep_indices=None, video_keep_indices=None` |
| early-return 条件 | `if keep_indices is None` | `if keep_indices is None and video_keep_indices is None` |
| 图像处理 | 直接展开在函数体内 | 用 `if keep_indices is not None and pixel_values is not None` 守卫 |
| 视频处理 | 始终走原始无剪枝路径 | `video_keep_indices is not None` 时走剪枝分支，否则走原始路径 |

视频剪枝分支的逻辑与图像完全对称：

```python
if pixel_values_videos is not None and video_grid_thw is not None:
    if video_keep_indices is not None:
        merged_video_keep = [idx[::block_size] // block_size for idx in video_keep_indices]
        # 1. 调用 get_video_features（内部转 get_image_features），传入 keep_indices
        video_outputs = self.get_video_features(
            pixel_values_videos, video_grid_thw, keep_indices=vit_video_keep
        )
        # 2. 按 merged_indices 选择 embeds（vit_prune_layer=-1 时）
        # 3. 用 _adjust_inputs_for_dedup + token_id=video_token_id 裁剪序列
        # 4. 用 _apply_keep_mask 裁剪 position_ids / attention_mask
    else:
        # 无剪枝，走原始路径
        video_outputs = self.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True)
    # 5. scatter video embeds
```

**`get_video_features` 为何能传 `keep_indices`**：Qwen3.5 的 `get_video_features` 直接调用 `self.get_image_features(..., **kwargs)`，而 `get_image_features` 已被 patch 为 `_model_get_image_features`，后者显式接受 `keep_indices` 参数，并负责 ViT forward + VIT 日志写入。

---

### 3. `_cg_forward` — 计算并传递 `video_keep_indices`

原来只对图像计算 keep_indices，现在增加对视频的对称处理：

```python
# 新增：视频 keep_indices 计算
if PIXELPRUNE_ENABLED and pixel_values_videos is not None and video_grid_thw is not None:
    pixel_values_videos_norm = normalize_pixel_values_for_selector(pixel_values_videos)
    merged_video_keep_indices = compute_merged_keep_indices(
        pixel_values_videos_norm, video_grid_thw, spatial_merge_size=spatial_merge_size,
    )
    video_keep_indices = merged_indices_to_patch_indices(merged_video_keep_indices, block_size, ...)
    _store_dedup_stats(original_merged_lengths_v, new_merged_lengths_v, input_ids, video_grid_thw, ...)

# baseline 模式（PIXELPRUNE_ENABLED=false）也写 vit log（100% retain）
elif SHOULD_PROFILE_VIT and pixel_values_videos is not None and video_grid_thw is not None:
    _store_dedup_stats(original_merged_lengths_v, original_merged_lengths_v, ...)
```

`_store_dedup_stats` 写入 `_pending_vit_record[rank]`，后续 `_model_get_image_features` 会补充 ViT latency/FLOPS 信息并写到 `.vit.rank0.jsonl`。

`video_keep_indices` 通过 `_orig_cg_forward(..., video_keep_indices=video_keep_indices)` → 原始 CG forward 的 `**kwargs` → `self.model(**kwargs)` → `_model_forward(video_keep_indices=...)` 传递。

---

## 数据流总览

```
_cg_forward
  ├─ compute_merged_keep_indices(pixel_values_videos, video_grid_thw)
  ├─ _store_dedup_stats(video stats)          → _pending_vit_record[rank]
  └─ _orig_cg_forward(..., video_keep_indices=...)
       └─ self.model(video_keep_indices=...)  → _model_forward
            ├─ get_video_features(keep_indices=vit_video_keep)
            │    └─ get_image_features        → _model_get_image_features
            │         ├─ _vision_forward (带剪枝)
            │         └─ 写 .vit.rank0.jsonl  ← _pending_vit_record flush
            ├─ _adjust_inputs_for_dedup(token_id=video_token_id)
            │    → 裁剪掉多余的 video token 位置
            ├─ _apply_keep_mask → 裁剪 position_ids / attention_mask
            └─ scatter video embeds → language_model
```

---

## 计时埋点补充（第三轮）

原来 vit.jsonl 只记录 `selector_latency_ms`（仅 `compute_merged_keep_indices` 的耗时）。为定位 prune 路径各步骤开销，新增以下四个计时字段：

| 字段 | 计时范围 | 代码位置 |
|------|---------|---------|
| `normalize_latency_ms` | `normalize_pixel_values_for_selector` | `_cg_forward`，selector 之前 |
| `index_convert_latency_ms` | `merged_indices_to_patch_indices` | `_cg_forward`，selector 之后 |
| `adjust_latency_ms` | `_adjust_inputs_for_dedup` | `_model_forward`，视频/图像剪枝块内 |
| `apply_mask_latency_ms` | `_apply_keep_mask` | `_model_forward`，adjust 之后 |

前两个字段在 `_cg_forward` 中测量并通过 `_store_dedup_stats` 写入 `_pending_vit_record`，初始值为 None；后两个字段在 `_model_forward` 中测量，通过 `_pending_vit_record[rank].update(...)` 填入，与现有的 `vit_latency_ms` 更新机制相同。

### jsonl 写入时序调整

新增计时字段带来一个竞争问题：`adjust` / `apply_mask` 在 `_model_forward` 中发生，而原来 `_model_get_image_features` 在 ViT forward 完成后立即 `pop` 并写入 jsonl，导致 `_model_forward` 再去 update 时 record 已不存在。

修复方式：

- `_model_get_image_features`：将 `_write_jsonl(_pending_vit_record.pop(rank), ...)` 改为只 `update`，不写不 pop
- `_model_forward` 图像/视频剪枝块：update adjust/apply_mask 后调用 `flush_pending_vit_record(rank)` 写入
- `_model_forward` early-return（baseline 路径）：`_orig_model_forward` 返回后调用 `flush_pending_vit_record(rank)`，保证 baseline profiling 模式下记录也能落盘

---

## 运行时 Bug 修复（第二轮）

初版视频支持上线后跑 Video-MME 出现两处运行时错误，分别修复如下。

---

### Bug 1：`pred_2d.py` — 多帧视频 reshape 失败

**错误**
```
RuntimeError: shape '[20, 36, 3072]' is invalid for input of size 70778880
File "pred_2d.py", line 157, in _select_2d_loco_fast
    g = tokens.view(h, w, D)
```

**原因**：`select()` 对每个 `(t, h, w)` 视频条目调用 `_select_2d_loco(img_merged, merged_h, merged_w)`，传入的 `img_merged` 包含了全部 `t * merged_h * merged_w` 个 merged token，而 `_select_2d_loco_fast` 内部做 `tokens.view(h, w, D)` 只期望 `h * w` 个 token（单帧），因此 reshape 失败。

**修复**（`pixelprune/methods/pred_2d.py`）：当 `t > 1` 时，逐帧切片调用 `_select_2d_loco`，并将每帧保留的索引加上帧偏移量后拼接：

```python
if t == 1:
    indices = self._select_2d_loco(img_merged, merged_h, merged_w, device)
else:
    frame_len = merged_h * merged_w
    frame_indices = [
        self._select_2d_loco(
            img_merged[f * frame_len:(f + 1) * frame_len],
            merged_h, merged_w, device,
        ) + f * frame_len
        for f in range(t)
    ]
    indices = torch.cat(frame_indices)
```

---

### Bug 2：`qwen3_5_hf_video.py` — `_adjust_inputs_for_dedup` 越界

**错误**
```
IndexError: index 1 is out of bounds for dimension 0 with size 1
File "qwen3_5_hf_video.py", line 237, in _adjust_inputs_for_dedup
    t, h, w = (int(x) for x in image_grid_thw[img_idx])
```

**原因**：Qwen3.5 把一个 t 帧视频拆分成 **t 个独立的 `vision_start + video_token` block** 写入 `input_ids`（见 `modeling_qwen3_5.py:1424` 的 `repeat_interleave` 逻辑）。`_adjust_inputs_for_dedup` 在 `input_ids` 中找到了 t 个视频位置（`image_positions` 长度为 t），却用 `video_grid_thw[img_idx]` 按顺序取格子——而 `video_grid_thw` 只有 1 条记录 `[t, h, w]`，导致 `img_idx=1` 越界。

**修复**（`_model_forward` 视频剪枝块，`qwen3_5_hf_video.py`）：调用 `_adjust_inputs_for_dedup` 前，将 `video_grid_thw` 从 `[[t, h, w]]` 展开为 `t × [[1, h, w]]`，同时把 `merged_video_keep` 和 `video_embeds` 按帧切分：

```python
expanded_grid_list, expanded_merged_keep, expanded_embeds = [], [], []
for v_idx, (t_v, h_v, w_v) in enumerate(video_grid_thw.tolist()):
    mh_v, mw_v = h_v // spatial_merge_size, w_v // spatial_merge_size
    frame_sz = mh_v * mw_v
    v_keep = merged_video_keep[v_idx]   # 整段视频的 merged 保留索引
    v_emb  = video_embeds[v_idx]        # 整段视频的已剪枝 embeddings
    emb_off = 0
    for f in range(t_v):
        lo, hi = f * frame_sz, (f + 1) * frame_sz
        frame_mask = (v_keep >= lo) & (v_keep < hi)
        n_kept_f = int(frame_mask.sum())
        expanded_merged_keep.append(v_keep[frame_mask] - lo)        # 帧内相对索引
        expanded_embeds.append(v_emb[emb_off : emb_off + n_kept_f])
        expanded_grid_list.append([1, h_v, w_v])
        emb_off += n_kept_f
expanded_video_grid_thw = torch.tensor(
    expanded_grid_list, dtype=video_grid_thw.dtype, device=video_grid_thw.device
)
# 展开后传给 _adjust_inputs_for_dedup
inputs_embeds, input_ids, video_keep_mask = _adjust_inputs_for_dedup(
    self, inputs_embeds, input_ids,
    expanded_embeds, expanded_video_grid_thw, "left",
    merged_indices=expanded_merged_keep,
    token_id=self.config.video_token_id,
)
```

展开逻辑与 `get_placeholder_mask` 内部的 `repeat_interleave` 完全对应，保证 `image_positions` 数量和 grid 条目数一致。

---

## 帧采样说明（Video-MME_64frame）

### 数据集侧 nframe 对视频模型无效

`Video-MME_64frame` 在 VLMEvalKit 中定义为：

```python
'Video-MME_64frame': partial(VideoMME, dataset='Video-MME', nframe=64)
```

数据集的 `nframe=64` 只在 `video_llm=False` 时生效（此时 VLMEvalKit 负责提取帧并以图像列表传入）。
但 Qwen3.5-HF 模型设置了 `VIDEO_LLM = True`（`eval/vlmeval/vlm/qwen3_vl/model.py:274`），
`build_prompt`（`/data/lijie/VLMEvalKit/vlmeval/dataset/videomme.py:202`）走的是：

```python
if video_llm:
    message.append(dict(type='video', value=<mp4路径>))
```

即直接传原始 mp4，**数据集的 nframe=64 完全不生效**，帧缓存目录（`LMUData/images/Video-MME/`）也因此为空。

### 真正生效的 nframe

模型侧在 `eval/vlmeval/config.py:1300` 配置：

```python
nframe=int(os.environ.get("nframe", 64))
```

该值经 `model.py` → `item['nframes']=64` → `qwen_vl_utils.smart_nframes` → 最终采样：

```python
idx = torch.linspace(0, total_frames - 1, 64).round().long()
```

在视频首尾之间均匀取 64 帧（无随机，结果固定）。如需修改帧数，应通过 `export nframe=N` 设置，
而非修改 `Video-MME_Nframe` 的数据集名。

---

## 性能分析

### 计时分解（Video-MME_64frame）

PixelPrune 引入的额外代码开销（各阶段均值）：

| 步骤 | 均值 |
|------|------|
| normalize_latency_ms | ~1 ms |
| selector_latency_ms | ~10 ms |
| index_convert_latency_ms | ~13 ms |
| expand_latency_ms | ~6 ms |
| adjust_latency_ms | ~4 ms |
| apply_mask_latency_ms | ~1 ms |
| **prune code 合计** | **~35 ms** |

额外代码开销约 35ms，相对于秒级的 ViT/LLM 时延可忽略不计。

---

### Bug：`_vision_forward` 中 `cu_seqlens` 构造错误导致视频 ViT 暴慢

#### 现象（修复前，T20260514-225644）

| 指标 | Baseline | PixelPrune（修复前） |
|------|----------|---------------------|
| ViT latency | 611 ms | 4692 ms（**7.7x 更慢**） |
| LLM prefill | 3989 ms | 3438 ms（1.16x 更快） |
| TTFT | 4624 ms | 8240 ms（**1.78x 更慢**） |
| retain ratio | 1.00 | 0.87 |

#### 根因

原始 ViT forward 对每帧做独立 attention，`cu_seqlens` 以帧为单位划分：

```python
# 原始：grid_thw=[[32,40,72]] → 32 帧，每帧 2880 tokens
cu_seqlens = [0, 2880, 5760, ..., 92160]   # 32 个子序列
# attention 复杂度：32 × 2880² ≈ 2.65 亿/层
```

`_vision_forward` 中剪枝后直接用 `len(keep_indices[i])` 构造 `cu_seqlens`：

```python
# 修复前：所有保留 token 合并为一条长序列
seq_lengths = [len(idx) for idx in keep_indices]  # → [80060]
cu_seqlens = [0, 80060]
# attention 复杂度：80060² ≈ 64 亿/层（24x）
```

token 减少了 13%，但 attention 计算量反而增大了约 24x，导致 ViT 从 611ms 暴增到 4692ms。

#### 修复（`qwen3_5_hf_video.py`）

新增 `_frame_cu_seqlens`，按帧拆分 keep_indices 恢复原始 per-frame attention pattern：

```python
def _frame_cu_seqlens(keep_indices, grid_thw, device):
    seq_lengths = []
    for seq_idx, (t, h, w) in enumerate(grid_thw.tolist()):
        frame_sz = h * w
        idx = keep_indices[seq_idx]
        for f in range(t):
            lo, hi = f * frame_sz, (f + 1) * frame_sz
            seq_lengths.append(int(((idx >= lo) & (idx < hi)).sum().item()))
    return F.pad(torch.tensor(seq_lengths, ...).cumsum(0), (1, 0), value=0)
```

`_vision_forward` 中两处（`vit_prune_layer==0` 和中间层剪枝）均替换为 `_frame_cu_seqlens`。

#### 修复后效果（T20260514-233131，Video-MME_64frame，16 samples）

| 指标 | Baseline | PixelPrune（修复后） | Speedup |
|------|----------|---------------------|---------|
| ViT latency | 611 ms | 428 ms | **1.43x** |
| LLM prefill | 3989 ms | 2739 ms | **1.46x** |
| TTFT | 4624 ms | 3220 ms | **1.44x** |
| retain ratio | 1.00 | 0.70 | — |
| merged tokens | 23040 | 16169 | — |

ViT 和 LLM prefill 均正常加速，整体 TTFT 达到 **1.44x 加速**。

retain=1.0 样本（无剪枝）ViT≈598ms、prefill≈3977ms，与 baseline 吻合，验证修复正确性。

---

## 兼容性说明

- **图像数据集不受影响**：`video_keep_indices` 默认为 `None`，图像路径代码路径与原来完全相同。
- **混合输入（图像+视频）**：图像和视频分别独立计算 keep_indices，串行处理。  
  注意：两者都调用 `_store_dedup_stats`，后者会覆盖 `_pending_vit_record[rank]`，vit log 只记录最后一次（视频）的统计。混合场景的日志精度有限，纯视频/纯图像场景均正确。
- **`qwen3_vl_hf.py`**：存在相同问题，本次未修复。如需对 Qwen3-VL 模型跑视频，需做同样改动。

---

## 帧间去重（Inter-Frame Deduplication）

### 动机

现有 PixelPrune 对视频做**帧内空间去重**：对每帧独立运行 pred_2d，去掉帧内空间冗余的 token。
但视频还存在大量**时间冗余**：静态背景的 token 在相邻帧几乎不变。帧间去重在此基础上额外利用这部分冗余。

### 新增文件

`pixelprune/methods/interframe.py`，注册三个新方法：

| 方法名 | 策略 | 适用场景 |
|--------|------|---------|
| `interframe` | 纯时间：仅比较相邻帧对应位置 | 时间冗余为主的静态视频 |
| `pred_2d_union` | 空间 ∪ 时间：两者任一重要则保留 | 平衡保留，剪枝温和 |
| `pred_2d_seq` | 空间 ∩ 时间：两者均重要才保留 | 最激进剪枝 |

对 `t=1`（图像/单帧）输入：
- `interframe` 退化为全保留（retain=1.0）
- `pred_2d_union` / `pred_2d_seq` 退化为标准 pred_2d

### 核心实现：`_temporal_keep_mask`

```python
def _temporal_keep_mask(tokens, t, h, w, method, threshold, device):
    """True = 该位置与前一帧不相似（应保留）。Frame 0 全为 True。"""
    D = tokens.shape[-1]
    g = tokens.view(t, h, w, D)          # [t, h, w, D]

    if t == 1:
        return torch.ones(h * w, dtype=torch.bool, device=device)

    # 向量化：一次比较所有相邻帧对
    dissimilar = ~_sim2d(g[1:], g[:-1], method, threshold)  # [t-1, h, w]

    frame0 = torch.ones(1, h, w, dtype=torch.bool, device=device)
    return torch.cat([frame0, dissimilar], dim=0).reshape(t * h * w)
```

- 复用 `pred_2d.py` 中的 `_sim2d`，支持 mae / rmse / max / exact 度量
- 共用 `PIXELPRUNE_THRESHOLD`，无需新增环境变量
- 参考帧：始终比较原始前一帧（frame f-1），不做 P-frame 跟踪

### 激活方式

```bash
# 纯时间去重
PIXELPRUNE_METHOD=interframe PIXELPRUNE_THRESHOLD=0.05 bash scripts/eval_doc_qwen35.sh "Video-MME_64frame"

# 空间 ∪ 时间（并集，温和）
PIXELPRUNE_METHOD=pred_2d_union PIXELPRUNE_THRESHOLD=0.05 bash scripts/eval_doc_qwen35.sh "Video-MME_64frame"

# 空间 ∩ 时间（交集，最激进）
PIXELPRUNE_METHOD=pred_2d_seq PIXELPRUNE_THRESHOLD=0.05 bash scripts/eval_doc_qwen35.sh "Video-MME_64frame"
```

### 与现有架构的关系

不需要修改 patch 层（`qwen3_5_hf_video.py`）或 `core.py`。
三个新 selector 注册到 `_REGISTRY` 后，`compute_merged_keep_indices` 按 `PIXELPRUNE_METHOD` 自动 dispatch。
`Pred2DUnionSelector` / `Pred2DSeqSelector` 继承 `Pred2DSelector`，复用 `_select_2d_loco_fast_video` 做空间部分。
