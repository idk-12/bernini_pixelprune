"""
帧间去重（Inter-Frame Deduplication）选择器。

提供三种利用视频时间冗余的 token 选择策略：

    interframe    — 纯时间：比较相邻帧对应位置，仅保留变化的 token。
    pred_2d_union — 空间 ∪ 时间：空间去重（pred_2d）或时间去重任一认为应保留即保留。
    pred_2d_seq   — 空间 ∩ 时间：空间去重 AND 时间去重都认为应保留才保留（最激进）。

所有方法对 t=1（单帧/图像）的输入均退化为：
    interframe    → 全部保留（无时间信息可比较）
    pred_2d_union → 等价于 pred_2d（仅空间）
    pred_2d_seq   → 等价于 pred_2d（仅空间）

参考帧策略：始终比较原始前一帧（frame f-1），不做 P-frame 跟踪。
阈值：共用 PIXELPRUNE_THRESHOLD，与 pred_2d 语义一致。
"""

from __future__ import annotations

from typing import List

import torch

from .base import BasePatchSelector
from .pred_2d import Pred2DSelector, _sim2d


def _temporal_keep_mask(
    tokens: torch.Tensor,
    t: int,
    h: int,
    w: int,
    method: str,
    threshold: float,
    device: torch.device,
) -> torch.Tensor:
    """
    返回 bool 张量 [t*h*w]，True 表示该位置与前一帧不相似（应保留）。
    Frame 0 全为 True（无参考帧，无条件保留）。

    完全向量化：一次 kernel 处理所有帧对。
    """
    D = tokens.shape[-1]
    g = tokens.view(t, h, w, D)

    if t == 1:
        return torch.ones(h * w, dtype=torch.bool, device=device)

    # g[1:] vs g[:-1]，shape [t-1, h, w, D]
    dissimilar = ~_sim2d(g[1:], g[:-1], method, threshold)  # [t-1, h, w]

    frame0 = torch.ones(1, h, w, dtype=torch.bool, device=device)
    return torch.cat([frame0, dissimilar], dim=0).reshape(t * h * w)


class InterFrameSelector(BasePatchSelector):
    """
    纯时间去重：对每个位置 (r,c)，比较帧 f 与帧 f-1，相似则丢弃。
    第 0 帧全部保留（无前驱帧）。对单帧输入退化为全保留。
    """

    name = "interframe"
    aliases = ["temporal"]

    def select(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        spatial_merge_size: int = 2,
    ) -> List[torch.Tensor]:
        merged_pv, merged_lengths = self._prepare_merged(
            pixel_values, image_grid_thw, spatial_merge_size
        )
        device = pixel_values.device
        result = []
        offset = 0

        for length, (t, h, w) in zip(merged_lengths, image_grid_thw.tolist()):
            img_merged = merged_pv[offset:offset + length]
            mh = h // spatial_merge_size
            mw = w // spatial_merge_size

            if t == 1:
                result.append(torch.arange(length, device=device))
            else:
                mask = _temporal_keep_mask(
                    img_merged, t, mh, mw, self.method, self.threshold, device
                )
                result.append(mask.nonzero(as_tuple=False)[:, 0])
            offset += length

        return result


class Pred2DUnionSelector(Pred2DSelector):
    """
    空间 ∪ 时间：pred_2d 空间去重 OR 帧间时间去重，任一认为重要则保留。
    比纯 pred_2d 更多保留时间变化的 token；比 interframe 更多保留空间细节。
    对 t=1 退化为标准 pred_2d。
    """

    name = "pred_2d_union"

    def select(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        spatial_merge_size: int = 2,
    ) -> List[torch.Tensor]:
        merged_pv, merged_lengths = self._prepare_merged(
            pixel_values, image_grid_thw, spatial_merge_size
        )
        device = pixel_values.device
        result = []
        offset = 0

        for length, (t, h, w) in zip(merged_lengths, image_grid_thw.tolist()):
            img_merged = merged_pv[offset:offset + length]
            mh = h // spatial_merge_size
            mw = w // spatial_merge_size

            if t == 1:
                result.append(self._select_2d_loco(img_merged, mh, mw, device))
            else:
                spatial_indices = self._select_2d_loco_fast_video(
                    img_merged, t, mh, mw, device
                )
                temporal_mask = _temporal_keep_mask(
                    img_merged, t, mh, mw, self.method, self.threshold, device
                )
                # Frame 0 无前帧参考，temporal 不应介入 union 决策，否则
                # all_True | spatial = all_True，导致第 0 帧空间去重完全失效。
                temporal_mask[:mh * mw] = False
                spatial_mask = torch.zeros(length, dtype=torch.bool, device=device)
                spatial_mask[spatial_indices] = True
                result.append(
                    (spatial_mask | temporal_mask).nonzero(as_tuple=False)[:, 0]
                )
            offset += length

        return result


class Pred2DSeqSelector(Pred2DSelector):
    """
    空间 ∩ 时间（顺序串联）：pred_2d 空间去重 AND 帧间时间去重，两者都认为重要才保留。
    剪枝最激进，保留 token 数最少。
    对 t=1 退化为标准 pred_2d。
    """

    name = "pred_2d_seq"

    def select(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        spatial_merge_size: int = 2,
    ) -> List[torch.Tensor]:
        merged_pv, merged_lengths = self._prepare_merged(
            pixel_values, image_grid_thw, spatial_merge_size
        )
        device = pixel_values.device
        result = []
        offset = 0

        for length, (t, h, w) in zip(merged_lengths, image_grid_thw.tolist()):
            img_merged = merged_pv[offset:offset + length]
            mh = h // spatial_merge_size
            mw = w // spatial_merge_size

            if t == 1:
                result.append(self._select_2d_loco(img_merged, mh, mw, device))
            else:
                spatial_indices = self._select_2d_loco_fast_video(
                    img_merged, t, mh, mw, device
                )
                temporal_mask = _temporal_keep_mask(
                    img_merged, t, mh, mw, self.method, self.threshold, device
                )
                spatial_mask = torch.zeros(length, dtype=torch.bool, device=device)
                spatial_mask[spatial_indices] = True
                keep_mask = spatial_mask & temporal_mask
                keep_by_frame = keep_mask.view(t, mh * mw)

                # The sequential intersection can be empty for a frame: its
                # spatial anchors may be temporally unchanged while all
                # temporally changed tokens are spatially predictable.  Keep
                # the per-frame spatial anchor in that case so downstream
                # video encoders always receive at least one token per frame.
                missing_frames = ~keep_by_frame.any(dim=1)
                if missing_frames.any():
                    frame_ids = missing_frames.nonzero(as_tuple=False).flatten()
                    keep_by_frame[frame_ids, 0] = True

                result.append(keep_mask.nonzero(as_tuple=False)[:, 0])
            offset += length

        return result
