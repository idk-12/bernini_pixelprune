# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Variable-length attention with an auto-selected backend.

Backend priority, probed once at import time:
  1. FlashAttention-3  (``flash_attn_interface``)  -- fastest on Hopper / H100
  2. FlashAttention-2  (``flash_attn``)            -- general CUDA GPUs
  3. PyTorch SDPA                                  -- always available, no extra dep

All backends share the same varlen contract: ``q``/``k``/``v`` are packed as
``[total_tokens, num_heads, head_dim]`` and ``cu_seqlens_*`` give the per-sample
offsets into ``total_tokens``.
"""

import math
from typing import Tuple

import torch
import torch.nn.functional as F

_BACKEND = None
_flash_varlen = None


def _select_backend():
    global _BACKEND, _flash_varlen
    if _BACKEND is not None:
        return
    try:
        from flash_attn_interface import flash_attn_varlen_func  # FA3

        _flash_varlen, _BACKEND = flash_attn_varlen_func, "fa3"
        return
    except Exception:
        pass
    try:
        from flash_attn import flash_attn_varlen_func  # FA2

        _flash_varlen, _BACKEND = flash_attn_varlen_func, "fa2"
        return
    except Exception:
        pass
    _BACKEND = "sdpa"


def get_attention_backend() -> str:
    _select_backend()
    return _BACKEND


def _sdpa_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, causal):
    """Varlen attention via SDPA: run each sample's segment, then concatenate."""
    cq = cu_seqlens_q.tolist()
    ck = cu_seqlens_k.tolist()
    outs = []
    for i in range(len(cq) - 1):
        # [seq, H, D] -> [1, H, seq, D]
        qi = q[cq[i] : cq[i + 1]].transpose(0, 1).unsqueeze(0)
        ki = k[ck[i] : ck[i + 1]].transpose(0, 1).unsqueeze(0)
        vi = v[ck[i] : ck[i + 1]].transpose(0, 1).unsqueeze(0)
        oi = F.scaled_dot_product_attention(qi, ki, vi, is_causal=causal)
        outs.append(oi.squeeze(0).transpose(0, 1))  # back to [seq, H, D]
    return torch.cat(outs, dim=0)


def varlen_attention(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    causal: bool = False,
):
    """Variable-length attention. Returns ``[total_q_tokens, num_heads, head_dim]``."""
    _select_backend()

    if _BACKEND == "fa3":
        out = _flash_varlen(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=int(max_seqlen_q),
            max_seqlen_k=int(max_seqlen_k),
            causal=causal,
        )
        return out[0] if isinstance(out, tuple) else out

    if _BACKEND == "fa2":
        return _flash_varlen(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            int(max_seqlen_q),
            int(max_seqlen_k),
            causal=causal,
        )

    return _sdpa_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, causal)


# --------------------------------------------------------------------------- #
# RainFusion for Bernini's packed editing sequence
# --------------------------------------------------------------------------- #

def _check_rainfusion_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    target_start: int,
    target_length: int,
    target_grid: Tuple[int, int, int],
    sparsity: float,
) -> None:
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("RainFusion expects packed TND tensors with shape [tokens, heads, head_dim].")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("Bernini RainFusion currently supports self-attention with equal Q/K/V shapes.")
    if query.device.type != "npu":
        raise RuntimeError("RainFusion is only available on Ascend NPU tensors.")
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"rainfusion_sparsity must be in [0, 1), got {sparsity}.")
    if target_start < 0 or target_length <= 0 or target_start + target_length != query.shape[0]:
        raise ValueError("The target segment must be the final, non-empty segment of the packed self-attention sequence.")
    if math.prod(target_grid) != target_length:
        raise ValueError(
            f"target_grid={target_grid} contains {math.prod(target_grid)} tokens, expected {target_length}."
        )


def _pool_blocks_bsnd(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Wan v3 average pooling: BSND tokens to one descriptor per block."""
    batch, seqlen, heads, head_dim = x.shape
    full_blocks, tail = divmod(seqlen, block_size)
    parts = []
    if full_blocks:
        full = x[:, : full_blocks * block_size]
        parts.append(full.view(batch, full_blocks, block_size, heads, head_dim).mean(dim=2))
    if tail:
        parts.append(x[:, full_blocks * block_size :].mean(dim=1, keepdim=True))
    return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]


def _keep_k_from_sparsity(scores: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Return the per-batch/per-head KV-block budget for configured sparsity."""
    kv_blocks = scores.shape[-1]
    keep_k = max(1, min(kv_blocks, math.ceil((1.0 - sparsity) * kv_blocks)))
    return torch.full(scores.shape[:2], keep_k, dtype=torch.int32, device=scores.device)


def _v3_topk_mask(scores: torch.Tensor, keep_per_head: torch.Tensor) -> torch.Tensor:
    """Build Wan v3's boolean [B, H, Q-block, KV-block] selector mask."""
    batch, heads, query_blocks, kv_blocks = scores.shape
    max_k = min(int(keep_per_head.max().item()), kv_blocks)
    topk_idx = torch.topk(scores, k=max_k, dim=-1).indices
    ranks = torch.arange(max_k, device=scores.device).view(1, 1, 1, max_k)
    keep = (ranks < keep_per_head[:, :, None, None]).expand_as(topk_idx)
    mask = torch.zeros(
        batch, heads, query_blocks, kv_blocks, dtype=torch.bool, device=scores.device
    )
    return mask.scatter_(-1, topk_idx, keep)


def _blocks_overlapping_range(
    seq_len: int,
    start: int,
    end: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return blocks whose token intervals overlap ``[start, end)``."""
    block_starts = torch.arange(0, seq_len, block_size, device=device)
    block_ends = torch.clamp(block_starts + block_size, max=seq_len)
    return (block_starts < end) & (block_ends > start)


def _protect_target_first_frame(
    mask: torch.Tensor,
    *,
    seq_len: int,
    target_start: int,
    target_grid: Tuple[int, int, int],
    block_size: int,
) -> torch.Tensor:
    """Keep all attention to and from the target video's first frame dense."""
    first_frame_end = target_start + target_grid[1] * target_grid[2]
    first_frame_blocks = _blocks_overlapping_range(
        seq_len,
        target_start,
        first_frame_end,
        block_size,
        mask.device,
    )
    mask[:, :, first_frame_blocks, :] = True
    mask[:, :, :, first_frame_blocks] = True
    return mask


def _selector_indices(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a v3 block mask to RainFusion's selected-index/count format."""
    kv_blocks = mask.shape[-1]
    block_ids = torch.arange(kv_blocks, device=mask.device, dtype=torch.int64)
    block_ids = block_ids.view(1, 1, 1, kv_blocks)
    counts = mask.sum(dim=-1, dtype=torch.int64)
    ranked = torch.where(mask, block_ids, block_ids + kv_blocks).sort(dim=-1).values
    slots = torch.arange(kv_blocks, device=mask.device).view(1, 1, 1, kv_blocks)
    indices = torch.where(slots < counts.unsqueeze(-1), ranked, -torch.ones_like(ranked))
    # The MindIE v3 kernel accepts [Q-block, H, slots] and [Q-block, H].
    return indices[0].transpose(0, 1).contiguous(), counts[0].transpose(0, 1).contiguous()


def build_editing_rainfusion_selector(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    target_start: int,
    target_grid: Tuple[int, int, int],
    sparsity: float = 0.7,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a full-visual-sequence RainFusion selector.

    Every visual query block, including reference-video/image blocks, uses the
    configured top-k KV budget. The target video's first-frame query and KV
    blocks remain fully connected.
    """
    if query.ndim != 3 or key.ndim != 3 or query.shape != key.shape:
        raise ValueError(
            "RainFusion selector expects equal packed "
            "[tokens, heads, head_dim] Q/K tensors."
        )
    seq_len = query.shape[0]
    target_length = math.prod(target_grid)
    if target_start < 0 or target_start + target_length != seq_len:
        raise ValueError(
            "The target grid must describe the final segment of the packed "
            "editing sequence."
        )
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"rainfusion_sparsity must be in [0, 1), got {sparsity}.")

    q_bsnd = query.unsqueeze(0)
    k_bsnd = key.unsqueeze(0)
    query_pool = _pool_blocks_bsnd(q_bsnd, block_size)
    key_pool = _pool_blocks_bsnd(k_bsnd, block_size)

    scores = torch.einsum("blnd,bsnd->bnls", query_pool, key_pool)
    scores = F.softmax(scores * (query.shape[-1] ** -0.5), dim=-1)
    keep_per_head = _keep_k_from_sparsity(scores, sparsity)
    mask = _v3_topk_mask(scores, keep_per_head)

    mask = _protect_target_first_frame(
        mask,
        seq_len=seq_len,
        target_start=target_start,
        target_grid=target_grid,
        block_size=block_size,
    )
    return _selector_indices(mask)


def editing_rainfusion_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    target_start: int,
    target_length: int,
    target_grid: Tuple[int, int, int],
    sparsity: float,
    block_size: int = 128,
) -> torch.Tensor:
    """Run Wan2.2 RainFusion v3's sparse student path on one packed QKV.

    This is intentionally the static attention path only: block pooling,
    sparsity-controlled routing, first-frame protection, BNSD kernel, and output.
    It has no timestep skip, dense teacher, compensation, or residual cache.
    """
    _check_rainfusion_inputs(query, key, value, target_start, target_length, target_grid, sparsity)

    try:
        from mindiesd.utils.get_platform import is_a5_device
    except ImportError:
        is_a5 = False
    else:
        is_a5 = bool(is_a5_device())

    if is_a5:
        try:
            from mindiesd.layers.flash_attn.sparse_flash_attn import sparse_attention
        except ImportError as exc:
            raise RuntimeError(
                "Ascend 950 RainFusion requires the MindIE-SD public sparse_attention API."
            ) from exc

        # The public API maps rf_v2 to RF-v3 on A5 and forces the kernel's
        # required inner_precise=4.  video_spans keeps the packed prefix as
        # dense context while sparsifying and first-frame-protecting the target.
        output = sparse_attention(
            query.unsqueeze(0),
            key.unsqueeze(0),
            value.unsqueeze(0),
            scale=query.shape[-1] ** -0.5,
            head_num=query.shape[1],
            input_layout="BSND",
            inner_precise=4,
            sparse_type="rf_v2",
            block_size=block_size,
            sparsity=sparsity,
            video_spans=[
                {
                    "start": target_start,
                    "latent_shape": list(target_grid),
                }
            ],
        )
        return output.squeeze(0).reshape_as(query).contiguous()

    try:
        from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2 import rain_fusion_attention
    except ImportError as exc:
        raise RuntimeError("RainFusion requires a compatible MindIE-SD installation.") from exc

    select_idx, select_num_idx = build_editing_rainfusion_selector(
        query,
        key,
        target_start=target_start,
        target_grid=target_grid,
        sparsity=sparsity,
        block_size=block_size,
    )
    q_bnsd = query.unsqueeze(0).transpose(1, 2).contiguous()
    k_bnsd = key.unsqueeze(0).transpose(1, 2).contiguous()
    v_bnsd = value.unsqueeze(0).transpose(1, 2).contiguous()
    output = rain_fusion_attention(
        q_bnsd,
        k_bnsd,
        v_bnsd,
        scale=query.shape[-1] ** -0.5,
        head_num=query.shape[1],
        input_layout="BNSD",
        select_idx=select_idx,
        select_num_idx=select_num_idx,
        blockshape=[block_size, block_size],
        actual_seq_lengths=[query.shape[0]],
        actual_seq_lengths_kv=[key.shape[0]],
    )
    return output.transpose(1, 2).reshape_as(query).contiguous()
