"""Bernini-specific PixelPrune selection, metadata, and profiling helpers."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.distributed as dist

from .device import accelerator_module, synchronize

logger = logging.getLogger("bernini.pixelprune")

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off", ""}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


@dataclass(frozen=True)
class PixelPruneConfig:
    enabled: bool = False
    threshold: float = 0.0
    method: str = "pred_2d"
    verbose: bool = False
    profile: bool = False
    metrics_output: Optional[str] = None
    warmup: bool = False

    @classmethod
    def from_env(cls) -> "PixelPruneConfig":
        return cls(
            enabled=_env_bool("PIXELPRUNE_ENABLED", False),
            threshold=float(os.environ.get("PIXELPRUNE_THRESHOLD", "0") or 0),
            method=os.environ.get("PIXELPRUNE_METHOD", "pred_2d").strip().lower(),
            verbose=_env_bool("PIXELPRUNE_VERBOSE", False),
            profile=_env_bool("PIXELPRUNE_PROFILE", False),
            metrics_output=os.environ.get("PIXELPRUNE_METRICS_OUTPUT") or None,
            warmup=_env_bool("PIXELPRUNE_WARMUP", False),
        )


@dataclass
class VisualItemMetadata:
    item_type: str
    role: str
    grid_thw: list[int]
    original_raw_patch_count: int
    original_merged_token_count: int
    kept_merged_token_count: int
    merged_keep_indices: Optional[torch.Tensor] = None
    per_frame_kept_token_count: list[int] = field(default_factory=list)
    selector_latency_ms: float = 0.0
    index_conversion_latency_ms: float = 0.0
    vit_latency_ms: float = 0.0

    @property
    def reduction_ratio(self) -> float:
        if self.original_merged_token_count == 0:
            return 0.0
        return 1.0 - self.kept_merged_token_count / self.original_merged_token_count

    def serializable(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("merged_keep_indices", None)
        value["token_reduction_ratio"] = self.reduction_ratio
        return value


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def rank_metrics_path(path: str, rank: Optional[int] = None) -> Path:
    rank = get_rank() if rank is None else rank
    output = Path(path)
    suffix = output.suffix or ".jsonl"
    stem = output.name[: -len(output.suffix)] if output.suffix else output.name
    return output.with_name(f"{stem}.rank{rank}{suffix}")


def append_jsonl(path: str, record: dict[str, Any]) -> Path:
    output = rank_metrics_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output


class CudaStageTimer:
    """CUDA-event timer with a CPU fallback for unit tests."""

    def __init__(self, enabled: bool, device: Optional[torch.device] = None):
        self.enabled = enabled
        self.device = device
        self.elapsed_ms = 0.0
        self._start_event = None
        self._end_event = None
        self._cpu_start = None

    def __enter__(self) -> "CudaStageTimer":
        if not self.enabled:
            return self
        module = accelerator_module(self.device) if self.device is not None else None
        if module is not None and module.is_available():
            self._start_event = module.Event(enable_timing=True)
            self._end_event = module.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._cpu_start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self.enabled:
            return
        if self._start_event is not None:
            self._end_event.record()
            self._end_event.synchronize()
            self.elapsed_ms = self._start_event.elapsed_time(self._end_event)
        else:
            self.elapsed_ms = (time.perf_counter() - self._cpu_start) * 1000


def _load_selector():
    try:
        from pixelprune import compute_merged_keep_indices
    except ImportError as exc:
        pixelprune_root = Path(
            os.environ.get("PIXELPRUNE_ROOT", "/data/lijie/PixelPrune")
        )
        if (pixelprune_root / "pixelprune" / "__init__.py").is_file():
            sys.path.insert(0, str(pixelprune_root))
            from pixelprune import compute_merged_keep_indices
        else:
            raise RuntimeError(
                "PixelPrune is enabled but the 'pixelprune' package is unavailable. "
                "Install it in the 'bernini' Conda environment or set "
                "PIXELPRUNE_ROOT to the PixelPrune checkout."
            ) from exc
    return compute_merged_keep_indices


def validate_merged_keep_indices(
    keep_indices: Sequence[torch.Tensor],
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> None:
    if len(keep_indices) != len(grid_thw):
        raise ValueError(
            f"expected {len(grid_thw)} keep-index tensors, got {len(keep_indices)}"
        )
    for item_index, (keep, grid) in enumerate(zip(keep_indices, grid_thw.tolist())):
        t, h, w = (int(value) for value in grid)
        if h % spatial_merge_size or w % spatial_merge_size:
            raise ValueError(
                f"visual item {item_index} grid {(t, h, w)} is not divisible by "
                f"spatial_merge_size={spatial_merge_size}"
            )
        total = t * (h // spatial_merge_size) * (w // spatial_merge_size)
        keep = keep.detach()
        if keep.ndim != 1 or keep.dtype != torch.long:
            raise ValueError("merged keep indices must be one-dimensional torch.long tensors")
        if keep.numel() == 0:
            raise ValueError(f"visual item {item_index} cannot prune every merged token")
        if not torch.equal(keep, torch.unique(keep, sorted=True)):
            raise ValueError(f"visual item {item_index} keep indices must be sorted and unique")
        if int(keep[0]) < 0 or int(keep[-1]) >= total:
            raise ValueError(
                f"visual item {item_index} keep indices are outside [0, {total})"
            )
        merged_per_frame = (h // spatial_merge_size) * (w // spatial_merge_size)
        for frame_index in range(t):
            frame_count = (
                (keep >= frame_index * merged_per_frame)
                & (keep < (frame_index + 1) * merged_per_frame)
            ).sum()
            if int(frame_count) == 0:
                raise ValueError(
                    f"visual item {item_index} frame {frame_index} has no kept token"
                )


def merged_to_raw_patch_indices(
    keep_indices: Sequence[torch.Tensor],
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Expand item-local merged indices into packed, global raw-patch indices."""

    validate_merged_keep_indices(keep_indices, grid_thw, spatial_merge_size)
    block_size = spatial_merge_size**2
    selected = []
    raw_offset = 0
    for keep, grid in zip(keep_indices, grid_thw.tolist()):
        keep = keep.to(device=device, dtype=torch.long)
        offsets = torch.arange(block_size, device=device, dtype=torch.long)
        selected.append((keep[:, None] * block_size + offsets).reshape(-1) + raw_offset)
        raw_offset += int(grid[0]) * int(grid[1]) * int(grid[2])
    return torch.cat(selected)


def compute_visual_keep_metadata(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    item_type: str,
    roles: Sequence[str],
    spatial_merge_size: int,
    config: PixelPruneConfig,
    selector_device: Optional[torch.device] = None,
) -> list[VisualItemMetadata]:
    """Run PixelPrune only for source-input items and return item-local metadata."""

    if len(roles) != len(grid_thw):
        raise ValueError(f"roles/grid mismatch: {len(roles)} roles for {len(grid_thw)} items")

    item_patch_values = []
    offset = 0
    for grid in grid_thw.tolist():
        raw_count = int(grid[0]) * int(grid[1]) * int(grid[2])
        item_patch_values.append(pixel_values[offset : offset + raw_count])
        offset += raw_count
    if offset != pixel_values.shape[0]:
        raise ValueError(
            f"grid describes {offset} raw patches but pixel_values has {pixel_values.shape[0]}"
        )

    source_indices = [index for index, role in enumerate(roles) if role == "source-input"]
    source_keeps: dict[int, torch.Tensor] = {}
    selector_latency_ms = 0.0
    if config.enabled and source_indices:
        selector = _load_selector()
        start = time.perf_counter()
        source_pixels = torch.cat(
            [item_patch_values[index] for index in source_indices]
        )
        source_grids = grid_thw[source_indices]
        if selector_device is not None:
            source_pixels = source_pixels.to(selector_device)
            source_grids = source_grids.to(selector_device)
        selected = selector(
            source_pixels,
            source_grids,
            spatial_merge_size=spatial_merge_size,
            method=config.method,
            metric="max",
            threshold=config.threshold,
        )
        if source_pixels.device.type in ("cuda", "npu"):
            synchronize(source_pixels.device)
        selector_latency_ms = (time.perf_counter() - start) * 1000
        validate_merged_keep_indices(selected, source_grids, spatial_merge_size)
        source_keeps = dict(zip(source_indices, selected))

    metadata = []
    for index, (grid, role) in enumerate(zip(grid_thw.tolist(), roles)):
        t, h, w = (int(value) for value in grid)
        merged_h = h // spatial_merge_size
        merged_w = w // spatial_merge_size
        merged_count = t * merged_h * merged_w
        keep = source_keeps.get(index)
        if role == "source-input" and keep is None:
            keep = torch.arange(merged_count, device=pixel_values.device, dtype=torch.long)
        per_frame = []
        if keep is not None:
            frame_size = merged_h * merged_w
            per_frame = [
                int(((keep >= frame * frame_size) & (keep < (frame + 1) * frame_size)).sum())
                for frame in range(t)
            ]
        item_metadata = VisualItemMetadata(
            item_type=item_type,
            role=role,
            grid_thw=[t, h, w],
            original_raw_patch_count=t * h * w,
            original_merged_token_count=merged_count,
            kept_merged_token_count=int(keep.numel()) if keep is not None else merged_count,
            merged_keep_indices=keep,
            per_frame_kept_token_count=per_frame,
            selector_latency_ms=selector_latency_ms if index in source_keeps else 0.0,
        )
        metadata.append(item_metadata)
        if config.verbose:
            logger.info(
                "PixelPrune %s %s grid=%s raw=%d merged=%d kept=%d reduction=%.2f%% "
                "per_frame=%s selector=%.3fms",
                item_type,
                role,
                item_metadata.grid_thw,
                item_metadata.original_raw_patch_count,
                merged_count,
                item_metadata.kept_merged_token_count,
                item_metadata.reduction_ratio * 100,
                per_frame,
                item_metadata.selector_latency_ms,
            )
    return metadata


def metadata_keep_indices(
    metadata: Sequence[VisualItemMetadata],
) -> Optional[list[torch.Tensor]]:
    """Return sparse-forward indices when at least one source item was pruned."""

    keep_indices = []
    any_pruned = False
    for item in metadata:
        if item.merged_keep_indices is None:
            keep = torch.arange(
                item.original_merged_token_count, dtype=torch.long
            )
        else:
            keep = item.merged_keep_indices
            any_pruned |= item.kept_merged_token_count < item.original_merged_token_count
        keep_indices.append(keep)
    return keep_indices if any_pruned else None


def compress_planner_visual_inputs(
    tokenized_example: dict[str, Any],
    source_keep_indices: Sequence[torch.Tensor],
    vision_start_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Compress source visual-input positions while preserving dense positions.

    ``position_ids`` must already have been computed from the original dense
    grids. The returned mask indexes that original dense sequence.
    """

    if not source_keep_indices:
        return torch.ones_like(tokenized_example["input_ids"], dtype=torch.bool)

    visual_input_mask = tokenized_example["visual_input_token_mask"]
    padded_source_ids = tokenized_example["token_segment_ids"][visual_input_mask]
    source_segment_ids = torch.unique_consecutive(padded_source_ids)
    if len(source_segment_ids) != len(source_keep_indices):
        raise ValueError(
            "PixelPrune metadata/source segment mismatch: "
            f"{len(source_keep_indices)} keep-index tensors for "
            f"{len(source_segment_ids)} visual-input segments"
        )

    sequence_keep_mask = torch.ones_like(
        tokenized_example["input_ids"], dtype=torch.bool
    )
    for segment_id, keep_indices in zip(
        source_segment_ids.tolist(), source_keep_indices
    ):
        keep_indices = torch.as_tensor(keep_indices, dtype=torch.long)
        segment_positions = (
            visual_input_mask
            & (tokenized_example["token_segment_ids"] == segment_id)
        ).nonzero(as_tuple=False).flatten()
        if keep_indices.ndim != 1 or keep_indices.numel() == 0:
            raise ValueError("PixelPrune cannot remove every source visual token")
        if not torch.equal(keep_indices, torch.unique(keep_indices, sorted=True)):
            raise ValueError("source keep indices must be sorted and unique")
        if int(keep_indices[0]) < 0 or int(keep_indices[-1]) >= segment_positions.numel():
            raise ValueError(
                f"invalid source keep indices for visual segment {segment_id}"
            )
        sequence_keep_mask[segment_positions] = False
        sequence_keep_mask[segment_positions[keep_indices]] = True

    dense_output_query_count = int(
        tokenized_example["visual_output_token_mask"].sum()
    )
    sequence_fields = (
        "input_ids",
        "attention_mask",
        "labels",
        "position_ids",
        "visual_input_token_mask",
        "visual_output_token_mask",
        "token_types",
        "token_segment_ids",
        "flex_token_types",
    )
    for field in sequence_fields:
        value = tokenized_example[field]
        if field == "position_ids":
            tokenized_example[field] = value[:, sequence_keep_mask]
        else:
            tokenized_example[field] = value[sequence_keep_mask]

    from .data.utils.attention_utils import build_custom_attention_mask

    tokenized_example["attention_mask_4d"] = build_custom_attention_mask(
        token_type=tokenized_example["token_types"].unsqueeze(0),
        token_segment_ids=tokenized_example["token_segment_ids"].unsqueeze(0),
    )
    if vision_start_token_id is not None:
        tokenized_example["vision_start_indices"] = (
            (tokenized_example["input_ids"] == vision_start_token_id)
            .nonzero(as_tuple=False)
            .flatten()
            .tolist()
        )
    if int(tokenized_example["visual_output_token_mask"].sum()) != dense_output_query_count:
        raise AssertionError("PixelPrune changed the target visual-output query count")
    if int(tokenized_example["visual_input_token_mask"].sum()) != sum(
        int(torch.as_tensor(indices).numel()) for indices in source_keep_indices
    ):
        raise AssertionError(
            "compressed visual-input mask does not match source visual features"
        )
    return sequence_keep_mask
