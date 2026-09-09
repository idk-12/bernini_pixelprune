"""Device helpers for CUDA/NPU/CPU inference."""

from contextlib import nullcontext

import torch


def _ensure_npu():
    if hasattr(torch, "npu"):
        return True
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return False
    return hasattr(torch, "npu")


def npu_is_available() -> bool:
    return _ensure_npu() and torch.npu.is_available()


def get_inference_device(local_rank: int = 0) -> torch.device:
    if npu_is_available():
        return torch.device(f"npu:{local_rank}")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def set_device(device) -> None:
    device = torch.device(device)
    if device.type == "npu":
        _ensure_npu()
        torch.npu.set_device(device)
    elif device.type == "cuda":
        torch.cuda.set_device(device)


def distributed_backend(device) -> str:
    device = torch.device(device)
    if device.type == "npu":
        return "hccl"
    if device.type == "cuda":
        return "cuda:nccl,cpu:gloo"
    return "gloo"


def object_collective_device(device) -> torch.device:
    device = torch.device(device)
    if device.type in ("cuda", "npu"):
        return device
    return torch.device("cpu")


def accelerator_module(device):
    device = torch.device(device)
    if device.type == "npu" and _ensure_npu():
        return torch.npu
    if device.type == "cuda":
        return torch.cuda
    return None


def empty_cache(device=None) -> None:
    module = accelerator_module(device or ("npu" if npu_is_available() else "cuda"))
    if module is not None and module.is_available():
        module.empty_cache()


def synchronize(device=None) -> None:
    if device is None:
        device = "npu" if npu_is_available() else "cuda"
    device = torch.device(device)
    module = accelerator_module(device)
    if module is not None and module.is_available():
        try:
            module.synchronize(device)
        except TypeError:
            module.synchronize()


def manual_seed_all(seed: int) -> None:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if npu_is_available():
        torch.npu.manual_seed_all(seed)


def autocast(device, dtype):
    device = torch.device(device)
    if dtype == torch.float32:
        return nullcontext()
    if device.type in ("cuda", "npu"):
        return torch.amp.autocast(device.type, dtype=dtype)
    return nullcontext()


def reset_peak_memory_stats(device) -> None:
    module = accelerator_module(device)
    if module is not None and module.is_available():
        try:
            module.reset_peak_memory_stats(device)
        except TypeError:
            module.reset_peak_memory_stats()


def max_memory_allocated(device) -> int:
    module = accelerator_module(device)
    if module is None or not module.is_available():
        return 0
    try:
        return int(module.max_memory_allocated(device))
    except TypeError:
        return int(module.max_memory_allocated())


def max_memory_reserved(device) -> int:
    module = accelerator_module(device)
    if module is None or not module.is_available():
        return 0
    try:
        return int(module.max_memory_reserved(device))
    except TypeError:
        return int(module.max_memory_reserved())
