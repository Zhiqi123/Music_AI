"""硬件检测与 DataLoader 工作进程数估算。"""
from __future__ import annotations

import os

import torch


def get_device() -> torch.device:
    """返回实验设备；可用 ``CHAPTER06_DEVICE`` 显式选择 cpu/mps/cuda。"""
    requested = os.environ.get("CHAPTER06_DEVICE")
    if requested is not None:
        device_type = requested.strip().lower()
        if device_type not in {"cpu", "mps", "cuda"}:
            raise ValueError(
                "CHAPTER06_DEVICE must be one of: cpu, mps, cuda; "
                f"got {requested!r}"
            )
        if device_type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CHAPTER06_DEVICE=cuda, but CUDA is unavailable")
        if device_type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("CHAPTER06_DEVICE=mps, but MPS is unavailable")
        return torch.device(device_type)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_num_workers(reserve: int = 2) -> int:
    """按 CPU 核数自适应，保留 reserve 个核给主进程与系统。"""
    cpu = os.cpu_count() or 1
    return max(1, cpu - reserve)


def auto_batch_size(model_params_m: int = 86, base: int = 8) -> int:
    """根据可用内存/显存推断合理 batch_size。

    Args:
        model_params_m: 模型参数量（百万），保留以备将来按模型大小细化。
        base: 保守 baseline，所有设备都能跑。
    """
    if torch.cuda.is_available():
        avail_gb = torch.cuda.mem_get_info()[0] / 1e9
        if avail_gb >= 24:
            return base * 4
        if avail_gb >= 12:
            return base * 2
        return base

    if torch.backends.mps.is_available():
        try:
            import psutil
        except ImportError:
            return base
        avail_gb = psutil.virtual_memory().available / 1e9 - 8
        if avail_gb >= 32:
            return base * 4
        if avail_gb >= 16:
            return base * 2
        return base

    return base
