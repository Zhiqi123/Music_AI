"""Device selection helpers."""
from __future__ import annotations


def choose_device(requested: str = "auto") -> str:
    """Return ``cuda``, ``mps``, or ``cpu`` for PyTorch code."""
    if requested != "auto":
        return requested
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe_device(device: str) -> dict[str, str | bool]:
    """Return table-ready device details."""
    try:
        import torch
    except Exception as exc:
        return {"device": device, "torch_available": False, "notes": str(exc)}
    notes = ""
    if device == "cuda" and torch.cuda.is_available():
        notes = torch.cuda.get_device_name(0)
    elif device == "mps":
        notes = "Apple Metal Performance Shaders"
    elif device == "cpu":
        notes = "CPU"
    return {"device": device, "torch_available": True, "notes": notes}

