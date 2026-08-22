"""Small checkpoint helpers for Chapter 8 training scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def save_torch_checkpoint(path: Path | str, payload: dict[str, Any]) -> None:
    """Save a PyTorch checkpoint after creating parent directories."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_torch_checkpoint(path: Path | str, map_location: str = "cpu") -> dict[str, Any]:
    """Load a PyTorch checkpoint into a dictionary."""
    import torch

    payload = torch.load(Path(path), map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint is not a dictionary: {path}")
    return payload

