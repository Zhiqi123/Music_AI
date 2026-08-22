"""Autoregressive sampling helpers for the toy WaveNet."""
from __future__ import annotations

import numpy as np
import torch

from _common.audio_io import mu_law_decode
from wavenet.model import WaveNet


@torch.no_grad()
def sample_tokens(
    model: WaveNet,
    seed_tokens: torch.Tensor,
    num_new_tokens: int,
    temperature: float = 1.0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Generate new mu-law tokens one step at a time."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    model.eval()
    device = torch.device(device)
    tokens = seed_tokens.to(device).long()
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    for _ in range(num_new_tokens):
        logits = model(tokens)[:, :, -1] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokens


def tokens_to_audio(tokens: torch.Tensor | np.ndarray, quantization_channels: int = 256) -> np.ndarray:
    """Decode generated tokens to float audio."""
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.detach().cpu().numpy()
    return mu_law_decode(np.asarray(tokens).reshape(-1), quantization_channels)

