"""Generation entry points for the mini Codec-LM."""
from __future__ import annotations

import torch

from codec_lm.model import MiniCodecLM, generate_tokens


def sample_from_model(
    model: MiniCodecLM,
    seed: list[int] | torch.Tensor,
    length: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Sample token ids from a trained mini Codec-LM."""
    seed_tokens = torch.as_tensor(seed, dtype=torch.long)
    return generate_tokens(model, seed_tokens, length, temperature=temperature)

