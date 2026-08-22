"""A minimal causal Transformer over discrete audio-codec tokens."""
from __future__ import annotations

import torch
from torch import nn


class MiniCodecLM(nn.Module):
    """Small causal Transformer for next-token prediction."""

    def __init__(
        self,
        vocab_size: int = 1024,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        max_length: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.max_length = int(max_length)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return logits with shape ``(batch, time, vocab)``."""
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch, time)")
        batch, time = tokens.shape
        if time > self.max_length:
            raise ValueError(f"sequence length {time} exceeds max_length {self.max_length}")
        positions = torch.arange(time, device=tokens.device).unsqueeze(0).expand(batch, time)
        x = self.token_embedding(tokens.long()) + self.position_embedding(positions)
        mask = causal_mask(time, tokens.device)
        x = self.transformer(x, mask=mask)
        return self.output(x)


def causal_mask(length: int, device: torch.device | str) -> torch.Tensor:
    """Return an additive mask for causal self-attention."""
    return torch.full((length, length), float("-inf"), device=device).triu(1)


@torch.no_grad()
def generate_tokens(
    model: MiniCodecLM,
    seed_tokens: torch.Tensor,
    num_new_tokens: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Autoregressively sample token ids."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    model.eval()
    tokens = seed_tokens.long()
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    for _ in range(num_new_tokens):
        context = tokens[:, -model.max_length :]
        logits = model(context)[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokens

