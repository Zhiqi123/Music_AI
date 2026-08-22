"""Token datasets for mini Codec-LM experiments."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class RandomTokenDataset(Dataset):
    """Deterministic random token sequences for tests and smoke runs."""

    def __init__(
        self,
        num_examples: int = 128,
        sequence_length: int = 128,
        vocab_size: int = 1024,
        seed: int = 0,
    ) -> None:
        self.num_examples = int(num_examples)
        self.sequence_length = int(sequence_length)
        self.vocab_size = int(vocab_size)
        generator = torch.Generator().manual_seed(seed)
        self.tokens = torch.randint(
            0,
            vocab_size,
            (num_examples, sequence_length + 1),
            generator=generator,
        )

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.tokens[index]
        return row[:-1].clone(), row[1:].clone()


class TokenFileDataset(Dataset):
    """Next-token pairs from cached codec token tensors."""

    def __init__(
        self,
        token_paths: list[Path | str],
        sequence_length: int = 512,
        vocab_size: int | None = 1024,
    ) -> None:
        self.token_paths = [Path(path) for path in token_paths]
        if not self.token_paths:
            raise ValueError("token_paths must not be empty")
        self.sequence_length = int(sequence_length)
        self.vocab_size = vocab_size
        self._tokens = [self._load_and_flatten(path) for path in self.token_paths]

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = fit_token_length(self._tokens[index], self.sequence_length + 1)
        return row[:-1].clone(), row[1:].clone()

    def _load_and_flatten(self, path: Path) -> torch.Tensor:
        tensor = torch.load(path, map_location="cpu")
        tokens = flatten_codec_tokens(tensor).long()
        if self.vocab_size is not None and tokens.numel():
            if int(tokens.min()) < 0 or int(tokens.max()) >= self.vocab_size:
                raise ValueError(f"Token ids in {path} exceed vocab_size={self.vocab_size}")
        return tokens


def flatten_codec_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """Flatten EnCodec-style ``(batch, codebook, time)`` tokens by time."""
    tokens = torch.as_tensor(tokens)
    if tokens.ndim == 1:
        return tokens.long()
    if tokens.ndim == 2:
        return tokens.transpose(0, 1).reshape(-1).long()
    if tokens.ndim == 3:
        return tokens[0].transpose(0, 1).reshape(-1).long()
    raise ValueError("tokens must have 1, 2, or 3 dimensions")


def fit_token_length(tokens: torch.Tensor, length: int) -> torch.Tensor:
    """Trim or repeat tokens to an exact length."""
    if length < 1:
        raise ValueError("length must be positive")
    tokens = torch.as_tensor(tokens).long().reshape(-1)
    if tokens.numel() == 0:
        raise ValueError("tokens must not be empty")
    if tokens.numel() >= length:
        return tokens[:length]
    repeats = (length + tokens.numel() - 1) // tokens.numel()
    return tokens.repeat(repeats)[:length]


def find_token_files(cache_dir: Path | str) -> list[Path]:
    """Return cached token tensors in stable order."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return []
    return sorted(cache_dir.glob("*.tokens.pt"))
