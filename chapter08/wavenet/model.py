"""A compact WaveNet for teaching autoregressive waveform modeling."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1-D convolution that pads only on the left."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class WaveNetResidualBlock(nn.Module):
    """Gated dilated residual block with a skip projection."""

    def __init__(
        self,
        residual_channels: int,
        dilation_channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.filter_conv = CausalConv1d(
            residual_channels, dilation_channels, kernel_size, dilation
        )
        self.gate_conv = CausalConv1d(
            residual_channels, dilation_channels, kernel_size, dilation
        )
        self.residual_proj = nn.Conv1d(dilation_channels, residual_channels, kernel_size=1)
        self.skip_proj = nn.Conv1d(dilation_channels, skip_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gated = torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))
        residual = self.residual_proj(gated)
        skip = self.skip_proj(gated)
        return x + residual, skip


class WaveNet(nn.Module):
    """Small autoregressive WaveNet over mu-law tokens."""

    def __init__(
        self,
        quantization_channels: int = 256,
        residual_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 64,
        kernel_size: int = 2,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    ) -> None:
        super().__init__()
        self.quantization_channels = quantization_channels
        self.dilations = tuple(dilations)
        self.embedding = nn.Embedding(quantization_channels, residual_channels)
        self.blocks = nn.ModuleList(
            [
                WaveNetResidualBlock(
                    residual_channels=residual_channels,
                    dilation_channels=dilation_channels,
                    skip_channels=skip_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
                for dilation in self.dilations
            ]
        )
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, skip_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(skip_channels, quantization_channels, kernel_size=1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return logits with shape ``(batch, vocab, time)``."""
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch, time)")
        x = self.embedding(tokens.long()).transpose(1, 2)
        skip_total: torch.Tensor | None = None
        for block in self.blocks:
            x, skip = block(x)
            skip_total = skip if skip_total is None else skip_total + skip
        if skip_total is None:
            raise RuntimeError("WaveNet must contain at least one residual block")
        return self.output(skip_total)


def build_wavenet_from_config(config: dict) -> WaveNet:
    """Construct ``WaveNet`` from a nested config dictionary."""
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    layers = int(model_cfg.get("layers_per_cycle", 6))
    cycles = int(model_cfg.get("dilation_cycles", 1))
    dilations = tuple(2**i for i in range(layers)) * cycles
    return WaveNet(
        quantization_channels=int(data_cfg.get("quantization_channels", 256)),
        residual_channels=int(model_cfg.get("residual_channels", 32)),
        dilation_channels=int(model_cfg.get("dilation_channels", 32)),
        skip_channels=int(model_cfg.get("skip_channels", 64)),
        kernel_size=int(model_cfg.get("kernel_size", 2)),
        dilations=dilations,
    )

