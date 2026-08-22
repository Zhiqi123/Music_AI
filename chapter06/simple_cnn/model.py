"""SimpleAudioCNN：3 个 conv block + GAP + FC，参数量约 100k 量级。

输入：log-mel谱 (B, 1, 64, 235)，64 mel bins × 5秒 × hop=512/sr=24000 ≈ 235帧。
输出：n_classes 类 logits。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SimpleAudioCNN(nn.Module):
    def __init__(self, n_classes: int = 6, in_channels: int = 1, base: int = 32, gn_groups: int = 8):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(gn_groups, c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(gn_groups, c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(gn_groups, c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c3, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)

    @torch.no_grad()
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
