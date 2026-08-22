"""Tiny PyTorch models used only for architecture shape checks."""
from __future__ import annotations

from collections import OrderedDict


class TinySpectrogramUNet:  # pragma: no cover - replaced when torch is available
    """Placeholder that reports the optional PyTorch dependency clearly."""

    def __init__(self, *args, **kwargs) -> None:
        raise ImportError("TinySpectrogramUNet requires PyTorch")


try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - depends on local environment
    torch = None
else:

    class ConvBlock(nn.Module):
        """Small convolutional block for shape demonstrations."""

        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(1, out_channels),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(1, out_channels),
                nn.GELU(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class TinySpectrogramUNet(nn.Module):
        """A minimal U-Net that predicts a bounded spectrogram mask."""

        def __init__(self, in_channels: int = 2, base_channels: int = 8) -> None:
            super().__init__()
            self.enc1 = ConvBlock(in_channels, base_channels)
            self.enc2 = ConvBlock(base_channels, base_channels * 2)
            self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)
            self.dec2 = ConvBlock(base_channels * 6, base_channels * 2)
            self.dec1 = ConvBlock(base_channels * 3, base_channels)
            self.out = nn.Conv2d(base_channels, in_channels, kernel_size=1)

        def forward(
            self,
            x: torch.Tensor,
            return_shapes: bool = False,
        ) -> torch.Tensor | tuple[torch.Tensor, OrderedDict[str, tuple[int, ...]]]:
            shapes: OrderedDict[str, tuple[int, ...]] = OrderedDict()
            shapes["input"] = tuple(x.shape)

            e1 = self.enc1(x)
            shapes["enc1"] = tuple(e1.shape)
            p1 = F.avg_pool2d(e1, kernel_size=2)
            shapes["pool1"] = tuple(p1.shape)

            e2 = self.enc2(p1)
            shapes["enc2"] = tuple(e2.shape)
            p2 = F.avg_pool2d(e2, kernel_size=2)
            shapes["pool2"] = tuple(p2.shape)

            bottleneck = self.bottleneck(p2)
            shapes["bottleneck"] = tuple(bottleneck.shape)

            u2 = F.interpolate(bottleneck, size=e2.shape[-2:], mode="bilinear", align_corners=False)
            d2 = self.dec2(torch.cat([u2, e2], dim=1))
            shapes["dec2"] = tuple(d2.shape)

            u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
            d1 = self.dec1(torch.cat([u1, e1], dim=1))
            shapes["dec1"] = tuple(d1.shape)

            mask = torch.sigmoid(self.out(d1))
            shapes["mask"] = tuple(mask.shape)
            y = x * mask
            shapes["output"] = tuple(y.shape)
            if return_shapes:
                return y, shapes
            return y
