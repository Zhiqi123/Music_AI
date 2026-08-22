"""PANNs CNN14 架构及迁移学习封装。

复现 Qiuqiang Kong 等人的 CNN14（AudioSet pretrained），接受 32 kHz 原始波形，
内部完成log-mel谱提取与SpecAugmentation，输出2048维embedding和527类概率。

参考：https://github.com/qiuqiangkong/audioset_tagging_cnn
权重来源：Zenodo record 3987831 (Cnn14_mAP=0.431.pth)
"""
from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import LogmelFilterBank, Spectrogram

_CHECKPOINT_URL = (
    "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth"
)
_CHECKPOINT_SIZE = 327_428_481
_DEFAULT_CKPT_DIR = Path(__file__).parent / "outputs" / "checkpoints"


# ---------------------------------------------------------------------------
# Initialization helpers
# ---------------------------------------------------------------------------

def init_layer(layer: nn.Module) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias.data.fill_(0.0)


def init_bn(bn: nn.Module) -> None:
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


# ---------------------------------------------------------------------------
# Download utility
# ---------------------------------------------------------------------------

def download_cnn14_weights(output_dir: Path | str | None = None) -> Path:
    """Download CNN14 pretrained weights from Zenodo if not already present.

    若自动下载失败（SSL/网络问题），请手动下载后放到返回路径：
        https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth
    """
    output_dir = Path(output_dir) if output_dir else _DEFAULT_CKPT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "Cnn14_mAP=0.431.pth"
    if ckpt_path.exists():
        actual = ckpt_path.stat().st_size
        if actual == _CHECKPOINT_SIZE:
            return ckpt_path
        invalid = ckpt_path.with_suffix(ckpt_path.suffix + ".invalid")
        index = 1
        while invalid.exists():
            invalid = ckpt_path.with_suffix(ckpt_path.suffix + f".invalid{index}")
            index += 1
        ckpt_path.replace(invalid)
        print(
            f"本地CNN14 checkpoint大小不符（{actual}，预期{_CHECKPOINT_SIZE}）；"
            f"原文件已移至{invalid.name}，随后下载新副本"
        )

    import ssl
    print(f"Downloading CNN14 checkpoint to {ckpt_path} (~300MB) ...")
    tmp_path = ckpt_path.with_suffix(ckpt_path.suffix + ".part")
    tmp_path.unlink(missing_ok=True)
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(_CHECKPOINT_URL)
        with urllib.request.urlopen(req, context=ctx) as resp, open(tmp_path, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB", end="", flush=True)
            print()
    except (ssl.SSLError, urllib.error.URLError) as e:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            "无法通过经过验证的 TLS 连接下载 CNN14 权重。"
            f"请从官方 Zenodo 记录手动下载后放到：{ckpt_path}\n"
            f"下载地址：{_CHECKPOINT_URL}"
        ) from e

    actual = tmp_path.stat().st_size if tmp_path.exists() else -1
    if actual != _CHECKPOINT_SIZE:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"下载文件大小不符（{actual}，预期{_CHECKPOINT_SIZE}）。"
            f"请手动下载权重文件并放到：{ckpt_path}\n"
            f"下载地址：{_CHECKPOINT_URL}"
        )
    tmp_path.replace(ckpt_path)
    print(f"Done: {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.1f} MB)")
    return ckpt_path


# ---------------------------------------------------------------------------
# ConvBlock
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 pool_size: tuple[int, int] = (2, 2),
                 pool_type: str = "avg"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool_size = pool_size
        self.pool_type = pool_type
        self._init_weights()

    def _init_weights(self) -> None:
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if self.pool_type == "max":
            x = F.max_pool2d(x, self.pool_size)
        elif self.pool_type == "avg":
            x = F.avg_pool2d(x, self.pool_size)
        else:  # avg+max
            x = F.avg_pool2d(x, self.pool_size) + F.max_pool2d(x, self.pool_size)
        return x


# ---------------------------------------------------------------------------
# CNN14
# ---------------------------------------------------------------------------

class Cnn14(nn.Module):
    """PANNs CNN14：32 kHz波形 → 527类AudioSet概率与2048维embedding。"""

    def __init__(self, sample_rate: int = 32000, n_mels: int = 64,
                 fmin: int = 50, fmax: int = 14000, classes_num: int = 527):
        super().__init__()

        # --- Front-end: waveform → log-mel spectrogram ---
        window_size = 1024
        hop_size = 320
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size, hop_length=hop_size, win_length=window_size,
            window="hann", center=True, pad_mode="reflect", freeze_parameters=True,
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate, n_fft=window_size, n_mels=n_mels,
            fmin=fmin, fmax=fmax, ref=1.0, amin=1e-10, top_db=None,
            freeze_parameters=True,
        )
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2,
        )
        self.bn0 = nn.BatchNorm2d(n_mels)
        init_bn(self.bn0)

        # --- Backbone: 6 ConvBlocks ---
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        # 官方 CNN14 在第六个卷积块后直接做全局池化；用 (1, 1)
        # 保留 ConvBlock 的统一接口，但不再下采样时频尺寸。
        self.conv_block6 = ConvBlock(1024, 2048, pool_size=(1, 1))

        # --- Head ---
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """从32 kHz波形计算2048维embedding。"""
        # Front-end
        x = self.spectrogram_extractor(waveform)  # (B, 1, T, n_fft//2+1)
        x = self.logmel_extractor(x)              # (B, 1, T, n_mels)

        x = x.transpose(1, 3)                     # (B, n_mels, T, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)                     # (B, 1, T, n_mels)

        if self.training:
            x = self.spec_augmenter(x)

        # Backbone
        x = self.conv_block1(x)   # (B, 64, T/2, n_mels/2)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # 频率维取均值；时间维最大值与均值相加，形成全局汇聚向量。
        x = torch.mean(x, dim=3)          # (B, 2048, T')
        (x1, _) = torch.max(x, dim=2)    # (B, 2048)
        x2 = torch.mean(x, dim=2)        # (B, 2048)
        x = x1 + x2                       # (B, 2048)

        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        return F.dropout(x, p=0.5, training=self.training)

    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            waveform: (B, samples) raw audio at 32 kHz.
        Returns:
            dict with 'clipwise_output', 'embedding'.
        """
        embedding = self.forward_features(waveform)
        clipwise_output = torch.sigmoid(self.fc_audioset(embedding))

        return {"clipwise_output": clipwise_output, "embedding": embedding}


# ---------------------------------------------------------------------------
# TransferCnn14
# ---------------------------------------------------------------------------

class TransferCnn14(nn.Module):
    """CNN14 backbone + custom classification head for downstream tasks."""

    def __init__(self, n_classes: int, freeze_base: bool = False,
                 sample_rate: int = 32000):
        super().__init__()
        self.base = Cnn14(sample_rate=sample_rate)
        self.fc_transfer = nn.Linear(2048, n_classes, bias=True)
        init_layer(self.fc_transfer)

        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

    def load_from_pretrain(self, checkpoint_path: str | Path) -> None:
        """把AudioSet预训练CNN14权重载入骨干网络。"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.base.load_state_dict(checkpoint["model"])

    def train(self, mode: bool = True):
        # backbone 完全冻结时，强制 base 进入 eval 模式：
        # 仅 requires_grad=False 不够，BatchNorm 的 running_mean/var 仍会被
        # 下游数据更新；Feature Extraction 在本节定义为固定这些预训练统计量。
        super().train(mode)
        if mode and not any(p.requires_grad for p in self.base.parameters()):
            self.base.eval()
        return self

    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            waveform: (B, samples) raw audio at 32 kHz.
        Returns:
            `clipwise_output`沿用现有接口名，但这里保存未经softmax的下游logits；
            `embedding`为2048维骨干输出。
        """
        embedding = self.base.forward_features(waveform)
        logits = self.fc_transfer(embedding)
        return {"clipwise_output": logits, "embedding": embedding}

    @torch.no_grad()
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
