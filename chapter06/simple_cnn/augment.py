"""音频与频谱数据增强（6.2 简单 CNN）。

两层结构：
  - 波形层（读取波形后、log-mel计算前）：高斯加噪、音量抖动、pitch shift、time stretch
  - 频谱层（log-mel之后）：SpecAugment时间/频率掩码

通过 AugmentConfig.enabled 总开关控制；enabled=False 时所有增强都为 no-op，
方便"无增强 vs 全套增强"二元对比。
"""
from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass
class AugmentConfig:
    enabled: bool = False
    # 波形层
    noise_std: float = 0.003             # 高斯白噪声标准差（相对单位幅度）
    gain_db_range: tuple[float, float] = (-3.0, 3.0)
    pitch_semitones: float = 1.0         # ± 半音
    time_stretch_range: tuple[float, float] = (0.95, 1.05)
    aug_prob: float = 0.3                # 每种波形增强独立采样的施用概率
    # 频谱层（SpecAugment）
    n_time_masks: int = 2
    time_mask_max: int = 10              # 帧（hop=512、sr=24000 → ~0.02 秒/帧 × 10）
    n_freq_masks: int = 2
    freq_mask_max: int = 8               # mel bins
    spec_aug_prob: float = 0.4
    # 批次层Mixup：λ ~ Beta(α, α)，α=0表示关闭
    mixup_alpha: float = 0.2


def apply_waveform_augment(
    audio: np.ndarray, sr: int, cfg: AugmentConfig, rng: np.random.Generator,
) -> np.ndarray:
    """对单段单声道波形施加波形层增强（每种独立采样）。"""
    if not cfg.enabled:
        return audio
    out = audio.astype(np.float32, copy=True)
    # 高斯加噪
    if rng.random() < cfg.aug_prob:
        out = out + rng.normal(0.0, cfg.noise_std, size=out.shape).astype(np.float32)
    # 音量抖动（dB 域）
    if rng.random() < cfg.aug_prob:
        gain_db = rng.uniform(*cfg.gain_db_range)
        out = out * float(10.0 ** (gain_db / 20.0))
    # pitch shift
    if rng.random() < cfg.aug_prob:
        n_steps = rng.uniform(-cfg.pitch_semitones, cfg.pitch_semitones)
        out = librosa.effects.pitch_shift(out, sr=sr, n_steps=n_steps)
    # time stretch（拉伸/压缩后裁切或零填充回原长度）
    if rng.random() < cfg.aug_prob:
        rate = rng.uniform(*cfg.time_stretch_range)
        stretched = librosa.effects.time_stretch(out, rate=rate)
        n = len(out)
        if len(stretched) >= n:
            out = stretched[:n]
        else:
            pad = np.zeros(n, dtype=np.float32)
            pad[: len(stretched)] = stretched
            out = pad
    return out


def apply_spec_augment(
    spec: np.ndarray, cfg: AugmentConfig, rng: np.random.Generator,
) -> np.ndarray:
    """对log-mel谱（shape (n_mels, n_frames)）施加SpecAugment时频掩码。"""
    if not cfg.enabled or rng.random() >= cfg.spec_aug_prob:
        return spec
    out = spec.copy()
    n_mels, n_frames = out.shape
    fill = float(out.min())
    for _ in range(cfg.n_time_masks):
        t = int(rng.integers(0, max(1, cfg.time_mask_max + 1)))
        if t == 0 or t >= n_frames:
            continue
        t0 = int(rng.integers(0, n_frames - t + 1))
        out[:, t0 : t0 + t] = fill
    for _ in range(cfg.n_freq_masks):
        f = int(rng.integers(0, max(1, cfg.freq_mask_max + 1)))
        if f == 0 or f >= n_mels:
            continue
        f0 = int(rng.integers(0, n_mels - f + 1))
        out[f0 : f0 + f, :] = fill
    return out
