"""PyTorch Dataset：基于CTMP清洗切片（24 kHz、5秒wav）的log-mel谱。

数据流：
  ctmp_loader.load_ctmp_segments(seed, split)
    → soundfile.read(audio_path)
    → 可选波形增强
    → compute_logmel()
    → Tensor (1, n_mels, n_frames)

未增强样本按音频路径惰性缓存log-mel谱，供验证、测试和无增强训练重复使用；
启用波形增强的训练样本仍逐次重新计算，保留每轮随机变换。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .augment import AugmentConfig, apply_spec_augment, apply_waveform_augment

SAMPLE_RATE = 24000
CLIP_SEC = 5.0
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SEC)  # 120000
N_FRAMES = CLIP_SAMPLES // HOP_LENGTH + 1   # 235


def compute_logmel(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop: int = HOP_LENGTH,
) -> np.ndarray:
    """计算log-mel谱。返回shape (n_mels, n_frames) float32。"""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop, power=2.0,
    )
    spec = librosa.power_to_db(mel, ref=1.0, top_db=80.0).astype(np.float32)
    if not np.isfinite(spec).all():
        spec = np.nan_to_num(spec, nan=-80.0, posinf=0.0, neginf=-80.0)
    return spec


@lru_cache(maxsize=4096)
def _cached_logmel(audio_path: str) -> np.ndarray:
    """读取冻结切片并缓存未增强的log-mel谱。"""
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if len(audio) < CLIP_SAMPLES:
        audio = np.pad(audio, (0, CLIP_SAMPLES - len(audio)))
    elif len(audio) > CLIP_SAMPLES:
        audio = audio[:CLIP_SAMPLES]
    return compute_logmel(audio, sr)


class MelSegmentDataset(Dataset):
    """从 CTMP 切片记录列表构建 Dataset。

    每条 record 来自 ctmp_loader.load_ctmp_segments()，含 audio_path 字段。
    """

    def __init__(
        self,
        records: list[dict],
        label_map: dict[str, int],
        mode: str = "train",
        aug: Optional[AugmentConfig] = None,
        seed: int = 42,
    ):
        assert mode in ("train", "eval")
        self.records = records
        self.label_map = label_map
        self.mode = mode
        self.aug = aug or AugmentConfig(enabled=False)
        self._rng_seeds = np.random.default_rng(seed).integers(
            0, 2**31 - 1, size=len(records)
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        use_waveform_aug = self.mode == "train" and self.aug.enabled
        if use_waveform_aug:
            audio, sr = sf.read(rec["audio_path"], dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if len(audio) < CLIP_SAMPLES:
                audio = np.pad(audio, (0, CLIP_SAMPLES - len(audio)))
            elif len(audio) > CLIP_SAMPLES:
                audio = audio[:CLIP_SAMPLES]
            rng = np.random.default_rng(
                int(self._rng_seeds[idx]) + int(torch.randint(0, 2**31 - 1, (1,)).item())
            )
            audio = apply_waveform_augment(audio, SAMPLE_RATE, self.aug, rng)
            spec = compute_logmel(audio, SAMPLE_RATE)
        else:
            rng = None
            spec = _cached_logmel(str(Path(rec["audio_path"]).resolve()))

        # 频谱增强（仅训练）
        if self.mode == "train" and self.aug.enabled and rng is not None:
            spec = apply_spec_augment(spec, self.aug, rng)

        # 固定偏置平移。ref=1.0 时 0 dB 对应功率1，样本峰值不必为0 dB；
        # top_db=80只把每个样本的动态范围截在其峰值以下80 dB内。
        spec = spec - (-40.0)

        x = torch.from_numpy(np.asarray(spec, dtype=np.float32)).unsqueeze(0)
        y = self.label_map[rec["family_label"]]
        return x, y
