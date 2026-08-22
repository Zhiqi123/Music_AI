"""PyTorch Dataset：CTMP 切片 → 32 kHz 波形，供 PANNs CNN14 使用。

数据流：
  ctmp_loader.load_ctmp_segments(seed, split)
    → soundfile.read(audio_path)   # 24 kHz mono
    → librosa.resample → 32 kHz
    → Tensor (samples,)
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

PANNS_SR = 32000
SOURCE_SR = 24000
CLIP_SEC = 5.0
CLIP_SAMPLES_32K = int(PANNS_SR * CLIP_SEC)  # 160000


@lru_cache(maxsize=4096)
def _cached_resampled_waveform(audio_path: str, target_sr: int) -> np.ndarray:
    """读取冻结切片、重采样并缓存定长波形。"""
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    clip_samples = int(target_sr * CLIP_SEC)
    if len(audio) < clip_samples:
        audio = np.pad(audio, (0, clip_samples - len(audio)))
    elif len(audio) > clip_samples:
        audio = audio[:clip_samples]
    return np.asarray(audio, dtype=np.float32)


class PannsSegmentDataset(Dataset):
    """从 CTMP 切片记录构建 Dataset，输出 32 kHz 波形 tensor。"""

    def __init__(
        self,
        records: list[dict],
        label_map: dict[str, int],
        target_sr: int = PANNS_SR,
    ):
        self.records = records
        self.label_map = label_map
        self.target_sr = target_sr
        self._clip_samples = int(target_sr * CLIP_SEC)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        audio = _cached_resampled_waveform(
            str(Path(rec["audio_path"]).resolve()), self.target_sr,
        )
        x = torch.from_numpy(audio)
        y = self.label_map[rec["family_label"]]
        return x, y
