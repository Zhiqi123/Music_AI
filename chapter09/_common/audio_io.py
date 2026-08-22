"""音频读取,第九章各 Notebook 共用;统一单声道 float32。"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "numba_cache"))

import librosa
import numpy as np
import soundfile as sf


def load_audio(
    path: Path | str,
    sr: int | None = 22050,
    duration: float | None = None,
    offset: float = 0.0,
) -> tuple[np.ndarray, int]:
    """读取音频为单声道 float32,按需重采样与裁剪。"""
    path = Path(path)
    audio, source_sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr is not None and source_sr != sr:
        audio = librosa.resample(audio, orig_sr=source_sr, target_sr=sr).astype(np.float32)
        source_sr = sr
    if offset or duration is not None:
        start = int(offset * source_sr)
        end = start + int(duration * source_sr) if duration is not None else len(audio)
        audio = audio[start:end]
    return audio, source_sr
