"""Audio loading, saving, cropping, and mu-law quantization helpers."""
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
    sr: int | None = None,
    mono: bool = True,
    duration: float | None = None,
    start: float = 0.0,
) -> tuple[np.ndarray, int]:
    """Load audio as float32.

    Mono output has shape ``(samples,)``. Multi-channel output is channel-first:
    ``(channels, samples)``.
    """
    path = Path(path)
    audio, source_sr = sf.read(path, always_2d=False, dtype="float32")

    if audio.ndim == 2:
        audio = audio.mean(axis=1) if mono else audio.T

    if start or duration is not None:
        audio = crop_audio(audio, source_sr, start=start, duration=duration)

    if sr is not None and sr != source_sr:
        audio = resample_audio(audio, source_sr, sr)
        source_sr = sr

    return np.asarray(audio, dtype=np.float32), int(source_sr)


def save_audio(path: Path | str, audio: np.ndarray, sr: int) -> None:
    """Save mono or channel-first audio to a soundfile-supported path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = to_soundfile_shape(np.asarray(audio, dtype=np.float32))
    sf.write(path, np.clip(data, -1.0, 1.0), sr)


def normalize_peak(audio: np.ndarray, peak: float = 0.98) -> np.ndarray:
    """Scale audio so the largest absolute sample equals ``peak``."""
    if peak <= 0:
        raise ValueError("peak must be positive")
    audio = np.asarray(audio, dtype=np.float32)
    current = float(np.max(np.abs(audio))) if audio.size else 0.0
    if current == 0.0:
        return audio.copy()
    return (audio * (peak / current)).astype(np.float32)


def crop_audio(
    audio: np.ndarray,
    sr: int,
    start: float = 0.0,
    duration: float | None = None,
) -> np.ndarray:
    """Crop along the last axis, preserving mono or channel-first layout."""
    if start < 0:
        raise ValueError("start must be non-negative")
    audio = np.asarray(audio)
    start_sample = int(round(start * sr))
    if duration is None:
        return audio[..., start_sample:].copy()
    if duration < 0:
        raise ValueError("duration must be non-negative")
    end_sample = start_sample + int(round(duration * sr))
    return audio[..., start_sample:end_sample].copy()


def pad_or_trim(audio: np.ndarray, target_samples: int) -> np.ndarray:
    """Return exactly ``target_samples`` along the last axis."""
    if target_samples < 0:
        raise ValueError("target_samples must be non-negative")
    audio = np.asarray(audio, dtype=np.float32)
    current = audio.shape[-1]
    if current == target_samples:
        return audio.copy()
    if current > target_samples:
        return audio[..., :target_samples].copy()
    pad_width = [(0, 0)] * audio.ndim
    pad_width[-1] = (0, target_samples - current)
    return np.pad(audio, pad_width).astype(np.float32)


def resample_audio(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """Resample mono or channel-first audio."""
    audio = np.asarray(audio, dtype=np.float32)
    if source_sr == target_sr:
        return audio.copy()
    if audio.ndim == 1:
        return librosa.resample(audio, orig_sr=source_sr, target_sr=target_sr).astype(
            np.float32
        )
    return np.vstack(
        [
            librosa.resample(channel, orig_sr=source_sr, target_sr=target_sr)
            for channel in audio
        ]
    ).astype(np.float32)


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert mono or channel-first audio to mono."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=0).astype(np.float32)


def mu_law_encode(
    audio: np.ndarray,
    quantization_channels: int = 256,
) -> np.ndarray:
    """Quantize audio in ``[-1, 1]`` with mu-law companding."""
    if quantization_channels < 2:
        raise ValueError("quantization_channels must be at least 2")
    mu = float(quantization_channels - 1)
    audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    companded = np.sign(audio) * np.log1p(mu * np.abs(audio)) / np.log1p(mu)
    encoded = ((companded + 1.0) * 0.5 * mu + 0.5).astype(np.int64)
    return np.clip(encoded, 0, quantization_channels - 1)


def mu_law_decode(
    encoded: np.ndarray,
    quantization_channels: int = 256,
) -> np.ndarray:
    """Decode mu-law integer tokens back to float audio."""
    if quantization_channels < 2:
        raise ValueError("quantization_channels must be at least 2")
    mu = float(quantization_channels - 1)
    x = 2.0 * np.asarray(encoded, dtype=np.float32) / mu - 1.0
    audio = np.sign(x) * (np.power(1.0 + mu, np.abs(x)) - 1.0) / mu
    return audio.astype(np.float32)


def to_soundfile_shape(audio: np.ndarray) -> np.ndarray:
    """Convert channel-first audio to soundfile's samples-first layout."""
    audio = np.asarray(audio)
    if audio.ndim == 1:
        return audio
    if audio.ndim != 2:
        raise ValueError("audio must be mono or 2-D channel-first data")
    if audio.shape[0] <= 8 and audio.shape[0] <= audio.shape[1]:
        return audio.T
    return audio

