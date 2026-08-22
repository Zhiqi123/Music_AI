"""STFT, phase, and mask utilities for source separation examples."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "numba_cache"))

import librosa
import numpy as np


def stft(
    audio: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    window: str = "hann",
    center: bool = True,
) -> np.ndarray:
    """Compute a complex STFT. Time is expected on the last axis."""
    return librosa.stft(
        np.asarray(audio),
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=center,
    )


def istft(
    spec: np.ndarray,
    hop_length: int = 512,
    length: int | None = None,
    window: str = "hann",
    center: bool = True,
) -> np.ndarray:
    """Invert a complex STFT."""
    return librosa.istft(
        spec,
        hop_length=hop_length,
        window=window,
        center=center,
        length=length,
    )


def magphase(spec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return magnitude and unit-magnitude phase."""
    magnitude = np.abs(spec)
    phase = np.exp(1j * np.angle(spec))
    return magnitude, phase


def apply_mask(mixture_spec: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a real-valued mask to a complex mixture spectrogram."""
    return np.asarray(mask) * np.asarray(mixture_spec)


def ideal_ratio_mask(
    target_mag: np.ndarray,
    source_mags: list[np.ndarray],
    eps: float = 1e-8,
) -> np.ndarray:
    """Ideal ratio mask for one target magnitude against all source magnitudes."""
    denominator = _sum_magnitudes(source_mags)
    mask = np.asarray(target_mag, dtype=np.float64) / np.maximum(denominator, eps)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def ideal_binary_mask(target_mag: np.ndarray, source_mags: list[np.ndarray]) -> np.ndarray:
    """Ideal binary mask: 1 where target magnitude is the strongest source."""
    stacked = np.stack([np.asarray(mag) for mag in source_mags], axis=0)
    target_mag = np.asarray(target_mag)
    if target_mag.shape != stacked.shape[1:]:
        raise ValueError("target_mag shape must match source magnitudes")
    return (target_mag >= np.max(stacked, axis=0)).astype(np.float32)


def reconstruct_with_mixture_phase(
    magnitude: np.ndarray,
    mixture_spec: np.ndarray,
    hop_length: int = 512,
    length: int | None = None,
) -> np.ndarray:
    """Invert an estimated magnitude by reusing the mixture phase."""
    _, phase = magphase(mixture_spec)
    return istft(np.asarray(magnitude) * phase, hop_length=hop_length, length=length)


def amplitude_to_db(magnitude: np.ndarray, ref: float | None = None) -> np.ndarray:
    """Convert magnitude to dB for plotting."""
    if ref is None:
        ref_value = float(np.max(magnitude)) if np.size(magnitude) else 1.0
    else:
        ref_value = ref
    return librosa.amplitude_to_db(np.asarray(magnitude), ref=max(ref_value, 1e-12))


def _sum_magnitudes(source_mags: list[np.ndarray]) -> np.ndarray:
    if not source_mags:
        raise ValueError("source_mags must not be empty")
    stacked = np.stack([np.asarray(mag, dtype=np.float64) for mag in source_mags], axis=0)
    return np.sum(stacked, axis=0)
