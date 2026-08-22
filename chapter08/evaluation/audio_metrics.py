"""Lightweight audio metrics for generation notebooks."""
from __future__ import annotations

import numpy as np

from _common.audio_io import to_mono


def rms(audio: np.ndarray) -> float:
    """Root mean square level."""
    y = to_mono(audio)
    return float(np.sqrt(np.mean(np.square(y))) if y.size else 0.0)


def peak_amplitude(audio: np.ndarray) -> float:
    """Peak absolute amplitude."""
    y = to_mono(audio)
    return float(np.max(np.abs(y)) if y.size else 0.0)


def silence_ratio(audio: np.ndarray, threshold: float = 1e-3) -> float:
    """Fraction of samples whose absolute amplitude stays below ``threshold``."""
    y = to_mono(audio)
    if y.size == 0:
        return 0.0
    return float(np.mean(np.abs(y) < threshold))


def clipping_ratio(audio: np.ndarray, threshold: float = 0.99) -> float:
    """Fraction of samples whose absolute amplitude reaches ``threshold`` (near full scale)."""
    y = to_mono(audio)
    if y.size == 0:
        return 0.0
    return float(np.mean(np.abs(y) >= threshold))


def loudness_lufs(audio: np.ndarray, floor_db: float = -120.0) -> float:
    """Teaching approximation of integrated LUFS from full-window mean square."""
    y = to_mono(audio)
    if y.size == 0:
        return floor_db
    mean_square = float(np.mean(np.square(y)))
    if mean_square <= 0.0:
        return floor_db
    return max(float(-0.691 + 10.0 * np.log10(mean_square)), floor_db)


def zero_crossing_rate(audio: np.ndarray) -> float:
    """Fraction of adjacent samples with a sign change."""
    y = to_mono(audio)
    if y.size < 2:
        return 0.0
    return float(np.mean(np.signbit(y[:-1]) != np.signbit(y[1:])))


def spectral_flatness(audio: np.ndarray, eps: float = 1e-8) -> float:
    """Geometric mean divided by arithmetic mean of magnitude spectrum."""
    y = to_mono(audio)
    if y.size == 0:
        return 0.0
    mag = np.abs(np.fft.rfft(y)) + eps
    return float(np.exp(np.mean(np.log(mag))) / np.mean(mag))


def spectral_centroid(audio: np.ndarray, sr: int, eps: float = 1e-8) -> float:
    """Single-window spectral centroid in Hz."""
    y = to_mono(audio)
    if y.size == 0:
        return 0.0
    mag = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(y.size, d=1.0 / sr)
    return float(np.sum(freqs * mag) / (np.sum(mag) + eps))


def simple_embedding(audio: np.ndarray, sr: int) -> np.ndarray:
    """Small hand-crafted embedding for teaching proxy distances."""
    return np.asarray(
        [
            rms(audio),
            zero_crossing_rate(audio),
            spectral_flatness(audio),
            spectral_centroid(audio, sr) / max(sr / 2.0, 1.0),
        ],
        dtype=np.float64,
    )


def fad_proxy(reference_audio: list[np.ndarray], generated_audio: list[np.ndarray], sr: int) -> float:
    """Teaching proxy for distribution distance; not a formal FAD implementation."""
    if not reference_audio or not generated_audio:
        return float("nan")
    ref = np.vstack([simple_embedding(audio, sr) for audio in reference_audio])
    gen = np.vstack([simple_embedding(audio, sr) for audio in generated_audio])
    mean_distance = np.sum((ref.mean(axis=0) - gen.mean(axis=0)) ** 2)
    std_distance = np.sum((ref.std(axis=0) - gen.std(axis=0)) ** 2)
    return float(mean_distance + std_distance)
