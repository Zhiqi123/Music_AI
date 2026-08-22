"""Synthetic mixtures used when no external audio is available."""
from __future__ import annotations

import numpy as np

from .audio_io import normalize_peak


def make_synthetic_mixture(
    sr: int = 22050,
    duration: float = 4.0,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Create harmonic, bass, and percussive sources plus their mixture."""
    n = int(round(sr * duration))
    t = np.arange(n) / sr
    harmonic = _melody(t, sr)
    bass = _bass(t)
    percussive = _percussive(n, sr, seed=seed)
    mixture = harmonic + bass + percussive

    scale = 0.92 / max(float(np.max(np.abs(mixture))), 1e-8)
    return {
        "harmonic": (harmonic * scale).astype(np.float32),
        "bass": (bass * scale).astype(np.float32),
        "percussive": (percussive * scale).astype(np.float32),
        "mixture": (mixture * scale).astype(np.float32),
    }


def _melody(t: np.ndarray, sr: int) -> np.ndarray:
    notes = [261.63, 329.63, 392.00, 523.25, 392.00, 329.63, 293.66, 261.63]
    note_len = len(t) // len(notes)
    y = np.zeros_like(t)
    for i, freq in enumerate(notes):
        start = i * note_len
        end = len(t) if i == len(notes) - 1 else (i + 1) * note_len
        local_t = np.arange(end - start) / sr
        tone = sum((1.0 / k) * np.sin(2 * np.pi * freq * k * local_t) for k in range(1, 5))
        y[start:end] = 0.28 * tone * _adsr(end - start, sr)
    return y


def _bass(t: np.ndarray) -> np.ndarray:
    root = np.where((t % 2.0) < 1.0, 65.41, 98.00)
    return 0.25 * np.sin(2 * np.pi * root * t) + 0.08 * np.sin(4 * np.pi * root * t)


def _percussive(n: int, sr: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.float64)
    burst_len = int(0.075 * sr)
    decay = np.exp(-np.linspace(0, 7.0, burst_len))
    for beat in np.arange(0.0, n / sr, 0.5):
        start = int(round(beat * sr))
        end = min(start + burst_len, n)
        size = end - start
        noise = rng.normal(0.0, 1.0, size)
        y[start:end] += 0.35 * noise * decay[:size]
        if start < n:
            y[start] += 1.0
    return normalize_peak(y, peak=0.5)


def _adsr(n: int, sr: int) -> np.ndarray:
    attack = min(int(0.02 * sr), n)
    release = min(int(0.08 * sr), max(n - attack, 0))
    sustain = max(n - attack - release, 0)
    return np.concatenate(
        [
            np.linspace(0.0, 1.0, attack, endpoint=False),
            np.full(sustain, 0.85),
            np.linspace(0.85, 0.0, release, endpoint=True),
        ]
    )[:n]

