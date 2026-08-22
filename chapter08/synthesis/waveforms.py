"""Synthetic waveform generators for audio-generation demos."""
from __future__ import annotations

import numpy as np

from _common.audio_io import normalize_peak


def sine_wave(
    frequency: float = 440.0,
    duration: float = 1.0,
    sr: int = 16000,
    phase: float = 0.0,
    amplitude: float = 0.8,
) -> np.ndarray:
    """Generate a sine wave."""
    t = np.arange(int(round(duration * sr)), dtype=np.float32) / sr
    return (amplitude * np.sin(2.0 * np.pi * frequency * t + phase)).astype(np.float32)


def harmonic_stack(
    fundamental: float = 220.0,
    duration: float = 1.0,
    sr: int = 16000,
    harmonics: int = 6,
    decay: float = 1.0,
) -> np.ndarray:
    """Generate a harmonic tone with decreasing overtone amplitudes."""
    if harmonics < 1:
        raise ValueError("harmonics must be positive")
    t = np.arange(int(round(duration * sr)), dtype=np.float32) / sr
    audio = np.zeros_like(t)
    for harmonic in range(1, harmonics + 1):
        weight = 1.0 / (harmonic**decay)
        audio += weight * np.sin(2.0 * np.pi * fundamental * harmonic * t)
    return normalize_peak(audio, peak=0.8)


def linear_chirp(
    start_hz: float = 120.0,
    end_hz: float = 1200.0,
    duration: float = 1.0,
    sr: int = 16000,
) -> np.ndarray:
    """Generate a linear frequency sweep."""
    t = np.arange(int(round(duration * sr)), dtype=np.float32) / sr
    rate = (end_hz - start_hz) / max(duration, 1e-6)
    phase = 2.0 * np.pi * (start_hz * t + 0.5 * rate * t * t)
    return (0.75 * np.sin(phase)).astype(np.float32)


def noise_burst(
    duration: float = 1.0,
    sr: int = 16000,
    seed: int = 0,
    decay_seconds: float = 0.25,
) -> np.ndarray:
    """Generate a decaying noise burst."""
    rng = np.random.default_rng(seed)
    n = int(round(duration * sr))
    t = np.arange(n, dtype=np.float32) / sr
    envelope = np.exp(-t / max(decay_seconds, 1e-6))
    audio = rng.normal(0.0, 1.0, n).astype(np.float32) * envelope
    return normalize_peak(audio, peak=0.8)


def adsr_envelope(
    num_samples: int,
    sr: int = 16000,
    attack: float = 0.02,
    decay: float = 0.08,
    sustain_level: float = 0.6,
    release: float = 0.12,
) -> np.ndarray:
    """Create a simple ADSR envelope."""
    if num_samples < 0:
        raise ValueError("num_samples must be non-negative")
    attack_n = min(num_samples, int(round(attack * sr)))
    decay_n = min(max(num_samples - attack_n, 0), int(round(decay * sr)))
    release_n = min(max(num_samples - attack_n - decay_n, 0), int(round(release * sr)))
    sustain_n = max(num_samples - attack_n - decay_n - release_n, 0)

    parts = []
    if attack_n:
        parts.append(np.linspace(0.0, 1.0, attack_n, endpoint=False))
    if decay_n:
        parts.append(np.linspace(1.0, sustain_level, decay_n, endpoint=False))
    if sustain_n:
        parts.append(np.full(sustain_n, sustain_level))
    if release_n:
        parts.append(np.linspace(sustain_level, 0.0, release_n, endpoint=True))
    if not parts:
        return np.zeros(num_samples, dtype=np.float32)
    envelope = np.concatenate(parts)
    return envelope[:num_samples].astype(np.float32)


def random_tone(
    duration: float = 1.0,
    sr: int = 16000,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a deterministic random tone for toy datasets."""
    rng = np.random.default_rng(seed)
    fundamental = float(rng.uniform(80.0, 880.0))
    harmonics = int(rng.integers(1, 8))
    decay = float(rng.uniform(0.7, 1.4))
    audio = harmonic_stack(fundamental, duration, sr, harmonics=harmonics, decay=decay)
    envelope = adsr_envelope(audio.size, sr=sr)
    return normalize_peak(audio * envelope, peak=0.8)


def waveform_gallery(duration: float = 1.0, sr: int = 16000) -> dict[str, np.ndarray]:
    """Return a small set of contrasting synthetic signals."""
    return {
        "sine": sine_wave(440.0, duration, sr),
        "harmonic": harmonic_stack(220.0, duration, sr),
        "chirp": linear_chirp(120.0, 1600.0, duration, sr),
        "noise_burst": noise_burst(duration, sr),
    }

