"""Readable HPSS implementation based on median filtering and soft masks."""
from __future__ import annotations

from scipy.ndimage import median_filter
import numpy as np

from chapter07._common.spectrogram import apply_mask, istft, stft


def median_filter_spectrogram(
    mag: np.ndarray,
    kernel_harmonic: int = 31,
    kernel_percussive: int = 31,
) -> tuple[np.ndarray, np.ndarray]:
    """Return harmonic- and percussive-enhanced magnitude estimates."""
    mag = _validate_magnitude(mag)
    kernel_harmonic = _ensure_odd_kernel(kernel_harmonic)
    kernel_percussive = _ensure_odd_kernel(kernel_percussive)
    harmonic = median_filter(mag, size=(1, kernel_harmonic), mode="reflect")
    percussive = median_filter(mag, size=(kernel_percussive, 1), mode="reflect")
    return harmonic, percussive


def compute_hpss_masks(
    mag: np.ndarray,
    kernel_harmonic: int = 31,
    kernel_percussive: int = 31,
    margin: float | tuple[float, float] = 1.0,
    power: float = 2.0,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute HPSS soft masks from a nonnegative magnitude spectrogram."""
    if power <= 0:
        raise ValueError("power must be positive")
    margin_h, margin_p = _parse_margin(margin)
    harmonic, percussive = median_filter_spectrogram(
        mag,
        kernel_harmonic=kernel_harmonic,
        kernel_percussive=kernel_percussive,
    )

    harmonic_power = np.maximum(harmonic, 0.0) ** power
    percussive_power = np.maximum(percussive, 0.0) ** power
    harmonic_mask = harmonic_power / (
        harmonic_power + (margin_h * percussive_power) + eps
    )
    percussive_mask = percussive_power / (
        (margin_p * harmonic_power) + percussive_power + eps
    )
    return _clip_mask(harmonic_mask), _clip_mask(percussive_mask)


def hpss_separate(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    kernel_harmonic: int = 31,
    kernel_percussive: int = 31,
    margin: float | tuple[float, float] = 1.0,
) -> dict[str, np.ndarray]:
    """Separate a mono mixture into harmonic and percussive estimates."""
    del sr
    audio = _to_mono(audio)
    mixture_spec = stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(mixture_spec)
    harmonic_mask, percussive_mask = compute_hpss_masks(
        mag,
        kernel_harmonic=kernel_harmonic,
        kernel_percussive=kernel_percussive,
        margin=margin,
    )
    return {
        "harmonic": istft(
            apply_mask(mixture_spec, harmonic_mask),
            hop_length=hop_length,
            length=audio.size,
        ).astype(np.float32),
        "percussive": istft(
            apply_mask(mixture_spec, percussive_mask),
            hop_length=hop_length,
            length=audio.size,
        ).astype(np.float32),
    }


def _validate_magnitude(mag: np.ndarray) -> np.ndarray:
    mag = np.asarray(mag, dtype=np.float64)
    if mag.ndim != 2:
        raise ValueError("mag must be a 2-D magnitude spectrogram")
    if np.any(mag < 0):
        raise ValueError("mag must be nonnegative")
    return mag


def _ensure_odd_kernel(size: int) -> int:
    size = int(size)
    if size < 1:
        raise ValueError("kernel size must be positive")
    return size if size % 2 else size + 1


def _parse_margin(margin: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(margin, tuple):
        margin_h, margin_p = margin
    else:
        margin_h = margin_p = margin
    if margin_h < 1.0 or margin_p < 1.0:
        raise ValueError("margin must be at least 1.0")
    return float(margin_h), float(margin_p)


def _clip_mask(mask: np.ndarray) -> np.ndarray:
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def _to_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return np.mean(audio, axis=0)
    raise ValueError("audio must be mono or channel-first stereo")
