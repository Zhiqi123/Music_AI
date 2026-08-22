"""Simplified REPET implementation for repeated-background demonstrations."""
from __future__ import annotations

import numpy as np

from chapter07._common.spectrogram import apply_mask, istft, stft


def self_similarity_matrix(mag: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Frame-wise cosine similarity for a magnitude spectrogram."""
    mag = _validate_magnitude(mag)
    columns = mag / (np.linalg.norm(mag, axis=0, keepdims=True) + eps)
    return np.clip(columns.T @ columns, 0.0, 1.0).astype(np.float32)


def estimate_period_from_similarity(
    similarity: np.ndarray,
    min_period_frames: int = 8,
    max_period_frames: int | None = None,
) -> int:
    """Estimate a repeating period from high-scoring similarity diagonals."""
    similarity = np.asarray(similarity, dtype=np.float64)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be a square matrix")
    n_frames = similarity.shape[0]
    if n_frames < 2:
        return 1
    max_period = n_frames // 2 if max_period_frames is None else max_period_frames
    max_period = max(min(int(max_period), n_frames - 1), 1)
    min_period = min(max(int(min_period_frames), 1), max_period)

    scores = [
        float(np.mean(np.diag(similarity, k=lag)))
        for lag in range(min_period, max_period + 1)
    ]
    return int(min_period + int(np.argmax(scores)))


def estimate_repeating_background(
    mag: np.ndarray,
    beat_period_frames: int | None = None,
    min_period_frames: int = 8,
    max_period_frames: int | None = None,
) -> np.ndarray:
    """Estimate repeated background magnitude by median over same-period frames."""
    mag = _validate_magnitude(mag)
    n_frames = mag.shape[1]
    if n_frames == 1:
        return mag.astype(np.float32)
    if beat_period_frames is None:
        similarity = self_similarity_matrix(mag)
        beat_period_frames = estimate_period_from_similarity(
            similarity,
            min_period_frames=min_period_frames,
            max_period_frames=max_period_frames,
        )
    period = max(int(beat_period_frames), 1)

    background = np.zeros_like(mag, dtype=np.float64)
    for frame in range(n_frames):
        indices = np.arange(frame % period, n_frames, period)
        if indices.size < 2:
            indices = np.arange(n_frames)
        background[:, frame] = np.median(mag[:, indices], axis=1)
    return np.minimum(background, mag).astype(np.float32)


def repet_masks(
    mag: np.ndarray,
    beat_period_frames: int | None = None,
    min_period_frames: int = 8,
    max_period_frames: int | None = None,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return background mask, foreground mask, and background magnitude."""
    mag = _validate_magnitude(mag)
    background_mag = estimate_repeating_background(
        mag,
        beat_period_frames=beat_period_frames,
        min_period_frames=min_period_frames,
        max_period_frames=max_period_frames,
    )
    foreground_mag = np.maximum(mag - background_mag, 0.0)
    background_mask = background_mag / (background_mag + foreground_mag + eps)
    foreground_mask = foreground_mag / (background_mag + foreground_mag + eps)
    return (
        np.clip(background_mask, 0.0, 1.0).astype(np.float32),
        np.clip(foreground_mask, 0.0, 1.0).astype(np.float32),
        background_mag.astype(np.float32),
    )


def repet_separate(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    beat_period_frames: int | None = None,
) -> dict[str, np.ndarray]:
    """Separate a mono mixture into repeated background and foreground residual."""
    del sr
    audio = _to_mono(audio)
    mixture_spec = stft(audio, n_fft=n_fft, hop_length=hop_length)
    background_mask, foreground_mask, _ = repet_masks(
        np.abs(mixture_spec),
        beat_period_frames=beat_period_frames,
    )
    return {
        "background": istft(
            apply_mask(mixture_spec, background_mask),
            hop_length=hop_length,
            length=audio.size,
        ).astype(np.float32),
        "foreground": istft(
            apply_mask(mixture_spec, foreground_mask),
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


def _to_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return np.mean(audio, axis=0)
    raise ValueError("audio must be mono or channel-first stereo")
