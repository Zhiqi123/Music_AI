"""Codec reconstruction metrics."""
from __future__ import annotations

import numpy as np

from _common.audio_io import to_mono


def reconstruction_mse(reference: np.ndarray, reconstruction: np.ndarray) -> float:
    """Mean squared error after trimming to common length."""
    ref = to_mono(reference)
    rec = to_mono(reconstruction)
    n = min(ref.size, rec.size)
    if n == 0:
        return float("nan")
    return float(np.mean((ref[:n] - rec[:n]) ** 2))


def reconstruction_snr_db(reference: np.ndarray, reconstruction: np.ndarray, eps: float = 1e-8) -> float:
    """Signal-to-noise ratio in dB after trimming to common length."""
    ref = to_mono(reference)
    rec = to_mono(reconstruction)
    n = min(ref.size, rec.size)
    if n == 0:
        return float("nan")
    signal = float(np.sum(ref[:n] ** 2))
    noise = float(np.sum((ref[:n] - rec[:n]) ** 2))
    return float(10.0 * np.log10((signal + eps) / (noise + eps)))

