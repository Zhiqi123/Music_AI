"""Objective and diagnostic metrics for source separation."""
from __future__ import annotations

import math
import warnings

import numpy as np


def scale_invariant_projection(
    reference: np.ndarray,
    estimate: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Project ``estimate`` onto ``reference``."""
    reference, estimate = _align_vectors(reference, estimate)
    ref_energy = float(np.dot(reference, reference))
    if ref_energy <= eps:
        return np.full_like(reference, np.nan, dtype=np.float64)
    scale = float(np.dot(estimate, reference)) / ref_energy
    return scale * reference


def si_sdr(
    reference: np.ndarray,
    estimate: np.ndarray,
    eps: float = 1e-8,
    zero_mean: bool = True,
) -> float:
    """Scale-invariant signal-to-distortion ratio in dB."""
    reference, estimate = _align_vectors(reference, estimate)
    if zero_mean:
        reference = reference - np.mean(reference)
        estimate = estimate - np.mean(estimate)

    target = scale_invariant_projection(reference, estimate, eps=eps)
    if np.isnan(target).any():
        return float("nan")

    residual = estimate - target
    target_energy = float(np.dot(target, target))
    residual_energy = float(np.dot(residual, residual))
    if target_energy <= eps:
        return float("nan")
    if residual_energy <= eps:
        return float("inf")
    return 10.0 * math.log10(target_energy / residual_energy)


def bss_eval_sdr(
    reference: np.ndarray,
    estimate: np.ndarray,
) -> float:
    """BSS Eval SDR in dB, via :mod:`mir_eval`.

    Unlike :func:`si_sdr`, which only corrects a global scale, BSS Eval
    matches the estimate to the reference with an optimal time-invariant
    linear distortion filter (length 512, Vincent 2006) before measuring the
    residual error. This is the SDR reported by ``museval`` (BSSEval v4) on
    MUSDB18 and used as the ranking metric of the MDX / SDX challenges.
    """
    import mir_eval

    reference, estimate = _align_vectors(reference, estimate)
    ref = np.asarray(reference, dtype=np.float64)[None, :]
    est = np.asarray(estimate, dtype=np.float64)[None, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        sdr, _isr, _sir, _sar = mir_eval.separation.bss_eval_sources(
            ref, est, compute_permutation=False
        )
    return float(sdr[0])


def energy_ratio(
    audio: np.ndarray,
    reference: np.ndarray | None = None,
    eps: float = 1e-12,
) -> float:
    """Return signal energy, or energy divided by reference energy."""
    audio_vec = _as_vector(audio)
    if reference is None:
        return float(np.dot(audio_vec, audio_vec))
    audio_vec, ref_vec = _align_vectors(audio, reference)
    return float(np.dot(audio_vec, audio_vec) / (np.dot(ref_vec, ref_vec) + eps))


def rms_db(audio: np.ndarray, ref: float = 1.0, eps: float = 1e-12) -> float:
    """Root-mean-square level in dBFS-like units."""
    audio_vec = _as_vector(audio)
    if audio_vec.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(audio_vec**2)))
    return 20.0 * math.log10((rms + eps) / ref)


def reconstruction_error(
    mixture: np.ndarray,
    estimates: dict[str, np.ndarray],
    eps: float = 1e-12,
) -> float:
    """Relative L2 error between mixture and the sum of estimated stems."""
    if not estimates:
        raise ValueError("estimates must not be empty")

    mixture = np.asarray(mixture, dtype=np.float64)
    n = min([mixture.shape[-1], *[np.asarray(x).shape[-1] for x in estimates.values()]])
    mixture = mixture[..., :n]
    total = np.zeros_like(mixture, dtype=np.float64)

    for name, estimate in estimates.items():
        estimate = np.asarray(estimate, dtype=np.float64)[..., :n]
        if estimate.shape != mixture.shape:
            raise ValueError(f"estimate shape for {name!r} does not match mixture")
        total += estimate

    return float(np.linalg.norm((mixture - total).ravel()) / (np.linalg.norm(mixture.ravel()) + eps))


def absent_stem_energy(
    estimates: dict[str, np.ndarray],
    absent_stems: list[str],
    mixture: np.ndarray,
) -> dict[str, float]:
    """Energy ratio for stems confirmed absent by the case manifest."""
    return {
        stem: energy_ratio(estimates[stem], mixture)
        for stem in absent_stems
        if stem in estimates
    }


def _align_vectors(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a_vec = _as_vector(a)
    b_vec = _as_vector(b)
    n = min(a_vec.size, b_vec.size)
    if n == 0:
        raise ValueError("signals must not be empty")
    return a_vec[:n], b_vec[:n]


def _as_vector(audio: np.ndarray) -> np.ndarray:
    return np.asarray(audio, dtype=np.float64).reshape(-1)

