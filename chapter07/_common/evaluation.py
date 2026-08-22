"""Evaluation helpers for Chapter 7 source-separation notebooks."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from .audio_io import load_audio
from .metrics import bss_eval_sdr, rms_db, si_sdr
from .spectrogram import apply_mask, ideal_ratio_mask, istft, stft

MUSDB_4STEMS = ("vocals", "drums", "bass", "other")
MUSDB_ACCOMPANIMENT_STEMS = ("drums", "bass", "other")
MUSDB_OTHER_EXTENSION_STEMS = ("other", "guitar", "piano")

# SDR (BSS Eval) is the field's standard metric and is listed first; SI-SDR is
# kept as a scale-invariant companion for comparison.
METRIC_BASE_FIELDS = ["track", "model", "stem", "sdr", "si_sdr"]
MetricValue = str | float | int


def metric_rows(
    track_id: str,
    model_name: str,
    references: dict[str, np.ndarray],
    estimates: dict[str, np.ndarray],
    extra: dict[str, MetricValue] | None = None,
    estimate_sources: dict[str, tuple[str, ...] | list[str] | str] | None = None,
) -> list[dict[str, MetricValue]]:
    """Build stem-wise metric rows (SDR + SI-SDR) for references that have estimates."""
    rows: list[dict[str, MetricValue]] = []
    extra = extra or {}
    estimate_sources = estimate_sources or {}
    for stem, reference in references.items():
        if stem == "mixture" or stem not in estimates:
            continue
        estimate = estimates[stem]
        row: dict[str, MetricValue] = {
            "track": track_id,
            "model": model_name,
            "stem": stem,
            "sdr": bss_eval_sdr(reference, estimate),
            "si_sdr": si_sdr(reference, estimate),
        }
        row.update(extra)
        if stem in estimate_sources:
            row["estimate_stems"] = _format_estimate_sources(estimate_sources[stem])
        rows.append(row)
    return rows


# Backward-compatible alias retained for callers/tests that predate the SDR
# addition; the rows it returns now carry both ``sdr`` and ``si_sdr``.
sisdr_rows = metric_rows


def mixture_as_estimates(
    mixture: np.ndarray,
    stems: tuple[str, ...] = ("vocals", "drums", "bass", "other"),
) -> dict[str, np.ndarray]:
    """Use the mixture itself as a simple lower-bound baseline for every stem."""
    return {stem: np.asarray(mixture).copy() for stem in stems}


def oracle_irm_estimates(
    mixture: np.ndarray,
    references: dict[str, np.ndarray],
    stems: tuple[str, ...] = ("vocals", "drums", "bass", "other"),
    n_fft: int = 2048,
    hop_length: int = 512,
) -> dict[str, np.ndarray]:
    """Estimate stems with ideal ratio masks and mixture phase."""
    mixture_mono = _to_mono(mixture)
    refs = {stem: _to_mono(references[stem]) for stem in stems if stem in references}
    if not refs:
        return {}

    n = min([mixture_mono.size, *[audio.size for audio in refs.values()]])
    mixture_mono = mixture_mono[:n]
    refs = {stem: audio[:n] for stem, audio in refs.items()}

    mixture_spec = stft(mixture_mono, n_fft=n_fft, hop_length=hop_length)
    ref_specs = {
        stem: stft(audio, n_fft=n_fft, hop_length=hop_length)
        for stem, audio in refs.items()
    }
    ref_mags = {stem: np.abs(spec) for stem, spec in ref_specs.items()}
    mag_list = list(ref_mags.values())

    estimates: dict[str, np.ndarray] = {}
    for stem, mag in ref_mags.items():
        mask = ideal_ratio_mask(mag, mag_list)
        estimates[stem] = istft(
            apply_mask(mixture_spec, mask),
            hop_length=hop_length,
            length=n,
        ).astype(np.float32)
    return estimates


def musdb_reference_tasks(
    references: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    """Return reference tasks available from MUSDB-style stems."""
    tasks: dict[str, dict[str, np.ndarray]] = {}
    if all(stem in references for stem in MUSDB_4STEMS):
        tasks["musdb_4stem"] = {stem: references[stem] for stem in MUSDB_4STEMS}
    if "vocals" in references and all(stem in references for stem in MUSDB_ACCOMPANIMENT_STEMS):
        accompaniment = sum_stems(references, MUSDB_ACCOMPANIMENT_STEMS)
        if accompaniment is not None:
            tasks["vocals_accompaniment_2stem"] = {
                "vocals": references["vocals"],
                "accompaniment": accompaniment,
            }
    return tasks


def map_estimates_to_musdb_tasks(
    estimates: dict[str, np.ndarray],
) -> dict[str, tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]]]]:
    """Map model-native stems to MUSDB 4-stem and 2-stem evaluation tasks."""
    mapped: dict[str, tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]]]] = {}

    four_stem_estimates: dict[str, np.ndarray] = {}
    four_stem_sources: dict[str, tuple[str, ...]] = {}
    for stem in ("vocals", "drums", "bass"):
        if stem in estimates:
            four_stem_estimates[stem] = estimates[stem]
            four_stem_sources[stem] = (stem,)
    other = sum_stems(estimates, MUSDB_OTHER_EXTENSION_STEMS)
    other_sources = tuple(stem for stem in MUSDB_OTHER_EXTENSION_STEMS if stem in estimates)
    if other is not None:
        four_stem_estimates["other"] = other
        four_stem_sources["other"] = other_sources
    if any(stem in four_stem_estimates for stem in ("drums", "bass", "other")):
        mapped["musdb_4stem"] = (four_stem_estimates, four_stem_sources)

    two_stem_estimates: dict[str, np.ndarray] = {}
    two_stem_sources: dict[str, tuple[str, ...]] = {}
    if "vocals" in estimates:
        two_stem_estimates["vocals"] = estimates["vocals"]
        two_stem_sources["vocals"] = ("vocals",)
    if "accompaniment" in estimates:
        two_stem_estimates["accompaniment"] = estimates["accompaniment"]
        two_stem_sources["accompaniment"] = ("accompaniment",)
    else:
        accompaniment = sum_stems(
            estimates,
            ("drums", "bass", "other", "guitar", "piano"),
        )
        accompaniment_sources = tuple(
            stem
            for stem in ("drums", "bass", "other", "guitar", "piano")
            if stem in estimates
        )
        if accompaniment is not None:
            two_stem_estimates["accompaniment"] = accompaniment
            two_stem_sources["accompaniment"] = accompaniment_sources
    if two_stem_estimates:
        mapped["vocals_accompaniment_2stem"] = (two_stem_estimates, two_stem_sources)

    return mapped


def sum_stems(
    audio_by_stem: dict[str, np.ndarray],
    stems: tuple[str, ...] | list[str],
) -> np.ndarray | None:
    """Sum available stems after trimming all arrays to their common length."""
    selected = [np.asarray(audio_by_stem[stem], dtype=np.float32) for stem in stems if stem in audio_by_stem]
    if not selected:
        return None
    length = min(audio.shape[-1] for audio in selected)
    if length == 0:
        return np.zeros(0, dtype=np.float32)
    total = np.zeros_like(selected[0][..., :length], dtype=np.float32)
    for audio in selected:
        trimmed = audio[..., :length]
        if trimmed.shape != total.shape:
            trimmed = _to_mono(trimmed)
            if total.ndim != 1:
                total = _to_mono(total)
        total = total + trimmed
    return total.astype(np.float32)


def write_metric_rows(path: Path, rows: list[dict[str, MetricValue]]) -> None:
    """Write metric rows to CSV with a stable schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _metric_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean_metric_by_model_stem(
    rows: list[dict[str, MetricValue]],
    metric: str,
) -> dict[tuple[str, str], float]:
    """Aggregate one metric column by ``(model, stem)``, taking the mean."""
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if metric not in row:
            continue
        value = float(row[metric])
        if not np.isfinite(value):
            continue
        key = (str(row["model"]), str(row["stem"]))
        grouped.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def mean_sisdr_by_model_stem(
    rows: list[dict[str, MetricValue]],
) -> dict[tuple[str, str], float]:
    """Aggregate SI-SDR rows by ``(model, stem)``."""
    return _mean_metric_by_model_stem(rows, "si_sdr")


def mean_sdr_by_model_stem(
    rows: list[dict[str, MetricValue]],
) -> dict[tuple[str, str], float]:
    """Aggregate BSS Eval SDR rows by ``(model, stem)``."""
    return _mean_metric_by_model_stem(rows, "sdr")


def _to_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return np.mean(audio, axis=0)
    raise ValueError("audio must be mono or channel-first stereo")


def _format_estimate_sources(value: tuple[str, ...] | list[str] | str) -> str:
    if isinstance(value, str):
        return value
    return ";".join(str(item) for item in value)


def _metric_fieldnames(rows: list[dict[str, MetricValue]]) -> list[str]:
    fieldnames = list(METRIC_BASE_FIELDS)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def energy_based_segment_candidates(
    track_dir: Path,
    duration: float = 20.0,
    sr: int | None = None,
    hop: float = 5.0,
    n_candidates: int | None = None,
    min_stem_energy_db: float = -60.0,
) -> list[dict[str, object]]:
    """Return sliding-window segment candidates for a MUSDB-style track.

    Candidates are ranked by total mixture energy (descending). When the track
    directory contains MUSDB stems, per-stem RMS levels are included so the
    caller can avoid windows where an instrument is nearly silent.

    .. note::
        Mixture energy is a coarse proxy. A high-energy window may still miss a
        soft stem, and a low-energy window does not mean the separation model
        is wrong. Use this helper to avoid misleading SI-SDR caused by quiet
        intros or breaks, not as a ground-truth instrument detector.
    """
    track_dir = Path(track_dir)
    mixture_path = track_dir / "mixture.wav"
    if not mixture_path.exists():
        return []

    info = load_audio(mixture_path, sr=sr, mono=True)
    mixture, loaded_sr = info[0], info[1]
    total_duration = mixture.shape[-1] / loaded_sr
    if duration >= total_duration:
        stem_rms = _stem_rms_at_start(track_dir, start=0.0, duration=duration, sr=loaded_sr)
        return [
            {
                "start_sec": 0.0,
                "duration_sec": duration,
                "mixture_energy_db": _signal_energy_db(mixture),
                "stem_rms_db": stem_rms,
                "all_stems_above_threshold": _all_above(stem_rms, min_stem_energy_db),
            }
        ]

    window_samples = int(round(duration * loaded_sr))
    hop_samples = max(1, int(round(hop * loaded_sr)))
    candidates: list[dict[str, object]] = []
    for start_sample in range(0, mixture.shape[-1] - window_samples + 1, hop_samples):
        window = mixture[..., start_sample : start_sample + window_samples]
        start_sec = start_sample / loaded_sr
        stem_rms = _stem_rms_at_start(track_dir, start=start_sec, duration=duration, sr=loaded_sr)
        candidates.append(
            {
                "start_sec": start_sec,
                "duration_sec": duration,
                "mixture_energy_db": _signal_energy_db(window),
                "stem_rms_db": stem_rms,
                "all_stems_above_threshold": _all_above(stem_rms, min_stem_energy_db),
            }
        )

    candidates.sort(key=lambda item: float(item["mixture_energy_db"]), reverse=True)
    if n_candidates is not None:
        candidates = candidates[:n_candidates]
    return candidates


def pick_evaluation_start_sec(
    track_dir: Path,
    duration: float = 20.0,
    sr: int | None = None,
    hop: float = 5.0,
    min_stem_energy_db: float = -60.0,
    fallback_start_sec: float = 0.0,
) -> float:
    """Pick a start time for evaluation, falling back when stems are too quiet.

    The function prefers the highest-energy candidate where all available stems
    exceed ``min_stem_energy_db``. If no candidate satisfies the threshold, it
    returns ``fallback_start_sec`` and the caller should note that SI-SDR may be
    dominated by near-silent stems.
    """
    candidates = energy_based_segment_candidates(
        track_dir,
        duration=duration,
        sr=sr,
        hop=hop,
        min_stem_energy_db=min_stem_energy_db,
    )
    for candidate in candidates:
        if bool(candidate["all_stems_above_threshold"]):
            return float(candidate["start_sec"])
    return fallback_start_sec


def _stem_rms_at_start(
    track_dir: Path,
    start: float,
    duration: float,
    sr: int,
) -> dict[str, float]:
    """Load available MUSDB stems for a window and return RMS levels in dB."""
    levels: dict[str, float] = {}
    for stem in MUSDB_4STEMS:
        path = Path(track_dir) / f"{stem}.wav"
        if not path.exists():
            continue
        try:
            audio, _ = load_audio(path, sr=sr, mono=True, start=start, duration=duration)
        except Exception:
            continue
        if audio.size:
            levels[stem] = rms_db(audio)
    return levels


def _signal_energy_db(audio: np.ndarray, eps: float = 1e-12) -> float:
    vec = np.asarray(audio, dtype=np.float64).reshape(-1)
    return 10.0 * math.log10(float(np.dot(vec, vec)) + eps)


def _all_above(levels: dict[str, float], threshold: float) -> bool:
    if not levels:
        return True
    return all(value > threshold for value in levels.values())
