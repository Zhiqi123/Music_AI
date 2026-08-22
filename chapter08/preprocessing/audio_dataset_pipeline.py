"""Audio cleaning, segmentation, augmentation, and manifest export."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from _common.audio_io import load_audio, normalize_peak, pad_or_trim, save_audio, to_mono
from _common.paths import portable_path
from _common.plotting import finish_figure, setup_plot_style
from _common.tables import write_rows
from finetune.ace_step_lora_dataset import resolve_audio_path


CHAPTER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CHAPTER_ROOT.parents[1]

QUALITY_FIELDS = [
    "case_id",
    "source_path",
    "status",
    "decision",
    "sample_rate",
    "channels",
    "duration_sec",
    "peak_dbfs",
    "rms_dbfs",
    "clipping_ratio",
    "silence_ratio",
    "prompt_en",
    "prompt_cn",
    "detail",
]

SEGMENT_FIELDS = [
    "case_id",
    "source_case_id",
    "split",
    "path",
    "source_path",
    "conditioning_type",
    "prompt_en",
    "prompt_cn",
    "category",
    "sample_rate",
    "duration_sec",
    "transform",
]

AUGMENTATION_FIELDS = [
    "case_id",
    "source_case_id",
    "path",
    "sample_rate",
    "duration_sec",
    "policy",
    "transform",
    "detail",
]

TRAINING_USE_FIELDS = [
    "output",
    "used_by",
    "conditioning",
    "why_it_matters",
]


@dataclass(frozen=True)
class AuthorAudioItem:
    """One row from the author-audio manifest."""

    case_id: str
    path: Path
    split: str
    category: str
    prompt_en: str
    prompt_cn: str


def load_author_audio_items(
    manifest_csv: Path | str,
    audio_root: Path | str,
    limit: int | None = None,
) -> list[AuthorAudioItem]:
    """Load author-owned audio rows and resolve local file paths."""
    manifest_csv = Path(manifest_csv)
    audio_root = Path(audio_root)
    items: list[AuthorAudioItem] = []
    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_path = row.get("path", "").strip()
            if not raw_path:
                continue
            audio_path = resolve_audio_path(raw_path, manifest_path=manifest_csv, audio_root=audio_root)
            items.append(
                AuthorAudioItem(
                    case_id=row.get("case_id", audio_path.stem).strip() or audio_path.stem,
                    path=audio_path,
                    split=(row.get("split") or "train").strip() or "train",
                    category=(row.get("category") or "").strip(),
                    prompt_en=(row.get("prompt_en") or row.get("caption") or "").strip(),
                    prompt_cn=(row.get("prompt_cn") or "").strip(),
                )
            )
            if limit is not None and len(items) >= limit:
                break
    return items


def build_audio_quality_rows(
    items: Iterable[AuthorAudioItem],
    chapter_root: Path | str = CHAPTER_ROOT,
    min_duration_sec: float = 2.0,
) -> list[dict[str, object]]:
    """Build a header and signal-quality report for training audio."""
    root = Path(chapter_root)
    rows = []
    for item in items:
        rows.append(audio_quality_row(item, root, min_duration_sec=min_duration_sec))
    return rows


def audio_quality_row(
    item: AuthorAudioItem,
    chapter_root: Path,
    min_duration_sec: float = 2.0,
) -> dict[str, object]:
    """Return one data-quality row for a manifest item."""
    try:
        info = sf.info(item.path)
        audio, _ = load_audio(item.path, sr=None, mono=True)
        stats = audio_signal_stats(audio)
    except Exception as exc:
        return {
            "case_id": item.case_id,
            "source_path": portable_path(item.path, chapter_root),
            "status": "unreadable",
            "decision": "reject",
            "sample_rate": "",
            "channels": "",
            "duration_sec": "",
            "peak_dbfs": "",
            "rms_dbfs": "",
            "clipping_ratio": "",
            "silence_ratio": "",
            "prompt_en": item.prompt_en,
            "prompt_cn": item.prompt_cn,
            "detail": f"{type(exc).__name__}: {exc}",
        }

    detail: list[str] = []
    decision = "keep"
    status = "ok"
    if info.duration < min_duration_sec:
        status = "too_short"
        decision = "reject"
        detail.append(f"duration<{min_duration_sec:g}s")
    if stats["clipping_ratio"] > 0.001:
        status = "warning" if decision == "keep" else status
        detail.append("possible clipping")
    if stats["silence_ratio"] > 0.8:
        status = "warning" if decision == "keep" else status
        detail.append("mostly silent")
    if stats["rms_dbfs"] < -70:
        status = "warning" if decision == "keep" else status
        detail.append("very low level")

    return {
        "case_id": item.case_id,
        "source_path": portable_path(item.path, chapter_root),
        "status": status,
        "decision": decision,
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "duration_sec": round(float(info.duration), 3),
        "peak_dbfs": round(stats["peak_dbfs"], 3),
        "rms_dbfs": round(stats["rms_dbfs"], 3),
        "clipping_ratio": round(stats["clipping_ratio"], 6),
        "silence_ratio": round(stats["silence_ratio"], 6),
        "prompt_en": item.prompt_en,
        "prompt_cn": item.prompt_cn,
        "detail": "; ".join(detail) if detail else "usable for segmentation",
    }


def audio_signal_stats(audio: np.ndarray) -> dict[str, float]:
    """Compute small signal metrics used for dataset cleaning decisions."""
    y = to_mono(np.asarray(audio, dtype=np.float32))
    finite = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(finite))) if finite.size else 0.0
    rms = float(np.sqrt(np.mean(finite.astype(np.float64) ** 2))) if finite.size else 0.0
    clipping_ratio = float(np.mean(np.abs(finite) >= 0.999)) if finite.size else 0.0
    silence_ratio = float(np.mean(np.abs(finite) < 1e-4)) if finite.size else 1.0
    return {
        "peak_dbfs": amplitude_to_dbfs(peak),
        "rms_dbfs": amplitude_to_dbfs(rms),
        "clipping_ratio": clipping_ratio,
        "silence_ratio": silence_ratio,
    }


def amplitude_to_dbfs(value: float) -> float:
    """Convert a linear amplitude to dBFS with a floor."""
    return float(20.0 * np.log10(max(float(value), 1e-12)))


def clean_audio(
    path: Path | str,
    target_sr: int = 32000,
    trim_top_db: float = 45.0,
    rms_target_dbfs: float = -20.0,
) -> tuple[np.ndarray, int]:
    """Load mono audio, resample, trim silence, and normalize RMS."""
    audio, sr = load_audio(path, sr=target_sr, mono=True)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    trimmed, _ = librosa.effects.trim(audio, top_db=trim_top_db)
    if trimmed.size:
        audio = trimmed.astype(np.float32)
    audio = rms_normalize(audio, target_dbfs=rms_target_dbfs)
    audio = normalize_peak(audio, peak=0.98)
    return audio.astype(np.float32), sr


def rms_normalize(audio: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    """Normalize RMS to an approximate dBFS target, then leave peak limiting to caller."""
    y = np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2))) if y.size else 0.0
    if rms <= 1e-12:
        return y.copy()
    target = 10.0 ** (float(target_dbfs) / 20.0)
    return (y * (target / rms)).astype(np.float32)


def segment_audio(audio: np.ndarray, sr: int, segment_sec: float, hop_sec: float) -> list[np.ndarray]:
    """Cut cleaned audio into fixed-length training windows."""
    target = int(round(segment_sec * sr))
    hop = max(1, int(round(hop_sec * sr)))
    if audio.size <= target:
        return [pad_or_trim(audio, target)]
    segments = []
    for start in range(0, max(1, audio.size - target + 1), hop):
        segments.append(pad_or_trim(audio[start : start + target], target))
    return segments


def limit_peak(audio: np.ndarray, peak: float = 0.98) -> np.ndarray:
    """Scale only when an augmented signal would exceed the target peak."""
    y = np.asarray(audio, dtype=np.float32)
    current = float(np.max(np.abs(y))) if y.size else 0.0
    if current <= peak or current <= 1e-12:
        return y.copy()
    return (y * (peak / current)).astype(np.float32)


def add_noise_at_snr(
    audio: np.ndarray,
    rng: np.random.Generator,
    snr_db: float,
) -> np.ndarray:
    """Add white noise at an approximate signal-to-noise ratio."""
    y = np.asarray(audio, dtype=np.float32)
    if not y.size:
        return y.copy()
    signal_rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    if signal_rms <= 1e-12:
        return y.copy()
    noise = rng.normal(0.0, 1.0, size=y.shape).astype(np.float32)
    noise_rms = float(np.sqrt(np.mean(noise.astype(np.float64) ** 2)))
    scale = signal_rms / (max(noise_rms, 1e-12) * (10.0 ** (snr_db / 20.0)))
    return (y + noise * scale).astype(np.float32)


def lowpass_fft(audio: np.ndarray, sr: int, cutoff_hz: float, transition_hz: float = 1500.0) -> np.ndarray:
    """Apply a small FFT-domain low-pass filter for channel/codec robustness demos."""
    y = np.asarray(audio, dtype=np.float32)
    if y.size == 0:
        return y.copy()
    cutoff_hz = min(float(cutoff_hz), float(sr) * 0.49)
    transition_hz = max(float(transition_hz), 1.0)
    freqs = np.fft.rfftfreq(y.size, d=1.0 / float(sr))
    spectrum = np.fft.rfft(y.astype(np.float64))
    weights = np.ones_like(freqs)
    rolloff_start = cutoff_hz
    rolloff_end = min(float(sr) * 0.5, cutoff_hz + transition_hz)
    transition = (freqs > rolloff_start) & (freqs < rolloff_end)
    weights[freqs >= rolloff_end] = 0.0
    if np.any(transition):
        weights[transition] = (rolloff_end - freqs[transition]) / (rolloff_end - rolloff_start)
    filtered = np.fft.irfft(spectrum * weights, n=y.size)
    return filtered.astype(np.float32)


def augment_segment(audio: np.ndarray, sr: int, rng: np.random.Generator) -> list[tuple[str, np.ndarray, str, str]]:
    """Return deterministic purpose-driven augmentations for a small author dataset."""
    y = np.asarray(audio, dtype=np.float32)
    gain_minus_6db = y * (10.0 ** (-6.0 / 20.0))
    noise_24db = add_noise_at_snr(y, rng, snr_db=24.0)

    shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=1.0).astype(np.float32)
    shifted_stretched = librosa.effects.time_stretch(shifted, rate=0.97).astype(np.float32)
    shifted_stretched = pad_or_trim(shifted_stretched, y.size)

    lowpassed = lowpass_fft(y, sr=sr, cutoff_hz=min(12000.0, sr * 0.42))
    lowpass_noise = add_noise_at_snr(lowpassed, rng, snr_db=30.0)

    combo = librosa.effects.pitch_shift(y, sr=sr, n_steps=-0.5).astype(np.float32)
    combo = librosa.effects.time_stretch(combo, rate=1.03).astype(np.float32)
    combo = pad_or_trim(combo, y.size) * (10.0 ** (-2.0 / 20.0))
    combo = add_noise_at_snr(combo, rng, snr_db=28.0)
    return [
        (
            "gain_minus_6db",
            limit_peak(gain_minus_6db),
            "loudness_robustness",
            "gain=-6 dB; useful when absolute playback level is not a conditioning target",
        ),
        (
            "noise_24db_snr",
            limit_peak(noise_24db),
            "recording_robustness",
            "add broadband noise at about 24 dB SNR",
        ),
        (
            "pitch_up_1st_time_stretch_0_97",
            limit_peak(shifted_stretched),
            "label_preserving_pitch_tempo",
            "pitch=+1 semitone and tempo rate=0.97; avoid when exact pitch or tempo is the label",
        ),
        (
            "lowpass_12khz_noise_30db_snr",
            limit_peak(lowpass_noise),
            "channel_codec_robustness",
            "low-pass near 12 kHz and add about 30 dB SNR noise",
        ),
        (
            "combo_pitch_tempo_gain_noise",
            limit_peak(combo),
            "small_data_combo",
            "pitch=-0.5 semitone, tempo rate=1.03, gain=-2 dB, noise about 28 dB SNR",
        ),
    ]


def prepare_author_audio_dataset(
    manifest_csv: Path | str,
    audio_root: Path | str,
    output_audio_dir: Path | str,
    output_table_dir: Path | str,
    chapter_root: Path | str = CHAPTER_ROOT,
    target_sr: int = 32000,
    segment_sec: float = 8.0,
    hop_sec: float = 8.0,
    max_files: int = 3,
    max_segments_per_file: int = 2,
    seed: int = 8,
) -> dict[str, list[dict[str, object]]]:
    """Prepare a small but real training-data slice from author audio."""
    root = Path(chapter_root)
    output_audio_dir = Path(output_audio_dir)
    output_table_dir = Path(output_table_dir)
    segments_dir = output_audio_dir / "segments"
    augmentations_dir = output_audio_dir / "augmentations"
    segments_dir.mkdir(parents=True, exist_ok=True)
    augmentations_dir.mkdir(parents=True, exist_ok=True)
    output_table_dir.mkdir(parents=True, exist_ok=True)

    items = load_author_audio_items(manifest_csv, audio_root, limit=max_files)
    quality_rows = build_audio_quality_rows(items, chapter_root=root)
    write_rows(output_table_dir / "08_0_author_audio_quality.csv", quality_rows, fieldnames=QUALITY_FIELDS)

    keep_by_case = {row["case_id"]: row["decision"] == "keep" for row in quality_rows}
    segment_rows: list[dict[str, object]] = []
    augmentation_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)

    for item in items:
        if not keep_by_case.get(item.case_id, False):
            continue
        cleaned, sr = clean_audio(item.path, target_sr=target_sr)
        for index, segment in enumerate(segment_audio(cleaned, sr, segment_sec, hop_sec)[:max_segments_per_file]):
            segment_id = f"{item.case_id}_seg{index:03d}"
            segment_path = segments_dir / f"{segment_id}_clean.wav"
            save_audio(segment_path, segment, sr)
            segment_rows.append(
                {
                    "case_id": segment_id,
                    "source_case_id": item.case_id,
                    "split": item.split,
                    "path": portable_path(segment_path, root),
                    "source_path": portable_path(item.path, root),
                    "conditioning_type": "text_prompt",
                    "prompt_en": item.prompt_en,
                    "prompt_cn": item.prompt_cn,
                    "category": item.category,
                    "sample_rate": sr,
                    "duration_sec": segment_sec,
                    "transform": "resample_mono_trim_rms_peak_segment",
                }
            )
            if index == 0:
                for aug_name, aug_audio, policy, detail in augment_segment(segment, sr, rng):
                    aug_id = f"{segment_id}_{aug_name}"
                    aug_path = augmentations_dir / f"{aug_id}.wav"
                    save_audio(aug_path, aug_audio, sr)
                    augmentation_rows.append(
                        {
                            "case_id": aug_id,
                            "source_case_id": segment_id,
                            "path": portable_path(aug_path, root),
                            "sample_rate": sr,
                            "duration_sec": segment_sec,
                            "policy": policy,
                            "transform": aug_name,
                            "detail": detail,
                        }
                    )

    write_rows(output_table_dir / "08_0_prepared_segments.csv", segment_rows, fieldnames=SEGMENT_FIELDS)
    write_rows(
        output_table_dir / "08_0_augmentation_manifest.csv",
        augmentation_rows,
        fieldnames=AUGMENTATION_FIELDS,
    )
    write_rows(
        output_table_dir / "08_0_training_data_usage.csv",
        training_data_usage_rows(),
        fieldnames=TRAINING_USE_FIELDS,
    )
    return {
        "quality": quality_rows,
        "segments": segment_rows,
        "augmentations": augmentation_rows,
        "training_usage": training_data_usage_rows(),
    }


def training_data_usage_rows() -> list[dict[str, object]]:
    """Explain how prepared audio artifacts feed the chapter's model families."""
    return [
        {
            "output": "08_0_author_audio_quality.csv",
            "used_by": "all training or fine-tuning paths",
            "conditioning": "quality filters before manifest export",
            "why_it_matters": "rejects unreadable, too short, mostly silent, or clipped files before training.",
        },
        {
            "output": "08_0_prepared_segments.csv",
            "used_by": "WaveNet, codec token cache, MusicGen fine-tuning, ACE-Step LoRA",
            "conditioning": "audio segment plus text prompt/caption",
            "why_it_matters": "fixed sample rate and duration make batches stable and keep text-audio pairs aligned.",
        },
        {
            "output": "08_0_augmentation_manifest.csv",
            "used_by": "small-data fine-tuning and robustness experiments",
            "conditioning": "same prompt, transformed audio target",
            "why_it_matters": "records purpose-driven gain, noise, pitch, tempo, and channel variants, including combined transforms.",
        },
    ]


def plot_preprocessing_summary(
    source_path: Path | str,
    cleaned_path: Path | str,
    augmented_path: Path | str,
    out_path: Path | str | None = None,
) -> plt.Figure:
    """绘制原始、清洗后和增强后的波形片段。"""
    setup_plot_style()
    source, source_sr = load_audio(source_path, sr=32000, mono=True, duration=8.0)
    cleaned, cleaned_sr = load_audio(cleaned_path, sr=32000, mono=True)
    augmented, augmented_sr = load_audio(augmented_path, sr=32000, mono=True)
    examples = [
        ("原始片段", source, source_sr),
        ("清洗后的训练片段", cleaned, cleaned_sr),
        ("增强片段", augmented, augmented_sr),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(9, 5.2), sharex=False)
    for ax, (title, audio, sr) in zip(axes, examples):
        t = np.arange(audio.size) / sr
        ax.plot(t, audio, color="0.15", linewidth=0.6)
        ax.set_title(title)
        ax.set_xlabel("时间（秒）")
        ax.set_ylabel("振幅")
        ax.set_ylim(-1.05, 1.05)
    return finish_figure(fig, Path(out_path) if out_path is not None else None)
