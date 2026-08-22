"""Standard rows for Chapter 8 model comparison outputs."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping

from _common.paths import portable_path
from _common.tables import append_row, write_rows


MODEL_COMPARISON_FIELDS = [
    "model_name",
    "prompt_id",
    "dataset_context",
    "duration_sec",
    "wall_time_sec",
    "device",
    "peak_memory_note",
    "clap_score",
    "fad_proxy",
    "spectral_flatness",
    "loudness_lufs",
    "human_rating_mean",
    "failure_notes",
    "output_audio_path",
]


def append_model_comparison(path: Path | str, row: Mapping[str, object]) -> None:
    """Append one model-comparison row with the chapter schema."""
    append_row(path, normalize_model_comparison_row(row), fieldnames=MODEL_COMPARISON_FIELDS)


def write_model_comparison(path: Path | str, rows: Iterable[Mapping[str, object]]) -> None:
    """Write model-comparison rows with the chapter schema."""
    write_rows(path, [normalize_model_comparison_row(row) for row in rows], fieldnames=MODEL_COMPARISON_FIELDS)


def build_model_comparison_rows(
    metric_rows: Iterable[Mapping[str, object]] = (),
    runner_status_rows: Iterable[Mapping[str, object]] = (),
    fad_proxy_value: object = "",
    clap_score_rows: Iterable[Mapping[str, object]] = (),
) -> list[dict[str, object]]:
    """Build comparison rows from generated-audio metrics and runner statuses."""
    clap_by_audio = best_clap_by_audio(clap_score_rows)
    rows = [
        comparison_row_from_metric(row, fad_proxy_value=fad_proxy_value, clap_by_audio=clap_by_audio)
        for row in metric_rows
    ]
    rows.extend(comparison_rows_from_runner_statuses(runner_status_rows))
    return rows


def comparison_row_from_metric(
    metric_row: Mapping[str, object],
    fad_proxy_value: object = "",
    clap_by_audio: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Convert one audio-metric row into the model-comparison schema."""
    audio_path = portable_path(str(metric_row.get("audio_path", "")), Path("."))
    return normalize_model_comparison_row(
        {
            "model_name": infer_model_name(audio_path),
            "prompt_id": infer_prompt_id(audio_path),
            "dataset_context": infer_dataset_context(audio_path),
            "duration_sec": metric_row.get("duration_sec", ""),
            "wall_time_sec": "",
            "device": "",
            "peak_memory_note": "",
            "clap_score": (clap_by_audio or {}).get(audio_path, ""),
            "fad_proxy": fad_proxy_value,
            "spectral_flatness": metric_row.get("spectral_flatness", ""),
            "loudness_lufs": metric_row.get("loudness_lufs", ""),
            "human_rating_mean": "",
            "failure_notes": "",
            "output_audio_path": audio_path,
        }
    )


def comparison_rows_from_runner_statuses(
    runner_status_rows: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Convert unavailable runner statuses into comparison rows with failure notes."""
    rows = []
    for status_row in runner_status_rows:
        status = str(status_row.get("status", ""))
        if status == "available":
            continue
        reason = str(status_row.get("reason", "")).strip()
        next_action = str(status_row.get("next_action", "")).strip()
        failure = status
        if reason:
            failure += f": {reason}"
        if next_action:
            failure += f" | next: {next_action}"
        rows.append(
            normalize_model_comparison_row(
                {
                    "model_name": status_row.get("model_name", ""),
                    "prompt_id": "",
                    "dataset_context": "pretrained runner status",
                    "duration_sec": "",
                    "wall_time_sec": "",
                    "device": "",
                    "peak_memory_note": status_row.get("estimated_vram_gb", ""),
                    "clap_score": "",
                    "fad_proxy": "",
                    "spectral_flatness": "",
                    "loudness_lufs": "",
                    "human_rating_mean": "",
                    "failure_notes": failure,
                    "output_audio_path": "",
                }
            )
        )
    return rows


def write_model_comparison_from_outputs(
    chapter_root: Path | str = Path("."),
    output_csv: Path | str | None = None,
) -> list[dict[str, object]]:
    """Write a comparison table from current Chapter 8 output CSV files."""
    chapter_root = Path(chapter_root)
    table_dir = chapter_root / "outputs" / "tables"
    metrics_dir = chapter_root / "outputs" / "metrics"
    output_csv = Path(output_csv) if output_csv is not None else table_dir / "08_model_comparison.csv"
    metric_rows = read_csv_rows(metrics_dir / "08_7_audio_metrics.csv")
    runner_rows = read_csv_rows(table_dir / "08_model_runner_status.csv")
    fad_value = read_first_value(metrics_dir / "08_7_fad_proxy.csv", "fad_proxy")
    clap_rows = read_csv_rows(metrics_dir / "08_7_clap_scores.csv")
    rows = build_model_comparison_rows(
        metric_rows=metric_rows,
        runner_status_rows=runner_rows,
        fad_proxy_value=fad_value,
        clap_score_rows=clap_rows,
    )
    write_model_comparison(output_csv, rows)
    return rows


def normalize_model_comparison_row(row: Mapping[str, object]) -> dict[str, object]:
    """Return a row containing every comparison-table field."""
    normalized = {field: row.get(field, "") for field in MODEL_COMPARISON_FIELDS}
    normalized["model_name"] = portable_path(normalized.get("model_name", ""), Path("."))
    normalized["output_audio_path"] = portable_path(normalized.get("output_audio_path", ""), Path("."))
    return normalized


def infer_model_name(audio_path: Path | str) -> str:
    """Infer a teaching model/source label from a generated audio path."""
    path = Path(str(audio_path))
    parent = path.parent.name
    stem = path.stem
    if parent == "08_1":
        return "synthesis_examples"
    if parent == "08_2":
        return "mulaw_codec_demo"
    if parent == "08_3":
        return "toy_wavenet_nsynth"
    if parent == "08_5":
        return "mini_codec_lm_teaching"
    if "musicgen" in parent.lower():
        return "musicgen"
    if "audioldm" in parent.lower():
        return "audioldm2"
    if "stable" in parent.lower():
        return "stable_audio_open"
    if "yue" in parent.lower():
        return "yue"
    if "ace" in parent.lower():
        return "ace_step"
    return parent or stem or "unknown"


def infer_prompt_id(audio_path: Path | str) -> str:
    """Infer a stable prompt or case id from an audio path."""
    path = Path(str(audio_path))
    parent = path.parent.name
    if parent in {"08_1", "08_2", "08_3"}:
        return path.stem
    return path.stem


def infer_dataset_context(audio_path: Path | str) -> str:
    """Infer the dataset or demonstration context from a generated audio path."""
    path = Path(str(audio_path))
    parent = path.parent.name
    if parent == "08_1":
        return "synthetic waveform demo"
    if parent == "08_2":
        return "synthetic harmonic waveform"
    if parent == "08_3":
        return "NSynth subset"
    if parent == "08_5":
        return "FMA small teaching tokens"
    if parent.startswith("08_6"):
        return "pretrained generation"
    return "chapter08 generated audio"


def best_clap_by_audio(clap_score_rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Return the best CLAP score per audio path."""
    best: dict[str, float] = {}
    for row in clap_score_rows:
        audio_path = portable_path(str(row.get("audio_path", "")), Path("."))
        if not audio_path:
            continue
        try:
            score = float(row.get("clap_score", ""))
        except (TypeError, ValueError):
            continue
        if audio_path not in best or score > best[audio_path]:
            best[audio_path] = score
    return best


def read_csv_rows(path: Path | str) -> list[dict[str, str]]:
    """Read CSV rows, returning an empty list when the file is absent."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_first_value(path: Path | str, field: str) -> str:
    """Return the first CSV value for a field, or an empty string."""
    rows = read_csv_rows(path)
    if not rows:
        return ""
    return rows[0].get(field, "")
