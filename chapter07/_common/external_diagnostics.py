"""No-reference diagnostics for external source-separation cases."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .metrics import energy_ratio, reconstruction_error, rms_db


AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".aif", ".aiff", ".ogg")

DIAGNOSTIC_FIELDS = [
    "case_id",
    "model",
    "mode",
    "stem",
    "reference_type",
    "expected_present",
    "expected_absent",
    "expected_uncertain",
    "rms_db",
    "energy_ratio",
    "absent_energy_ratio",
    "known_present_energy_ratio",
    "reconstruction_error",
    "notes",
]

DIAGNOSTIC_SUMMARY_FIELDS = [
    "case_id",
    "model",
    "mode",
    "reference_type",
    "stem_count",
    "reconstruction_error",
    "absent_energy_ratio_sum",
    "absent_energy_ratio_max",
    "absent_stem_with_max",
    "uncertain_energy_ratio_sum",
    "known_present_energy_ratio_sum",
    "dominant_stem",
    "dominant_energy_ratio",
    "review_priority",
]


_STEM_ALIASES = {
    "vocal": "vocals",
    "vox": "vocals",
    "voice": "vocals",
    "voices": "vocals",
    "lead_vocal": "vocals",
    "lead_vocals": "vocals",
    "ins": "accompaniment",
    "instrumental": "accompaniment",
    "instruments": "accompaniment",
    "backing": "accompaniment",
    "no_vocals": "accompaniment",
    "pianos": "piano",
    "keys": "piano",
    "keyboards": "piano",
    "basses": "bass",
    "drum": "drums",
    "drums_sfx": "drums",
    "drum_sfx": "drums",
    "drumsfx": "drums",
    "sfx_drums": "drums",
    "synth": "synths",
    "synthesizer": "synths",
    "synthesizers": "synths",
    "choir": "choirs",
    "woods": "woodwinds",
    "woodwind": "woodwinds",
    "string": "strings",
    "brasses": "brass",
    "percussions": "percussion",
    "others": "other",
}


def parse_semicolon_list(value: str | None) -> list[str]:
    """Parse semicolon-separated manifest fields."""
    if not value:
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def read_external_manifest(path: Path) -> list[dict[str, str]]:
    """Read an external-case CSV manifest."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_output_audio_manifest(path: Path) -> list[dict[str, str]]:
    """Read the 07_4 output-audio manifest."""
    return read_external_manifest(path)


def resolve_repo_path(value: str | Path, repo_root: Path) -> Path:
    """Resolve an absolute or repo-relative path."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path(repo_root) / path).resolve()


def collect_manifest_output_paths(
    manifest_path: Path,
    repo_root: Path,
    case_ids: list[str] | set[str] | tuple[str, ...] | None = None,
) -> dict[str, dict[str, dict[str, Path]]]:
    """Collect 07_4 output stems as case -> model -> stem -> path."""
    wanted = set(case_ids or [])
    grouped: dict[str, dict[str, dict[str, Path]]] = {}
    for row in read_output_audio_manifest(manifest_path):
        case_id = str(row.get("case_id", "")).strip()
        model_id = str(row.get("model_id") or row.get("model") or "").strip()
        stem = canonical_stem_name(str(row.get("stem", "")).strip(), case_id)
        path_value = str(row.get("path", "")).strip()
        if not case_id or not model_id or not stem or not path_value:
            continue
        if wanted and case_id not in wanted:
            continue
        path = resolve_repo_path(path_value, repo_root)
        if not _is_audio_file(path):
            continue
        grouped.setdefault(case_id, {}).setdefault(model_id, {})[stem] = path
    return grouped


def discover_output_audio_paths(
    output_root: Path,
    case_ids: list[str] | set[str] | tuple[str, ...] | None = None,
) -> dict[str, dict[str, dict[str, Path]]]:
    """Discover model outputs directly from output_audio/07_4 when no CSV exists."""
    output_root = Path(output_root)
    wanted = set(case_ids or [])
    grouped: dict[str, dict[str, dict[str, Path]]] = {}
    if not output_root.exists():
        return grouped

    for case_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
        case_id = case_dir.name
        if case_id.startswith("_") or (wanted and case_id not in wanted):
            continue
        for model_dir in sorted(path for path in case_dir.iterdir() if path.is_dir()):
            stems: dict[str, Path] = {}
            for path in _iter_audio_files(model_dir, recursive=True):
                clean_name = path.stem.lower().replace("-", "_").replace(" ", "_")
                if clean_name == "input" or _is_mixture_like(path):
                    continue
                stem = canonical_stem_name(path.stem, case_id)
                stems.setdefault(stem, path)
            if stems:
                grouped.setdefault(case_id, {})[model_dir.name] = stems
    return grouped


def output_mode_from_stems(stems: list[str] | tuple[str, ...] | set[str]) -> str:
    """Infer a concise stem-taxonomy label from output names."""
    names = set(stems)
    if names == {"vocals", "accompaniment"}:
        return "2stems"
    if {"vocals", "drums", "bass", "other", "guitar", "piano"}.issubset(names):
        return "6stems"
    if {"vocals", "drums", "bass", "other"}.issubset(names):
        return "4stems"
    if {"vocals", "accompaniment"}.issubset(names):
        return "2stems_plus"
    return f"{len(names)}stems"


def repo_relative_path(path: Path | str, repo_root: Path) -> str:
    """Return a repo-relative display path, avoiding machine-specific prefixes."""
    path = Path(path)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError:
        return path.name


def find_case_mixture_path(
    case_dir: Path,
    manifest_row: dict[str, str] | None = None,
    repo_root: Path | None = None,
) -> Path | None:
    """Find the mixture/input audio for one external case."""
    case_dir = Path(case_dir)
    manifest_row = manifest_row or {}
    repo_root = Path(repo_root) if repo_root is not None else case_dir

    for key in ("mixture_file", "mixture_path", "input_file", "input_path"):
        value = manifest_row.get(key)
        if not value:
            continue
        path = resolve_audio_path(value, case_dir, repo_root)
        if path.exists():
            return path

    candidates = []
    for name in ("mixture.wav", "mix.wav"):
        path = case_dir / name
        if path.exists():
            candidates.append(path)
    candidates.extend(path for path in _iter_audio_files(case_dir) if _is_mixture_like(path))
    return sorted(set(candidates))[0] if candidates else None


def discover_case_stem_paths(
    case_dir: Path,
    manifest_row: dict[str, str] | None = None,
    repo_root: Path | None = None,
    mixture_path: Path | None = None,
) -> dict[str, Path]:
    """Discover owned/project stems from manifest hints and case filenames."""
    case_dir = Path(case_dir)
    manifest_row = manifest_row or {}
    repo_root = Path(repo_root) if repo_root is not None else case_dir
    case_id = manifest_row.get("case_id", "")
    mixture_path = Path(mixture_path).resolve() if mixture_path else None
    stems: dict[str, Path] = {}

    for item in parse_semicolon_list(manifest_row.get("stem_files")):
        stem_name, rel_path = _split_stem_file_item(item)
        path = resolve_audio_path(rel_path, case_dir, repo_root)
        if path.exists() and _is_audio_file(path) and not _same_path(path, mixture_path):
            stems.setdefault(canonical_stem_name(stem_name or path.stem, case_id), path)

    stem_dirs = parse_semicolon_list(manifest_row.get("stem_dirs")) or ["stems", "groups", "."]
    for rel_dir in stem_dirs:
        root = resolve_audio_path(rel_dir, case_dir, repo_root)
        if not root.exists() or not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            if not _is_audio_file(path) or _same_path(path, mixture_path):
                continue
            stem = canonical_stem_name(path.stem, case_id)
            if stem in {"mix", "mixture", "full"} or _is_mixture_like(path):
                continue
            stems.setdefault(stem, path)
    return stems


def find_mixture_paths(root: Path) -> list[Path]:
    """Find mixture-like audio files below an author-audio root."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(path for path in _iter_audio_files(root, recursive=True) if _is_mixture_like(path))


def canonical_stem_name(name: str, case_id: str | None = None) -> str:
    """Normalize project-specific stem filenames to stable teaching labels."""
    clean = Path(str(name)).stem.lower()
    clean = clean.replace("-", "_").replace(" ", "_")
    prefixes = [case_id or "", "xiaohetang", "orchestra"]
    for prefix in prefixes:
        prefix = prefix.lower().replace("-", "_").replace(" ", "_")
        if prefix and clean.startswith(prefix + "_"):
            clean = clean[len(prefix) + 1 :]
    clean = "_".join(part for part in clean.split("_") if part)
    return _STEM_ALIASES.get(clean, clean)


def resolve_audio_path(value: str | Path, case_dir: Path, repo_root: Path) -> Path:
    """Resolve manifest paths relative to case dir first, then repo root."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    case_candidate = Path(case_dir) / path
    if case_candidate.exists():
        return case_candidate
    return Path(repo_root) / path


def diagnostic_rows(
    case_id: str,
    model: str,
    mode: str,
    mixture: np.ndarray,
    estimates: dict[str, np.ndarray],
    reference_type: str = "no_reference",
    expected_present: list[str] | None = None,
    expected_absent: list[str] | None = None,
    expected_uncertain: list[str] | None = None,
    notes: str = "",
) -> list[dict[str, str | float]]:
    """Build no-reference diagnostic rows for estimated stems."""
    expected_present = expected_present or []
    expected_absent = expected_absent or []
    expected_uncertain = expected_uncertain or []
    recon_error = reconstruction_error(mixture, estimates) if estimates else float("nan")

    rows = []
    for stem, audio in sorted(estimates.items()):
        ratio = energy_ratio(audio, mixture)
        is_absent = stem in expected_absent
        is_present = stem in expected_present
        rows.append(
            {
                "case_id": case_id,
                "model": model,
                "mode": mode,
                "stem": stem,
                "reference_type": reference_type,
                "expected_present": ";".join(expected_present),
                "expected_absent": ";".join(expected_absent),
                "expected_uncertain": ";".join(expected_uncertain),
                "rms_db": rms_db(audio),
                "energy_ratio": ratio,
                "absent_energy_ratio": ratio if is_absent else "",
                "known_present_energy_ratio": ratio if is_present else "",
                "reconstruction_error": recon_error,
                "notes": notes,
            }
        )
    return rows


def write_diagnostic_rows(path: Path, rows: list[dict[str, str | float]]) -> None:
    """Write diagnostic rows to CSV with a stable schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DIAGNOSTIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarize_diagnostic_rows(rows: list[dict[str, str | float]]) -> list[dict[str, str | float | int]]:
    """Aggregate stem-level diagnostics into one row per case/model/mode."""
    grouped: dict[tuple[str, str, str, str], dict[str, str | float | int]] = {}
    for row in rows:
        key = (
            str(row.get("case_id", "")),
            str(row.get("model", "")),
            str(row.get("mode", "")),
            str(row.get("reference_type", "")),
        )
        item = grouped.setdefault(
            key,
            {
                "case_id": key[0],
                "model": key[1],
                "mode": key[2],
                "reference_type": key[3],
                "stem_count": 0,
                "reconstruction_error": "",
                "absent_energy_ratio_sum": 0.0,
                "absent_energy_ratio_max": 0.0,
                "absent_stem_with_max": "",
                "uncertain_energy_ratio_sum": 0.0,
                "known_present_energy_ratio_sum": 0.0,
                "dominant_stem": "",
                "dominant_energy_ratio": 0.0,
                "review_priority": "low",
            },
        )
        item["stem_count"] = int(item["stem_count"]) + 1

        recon = _float_or_nan(row.get("reconstruction_error"))
        if not np.isnan(recon):
            item["reconstruction_error"] = recon

        stem = str(row.get("stem", ""))
        energy = _float_or_zero(row.get("energy_ratio"))
        if energy > float(item["dominant_energy_ratio"]):
            item["dominant_stem"] = stem
            item["dominant_energy_ratio"] = energy

        absent = row.get("absent_energy_ratio")
        if absent != "":
            absent_value = _float_or_zero(absent)
            item["absent_energy_ratio_sum"] = float(item["absent_energy_ratio_sum"]) + absent_value
            if absent_value > float(item["absent_energy_ratio_max"]):
                item["absent_energy_ratio_max"] = absent_value
                item["absent_stem_with_max"] = stem

        if stem in parse_semicolon_list(str(row.get("expected_uncertain", ""))):
            item["uncertain_energy_ratio_sum"] = (
                float(item["uncertain_energy_ratio_sum"]) + energy
            )

        known_present = row.get("known_present_energy_ratio")
        if known_present != "":
            item["known_present_energy_ratio_sum"] = (
                float(item["known_present_energy_ratio_sum"]) + _float_or_zero(known_present)
            )

    summaries = list(grouped.values())
    for item in summaries:
        item["review_priority"] = _review_priority(
            _float_or_nan(item.get("reconstruction_error")),
            float(item["absent_energy_ratio_sum"]),
            float(item["absent_energy_ratio_max"]),
            float(item["uncertain_energy_ratio_sum"]),
        )
    return sorted(
        summaries,
        key=lambda item: (
            str(item["case_id"]),
            _priority_rank(str(item["review_priority"])),
            -float(item["absent_energy_ratio_max"]),
            str(item["model"]),
        ),
    )


def write_diagnostic_summary(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    """Write case/model diagnostic summaries."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DIAGNOSTIC_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def resolve_case_dir(case_dir: str, repo_root: Path) -> Path:
    """Resolve manifest case paths written from repo root or as absolute paths."""
    path = Path(case_dir).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _split_stem_file_item(item: str) -> tuple[str, str]:
    if "=" not in item:
        return "", item
    stem, rel_path = item.split("=", 1)
    return stem.strip(), rel_path.strip()


def _iter_audio_files(root: Path, recursive: bool = False) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.glob("*")
    return [path for path in iterator if _is_audio_file(path)]


def _is_audio_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS


def _is_mixture_like(path: Path) -> bool:
    name = path.stem.lower().replace("-", "_").replace(" ", "_")
    return (
        name in {"mix", "mixture", "full"}
        or name.endswith("_mix")
        or name.endswith("_mixture")
        or name.endswith("_full")
    )


def _same_path(path: Path, other: Path | None) -> bool:
    if other is None:
        return False
    try:
        return path.resolve() == other.resolve()
    except OSError:
        return False


def _float_or_zero(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _float_or_nan(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _review_priority(
    reconstruction_error_value: float,
    absent_sum: float,
    absent_max: float,
    uncertain_sum: float,
) -> str:
    if absent_max >= 0.05 or (
        not np.isnan(reconstruction_error_value) and reconstruction_error_value >= 0.25
    ):
        return "high"
    if absent_sum >= 0.01 or uncertain_sum >= 0.20 or (
        not np.isnan(reconstruction_error_value) and reconstruction_error_value >= 0.15
    ):
        return "medium"
    return "low"


def _priority_rank(value: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(value, 3)
