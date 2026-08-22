"""Manifest handling for ACE-Step LoRA personalization."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from _common.paths import portable_path


CHAPTER_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_LORA_MANIFEST_COLUMNS = (
    "case_id",
    "path",
    "category",
    "source_type",
    "license_status",
    "can_redistribute",
    "prompt_cn",
    "prompt_en",
)


@dataclass(frozen=True)
class LoRAAudioItem:
    case_id: str
    path: Path
    prompt_cn: str = ""
    prompt_en: str = ""
    category: str = ""
    notes: str = ""


@dataclass(frozen=True)
class LoRAManifestIssue:
    row_index: int
    case_id: str
    field: str
    severity: str
    message: str

    def as_row(self) -> dict[str, object]:
        return {
            "row_index": self.row_index,
            "case_id": self.case_id,
            "field": self.field,
            "severity": self.severity,
            "message": self.message,
        }


def load_lora_manifest(path: Path | str, audio_root: Path | str | None = None) -> list[LoRAAudioItem]:
    """Load the author-audio manifest for personalization experiments."""
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        items = []
        for row in reader:
            raw_path = Path(row["path"])
            audio_path = resolve_audio_path(raw_path, manifest_path=path, audio_root=audio_root)
            items.append(
                LoRAAudioItem(
                    case_id=row.get("case_id", audio_path.stem),
                    path=audio_path,
                    prompt_cn=row.get("prompt_cn", ""),
                    prompt_en=row.get("prompt_en", ""),
                    category=row.get("category", ""),
                    notes=row.get("notes", ""),
                )
            )
    return items


def validate_lora_manifest(
    path: Path | str,
    audio_root: Path | str | None = None,
    require_files: bool = True,
) -> list[LoRAManifestIssue]:
    """Validate manifest schema and local audio references."""
    path = Path(path)
    issues: list[LoRAManifestIssue] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        for column in REQUIRED_LORA_MANIFEST_COLUMNS:
            if column not in fieldnames:
                issues.append(
                    LoRAManifestIssue(
                        row_index=0,
                        case_id="",
                        field=column,
                        severity="error",
                        message="required column is missing",
                    )
                )
        if issues:
            return issues
        seen_case_ids: set[str] = set()
        for row_index, row in enumerate(reader, start=1):
            case_id = row.get("case_id", "").strip()
            if not case_id:
                issues.append(_issue(row_index, case_id, "case_id", "case_id is empty"))
            elif case_id in seen_case_ids:
                issues.append(_issue(row_index, case_id, "case_id", "case_id is duplicated"))
            seen_case_ids.add(case_id)

            raw_path = row.get("path", "").strip()
            if not raw_path:
                issues.append(_issue(row_index, case_id, "path", "path is empty"))
            elif require_files:
                audio_path = resolve_audio_path(raw_path, manifest_path=path, audio_root=audio_root)
                if not audio_path.exists():
                    issues.append(
                        _issue(
                            row_index,
                            case_id,
                            "path",
                            f"audio file does not exist: {portable_path(audio_path, CHAPTER_ROOT)}",
                        )
                    )

            if not row.get("prompt_cn", "").strip() and not row.get("prompt_en", "").strip():
                issues.append(
                    _issue(row_index, case_id, "prompt", "at least one prompt field must be filled")
                )

            can_redistribute = row.get("can_redistribute", "").strip().lower()
            if can_redistribute not in {"true", "false", "yes", "no", "1", "0"}:
                issues.append(
                    _issue(
                        row_index,
                        case_id,
                        "can_redistribute",
                        "expected a boolean-like value",
                        severity="warning",
                    )
                )
    return issues


def resolve_audio_path(
    raw_path: Path | str,
    manifest_path: Path | str,
    audio_root: Path | str | None = None,
) -> Path:
    """Resolve manifest audio paths against likely local bases."""
    raw_path = Path(raw_path)
    manifest_path = Path(manifest_path)
    if raw_path.is_absolute():
        return raw_path
    candidates: list[Path] = []
    if audio_root is not None:
        candidates.append(Path(audio_root) / raw_path)
    candidates.append(manifest_path.parent / raw_path)
    candidates.append(Path.cwd() / raw_path)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return candidates[0].resolve()


def _issue(
    row_index: int,
    case_id: str,
    field: str,
    message: str,
    severity: str = "error",
) -> LoRAManifestIssue:
    return LoRAManifestIssue(
        row_index=row_index,
        case_id=case_id,
        field=field,
        severity=severity,
        message=message,
    )
