"""MusicGen fine-tuning preparation and command templates."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

from _common.config import load_yaml_config
from _common.dataset_registry import asset_status
from _common.paths import command_relative_path, portable_path
from _common.tables import write_rows
from finetune.ace_step_lora_dataset import resolve_audio_path
from checks.setup_guidance import AUDIOCRAFT_FINETUNE_HINT


CHAPTER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CHAPTER_ROOT.parents[1]

MUSICGEN_REQUIRED_COLUMNS = (
    "case_id",
    "path",
    "license_status",
    "can_redistribute",
)

MUSICGEN_ISSUE_FIELDS = ["row_index", "case_id", "field", "severity", "message"]

MUSICGEN_PLAN_FIELDS = [
    "pathway",
    "asset_available",
    "audiocraft_available",
    "dora_available",
    "manifest_csv",
    "audio_root",
    "train_jsonl",
    "valid_jsonl",
    "checkpoint_dir",
    "train_command",
    "detail",
]


@dataclass(frozen=True)
class MusicGenFineTuneItem:
    case_id: str
    path: Path
    description: str
    split: str = "train"
    duration_sec: float | None = None
    license_status: str = ""
    can_redistribute: str = ""

    def as_audiocraft_row(self) -> dict[str, object]:
        """Return a JSONL row suitable for a MusicGen training manifest."""
        row: dict[str, object] = {
            "key": self.case_id,
            "path": command_relative_path(self.path, CHAPTER_ROOT),
            "description": self.description,
            "split": self.split,
        }
        if self.duration_sec is not None:
            row["duration"] = self.duration_sec
        return row


@dataclass(frozen=True)
class MusicGenManifestIssue:
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


def validate_musicgen_manifest(
    path: Path | str,
    audio_root: Path | str | None = None,
    require_files: bool = True,
) -> list[MusicGenManifestIssue]:
    """Validate the audio/text manifest used for MusicGen fine-tuning."""
    path = Path(path)
    issues: list[MusicGenManifestIssue] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        for column in MUSICGEN_REQUIRED_COLUMNS:
            if column not in fieldnames:
                issues.append(_issue(0, "", column, "required column is missing"))
        has_caption = "caption" in fieldnames or "prompt_en" in fieldnames or "prompt_cn" in fieldnames
        if not has_caption:
            issues.append(_issue(0, "", "caption", "caption, prompt_en, or prompt_cn is required"))
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

            if not row_caption(row):
                issues.append(_issue(row_index, case_id, "caption", "caption text is empty"))

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


def load_musicgen_manifest(
    path: Path | str,
    audio_root: Path | str | None = None,
    default_split: str = "train",
) -> list[MusicGenFineTuneItem]:
    """Load a MusicGen fine-tuning manifest into typed items."""
    path = Path(path)
    items: list[MusicGenFineTuneItem] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_path = row.get("path", "").strip()
            audio_path = resolve_audio_path(raw_path, manifest_path=path, audio_root=audio_root)
            items.append(
                MusicGenFineTuneItem(
                    case_id=row.get("case_id", audio_path.stem).strip(),
                    path=audio_path,
                    description=row_caption(row),
                    split=(row.get("split") or default_split).strip() or default_split,
                    duration_sec=parse_float(row.get("duration_sec", "")),
                    license_status=row.get("license_status", ""),
                    can_redistribute=row.get("can_redistribute", ""),
                )
            )
    return items


def export_audiocraft_jsonl(
    items: Iterable[MusicGenFineTuneItem],
    output_jsonl: Path | str,
    split: str = "train",
) -> list[dict[str, object]]:
    """Write one AudioCraft-style JSONL file for a split."""
    output_jsonl = Path(output_jsonl)
    rows = [item.as_audiocraft_row() for item in items if item.split == split]
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def build_musicgen_finetune_command(config: dict[str, Any] | None = None) -> list[str]:
    """Return the AudioCraft/Dora command template for MusicGen fine-tuning."""
    config = config or load_yaml_config(CHAPTER_ROOT / "configs" / "musicgen_finetune.yaml")
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    output_cfg = config.get("outputs", {})
    solver = str(training_cfg.get("solver", "musicgen/musicgen_base_32khz"))
    model_scale = str(training_cfg.get("model_scale", "small"))
    return [
        "dora",
        "run",
        f"solver={solver}",
        f"model/lm/model_scale={model_scale}",
        f"continue_from={config.get('base_model', 'facebook/musicgen-small')}",
        f"dset.train={data_cfg.get('train_jsonl', 'outputs/generated/musicgen_finetune/train.jsonl')}",
        f"dset.valid={data_cfg.get('valid_jsonl', 'outputs/generated/musicgen_finetune/valid.jsonl')}",
        f"optim.batch_size={training_cfg.get('batch_size', 1)}",
        f"optim.lr={training_cfg.get('learning_rate', 1e-4)}",
        f"optim.max_steps={training_cfg.get('max_steps', 1000)}",
        f"checkpoint.save_folder={output_cfg.get('checkpoint_dir', 'outputs/checkpoints/musicgen_finetune')}",
    ]


def build_musicgen_finetune_plan(
    config: dict[str, Any] | None = None,
    chapter_root: Path | str = CHAPTER_ROOT,
    project_root: Path | str | None = None,
) -> list[dict[str, object]]:
    """Return the table row describing the MusicGen fine-tuning path."""
    chapter_root = Path(chapter_root)
    root = Path(project_root) if project_root is not None else chapter_root.resolve().parents[1]
    config = config or load_yaml_config(chapter_root / "configs" / "musicgen_finetune.yaml")
    data_cfg = config.get("data", {})
    output_cfg = config.get("outputs", {})
    asset = asset_status("audio_author_ch08", project_root=root)
    audiocraft_available = importlib.util.find_spec("audiocraft") is not None
    dora_available = shutil.which("dora") is not None
    detail = "ready to export manifests and run command"
    if not asset.ok:
        detail = asset.spec.download_hint
    elif not audiocraft_available or not dora_available:
        detail = AUDIOCRAFT_FINETUNE_HINT
    return [
        {
            "pathway": "musicgen_finetune_reference",
            "asset_available": asset.ok,
            "audiocraft_available": audiocraft_available,
            "dora_available": dora_available,
            "manifest_csv": data_cfg.get("manifest_csv", ""),
            "audio_root": data_cfg.get("audio_root", ""),
            "train_jsonl": data_cfg.get("train_jsonl", ""),
            "valid_jsonl": data_cfg.get("valid_jsonl", ""),
            "checkpoint_dir": output_cfg.get("checkpoint_dir", ""),
            "train_command": " ".join(build_musicgen_finetune_command(config)),
            "detail": detail,
        }
    ]


def write_musicgen_finetune_plan(
    output_csv: Path | str | None = None,
    config: dict[str, Any] | None = None,
    chapter_root: Path | str = CHAPTER_ROOT,
    project_root: Path | str | None = None,
) -> list[dict[str, object]]:
    """Write and return the MusicGen fine-tuning plan table."""
    chapter_root = Path(chapter_root)
    config = config or load_yaml_config(chapter_root / "configs" / "musicgen_finetune.yaml")
    output_csv = output_csv or chapter_root / config.get("outputs", {}).get(
        "table_csv", "outputs/tables/08_8_musicgen_finetune_plan.csv"
    )
    rows = build_musicgen_finetune_plan(config, chapter_root=chapter_root, project_root=project_root)
    write_rows(output_csv, rows, fieldnames=MUSICGEN_PLAN_FIELDS)
    return rows


def row_caption(row: dict[str, str]) -> str:
    """Return the preferred text condition for a manifest row."""
    return (
        row.get("caption", "").strip()
        or row.get("prompt_en", "").strip()
        or row.get("prompt_cn", "").strip()
    )


def parse_float(value: str) -> float | None:
    """Parse a float field, returning None for blank values."""
    value = str(value).strip()
    if not value:
        return None
    return float(value)


def _issue(
    row_index: int,
    case_id: str,
    field: str,
    message: str,
    severity: str = "error",
) -> MusicGenManifestIssue:
    return MusicGenManifestIssue(
        row_index=row_index,
        case_id=case_id,
        field=field,
        severity=severity,
        message=message,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MusicGen fine-tuning manifests.")
    parser.add_argument("--config", default=str(CHAPTER_ROOT / "configs" / "musicgen_finetune.yaml"))
    parser.add_argument("--manifest")
    parser.add_argument("--audio-root")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--no-require-files", action="store_true")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    data_cfg = config.get("data", {})
    manifest = Path(args.manifest or data_cfg.get("manifest_csv", ""))
    if not manifest.is_absolute():
        manifest = CHAPTER_ROOT / manifest
    audio_root = args.audio_root or data_cfg.get("audio_root")
    audio_root_path = Path(audio_root) if audio_root else None
    if audio_root_path is not None and not audio_root_path.is_absolute():
        audio_root_path = CHAPTER_ROOT / audio_root_path

    plan_rows = write_musicgen_finetune_plan(config=config, chapter_root=CHAPTER_ROOT)
    print(plan_rows[0])
    if args.export:
        issues = validate_musicgen_manifest(
            manifest,
            audio_root=audio_root_path,
            require_files=not args.no_require_files,
        )
        if any(issue.severity == "error" for issue in issues):
            for issue in issues:
                print(issue.as_row())
            raise SystemExit(1)
        items = load_musicgen_manifest(manifest, audio_root=audio_root_path)
        export_audiocraft_jsonl(items, CHAPTER_ROOT / data_cfg["train_jsonl"], split="train")
        export_audiocraft_jsonl(items, CHAPTER_ROOT / data_cfg["valid_jsonl"], split="valid")


if __name__ == "__main__":
    main()
