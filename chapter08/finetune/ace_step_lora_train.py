"""ACE-Step LoRA training command builder."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from _common.config import load_yaml_config
from _common.dataset_registry import asset_status
from _common.tables import write_rows
from model_runners.ace_step import ACEStepRunner


CHAPTER_ROOT = Path(__file__).resolve().parents[1]

LORA_PLAN_FIELDS = [
    "pathway",
    "asset_available",
    "runner_status",
    "runner_reason",
    "config_path",
    "manifest_csv",
    "audio_root",
    "sample_rate",
    "rank",
    "alpha",
    "batch_size",
    "max_steps",
    "checkpoint_dir",
    "train_command",
    "detail",
]


def build_lora_train_command(config_path: Path | str = "configs/ace_step_lora.yaml") -> list[str]:
    """Return the command used once ACE-Step's training package is installed."""
    return ["python", "-m", "ace_step.train_lora", "--config", str(config_path)]


def build_lora_train_plan(
    config: dict[str, Any] | None = None,
    config_path: Path | str = "configs/ace_step_lora.yaml",
    chapter_root: Path | str = CHAPTER_ROOT,
    project_root: Path | str | None = None,
) -> list[dict[str, object]]:
    """Return the table row describing the ACE-Step LoRA training path."""
    chapter_root = Path(chapter_root)
    root = Path(project_root) if project_root is not None else chapter_root.resolve().parents[1]
    config = config or load_yaml_config(resolve_config_path(config_path, chapter_root))
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    output_cfg = config.get("outputs", {})
    asset = asset_status("audio_author_ch08", project_root=root)
    runner_status = ACEStepRunner().check_environment()
    detail = "ready to validate manifest and run training command"
    if not asset.ok:
        detail = asset.spec.download_hint
    elif not runner_status.available:
        detail = runner_status.next_action or runner_status.reason
    return [
        {
            "pathway": "ace_step_lora_personalization",
            "asset_available": asset.ok,
            "runner_status": runner_status.status,
            "runner_reason": runner_status.reason,
            "config_path": str(config_path),
            "manifest_csv": data_cfg.get("manifest_csv", ""),
            "audio_root": data_cfg.get("audio_root", ""),
            "sample_rate": data_cfg.get("sample_rate", ""),
            "rank": training_cfg.get("rank", ""),
            "alpha": training_cfg.get("alpha", ""),
            "batch_size": training_cfg.get("batch_size", ""),
            "max_steps": training_cfg.get("max_steps", ""),
            "checkpoint_dir": output_cfg.get("checkpoint_dir", ""),
            "train_command": " ".join(build_lora_train_command(config_path)),
            "detail": detail,
        }
    ]


def write_lora_train_plan(
    output_csv: Path | str | None = None,
    config: dict[str, Any] | None = None,
    config_path: Path | str = "configs/ace_step_lora.yaml",
    chapter_root: Path | str = CHAPTER_ROOT,
    project_root: Path | str | None = None,
) -> list[dict[str, object]]:
    """Write and return the ACE-Step LoRA training plan table."""
    chapter_root = Path(chapter_root)
    config = config or load_yaml_config(resolve_config_path(config_path, chapter_root))
    output_csv = output_csv or chapter_root / config.get("outputs", {}).get(
        "table_csv", "outputs/tables/08_8_lora_plan.csv"
    )
    rows = build_lora_train_plan(
        config=config,
        config_path=config_path,
        chapter_root=chapter_root,
        project_root=project_root,
    )
    write_rows(output_csv, rows, fieldnames=LORA_PLAN_FIELDS)
    return rows


def resolve_config_path(config_path: Path | str, chapter_root: Path | str = CHAPTER_ROOT) -> Path:
    """Resolve a LoRA config path against the Chapter 8 root."""
    path = Path(config_path)
    if path.is_absolute():
        return path
    return Path(chapter_root) / path


def lora_not_ready_message() -> str:
    """Message shown by the Notebook before external ACE-Step setup is complete."""
    return (
        "ACE-Step LoRA requires the official ACE-Step training environment, local source "
        "under external/ACE-Step, and pretrained weights under models/ace_step_v1_3_5b. "
        "Prepare those assets, then run the command from build_lora_train_command()."
    )
