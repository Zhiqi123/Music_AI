"""Tiny end-to-end training checks for Chapter 8 model entry points."""
from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from _common.tables import write_rows
from codec_lm.train import train_codec_lm
from wavenet.train import train_wavenet


CHAPTER_ROOT = Path(__file__).resolve().parents[1]

TRAINING_SMOKE_FIELDS = [
    "check_id",
    "model",
    "status",
    "train_loss",
    "output_checkpoint",
    "history_csv",
    "history_rows",
    "detail",
]

EXPECTED_TRAINING_SMOKE_CHECKS = (
    "wavenet_synthetic_1step",
    "codec_lm_random_1step",
)


def build_training_smoke_rows(chapter_root: Path | str = CHAPTER_ROOT) -> list[dict[str, object]]:
    """Run all local training smoke checks and return table rows."""
    root = Path(chapter_root).resolve()
    return [
        run_wavenet_smoke(root),
        run_codec_lm_smoke(root),
    ]


def write_training_smoke(
    output_csv: Path | str = CHAPTER_ROOT / "outputs" / "tables" / "08_training_smoke.csv",
    chapter_root: Path | str = CHAPTER_ROOT,
) -> list[dict[str, object]]:
    """Run smoke checks and write the summary CSV."""
    rows = build_training_smoke_rows(chapter_root)
    write_rows(output_csv, rows, fieldnames=TRAINING_SMOKE_FIELDS)
    return rows


def training_smoke_failures(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return smoke rows that did not complete successfully."""
    return [row for row in rows if row.get("status") != "ok"]


def run_wavenet_smoke(chapter_root: Path) -> dict[str, object]:
    """Train toy WaveNet for one synthetic optimizer step."""
    history_csv = chapter_root / "outputs" / "tables" / "08_training_smoke_wavenet_history.csv"
    checkpoint = chapter_root / "outputs" / "checkpoints" / "smoke_wavenet" / "last.pt"
    config: dict[str, Any] = {
        "seed": 1308,
        "device": "cpu",
        "data": {
            "source": "synthetic",
            "num_examples": 4,
            "sample_rate": 8000,
            "window_samples": 64,
            "quantization_channels": 32,
        },
        "model": {
            "residual_channels": 8,
            "dilation_channels": 8,
            "skip_channels": 16,
            "kernel_size": 2,
            "dilation_cycles": 1,
            "layers_per_cycle": 3,
        },
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 1e-3,
            "max_steps_per_epoch": 1,
        },
        "outputs": {
            "checkpoint_dir": str(checkpoint.parent),
            "history_csv": str(history_csv),
        },
    }
    return run_training_case(
        check_id="wavenet_synthetic_1step",
        model="toy_wavenet",
        train_fn=train_wavenet,
        config=config,
        checkpoint=checkpoint,
        history_csv=history_csv,
        chapter_root=chapter_root,
    )


def run_codec_lm_smoke(chapter_root: Path) -> dict[str, object]:
    """Train mini Codec-LM for one random-token optimizer step."""
    history_csv = chapter_root / "outputs" / "tables" / "08_training_smoke_codec_lm_history.csv"
    checkpoint = chapter_root / "outputs" / "checkpoints" / "smoke_codec_lm" / "last.pt"
    config: dict[str, Any] = {
        "seed": 2308,
        "device": "cpu",
        "data": {
            "source": "random",
            "num_examples": 4,
        },
        "model": {
            "vocab_size": 32,
            "d_model": 16,
            "n_heads": 4,
            "n_layers": 1,
            "max_length": 16,
        },
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 1e-3,
            "max_steps_per_epoch": 1,
        },
        "outputs": {
            "checkpoint_dir": str(checkpoint.parent),
            "history_csv": str(history_csv),
        },
    }
    return run_training_case(
        check_id="codec_lm_random_1step",
        model="mini_codec_lm",
        train_fn=train_codec_lm,
        config=config,
        checkpoint=checkpoint,
        history_csv=history_csv,
        chapter_root=chapter_root,
    )


def run_training_case(
    check_id: str,
    model: str,
    train_fn: Callable[[dict[str, Any]], list[dict[str, float | int | str]]],
    config: dict[str, Any],
    checkpoint: Path,
    history_csv: Path,
    chapter_root: Path,
) -> dict[str, object]:
    """Run one tiny training job and normalize its result."""
    try:
        torch.manual_seed(int(config.get("seed", 0)))
        history = train_fn(config)
        train_loss = latest_loss(history)
        history_rows = count_csv_rows(history_csv)
        status, detail = validate_training_outputs(train_loss, checkpoint, history_csv, history_rows)
    except Exception as exc:  # pragma: no cover - failure details are surfaced in the row.
        train_loss = ""
        history_rows = count_csv_rows(history_csv) if history_csv.exists() else ""
        status = "failed"
        detail = f"{type(exc).__name__}: {exc}"
    return {
        "check_id": check_id,
        "model": model,
        "status": status,
        "train_loss": round(float(train_loss), 6) if isinstance(train_loss, float) else train_loss,
        "output_checkpoint": relative_to_root(checkpoint, chapter_root),
        "history_csv": relative_to_root(history_csv, chapter_root),
        "history_rows": history_rows,
        "detail": detail,
    }


def latest_loss(history: list[dict[str, float | int | str]]) -> float | None:
    """Return the final train loss from a training history."""
    if not history:
        return None
    value = history[-1].get("train_loss")
    if value is None:
        return None
    return float(value)


def validate_training_outputs(
    train_loss: float | None,
    checkpoint: Path,
    history_csv: Path,
    history_rows: int,
) -> tuple[str, str]:
    """Validate loss, checkpoint, and history artifacts for one smoke run."""
    if train_loss is None:
        return "invalid", "training returned no train_loss"
    if not math.isfinite(train_loss) or train_loss <= 0:
        return "invalid", f"non-positive or non-finite train_loss={train_loss}"
    if not checkpoint.exists() or checkpoint.stat().st_size <= 0:
        return "invalid", f"checkpoint missing or empty: {checkpoint}"
    if not history_csv.exists() or history_rows < 1:
        return "invalid", f"history CSV missing or empty: {history_csv}"
    return "ok", "one optimizer step, history row, and checkpoint verified"


def count_csv_rows(path: Path) -> int:
    """Count data rows in a CSV file."""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def relative_to_root(path: Path, root: Path) -> str:
    """Return a stable display path relative to the chapter root when possible."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chapter 8 tiny training smoke checks.")
    parser.add_argument(
        "--output-csv",
        default=str(CHAPTER_ROOT / "outputs" / "tables" / "08_training_smoke.csv"),
    )
    parser.add_argument("--chapter-root", default=str(CHAPTER_ROOT))
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args()

    rows = write_training_smoke(args.output_csv, args.chapter_root)
    for row in rows:
        print(row)
    failures = training_smoke_failures(rows)
    if failures and not args.no_strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
