"""Demucs command-line runner with a Chapter 7 compatible interface."""
from __future__ import annotations

import sys
from pathlib import Path

from .base import Separator, SeparationResult, audio_sample_rate, collect_stem_paths


class DemucsSeparator(Separator):
    model_name = "demucs"
    supported_stems = ("vocals", "drums", "bass", "other", "guitar", "piano", "accompaniment")
    package_name = "demucs"
    command_name = "demucs"

    def __init__(self, model: str = "htdemucs", device: str = "cpu", jobs: int = 0) -> None:
        self.model = model
        self.device = device
        self.jobs = int(jobs)

    def build_command(
        self,
        input_path: Path,
        output_dir: Path,
        stems: str = "4stems",
        segment: float | None = None,
    ) -> list[str]:
        mode = _normalize_mode(stems)
        model = self.model
        if mode == "6stems" and model == "htdemucs":
            model = "htdemucs_6s"

        command = [
            sys.executable,
            "-m",
            "demucs.separate",
            "-n",
            model,
            "-o",
            str(output_dir),
            "--filename",
            "{track}/{stem}.{ext}",
            "-d",
            self.device,
            "--float32",
        ]
        if self.jobs > 0:
            command.extend(["-j", str(self.jobs)])
        if segment is not None:
            command.extend(["--segment", str(float(segment))])
        if mode == "2stems":
            command.extend(["--two-stems", "vocals"])
        command.append(str(input_path))
        return command

    def collect_result(
        self,
        input_path: Path,
        output_dir: Path,
        command: list[str] | None = None,
    ) -> SeparationResult:
        stems = collect_stem_paths(output_dir, aliases={"no_vocals": "accompaniment"})
        return SeparationResult(
            model_name=self.model_name,
            input_path=Path(input_path),
            output_dir=Path(output_dir),
            stems=stems,
            sample_rate=audio_sample_rate(input_path),
            command=list(command or []),
            notes=f"demucs model={self.model}",
        )


def _normalize_mode(stems: str) -> str:
    mode = stems.lower().replace("-", "")
    if mode not in {"2stems", "4stems", "6stems"}:
        raise ValueError("Demucs stems must be one of: 2stems, 4stems, 6stems")
    return mode
