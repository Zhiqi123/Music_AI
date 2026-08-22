"""Spleeter command-line runner for optional Chapter 7 comparisons."""
from __future__ import annotations

from pathlib import Path

from .base import Separator, SeparationResult, audio_sample_rate, collect_stem_paths


class SpleeterSeparator(Separator):
    model_name = "spleeter"
    supported_stems = ("vocals", "accompaniment", "drums", "bass", "piano", "other")
    package_name = "spleeter"
    command_name = "spleeter"

    def __init__(self, profile: str = "spleeter:2stems", codec: str = "wav") -> None:
        self.profile = profile
        self.codec = codec

    def build_command(
        self,
        input_path: Path,
        output_dir: Path,
        stems: str = "2stems",
        segment: float | None = None,
    ) -> list[str]:
        del segment
        profile = self.profile
        if stems:
            mode = stems.lower().replace("-", "")
            profile = {
                "2stems": "spleeter:2stems",
                "4stems": "spleeter:4stems",
                "5stems": "spleeter:5stems",
            }.get(mode, profile)
        return [
            "spleeter",
            "separate",
            "-p",
            profile,
            "-o",
            str(output_dir),
            "-c",
            self.codec,
            str(input_path),
        ]

    def collect_result(
        self,
        input_path: Path,
        output_dir: Path,
        command: list[str] | None = None,
    ) -> SeparationResult:
        return SeparationResult(
            model_name=self.model_name,
            input_path=Path(input_path),
            output_dir=Path(output_dir),
            stems=collect_stem_paths(output_dir, aliases={"accompaniment": "accompaniment"}),
            sample_rate=audio_sample_rate(input_path),
            command=list(command or []),
            notes=f"spleeter profile={self.profile}",
        )
