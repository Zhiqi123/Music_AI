"""audio-separator/UVR runner for optional RoFormer-family checkpoints."""
from __future__ import annotations

import sys
from pathlib import Path
import re

from .base import Separator, SeparationResult, audio_sample_rate, collect_stem_paths


_STEM_ALIASES = {
    "instrumental": "accompaniment",
    "no_vocals": "accompaniment",
    "no vocals": "accompaniment",
    "vocals": "vocals",
    "vocal": "vocals",
    "drums": "drums",
    "drum": "drums",
    "bass": "bass",
    "guitar": "guitar",
    "piano": "piano",
    "other": "other",
}


class AudioSeparatorRunner(Separator):
    """Wrapper around the `audio-separator` CLI.

    The default model is a BS-RoFormer vocal/accompaniment checkpoint exposed by
    the audio-separator package. The package downloads model files on demand.
    """

    model_name = "audio_separator_roformer"
    supported_stems = (
        "vocals",
        "accompaniment",
        "drums",
        "bass",
        "guitar",
        "piano",
        "other",
    )
    package_name = "audio_separator"
    command_name = None

    def __init__(
        self,
        model_filename: str = "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        model_file_dir: Path | str | None = None,
        output_format: str = "WAV",
        sample_rate: int = 44100,
        log_level: str = "warning",
    ) -> None:
        self.model_filename = model_filename
        self.model_file_dir = Path(model_file_dir) if model_file_dir else None
        self.output_format = output_format
        self.sample_rate = int(sample_rate)
        self.log_level = log_level

    def build_command(
        self,
        input_path: Path,
        output_dir: Path,
        stems: str = "2stems",
        segment: float | None = None,
    ) -> list[str]:
        del stems
        script = Path(__file__).with_name("audio_separator_cli.py")
        command = [
            sys.executable,
            str(script),
            "--log_level",
            self.log_level,
            "-m",
            self.model_filename,
            "--output_format",
            self.output_format,
            "--output_dir",
            str(output_dir),
            "--sample_rate",
            str(self.sample_rate),
        ]
        if self.model_file_dir is not None:
            command.extend(["--model_file_dir", str(self.model_file_dir)])
        if segment is not None:
            command.extend(["--chunk_duration", str(float(segment))])
        command.append(str(input_path))
        return command

    def build_download_command(self) -> list[str]:
        script = Path(__file__).with_name("audio_separator_cli.py")
        command = [
            sys.executable,
            str(script),
            "--log_level",
            self.log_level,
            "-m",
            self.model_filename,
            "--download_model_only",
        ]
        if self.model_file_dir is not None:
            command.extend(["--model_file_dir", str(self.model_file_dir)])
        return command

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
            stems=_collect_audio_separator_stems(output_dir),
            sample_rate=audio_sample_rate(input_path),
            command=list(command or []),
            notes=f"audio-separator model={self.model_filename}",
        )


def _collect_audio_separator_stems(output_dir: Path) -> dict[str, Path]:
    raw = collect_stem_paths(output_dir)
    normalized: dict[str, Path] = {}
    for name, path in raw.items():
        stem = _stem_from_audio_separator_name(name)
        normalized.setdefault(stem, path)
    return normalized


def _stem_from_audio_separator_name(name: str) -> str:
    clean = str(name).lower().replace("-", "_")
    parenthetical = re.findall(r"\(([^)]+)\)", clean)
    candidates = parenthetical + re.split(r"[_\s]+", clean)
    for candidate in candidates:
        key = candidate.strip().replace("_", " ")
        if key in _STEM_ALIASES:
            return _STEM_ALIASES[key]
        key = candidate.strip().replace(" ", "_")
        if key in _STEM_ALIASES:
            return _STEM_ALIASES[key]
    return clean
