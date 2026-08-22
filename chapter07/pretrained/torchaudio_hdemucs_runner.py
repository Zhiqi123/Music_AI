"""Torchaudio HDemucs bundle runner."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .base import DependencyStatus, Separator, SeparationResult, audio_sample_rate, collect_stem_paths


class TorchaudioHDemucsSeparator(Separator):
    model_name = "torchaudio_hdemucs"
    supported_stems = ("vocals", "drums", "bass", "other")
    package_name = "torchaudio"
    command_name = None

    def __init__(
        self,
        bundle: str = "HDEMUCS_HIGH_MUSDB_PLUS",
        device: str = "cpu",
        overlap: float = 1.0,
    ) -> None:
        self.bundle = bundle
        self.device = device
        self.overlap = float(overlap)

    def dependency_status(self) -> DependencyStatus:
        torch_ok = importlib.util.find_spec("torch") is not None
        torchaudio_ok = importlib.util.find_spec("torchaudio") is not None
        notes = "available" if torch_ok and torchaudio_ok else "Missing torch or torchaudio"
        return DependencyStatus(
            name=self.model_name,
            package_available=torch_ok and torchaudio_ok,
            command_available=False,
            notes=notes,
        )

    def build_command(
        self,
        input_path: Path,
        output_dir: Path,
        stems: str = "4stems",
        segment: float | None = None,
    ) -> list[str]:
        mode = stems.lower().replace("-", "")
        if mode != "4stems":
            raise ValueError("Torchaudio HDemucs bundle supports 4stems in this wrapper")
        script = Path(__file__).with_name("torchaudio_hdemucs_cli.py")
        command = [
            sys.executable,
            str(script),
            str(input_path),
            "--outdir",
            str(output_dir),
            "--bundle",
            self.bundle,
            "--device",
            self.device,
            "--overlap",
            str(self.overlap),
        ]
        if segment is not None:
            command.extend(["--segment", str(float(segment))])
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
            stems=collect_stem_paths(output_dir),
            sample_rate=audio_sample_rate(input_path),
            command=list(command or []),
            notes=f"torchaudio bundle={self.bundle}",
        )
