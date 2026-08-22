"""Open-Unmix command-line runner with a Chapter 7 compatible interface."""
from __future__ import annotations

import sys
from pathlib import Path

from .base import Separator, SeparationResult, audio_sample_rate, collect_stem_paths


class OpenUnmixSeparator(Separator):
    model_name = "openunmix"
    supported_stems = ("vocals", "drums", "bass", "other", "accompaniment")
    package_name = "openunmix"
    command_name = None

    def __init__(
        self,
        model: str = "umxhq",
        targets: tuple[str, ...] = ("vocals", "drums", "bass", "other"),
        no_cuda: bool = True,
        niter: int = 1,
    ) -> None:
        self.model = model
        self.targets = targets
        self.no_cuda = no_cuda
        self.niter = int(niter)

    def build_command(
        self,
        input_path: Path,
        output_dir: Path,
        stems: str = "4stems",
        segment: float | None = None,
    ) -> list[str]:
        mode = stems.lower().replace("-", "")
        if mode not in {"4stems", "2stems"}:
            raise ValueError("Open-Unmix supports 4stems or aggregated 2stems in this wrapper")

        script = Path(__file__).with_name("openunmix_cli.py")
        command = [
            sys.executable,
            str(script),
            str(input_path),
            "--model",
            self.model,
            "--outdir",
            str(output_dir),
            "--ext",
            ".wav",
            "--niter",
            str(self.niter),
        ]
        if self.no_cuda:
            command.append("--no-cuda")
        if mode == "2stems":
            command.extend(
                [
                    "--aggregate",
                    '{"vocals":["vocals"],"accompaniment":["drums","bass","other"]}',
                ]
            )
        elif self.targets:
            command.extend(["--targets", *self.targets])
        if segment is not None:
            command.extend(["--duration", str(float(segment))])
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
            notes=f"openunmix model={self.model}",
        )
