"""Optional RoFormer runner placeholder."""
from __future__ import annotations

from pathlib import Path

from .base import SeparationResult, Separator, SeparatorUnavailable


class RoFormerSeparator(Separator):
    model_name = "roformer"
    supported_stems = ("vocals", "drums", "bass", "other")
    package_name = None
    command_name = None

    def __init__(self, checkpoint: Path | None = None, config: Path | None = None) -> None:
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.config = Path(config) if config else None

    def build_command(
        self,
        input_path: Path,
        output_dir: Path,
        stems: str = "4stems",
        segment: float | None = None,
    ) -> list[str]:
        del input_path, output_dir, stems, segment
        raise SeparatorUnavailable(
            "RoFormer is optional in this chapter; provide a concrete package/checkpoint "
            "before enabling inference."
        )

    def separate(
        self,
        input_path: Path,
        output_dir: Path,
        stems: str = "4stems",
        segment: float | None = None,
    ) -> SeparationResult:
        del input_path, output_dir, stems, segment
        raise SeparatorUnavailable(
            "RoFormer runner is a documented extension point, not a default executable path."
        )
