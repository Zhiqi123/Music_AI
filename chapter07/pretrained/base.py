"""Common interfaces for optional pretrained source-separation runners."""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import shutil
import subprocess
from typing import Sequence

import soundfile as sf


class SeparatorUnavailable(RuntimeError):
    """Raised when an optional separator dependency is not available."""


@dataclass(frozen=True)
class DependencyStatus:
    name: str
    package_available: bool
    command_available: bool
    command_path: str | None = None
    notes: str = ""

    @property
    def available(self) -> bool:
        return self.package_available or self.command_available


@dataclass
class SeparationResult:
    model_name: str
    input_path: Path
    output_dir: Path
    stems: dict[str, Path]
    sample_rate: int
    command: list[str] = field(default_factory=list)
    notes: str = ""

    def as_rows(self) -> list[dict[str, str | int]]:
        """Return rows suitable for a small manifest table."""
        return [
            {
                "model": self.model_name,
                "stem": stem,
                "path": str(path),
                "sample_rate": self.sample_rate,
                "notes": self.notes,
            }
            for stem, path in sorted(self.stems.items())
        ]


class Separator:
    """Base class for command-backed pretrained separators."""

    model_name: str = ""
    supported_stems: tuple[str, ...] = ()
    package_name: str | None = None
    command_name: str | None = None

    def dependency_status(self) -> DependencyStatus:
        return dependency_status(
            self.model_name,
            package_name=self.package_name,
            command_name=self.command_name,
        )

    def build_command(
        self,
        input_path: Path,
        output_dir: Path,
        stems: str = "4stems",
        segment: float | None = None,
    ) -> list[str]:
        raise NotImplementedError

    def separate(
        self,
        input_path: Path,
        output_dir: Path,
        stems: str = "4stems",
        segment: float | None = None,
    ) -> SeparationResult:
        status = self.dependency_status()
        if not status.available:
            raise SeparatorUnavailable(status.notes)

        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        command = self.build_command(input_path, output_dir, stems=stems, segment=segment)
        subprocess.run(command, check=True)
        return self.collect_result(input_path, output_dir, command=command)

    def collect_result(
        self,
        input_path: Path,
        output_dir: Path,
        command: Sequence[str] | None = None,
    ) -> SeparationResult:
        return SeparationResult(
            model_name=self.model_name,
            input_path=Path(input_path),
            output_dir=Path(output_dir),
            stems=collect_stem_paths(output_dir),
            sample_rate=audio_sample_rate(input_path),
            command=list(command or []),
        )


def dependency_status(
    name: str,
    package_name: str | None = None,
    command_name: str | None = None,
) -> DependencyStatus:
    package_available = bool(package_name and importlib.util.find_spec(package_name))
    command_path = shutil.which(command_name) if command_name else None
    command_available = command_path is not None
    if package_available or command_available:
        notes = "available"
    else:
        pieces = []
        if package_name:
            pieces.append(f"Python package {package_name!r}")
        if command_name:
            pieces.append(f"command {command_name!r}")
        notes = "Missing " + " and ".join(pieces)
    return DependencyStatus(
        name=name,
        package_available=package_available,
        command_available=command_available,
        command_path=command_path,
        notes=notes,
    )


def collect_stem_paths(
    output_dir: Path,
    aliases: dict[str, str] | None = None,
    extensions: tuple[str, ...] = (".wav", ".flac", ".mp3"),
) -> dict[str, Path]:
    """Collect separator outputs by filename stem."""
    output_dir = Path(output_dir)
    aliases = aliases or {}
    stems: dict[str, Path] = {}
    if not output_dir.exists():
        return stems
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        name = aliases.get(path.stem.lower(), path.stem.lower())
        stems.setdefault(name, path)
    return stems


def audio_sample_rate(path: Path) -> int:
    return int(sf.info(path).samplerate)


def shell_join(command: Sequence[str]) -> str:
    """Return a readable command string without invoking a shell."""
    return " ".join(_quote_arg(arg) for arg in command)


def _quote_arg(arg: str) -> str:
    arg = str(arg)
    if not arg or any(ch.isspace() for ch in arg):
        return "'" + arg.replace("'", "'\"'\"'") + "'"
    return arg
