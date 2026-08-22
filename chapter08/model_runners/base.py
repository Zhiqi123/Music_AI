"""Common interfaces for pretrained audio-generation runners."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Mapping
from typing import Sequence

from _common.paths import portable_text
from _common.tables import write_rows


class RunnerState(str, Enum):
    AVAILABLE = "available"
    MISSING_DEPENDENCY = "missing_dependency"
    MISSING_WEIGHTS = "missing_weights"
    LICENSE_REQUIRED = "license_required"
    INSUFFICIENT_MEMORY = "insufficient_memory"
    DISABLED_BY_CONFIG = "disabled_by_config"
    ERROR = "error"


@dataclass(frozen=True)
class RunnerStatus:
    model_name: str
    runner: str
    status: str
    reason: str = ""
    next_action: str = ""
    license_note: str = ""
    estimated_vram_gb: float | None = None

    @property
    def available(self) -> bool:
        return self.status == RunnerState.AVAILABLE.value

    def as_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    prompt_id: str = "default"
    duration_sec: float = 8.0
    output_dir: Path = Path("output_audio")
    seed: int | None = None
    reference_audio: Path | None = None
    extra: dict[str, object] | None = None


@dataclass(frozen=True)
class GenerationResult:
    model_name: str
    prompt_id: str
    output_audio_path: Path | None
    sample_rate: int | None = None
    duration_sec: float | None = None
    wall_time_sec: float | None = None
    device: str = ""
    notes: str = ""

    def as_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExternalCommand:
    """A command-backed model call that can be displayed or executed."""

    command: tuple[str, ...]
    cwd: Path | None = None
    expected_output: Path | None = None
    notes: str = ""

    def shell_command(self) -> str:
        return shell_join(self.command)


class GenerationRunner:
    """Base class for model-specific generation adapters."""

    model_name: str = ""
    runner_name: str = ""
    estimated_vram_gb: float | None = None
    license_note: str = ""

    def check_environment(self) -> RunnerStatus:
        raise NotImplementedError

    def generate(self, request: GenerationRequest) -> GenerationResult:
        raise NotImplementedError


def dependency_available(package_name: str | None = None, command_name: str | None = None) -> bool:
    """Return true if a Python package or command can be resolved."""
    package_ok = bool(package_name and importlib.util.find_spec(package_name))
    command_ok = bool(command_name and shutil.which(command_name))
    return package_ok or command_ok


def dependency_status(
    model_name: str,
    runner: str,
    package_names: Sequence[str] = (),
    command_names: Sequence[str] = (),
    license_note: str = "",
    estimated_vram_gb: float | None = None,
    next_action: str = "",
    requires_license: bool = False,
) -> RunnerStatus:
    """Build a standard status row from dependency probes."""
    missing_packages = [name for name in package_names if not importlib.util.find_spec(name)]
    missing_commands = [name for name in command_names if not shutil.which(name)]
    if requires_license:
        return RunnerStatus(
            model_name=model_name,
            runner=runner,
            status=RunnerState.LICENSE_REQUIRED.value,
            reason="model access may require accepting a license or logging in",
            next_action=next_action,
            license_note=license_note,
            estimated_vram_gb=estimated_vram_gb,
        )
    if missing_packages or missing_commands:
        reason_bits = []
        if missing_packages:
            reason_bits.append("missing packages: " + ", ".join(missing_packages))
        if missing_commands:
            reason_bits.append("missing commands: " + ", ".join(missing_commands))
        return RunnerStatus(
            model_name=model_name,
            runner=runner,
            status=RunnerState.MISSING_DEPENDENCY.value,
            reason="; ".join(reason_bits),
            next_action=next_action,
            license_note=license_note,
            estimated_vram_gb=estimated_vram_gb,
        )
    return RunnerStatus(
        model_name=model_name,
        runner=runner,
        status=RunnerState.AVAILABLE.value,
        reason="dependencies resolved",
        next_action="run the model notebook",
        license_note=license_note,
        estimated_vram_gb=estimated_vram_gb,
    )


def write_runner_status_table(path: Path | str, statuses: Sequence[RunnerStatus]) -> None:
    """Write runner statuses with stable Chapter 8 fields."""
    write_rows(
        path,
        [status.as_row() for status in statuses],
        fieldnames=[
            "model_name",
            "runner",
            "status",
            "reason",
            "next_action",
            "license_note",
            "estimated_vram_gb",
        ],
    )


def shell_join(command: Sequence[str]) -> str:
    """Return a readable shell command without invoking a shell."""
    return " ".join(_quote_arg(str(arg)) for arg in command)


def render_command_template(
    template: Sequence[str],
    values: Mapping[str, object],
) -> tuple[str, ...]:
    """Render ``{placeholder}`` fields in a command template."""
    return tuple(str(part).format(**values) for part in template)


def run_external_command(
    command: Sequence[str],
    cwd: Path | str | None = None,
    timeout: float | None = None,
) -> float:
    """Run a command-backed model call and return wall time in seconds."""
    start = time.perf_counter()
    subprocess.run(list(command), cwd=cwd, timeout=timeout, check=True)
    return time.perf_counter() - start


def resolve_model_source(
    default_model_name: str,
    local_dir: Path | str,
    env_var: str,
) -> str:
    """Return an env override, local model cache, or remote Hugging Face id."""
    override = os.getenv(env_var, "").strip()
    if override:
        return override
    path = Path(local_dir)
    if path.exists():
        return str(path)
    return default_model_name


def model_load_error_message(
    model_name: str,
    exc: BaseException,
    install_hint: str,
    download_hint: str,
) -> str:
    """Return a concise notebook-facing message for Hub/cache/model-file failures."""
    model_label = portable_text(model_name, Path("."))
    return (
        f"Could not load {model_label}.\n"
        f"{portable_text(f'{type(exc).__name__}: {exc}', Path('.'))}\n\n"
        "Check that the model environment is active and local model files are complete. "
        "If the runner is loading from the Hub, also check that huggingface.co is reachable "
        "and any gated model license has been accepted with the same account used by hf auth login.\n"
        f"{install_hint}\n"
        f"{download_hint}"
    )


def _quote_arg(arg: str) -> str:
    if not arg or any(ch.isspace() for ch in arg):
        return "'" + arg.replace("'", "'\"'\"'") + "'"
    return arg
