"""MusicGen runner scaffold."""
from __future__ import annotations

import importlib.util
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import sys
import time
import types

from _common.device_utils import choose_device
from model_runners.base import (
    GenerationRequest,
    GenerationResult,
    GenerationRunner,
    RunnerState,
    RunnerStatus,
    dependency_status,
    model_load_error_message,
    resolve_model_source,
)
from checks.setup_guidance import AUDIOCRAFT_INSTALL_HINT, MUSICGEN_DOWNLOAD_HINT


class MusicGenRunner(GenerationRunner):
    model_name = "facebook/musicgen-small"
    runner_name = "audiocraft.musicgen"
    estimated_vram_gb = 6.0
    license_note = "Check the selected MusicGen checkpoint license before distribution."
    local_model_dir = Path(__file__).resolve().parents[1] / "models" / "facebook_musicgen_small"
    local_t5_dir = Path(__file__).resolve().parents[1] / "models" / "t5_base"
    model_env_var = "CHAPTER08_MUSICGEN_MODEL"
    t5_env_var = "CHAPTER08_T5_MODEL"

    def check_environment(self):
        status = dependency_status(
            self.model_name,
            self.runner_name,
            package_names=("audiocraft", "torch", "numpy", "transformers", "librosa", "soundfile"),
            next_action=AUDIOCRAFT_INSTALL_HINT + " Then run 08_6b_musicgen_inference.ipynb.",
            license_note=self.license_note,
            estimated_vram_gb=self.estimated_vram_gb,
        )
        if not status.available:
            return status
        compatibility_reason = musicgen_dependency_compatibility_reason()
        if compatibility_reason:
            return type(status)(
                model_name=status.model_name,
                runner=status.runner,
                status="missing_dependency",
                reason=compatibility_reason,
                next_action=AUDIOCRAFT_INSTALL_HINT + " Then restart the Notebook kernel.",
                license_note=status.license_note,
                estimated_vram_gb=status.estimated_vram_gb,
            )
        weight_status = self._local_weight_status()
        if weight_status is not None:
            return weight_status
        return status

    def generate(self, request: GenerationRequest) -> GenerationResult:
        status = self.check_environment()
        if not status.available:
            raise RuntimeError(status.reason)
        from _common.audio_io import save_audio

        extra = request.extra or {}
        model, model_name, device, device_note = self._load_model(extra)
        start = time.perf_counter()
        model.set_generation_params(duration=float(request.duration_sec))
        wav = model.generate([request.prompt], progress=True)[0].detach().cpu().numpy()
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{request.prompt_id}_musicgen.wav"
        save_audio(output_path, wav, int(model.sample_rate))
        return GenerationResult(
            model_name=model_name,
            prompt_id=request.prompt_id,
            output_audio_path=output_path,
            sample_rate=int(model.sample_rate),
            duration_sec=request.duration_sec,
            wall_time_sec=time.perf_counter() - start,
            device=device,
            notes=device_note,
        )

    def generate_continuation(self, request: GenerationRequest, audio_prompt_path: Path | str) -> GenerationResult:
        """Generate MusicGen audio conditioned on a real audio prefix."""
        status = self.check_environment()
        if not status.available:
            raise RuntimeError(status.reason)
        from _common.audio_io import load_audio, save_audio

        import torch

        extra = request.extra or {}
        model, model_name, device, device_note = self._load_model(extra)
        prompt_duration = float(extra.get("prompt_duration_sec", 2.0))
        prompt_audio, prompt_sr = load_audio(audio_prompt_path, sr=None, mono=True, duration=prompt_duration)
        prompt_tensor = torch.from_numpy(prompt_audio).float().unsqueeze(0).unsqueeze(0).to(device)
        start = time.perf_counter()
        model.set_generation_params(duration=float(request.duration_sec))
        wav = model.generate_continuation(
            prompt_tensor,
            prompt_sample_rate=int(prompt_sr),
            descriptions=[request.prompt],
            progress=True,
        )[0].detach().cpu().numpy()
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{request.prompt_id}_musicgen_continuation.wav"
        save_audio(output_path, wav, int(model.sample_rate))
        return GenerationResult(
            model_name=model_name,
            prompt_id=request.prompt_id,
            output_audio_path=output_path,
            sample_rate=int(model.sample_rate),
            duration_sec=request.duration_sec,
            wall_time_sec=time.perf_counter() - start,
            device=device,
            notes=device_note,
        )

    def _load_model(self, extra: dict[str, object]):
        """Load MusicGen after applying local T5 and xformers compatibility patches."""
        install_xformers_compat_if_missing()
        try:
            from audiocraft.models import MusicGen
        except Exception as exc:
            raise RuntimeError(
                "Could not import AudioCraft MusicGen in the current kernel.\n"
                f"{type(exc).__name__}: {exc}\n\n"
                "Use the dedicated chapter08-audiocraft environment and the pinned package recipe. "
                "In particular, keep numpy==1.26.4 and transformers==4.31.0 for torch==2.1.0. "
                f"{AUDIOCRAFT_INSTALL_HINT}"
            ) from exc

        device, device_note = choose_musicgen_device(str(extra.get("device", "auto")))
        requested_model = str(extra.get("model_name", self.model_name))
        model_name = resolve_model_source(requested_model, self.local_model_dir, self.model_env_var)
        t5_source = self.resolve_t5_source()
        if Path(t5_source).exists():
            patch_t5_conditioner_sources(t5_source)
        try:
            model = MusicGen.get_pretrained(model_name, device=device)
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                model_load_error_message(
                    model_name,
                    exc,
                    AUDIOCRAFT_INSTALL_HINT,
                    MUSICGEN_DOWNLOAD_HINT,
                )
            ) from exc
        return model, model_name, device, device_note

    def resolve_t5_source(self) -> str:
        override = os.getenv(self.t5_env_var, "").strip()
        if override:
            return override
        if self.local_t5_dir.exists():
            return str(self.local_t5_dir)
        return "t5-base"

    def _local_weight_status(self) -> RunnerStatus | None:
        missing: list[str] = []
        model_source = resolve_model_source(self.model_name, self.local_model_dir, self.model_env_var)
        model_path = Path(model_source)
        if model_path.exists():
            missing_model_files = _missing_required_files(
                model_path,
                required=("state_dict.bin", "compression_state_dict.bin"),
                alternatives=(),
            )
            if missing_model_files:
                missing.append(f"incomplete MusicGen checkpoint: {', '.join(missing_model_files)}")
        elif not os.getenv(self.model_env_var, "").strip():
            missing.append("missing MusicGen checkpoint directory: chapter08/models/facebook_musicgen_small")

        t5_source = self.resolve_t5_source()
        t5_path = Path(t5_source)
        if t5_path.exists():
            missing_t5_files = _missing_required_files(
                t5_path,
                required=("config.json",),
                alternatives=(("model.safetensors", "pytorch_model.bin"), ("spiece.model", "tokenizer.json")),
            )
            if missing_t5_files:
                missing.append(f"incomplete T5 text encoder: {', '.join(missing_t5_files)}")
        elif not os.getenv(self.t5_env_var, "").strip():
            missing.append("missing T5 text encoder directory: chapter08/models/t5_base")

        if not missing:
            return None
        return RunnerStatus(
            model_name=self.model_name,
            runner=self.runner_name,
            status=RunnerState.MISSING_WEIGHTS.value,
            reason="; ".join(missing),
            next_action=MUSICGEN_DOWNLOAD_HINT,
            license_note=self.license_note,
            estimated_vram_gb=self.estimated_vram_gb,
        )


def musicgen_dependency_compatibility_reason() -> str:
    """Return a concise reason when installed packages are known incompatible."""
    try:
        numpy_version = version("numpy")
        transformers_version = version("transformers")
        torch_version = version("torch")
    except PackageNotFoundError:
        return ""

    problems: list[str] = []
    if _major_version(numpy_version) >= 2:
        problems.append(f"numpy {numpy_version} is installed; use numpy==1.26.4")
    if not transformers_version.startswith("4.31."):
        problems.append(
            f"transformers {transformers_version} is installed; use transformers==4.31.0 with torch {torch_version}"
        )
    if problems:
        return "; ".join(problems)
    return ""


def choose_musicgen_device(requested: str = "auto") -> tuple[str, str]:
    """Return an AudioCraft-compatible device.

    The shared chapter policy still detects devices as cuda -> mps -> cpu, but
    AudioCraft 1.3.0 can hit ``torch.autocast(device_type="mps")`` under
    torch 2.1, which raises at model load time. Use CPU instead of failing on
    Apple MPS machines.
    """
    device = choose_device(requested)
    if device == "mps":
        return (
            "cpu",
            "AudioCraft/MusicGen under torch 2.1 does not support MPS autocast; this runner used CPU instead.",
        )
    return device, ""


def _major_version(text: str) -> int:
    try:
        return int(text.split(".", 1)[0])
    except (TypeError, ValueError):
        return 0


def patch_t5_conditioner_sources(local_t5_dir: Path | str, model_name: str = "t5-base") -> bool:
    """Redirect AudioCraft's T5 conditioner to a local text-encoder directory."""
    path = Path(local_t5_dir)
    if not path.exists():
        return False

    from audiocraft.modules import conditioners

    marker = str(path.resolve())
    if getattr(conditioners, "_chapter08_t5_source", None) == marker:
        return True

    if not hasattr(conditioners, "_chapter08_original_t5_tokenizer_from_pretrained"):
        conditioners._chapter08_original_t5_tokenizer_from_pretrained = conditioners.T5Tokenizer.from_pretrained
    if not hasattr(conditioners, "_chapter08_original_t5_encoder_from_pretrained"):
        conditioners._chapter08_original_t5_encoder_from_pretrained = conditioners.T5EncoderModel.from_pretrained

    original_tokenizer_from_pretrained = conditioners._chapter08_original_t5_tokenizer_from_pretrained
    original_encoder_from_pretrained = conditioners._chapter08_original_t5_encoder_from_pretrained

    def _source_name(name: str) -> str:
        return marker if name == model_name else name

    def _tokenizer_from_pretrained(cls, name: str, *args, **kwargs):
        if name == model_name:
            kwargs.setdefault("local_files_only", True)
        return original_tokenizer_from_pretrained(_source_name(name), *args, **kwargs)

    def _encoder_from_pretrained(cls, name: str, *args, **kwargs):
        if name == model_name:
            kwargs.setdefault("local_files_only", True)
        return original_encoder_from_pretrained(_source_name(name), *args, **kwargs)

    conditioners.T5Tokenizer.from_pretrained = classmethod(_tokenizer_from_pretrained)
    conditioners.T5EncoderModel.from_pretrained = classmethod(_encoder_from_pretrained)
    conditioners._chapter08_t5_source = marker
    return True


def _missing_required_files(
    path: Path,
    required: tuple[str, ...],
    alternatives: tuple[tuple[str, ...], ...],
) -> list[str]:
    missing = [name for name in required if not (path / name).exists()]
    for names in alternatives:
        if not any((path / name).exists() for name in names):
            missing.append(" or ".join(names))
    return missing


def install_xformers_compat_if_missing() -> bool:
    """Install a minimal xformers shim for AudioCraft CPU/MPS inference.

    AudioCraft 1.3.0 imports ``from xformers import ops`` at module import time
    even when its default transformer config uses PyTorch attention. Some local
    teaching environments cannot install the pinned xformers release. This shim
    is intentionally tiny: it supports the non-memory-efficient path and raises
    a clear error if a true xformers attention kernel is requested.
    """
    if "xformers" in sys.modules:
        return False
    try:
        xformers_spec = importlib.util.find_spec("xformers")
    except ValueError:
        xformers_spec = None
    if xformers_spec is not None:
        return False

    import torch

    xformers_module = types.ModuleType("xformers")
    ops_module = types.ModuleType("xformers.ops")

    class LowerTriangularMask:
        """Placeholder used only before memory-efficient attention is called."""

    def memory_efficient_attention(*args, **kwargs):
        raise RuntimeError(
            "xformers is not installed. Run MusicGen with AudioCraft's default "
            "PyTorch attention path, or use a CUDA environment where a real "
            "xformers build is supported for memory_efficient=True."
        )

    ops_module.unbind = torch.unbind
    ops_module.LowerTriangularMask = LowerTriangularMask
    ops_module.memory_efficient_attention = memory_efficient_attention
    xformers_module.ops = ops_module
    sys.modules["xformers"] = xformers_module
    sys.modules["xformers.ops"] = ops_module
    return True
