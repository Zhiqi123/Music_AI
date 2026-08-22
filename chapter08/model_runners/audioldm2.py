"""AudioLDM2 runner scaffold."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import time

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
from checks.setup_guidance import AUDIOLDM2_DOWNLOAD_HINT, AUDIOLDM2_INSTALL_HINT


class AudioLDM2Runner(GenerationRunner):
    model_name = "cvssp/audioldm2"
    runner_name = "diffusers.audioldm2"
    estimated_vram_gb = 8.0
    license_note = "Check the AudioLDM2 model card before publishing generated examples."
    local_model_dir = Path(__file__).resolve().parents[1] / "models" / "cvssp_audioldm2"
    model_env_var = "CHAPTER08_AUDIOLDM2_MODEL"

    def check_environment(self):
        status = dependency_status(
            self.model_name,
            self.runner_name,
            package_names=(
                "diffusers",
                "transformers",
                "accelerate",
                "torch",
                "huggingface_hub",
                "safetensors",
                "librosa",
                "soundfile",
            ),
            next_action=AUDIOLDM2_INSTALL_HINT + " Then rerun 08_6c_audioldm2_inference.ipynb.",
            license_note=self.license_note,
            estimated_vram_gb=self.estimated_vram_gb,
        )
        if not status.available:
            return status
        compatibility_reason = audioldm2_dependency_compatibility_reason()
        if compatibility_reason:
            return RunnerStatus(
                model_name=self.model_name,
                runner=self.runner_name,
                status=RunnerState.MISSING_DEPENDENCY.value,
                reason=compatibility_reason,
                next_action=AUDIOLDM2_INSTALL_HINT + " Then rerun 08_6c_audioldm2_inference.ipynb.",
                license_note=self.license_note,
                estimated_vram_gb=self.estimated_vram_gb,
            )
        return status

    def generate(self, request: GenerationRequest) -> GenerationResult:
        status = self.check_environment()
        if not status.available:
            raise RuntimeError(status.reason)
        from _common.audio_io import save_audio

        import torch
        try:
            from diffusers import AudioLDM2Pipeline
        except ImportError as exc:
            raise RuntimeError(
                "The installed diffusers package does not expose AudioLDM2Pipeline.\n"
                + AUDIOLDM2_INSTALL_HINT
            ) from exc

        extra = request.extra or {}
        device = choose_device(str(extra.get("device", "auto")))
        requested_model = str(extra.get("model_name", self.model_name))
        model_name = resolve_model_source(requested_model, self.local_model_dir, self.model_env_var)
        num_inference_steps = int(extra.get("num_inference_steps", 25))
        guidance_scale = float(extra.get("guidance_scale", 3.5))
        negative_prompt = str(extra.get("negative_prompt", ""))
        sample_rate = int(extra.get("sample_rate", 16000))
        dtype = torch.float16 if device == "cuda" else torch.float32
        start = time.perf_counter()
        try:
            pipe = AudioLDM2Pipeline.from_pretrained(model_name, torch_dtype=dtype)
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                model_load_error_message(
                    requested_model,
                    exc,
                    AUDIOLDM2_INSTALL_HINT,
                    AUDIOLDM2_DOWNLOAD_HINT,
                )
            ) from exc
        pipe = pipe.to(device)
        try:
            output = pipe(
                request.prompt,
                num_inference_steps=num_inference_steps,
                audio_length_in_s=float(request.duration_sec),
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt or None,
            )
        except AttributeError as exc:
            if "_get_initial_cache_position" in str(exc):
                raise RuntimeError(
                    "AudioLDM2 failed because the installed Transformers version is incompatible. "
                    "Use transformers==4.49.0 in venv_ch08_diffusers.\n"
                    + AUDIOLDM2_INSTALL_HINT
                ) from exc
            raise
        audio = output.audios[0]
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{request.prompt_id}_audioldm2.wav"
        save_audio(output_path, audio, sample_rate)
        return GenerationResult(
            model_name=model_name,
            prompt_id=request.prompt_id,
            output_audio_path=output_path,
            sample_rate=sample_rate,
            duration_sec=request.duration_sec,
            wall_time_sec=time.perf_counter() - start,
            device=device,
        )


def audioldm2_dependency_compatibility_reason() -> str:
    """Return a notebook-facing message for known AudioLDM2 version conflicts."""
    try:
        transformers_version = version("transformers")
    except PackageNotFoundError:
        return ""
    if _version_at_least(transformers_version, (4, 50)):
        return (
            f"transformers {transformers_version} is installed; use transformers==4.49.0 for AudioLDM2. "
            "Transformers 4.50+ removes generation helpers from GPT2Model, while "
            "diffusers.AudioLDM2Pipeline still calls them."
        )
    return ""


def _version_at_least(version_text: str, minimum: tuple[int, int]) -> bool:
    parts = version_text.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1])
    except (IndexError, ValueError):
        return False
    return (major, minor) >= minimum
