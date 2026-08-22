"""Stable Audio Open runner scaffold."""
from __future__ import annotations

import json
import math
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import time

import numpy as np

from _common.paths import portable_path
from _common.device_utils import choose_device
from model_runners.base import (
    GenerationRequest,
    GenerationResult,
    GenerationRunner,
    RunnerState,
    RunnerStatus,
    dependency_status,
    resolve_model_source,
)
from checks.setup_guidance import (
    STABLE_AUDIO_NUMPY_ABI_HINT,
    STABLE_AUDIO_OPEN_DOWNLOAD_HINT,
    STABLE_AUDIO_OPEN_INSTALL_HINT,
)


def stable_audio_waveform_for_wav(audio, peak: float = 0.98) -> tuple[np.ndarray, dict[str, float]]:
    """Prepare Stable Audio output for audible WAV export.

    Stable Audio Open can return a very low-amplitude decoded waveform. Writing
    that directly as the default 16-bit WAV subtype can quantize every sample to
    zero, so we follow the model-card convention and peak-normalize before
    saving.
    """
    from _common.audio_io import normalize_peak

    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim == 3 and waveform.shape[0] == 1:
        waveform = waveform[0]
    if waveform.ndim not in (1, 2):
        raise ValueError("Stable Audio output must be mono, stereo, or a single batched waveform")
    if not np.all(np.isfinite(waveform)):
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)

    raw_peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    raw_rms = float(np.sqrt(np.mean(np.square(waveform)))) if waveform.size else 0.0
    if raw_peak <= 1e-10:
        raise RuntimeError(
            "Stable Audio Open returned an all-zero or near-zero waveform before saving. "
            "Use a CUDA machine for this Notebook or reduce num_inference_steps for a quick diagnostic run."
        )

    normalized = normalize_peak(waveform, peak=peak)
    normalized_peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    normalized_rms = float(np.sqrt(np.mean(np.square(normalized)))) if normalized.size else 0.0
    return normalized, {
        "raw_peak": raw_peak,
        "raw_rms": raw_rms,
        "normalized_peak": normalized_peak,
        "normalized_rms": normalized_rms,
    }


def stable_audio_select_decode_window(
    audio,
    duration_sec: float,
    sample_rate: int,
    near_zero: float = 1e-10,
) -> tuple[np.ndarray, dict[str, float]]:
    """Select the requested Stable Audio window, falling back to the loudest window."""
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim == 3 and waveform.shape[0] == 1:
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform[None, :]
    if waveform.ndim != 2:
        raise ValueError("decoded Stable Audio must be channel-first audio")
    if not np.all(np.isfinite(waveform)):
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)

    target_samples = min(int(round(duration_sec * sample_rate)), waveform.shape[-1])
    full_peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    full_rms = float(np.sqrt(np.mean(np.square(waveform)))) if waveform.size else 0.0
    if full_peak <= near_zero or target_samples <= 0:
        raise RuntimeError(
            "Stable Audio Open returned an all-zero or near-zero full waveform. "
            "Use a CUDA machine for this Notebook or reduce num_inference_steps for a quick diagnostic run."
        )

    requested = waveform[..., :target_samples]
    requested_peak = float(np.max(np.abs(requested))) if requested.size else 0.0
    requested_rms = float(np.sqrt(np.mean(np.square(requested)))) if requested.size else 0.0
    if requested_peak > near_zero:
        return requested, {
            "full_peak": full_peak,
            "full_rms": full_rms,
            "requested_peak": requested_peak,
            "requested_rms": requested_rms,
            "selected_start_sec": 0.0,
        }

    selected_start = _loudest_window_start(waveform, target_samples, sample_rate)
    selected = waveform[..., selected_start : selected_start + target_samples]
    selected_peak = float(np.max(np.abs(selected))) if selected.size else 0.0
    selected_rms = float(np.sqrt(np.mean(np.square(selected)))) if selected.size else 0.0
    if selected_peak <= near_zero:
        raise RuntimeError(
            "Stable Audio decoded a non-zero full tensor but no audible target-length window "
            "could be found. Use a CUDA machine or the official stable-audio-tools path."
        )
    return selected, {
        "full_peak": full_peak,
        "full_rms": full_rms,
        "requested_peak": requested_peak,
        "requested_rms": requested_rms,
        "selected_start_sec": float(selected_start / sample_rate),
        "selected_peak": selected_peak,
        "selected_rms": selected_rms,
    }


def _loudest_window_start(audio: np.ndarray, window_samples: int, sample_rate: int) -> int:
    if audio.shape[-1] <= window_samples:
        return 0
    hop = max(1, int(round(sample_rate * 0.5)))
    starts = list(range(0, audio.shape[-1] - window_samples + 1, hop))
    last_start = audio.shape[-1] - window_samples
    if starts[-1] != last_start:
        starts.append(last_start)
    energies = [
        float(np.mean(np.square(audio[..., start : start + window_samples])))
        for start in starts
    ]
    return int(starts[int(np.argmax(energies))])


def tensor_peak_rms(tensor) -> dict[str, bool | float | tuple[int, ...] | str]:
    """Return compact diagnostics for torch tensors without keeping GPU refs."""
    values = tensor.detach().to("cpu").float()
    finite = bool(values.isfinite().all()) if values.numel() else True
    peak = float(values.abs().max()) if values.numel() else 0.0
    rms = float(values.square().mean().sqrt()) if values.numel() else 0.0
    return {
        "shape": tuple(int(dim) for dim in values.shape),
        "device": str(tensor.device),
        "peak": peak,
        "rms": rms,
        "finite": finite,
    }


def choose_stable_audio_device(requested: str = "auto") -> tuple[str, str]:
    """Return a Stable Audio Open compatible device.

    The shared chapter policy detects devices as cuda -> mps -> cpu. Stable
    Audio Open is most reliable through its official tools backend on CUDA or
    CPU, so this runner uses cuda -> cpu instead.
    """
    device = choose_device(requested)
    if device == "mps":
        return (
            "cpu",
            "Stable Audio Open uses the official stable-audio-tools backend here; this runner used CPU instead of MPS.",
        )
    return device, ""


def stable_audio_model_files(model_dir: Path) -> dict[str, Path | None]:
    """Return local Stable Audio Open files used by stable-audio-tools."""
    config_path = model_dir / "model_config.json"
    ckpt_path = model_dir / "model.safetensors"
    if not ckpt_path.exists():
        ckpt_path = model_dir / "model.ckpt"
    vae_path = model_dir / "vae_model.ckpt"
    return {
        "config": config_path if config_path.exists() else None,
        "checkpoint": ckpt_path if ckpt_path.exists() else None,
        "vae_checkpoint": vae_path if vae_path.exists() else None,
    }


def stable_audio_t5_files(t5_dir: Path) -> dict[str, Path | None]:
    """Return local T5 files needed by stable-audio-tools."""
    model_file = t5_dir / "model.safetensors"
    if not model_file.exists():
        model_file = t5_dir / "pytorch_model.bin"
    return {
        "t5_config": t5_dir / "config.json" if (t5_dir / "config.json").exists() else None,
        "t5_model": model_file if model_file.exists() else None,
        "t5_tokenizer": t5_dir / "spiece.model" if (t5_dir / "spiece.model").exists() else None,
    }


def stable_audio_sample_size(duration_sec: float, sample_rate: int, model_config: dict) -> int:
    """Return a valid sample count for Stable Audio Open generation."""
    max_sample_size = int(model_config.get("sample_size", int(duration_sec * sample_rate)))
    downsampling_ratio = int(
        model_config.get("model", {})
        .get("pretransform", {})
        .get("config", {})
        .get("downsampling_ratio", 2048)
    )
    target = max(1, int(math.ceil(duration_sec * sample_rate)))
    if downsampling_ratio > 1:
        target = int(math.ceil(target / downsampling_ratio) * downsampling_ratio)
    return min(target, max_sample_size)


def patch_stable_audio_t5_model_path(model_config: dict, t5_dir: Path) -> dict:
    """Point stable-audio-tools T5 conditioners to a local T5 directory."""
    patched = dict(model_config)
    model_section = dict(patched.get("model", {}))
    conditioning = dict(model_section.get("conditioning", {}))
    configs = []
    for conditioner in conditioning.get("configs", []):
        conditioner = dict(conditioner)
        if conditioner.get("type") == "t5":
            config = dict(conditioner.get("config", {}))
            config["model_path"] = str(t5_dir)
            conditioner["config"] = config
        configs.append(conditioner)
    conditioning["configs"] = configs
    model_section["conditioning"] = conditioning
    patched["model"] = model_section
    return patched


def load_stable_audio_tools_model(model_dir: Path, t5_dir: Path):
    """Load Stable Audio Open from local files with the official tools API."""
    try:
        from stable_audio_tools.models.factory import create_model_from_config
        from stable_audio_tools.models.utils import load_ckpt_state_dict
    except ValueError as exc:
        dependency_error = stable_audio_dependency_runtime_error(exc)
        if dependency_error is not None:
            raise dependency_error from exc
        raise

    files = stable_audio_model_files(model_dir)
    config_path = files["config"]
    ckpt_path = files["checkpoint"]
    if config_path is None or ckpt_path is None:
        raise RuntimeError(
            "Stable Audio Open local files are incomplete. Download the model to "
            "chapter08/models/stabilityai_stable_audio_open_1_0 before running this Notebook."
        )

    with Path(config_path).open("r", encoding="utf-8") as handle:
        model_config = json.load(handle)
    model_config = patch_stable_audio_t5_model_path(model_config, t5_dir)
    model = create_model_from_config(model_config)
    model.load_state_dict(load_ckpt_state_dict(str(ckpt_path)))
    return model, model_config


def stable_audio_dependency_runtime_error(exc: BaseException) -> RuntimeError | None:
    """Return a notebook-facing dependency message for known binary ABI failures."""
    message = str(exc)
    if "numpy.dtype size changed" in message:
        return RuntimeError(
            "Stable Audio dependency ABI mismatch: PyWavelets was built for a different "
            "NumPy ABI than the NumPy currently imported by this kernel. "
            + STABLE_AUDIO_NUMPY_ABI_HINT
        )
    if "libtorchaudio" in message or ("Symbol not found" in message and "torchaudio" in message):
        return RuntimeError(
            "Stable Audio dependency mismatch: torchaudio was built for a different torch "
            "binary than the one currently imported by this kernel. "
            + STABLE_AUDIO_NUMPY_ABI_HINT
        )
    return None


def stable_audio_torch_compatibility_reason() -> str:
    """Return a reason when the pinned Stable Audio torch binary set is inconsistent."""
    expected = {
        "torch": "2.7.1",
        "torchaudio": "2.7.1",
        "torchvision": "0.22.1",
    }
    observed: dict[str, str] = {}
    try:
        for package_name in expected:
            observed[package_name] = version(package_name).split("+", 1)[0]
    except PackageNotFoundError as exc:
        return f"missing {exc.name}"
    mismatched = [
        f"{package_name} {observed[package_name]} != {expected_version}"
        for package_name, expected_version in expected.items()
        if observed[package_name] != expected_version
    ]
    if mismatched:
        return "; ".join(mismatched)
    return ""


class StableAudioOpenRunner(GenerationRunner):
    model_name = "stabilityai/stable-audio-open-1.0"
    runner_name = "stable_audio_tools"
    estimated_vram_gb = 8.0
    license_note = "Gated Hugging Face model; accept the provider license before downloading weights."
    local_model_dir = Path(__file__).resolve().parents[1] / "models" / "stabilityai_stable_audio_open_1_0"
    local_t5_dir = Path(__file__).resolve().parents[1] / "models" / "t5_base"
    model_env_var = "CHAPTER08_STABLE_AUDIO_MODEL"
    t5_env_var = "CHAPTER08_T5_MODEL"

    def check_environment(self):
        status = dependency_status(
            self.model_name,
            self.runner_name,
            package_names=(
                "stable_audio_tools",
                "torch",
                "torchaudio",
                "torchvision",
                "pywt",
                "pytorch_lightning",
                "huggingface_hub",
                "safetensors",
                "librosa",
                "soundfile",
            ),
            next_action=STABLE_AUDIO_OPEN_INSTALL_HINT + " Then rerun 08_6d_stable_audio_open_inference.ipynb.",
            license_note=self.license_note,
            estimated_vram_gb=self.estimated_vram_gb,
        )
        if not status.available:
            return status
        abi_status = self._binary_dependency_status()
        if abi_status is not None:
            return abi_status
        weight_status = self._local_weight_status()
        if weight_status is not None:
            return weight_status
        return status

    def _binary_dependency_status(self) -> RunnerStatus | None:
        torch_reason = stable_audio_torch_compatibility_reason()
        if torch_reason:
            return RunnerStatus(
                model_name=self.model_name,
                runner=self.runner_name,
                status=RunnerState.MISSING_DEPENDENCY.value,
                reason="incompatible Stable Audio torch binary set: " + torch_reason,
                next_action=STABLE_AUDIO_NUMPY_ABI_HINT + " Then rerun 08_6d.",
                license_note=self.license_note,
                estimated_vram_gb=self.estimated_vram_gb,
            )
        try:
            import numpy  # noqa: F401
            import pywt  # noqa: F401
            import torch  # noqa: F401
            import torchaudio  # noqa: F401
            import torchvision  # noqa: F401
        except ValueError as exc:
            dependency_error = stable_audio_dependency_runtime_error(exc)
            if dependency_error is not None:
                return RunnerStatus(
                    model_name=self.model_name,
                    runner=self.runner_name,
                    status=RunnerState.MISSING_DEPENDENCY.value,
                    reason=str(dependency_error),
                    next_action=STABLE_AUDIO_NUMPY_ABI_HINT + " Then rerun 08_6d.",
                    license_note=self.license_note,
                    estimated_vram_gb=self.estimated_vram_gb,
                )
            raise
        except OSError as exc:
            dependency_error = stable_audio_dependency_runtime_error(exc)
            if dependency_error is not None:
                return RunnerStatus(
                    model_name=self.model_name,
                    runner=self.runner_name,
                    status=RunnerState.MISSING_DEPENDENCY.value,
                    reason=str(dependency_error),
                    next_action=STABLE_AUDIO_NUMPY_ABI_HINT + " Then rerun 08_6d.",
                    license_note=self.license_note,
                    estimated_vram_gb=self.estimated_vram_gb,
                )
            raise
        return None

    def _local_weight_status(self) -> RunnerStatus | None:
        model_name = resolve_model_source(self.model_name, self.local_model_dir, self.model_env_var)
        path = Path(model_name)
        if not path.exists():
            return RunnerStatus(
                model_name=self.model_name,
                runner=self.runner_name,
                status=RunnerState.MISSING_WEIGHTS.value,
                reason="local Stable Audio Open model directory is missing",
                next_action=STABLE_AUDIO_OPEN_DOWNLOAD_HINT + " Then rerun 08_6d_stable_audio_open_inference.ipynb.",
                license_note=self.license_note,
                estimated_vram_gb=self.estimated_vram_gb,
            )
        files = stable_audio_model_files(path)
        missing = [name for name, file_path in files.items() if file_path is None and name != "vae_checkpoint"]
        t5_path = Path(resolve_model_source("t5-base", self.local_t5_dir, self.t5_env_var))
        t5_missing = [name for name, file_path in stable_audio_t5_files(t5_path).items() if file_path is None]
        missing.extend(t5_missing)
        if missing:
            return RunnerStatus(
                model_name=self.model_name,
                runner=self.runner_name,
                status=RunnerState.MISSING_WEIGHTS.value,
                reason="missing local Stable Audio Open or T5 files: " + ", ".join(missing),
                next_action=STABLE_AUDIO_OPEN_DOWNLOAD_HINT + " Then rerun 08_6d_stable_audio_open_inference.ipynb.",
                license_note=self.license_note,
                estimated_vram_gb=self.estimated_vram_gb,
            )
        return None

    def generate(self, request: GenerationRequest) -> GenerationResult:
        status = self.check_environment()
        if not status.available:
            raise RuntimeError(f"{status.status}: {status.reason}")
        from _common.audio_io import save_audio

        import torch
        try:
            from stable_audio_tools.inference.generation import generate_diffusion_cond
        except ValueError as exc:
            dependency_error = stable_audio_dependency_runtime_error(exc)
            if dependency_error is not None:
                raise dependency_error from exc
            raise

        extra = request.extra or {}
        device, device_note = choose_stable_audio_device(str(extra.get("device", "auto")))
        requested_model = str(extra.get("model_name", self.model_name))
        model_name = resolve_model_source(requested_model, self.local_model_dir, self.model_env_var)
        t5_name = resolve_model_source("t5-base", self.local_t5_dir, self.t5_env_var)
        num_inference_steps = int(extra.get("num_inference_steps", 50))
        guidance_scale = float(extra.get("guidance_scale", 7.0))
        model_dir = Path(model_name)
        t5_dir = Path(t5_name)
        start = time.perf_counter()
        print(
            "Stable Audio: loading official stable-audio-tools model "
            f"model={portable_path(model_dir)} device={device}",
            flush=True,
        )
        print(f"Stable Audio: using local T5 text encoder {portable_path(t5_dir)}", flush=True)
        if device_note:
            print(f"Stable Audio: {device_note}", flush=True)
        try:
            model, model_config = load_stable_audio_tools_model(model_dir, t5_dir)
        except ValueError as exc:
            dependency_error = stable_audio_dependency_runtime_error(exc)
            if dependency_error is not None:
                raise dependency_error from exc
            raise
        sample_rate = int(extra.get("sample_rate", model_config.get("sample_rate", 44100)))
        sample_size = stable_audio_sample_size(float(request.duration_sec), sample_rate, model_config)
        model = model.to(device)
        model.eval()
        if request.seed is not None:
            torch.manual_seed(int(request.seed))
            if device == "cuda":
                torch.cuda.manual_seed_all(int(request.seed))

        print(
            "Stable Audio: sampling "
            f"duration={request.duration_sec}s sample_size={sample_size} "
            f"steps={num_inference_steps} guidance={guidance_scale}",
            flush=True,
        )
        conditioning = [
            {
                "prompt": request.prompt,
                "seconds_start": float(extra.get("seconds_start", 0.0)),
                "seconds_total": float(request.duration_sec),
            }
        ]
        with torch.no_grad():
            generated = generate_diffusion_cond(
                model,
                steps=num_inference_steps,
                cfg_scale=guidance_scale,
                conditioning=conditioning,
                sample_size=sample_size,
                sigma_min=float(extra.get("sigma_min", 0.3)),
                sigma_max=float(extra.get("sigma_max", 500.0)),
                sampler_type=str(extra.get("sampler_type", "dpmpp-3m-sde")),
                device=device,
            )
        tensor_stats = tensor_peak_rms(generated)
        print(
            "Stable Audio: generated tensor stats "
            f"shape={tensor_stats['shape']} device={tensor_stats['device']} "
            f"peak={tensor_stats['peak']:.3e} rms={tensor_stats['rms']:.3e} "
            f"finite={tensor_stats['finite']}",
            flush=True,
        )
        if not bool(tensor_stats["finite"]):
            raise RuntimeError(
                "Stable Audio Open returned NaN or Inf audio. This indicates a runtime/backend problem. "
                "Use a CUDA machine for this Notebook or reduce num_inference_steps for a quick diagnostic run."
            )
        if float(tensor_stats["peak"]) <= 1e-10:
            raise RuntimeError(
                "Stable Audio Open returned all-zero or near-zero audio. Use a CUDA machine for "
                "this Notebook or reduce num_inference_steps for a quick diagnostic run."
            )
        audio_tensor = generated.detach().to("cpu").float()
        if audio_tensor.ndim == 3:
            audio_tensor = audio_tensor[0]
        print("Stable Audio: writing audio", flush=True)
        selected_audio, window_stats = stable_audio_select_decode_window(
            audio_tensor.numpy(),
            duration_sec=float(request.duration_sec),
            sample_rate=sample_rate,
        )
        print(
            "Stable Audio: decode window stats "
            f"full_peak={window_stats['full_peak']:.3e} full_rms={window_stats['full_rms']:.3e} "
            f"requested_peak={window_stats['requested_peak']:.3e} "
            f"requested_rms={window_stats['requested_rms']:.3e} "
            f"selected_start={window_stats['selected_start_sec']:.2f}s",
            flush=True,
        )
        audio, audio_stats = stable_audio_waveform_for_wav(selected_audio)
        print(
            "Stable Audio: audio stats "
            f"raw_peak={audio_stats['raw_peak']:.3e} raw_rms={audio_stats['raw_rms']:.3e} "
            f"saved_peak={audio_stats['normalized_peak']:.3f} saved_rms={audio_stats['normalized_rms']:.3f}",
            flush=True,
        )
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{request.prompt_id}_stable_audio_open.wav"
        save_audio(output_path, audio, sample_rate)
        print(f"Stable Audio: saved {portable_path(output_path)}", flush=True)
        return GenerationResult(
            model_name=model_name,
            prompt_id=request.prompt_id,
            output_audio_path=output_path,
            sample_rate=sample_rate,
            duration_sec=request.duration_sec,
            wall_time_sec=time.perf_counter() - start,
            device=device,
            notes=device_note or RunnerState.AVAILABLE.value,
        )
