"""ACE-Step generation and LoRA runner scaffold."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import time

from _common.device_utils import choose_device
from model_runners.base import (
    ExternalCommand,
    GenerationRequest,
    GenerationResult,
    GenerationRunner,
    RunnerStatus,
    RunnerState,
    dependency_status,
)


CHAPTER_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = CHAPTER_ROOT.parent
ACE_STEP_SOURCE_DIR = CODE_ROOT / "external" / "ACE-Step"
ACE_STEP_CHECKPOINT_DIR = CHAPTER_ROOT / "models" / "ace_step_v1_3_5b"
ACE_STEP_CHECKPOINT_REL = "chapter08/models/ace_step_v1_3_5b"
ACE_STEP_REQUIRED_DIRS = (
    "music_dcae_f8c8",
    "music_vocoder",
    "ace_step_transformer",
    "umt5-base",
)


def checkpoint_dir_has_weights(path: Path) -> bool:
    """Return true for direct ``hf download --local-dir`` or HF cache layouts."""
    if all((path / dirname).exists() for dirname in ACE_STEP_REQUIRED_DIRS):
        return True
    snapshot_root = path / "models--ACE-Step--ACE-Step-v1-3.5B" / "snapshots"
    if snapshot_root.exists():
        return any(
            all((snapshot / dirname).exists() for dirname in ACE_STEP_REQUIRED_DIRS)
            for snapshot in snapshot_root.iterdir()
            if snapshot.is_dir()
        )
    return False


class ACEStepRunner(GenerationRunner):
    model_name = "ACE-Step/ACE-Step-v1-3.5B"
    runner_name = "ace_step"
    estimated_vram_gb = 12.0
    license_note = "Check ACE-Step license and LoRA training terms before distributing outputs."

    def check_environment(self):
        package_ok = importlib.util.find_spec("acestep") is not None
        command_ok = shutil.which("acestep") is not None
        if not (package_ok or command_ok):
            return dependency_status(
                self.model_name,
                self.runner_name,
                package_names=("acestep",),
                next_action=(
                    "install ACE-Step from CODE/external/ACE-Step with "
                    "'python -m pip install -e external/ACE-Step', then rerun this Notebook"
                ),
                license_note=self.license_note,
                estimated_vram_gb=self.estimated_vram_gb,
            )
        if not checkpoint_dir_has_weights(ACE_STEP_CHECKPOINT_DIR):
            return RunnerStatus(
                model_name=self.model_name,
                runner=self.runner_name,
                status=RunnerState.MISSING_WEIGHTS.value,
                reason=f"missing ACE-Step checkpoint directories under {ACE_STEP_CHECKPOINT_REL}",
                next_action=(
                    "download ACE-Step/ACE-Step-v1-3.5B with "
                    f"'hf download ACE-Step/ACE-Step-v1-3.5B --local-dir {ACE_STEP_CHECKPOINT_REL}', "
                    "or manually place music_dcae_f8c8, music_vocoder, ace_step_transformer, "
                    "and umt5-base under that directory"
                ),
                license_note=self.license_note,
                estimated_vram_gb=self.estimated_vram_gb,
            )
        return dependency_status(
            self.model_name,
            self.runner_name,
            package_names=("torch", "torchaudio", "torchcodec"),
            next_action=(
                "install ACE-Step runtime helpers with "
                "'python -m pip install pandas PyYAML torchcodec ipykernel ipywidgets', "
                "then rerun 08_6f_ace_step_generation.ipynb"
            ),
            license_note=self.license_note,
            estimated_vram_gb=self.estimated_vram_gb,
        )

    def build_gradio_command(
        self,
        port: int = 7865,
        checkpoint_dir: Path | str = Path("models/ace_step_v1_3_5b"),
        bf16: bool = False,
    ) -> ExternalCommand:
        return ExternalCommand(
            command=(
                "acestep",
                "--checkpoint_path",
                str(checkpoint_dir),
                "--port",
                str(port),
                "--bf16",
                "true" if bf16 else "false",
            ),
            notes="Launch the ACE-Step Gradio app for manual prompt experiments.",
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        status = self.check_environment()
        if not status.available:
            raise RuntimeError(f"{status.status}: {status.reason}")
        from acestep.pipeline_ace_step import ACEStepPipeline

        extra = request.extra or {}
        device = choose_device(str(extra.get("device", "auto")))
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{request.prompt_id}_ace_step.wav"
        start = time.perf_counter()
        pipeline = ACEStepPipeline(
            checkpoint_dir=str(extra.get("checkpoint_dir", "models/ace_step_v1_3_5b")),
            device_id=extra.get("device_id", 0),
            dtype=str(extra.get("dtype", "float32")),
        )
        pipeline(
            audio_duration=float(request.duration_sec),
            prompt=request.prompt,
            lyrics=str(extra.get("lyrics", "")),
            infer_step=int(extra.get("infer_steps", 60)),
            guidance_scale=float(extra.get("guidance_scale", 15.0)),
            scheduler_type=str(extra.get("scheduler_type", "euler")),
            cfg_type=str(extra.get("cfg_type", "apg")),
            omega_scale=float(extra.get("omega_scale", 10.0)),
            manual_seeds=str(request.seed if request.seed is not None else extra.get("seed", 0)),
            save_path=str(output_path),
        )
        return GenerationResult(
            model_name=self.model_name,
            prompt_id=request.prompt_id,
            output_audio_path=output_path,
            duration_sec=request.duration_sec,
            wall_time_sec=time.perf_counter() - start,
            device=device,
            notes=RunnerState.AVAILABLE.value,
        )
