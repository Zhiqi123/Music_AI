"""YuE full-song generation runner scaffold."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import time

from _common.paths import command_relative_path, portable_text
from model_runners.base import (
    ExternalCommand,
    GenerationRequest,
    GenerationResult,
    GenerationRunner,
    RunnerState,
    RunnerStatus,
    dependency_status,
    render_command_template,
)
from checks.setup_guidance import YUE_INSTALL_HINT


class YuERunner(GenerationRunner):
    model_name = "m-a-p/YuE"
    runner_name = "yue.cli"
    estimated_vram_gb = 16.0
    license_note = "Full-song checkpoints may require model-card review and substantial GPU memory."
    repo_env = "CHAPTER08_YUE_REPO"
    chapter_root = Path(__file__).resolve().parents[1]
    default_repo = Path(__file__).resolve().parents[2] / "external" / "YuE" / "inference"
    default_stage1_dir = chapter_root / "models" / "m_a_p_yue_s1_7b_anneal_en_cot"
    default_stage2_dir = chapter_root / "models" / "m_a_p_yue_s2_1b_general"
    required_xcodec_files = (
        "xcodec_mini_infer/final_ckpt/config.yaml",
        "xcodec_mini_infer/final_ckpt/ckpt_00360000.pth",
        "xcodec_mini_infer/decoders/config.yaml",
        "xcodec_mini_infer/decoders/decoder_131000.pth",
        "xcodec_mini_infer/decoders/decoder_151000.pth",
    )

    def check_environment(self):
        repo = self.repo_path()
        if repo is None or not (repo / "infer.py").exists():
            return RunnerStatus(
                model_name=self.model_name,
                runner=self.runner_name,
                status=RunnerState.MISSING_WEIGHTS.value,
                reason=(
                    "missing YuE repository with infer.py; checked "
                    f"{self.repo_env} and CODE/external/YuE/inference"
                ),
                next_action=YUE_INSTALL_HINT,
                license_note=self.license_note,
                estimated_vram_gb=self.estimated_vram_gb,
            )
        missing_xcodec = [path for path in self.required_xcodec_files if not (repo / path).exists()]
        if missing_xcodec:
            return RunnerStatus(
                model_name=self.model_name,
                runner=self.runner_name,
                status=RunnerState.MISSING_WEIGHTS.value,
                reason="missing YuE xcodec files: " + ", ".join(missing_xcodec),
                next_action=YUE_INSTALL_HINT,
                license_note=self.license_note,
                estimated_vram_gb=self.estimated_vram_gb,
            )
        missing_models = self.missing_local_models()
        if missing_models:
            return RunnerStatus(
                model_name=self.model_name,
                runner=self.runner_name,
                status=RunnerState.MISSING_WEIGHTS.value,
                reason="missing local YuE checkpoints: " + ", ".join(missing_models),
                next_action=YUE_INSTALL_HINT,
                license_note=self.license_note,
                estimated_vram_gb=self.estimated_vram_gb,
            )
        dependencies = dependency_status(
            self.model_name,
            self.runner_name,
            package_names=(
                "torch",
                "torchaudio",
                "transformers",
                "pandas",
                "omegaconf",
                "einops",
                "soundfile",
                "tqdm",
            ),
            next_action=YUE_INSTALL_HINT + " Then run 08_6e_yue_full_song_generation.ipynb.",
            license_note=self.license_note,
            estimated_vram_gb=self.estimated_vram_gb,
        )
        if not dependencies.available:
            return dependencies
        runtime_status = self.cuda_flash_attention_status()
        if runtime_status is not None:
            return runtime_status
        return dependencies

    def cuda_flash_attention_status(self) -> RunnerStatus | None:
        import torch

        if not torch.cuda.is_available():
            return RunnerStatus(
                model_name=self.model_name,
                runner=self.runner_name,
                status=RunnerState.INSUFFICIENT_MEMORY.value,
                reason=(
                    "YuE official infer.py is a CUDA/FlashAttention2 full-song pipeline; "
                    "the current kernel reports cuda=False. CPU/MPS execution is not used "
                    "by this notebook because it can run for hours and then fail. On Apple "
                    "Silicon, large unified memory does not provide CUDA or FlashAttention2, "
                    "so a Mac can prepare assets and inspect the command but is not a reliable "
                    "local YuE generation target."
                ),
                next_action=(
                    "Use a Linux/WSL2/remote CUDA environment with enough VRAM, install "
                    "flash-attn after installing the matching CUDA PyTorch build, then select "
                    "the Python 3.10 (chapter08-yue) kernel and rerun."
                ),
                license_note=self.license_note,
                estimated_vram_gb=self.estimated_vram_gb,
            )
        if importlib.util.find_spec("flash_attn") is None:
            return RunnerStatus(
                model_name=self.model_name,
                runner=self.runner_name,
                status=RunnerState.MISSING_DEPENDENCY.value,
                reason=(
                    "missing package: flash_attn. YuE official infer.py sets "
                    "attn_implementation='flash_attention_2'."
                ),
                next_action=(
                    "In the CUDA YuE environment, run: python -m pip install flash-attn --no-build-isolation. "
                    "If that build fails, use a CUDA/PyTorch/flash-attn combination supported by the "
                    "official YuE instructions."
                ),
                license_note=self.license_note,
                estimated_vram_gb=self.estimated_vram_gb,
            )
        return None

    def repo_path(self) -> Path | None:
        value = os.getenv(self.repo_env, "").strip()
        if value:
            return Path(value).expanduser()
        return self.default_repo

    def chapter_path(self, value: Path | str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.chapter_root / path

    def missing_local_models(self) -> list[str]:
        missing = []
        for label, path in (
            ("stage1:chapter08/models/m_a_p_yue_s1_7b_anneal_en_cot", self.default_stage1_dir),
            ("stage2:chapter08/models/m_a_p_yue_s2_1b_general", self.default_stage2_dir),
        ):
            if not self.model_dir_ready(path):
                missing.append(label)
        return missing

    @staticmethod
    def model_dir_ready(path: Path) -> bool:
        if not path.exists():
            return False
        has_config = (path / "config.json").exists()
        has_indexed_safetensors = (path / "model.safetensors.index.json").exists() and any(
            path.glob("model-*.safetensors")
        )
        has_single_weight = any(
            (path / filename).exists()
            for filename in ("model.safetensors", "pytorch_model.bin", "pytorch_model.pt")
        )
        return has_config and (has_indexed_safetensors or has_single_weight)

    def model_arg(self, value: object, repo: Path) -> str:
        text = str(value)
        path = Path(text)
        if path.is_absolute():
            return text
        if path.parts and path.parts[0] == "models":
            return command_relative_path(self.chapter_path(path), repo)
        return text

    def build_command(self, request: GenerationRequest) -> ExternalCommand:
        extra = request.extra or {}
        repo = Path(extra.get("repo_path") or self.repo_path() or ".").expanduser()
        output_dir = Path(request.output_dir)
        template = extra.get("command_template")
        if template is None:
            template = (
                "python",
                "infer.py",
                "--stage1_model",
                "{stage1_model}",
                "--stage2_model",
                "{stage2_model}",
                "--genre_txt",
                "{genre_txt}",
                "--lyrics_txt",
                "{lyrics_file}",
                "--run_n_segments",
                str(extra.get("run_n_segments", 2)),
                "--output_dir",
                "{output_dir}",
            )
        values = {
            "genre_txt": command_relative_path(
                self.chapter_path(extra.get("genre_txt", "data_manifests/yue_genre.example.txt")),
                repo,
            ),
            "lyrics_file": command_relative_path(
                self.chapter_path(extra.get("lyrics_file", "data_manifests/yue_prompt.example.txt")),
                repo,
            ),
            "output_dir": command_relative_path(self.chapter_path(output_dir), repo),
            "stage1_model": self.model_arg(
                extra.get("stage1_model", "models/m_a_p_yue_s1_7b_anneal_en_cot"),
                repo,
            ),
            "stage2_model": self.model_arg(
                extra.get("stage2_model", "models/m_a_p_yue_s2_1b_general"),
                repo,
            ),
            "prompt_id": request.prompt_id,
        }
        return ExternalCommand(
            command=render_command_template(template, values),
            cwd=repo,
            expected_output=output_dir,
            notes="YuE writes one or more audio files under output_dir.",
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        command = self.build_command(request)
        extra = request.extra or {}
        if not bool(extra.get("run_external", False)):
            return GenerationResult(
                model_name=self.model_name,
                prompt_id=request.prompt_id,
                output_audio_path=command.expected_output,
                duration_sec=request.duration_sec,
                notes=command.shell_command(),
            )
        status = self.check_environment()
        if not status.available:
            raise RuntimeError(
                "YuE runner is not available in the current kernel.\n"
                f"{status.reason}\n{status.next_action}"
            )
        wall_time = run_yue_external_command(command)
        return GenerationResult(
            model_name=self.model_name,
            prompt_id=request.prompt_id,
            output_audio_path=command.expected_output,
            duration_sec=request.duration_sec,
            wall_time_sec=wall_time,
            notes=command.shell_command(),
        )


def run_yue_external_command(command: ExternalCommand, tail_lines: int = 80) -> float:
    """Run YuE's official command while preserving useful failure output."""
    start = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        list(command.command),
        cwd=command.cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    tail: list[str] = []
    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            tail.append(line)
            if len(tail) > tail_lines:
                tail.pop(0)
        return_code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        raise
    wall_time = time.perf_counter() - start
    if return_code != 0:
        tail_text = portable_text("".join(tail), Path(".")).strip()
        raise RuntimeError(
            "YuE command failed with exit status "
            f"{return_code} after {wall_time:.1f}s.\n"
            f"Command: {command.shell_command()}\n"
            f"cwd: {portable_text(str(command.cwd or ''), Path('.'))}\n"
            "Last YuE output:\n"
            f"{tail_text}"
        )
    return wall_time
