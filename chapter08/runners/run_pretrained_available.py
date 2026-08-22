"""Check pretrained runners and run only available model notebooks."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Callable

from _common.config import load_yaml_config
from _common.tables import write_rows
from model_runners.ace_step import ACEStepRunner
from model_runners.audioldm2 import AudioLDM2Runner
from model_runners.base import GenerationRequest, RunnerStatus, write_runner_status_table
from model_runners.musicgen import MusicGenRunner
from model_runners.stable_audio_open import StableAudioOpenRunner
from model_runners.yue import YuERunner


RUNNER_NOTEBOOKS = {
    "facebook/musicgen-small": "08_6b_musicgen_inference.ipynb",
    "cvssp/audioldm2": "08_6c_audioldm2_inference.ipynb",
    "stabilityai/stable-audio-open-1.0": "08_6d_stable_audio_open_inference.ipynb",
    "m-a-p/YuE": "08_6e_yue_full_song_generation.ipynb",
    "ACE-Step/ACE-Step-v1-3.5B": "08_6f_ace_step_generation.ipynb",
}

RUNNER_CONFIGS = {
    "facebook/musicgen-small": "configs/musicgen_inference.yaml",
    "cvssp/audioldm2": "configs/audioldm2_inference.yaml",
    "stabilityai/stable-audio-open-1.0": "configs/stable_audio_open_inference.yaml",
    "m-a-p/YuE": "configs/yue_inference.yaml",
    "ACE-Step/ACE-Step-v1-3.5B": "configs/ace_step_inference.yaml",
}


def main() -> None:
    runners = [
        MusicGenRunner(),
        AudioLDM2Runner(),
        StableAudioOpenRunner(),
        YuERunner(),
        ACEStepRunner(),
    ]
    statuses = [runner.check_environment() for runner in runners]
    write_runner_status_table("outputs/tables/08_model_runner_status.csv", statuses)
    plan_rows = build_run_plan_rows(statuses, notebook_root=Path.cwd())
    write_rows("outputs/tables/08_pretrained_run_plan.csv", plan_rows, fieldnames=RUN_PLAN_FIELDS)
    request_rows = build_request_plan_rows(statuses, notebook_root=Path.cwd())
    write_rows(
        "outputs/tables/08_pretrained_request_plan.csv",
        request_rows,
        fieldnames=REQUEST_PLAN_FIELDS,
    )
    run_available_notebooks(statuses, notebook_root=Path.cwd())


def run_available_notebooks(
    statuses: list[RunnerStatus],
    notebook_root: Path | str = Path("."),
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> None:
    """Execute pretrained notebooks whose dependencies and access checks are available."""
    notebook_root = Path(notebook_root)
    for status in statuses:
        notebook = RUNNER_NOTEBOOKS[status.model_name]
        if not status.available:
            print(f"skip {notebook}: {status.status} - {status.reason}")
            continue
        if not (notebook_root / notebook).exists():
            print(f"skip missing notebook: {notebook}")
            continue
        runner(
            [sys.executable, "-m", "nbconvert", "--execute", "--to", "notebook", "--inplace", notebook],
            cwd=notebook_root,
            check=True,
        )


RUN_PLAN_FIELDS = [
    "model_name",
    "notebook",
    "status",
    "will_run",
    "reason",
    "next_action",
]

REQUEST_PLAN_FIELDS = [
    "model_name",
    "runner",
    "notebook",
    "status",
    "will_run",
    "prompt_id",
    "prompt_preview",
    "duration_sec",
    "output_dir",
    "table_csv",
    "command_preview",
    "next_action",
]


def build_run_plan_rows(
    statuses: list[RunnerStatus],
    notebook_root: Path | str = Path("."),
) -> list[dict[str, object]]:
    """Return table rows describing what run_pretrained_available will execute."""
    notebook_root = Path(notebook_root)
    rows = []
    for status in statuses:
        notebook = RUNNER_NOTEBOOKS[status.model_name]
        notebook_exists = (notebook_root / notebook).exists()
        will_run = status.available and notebook_exists
        reason = status.reason
        next_action = status.next_action
        if status.available and not notebook_exists:
            reason = "notebook is missing"
            next_action = f"create {notebook}"
        rows.append(
            {
                "model_name": status.model_name,
                "notebook": notebook,
                "status": status.status,
                "will_run": will_run,
                "reason": reason,
                "next_action": next_action,
            }
        )
    return rows


def build_request_plan_rows(
    statuses: list[RunnerStatus],
    notebook_root: Path | str = Path("."),
) -> list[dict[str, object]]:
    """Return rows describing model requests used by pretrained notebooks."""
    notebook_root = Path(notebook_root)
    rows = []
    for status in statuses:
        notebook = RUNNER_NOTEBOOKS[status.model_name]
        config_path = notebook_root / RUNNER_CONFIGS[status.model_name]
        config = load_yaml_config(config_path) if config_path.exists() else {}
        prompt = first_prompt(config)
        output_dir = str(config.get("outputs", {}).get("audio_dir", ""))
        table_csv = str(config.get("outputs", {}).get("table_csv", ""))
        command_preview = build_command_preview(status.model_name, config, output_dir, notebook)
        rows.append(
            {
                "model_name": status.model_name,
                "runner": status.runner,
                "notebook": notebook,
                "status": status.status,
                "will_run": status.available and (notebook_root / notebook).exists(),
                "prompt_id": prompt.get("prompt_id", ""),
                "prompt_preview": prompt.get("text", "")[:160],
                "duration_sec": config.get("duration_seconds", ""),
                "output_dir": output_dir,
                "table_csv": table_csv,
                "command_preview": command_preview,
                "next_action": status.next_action,
            }
        )
    return rows


def first_prompt(config: dict) -> dict[str, str]:
    """Return the first configured prompt row, if present."""
    prompts = config.get("prompts") or []
    if prompts:
        return {
            "prompt_id": str(prompts[0].get("prompt_id", "")),
            "text": str(prompts[0].get("text", "")),
        }
    return {"prompt_id": "", "text": ""}


def build_command_preview(
    model_name: str,
    config: dict,
    output_dir: str,
    notebook: str,
) -> str:
    """Return a command or execution preview without running a model."""
    if model_name == "m-a-p/YuE":
        runner = YuERunner()
        request = GenerationRequest(
            prompt="",
            prompt_id="yue_demo",
            duration_sec=0,
            output_dir=Path(output_dir or "output_audio/08_6e_yue"),
            extra={
                "command_template": config.get("command_template"),
                "genre_txt": config.get("genre_txt", "data_manifests/yue_genre.example.txt"),
                "lyrics_file": config.get("lyrics_file", "data_manifests/yue_prompt.example.txt"),
                "stage1_model": config.get("stage1_model_dir", "models/m_a_p_yue_s1_7b_anneal_en_cot"),
                "stage2_model": config.get("stage2_model_dir", "models/m_a_p_yue_s2_1b_general"),
            },
        )
        return runner.build_command(request).shell_command()
    if model_name == "ACE-Step/ACE-Step-v1-3.5B":
        return ACEStepRunner().build_gradio_command(
            port=7865,
            checkpoint_dir=config.get("checkpoint_dir", "models/ace_step_v1_3_5b"),
            bf16=bool(config.get("bf16", False)),
        ).shell_command()
    return f"python -m nbconvert --execute --to notebook --inplace {notebook}"


if __name__ == "__main__":
    main()
