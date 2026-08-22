"""Execute light Chapter 8 notebooks that do not load large model weights."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
import subprocess
import sys

from _common.paths import portable_path
from _common.tables import write_rows


LIGHT_NOTEBOOKS = [
    "08_0_audio_data_pipeline.ipynb",
    "08_6a_pretrained_overview.ipynb",
    "08_7_prompt_control_and_evaluation.ipynb",
    "08_8_lora_personalization.ipynb",
]

NOTEBOOK_RUN_FIELDS = ["notebook", "status", "command", "detail"]


def nbconvert_command(notebook: str) -> list[str]:
    """Return the standard in-place execution command for one notebook."""
    return [
        sys.executable,
        "-m",
        "nbconvert",
        "--execute",
        "--to",
        "notebook",
        "--inplace",
        notebook,
    ]


def run_notebooks(
    notebooks: Sequence[str] = LIGHT_NOTEBOOKS,
    notebook_root: Path | str = Path("."),
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> list[dict[str, object]]:
    """Execute present notebooks and return status rows."""
    notebook_root = Path(notebook_root)
    rows: list[dict[str, object]] = []
    for notebook in notebooks:
        path = notebook_root / notebook
        if not path.exists():
            print(f"skip missing notebook: {notebook}")
            rows.append({"notebook": notebook, "status": "missing", "command": "", "detail": "not found"})
            continue
        command = nbconvert_command(notebook)
        command_text = " ".join([portable_path(command[0], notebook_root), *command[1:]])
        try:
            runner(command, cwd=notebook_root, check=True)
        except subprocess.CalledProcessError as exc:
            rows.append(
                {
                    "notebook": notebook,
                    "status": "failed",
                    "command": command_text,
                    "detail": f"returncode={exc.returncode}",
                }
            )
            break
        rows.append(
            {
                "notebook": notebook,
                "status": "executed",
                "command": command_text,
                "detail": "",
            }
        )
    return rows


def notebook_run_failures(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return notebook rows that did not execute successfully."""
    return [row for row in rows if row.get("status") != "executed"]


def write_notebook_run_status(output_csv: Path | str, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Write notebook execution rows with a stable schema."""
    write_rows(output_csv, rows, fieldnames=NOTEBOOK_RUN_FIELDS)
    return rows


def main() -> None:
    rows = run_notebooks()
    write_notebook_run_status("outputs/tables/08_light_notebook_run_status.csv", rows)
    executed = sum(1 for row in rows if row["status"] == "executed")
    print(f"Executed {executed} light notebooks")
    failures = notebook_run_failures(rows)
    if failures:
        details = "\n".join(f"- {row['notebook']}: {row['status']} {row.get('detail', '')}" for row in failures)
        raise RuntimeError("Light notebook execution failed:\n" + details)


if __name__ == "__main__":
    main()
