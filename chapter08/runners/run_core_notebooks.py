"""Execute core Chapter 8 notebooks when they exist."""
from __future__ import annotations

from runners.run_light_notebooks import notebook_run_failures, run_notebooks, write_notebook_run_status


CORE_NOTEBOOKS = [
    "08_1_generation_paradigms.ipynb",
    "08_2_wavenet_receptive_field.ipynb",
    "08_3_wavenet_training_nsynth.ipynb",
    "08_4_encodec_codec_tokens.ipynb",
    "08_5_mini_codec_lm_training.ipynb",
]


def main() -> None:
    rows = run_notebooks(CORE_NOTEBOOKS)
    write_notebook_run_status("outputs/tables/08_core_notebook_run_status.csv", rows)
    executed = sum(1 for row in rows if row["status"] == "executed")
    print(f"Executed {executed} core notebooks")
    failures = notebook_run_failures(rows)
    if failures:
        details = "\n".join(f"- {row['notebook']}: {row['status']} {row.get('detail', '')}" for row in failures)
        raise RuntimeError("Core notebook execution failed:\n" + details)


if __name__ == "__main__":
    main()
