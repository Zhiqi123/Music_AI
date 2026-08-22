"""Execute Chapter 7 notebooks linearly in the current Python environment."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path


DEFAULT_NOTEBOOKS = [
    "07_1_problem_masks_phase.ipynb",
    "07_2_classic_hpss_nmf_repet.ipynb",
    "07_3_unet_roformer_visualized.ipynb",
    "07_4_pretrained_demucs_openunmix.ipynb",
    "07_5_musdb18hq_evaluation.ipynb",
    "07_6_external_case_studies.ipynb",
]


def execute_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    print(f"running {path.name}")
    for index, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        try:
            exec(compile(source, f"{path.name}:cell-{index}", "exec"), namespace)
        except Exception:
            print(f"failed {path.name} cell {index}", file=sys.stderr)
            traceback.print_exc()
            raise


def main(argv: list[str]) -> int:
    chapter_dir = Path(__file__).resolve().parent
    os.chdir(chapter_dir)
    cache_root = Path(tempfile.gettempdir())
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "chapter07_cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "mplconfig"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba_cache"))

    try:
        import matplotlib

        matplotlib.use("Agg")
    except Exception:
        pass

    notebooks = argv or DEFAULT_NOTEBOOKS
    for name in notebooks:
        execute_notebook(chapter_dir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
