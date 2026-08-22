"""按 Notebook 声明运行环境,供各 Notebook 开头自检。

09_1/09_2/09_3/09_5 用主 anaconda 环境;09_4 需独立环境 CODE/venv_ch09_clap
(按 CODE/chapter06/pretrained_transformer/ENV_NOTES.md 的 pin 集复刻,见写作指南 §11.2)。
"""
from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

CHAPTER_ROOT = Path(__file__).resolve().parents[1]
CLAP_VENV = CHAPTER_ROOT.parent / "venv_ch09_clap"

_MAIN = "主 anaconda(Python 3.13)"
NOTEBOOK_ENVS: dict[str, dict] = {
    "09_1_audio_fingerprint": {
        "env": _MAIN,
        "packages": ["numpy", "scipy", "librosa", "soundfile", "matplotlib", "pandas", "pretty_midi"],
        "cli": ["fluidsynth", "ffmpeg"],
    },
    "09_2_qbh_dtw": {
        "env": _MAIN,
        "packages": ["numpy", "scipy", "librosa", "soundfile", "matplotlib", "pretty_midi", "pandas"],
    },
    "09_3_cover_song": {
        "env": _MAIN,
        "packages": ["numpy", "scipy", "librosa", "soundfile", "matplotlib", "pretty_midi", "pandas"],
        "cli": ["fluidsynth"],
    },
    "09_4_clap_retrieval": {
        "env": f"独立环境 {CLAP_VENV}",
        "packages": ["laion_clap", "torch", "librosa", "numpy", "pandas", "matplotlib", "sklearn"],
    },
    "09_5_recommendation_toy": {"env": _MAIN, "packages": ["numpy", "matplotlib", "pandas"]},
}


def check_notebook_env(notebook: str) -> bool:
    """检查当前解释器是否满足指定 Notebook 的依赖;打印逐项结果,返回整体是否通过。"""
    spec = NOTEBOOK_ENVS[notebook]
    ok = True
    print(f"{notebook}:目标环境 {spec['env']};当前 {sys.executable}")
    for pkg in spec["packages"]:
        try:
            importlib.import_module(pkg)
            print(f"  [ok] import {pkg}")
        except ImportError:
            print(f"  [缺] import {pkg}")
            ok = False
    for cli in spec.get("cli", []):
        path = shutil.which(cli)
        print(f"  [{'ok' if path else '缺'}] CLI {cli}" + (f"({path})" if path else ""))
        ok = ok and path is not None
    return ok
