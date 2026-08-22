"""Notebook 环境检查工具。

每个 chapter06 的 notebook 在 §1 调用 `check_environment(...)`，提前发现：
  - 依赖包缺失（给出可直接复制的 pip 命令）
  - CTMP 清洗输出目录缺失（给出生成命令）
  - CTIS 数据缓存缺失（首次运行会自动下载，这里只做提示）

设计原则：只诊断，不在 notebook 里替用户执行 `pip install` / 数据下载——
这些动作要么涉及联网、要么会污染当前 venv，应该由用户显式确认。
"""
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# 包名（pip）→ 导入名（import）映射。值为 None 表示同名。
# 加新依赖时只在这里登记一次，全 notebook 共享。
_PACKAGE_IMPORT_NAME: dict[str, str] = {
    "scikit-learn": "sklearn",
    "laion-clap": "laion_clap",
    "Pillow": "PIL",
    "PyYAML": "yaml",
}

# Notebook 维度的依赖配方。新 notebook 在这里加一行即可。
PRESETS: dict[str, list[str]] = {
    "06_1a": ["numpy", "pandas", "matplotlib", "datasets", "soundfile"],
    "06_1b": [
        "numpy", "pandas", "matplotlib", "soundfile",
        "scikit-learn", "xgboost", "librosa",
    ],
    "06_2": [
        "numpy", "pandas", "matplotlib", "soundfile",
        "scikit-learn", "torch", "librosa",
    ],
    "06_3": [
        "numpy", "pandas", "matplotlib", "soundfile",
        "scikit-learn", "torch", "torchlibrosa", "librosa",
    ],
    "06_4_ast": [
        "numpy", "pandas", "matplotlib", "soundfile",
        "scikit-learn", "torch", "transformers", "librosa",
    ],
    "06_4_clap": [
        "numpy", "pandas", "matplotlib", "soundfile",
        "scikit-learn", "torch", "transformers", "laion-clap", "librosa",
    ],
}


@dataclass
class CheckResult:
    ok: bool
    missing: list[str]
    install_cmd: str
    notes: list[str]


def _import_name(pkg: str) -> str:
    return _PACKAGE_IMPORT_NAME.get(pkg, pkg.replace("-", "_"))


def _is_installed(pkg: str) -> bool:
    """find_spec 比 import 便宜：不会触发模块体执行。"""
    try:
        return importlib.util.find_spec(_import_name(pkg)) is not None
    except (ImportError, ValueError):
        # 极少数包（如 numpy ABI 不兼容）find_spec 也会抛错——按缺失处理
        return False


def _ctmp_output_dir() -> Path:
    """与 ctmp_loader 完全一致的解析逻辑，避免循环 import。"""
    env = os.environ.get("CTMP_OUTPUT_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent.parent / "_data_pipeline" / "output"


def check_environment(
    notebook: Optional[str] = None,
    extra_packages: Iterable[str] = (),
    require_ctmp: bool = False,
    require_ctis_cache: bool = False,
    ctmp_seeds: Iterable[int] = (0, 1, 2),
    raise_on_error: bool = True,
) -> CheckResult:
    """对当前 notebook 做依赖与数据齐备性检查。

    Args:
        notebook: PRESETS 的 key（如 '06_1b'）；不传则只检查 extra_packages。
        extra_packages: 临时追加的依赖（pip 名）。
        require_ctmp: 检查 _data_pipeline/output/ 与 segment_manifest_seed{*}.csv。
        require_ctis_cache: 检查 datasets/CCMUSIC_CTIS/default/ 是否已落盘。
        ctmp_seeds: 检查这几个 seed 的 manifest 是否齐全。
        raise_on_error: True 时缺失即抛 RuntimeError；False 时只返回 CheckResult。

    Returns:
        CheckResult。打印好友提示并按 raise_on_error 决定是否抛错。
    """
    packages: list[str] = []
    if notebook is not None:
        if notebook not in PRESETS:
            raise KeyError(f"未知 notebook preset: {notebook}（合法：{sorted(PRESETS)}）")
        packages.extend(PRESETS[notebook])
    packages.extend(extra_packages)

    # 去重保序
    seen: set[str] = set()
    packages = [p for p in packages if not (p in seen or seen.add(p))]

    missing = [p for p in packages if not _is_installed(p)]
    install_cmd = ""
    if missing:
        install_cmd = f"pip install {' '.join(missing)}"

    notes: list[str] = []
    if require_ctmp:
        out_dir = _ctmp_output_dir()
        if not out_dir.exists():
            notes.append(
                f"[CTMP] 找不到清洗输出目录：{out_dir}\n"
                f"        请先按 CODE/chapter06/_data_pipeline/README.md 跑一遍 pipeline，"
                f"或设置环境变量 CTMP_OUTPUT_DIR 指向已有 output/。"
            )
        else:
            seeds_missing = [
                s for s in ctmp_seeds
                if not (out_dir / f"segment_manifest_seed{s}.csv").exists()
            ]
            if seeds_missing:
                notes.append(
                    f"[CTMP] {out_dir} 存在，但缺少 seed manifest：{seeds_missing}\n"
                    f"        重新执行 pipeline 时确保 freeze_config.yaml 的 seeds 包含这几个值。"
                )

    if require_ctis_cache:
        ctis_cache = Path(__file__).resolve().parents[2] / "datasets" / "CCMUSIC_CTIS" / "default"
        if not ctis_cache.exists():
            notes.append(
                f"[CTIS] 数据缓存不存在：{ctis_cache}\n"
                f"        06_1a 首次运行会自动从 HuggingFace 下载 ccmusic-database/CTIS"
                f"（约 600 MB），需要联网。"
            )

    ok = not missing and not notes
    result = CheckResult(ok=ok, missing=missing, install_cmd=install_cmd, notes=notes)

    _print_report(result, packages)

    if not ok and raise_on_error:
        msg_parts = []
        if missing:
            msg_parts.append(f"缺失依赖 {missing}，请运行：{install_cmd}")
        if notes:
            msg_parts.append("数据未就绪：\n  - " + "\n  - ".join(notes))
        raise RuntimeError("环境检查未通过。\n" + "\n".join(msg_parts))

    return result


def _print_report(result: CheckResult, checked: list[str]) -> None:
    if result.ok:
        pkgs = ", ".join(checked) if checked else "(无)"
        print(f"[env_check] OK — 依赖齐全：{pkgs}")
        return

    print("[env_check] 检测到问题：")
    if result.missing:
        print(f"  - 缺失 Python 包：{result.missing}")
        print(f"    安装命令：{result.install_cmd}")
        print("    （在当前 venv 内运行；conda 用户可改用 `conda install -c conda-forge ...`）")
    for note in result.notes:
        print(f"  - {note}")
