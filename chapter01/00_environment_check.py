#!/usr/bin/env python3
"""报告第一章使用的解释器与轻量依赖。"""

from __future__ import annotations

import importlib
import platform
import sys
from importlib import metadata
from pathlib import Path


TESTED_PYTHON = (3, 11, 9)
REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "ipykernel": "ipykernel",
    "nbconvert": "nbconvert",
}


def print_environment() -> None:
    """打印用于识别当前解释器和工作目录的信息。"""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    print("第 1 章环境检查")
    print("=" * 30)
    print(f"Python 版本  : {platform.python_version()}")
    print(f"Python 实现  : {platform.python_implementation()}")
    print(f"sys.executable: {sys.executable}")
    print(f"当前 cwd     : {Path.cwd().resolve()}")
    print(f"项目根目录   : {project_root}")
    print(f"操作系统平台 : {platform.platform()}")


def check_python_version() -> bool:
    """返回当前解释器是否属于本章采用的 Python 3.11 系列。"""
    current = sys.version_info[:3]
    if current == TESTED_PYTHON:
        version = ".".join(str(part) for part in current)
        print(f"[通过] Python {version} 与实测基线完全一致。")
        return True

    expected = ".".join(str(part) for part in TESTED_PYTHON)
    actual = ".".join(str(part) for part in current)
    if current[:2] == TESTED_PYTHON[:2]:
        print(f"[提示] 当前为 Python {actual}；实测补丁版本是 {expected}。")
        print("       二者同属 Python 3.11，但仍应重新执行本章 Notebook 核对结果。")
        return True

    print(f"[警告] Python {actual} 不在本章实测的 3.11 系列内。")
    print(f"       请使用 Python {expected} 建立环境，再重新运行本脚本。")
    return False


def check_packages() -> tuple[list[str], list[str]]:
    """检查所需发行包的版本元数据与实际导入。"""
    missing: list[str] = []
    broken: list[str] = []

    print("\n本章依赖")
    print("-" * 30)
    for distribution_name, import_name in REQUIRED_PACKAGES.items():
        try:
            version = metadata.version(distribution_name)
        except metadata.PackageNotFoundError:
            print(f"[缺少] {distribution_name}")
            missing.append(distribution_name)
            continue

        try:
            importlib.import_module(import_name)
        except ImportError as error:
            print(f"[导入失败] {distribution_name} {version}: {error}")
            broken.append(distribution_name)
            continue

        print(f"[通过] {distribution_name} {version}")

    return missing, broken


def main() -> int:
    """运行全部检查，并返回适合终端使用的进程状态码。"""
    print_environment()
    python_matches = check_python_version()
    missing, broken = check_packages()

    if missing or broken:
        requirements_path = Path(__file__).resolve().with_name("requirements.txt")
        print("\n本章环境不完整。")
        if not python_matches:
            print("请先建立并激活 Python 3.11 环境，然后运行：")
            print(f'  python -m pip install -r "{requirements_path}"')
        elif broken:
            print("当前解释器中的包无法正常导入，可先重新安装本章依赖：")
            print(f'  "{sys.executable}" -m pip install --force-reinstall -r "{requirements_path}"')
        else:
            print("在当前环境中安装缺少的依赖：")
            print(f'  "{sys.executable}" -m pip install -r "{requirements_path}"')
        return 1

    if not python_matches:
        print("\n依赖均可导入，但当前 Python 不属于本章采用的版本系列。")
        return 2

    print("\n第 1 章环境检查通过。")
    print("请在每个 Notebook 中再次打印 sys.executable，并与上方路径比较。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
