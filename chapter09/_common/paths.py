"""面向读者的路径格式化,避免在输出里泄露机器特定的绝对路径。"""
from __future__ import annotations

from pathlib import Path

CHAPTER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CHAPTER_ROOT.parents[1]


def portable_path(value: Path | str | None, primary_root: Path | str | None = None) -> str:
    """把绝对路径转成相对项目根的可读形式。"""
    if value is None:
        return ""
    text = str(value)
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        return path.as_posix()
    roots = unique_roots(primary_root, CHAPTER_ROOT, PROJECT_ROOT)
    candidates = [path.absolute(), path.resolve()]
    for root in roots:
        root_candidates = [root.absolute(), root.resolve()]
        for candidate in candidates:
            for root_candidate in root_candidates:
                try:
                    return candidate.relative_to(root_candidate).as_posix()
                except ValueError:
                    continue
    return path.as_posix()


def unique_roots(*roots: Path | str | None) -> list[Path]:
    """保序去重,忽略空值。"""
    result: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        if root is None:
            continue
        path = Path(root)
        key = str(path.resolve())
        if key not in seen:
            result.append(path)
            seen.add(key)
    return result
