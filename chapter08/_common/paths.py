"""Portable path formatting for Chapter 8 public outputs."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


CHAPTER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CHAPTER_ROOT.parents[1]


def portable_path(value: Path | str | None, primary_root: Path | str | None = None) -> str:
    """Return a reader-facing path without leaking a machine-specific root."""
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


def portable_path_list(
    values: Iterable[Path | str],
    primary_root: Path | str | None = None,
    separator: str = ";",
) -> str:
    """Join paths after formatting each one with ``portable_path``."""
    return separator.join(portable_path(value, primary_root=primary_root) for value in values)


def portable_text(value: object, primary_root: Path | str | None = None) -> str:
    """Replace known absolute roots inside a longer reader-facing string."""
    text = str(value)
    for root in unique_roots(primary_root, CHAPTER_ROOT, PROJECT_ROOT):
        resolved = root.resolve()
        replacement = ""
        if primary_root is None and resolved == CHAPTER_ROOT.resolve():
            replacement = "CODE/chapter08"
        prefix = str(resolved) + os.sep
        text = text.replace(prefix, replacement + ("/" if replacement else ""))
        text = text.replace(str(resolved), replacement or ".")
    return text


def command_relative_path(value: Path | str, cwd: Path | str) -> str:
    """Return a path suitable for a command executed from ``cwd``."""
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    cwd_path = Path(cwd).expanduser()
    if not cwd_path.is_absolute():
        cwd_path = (Path.cwd() / cwd_path).resolve()
    try:
        return Path(os.path.relpath(path.resolve(), start=cwd_path.resolve())).as_posix()
    except ValueError:
        return portable_path(path)


def unique_roots(*roots: Path | str | None) -> list[Path]:
    """Return roots in order, removing duplicates and blanks."""
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
