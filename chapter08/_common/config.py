"""YAML configuration helpers."""
from __future__ import annotations

from copy import deepcopy
import collections
import collections.abc
from pathlib import Path
from typing import Any, Mapping

import yaml

if not hasattr(collections, "Hashable"):
    collections.Hashable = collections.abc.Hashable


def load_yaml_config(path: Path | str) -> dict[str, Any]:
    """Load a YAML file into a plain dictionary."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def deep_update(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating inputs."""
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def get_by_dotted_key(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Read ``a.b.c`` from nested dictionaries."""
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def set_by_dotted_key(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``a.b.c`` in a nested dictionary, creating intermediate dicts."""
    current = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
        if not isinstance(current, dict):
            raise ValueError(f"Cannot set nested key through non-dict field: {part}")
    current[parts[-1]] = value
