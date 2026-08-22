"""Dataset path helpers for optional Chapter 7 data."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .audio_io import load_audio


CODE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MUSDB_ROOT = CODE_ROOT / "datasets" / "MUSDB18-HQ"
DEFAULT_AUTHOR_AUDIO_ROOT = CODE_ROOT / "datasets" / "audio_author"


def find_musdb_root(config_path: str | Path | None = None) -> Path | None:
    """Return an existing MUSDB18-HQ root, or ``None``."""
    candidates = []
    env_path = os.environ.get("MUSDB18HQ_ROOT")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    config_root = _path_from_config(config_path, "musdb18hq_root")
    if config_root is not None:
        candidates.append(config_root)
    candidates.extend(
        [
            DEFAULT_MUSDB_ROOT,
            CODE_ROOT / "datasets" / "musdb18-hq",
            CODE_ROOT / "datasets" / "MUSDB18-HQ",
            CODE_ROOT / "datasets" / "musdb18hq",
        ]
    )

    for path in candidates:
        path = path.expanduser().resolve()
        if (
            _case_exact_path_exists(path)
            and any((path / split).exists() for split in ("test", "train", "valid"))
        ):
            return path
    return None


def find_author_audio_root(config_path: str | Path | None = None) -> Path | None:
    """Return an existing author-audio root, or ``None``."""
    candidates = []
    env_path = os.environ.get("CHAPTER07_AUTHOR_AUDIO_ROOT")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    config_root = _path_from_config(config_path, "audio_author_root")
    if config_root is not None:
        candidates.append(config_root)
    candidates.append(DEFAULT_AUTHOR_AUDIO_ROOT)

    for path in candidates:
        path = path.expanduser().resolve()
        if path.exists():
            return path
    return None


def _case_exact_path_exists(path: Path) -> bool:
    """Return True only when the final path component exists with matching case."""
    if not path.exists():
        return False
    try:
        return any(child.name == path.name for child in path.parent.iterdir())
    except OSError:
        return True


def list_musdb_tracks(root: Path, split: str = "test") -> list[Path]:
    """List MUSDB track directories containing ``mixture.wav``."""
    split_dir = Path(root) / split
    if not split_dir.exists():
        return []
    return sorted(path for path in split_dir.iterdir() if (path / "mixture.wav").exists())


def load_musdb_track(
    track_dir: Path,
    duration: float | None = 30.0,
    sr: int | None = None,
    start: float = 0.0,
    mono: bool = False,
) -> dict[str, np.ndarray]:
    """Load available MUSDB stems from a track directory."""
    stems = {}
    target_sr = sr
    for stem in ("mixture", "vocals", "drums", "bass", "other"):
        path = Path(track_dir) / f"{stem}.wav"
        if path.exists():
            audio, loaded_sr = load_audio(
                path,
                sr=target_sr,
                mono=mono,
                start=start,
                duration=duration,
            )
            if target_sr is None:
                target_sr = loaded_sr
            stems[stem] = audio
    return stems


def _path_from_config(config_path: str | Path | None, key: str) -> Path | None:
    if config_path is None:
        return None
    path = Path(config_path).expanduser()
    if not path.exists():
        return None

    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        value = data.get(key)
    except Exception:
        value = _simple_yaml_value(path, key)

    if not value:
        return None
    value_path = Path(str(value)).expanduser()
    if value_path.is_absolute():
        return value_path
    return (path.parent / value_path).resolve()


def _simple_yaml_value(path: Path, key: str) -> str | None:
    prefix = f"{key}:"
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.split("#", 1)[0].strip()
        if clean.startswith(prefix):
            return clean[len(prefix) :].strip().strip("'\"") or None
    return None
