"""Datasets for toy WaveNet training."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from _common.audio_io import load_audio, mu_law_encode, normalize_peak, pad_or_trim
from _common.dataset_registry import check_required_assets, load_nsynth_metadata
from synthesis.waveforms import random_tone


CHAPTER_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AudioWindowConfig:
    sample_rate: int = 16000
    window_samples: int = 4096
    quantization_channels: int = 256


def waveform_to_training_pair(
    audio: np.ndarray,
    config: AudioWindowConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert one waveform into input and next-token targets."""
    audio = normalize_peak(pad_or_trim(audio, config.window_samples), peak=0.98)
    tokens = mu_law_encode(audio, config.quantization_channels)
    return (
        torch.as_tensor(tokens[:-1], dtype=torch.long),
        torch.as_tensor(tokens[1:], dtype=torch.long),
    )


class SyntheticWaveformDataset(Dataset):
    """Deterministic synthetic waveform windows for smoke tests and demos."""

    def __init__(
        self,
        num_examples: int = 128,
        config: AudioWindowConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.num_examples = int(num_examples)
        self.config = config or AudioWindowConfig()
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        duration = self.config.window_samples / self.config.sample_rate
        audio = random_tone(duration=duration, sr=self.config.sample_rate, seed=self.seed + index)
        return waveform_to_training_pair(audio, self.config)


class NSynthWaveformDataset(Dataset):
    """NSynth JSON/WAV windows for toy WaveNet training."""

    def __init__(
        self,
        split: str = "valid",
        max_files: int | None = 256,
        config: AudioWindowConfig | None = None,
        project_root: Path | str | None = None,
    ) -> None:
        self.config = config or AudioWindowConfig()
        check_required_assets(["nsynth"], stop=True, project_root=project_root)
        self.rows = load_nsynth_metadata(split=split, limit=max_files, project_root=project_root)
        if not self.rows:
            raise ValueError(f"No NSynth rows found for split={split!r}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        path = Path(row["audio_path"])
        audio, _ = load_audio(path, sr=self.config.sample_rate, mono=True)
        return waveform_to_training_pair(audio, self.config)


class ManifestWaveformDataset(Dataset):
    """Audio waveform windows from a CSV manifest with a ``path`` column."""

    def __init__(
        self,
        manifest_csv: Path | str,
        max_files: int | None = None,
        config: AudioWindowConfig | None = None,
    ) -> None:
        self.config = config or AudioWindowConfig()
        self.manifest_csv = Path(manifest_csv)
        self.rows = load_audio_manifest(self.manifest_csv, max_files=max_files)
        if not self.rows:
            raise ValueError(f"No audio rows found in manifest: {self.manifest_csv}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.rows[index]
        audio, _ = load_audio(path, sr=self.config.sample_rate, mono=True)
        return waveform_to_training_pair(audio, self.config)


def load_audio_manifest(manifest_csv: Path | str, max_files: int | None = None) -> list[Path]:
    """Load audio paths from a manifest, resolving relative paths beside the CSV."""
    manifest_csv = Path(manifest_csv)
    if not manifest_csv.is_absolute():
        cwd_candidate = Path.cwd() / manifest_csv
        manifest_csv = cwd_candidate if cwd_candidate.exists() else CHAPTER_ROOT / manifest_csv
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Audio manifest not found: {manifest_csv}")
    rows: list[Path] = []
    with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "path" not in (reader.fieldnames or []):
            raise ValueError(f"Audio manifest must include a 'path' column: {manifest_csv}")
        for raw in reader:
            value = (raw.get("path") or "").strip()
            if not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = manifest_csv.parent / path
            rows.append(path.resolve())
            if max_files is not None and len(rows) >= max_files:
                break
    missing = [path for path in rows if not path.exists()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(f"Manifest references missing audio file(s): {preview}")
    return rows


def build_wavenet_dataset(config: dict) -> Dataset:
    """Build the configured WaveNet dataset."""
    data_cfg = config.get("data", {})
    window_config = AudioWindowConfig(
        sample_rate=int(data_cfg.get("sample_rate", 16000)),
        window_samples=int(data_cfg.get("window_samples", 4096)),
        quantization_channels=int(data_cfg.get("quantization_channels", 256)),
    )
    source = data_cfg.get("source", "synthetic")
    manifest_csv = data_cfg.get("manifest_csv")
    if manifest_csv:
        return ManifestWaveformDataset(
            manifest_csv=manifest_csv,
            max_files=int(data_cfg.get("max_files", 256)),
            config=window_config,
        )
    if source == "synthetic":
        return SyntheticWaveformDataset(
            num_examples=int(data_cfg.get("num_examples", 128)),
            config=window_config,
            seed=int(config.get("seed", 0)),
        )
    if source == "nsynth":
        return NSynthWaveformDataset(
            split=str(data_cfg.get("split", "valid")),
            max_files=int(data_cfg.get("max_files", 256)),
            config=window_config,
        )
    raise ValueError(f"Unknown WaveNet data source: {source}")
