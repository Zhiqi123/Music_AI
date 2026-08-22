"""Local dataset discovery and required-asset checks for Chapter 8."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from _common.paths import portable_path, portable_path_list


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "CODE"
DATASETS_ROOT = CODE_ROOT / "datasets"


class MissingAssetError(RuntimeError):
    """Raised when a required local dataset or asset pack is missing."""


@dataclass(frozen=True)
class AssetSpec:
    asset_id: str
    label: str
    required_paths: tuple[str, ...]
    download_hint: str
    require_all: bool = True


@dataclass(frozen=True)
class AssetStatus:
    spec: AssetSpec
    project_root: Path
    existing_paths: tuple[Path, ...]

    @property
    def ok(self) -> bool:
        expected = [self.project_root / p for p in self.spec.required_paths]
        if self.spec.require_all:
            return all(path.exists() for path in expected)
        return any(path.exists() for path in expected)

    @property
    def missing_paths(self) -> tuple[Path, ...]:
        return tuple(
            self.project_root / p
            for p in self.spec.required_paths
            if not (self.project_root / p).exists()
        )


@dataclass(frozen=True)
class AssetCheckResult:
    statuses: tuple[AssetStatus, ...]
    message: str = ""

    @property
    def ok(self) -> bool:
        return all(status.ok for status in self.statuses)

    @property
    def missing(self) -> tuple[AssetStatus, ...]:
        return tuple(status for status in self.statuses if not status.ok)

    def format_message(self) -> str:
        if self.ok:
            return "All required Chapter 8 assets are available."
        lines = []
        if self.message:
            lines.append(self.message)
        lines.append("Missing required Chapter 8 assets:")
        for status in self.missing:
            lines.append(f"- {status.spec.asset_id}: {status.spec.label}")
            lines.append(f"  expected: {', '.join(portable_path(p, status.project_root) for p in status.missing_paths)}")
            lines.append(f"  download: {status.spec.download_hint}")
        return "\n".join(lines)

    def raise_if_missing(self) -> None:
        if not self.ok:
            raise MissingAssetError(self.format_message())


ASSETS: dict[str, AssetSpec] = {
    "nsynth": AssetSpec(
        asset_id="nsynth",
        label="NSynth JSON/WAV dataset",
        required_paths=(
            "CODE/datasets/nsynth/nsynth-valid/examples.json",
            "CODE/datasets/nsynth/nsynth-valid/audio",
        ),
        download_hint=(
            "hf download confit/nsynth nsynth-valid.jsonwav.tar.gz "
            "--repo-type dataset --local-dir CODE/datasets/nsynth && "
            "tar -xzf CODE/datasets/nsynth/nsynth-valid.jsonwav.tar.gz -C CODE/datasets/nsynth"
        ),
    ),
    "nsynth_train": AssetSpec(
        asset_id="nsynth_train",
        label="NSynth train split in JSON/WAV format",
        required_paths=(
            "CODE/datasets/nsynth/nsynth-train/examples.json",
            "CODE/datasets/nsynth/nsynth-train/audio",
        ),
        download_hint=(
            "hf download confit/nsynth nsynth-train.jsonwav.tar.gz "
            "--repo-type dataset --local-dir CODE/datasets/nsynth && "
            "tar -xzf CODE/datasets/nsynth/nsynth-train.jsonwav.tar.gz -C CODE/datasets/nsynth"
        ),
    ),
    "nsynth_test": AssetSpec(
        asset_id="nsynth_test",
        label="NSynth test split in JSON/WAV format",
        required_paths=(
            "CODE/datasets/nsynth/nsynth-test/examples.json",
            "CODE/datasets/nsynth/nsynth-test/audio",
        ),
        download_hint=(
            "hf download confit/nsynth nsynth-test.jsonwav.tar.gz "
            "--repo-type dataset --local-dir CODE/datasets/nsynth && "
            "tar -xzf CODE/datasets/nsynth/nsynth-test.jsonwav.tar.gz -C CODE/datasets/nsynth"
        ),
    ),
    "fma_small": AssetSpec(
        asset_id="fma_small",
        label="FMA small audio and metadata",
        required_paths=(
            "CODE/datasets/FMA_small/fma_small/fma_small",
            "CODE/datasets/FMA_small/fma_metadata/fma_metadata/tracks.csv",
        ),
        download_hint=(
            "Download fma_small.zip and fma_metadata.zip, then extract them under "
            "CODE/datasets/FMA_small/"
        ),
    ),
    "ctis": AssetSpec(
        asset_id="ctis",
        label="CCMUSIC CTIS Chinese instrument audio",
        required_paths=("CODE/datasets/CCMUSIC_CTIS",),
        download_hint="Place CCMUSIC_CTIS under CODE/datasets/CCMUSIC_CTIS/.",
    ),
    "chmusic": AssetSpec(
        asset_id="chmusic",
        label="ChMusic Chinese music examples",
        required_paths=("CODE/datasets/ChMusic",),
        download_hint="Place ChMusic under CODE/datasets/ChMusic/.",
    ),
    "audio_author_ch08": AssetSpec(
        asset_id="audio_author_ch08",
        label="Chapter 8 author-owned audio pack",
        required_paths=("CODE/datasets/audio_author/chapter_08_author",),
        download_hint=(
            "Download the author asset pack and extract it to "
            "CODE/datasets/audio_author/chapter_08_author/."
        ),
    ),
}


def check_required_assets(
    asset_ids: list[str],
    message: str = "",
    stop: bool = True,
    project_root: Path | str | None = None,
) -> AssetCheckResult:
    """Check local assets and raise with notebook-ready instructions when requested."""
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    statuses = tuple(asset_status(asset_id, root) for asset_id in asset_ids)
    result = AssetCheckResult(statuses=statuses, message=message)
    if not result.ok:
        print(result.format_message())
        if stop:
            result.raise_if_missing()
    return result


def asset_status(asset_id: str, project_root: Path | str | None = None) -> AssetStatus:
    """Return availability information for one registered asset."""
    if asset_id not in ASSETS:
        raise KeyError(f"Unknown Chapter 8 asset id: {asset_id}")
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    spec = ASSETS[asset_id]
    existing = tuple(root / p for p in spec.required_paths if (root / p).exists())
    return AssetStatus(spec=spec, project_root=root, existing_paths=existing)


def list_assets(project_root: Path | str | None = None) -> list[dict[str, str | bool]]:
    """Return registered asset statuses as table-ready dictionaries."""
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    rows = []
    for asset_id in sorted(ASSETS):
        status = asset_status(asset_id, root)
        rows.append(
            {
                "asset_id": asset_id,
                "label": status.spec.label,
                "available": status.ok,
                "paths": portable_path_list(status.existing_paths, primary_root=root),
                "download_hint": status.spec.download_hint,
            }
        )
    return rows


def nsynth_split_root(split: str = "valid", project_root: Path | str | None = None) -> Path:
    """Return the local NSynth split root."""
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    return root / "CODE" / "datasets" / "nsynth" / f"nsynth-{split}"


def load_nsynth_metadata(
    split: str = "valid",
    limit: int | None = None,
    project_root: Path | str | None = None,
) -> list[dict[str, object]]:
    """Load NSynth metadata rows and attach ``audio_path``."""
    split_root = nsynth_split_root(split, project_root)
    examples_path = split_root / "examples.json"
    if not examples_path.exists():
        raise MissingAssetError(f"Missing NSynth metadata: {examples_path}")
    with examples_path.open("r", encoding="utf-8") as handle:
        examples = json.load(handle)
    rows = []
    for note_id, metadata in examples.items():
        row = dict(metadata)
        row["note_id"] = note_id
        row["audio_path"] = split_root / "audio" / f"{note_id}.wav"
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def find_fma_audio_root(project_root: Path | str | None = None) -> Path:
    """Find the directory that directly contains FMA small shard folders."""
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    candidates = [
        root / "CODE" / "datasets" / "FMA_small" / "fma_small" / "fma_small",
        root / "CODE" / "datasets" / "FMA_small" / "fma_small",
        root / "CODE" / "datasets" / "FMA_small",
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.rglob("*.mp3")):
            return candidate
    raise MissingAssetError("Could not find FMA small audio files under CODE/datasets/FMA_small/")


def find_fma_metadata_root(project_root: Path | str | None = None) -> Path:
    """Find the FMA metadata directory containing ``tracks.csv``."""
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    candidates = [
        root / "CODE" / "datasets" / "FMA_small" / "fma_metadata" / "fma_metadata",
        root / "CODE" / "datasets" / "FMA_small" / "fma_metadata",
        root / "CODE" / "datasets" / "FMA_small",
    ]
    for candidate in candidates:
        if (candidate / "tracks.csv").exists():
            return candidate
    raise MissingAssetError("Could not find FMA metadata tracks.csv.")


def iter_fma_audio_files(
    limit: int | None = None,
    project_root: Path | str | None = None,
) -> Iterable[Path]:
    """Yield FMA small MP3 paths in stable order."""
    audio_root = find_fma_audio_root(project_root)
    count = 0
    for path in sorted(audio_root.rglob("*.mp3")):
        yield path
        count += 1
        if limit is not None and count >= limit:
            return
