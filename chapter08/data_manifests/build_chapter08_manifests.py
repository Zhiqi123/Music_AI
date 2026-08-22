"""Build real Chapter 8 subset manifests from local datasets."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

CHAPTER_ROOT = Path(__file__).resolve().parents[1]
if str(CHAPTER_ROOT) not in sys.path:
    sys.path.insert(0, str(CHAPTER_ROOT))

from _common.dataset_registry import (  # noqa: E402
    find_fma_audio_root,
    find_fma_metadata_root,
    load_nsynth_metadata,
)
from _common.tables import write_rows  # noqa: E402


NSYNTH_FIELDS = [
    "note_id",
    "path",
    "split",
    "instrument_family",
    "instrument_source",
    "pitch",
    "velocity",
    "sample_rate",
    "license_status",
    "notes",
]

FMA_FIELDS = [
    "track_id",
    "path",
    "genre_top",
    "split",
    "subset",
    "duration_sec",
    "license_status",
    "notes",
]


def build_nsynth_subset_manifest(
    out_csv: Path | str = CHAPTER_ROOT / "data_manifests" / "nsynth_subset.csv",
    split: str = "valid",
    limit: int = 256,
    project_root: Path | str | None = None,
) -> list[dict[str, object]]:
    """Write a stable NSynth subset manifest."""
    out_csv = Path(out_csv)
    rows = []
    for row in load_nsynth_metadata(split=split, limit=limit, project_root=project_root):
        audio_path = Path(row["audio_path"])
        rows.append(
            {
                "note_id": row["note_id"],
                "path": relative_to_manifest(audio_path, out_csv),
                "split": split,
                "instrument_family": row.get("instrument_family_str", ""),
                "instrument_source": row.get("instrument_source_str", ""),
                "pitch": row.get("pitch", ""),
                "velocity": row.get("velocity", ""),
                "sample_rate": row.get("sample_rate", ""),
                "license_status": "nsynth_license",
                "notes": "",
            }
        )
    write_rows(out_csv, rows, fieldnames=NSYNTH_FIELDS)
    return rows


def build_fma_small_subset_manifest(
    out_csv: Path | str = CHAPTER_ROOT / "data_manifests" / "fma_small_subset.csv",
    limit: int = 512,
    project_root: Path | str | None = None,
) -> list[dict[str, object]]:
    """Write a stable FMA small subset manifest with metadata fields."""
    out_csv = Path(out_csv)
    audio_root = find_fma_audio_root(project_root)
    metadata = load_fma_tracks_metadata(project_root)
    rows = []
    for track_id in sorted(metadata):
        meta = metadata[track_id]
        if meta.get("set", {}).get("subset") != "small":
            continue
        audio_path = fma_audio_path_for_track_id(track_id, audio_root)
        if not audio_path.exists():
            continue
        track_meta = meta.get("track", {})
        set_meta = meta.get("set", {})
        rows.append(
            {
                "track_id": track_id,
                "path": relative_to_manifest(audio_path, out_csv),
                "genre_top": track_meta.get("genre_top", ""),
                "split": set_meta.get("split", ""),
                "subset": set_meta.get("subset", ""),
                "duration_sec": track_meta.get("duration", ""),
                "license_status": track_meta.get("license", ""),
                "notes": "",
            }
        )
        if len(rows) >= limit:
            break
    write_rows(out_csv, rows, fieldnames=FMA_FIELDS)
    return rows


def load_fma_tracks_metadata(project_root: Path | str | None = None) -> dict[int, dict[str, dict[str, str]]]:
    """Parse FMA tracks.csv into ``track_id -> top_level -> field -> value``."""
    tracks_csv = find_fma_metadata_root(project_root) / "tracks.csv"
    with tracks_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        top = next(reader)
        sub = next(reader)
        next(reader)
        metadata: dict[int, dict[str, dict[str, str]]] = {}
        for raw in reader:
            if not raw or not raw[0].strip():
                continue
            track_id = int(raw[0])
            entry: dict[str, dict[str, str]] = {}
            for top_key, sub_key, value in zip(top[1:], sub[1:], raw[1:]):
                if not top_key or not sub_key:
                    continue
                entry.setdefault(top_key, {})[sub_key] = value
            metadata[track_id] = entry
    return metadata


def fma_audio_path_for_track_id(track_id: int, audio_root: Path | str) -> Path:
    """Return the expected FMA small audio path for a track id."""
    filename = f"{int(track_id):06d}.mp3"
    return Path(audio_root) / filename[:3] / filename


def relative_to_manifest(path: Path | str, manifest_csv: Path | str) -> str:
    """Return a manifest-stable relative path."""
    path = Path(path).resolve()
    base = Path(manifest_csv).resolve().parent
    return os.path.relpath(path, start=base)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chapter 8 dataset manifests.")
    parser.add_argument("--nsynth-limit", type=int, default=256)
    parser.add_argument("--nsynth-split", default="valid")
    parser.add_argument("--fma-limit", type=int, default=512)
    parser.add_argument("--output-dir", default=str(CHAPTER_ROOT / "data_manifests"))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nsynth_rows = build_nsynth_subset_manifest(
        output_dir / "nsynth_subset.csv",
        split=args.nsynth_split,
        limit=args.nsynth_limit,
    )
    fma_rows = build_fma_small_subset_manifest(
        output_dir / "fma_small_subset.csv",
        limit=args.fma_limit,
    )
    print(f"Wrote {len(nsynth_rows)} NSynth rows")
    print(f"Wrote {len(fma_rows)} FMA small rows")


if __name__ == "__main__":
    main()
