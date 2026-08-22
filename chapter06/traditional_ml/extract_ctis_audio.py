"""从 CCMUSIC CTIS default/train Arrow 文件中导出嵌入式音频为 .wav。

输出布局：
    <output-dir>/
      <cname>/
        <audio_path>.wav
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Optional, Tuple

try:
    import pyarrow as pa
    import pyarrow.ipc as ipc
except ModuleNotFoundError:
    print(
        "Missing dependency: pyarrow\n"
        "Install it with:\n"
        "  python3 -m pip install pyarrow\n",
        file=sys.stderr,
    )
    raise SystemExit(1)


DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "CCMUSIC_CTIS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract all embedded audio from CTIS default/train into instrument folders."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=(
            "CTIS dataset root (containing default/train/*.arrow). "
            f"Defaults to the in-repo cache: {DEFAULT_DATASET_ROOT}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <dataset-root>/audio_by_cname",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files if present.",
    )
    return parser.parse_args()


def iter_record_batches(arrow_files: Iterable[Path]) -> Iterator[Tuple[Path, pa.RecordBatch]]:
    for arrow_file in sorted(arrow_files):
        with pa.memory_map(str(arrow_file), "r") as source:
            for batch in ipc.open_stream(source):
                yield arrow_file, batch


def sanitize_name(name: str) -> str:
    bad_chars = '<>:"/\\|?*\0'
    sanitized = "".join("_" if ch in bad_chars else ch for ch in name).strip()
    return sanitized or "unnamed"


def ensure_wav_suffix(name: str) -> str:
    return name if name.lower().endswith(".wav") else f"{name}.wav"


def struct_path(struct_value: Optional[Mapping[str, object]]) -> str:
    return "" if not struct_value else str(struct_value.get("path") or "")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else (dataset_root / "audio_by_cname").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    default_dir = dataset_root / "default" / "train"
    if not default_dir.exists():
        raise SystemExit(f"Missing directory: {default_dir}")

    total_rows = 0
    written = 0
    skipped = 0

    for _, batch in iter_record_batches(default_dir.glob("*.arrow")):
        columns = {name: batch.column(i).to_pylist() for i, name in enumerate(batch.schema.names)}
        for audio, cname in zip(columns["audio"], columns["cname"]):
            total_rows += 1
            audio_bytes = None if not audio else audio.get("bytes")
            audio_path = struct_path(audio)
            if not audio_bytes:
                skipped += 1
                continue

            instrument_dir = output_dir / sanitize_name(str(cname))
            instrument_dir.mkdir(parents=True, exist_ok=True)

            filename = ensure_wav_suffix(sanitize_name(audio_path or f"sample_{total_rows}"))
            out_path = instrument_dir / filename

            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue

            out_path.write_bytes(audio_bytes)
            written += 1

    print("Done.")
    print(f"Dataset root: {dataset_root}")
    print(f"Output dir:   {output_dir}")
    print(f"Total rows seen: {total_rows}")
    print(f"Audio files written: {written}")
    print(f"Skipped: {skipped}")
    print("Note: eval splits do not contain audio and are not extracted.")


if __name__ == "__main__":
    main()
