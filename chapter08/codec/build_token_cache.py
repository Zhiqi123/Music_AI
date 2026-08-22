"""Build EnCodec token caches from FMA small or a manifest of audio files."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

from _common.dataset_registry import check_required_assets, iter_fma_audio_files
from _common.device_utils import choose_device
from _common.paths import portable_path
from _common.tables import write_rows
from codec.encodec_adapter import EncodecAdapter, check_encodec, write_encodec_artifacts


CHAPTER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FMA_AUDIO_MANIFEST = CHAPTER_ROOT / "data_manifests" / "fma_small_subset.csv"


def build_fma_token_cache(
    token_dir: Path | str = "outputs/generated/codec_tokens",
    manifest_csv: Path | str = "outputs/tables/08_5_token_cache_manifest.csv",
    audio_manifest_csv: Path | str | None = None,
    limit: int = 32,
    model_name: str = "24khz",
    bandwidth: float = 6.0,
    device: str = "auto",
    save_reconstruction: bool = False,
    skip_existing: bool = True,
) -> list[dict[str, object]]:
    """Encode FMA small tracks into cached EnCodec token tensors."""
    check_required_assets(
        ["fma_small"],
        message="FMA small is required to build the Chapter 8 Codec-LM token cache.",
        stop=True,
    )
    status = check_encodec()
    if not status.available:
        raise RuntimeError(f"{status.reason}. {status.next_action}")
    audio_paths = resolve_fma_audio_paths(audio_manifest_csv=audio_manifest_csv, limit=limit)
    return build_token_cache(
        audio_paths=audio_paths,
        token_dir=token_dir,
        manifest_csv=manifest_csv,
        model_name=model_name,
        bandwidth=bandwidth,
        device=device,
        save_reconstruction=save_reconstruction,
        skip_existing=skip_existing,
    )


def resolve_fma_audio_paths(
    audio_manifest_csv: Path | str | None = None,
    limit: int = 32,
) -> list[Path]:
    """Resolve the FMA audio slice, preferring the Chapter 8 manifest."""
    manifest = Path(audio_manifest_csv) if audio_manifest_csv else DEFAULT_FMA_AUDIO_MANIFEST
    if not manifest.is_absolute():
        cwd_candidate = Path.cwd() / manifest
        manifest = cwd_candidate if cwd_candidate.exists() else CHAPTER_ROOT / manifest
    if manifest.exists():
        return load_audio_manifest(manifest, limit=limit)
    if audio_manifest_csv is not None:
        raise FileNotFoundError(f"Audio manifest not found: {manifest}")
    return list(iter_fma_audio_files(limit=limit))


def load_audio_manifest(manifest_csv: Path | str, limit: int | None = None) -> list[Path]:
    """Load audio file paths from a manifest with paths relative to the CSV."""
    manifest_csv = Path(manifest_csv)
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
            if limit is not None and len(rows) >= limit:
                break
    missing = [path for path in rows if not path.exists()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(f"Manifest references missing audio file(s): {preview}")
    return rows


def build_token_cache(
    audio_paths: Iterable[Path | str],
    token_dir: Path | str,
    manifest_csv: Path | str,
    model_name: str = "24khz",
    bandwidth: float = 6.0,
    device: str = "auto",
    save_reconstruction: bool = False,
    skip_existing: bool = True,
    adapter: EncodecAdapter | None = None,
) -> list[dict[str, object]]:
    """Encode audio files into token tensors and write a manifest."""
    token_dir = Path(token_dir)
    manifest_csv = Path(manifest_csv)
    resolved_device = choose_device(device)
    adapter = adapter or EncodecAdapter(model_name=model_name, bandwidth=bandwidth, device=resolved_device)
    rows: list[dict[str, object]] = []
    for index, audio_path in enumerate(audio_paths, start=1):
        audio_path = Path(audio_path)
        paths = token_output_paths(audio_path, token_dir)
        if skip_existing and paths["tokens"].exists():
            codes_shape = cached_codes_shape(paths["tokens"])
            row = manifest_row(
                index=index,
                audio_path=audio_path,
                token_path=paths["tokens"],
                metadata_path=paths["metadata"],
                reconstruction_path=paths["reconstruction"] if save_reconstruction else None,
                model_name=model_name,
                bandwidth=bandwidth,
                device=resolved_device,
                status="cached",
                codes_shape=codes_shape,
            )
            rows.append(row)
            continue
        result = adapter.encode_decode_file(audio_path)
        paths = write_encodec_artifacts(
            audio_path,
            result,
            token_dir,
            save_reconstruction=save_reconstruction,
        )
        row = manifest_row(
            index=index,
            audio_path=audio_path,
            token_path=paths["tokens"],
            metadata_path=paths["metadata"],
            reconstruction_path=paths["reconstruction"] if save_reconstruction else None,
            model_name=model_name,
            bandwidth=bandwidth,
            device=resolved_device,
            status="encoded",
            codes_shape="x".join(str(dim) for dim in result.codes.shape),
        )
        rows.append(row)
    write_rows(manifest_csv, rows, fieldnames=TOKEN_CACHE_FIELDS)
    return rows


TOKEN_CACHE_FIELDS = [
    "index",
    "status",
    "audio_path",
    "token_path",
    "metadata_path",
    "reconstruction_path",
    "model_name",
    "bandwidth",
    "device",
    "codes_shape",
]


def token_output_paths(audio_path: Path | str, token_dir: Path | str) -> dict[str, Path]:
    """Import-safe wrapper around the adapter path convention."""
    from codec.encodec_adapter import encodec_output_paths

    return encodec_output_paths(audio_path, token_dir)


def cached_codes_shape(token_path: Path | str) -> str:
    """Read a cached token tensor and return its manifest shape string."""
    import torch

    token_path = Path(token_path)
    try:
        cached = torch.load(token_path, map_location="cpu", weights_only=True)
    except TypeError:
        cached = torch.load(token_path, map_location="cpu")
    if isinstance(cached, dict) and "codes" in cached:
        cached = cached["codes"]
    shape = getattr(cached, "shape", None)
    if shape is None:
        raise ValueError(f"Cached token file does not expose a tensor shape: {token_path}")
    return "x".join(str(dim) for dim in shape)


def manifest_row(
    index: int,
    audio_path: Path,
    token_path: Path,
    metadata_path: Path,
    reconstruction_path: Path | None,
    model_name: str,
    bandwidth: float,
    device: str,
    status: str,
    codes_shape: str,
) -> dict[str, object]:
    """Return one token-cache manifest row."""
    return {
        "index": index,
        "status": status,
        "audio_path": portable_path(audio_path, CHAPTER_ROOT),
        "token_path": portable_path(token_path, CHAPTER_ROOT),
        "metadata_path": portable_path(metadata_path, CHAPTER_ROOT),
        "reconstruction_path": portable_path(reconstruction_path, CHAPTER_ROOT),
        "model_name": model_name,
        "bandwidth": bandwidth,
        "device": device,
        "codes_shape": codes_shape,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chapter 8 EnCodec token cache.")
    parser.add_argument("--token-dir", default="outputs/generated/codec_tokens")
    parser.add_argument("--manifest-csv", default="outputs/tables/08_5_token_cache_manifest.csv")
    parser.add_argument(
        "--audio-manifest",
        default=None,
        help="Audio-source manifest with a path column. Defaults to data_manifests/fma_small_subset.csv when present.",
    )
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--model-name", default="24khz", choices=["24khz", "48khz"])
    parser.add_argument("--bandwidth", type=float, default=6.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-reconstruction", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    rows = build_fma_token_cache(
        token_dir=args.token_dir,
        manifest_csv=args.manifest_csv,
        audio_manifest_csv=args.audio_manifest,
        limit=args.limit,
        model_name=args.model_name,
        bandwidth=args.bandwidth,
        device=args.device,
        save_reconstruction=args.save_reconstruction,
        skip_existing=not args.overwrite,
    )
    print(f"Wrote {len(rows)} token-cache rows to {args.manifest_csv}")


if __name__ == "__main__":
    main()
