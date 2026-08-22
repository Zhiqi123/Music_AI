"""Build deterministic teaching audio-token caches from local audio files."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import torch

from _common.audio_io import load_audio
from _common.dataset_registry import check_required_assets
from _common.paths import portable_path
from _common.tables import write_rows
from codec.build_token_cache import resolve_fma_audio_paths


CHAPTER_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TeachingCodecConfig:
    sample_rate: int = 24000
    duration_sec: float = 4.0
    n_mels: int = 16
    n_fft: int = 1024
    hop_length: int = 512
    vocab_size: int = 256
    top_db: float = 80.0


TEACHING_TOKEN_CACHE_FIELDS = [
    "index",
    "status",
    "audio_path",
    "token_path",
    "metadata_path",
    "tokenizer",
    "sample_rate",
    "duration_sec",
    "n_mels",
    "hop_length",
    "n_fft",
    "vocab_size",
    "codes_shape",
]


def audio_to_teaching_tokens(
    audio_path: Path | str,
    config: TeachingCodecConfig | None = None,
) -> torch.Tensor:
    """Quantize log-mel energy into discrete teaching tokens."""
    config = config or TeachingCodecConfig()
    audio, sr = load_audio(
        audio_path,
        sr=config.sample_rate,
        mono=True,
        duration=config.duration_sec,
    )
    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")
    mel = librosa.feature.melspectrogram(
        y=np.asarray(audio, dtype=np.float32),
        sr=sr,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        power=2.0,
    )
    db = librosa.power_to_db(mel, ref=1.0, top_db=None)
    clipped = np.clip(db, -float(config.top_db), 0.0)
    normalized = (clipped + float(config.top_db)) / float(config.top_db)
    tokens = np.rint(normalized * float(config.vocab_size - 1)).astype(np.int64)
    return torch.from_numpy(tokens)


def teaching_tokens_to_audio(
    tokens: torch.Tensor | np.ndarray,
    config: TeachingCodecConfig | None = None,
    n_iter: int = 16,
) -> np.ndarray:
    """Invert teaching log-mel tokens into an approximate waveform."""
    config = config or TeachingCodecConfig()
    token_array = np.asarray(torch.as_tensor(tokens).detach().cpu(), dtype=np.float32)
    if token_array.ndim == 1:
        usable = (token_array.size // config.n_mels) * config.n_mels
        if usable == 0:
            raise ValueError("not enough flat tokens to form one mel frame")
        token_array = token_array[:usable].reshape(-1, config.n_mels).T
    elif token_array.ndim == 3:
        token_array = token_array[0]
    if token_array.ndim != 2:
        raise ValueError("teaching tokens must be flat, 2-D, or batch-first 3-D")
    token_array = np.clip(token_array, 0, config.vocab_size - 1)
    db = token_array / float(config.vocab_size - 1) * float(config.top_db) - float(config.top_db)
    mel_power = librosa.db_to_power(db, ref=1.0)
    audio = librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        power=2.0,
        n_iter=n_iter,
    )
    return np.asarray(audio, dtype=np.float32)


def build_teaching_fma_token_cache(
    token_dir: Path | str = "outputs/generated/teaching_codec_tokens",
    manifest_csv: Path | str = "outputs/tables/08_5_teaching_token_cache_manifest.csv",
    audio_manifest_csv: Path | str | None = "data_manifests/fma_small_subset.csv",
    limit: int = 32,
    config: TeachingCodecConfig | None = None,
    skip_existing: bool = True,
) -> list[dict[str, object]]:
    """Build teaching token caches from the FMA small manifest."""
    check_required_assets(
        ["fma_small"],
        message="FMA small is required to build teaching codec tokens.",
        stop=True,
    )
    audio_paths = resolve_fma_audio_paths(audio_manifest_csv=audio_manifest_csv, limit=limit)
    return build_teaching_token_cache(
        audio_paths=audio_paths,
        token_dir=token_dir,
        manifest_csv=manifest_csv,
        config=config,
        skip_existing=skip_existing,
    )


def build_teaching_token_cache(
    audio_paths: Iterable[Path | str],
    token_dir: Path | str,
    manifest_csv: Path | str,
    config: TeachingCodecConfig | None = None,
    skip_existing: bool = True,
) -> list[dict[str, object]]:
    """Write teaching token tensors and a cache manifest."""
    config = config or TeachingCodecConfig()
    token_dir = Path(token_dir)
    manifest_csv = Path(manifest_csv)
    rows = []
    for index, audio_path in enumerate(audio_paths, start=1):
        audio_path = Path(audio_path)
        paths = teaching_output_paths(audio_path, token_dir)
        if skip_existing and paths["tokens"].exists():
            status = "cached"
            tokens = torch.load(paths["tokens"], map_location="cpu")
        else:
            status = "encoded"
            tokens = audio_to_teaching_tokens(audio_path, config)
            write_teaching_artifacts(audio_path, tokens, paths, config)
        rows.append(teaching_manifest_row(index, audio_path, paths, config, status, tokens))
    write_rows(manifest_csv, rows, fieldnames=TEACHING_TOKEN_CACHE_FIELDS)
    return rows


def write_teaching_artifacts(
    audio_path: Path | str,
    tokens: torch.Tensor,
    paths: dict[str, Path],
    config: TeachingCodecConfig,
) -> None:
    """Write one teaching token tensor and JSON metadata."""
    paths["tokens"].parent.mkdir(parents=True, exist_ok=True)
    torch.save(tokens.long(), paths["tokens"])
    metadata = {
        "source_audio": portable_path(Path(audio_path), CHAPTER_ROOT),
        "token_path": portable_path(paths["tokens"], CHAPTER_ROOT),
        "metadata_path": portable_path(paths["metadata"], CHAPTER_ROOT),
        "tokenizer": "teaching_logmel_quantizer",
        "codes_shape": list(tokens.shape),
        **asdict(config),
    }
    paths["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def teaching_output_paths(audio_path: Path | str, token_dir: Path | str) -> dict[str, Path]:
    """Return stable paths for teaching-token artifacts."""
    audio_path = Path(audio_path)
    token_dir = Path(token_dir)
    return {
        "tokens": token_dir / f"{audio_path.stem}.tokens.pt",
        "metadata": token_dir / f"{audio_path.stem}.json",
    }


def teaching_manifest_row(
    index: int,
    audio_path: Path,
    paths: dict[str, Path],
    config: TeachingCodecConfig,
    status: str,
    tokens: torch.Tensor,
) -> dict[str, object]:
    """Return one teaching token-cache manifest row."""
    return {
        "index": index,
        "status": status,
        "audio_path": portable_path(audio_path, CHAPTER_ROOT),
        "token_path": portable_path(paths["tokens"], CHAPTER_ROOT),
        "metadata_path": portable_path(paths["metadata"], CHAPTER_ROOT),
        "tokenizer": "teaching_logmel_quantizer",
        "sample_rate": config.sample_rate,
        "duration_sec": config.duration_sec,
        "n_mels": config.n_mels,
        "hop_length": config.hop_length,
        "n_fft": config.n_fft,
        "vocab_size": config.vocab_size,
        "codes_shape": "x".join(str(dim) for dim in tokens.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chapter 8 teaching audio-token cache.")
    parser.add_argument("--token-dir", default="outputs/generated/teaching_codec_tokens")
    parser.add_argument("--manifest-csv", default="outputs/tables/08_5_teaching_token_cache_manifest.csv")
    parser.add_argument("--audio-manifest", default="data_manifests/fma_small_subset.csv")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--duration-sec", type=float, default=4.0)
    parser.add_argument("--n-mels", type=int, default=16)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--top-db", type=float, default=80.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = TeachingCodecConfig(
        sample_rate=args.sample_rate,
        duration_sec=args.duration_sec,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        vocab_size=args.vocab_size,
        top_db=args.top_db,
    )
    rows = build_teaching_fma_token_cache(
        token_dir=args.token_dir,
        manifest_csv=args.manifest_csv,
        audio_manifest_csv=args.audio_manifest,
        limit=args.limit,
        config=config,
        skip_existing=not args.overwrite,
    )
    print(f"Wrote {len(rows)} teaching token-cache rows to {args.manifest_csv}")


if __name__ == "__main__":
    main()
