"""Thin EnCodec adapter used by codec-token notebooks."""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path

import numpy as np

from _common.audio_io import load_audio, save_audio, to_mono
from _common.paths import portable_path


CHAPTER_ROOT = Path(__file__).resolve().parents[1]


class EncodecUnavailable(RuntimeError):
    """Raised when EnCodec dependencies are not installed."""


@dataclass(frozen=True)
class EncodecStatus:
    available: bool
    reason: str
    next_action: str


@dataclass(frozen=True)
class EncodecResult:
    """Outputs from one EnCodec encode/decode pass."""

    codes: object
    reconstruction: np.ndarray
    sample_rate: int
    bandwidth: float
    source_path: Path | None = None


def check_encodec() -> EncodecStatus:
    """Check whether the EnCodec package can be imported."""
    if importlib.util.find_spec("encodec") is None:
        return EncodecStatus(
            available=False,
            reason="missing Python package: encodec",
            next_action="python -m pip install encodec",
        )
    return EncodecStatus(True, "encodec package available", "run 08_4_encodec_codec_tokens.ipynb")


def require_encodec() -> None:
    """Raise with a concise install hint if EnCodec is unavailable."""
    status = check_encodec()
    if not status.available:
        raise EncodecUnavailable(f"{status.reason}. {status.next_action}")


def encodec_output_paths(audio_path: Path | str, token_dir: Path | str) -> dict[str, Path]:
    """Return stable paths for cached codec-token artifacts."""
    audio_path = Path(audio_path)
    token_dir = Path(token_dir)
    stem = audio_path.stem
    return {
        "tokens": token_dir / f"{stem}.tokens.pt",
        "metadata": token_dir / f"{stem}.json",
        "reconstruction": token_dir / f"{stem}.reconstruction.wav",
    }


class EncodecAdapter:
    """Small wrapper around the official EnCodec package."""

    def __init__(
        self,
        model_name: str = "24khz",
        bandwidth: float = 6.0,
        device: str = "cpu",
    ) -> None:
        require_encodec()
        import torch
        from encodec import EncodecModel

        self.torch = torch
        self.device = torch.device(device)
        self.bandwidth = float(bandwidth)
        if model_name == "24khz":
            self.model = EncodecModel.encodec_model_24khz()
        elif model_name == "48khz":
            self.model = EncodecModel.encodec_model_48khz()
        else:
            raise ValueError("model_name must be '24khz' or '48khz'")
        self.model.set_target_bandwidth(self.bandwidth)
        self.model.to(self.device)
        self.model.eval()
        self.sample_rate = int(self.model.sample_rate)
        self.channels = int(self.model.channels)

    def encode_decode_file(self, audio_path: Path | str) -> EncodecResult:
        """Encode and decode one audio file with the selected EnCodec model."""
        audio, _ = load_audio(audio_path, sr=self.sample_rate, mono=self.channels == 1)
        tensor = self._audio_to_tensor(audio)
        with self.torch.no_grad():
            encoded_frames = self.model.encode(tensor)
            reconstruction = self.model.decode(encoded_frames)
        codes = self.torch.cat([frame[0] for frame in encoded_frames], dim=-1)
        audio_np = reconstruction.squeeze(0).detach().cpu().numpy()
        if self.channels == 1:
            audio_np = to_mono(audio_np)
        return EncodecResult(
            codes=codes.detach().cpu(),
            reconstruction=np.asarray(audio_np, dtype=np.float32),
            sample_rate=self.sample_rate,
            bandwidth=self.bandwidth,
            source_path=Path(audio_path),
        )

    def write_reconstruction(self, audio_path: Path | str, token_dir: Path | str) -> EncodecResult:
        """Run encode/decode and save the reconstructed waveform."""
        result = self.encode_decode_file(audio_path)
        paths = encodec_output_paths(audio_path, token_dir)
        save_audio(paths["reconstruction"], result.reconstruction, result.sample_rate)
        return result

    def _audio_to_tensor(self, audio: np.ndarray):
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[None, :]
        if audio.shape[0] != self.channels:
            if self.channels == 1:
                audio = to_mono(audio)[None, :]
            else:
                audio = np.tile(to_mono(audio)[None, :], (self.channels, 1))
        return self.torch.from_numpy(audio).unsqueeze(0).to(self.device)


def write_encodec_artifacts(
    audio_path: Path | str,
    result: EncodecResult,
    token_dir: Path | str,
    save_reconstruction: bool = False,
) -> dict[str, Path]:
    """Write token tensor, metadata JSON, and reconstruction WAV when requested."""
    import torch

    paths = encodec_output_paths(audio_path, token_dir)
    paths["tokens"].parent.mkdir(parents=True, exist_ok=True)
    torch.save(result.codes, paths["tokens"])
    metadata = encodec_metadata(audio_path, result, paths, save_reconstruction)
    paths["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if save_reconstruction:
        save_audio(paths["reconstruction"], result.reconstruction, result.sample_rate)
    return paths


def encodec_metadata(
    audio_path: Path | str,
    result: EncodecResult,
    paths: dict[str, Path],
    save_reconstruction: bool = False,
) -> dict[str, object]:
    """Return stable metadata for a cached EnCodec token file."""
    codes = result.codes
    shape = list(codes.shape) if hasattr(codes, "shape") else []
    return {
        "source_audio": portable_path(Path(audio_path), CHAPTER_ROOT),
        "token_path": portable_path(paths["tokens"], CHAPTER_ROOT),
        "metadata_path": portable_path(paths["metadata"], CHAPTER_ROOT),
        "reconstruction_path": portable_path(paths["reconstruction"], CHAPTER_ROOT) if save_reconstruction else "",
        "sample_rate": result.sample_rate,
        "bandwidth": result.bandwidth,
        "codes_shape": shape,
    }
