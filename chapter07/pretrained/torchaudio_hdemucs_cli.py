"""Run a torchaudio HDemucs bundle on one audio file."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from chapter07._common.audio_io import load_audio, save_audio  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--bundle", default="HDEMUCS_HIGH_MUSDB_PLUS")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--segment", type=float, default=10.0)
    parser.add_argument("--overlap", type=float, default=1.0)
    args = parser.parse_args(argv)

    import torch
    import torchaudio

    bundle = getattr(torchaudio.pipelines, args.bundle)
    model = bundle.get_model().to(args.device).eval()
    sample_rate = int(bundle.sample_rate)
    # PyTorch/Torchaudio bundles expose source names on the model, not the bundle.
    source_names = tuple(model.sources)

    audio, _ = load_audio(args.input_path, sr=sample_rate, mono=False)
    audio = _as_stereo(audio)
    waveform = torch.from_numpy(audio).unsqueeze(0).to(args.device)

    with torch.inference_mode():
        estimates = _separate_with_overlap(
            model,
            waveform,
            sample_rate=sample_rate,
            segment_sec=args.segment,
            overlap_sec=args.overlap,
        )

    track_dir = args.outdir / args.input_path.stem
    for source_name, estimate in zip(source_names, estimates.cpu().numpy(), strict=False):
        save_audio(track_dir / f"{source_name}.wav", estimate.astype(np.float32), sample_rate)
    return 0


def _as_stereo(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return np.vstack([audio, audio])
    if audio.shape[0] == 1:
        return np.vstack([audio[0], audio[0]])
    if audio.shape[0] >= 2:
        return audio[:2]
    raise ValueError("audio must be mono or channel-first")


def _separate_with_overlap(
    model: object,
    waveform: object,
    sample_rate: int,
    segment_sec: float,
    overlap_sec: float,
) -> object:
    import torch

    total = int(waveform.shape[-1])
    segment = max(1, int(round(segment_sec * sample_rate)))
    overlap = max(0, int(round(overlap_sec * sample_rate)))
    if segment >= total:
        return model(waveform)[0][..., :total]

    step = max(1, segment - overlap)
    output = None
    weight = torch.zeros(total, device=waveform.device, dtype=waveform.dtype)

    for start in range(0, total, step):
        end = min(start + segment, total)
        chunk = waveform[..., start:end]
        valid = int(chunk.shape[-1])
        if valid < segment:
            chunk = torch.nn.functional.pad(chunk, (0, segment - valid))
        estimate = model(chunk)[0][..., :valid]
        if output is None:
            output = torch.zeros(
                estimate.shape[0],
                estimate.shape[1],
                total,
                device=estimate.device,
                dtype=estimate.dtype,
            )
        output[..., start:end] += estimate
        weight[start:end] += 1.0
        if end >= total:
            break

    if output is None:
        raise RuntimeError("no audio segment was processed")
    return output / weight.clamp_min(1.0)


if __name__ == "__main__":
    raise SystemExit(main())
