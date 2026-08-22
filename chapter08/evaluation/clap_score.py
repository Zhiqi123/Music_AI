"""CLAP-based text-audio similarity scoring with an explicit dependency check."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
from pathlib import Path
from typing import Sequence

import numpy as np

from _common.device_utils import choose_device
from _common.paths import portable_path
from _common.tables import write_rows


@dataclass(frozen=True)
class ClapStatus:
    backend: str
    available: bool
    reason: str
    next_action: str

    def as_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ClapScore:
    prompt_id: str
    audio_path: str
    prompt: str
    clap_score: float
    backend: str

    def as_row(self) -> dict[str, object]:
        return asdict(self)


def check_clap_dependency(preferred_backend: str = "laion_clap") -> ClapStatus:
    """Check whether a supported CLAP backend is importable."""
    if preferred_backend != "laion_clap":
        return ClapStatus(
            backend=preferred_backend,
            available=False,
            reason=f"unsupported CLAP backend: {preferred_backend}",
            next_action="Use preferred_backend='laion_clap'.",
        )
    if importlib.util.find_spec("laion_clap") is None:
        return ClapStatus(
            backend="laion_clap",
            available=False,
            reason="missing Python package: laion_clap",
            next_action="python -m pip install laion-clap",
        )
    return ClapStatus(
        backend="laion_clap",
        available=True,
        reason="laion_clap package available",
        next_action="set CHAPTER08_RUN_CLAP=1 to compute CLAP scores",
    )


def clap_score_unavailable_message() -> str:
    """Return the message shown when CLAP dependencies are absent."""
    return check_clap_dependency().next_action


def compute_clap_scores(
    audio_paths: Sequence[Path | str],
    prompts: Sequence[dict[str, str]],
    device: str = "auto",
) -> list[ClapScore]:
    """Compute CLAP cosine similarity rows using ``laion_clap``."""
    status = check_clap_dependency()
    if not status.available:
        raise RuntimeError(f"{status.reason}. {status.next_action}")
    import torch
    import laion_clap

    device = choose_device(device)
    model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
    model.load_ckpt()
    audio_path_strings = [str(Path(path)) for path in audio_paths]
    prompt_texts = [row["prompt"] for row in prompts]
    with torch.no_grad():
        audio_embeddings = model.get_audio_embedding_from_filelist(
            x=audio_path_strings,
            use_tensor=True,
        )
        text_embeddings = model.get_text_embedding(prompt_texts, use_tensor=True)
        similarities = cosine_similarity_matrix(
            audio_embeddings.detach().cpu().numpy(),
            text_embeddings.detach().cpu().numpy(),
        )
    scores: list[ClapScore] = []
    for prompt_index, prompt in enumerate(prompts):
        for audio_index, audio_path in enumerate(audio_path_strings):
            scores.append(
                ClapScore(
                    prompt_id=prompt.get("prompt_id", f"prompt_{prompt_index:03d}"),
                    audio_path=portable_path(audio_path, Path(".")),
                    prompt=prompt["prompt"],
                    clap_score=float(similarities[audio_index, prompt_index]),
                    backend=status.backend,
                )
            )
    return scores


def cosine_similarity_matrix(audio_embeddings: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
    """Return cosine similarities with shape ``(audio, text)``."""
    audio = _normalize_rows(np.asarray(audio_embeddings, dtype=np.float64))
    text = _normalize_rows(np.asarray(text_embeddings, dtype=np.float64))
    return audio @ text.T


def write_clap_status(path: Path | str, status: ClapStatus) -> None:
    """Write one CLAP dependency status row."""
    write_rows(path, [status.as_row()], fieldnames=["backend", "available", "reason", "next_action"])


def write_clap_scores(path: Path | str, scores: Sequence[ClapScore]) -> None:
    """Write CLAP scores with stable fields."""
    write_rows(
        path,
        [score.as_row() for score in scores],
        fieldnames=["prompt_id", "audio_path", "prompt", "clap_score", "backend"],
    )


def _normalize_rows(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("embedding matrix must be 2-D")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)
