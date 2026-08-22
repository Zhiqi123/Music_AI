"""PyTorch Dataset：CTMP 切片 → 16 kHz → AST processor → 缓存的输入特征。

每条 input_values 约占0.5 MB。模块按“音频路径 + processor来源”惰性缓存，
不同超参数配置和冻结划分可复用同一音频的确定性预处理结果。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import ASTFeatureExtractor

from .ast_model import _resolve_ast_source

AST_SR = 16000
CLIP_SEC = 5.0
AST_MAX_LENGTH = 1024
AST_N_MELS = 128


@lru_cache(maxsize=8)
def _get_processor(source: str) -> ASTFeatureExtractor:
    return ASTFeatureExtractor.from_pretrained(source)


@lru_cache(maxsize=4096)
def _cached_ast_input(audio_path: str, source: str) -> np.ndarray:
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != AST_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=AST_SR)
    target = int(AST_SR * CLIP_SEC)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]
    feat = _get_processor(source)(
        audio,
        sampling_rate=AST_SR,
        padding="max_length",
        max_length=AST_MAX_LENGTH,
        return_tensors="np",
    )
    return feat["input_values"][0].astype(np.float32)


class ASTSegmentDataset(Dataset):
    """CTMP 切片 → AST 输入 (max_length, n_mels)。"""

    def __init__(
        self,
        records: list[dict],
        label_map: dict[str, int],
        precompute: bool = True,
        hf_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
    ):
        self.records = records
        self.label_map = label_map
        self.source = _resolve_ast_source(hf_name)
        self.precompute = precompute
        self._cache: list[np.ndarray] | None = None
        if precompute:
            self._cache = self._precompute_all()

    def _load_one(self, rec: dict) -> np.ndarray:
        return _cached_ast_input(
            str(Path(rec["audio_path"]).resolve()), self.source,
        )

    def _precompute_all(self) -> list[np.ndarray]:
        return [self._load_one(r) for r in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        if self._cache is not None:
            x = torch.from_numpy(self._cache[idx])
        else:
            x = torch.from_numpy(self._load_one(rec))
        y = self.label_map[rec["family_label"]]
        return x, y
