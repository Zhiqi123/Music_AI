"""CLAP zero-shot 推理封装。

模型：laion-clap 包 + music_audioset_epoch_15_esc_90.14.pt
官方分发渠道：huggingface.co/lukewys/laion_clap (LAION 官方)

国内网络下推荐设 HF_ENDPOINT=https://hf-mirror.com，或预先用 curl
从 https://hf-mirror.com/lukewys/laion_clap/resolve/main/<ckpt> 下到本地 outputs/checkpoints/。
"""
from __future__ import annotations

import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

CLAP_SR = 48000
CLIP_SEC = 5.0
CLAP_CKPT_NAME = "music_audioset_epoch_15_esc_90.14.pt"
CLAP_CKPT_SIZE = 2_352_471_003  # bytes；用于检查下载完整性
_HF_HOST = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
CLAP_CKPT_URL = f"{_HF_HOST}/lukewys/laion_clap/resolve/main/{CLAP_CKPT_NAME}"


def _offline_mode_enabled() -> bool:
    truthy = {"1", "true", "yes", "on"}
    return any(
        os.environ.get(name, "").strip().lower() in truthy
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def download_clap_weights(target_dir: Path | None = None) -> Path:
    target_dir = target_dir or (Path(__file__).parent / "outputs" / "checkpoints")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / CLAP_CKPT_NAME
    if target.exists():
        actual = target.stat().st_size
        if actual == CLAP_CKPT_SIZE:
            return target
        invalid = target.with_suffix(target.suffix + ".invalid")
        index = 1
        while invalid.exists():
            invalid = target.with_suffix(target.suffix + f".invalid{index}")
            index += 1
        target.replace(invalid)
        print(
            f"本地CLAP checkpoint大小不符（{actual}，预期{CLAP_CKPT_SIZE}）；"
            f"原文件已移至{invalid.name}"
        )
    if _offline_mode_enabled():
        raise FileNotFoundError(
            f"离线模式下未找到完整CLAP checkpoint：{target}\n"
            f"请预先下载{CLAP_CKPT_SIZE}字节的文件，再重新执行"
        )
    print(f"downloading CLAP ckpt → {target} (~2.2 GB)")
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        urllib.request.urlretrieve(CLAP_CKPT_URL, tmp)
    except (ssl.SSLError, urllib.error.URLError) as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"自动下载失败：{exc}\n"
            f"手动方案：curl -L -C - -o {target} {CLAP_CKPT_URL}"
        ) from exc
    actual = tmp.stat().st_size
    if actual != CLAP_CKPT_SIZE:
        tmp.unlink()
        raise RuntimeError(
            f"下载不完整（{actual} vs 期望 {CLAP_CKPT_SIZE} 字节）；"
            f"用 curl -L -C - 重试以启用断点续传"
        )
    tmp.rename(target)
    return target


def _quantize(x: np.ndarray) -> np.ndarray:
    """laion-clap 推荐的 int16 quantize-then-float32 步骤。"""
    x = np.clip(x, -1.0, 1.0)
    x = (x * 32767.0).astype(np.int16)
    return (x / 32767.0).astype(np.float32)


def _load_audio_48k(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != CLAP_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=CLAP_SR)
    target = int(CLAP_SR * CLIP_SEC)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]
    return _quantize(audio)


class CLAPZeroShot:
    """加载 LAION-CLAP music checkpoint，提供 audio/text embedding 与分类接口。"""

    def __init__(self, ckpt_path: Path | None = None):
        import laion_clap
        self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        ckpt_path = ckpt_path or download_clap_weights()
        self.model.load_ckpt(str(ckpt_path))
        self.model.eval()

    @torch.no_grad()
    def encode_audio(
        self,
        audio_paths: list[str],
        *,
        batch_size: int = 16,
    ) -> torch.Tensor:
        """分批编码音频，避免把整个评估集同时送入HTSAT。"""
        if not audio_paths:
            raise ValueError("audio_paths不能为空")
        if batch_size < 1:
            raise ValueError("batch_size必须为正整数")
        embeddings = []
        for start in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[start : start + batch_size]
            wavs = np.stack([_load_audio_48k(p) for p in batch_paths])
            wavs_t = torch.from_numpy(wavs)
            embeddings.append(
                self.model.get_audio_embedding_from_data(
                    x=wavs_t, use_tensor=True,
                )
            )
        return torch.cat(embeddings, dim=0)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        return self.model.get_text_embedding(texts, use_tensor=True)

    @torch.no_grad()
    def classify(
        self,
        audio_paths: list[str],
        prompts: list[str],
    ) -> np.ndarray:
        """返回 (n_audio,) 的 argmax 索引。"""
        a = self.encode_audio(audio_paths)
        t = self.encode_text(prompts)
        sim = a @ t.T
        return sim.argmax(dim=1).cpu().numpy()
