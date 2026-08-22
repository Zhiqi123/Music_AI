"""CTIS 数据加载（6.1 传统机器学习管线）。

约束：仅 `default/train` split 含音频字节，eval_* 三 split 只含预算
特征（mel/cqt/chroma），不可用于 raw audio 训练。

数据来源：
  默认路径 `CODE/datasets/CCMUSIC_CTIS/default/` 不存在时，自动从
  HuggingFace `ccmusic-database/CTIS` 下载 default config 并 save_to_disk
  到该路径；存在时直接 load_from_disk，不联网。

  读者首次运行 Notebook 会触发一次下载（约 600 MB，含全部 split 的元数据
  与 default/train 的音频字节），之后离线复用。
"""
from __future__ import annotations

import hashlib
import io
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, load_dataset, load_from_disk

HF_REPO_ID = "ccmusic-database/CTIS"
DEFAULT_LOCAL_CACHE = Path(__file__).resolve().parents[2] / "datasets" / "CCMUSIC_CTIS" / "default"


def _ensure_local_cache(local_dir: Path) -> None:
    """若本地缓存不存在，从 HuggingFace 下载并落盘到 local_dir。

    下载只发生一次；之后所有调用都直接 load_from_disk，无网络依赖。
    """
    if local_dir.exists():
        return
    print(f"本地缓存不存在，从 HuggingFace 下载 {HF_REPO_ID} (default config) ...")
    print(f"  目标路径: {local_dir}")
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    ds_dict = load_dataset(HF_REPO_ID, "default")
    ds_dict.save_to_disk(str(local_dir))
    print(f"  下载完成，已缓存至 {local_dir}")


def _open_default_train(local_dir: Optional[Path] = None) -> Dataset:
    """读取 default/train，首次调用会自动下载并落盘到 local_dir。"""
    local_dir = Path(local_dir) if local_dir else DEFAULT_LOCAL_CACHE
    _ensure_local_cache(local_dir)
    ds = load_from_disk(str(local_dir))["train"]
    return ds.cast_column("audio", Audio(decode=False))


def get_top_n_classes(n: int = 10, local_dir: Optional[Path] = None) -> list[dict]:
    """按 default/train 样本数取前 n 个类，仅扫描元数据列。

    Returns:
        [{'label_idx': int, 'cname': str, 'pinyin': str, 'default_train': int}, ...]
        按 default_train 降序排列；同样本数按 label_idx 升序保稳定。
    """
    ds = _open_default_train(local_dir)
    meta = ds.remove_columns([c for c in ds.column_names if c not in ("label", "cname", "pinyin")])
    counter: Counter[int] = Counter()
    name_map: dict[int, tuple[str, str]] = {}
    for row in meta:
        counter[row["label"]] += 1
        name_map.setdefault(row["label"], (row["cname"], row["pinyin"]))

    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    return [
        {
            "label_idx": label_idx,
            "cname": name_map[label_idx][0],
            "pinyin": name_map[label_idx][1],
            "default_train": count,
        }
        for label_idx, count in ranked
    ]


def load_ctis_topN(
    n: int = 10,
    local_dir: Optional[Path] = None,
    return_audio: bool = True,
) -> tuple[list[dict], list[dict]]:
    """加载 CTIS default/train 中样本数排名前 n 的类。

    Args:
        n: 取前 n 个类，默认 10。
        local_dir: 本地缓存目录；None 则用仓库内置路径。
        return_audio: True 时解码音频为 float32 numpy 数组，False 时只返回元数据。

    Returns:
        (top_classes, samples)：
          top_classes 见 get_top_n_classes 返回；
          samples 中每项为 {'audio': np.ndarray|None, 'sr': int, 'label_idx': int,
                            'cname': str, 'audio_sha1': str}。
    """
    top_classes = get_top_n_classes(n, local_dir)
    keep_labels = {c["label_idx"] for c in top_classes}
    cname_map = {c["label_idx"]: c["cname"] for c in top_classes}

    ds = _open_default_train(local_dir)
    samples: list[dict] = []
    for row in ds:
        if row["label"] not in keep_labels:
            continue
        audio_bytes = row["audio"]["bytes"]
        item: dict = {
            "label_idx": row["label"],
            "cname": cname_map[row["label"]],
            "audio_sha1": hashlib.sha1(audio_bytes).hexdigest(),
            "sr": None,
            "audio": None,
        }
        if return_audio:
            audio_arr, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            if audio_arr.ndim > 1:
                audio_arr = audio_arr.mean(axis=1)
            item["audio"] = audio_arr
            item["sr"] = sr
        samples.append(item)
    return top_classes, samples


def compute_audio_sha1(audio_bytes: bytes) -> str:
    """对原始音频字节做 SHA1，用于跨 split 重叠检测。"""
    return hashlib.sha1(audio_bytes).hexdigest()
