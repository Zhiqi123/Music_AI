"""ChMusic 遗留探索加载器。

数据来源：CCMUSIC ChMusic（11 类民族乐器，本地副本每类 5 段曲目；
未附可核验的演奏者元数据）。

标签编号→中文名映射来自 InstrumentsDescription/MusicsList.png 的 OCR 整理，
编号严格对应 Musics/<id>.<take>.wav 文件名前缀。

该模块只供早期 6.1a 探索单元复现旧图，不属于当前冻结实验管线。
主实验统一读取 ``_data_pipeline/output/segment_manifest_seed*.csv``；其中
ChMusic 使用 24 kHz、5 秒切片和 6 类标签。不得用本模块的 3 秒、4 类
交集输出替代主实验数据。

核心约束：
  - 每段曲目 ~2 分钟立体声 44.1 kHz；本模块用 librosa.load 强制单声道 + 目标采样率。
  - 切片：固定 3 秒、0% 重叠、不足段零填充（不丢弃，最后一段较短时仍参与评估）。
  - 曲目级评估通过 source_file 字段聚合切片，避免长曲目对短曲目的样本数偏置。

与 CTIS top-10 的严格交集：二胡、琵琶、柳琴、唢呐（CTIS 端 '唢呐2' 经 display_cname
规范化后对齐）。中阮↔大阮形制差异明显，不视为同类。
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

DEFAULT_CHMUSIC_ROOT = (
    Path(__file__).resolve().parents[2] / "datasets" / "ChMusic"
)

# 编号 → 中文乐器名。来自 OCR 整理的官方演奏曲目列表。
CHMUSIC_LABEL_MAP: dict[int, str] = {
    1: "二胡",
    2: "琵琶",
    3: "三弦",
    4: "笛子",
    5: "唢呐",
    6: "坠琴",
    7: "中阮",
    8: "柳琴",
    9: "古筝",
    10: "扬琴",
    11: "笙",
}

# 与 CTIS top-10 的严格交集（顺序与 CTIS top10_class_list.json 默认顺序无关，
# 评估前再按调用方传入的类名顺序对齐）。
CTIS_INTERSECTION: list[str] = ["二胡", "琵琶", "柳琴", "唢呐"]


def _list_wav_files(root: Path) -> list[Path]:
    musics_dir = root / "Musics"
    if not musics_dir.exists():
        raise FileNotFoundError(f"ChMusic Musics/ 不存在: {musics_dir}")
    return sorted(musics_dir.glob("*.wav"))


def _parse_label_from_name(path: Path) -> int:
    # 文件名形如 "1.1.wav" / "11.5.wav"，第一个 '.' 前是乐器编号
    return int(path.stem.split(".")[0])


def _slice_3sec(audio: np.ndarray, sr: int, clip_sec: float) -> list[np.ndarray]:
    """3 秒、0% 重叠切片；最后不足 3 秒的尾段零填充后保留。"""
    clip_len = int(round(clip_sec * sr))
    n = len(audio)
    if n == 0:
        return []
    n_full = n // clip_len
    clips = [audio[i * clip_len : (i + 1) * clip_len] for i in range(n_full)]
    rem = n - n_full * clip_len
    if rem > 0:
        tail = np.zeros(clip_len, dtype=audio.dtype)
        tail[:rem] = audio[n_full * clip_len :]
        clips.append(tail)
    return clips


def load_chmusic_clips(
    root: Optional[Path] = None,
    target_sr: int = 44100,
    clip_sec: float = 3.0,
    intersection_only: bool = True,
) -> list[dict]:
    """扫描 ChMusic/Musics/*.wav，按固定窗切片，返回扁平的切片列表。

    Args:
        root: ChMusic 根目录；None 用仓库默认路径。
        target_sr: librosa.load 的目标采样率（默认 44100 与 CTIS 对齐，便于复用
            56 维特征链路）。CNN 路径可传 22050 节省一半 mel 计算量。
        clip_sec: 切片长度（秒），默认 3.0。
        intersection_only: True 时只返回与 CTIS top-10 严格交集 4 类。

    Returns:
        每条 dict:
            audio: shape (clip_len,) float32，已重采样为单声道
            sr: int，等于 target_sr
            label_idx: int，ChMusic 原始编号 1–11
            cname: str，乐器中文名
            source_file: str，原始 wav 文件名（不含路径），用于曲目级聚合
            clip_idx: int，本曲目内的切片序号（0 起）
            total_clips: int，本曲目共切出多少片
    """
    root = Path(root) if root is not None else DEFAULT_CHMUSIC_ROOT
    keep_names: Optional[set[str]] = set(CTIS_INTERSECTION) if intersection_only else None

    out: list[dict] = []
    for wav_path in _list_wav_files(root):
        label_idx = _parse_label_from_name(wav_path)
        cname = CHMUSIC_LABEL_MAP.get(label_idx)
        if cname is None:
            continue
        if keep_names is not None and cname not in keep_names:
            continue
        audio, _ = librosa.load(str(wav_path), sr=target_sr, mono=True)
        audio = audio.astype(np.float32, copy=False)
        clips = _slice_3sec(audio, target_sr, clip_sec)
        total = len(clips)
        for k, clip in enumerate(clips):
            out.append(
                {
                    "audio": clip,
                    "sr": target_sr,
                    "label_idx": label_idx,
                    "cname": cname,
                    "source_file": wav_path.name,
                    "clip_idx": k,
                    "total_clips": total,
                }
            )
    return out


def aggregate_clip_predictions(
    proba: np.ndarray,
    source_files: list[str],
) -> tuple[list[str], np.ndarray]:
    """按 source_file 把切片级概率聚合成曲目级概率（平均）。

    Args:
        proba: shape (n_clips, n_classes) 的概率矩阵。
        source_files: 长度 n_clips 的源文件名列表。

    Returns:
        (unique_files, file_proba)：
          unique_files 按首次出现顺序去重；
          file_proba shape (n_files, n_classes)，每行是该文件下所有切片概率的均值。
    """
    if len(source_files) != proba.shape[0]:
        raise ValueError("source_files 长度与 proba 行数不一致")
    groups: dict[str, list[int]] = defaultdict(list)
    order: list[str] = []
    for i, fn in enumerate(source_files):
        if fn not in groups:
            order.append(fn)
        groups[fn].append(i)
    file_proba = np.stack([proba[idxs].mean(axis=0) for idxs in (groups[f] for f in order)])
    return order, file_proba
