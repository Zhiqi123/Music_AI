"""读取 `_data_pipeline/output/` 的清洗冻结数据集（CTMP 切片）。

本模块是 6.1 / 6.2 的唯一数据入口。它只负责：
  - 找到 `_data_pipeline/output/` 目录（约定路径相对本仓库）
  - 解析 `segment_manifest_seed{seed}.csv`
  - 把 split（train/val/test/external_test）切出来
  - 返回 records: list[dict]，每条含 audio_path / family_label / source_dataset 等

切片是 5 秒、24 kHz、非重叠 wav 文件。下游训练时可流式 `soundfile.read` 即可，
无需波形缓存（清洗阶段已经做过重采样）。

如果 `_data_pipeline/output/` 不存在，会抛 FileNotFoundError 并提示如何生成。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_SR = 24000
DEFAULT_CLIP_SEC = 5.0
CTMP_CLASSES = ("二胡", "琵琶", "中阮", "笛子", "唢呐", "笙")  # 冻结类别集合
CTMP_SPLITS = ("train", "val", "test", "external_test")


def _default_pipeline_root() -> Path:
    """约定 _data_pipeline 与 _common 同级（都在 chapter06/ 下）。"""
    return Path(__file__).resolve().parent.parent / "_data_pipeline"


def get_ctmp_output_dir(pipeline_root: Optional[Path] = None) -> Path:
    """返回 `_data_pipeline/output/`。允许通过环境变量 CTMP_OUTPUT_DIR 覆盖。"""
    env = os.environ.get("CTMP_OUTPUT_DIR")
    if env:
        out = Path(env).expanduser().resolve()
    else:
        root = pipeline_root or _default_pipeline_root()
        out = root / "output"
    if not out.exists():
        raise FileNotFoundError(
            f"找不到 CTMP 清洗输出目录：{out}\n"
            f"请按 CODE/chapter06/_data_pipeline/README.md 跑一遍 pipeline，"
            f"或设置环境变量 CTMP_OUTPUT_DIR 指向已有的 output/ 目录。"
        )
    return out


def load_segment_manifest(seed: int = 0, output_dir: Optional[Path] = None) -> pd.DataFrame:
    """读 `segment_manifest_seed{seed}.csv`。"""
    out = output_dir or get_ctmp_output_dir()
    path = out / f"segment_manifest_seed{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(f"缺少 manifest：{path}")
    return pd.read_csv(path)


def load_ctmp_segments(
    seed: int = 0,
    split: str = "train",
    output_dir: Optional[Path] = None,
) -> list[dict]:
    """返回某个 seed × split 的切片记录列表。

    每条 record：
        {
          'audio_path': str  # 绝对路径，可直接 sf.read
          'family_label': str  # 中文，6 类之一
          'source_dataset': str  # 'CCMusic' 或 'ChMusic'
          'record_id': str
          'sample_id': str
          'segment_id': str
          'segment_index': int
          'is_padded': bool
          'active_frame_ratio': float
        }
    """
    assert split in CTMP_SPLITS, f"未知 split：{split}（合法：{CTMP_SPLITS}）"
    out = output_dir or get_ctmp_output_dir()
    df = load_segment_manifest(seed=seed, output_dir=out)
    sub = df[df["split"] == split]
    segments_root = out / f"segments_seed{seed}"
    records = []
    for row in sub.itertuples(index=False):
        records.append(
            {
                "audio_path": str((segments_root / row.segment_path).resolve()),
                "family_label": row.family_label,
                "source_dataset": row.source_dataset,
                "record_id": row.record_id,
                "sample_id": row.sample_id,
                "segment_id": row.segment_id,
                "segment_index": int(row.segment_index),
                "is_padded": bool(row.is_padded),
                "active_frame_ratio": float(row.active_frame_ratio),
            }
        )
    return records


def build_label_map(classes: tuple[str, ...] = CTMP_CLASSES) -> dict[str, int]:
    """中文类名 → 整数标签（按 CTMP_CLASSES 顺序固化）。"""
    return {c: i for i, c in enumerate(classes)}


__all__ = [
    "DEFAULT_SR",
    "DEFAULT_CLIP_SEC",
    "CTMP_CLASSES",
    "CTMP_SPLITS",
    "get_ctmp_output_dir",
    "load_segment_manifest",
    "load_ctmp_segments",
    "build_label_map",
]
