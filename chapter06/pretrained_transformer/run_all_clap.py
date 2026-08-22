"""跑 4 prompt × {test, external_test} CLAP zero-shot 评估。

CLAP 不在本章数据上训练；本脚本只评估冻结划分 0，不能据此估计跨划分波动。
CLAP music checkpoint应预先放到outputs/checkpoints/，当前文件大小为2,352,471,003字节。
离线模式下若文件缺失或大小不符，程序会直接报错；在线模式下才会尝试下载。

环境变量默认值（命令行可 override）：
- HF_ENDPOINT=https://hf-mirror.com
- HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1：cache 命中直接用
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from chapter06._common.ctmp_loader import (
    CTMP_CLASSES, build_label_map, load_ctmp_segments,
)
from .clap_model import CLAPZeroShot
from .prompts import PROMPT_TEMPLATES, render_prompts

OUT = Path(__file__).parent / "outputs" / "clap_prompt_comparison.csv"
CLAP_SUMMARY_METHOD = "CLAP · zero-shot (descriptive, split 0)"


def evaluate_encoded_split(
    clap: CLAPZeroShot,
    audio_embeddings,
    y_true: np.ndarray,
    prompts: list[str],
) -> tuple[np.ndarray, float, float]:
    text_embeddings = clap.encode_text(prompts)
    y_pred = (audio_embeddings @ text_embeddings.T).argmax(dim=1).cpu().numpy()
    return (
        y_pred,
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro"),
    )


def upsert_clap_summary(df_prev: pd.DataFrame, selected: pd.Series) -> pd.DataFrame:
    """以固定的descriptive结果更新全章汇总，避免重复追加同一行。"""
    if "方法" in df_prev.columns:
        df_prev = df_prev.loc[df_prev["方法"] != CLAP_SUMMARY_METHOD].copy()
    new_row = pd.DataFrame([{
        "方法": CLAP_SUMMARY_METHOD,
        "内部 Acc": f'{selected["internal_acc"]:.3f}',
        "内部 F1": f'{selected["internal_f1"]:.3f}',
        "外部 Acc": f'{selected["external_acc"]:.3f}',
        "外部 F1": f'{selected["external_f1"]:.3f}',
        "Gap (Acc)": f'{selected["internal_acc"] - selected["external_acc"]:+.3f}',
    }])
    return pd.concat([df_prev, new_row], ignore_index=True)


def main() -> pd.DataFrame:
    label_map = build_label_map()
    clap = CLAPZeroShot()

    test_segs = load_ctmp_segments(seed=0, split="test")
    ext_segs = load_ctmp_segments(seed=0, split="external_test")
    print(f"test={len(test_segs)} ext={len(ext_segs)}")

    paths_in = [s["audio_path"] for s in test_segs]
    paths_ex = [s["audio_path"] for s in ext_segs]
    y_in = np.array([label_map[s["family_label"]] for s in test_segs])
    y_ex = np.array([label_map[s["family_label"]] for s in ext_segs])
    audio_in = clap.encode_audio(paths_in)
    audio_ex = clap.encode_audio(paths_ex)

    rows = []
    for tname in PROMPT_TEMPLATES:
        prompts = render_prompts(tname)
        _, in_acc, in_f1 = evaluate_encoded_split(clap, audio_in, y_in, prompts)
        _, ex_acc, ex_f1 = evaluate_encoded_split(clap, audio_ex, y_ex, prompts)
        print(f"{tname:14s}  internal acc={in_acc:.3f} f1={in_f1:.3f} | "
              f"external acc={ex_acc:.3f} f1={ex_f1:.3f}")
        rows.append({
            "prompt": tname,
            "internal_acc": in_acc,
            "internal_f1": in_f1,
            "external_acc": ex_acc,
            "external_f1": ex_f1,
        })
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.4f")
    print(f"saved {OUT}")
    return df


if __name__ == "__main__":
    main()
