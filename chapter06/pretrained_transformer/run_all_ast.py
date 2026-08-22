"""跑 AST compact sweep，输出 ast_method_comparison.csv。

CSV schema 与 6.3 pretrained_cnn 对齐（中文列名 + mean±std 字符串），
便于后续合并到 full_comparison.csv。

默认使用 train 训练、val 选 best epoch，test/external_test 只做最终评估。
LoRA 与 FullFT 用 compact sweep 选验证集 macro-F1 最好的配置。
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pathlib import Path

import pandas as pd

from .sweep import (
    compact_sweep_configs,
    run_ast_config_grid,
    selected_config_table,
    select_best_specs_by_val,
    summarize_config_results,
    summarize_selected_methods,
)

OUT = Path(__file__).parent / "outputs" / "ast_method_comparison.csv"
SWEEP_OUT = Path(__file__).parent / "outputs" / "ast_hparam_sweep.csv"
SELECTED_OUT = Path(__file__).parent / "outputs" / "ast_selected_configs.csv"
RUN_CACHE = Path(__file__).parent / "outputs" / "run_cache" / "ast_val"

DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 32
DEFAULT_TUNE_BATCH_SIZE = 8


def main(epochs: int = DEFAULT_EPOCHS, seeds: tuple[int, ...] = (0, 1, 2)) -> pd.DataFrame:
    specs = compact_sweep_configs(
        epochs=epochs,
        batch_size=DEFAULT_BATCH_SIZE,
        tune_batch_size=DEFAULT_TUNE_BATCH_SIZE,
    )
    results = run_ast_config_grid(specs, seeds, RUN_CACHE)
    df_sweep = summarize_config_results(results)
    selected_specs = select_best_specs_by_val(specs, df_sweep)
    df_selected = selected_config_table(selected_specs, df_sweep)
    df = summarize_selected_methods(results, selected_specs)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df_sweep.to_csv(SWEEP_OUT, index=False)
    df_selected.to_csv(SELECTED_OUT, index=False)
    df.to_csv(OUT, index=False)
    print(f"saved {SWEEP_OUT} ({len(df_sweep)} rows)")
    print(f"saved {SELECTED_OUT} ({len(df_selected)} rows)")
    print(f"saved {OUT} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    main()
