"""Small AST hyperparameter sweeps for section 6.4."""
from __future__ import annotations

from dataclasses import replace
from math import ceil
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .train import ASTTrainConfig, run_ast_experiment


def _safe_tune_config(
    base: ASTTrainConfig,
    *,
    effective_batch_size: int,
    micro_batch_size: int,
) -> ASTTrainConfig:
    accum = max(1, ceil(effective_batch_size / micro_batch_size))
    return replace(base, batch_size=micro_batch_size, grad_accum_steps=accum)


def compact_sweep_configs(
    epochs: int = 20,
    batch_size: int = 32,
    tune_batch_size: int = 8,
) -> list[dict]:
    """Return a compact, readable grid for the three highest-priority optimizations."""
    base = ASTTrainConfig(epochs=epochs, batch_size=batch_size, early_stop_patience=5)
    safe_tune = _safe_tune_config(
        base, effective_batch_size=batch_size, micro_batch_size=tune_batch_size,
    )
    return [
        {
            "key": "feature_extraction",
            "mode": "feature_extraction",
            "family": "Feature Extraction",
            "display_name": "Feature Extraction",
            "cfg": base,
        },
        {
            "key": "lora_r4_a8_d0p1_lr1em4",
            "mode": "lora",
            "family": "LoRA",
            "display_name": "LoRA r=4 alpha=8 dropout=0.1",
            "cfg": replace(safe_tune, lora_r=4, lora_alpha=8, lora_dropout=0.1),
        },
        {
            "key": "lora_r8_a16_d0p1_lr1em4",
            "mode": "lora",
            "family": "LoRA",
            "display_name": "LoRA r=8 alpha=16 dropout=0.1",
            "cfg": replace(safe_tune, lora_r=8, lora_alpha=16, lora_dropout=0.1),
        },
        {
            "key": "lora_r16_a32_d0p1_lr1em4",
            "mode": "lora",
            "family": "LoRA",
            "display_name": "LoRA r=16 alpha=32 dropout=0.1",
            "cfg": replace(safe_tune, lora_r=16, lora_alpha=32, lora_dropout=0.1),
        },
        {
            "key": "fullft_lr5em6_wd1em4_ls0p1",
            "mode": "fine_tune",
            "family": "Full Fine-tune",
            "display_name": "FullFT lr=5e-6 wd=1e-4 ls=0.1",
            "cfg": replace(safe_tune, full_ft_lr=5e-6),
        },
        {
            "key": "fullft_lr1em5_wd1em4_ls0p1",
            "mode": "fine_tune",
            "family": "Full Fine-tune",
            "display_name": "FullFT lr=1e-5 wd=1e-4 ls=0.1",
            "cfg": replace(safe_tune, full_ft_lr=1e-5),
        },
        {
            "key": "fullft_lr2em5_wd1em4_ls0p1",
            "mode": "fine_tune",
            "family": "Full Fine-tune",
            "display_name": "FullFT lr=2e-5 wd=1e-4 ls=0.1",
            "cfg": replace(safe_tune, full_ft_lr=2e-5),
        },
    ]


def extended_sweep_configs(
    epochs: int = 20,
    batch_size: int = 32,
    tune_batch_size: int = 8,
) -> list[dict]:
    """A broader optional grid; the notebook uses compact_sweep_configs by default."""
    specs = compact_sweep_configs(
        epochs=epochs, batch_size=batch_size, tune_batch_size=tune_batch_size,
    )
    base = ASTTrainConfig(epochs=epochs, batch_size=batch_size, early_stop_patience=5)
    safe_tune = _safe_tune_config(
        base, effective_batch_size=batch_size, micro_batch_size=tune_batch_size,
    )
    specs.extend([
        {
            "key": "lora_r8_a16_d0p05_lr1em4",
            "mode": "lora",
            "family": "LoRA",
            "display_name": "LoRA r=8 alpha=16 dropout=0.05",
            "cfg": replace(safe_tune, lora_r=8, lora_alpha=16, lora_dropout=0.05),
        },
        {
            "key": "lora_r8_a16_d0p2_lr1em4",
            "mode": "lora",
            "family": "LoRA",
            "display_name": "LoRA r=8 alpha=16 dropout=0.2",
            "cfg": replace(safe_tune, lora_r=8, lora_alpha=16, lora_dropout=0.2),
        },
        {
            "key": "fullft_lr1em5_wd1em5_ls0p1",
            "mode": "fine_tune",
            "family": "Full Fine-tune",
            "display_name": "FullFT lr=1e-5 wd=1e-5 ls=0.1",
            "cfg": replace(safe_tune, full_ft_lr=1e-5, weight_decay=1e-5),
        },
        {
            "key": "fullft_lr1em5_wd5em4_ls0p1",
            "mode": "fine_tune",
            "family": "Full Fine-tune",
            "display_name": "FullFT lr=1e-5 wd=5e-4 ls=0.1",
            "cfg": replace(safe_tune, full_ft_lr=1e-5, weight_decay=5e-4),
        },
        {
            "key": "fullft_lr1em5_wd1em4_ls0p05",
            "mode": "fine_tune",
            "family": "Full Fine-tune",
            "display_name": "FullFT lr=1e-5 wd=1e-4 ls=0.05",
            "cfg": replace(safe_tune, full_ft_lr=1e-5, label_smoothing=0.05),
        },
    ])
    return specs


def run_ast_config_grid(
    specs: Iterable[dict],
    seeds: Iterable[int],
    cache_dir: Path,
) -> list[dict]:
    results = []
    for spec in specs:
        print(f"\n{'=' * 72}\n  {spec['display_name']}\n{'=' * 72}")
        for seed in seeds:
            result = run_ast_experiment(
                seed=seed,
                mode=spec["mode"],
                cfg=spec["cfg"],
                tag=f"{spec['display_name']}·split{seed}",
                cache_dir=cache_dir,
                cache_key=spec["key"],
            )
            result = dict(result)
            result.update({
                "config_key": spec["key"],
                "config_display": spec["display_name"],
                "family": spec["family"],
            })
            results.append(result)
    return results


def _mean_std(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values))


def summarize_config_results(results: list[dict]) -> pd.DataFrame:
    rows = []
    if not results:
        return pd.DataFrame(rows)
    config_order: dict[str, int] = {}
    for result in results:
        config_order.setdefault(result["config_key"], len(config_order))
    keys = sorted({r["config_key"] for r in results})
    for key in keys:
        sub = [r for r in results if r["config_key"] == key]
        first = sub[0]
        row = {
            "config_key": key,
            "mode": first["mode"],
            "family": first["family"],
            "display_name": first["config_display"],
            "n_seeds": len(sub),
            "train_config": first["train_config"],
        }
        for metric in ["val_acc", "val_f1", "test_acc", "test_f1", "ext_acc", "ext_f1"]:
            mean, std = _mean_std([r[metric] for r in sub])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[metric] = f"{mean:.3f} ± {std:.3f}"
        row["gap_acc_mean"] = row["test_acc_mean"] - row["ext_acc_mean"]
        row["Gap (Acc)"] = f"{row['gap_acc_mean']:+.3f}"
        row["_config_order"] = config_order[key]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["mode", "val_f1_mean", "val_acc_mean", "_config_order"],
        ascending=[True, False, False, True],
        kind="mergesort",
    ).drop(columns="_config_order").reset_index(drop=True)


def select_best_specs_by_val(specs: list[dict], summary: pd.DataFrame) -> list[dict]:
    selected = []
    by_key = {spec["key"]: spec for spec in specs}
    spec_order = {spec["key"]: idx for idx, spec in enumerate(specs)}
    feature = [spec for spec in specs if spec["mode"] == "feature_extraction"]
    if feature:
        selected.append(feature[0])
    for mode in ["lora", "fine_tune"]:
        sub = summary[summary["mode"] == mode].copy()
        if sub.empty:
            continue
        # 验证 macro-F1 和验证准确率完全并列时，按预先声明的候选顺序
        # 决定配置，避免结果依赖 pandas 对并列行的内部排序细节。
        # compact_sweep_configs() 对 LoRA 按较低秩在前、对 FullFT 按
        # 较低学习率在前排列，因此这一规则也保持了搜索定义中的优先级。
        sub["_spec_order"] = sub["config_key"].map(spec_order)
        best = sub.sort_values(
            ["val_f1_mean", "val_acc_mean", "_spec_order"],
            ascending=[False, False, True],
            kind="mergesort",
        ).iloc[0]
        selected.append(by_key[best["config_key"]])
    return selected


def summarize_selected_methods(results: list[dict], selected_specs: list[dict]) -> pd.DataFrame:
    method_names = {
        "feature_extraction": "AST · Feature Extraction",
        "lora": "AST · LoRA (best val)",
        "fine_tune": "AST · Full Fine-tune (best val)",
    }
    rows = []
    for spec in selected_specs:
        sub = [r for r in results if r["config_key"] == spec["key"]]
        if not sub:
            continue
        test_acc, test_acc_std = _mean_std([r["test_acc"] for r in sub])
        test_f1, test_f1_std = _mean_std([r["test_f1"] for r in sub])
        ext_acc, ext_acc_std = _mean_std([r["ext_acc"] for r in sub])
        ext_f1, ext_f1_std = _mean_std([r["ext_f1"] for r in sub])
        rows.append({
            "方法": method_names[spec["mode"]],
            "内部 Acc": f"{test_acc:.3f} ± {test_acc_std:.3f}",
            "内部 F1": f"{test_f1:.3f} ± {test_f1_std:.3f}",
            "外部 Acc": f"{ext_acc:.3f} ± {ext_acc_std:.3f}",
            "外部 F1": f"{ext_f1:.3f} ± {ext_f1_std:.3f}",
            "Gap (Acc)": f"{test_acc - ext_acc:+.3f}",
        })
    return pd.DataFrame(rows)


def selected_config_table(selected_specs: list[dict], summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in selected_specs:
        sub = summary[summary["config_key"] == spec["key"]]
        if sub.empty:
            continue
        row = sub.iloc[0]
        rows.append({
            "family": spec["family"],
            "selected_config": spec["display_name"],
            "config_key": spec["key"],
            "val_f1": row["val_f1"],
            "val_acc": row["val_acc"],
            "external_acc": row["ext_acc"],
        })
    return pd.DataFrame(rows)
