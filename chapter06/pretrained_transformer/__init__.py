"""6.4 预训练 Transformer 迁移学习：AST 三模式 + CLAP zero-shot。"""
from .ast_model import TransferAST, AST_HF_NAME
from .clap_model import CLAPZeroShot, CLAP_CKPT_URL, CLAP_CKPT_NAME, download_clap_weights
from .dataset import ASTSegmentDataset
from .prompts import PROMPT_TEMPLATES, INSTRUMENT_EN, render_prompts
from .transfer import freeze_backbone, apply_lora, fine_tune_all, count_trainable
from .train import ASTTrainConfig, run_ast_experiment, TRANSFER_MODES
from .sweep import (
    compact_sweep_configs,
    extended_sweep_configs,
    run_ast_config_grid,
    selected_config_table,
    select_best_specs_by_val,
    summarize_config_results,
    summarize_selected_methods,
)

__all__ = [
    "TransferAST", "AST_HF_NAME",
    "ASTSegmentDataset",
    "freeze_backbone", "apply_lora", "fine_tune_all", "count_trainable",
    "ASTTrainConfig", "run_ast_experiment", "TRANSFER_MODES",
    "compact_sweep_configs", "extended_sweep_configs", "run_ast_config_grid",
    "selected_config_table", "select_best_specs_by_val",
    "summarize_config_results", "summarize_selected_methods",
    "CLAPZeroShot", "CLAP_CKPT_URL", "CLAP_CKPT_NAME", "download_clap_weights",
    "PROMPT_TEMPLATES", "INSTRUMENT_EN", "render_prompts",
]
