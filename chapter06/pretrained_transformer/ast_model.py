"""AST 模型加载与分类头替换。

预训练链条：ImageNet-2012（DeiT）→ AudioSet → 下游
HF checkpoint: MIT/ast-finetuned-audioset-10-10-0.4593

本地权重目录：`outputs/checkpoints/ast/` 存有config、preprocessor配置和safetensors，
优先从本地加载（绕开 huggingface_hub 1.x + httpx 在国内网络下的偶发 client-closed 问题）。
本地文件缺失时会回退到HF Hub模型名；离线模式下该回退会直接报错。
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import ASTModel
from transformers.utils import logging as hf_logging

# AudioSet 预训练头 (classifier.dense / classifier.layernorm) 在我们换 fc_transfer 时
# 下游分类头与 AudioSet 头形状不同，因此加载时预期会报告对应键不匹配；只打印一次。
hf_logging.set_verbosity_error()

AST_HF_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_HIDDEN = 768
AST_LOCAL_DIR = Path(__file__).parent / "outputs" / "checkpoints" / "ast"


def _resolve_ast_source(hf_name: str) -> str:
    """本地 checkpoint 存在则返回本地路径，否则返回 HF 名。"""
    if (AST_LOCAL_DIR / "config.json").exists() and (
        AST_LOCAL_DIR / "model.safetensors"
    ).exists():
        return str(AST_LOCAL_DIR)
    return hf_name


class TransferAST(nn.Module):
    """HuggingFace ASTModel + Linear 分类头。"""

    def __init__(self, n_classes: int = 6, hf_name: str = AST_HF_NAME):
        super().__init__()
        self.base = ASTModel.from_pretrained(_resolve_ast_source(hf_name))
        self.fc_transfer = nn.Linear(AST_HIDDEN, n_classes)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """input_values: (B, max_length, n_mels)。返回 (B, n_classes)。"""
        outputs = self.base(input_values=input_values)
        # 该 AudioSet checkpoint 继承带 distillation token 的 DeiT 权重。
        # Transformers 的 ASTModel 按原始 AST 迁移方案对前两个特殊 token
        # 取平均，作为 768 维音频表示。
        return self.fc_transfer(outputs.pooler_output)
