"""AST 三种迁移学习策略。

- Feature Extraction: 冻结 ASTModel，只训练 fc_transfer
- LoRA: peft 注入低秩 adapter 到 query / value，仅训练 adapter + head
- Full Fine-tuning: 解冻所有层
"""
from __future__ import annotations

from peft import LoraConfig, get_peft_model

from .ast_model import TransferAST


def freeze_backbone(model: TransferAST) -> TransferAST:
    """冻结 ASTModel base，只保留 fc_transfer 可训练。"""
    for param in model.base.parameters():
        param.requires_grad = False
    for param in model.fc_transfer.parameters():
        param.requires_grad = True
    return model


def apply_lora(
    model: TransferAST,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
) -> TransferAST:
    """在 ASTModel 的 query / value 注入 LoRA，head 保持可训练。"""
    for param in model.base.parameters():
        param.requires_grad = False

    # transformers 5.9+ 的 AST attention 子模块命名为
    # base.layers.{i}.attention.{q_proj,k_proj,v_proj}（与 LLM 风格对齐）；
    # 5.5 及更早是 base.encoder.layer.{i}.attention.attention.{query,key,value}。
    # 这里按 5.9 命名匹配——若 PEFT 报"target modules not found"，请用
    # `[n for n,_ in model.base.named_modules() if 'attention' in n][:6]`
    # 自查一次模块名再调整。
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_dropout,
        bias="none",
    )
    model.base = get_peft_model(model.base, config)

    for param in model.fc_transfer.parameters():
        param.requires_grad = True
    return model


def fine_tune_all(model: TransferAST) -> TransferAST:
    """解冻所有参数。"""
    for param in model.parameters():
        param.requires_grad = True
    return model


def count_trainable(model) -> int:
    """统计可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
