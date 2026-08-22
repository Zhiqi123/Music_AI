"""三种经典迁移学习策略。

- Feature Extraction: 冻结 backbone，只训练分类头
- Full Fine-tuning: 解冻所有层
- Layer-wise LR: backbone 低学习率，head 高学习率
"""
from __future__ import annotations

import torch.nn as nn

from .model import TransferCnn14


def freeze_backbone(model: TransferCnn14) -> TransferCnn14:
    """冻结 CNN14 backbone 所有参数，只保留 fc_transfer 可训练。"""
    for param in model.base.parameters():
        param.requires_grad = False
    for param in model.fc_transfer.parameters():
        param.requires_grad = True
    return model


def fine_tune_all(model: TransferCnn14) -> TransferCnn14:
    """解冻下游前向路径；保留未使用的 AudioSet 分类头冻结。"""
    for param in model.parameters():
        param.requires_grad = True
    for param in model.base.fc_audioset.parameters():
        param.requires_grad = False
    return model


def layer_wise_lr(
    model: TransferCnn14,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-3,
) -> list[dict]:
    """返回 optimizer param_groups，backbone 用低 lr，head 用高 lr。

    用法：
        param_groups = layer_wise_lr(model, 1e-5, 1e-3)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    """
    fine_tune_all(model)
    return [
        {
            "params": (p for p in model.base.parameters() if p.requires_grad),
            "lr": backbone_lr,
        },
        {"params": model.fc_transfer.parameters(), "lr": head_lr},
    ]
