"""6.3 预训练 CNN 迁移学习：PANNs CNN14 + 三种经典迁移模式。"""
from .model import Cnn14, TransferCnn14
from .transfer import freeze_backbone, fine_tune_all, layer_wise_lr
from .train import TrainConfig, run_experiment

__all__ = [
    "Cnn14",
    "TransferCnn14",
    "freeze_backbone",
    "fine_tune_all",
    "layer_wise_lr",
    "TrainConfig",
    "run_experiment",
]
