"""6.2简单CNN：log-mel谱、3个卷积块、数据增强对比与外部评估。"""
from .augment import AugmentConfig, apply_spec_augment, apply_waveform_augment
from .dataset import (
    CLIP_SAMPLES,
    CLIP_SEC,
    HOP_LENGTH,
    N_FFT,
    N_FRAMES,
    N_MELS,
    SAMPLE_RATE,
    MelSegmentDataset,
    compute_logmel,
)
from .model import SimpleAudioCNN
from .train import TrainConfig, run_experiment

__all__ = [
    "AugmentConfig", "apply_waveform_augment", "apply_spec_augment",
    "MelSegmentDataset", "compute_logmel",
    "SAMPLE_RATE", "CLIP_SEC", "CLIP_SAMPLES", "N_MELS", "N_FFT", "HOP_LENGTH", "N_FRAMES",
    "SimpleAudioCNN",
    "TrainConfig", "run_experiment",
]
