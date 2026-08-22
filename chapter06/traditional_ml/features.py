"""音频特征提取（6.1 传统机器学习管线）。

特征构成（98 维）：
    MFCC mean(13) + MFCC std(13)
  + Delta-MFCC mean(13) + Delta-MFCC std(13)
  + Chroma mean(12) + Chroma std(12)
  + Spectral Contrast mean(7) + Spectral Contrast std(7)
  + Spectral Centroid mean/std
  + Zero-Crossing Rate mean/std
  + RMS mean/std
  + Onset Strength mean/std
  = 98

Delta-MFCC 描述 MFCC 随时间的局部变化；Spectral Contrast 描述各子带
的峰谷差；Onset Strength 描述起音相关的能量变化。这些统计量可为分类
提供补充信息，但其有效性取决于数据与录音条件，需要通过实验检验。
"""
from __future__ import annotations

import librosa
import numpy as np

SAMPLE_RATE = 24000
N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 13
N_CHROMA = 12
N_CONTRAST_BANDS = 7
FEATURE_DIM = 98


def extract_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """对单段单声道音频提取 98 维特征向量。

    Args:
        audio: 1D float numpy 数组，单声道波形。
        sr: 采样率。

    Returns:
        shape (98,) 的 float32 向量。
    """
    if audio.ndim != 1:
        raise ValueError(f"expected 1D mono audio, got shape {audio.shape}")

    # MFCC + Delta
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    delta_mfcc = librosa.feature.delta(mfcc)

    # Chroma
    chroma = librosa.feature.chroma_stft(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_chroma=N_CHROMA
    )

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_bands=N_CONTRAST_BANDS - 1,
    )

    # Global
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)
    onset_env = librosa.onset.onset_strength(
        y=audio, sr=sr, hop_length=HOP_LENGTH
    )

    parts = [
        mfcc.mean(axis=1), mfcc.std(axis=1),
        delta_mfcc.mean(axis=1), delta_mfcc.std(axis=1),
        chroma.mean(axis=1), chroma.std(axis=1),
        contrast.mean(axis=1), contrast.std(axis=1),
        np.array([centroid.mean(), centroid.std()]),
        np.array([zcr.mean(), zcr.std()]),
        np.array([rms.mean(), rms.std()]),
        np.array([onset_env.mean(), onset_env.std()]),
    ]
    feat = np.concatenate(parts).astype(np.float32)
    assert feat.shape == (FEATURE_DIM,), f"unexpected feature dim: {feat.shape}"
    return feat
