"""Visualization helpers for discrete codec tokens."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common.plotting import finish_figure, setup_plot_style


def codebook_usage(tokens: np.ndarray, vocab_size: int | None = None) -> dict[int, int]:
    """Count token ids across a token array."""
    flat = np.asarray(tokens).reshape(-1)
    if vocab_size is None:
        vocab_size = int(flat.max()) + 1 if flat.size else 0
    counts = np.bincount(flat.astype(int), minlength=vocab_size)
    return {index: int(count) for index, count in enumerate(counts) if count}


def plot_token_matrix(tokens: np.ndarray, out_path: Path | None = None) -> plt.Figure:
    """Plot token ids as a codebook-by-time image."""
    setup_plot_style()
    matrix = np.asarray(tokens)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    fig, ax = plt.subplots(figsize=(9, 3))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="gray_r")
    if matrix.shape[0] == 1:
        # 展平后的单条 token 流：横轴是序列位置而非帧，也不对应某个码本
        ax.set_xlabel("token 位置")
        ax.set_yticks([])
    else:
        ax.set_xlabel("帧")
        ax.set_ylabel("码本")
        if matrix.shape[0] <= 16:
            ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_title("编解码器 token 矩阵")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return finish_figure(fig, out_path)


def plot_codebook_usage(tokens: np.ndarray, out_path: Path | None = None) -> plt.Figure:
    """Plot non-empty codebook token counts."""
    setup_plot_style()
    usage = codebook_usage(tokens)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(list(usage.keys()), list(usage.values()), color="0.35")
    ax.set_xlabel("token ID")
    ax.set_ylabel("计数")
    ax.set_title("码本使用量")
    return finish_figure(fig, out_path)
