"""Receptive-field utilities for dilated causal convolution stacks."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common.plotting import finish_figure, setup_plot_style


def dilation_cycle(layers_per_cycle: int, cycles: int) -> tuple[int, ...]:
    """Return ``1, 2, 4, ...`` dilations repeated for each cycle."""
    if layers_per_cycle < 1:
        raise ValueError("layers_per_cycle must be positive")
    if cycles < 1:
        raise ValueError("cycles must be positive")
    base = tuple(2**i for i in range(layers_per_cycle))
    return base * cycles


def receptive_field_size(kernel_size: int, dilations: list[int] | tuple[int, ...]) -> int:
    """Compute samples visible to one output of a causal dilated stack."""
    if kernel_size < 1:
        raise ValueError("kernel_size must be positive")
    if any(dilation < 1 for dilation in dilations):
        raise ValueError("dilations must be positive")
    return 1 + (kernel_size - 1) * int(sum(dilations))


def dependency_matrix(
    sequence_length: int,
    kernel_size: int,
    dilations: list[int] | tuple[int, ...],
) -> np.ndarray:
    """Return a boolean matrix showing which inputs can affect each output."""
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    deps = np.eye(sequence_length, dtype=bool)
    for dilation in dilations:
        new_deps = np.zeros_like(deps)
        for out_index in range(sequence_length):
            for tap in range(kernel_size):
                prev_index = out_index - tap * dilation
                if prev_index >= 0:
                    new_deps[out_index] |= deps[prev_index]
        deps = new_deps
    return deps


def plot_receptive_field(
    sequence_length: int,
    kernel_size: int,
    dilations: list[int] | tuple[int, ...],
    out_path: Path | None = None,
) -> plt.Figure:
    """Visualize causal dependencies as a binary matrix."""
    setup_plot_style()
    matrix = dependency_matrix(sequence_length, kernel_size, dilations)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(matrix, origin="lower", aspect="auto", cmap="gray_r")
    ax.set_title(f"感受野 = {receptive_field_size(kernel_size, dilations)} 个采样点")
    ax.set_xlabel("输入采样点索引")
    ax.set_ylabel("输出采样点索引")
    return finish_figure(fig, out_path)
