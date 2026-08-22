"""MAGNeT-style masked-token generation demo."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from _common.plotting import finish_figure, setup_plot_style


def mask_schedule(length: int, steps: int) -> list[np.ndarray]:
    """Return boolean masks that reveal more tokens each step.

    The reveal order is a seeded random permutation, mirroring MAGNeT-style
    non-autoregressive filling instead of strict left-to-right decoding.
    """
    if length < 1 or steps < 1:
        raise ValueError("length and steps must be positive")
    order = np.random.default_rng(0).permutation(length)
    masks = []
    for step in range(steps):
        visible = int(round((step + 1) * length / steps))
        mask = np.ones(length, dtype=bool)
        mask[order[:visible]] = False
        masks.append(mask)
    return masks


def plot_masked_generation(
    out_path: Path | None = None,
    length: int = 32,
    steps: int = 5,
) -> plt.Figure:
    """Visualize iterative masked-token filling."""
    setup_plot_style()
    masks = mask_schedule(length, steps)
    matrix = np.vstack([mask.astype(float) for mask in masks])
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.imshow(matrix, aspect="auto", cmap="gray", vmin=0, vmax=1)
    ax.set_xlabel("token 位置")
    ax.set_ylabel("生成步")
    ax.legend(
        handles=[
            Patch(facecolor="black", edgecolor="0.25", label="已确定 token"),
            Patch(facecolor="white", edgecolor="0.25", label="遮盖"),
        ],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.08),
        fontsize=9,
        framealpha=0.9,
    )
    return finish_figure(fig, out_path)
