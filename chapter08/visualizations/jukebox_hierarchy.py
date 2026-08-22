"""Jukebox-style hierarchy visualization."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from _common.plotting import finish_figure, setup_plot_style


def plot_jukebox_hierarchy(out_path: Path | None = None) -> plt.Figure:
    """Draw a three-level VQ-VAE/prior hierarchy."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(6.2, 2.35))
    labels = ["波形", "底层 VQ 码", "中层 VQ 码", "顶层 VQ 码", "歌词/风格先验"]
    box_height = 0.34
    y_step = 0.56
    y = [index * y_step for index in range(len(labels))]
    widths = [6.1, 5.0, 3.85, 2.85, 2.25]
    for yi, width, label in zip(y, widths, labels):
        ax.add_patch(
            plt.Rectangle(
                (-width / 2, yi - box_height / 2),
                width,
                box_height,
                fill=False,
                edgecolor="0.2",
                linewidth=1.0,
            )
        )
        ax.text(0, yi, label, ha="center", va="center", fontsize=9)
        if yi > 0:
            previous_top = yi - y_step + box_height / 2
            current_bottom = yi - box_height / 2
            ax.annotate(
                "",
                xy=(0, current_bottom - 0.015),
                xytext=(0, previous_top + 0.015),
                arrowprops={"arrowstyle": "->", "color": "0.25", "linewidth": 0.9},
            )
    x_margin = 0.08
    y_margin = 0.07
    ax.set_xlim(-max(widths) / 2 - x_margin, max(widths) / 2 + x_margin)
    ax.set_ylim(min(y) - box_height / 2 - y_margin, max(y) + box_height / 2 + y_margin)
    ax.axis("off")
    return finish_figure(fig, out_path, pad_inches=0.02)
