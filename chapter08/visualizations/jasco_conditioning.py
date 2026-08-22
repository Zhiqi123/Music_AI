"""JASCO-style conditioning graph visualization."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from _common.plotting import finish_figure, setup_plot_style


def plot_conditioning_graph(out_path: Path | None = None) -> plt.Figure:
    """Draw text, chords, beat, and melody conditions feeding audio tokens."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(5.5, 2.2))
    nodes = {
        "文本": (-2.2, 0.84),
        "和弦": (-2.2, 0.28),
        "节拍": (-2.2, -0.28),
        "旋律": (-2.2, -0.84),
        "条件编码器": (-0.55, 0.0),
        "token 生成器": (1.25, 0.0),
    }
    sizes = {
        "文本": (1.18, 0.36),
        "和弦": (1.18, 0.36),
        "节拍": (1.18, 0.36),
        "旋律": (1.18, 0.36),
        "条件编码器": (1.36, 0.40),
        "token 生成器": (1.36, 0.40),
    }
    for label, (x, y) in nodes.items():
        width, height = sizes[label]
        ax.add_patch(
            plt.Rectangle(
                (x - width / 2, y - height / 2),
                width,
                height,
                fill=False,
                edgecolor="0.2",
                linewidth=1.0,
            )
        )
        ax.text(x, y, label, ha="center", va="center", fontsize=9)
    encoder_x, encoder_y = nodes["条件编码器"]
    encoder_width, _ = sizes["条件编码器"]
    encoder_left = (encoder_x - encoder_width / 2, encoder_y)
    for source in ["文本", "和弦", "节拍", "旋律"]:
        source_x, source_y = nodes[source]
        source_width, _ = sizes[source]
        source_right = (source_x + source_width / 2, source_y)
        ax.annotate(
            "",
            xy=encoder_left,
            xytext=source_right,
            arrowprops={
                "arrowstyle": "->",
                "color": "0.25",
                "linewidth": 0.9,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )
    generator_x, generator_y = nodes["token 生成器"]
    generator_width, _ = sizes["token 生成器"]
    ax.annotate(
        "",
        xy=(generator_x - generator_width / 2, generator_y),
        xytext=(encoder_x + encoder_width / 2, encoder_y),
        arrowprops={
            "arrowstyle": "->",
            "color": "0.25",
            "linewidth": 0.9,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )
    x_min = min(x - sizes[label][0] / 2 for label, (x, _) in nodes.items())
    x_max = max(x + sizes[label][0] / 2 for label, (x, _) in nodes.items())
    y_min = min(y - sizes[label][1] / 2 for label, (_, y) in nodes.items())
    y_max = max(y + sizes[label][1] / 2 for label, (_, y) in nodes.items())
    ax.set_xlim(x_min - 0.1, x_max + 0.1)
    ax.set_ylim(y_min - 0.08, y_max + 0.08)
    ax.axis("off")
    return finish_figure(fig, out_path, pad_inches=0.02)
