"""第九章 Notebook 共用的绘图风格与落盘助手。"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

GRAY_IMAGE_CMAP = "gray_r"
BAR_GRAY = "0.35"
LINE_GRAYS = ("0.12", "0.28", "0.44", "0.60", "0.76")
NOTEBOOK_DISPLAY_DPI = 120
FIGURE_SAVE_DPI = 600
ACADEMIC_CJK_FONTS = [
    "Source Han Sans SC",
    "Source Han Sans CN",
    "Source Han Sans",
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "PingFang SC",
    "Heiti SC",
    "STHeiti",
    "Songti SC",
    "SimSong",
    "Microsoft YaHei",
    "SimHei",
    "FandolHei",
    "FandolSong",
    "Arial Unicode MS",
    "DejaVu Sans",
]


def setup_plot_style() -> None:
    """灰度学术风格,与全书插图一致。"""
    plt.rcParams.update(
        {
            "figure.dpi": NOTEBOOK_DISPLAY_DPI,
            "savefig.dpi": FIGURE_SAVE_DPI,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "0.25",
            "axes.labelcolor": "0.10",
            "xtick.color": "0.10",
            "ytick.color": "0.10",
            "text.color": "0.10",
            "image.cmap": GRAY_IMAGE_CMAP,
            "font.family": "sans-serif",
            "font.sans-serif": ACADEMIC_CJK_FONTS,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def finish_figure(fig: plt.Figure, out_path: Path | str | None = None, pad_inches: float = 0.1) -> plt.Figure:
    """收紧布局;给出 out_path 时按出书分辨率落盘。"""
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=pad_inches, dpi=FIGURE_SAVE_DPI)
    return fig
