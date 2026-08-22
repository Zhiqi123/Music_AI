"""第十章 Notebook 共用的灰度绘图风格与落盘助手。"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

NOTEBOOK_DISPLAY_DPI = 120
FIGURE_SAVE_DPI = 600
ACADEMIC_CJK_FONTS = (
    "Source Han Sans SC",
    "Source Han Sans CN",
    "Source Han Sans",
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "PingFang SC",
    "Songti SC",
    "Heiti SC",
    "STHeiti",
    "FandolHei",
    "FandolSong",
    "Arial Unicode MS",
    "DejaVu Sans",
)


def select_academic_font() -> str:
    """选择当前环境中实际存在的中英文字体，避免静默回退。"""
    available = {entry.name for entry in font_manager.fontManager.ttflist}
    for candidate in ACADEMIC_CJK_FONTS:
        if candidate in available:
            return candidate
    return "DejaVu Sans"


def setup_plot_style() -> str:
    """设置全灰度、600 dpi 的书稿插图风格，并返回所用字体。"""
    selected_font = select_academic_font()
    plt.rcParams.update(
        {
            "figure.dpi": NOTEBOOK_DISPLAY_DPI,
            "savefig.dpi": FIGURE_SAVE_DPI,
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "axes.grid": True,
            "axes.grid.axis": "x",
            "grid.alpha": 0.25,
            "grid.color": "0.45",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "0.20",
            "axes.labelcolor": "0.08",
            "xtick.color": "0.08",
            "ytick.color": "0.08",
            "text.color": "0.08",
            "font.family": selected_font,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return selected_font


def finish_figure(
    fig: plt.Figure,
    out_path: Path | str,
    *,
    pad_inches: float = 0.08,
) -> Path:
    """收紧布局并按 600 dpi、白色背景保存 PNG。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        out_path,
        bbox_inches="tight",
        pad_inches=pad_inches,
        dpi=FIGURE_SAVE_DPI,
        facecolor="white",
        transparent=False,
    )
    return out_path
