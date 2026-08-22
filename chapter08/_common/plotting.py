"""Plot helpers shared by Chapter 8 notebooks."""
from __future__ import annotations

from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np

from .audio_io import to_mono


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
    """Use restrained grayscale plots suitable for the book."""
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


def plot_waveform(
    audio: np.ndarray,
    sr: int,
    title: str = "波形",
    out_path: Path | None = None,
) -> plt.Figure:
    """Plot a mono waveform."""
    setup_plot_style()
    y = to_mono(audio)
    t = np.arange(y.size) / sr
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.plot(t, y, color=LINE_GRAYS[0], linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("时间（秒）")
    ax.set_ylabel("振幅")
    ax.set_ylim(-1.05, 1.05)
    return finish_figure(fig, out_path)


def plot_spectrogram(
    audio: np.ndarray,
    sr: int,
    title: str | None = "声谱图",
    out_path: Path | None = None,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> plt.Figure:
    """Plot a dB magnitude spectrogram."""
    setup_plot_style()
    y = to_mono(audio)
    spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    db = librosa.amplitude_to_db(spec, ref=np.max)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    extent = [0, db.shape[1] * hop_length / sr, 0, sr / 2]
    image = ax.imshow(db, origin="lower", aspect="auto", extent=extent, cmap=GRAY_IMAGE_CMAP)
    if title:
        ax.set_title(title)
    ax.set_xlabel("时间（秒）")
    ax.set_ylabel("频率（Hz）")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return finish_figure(fig, out_path)


def finish_figure(
    fig: plt.Figure,
    out_path: Path | None = None,
    pad_inches: float = 0.1,
    tight: bool = True,
) -> plt.Figure:
    """Tighten a Matplotlib figure and save it when ``out_path`` is given."""
    if tight:
        fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=pad_inches, dpi=FIGURE_SAVE_DPI)
    return fig
