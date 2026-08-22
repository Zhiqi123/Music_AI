"""Plot helpers shared by Chapter 7 notebooks."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .spectrogram import amplitude_to_db, stft


GRAY_IMAGE_CMAP = "gray_r"
GRAY_MASK_CMAP = "gray_r"
BAR_GRAY = "0.35"
LINE_GRAYS = ("0.12", "0.28", "0.44", "0.60", "0.76")
NOTEBOOK_DISPLAY_DPI = 120
FIGURE_SAVE_DPI = 600

DISPLAY_TERMS = {
    "mixture": "混合信号",
    "source": "音源",
    "harmonic": "谐波源",
    "percussive": "打击源",
    "bass": "贝斯",
    "vocals": "人声",
    "vocal": "人声",
    "drums": "鼓",
    "other": "其他",
    "accompaniment": "伴奏",
    "piano": "钢琴",
    "guitar": "吉他",
    "synths": "合成器",
    "strings": "弦乐组",
    "woodwinds": "木管组",
    "brass": "铜管组",
    "choirs": "合唱",
    "percussion": "打击乐组",
    "background": "背景",
    "foreground": "前景",
    "no_vocals": "非人声",
    "low_components": "低频分量",
    "other_components": "其他分量",
}


def setup_plot_style() -> None:
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
            "font.sans-serif": [
                "Noto Sans CJK SC",
                "Source Han Sans SC",
                "PingFang SC",
                "Heiti SC",
                "Microsoft YaHei",
                "SimHei",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
        }
    )


def display_label(name: str) -> str:
    if ":" in name:
        prefix, suffix = name.split(":", 1)
        return f"{prefix}:{display_label(suffix)}"
    return DISPLAY_TERMS.get(name, name)


def plot_waveforms(
    sources: dict[str, np.ndarray],
    sr: int,
    out_path: Path | None = None,
) -> plt.Figure:
    setup_plot_style()
    fig, axes = plt.subplots(len(sources), 1, figsize=(10, 1.7 * len(sources)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (name, audio) in zip(axes, sources.items()):
        y = _to_mono(audio)
        t = np.arange(y.size) / sr
        ax.plot(t, y, linewidth=0.8, color=LINE_GRAYS[0])
        ax.set_ylabel(display_label(name))
        ax.set_ylim(-1.05, 1.05)
    axes[-1].set_xlabel("时间 (s)")
    return _finish(fig, out_path)


def plot_spectrogram(
    audio: np.ndarray,
    sr: int,
    title: str,
    out_path: Path | None = None,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> plt.Figure:
    setup_plot_style()
    y = _to_mono(audio)
    spec = stft(y, n_fft=n_fft, hop_length=hop_length)
    db = amplitude_to_db(np.abs(spec))
    fig, ax = plt.subplots(figsize=(9, 4))
    _imshow_time_frequency(ax, db, sr, hop_length, title)
    return _finish(fig, out_path)


def plot_stem_spectrogram_grid(
    stems: dict[str, np.ndarray],
    sr: int,
    out_path: Path | None = None,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> plt.Figure:
    setup_plot_style()
    n = len(stems)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.0 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (name, audio) in zip(axes, stems.items()):
        spec = stft(_to_mono(audio), n_fft=n_fft, hop_length=hop_length)
        db = amplitude_to_db(np.abs(spec))
        _imshow_time_frequency(ax, db, sr, hop_length, display_label(name))
    axes[-1].set_xlabel("时间 (s)")
    return _finish(fig, out_path)


def plot_energy_bars(values: dict[str, float], out_path: Path | None = None) -> plt.Figure:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7, 3))
    names = list(values)
    ax.bar([display_label(name) for name in names], [values[name] for name in names], color=BAR_GRAY)
    ax.set_ylabel("能量比")
    ax.tick_params(axis="x", rotation=25)
    return _finish(fig, out_path)


def plot_matrix_grid(
    matrices: dict[str, np.ndarray],
    out_path: Path | None = None,
    cmap: str = GRAY_IMAGE_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    ncols: int | None = None,
) -> plt.Figure:
    setup_plot_style()
    n = len(matrices)
    ncols = n if ncols is None else max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows), squeeze=False)
    flat_axes = axes.ravel()
    for ax, (name, matrix) in zip(flat_axes, matrices.items()):
        image = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(display_label(name))
        ax.set_xlabel("帧")
        ax.set_ylabel("频率 bin")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    for ax in flat_axes[n:]:
        ax.axis("off")
    return _finish(fig, out_path)


def _imshow_time_frequency(
    ax: plt.Axes,
    matrix: np.ndarray,
    sr: int,
    hop_length: int,
    title: str,
) -> None:
    extent = [0, matrix.shape[1] * hop_length / sr, 0, sr / 2]
    ax.imshow(matrix, origin="lower", aspect="auto", extent=extent, cmap=GRAY_IMAGE_CMAP)
    ax.set_title(title)
    ax.set_ylabel("频率 (Hz)")


def _to_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=0)


def _finish(fig: plt.Figure, out_path: Path | None) -> plt.Figure:
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=FIGURE_SAVE_DPI)
    return fig
