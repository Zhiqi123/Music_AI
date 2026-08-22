"""Riffusion-style spectrogram diffusion sketch."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common.plotting import finish_figure, setup_plot_style


def plot_riffusion_process(out_path: Path | None = None) -> plt.Figure:
    """Draw denoising steps over a spectrogram-like image."""
    setup_plot_style()
    rng = np.random.default_rng(0)
    base = np.outer(np.linspace(0.2, 1.0, 64), np.sin(np.linspace(0, 8, 96)) ** 2)
    noise = rng.normal(0.0, 1.0, base.shape)
    # Blend factor rises across frames so the time-frequency structure
    # emerges gradually instead of only appearing in the last frame.
    blend = [0.0, 0.55, 0.85, 1.0]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.6), sharex=True, sharey=True)
    for i, (ax, alpha) in enumerate(zip(axes, blend)):
        image = (1.0 - alpha) * noise + alpha * base
        span = image.max() - image.min()
        if span > 0:
            image = (image - image.min()) / span
        ax.imshow(image, origin="lower", aspect="auto", cmap="gray_r")
        ax.set_title(f"第 {i} 步")
        ax.set_xticks([])
        ax.set_yticks([])
    return finish_figure(fig, out_path)
