"""Toy implementation and visualization for state-space scanning."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common.plotting import finish_figure, setup_plot_style


def ssm_scan(inputs: np.ndarray, a: float = 0.9, b: float = 0.2) -> np.ndarray:
    """Run a scalar recurrent state-space scan."""
    state = 0.0
    outputs = []
    for value in np.asarray(inputs, dtype=np.float32):
        state = a * state + b * float(value)
        outputs.append(state)
    return np.asarray(outputs, dtype=np.float32)


def plot_scan(out_path: Path | None = None) -> plt.Figure:
    """Plot input and scanned state for a long-sequence modeling sketch."""
    setup_plot_style()
    x = np.zeros(80, dtype=np.float32)
    x[[5, 20, 55]] = [1.0, -0.7, 0.8]
    y = ssm_scan(x)
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.plot(x, label="输入", color="0.65", linewidth=1.2)
    ax.plot(y, label="状态", color="0.15", linewidth=1.5)
    ax.set_xlabel("token 索引")
    ax.legend(frameon=False)
    return finish_figure(fig, out_path)
