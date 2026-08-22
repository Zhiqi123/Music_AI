"""第六章共享绘图工具：中文字体配置 + 混淆矩阵 + 类名展示规范化。"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# CTIS 原始 cname 中部分类名带数字后缀（如 '唢呐2'），按乐器门类规范名展示
CNAME_DISPLAY_REMAP: dict[str, str] = {"唢呐2": "唢呐"}


def display_cname(raw: str) -> str:
    """把 CTIS 原始 cname 映射到展示名（如 '唢呐2' → '唢呐'）。"""
    return CNAME_DISPLAY_REMAP.get(raw, raw)


def setup_chinese_font() -> None:
    """把 matplotlib 中文字体优先级设为跨平台列表（macOS/Windows/Linux），避免 DejaVu Sans 缺字警告。"""
    plt.rcParams["font.sans-serif"] = [
        "Hiragino Sans GB",
        "PingFang SC",
        "Arial Unicode MS",
        "STHeiti",
        "Heiti TC",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def plot_confusion_matrix(
    ax,
    y_true,
    y_pred,
    class_names: list[str],
    title: str,
    *,
    labels=None,
):
    """在给定 ax 上绘制行归一化混淆矩阵。

    Args:
        ax: matplotlib Axes。
        y_true, y_pred: 真实/预测标签。
        class_names: 展示用类名列表，按对应标签的升序排列。
        title: 子图标题。
        labels: 传给 sklearn confusion_matrix 的标签顺序，None 时按数值升序。

    Returns:
        ax.imshow 返回的 image，便于外部统一加 colorbar。

    口径：色阶 = 行归一化召回率（0–1）；文本与色阶一致，均为百分比；0% 也标出。
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    n = cm.shape[0]
    im = ax.imshow(cm_norm, cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(title)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="0.85", linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    for i in range(n):
        for j in range(n):
            pct = cm_norm[i, j] * 100
            text_color = "white" if cm_norm[i, j] > 0.5 else "0.30"
            ax.text(
                j, i, f"{pct:.0f}",
                ha="center", va="center",
                color=text_color, fontsize=8,
            )
    return im


def add_recall_colorbar(fig, axes, im=None):
    """为一组并排的混淆矩阵子图加统一的行归一化召回率 colorbar。"""
    if im is None:
        im = axes[0].images[0]
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("行归一化召回率")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
    return cbar
