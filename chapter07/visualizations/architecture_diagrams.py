"""Architecture diagrams for source-separation teaching notebooks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Patch, Rectangle
import numpy as np

from chapter07._common.plotting import FIGURE_SAVE_DPI, LINE_GRAYS, setup_plot_style


@dataclass(frozen=True)
class TensorStage:
    name: str
    shape: tuple[int, int, int, int]
    role: str


@dataclass(frozen=True)
class FrequencyBand:
    name: str
    start_bin: int
    end_bin: int
    color: str


@dataclass(frozen=True)
class SkipConnectionSpec:
    start_stage: int
    end_stage: int
    y_offset: float
    label: str = "跳跃连接"
    curvature: float = -0.15
    label_y_offset: float = 0.03


def unet_shape_flow(
    batch: int = 1,
    channels: int = 2,
    freq_bins: int = 256,
    frames: int = 128,
    base_channels: int = 16,
) -> list[TensorStage]:
    """Return a compact U-Net tensor-shape trace."""
    if min(batch, channels, freq_bins, frames, base_channels) < 1:
        raise ValueError("all dimensions must be positive")
    f1, t1 = freq_bins, frames
    f2, t2 = max(freq_bins // 2, 1), max(frames // 2, 1)
    f3, t3 = max(freq_bins // 4, 1), max(frames // 4, 1)
    return [
        TensorStage("输入 STFT", (batch, channels, f1, t1), "幅度或复数通道"),
        TensorStage("编码器 1", (batch, base_channels, f1, t1), "局部时频模式"),
        TensorStage("编码器 2", (batch, base_channels * 2, f2, t2), "扩大上下文"),
        TensorStage("瓶颈层", (batch, base_channels * 4, f3, t3), "压缩表示"),
        TensorStage("解码器 2", (batch, base_channels * 2, f2, t2), "上采样 + 跳跃连接"),
        TensorStage("解码器 1", (batch, base_channels, f1, t1), "恢复分辨率"),
        TensorStage("掩蔽", (batch, channels, f1, t1), "有界源掩蔽"),
    ]


def make_frequency_bands(n_bins: int = 1024) -> list[FrequencyBand]:
    """Create coarse nonuniform bands for band-split diagrams."""
    if n_bins < 16:
        raise ValueError("n_bins must be at least 16")
    edges = [
        0,
        max(n_bins // 32, 1),
        max(n_bins // 16, 2),
        max(n_bins // 8, 3),
        max(n_bins // 4, 4),
        max(n_bins // 2, 5),
        n_bins,
    ]
    edges = sorted(set(edges))
    names = ["超低频", "低频", "中低频", "中频", "中高频", "高频"]
    colors = ["0.18", "0.30", "0.42", "0.54", "0.66", "0.78"]
    return [
        FrequencyBand(names[i], edges[i], edges[i + 1], colors[i])
        for i in range(len(edges) - 1)
    ]


def rotary_embedding_angles(seq_len: int = 64, dim: int = 16) -> np.ndarray:
    """Return RoPE angles with shape ``(seq_len, dim / 2)``."""
    if seq_len < 1 or dim < 2 or dim % 2:
        raise ValueError("seq_len must be positive and dim must be an even integer >= 2")
    positions = np.arange(seq_len, dtype=np.float64)[:, None]
    pair_ids = np.arange(dim // 2, dtype=np.float64)[None, :]
    inv_freq = 1.0 / (10000.0 ** (2.0 * pair_ids / dim))
    return (positions * inv_freq).astype(np.float32)


def plot_unet_shape_flow(
    stages: list[TensorStage] | None = None,
    out_path: Path | None = None,
    font_scale: float = 1.0,
    skip_connections: Sequence[SkipConnectionSpec] | None = None,
    figsize: tuple[float, float] = (13, 4.4),
    stage_x_min: float = 0.06,
    stage_x_max: float = 0.94,
    stage_y: float = 0.5,
    stage_width: float = 0.1,
    min_stage_height: float = 0.18,
    max_stage_height: float = 0.36,
    stage_title_font_size: float = 9,
    stage_shape_font_size: float = 8,
    stage_role_font_size: float = 9,
    skip_label_font_size: float = 8,
    note_font_size: float = 8,
    stage_title_y_offset: float = 0.01,
    stage_role_y_offset: float = -0.02,
    note_xy: tuple[float, float] = (0.02, 0.1),
    note_text: str = "B=批量，C=通道，F=频率 bin，T=时间帧",
    encoder_color: str = "0.30",
    bottleneck_color: str = "0.50",
    decoder_color: str = "0.70",
    block_edge_color: str = "0.2",
    arrow_color: str = "0.25",
    arrow_linewidth: float = 1.1,
    arrow_mutation_scale: float = 10,
    skip_color: str = "0.35",
    skip_linewidth: float = 1.0,
) -> plt.Figure:
    """Draw a compact U-Net shape-flow diagram."""
    setup_plot_style()
    if stages is None:
        stages = unet_shape_flow()
    _validate_font_scale(font_scale)
    if skip_connections is None:
        skip_connections = (
            SkipConnectionSpec(1, 5, 0.21),
            SkipConnectionSpec(2, 4, 0.31),
        )
    _validate_skip_connections(skip_connections, len(stages))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    xs = np.linspace(stage_x_min, stage_x_max, len(stages))
    y = stage_y
    heights = _normalized_heights([stage.shape[2] for stage in stages], min_stage_height, max_stage_height)
    widths = [stage_width] * len(stages)

    for idx, (stage, x, height, width) in enumerate(zip(stages, xs, heights, widths)):
        color = encoder_color if idx < 3 else bottleneck_color if idx == 3 else decoder_color
        rect = Rectangle(
            (x - width / 2, y - height / 2),
            width,
            height,
            linewidth=1.4,
            edgecolor=block_edge_color,
            facecolor=color,
            alpha=0.8,
        )
        ax.add_patch(rect)
        b, c, f, t = stage.shape
        ax.text(
            x,
            y + height / 2 + stage_title_y_offset,
            stage.name,
            ha="center",
            va="bottom",
            fontsize=_font_size(stage_title_font_size, font_scale),
        )
        ax.text(
            x,
            y,
            f"B{b} C{c}\nF{f} T{t}",
            ha="center",
            va="center",
            fontsize=_font_size(stage_shape_font_size, font_scale),
            color=_text_color_for_gray(color),
        )
        ax.text(
            x,
            y - height / 2 + stage_role_y_offset,
            stage.role,
            ha="center",
            va="top",
            fontsize=_font_size(stage_role_font_size, font_scale),
        )
        if idx < len(stages) - 1:
            _arrow(
                ax,
                xs[idx] + width / 2,
                y,
                xs[idx + 1] - widths[idx + 1] / 2,
                y,
                color=arrow_color,
                linewidth=arrow_linewidth,
                mutation_scale=arrow_mutation_scale,
            )

    for spec in skip_connections:
        _skip_connection(
            ax,
            xs[spec.start_stage],
            xs[spec.end_stage],
            y + spec.y_offset,
            spec.label,
            curvature=spec.curvature,
            label_y_offset=spec.label_y_offset,
            font_size=_font_size(skip_label_font_size, font_scale),
            color=skip_color,
            linewidth=skip_linewidth,
        )
    if note_text:
        ax.text(
            note_xy[0],
            note_xy[1],
            note_text,
            ha="left",
            va="bottom",
            fontsize=_font_size(note_font_size, font_scale),
            color="0.25",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return _finish(fig, out_path)


def plot_band_split_layout(
    bands: list[FrequencyBand] | None = None,
    n_frames: int = 96,
    out_path: Path | None = None,
    font_scale: float = 1.0,
    figsize: tuple[float, float] = (12.2, 4.8),
    width_ratios: tuple[float, float] = (1.05, 1.25),
    title_font_size: float = 11,
    axis_label_font_size: float = 10,
    tick_label_font_size: float = 8,
    legend_font_size: float = 8,
    legend_title_font_size: float = 9,
    block_font_size: float = 8,
    left_title: str = "非均匀频带",
    right_title: str = "逐频带建模与跨带融合",
    legend_title: str = "频带范围",
    legend_loc: str = "upper left",
    legend_bbox_to_anchor: tuple[float, float] = (1.02, 1.0),
    band_alpha: float = 0.82,
    band_edge_color: str = "0.96",
    block_height: float = 0.09,
    band_block_xy: tuple[float, float] = (0.03, 0.24),
    local_block_xy: tuple[float, float] = (0.45, 0.24),
    merge_block_xywh: tuple[float, float, float, float] = (0.84, 0.41, 0.13, 0.18),
    right_y_top: float = 0.82,
    right_y_bottom: float = 0.18,
    arrow_color: str = "0.25",
    arrow_linewidth: float = 1.1,
    arrow_mutation_scale: float = 10,
    band_to_local_arrow_gap: float = 0.02,
    local_to_merge_arrow_start_gap: float = 0.02,
    local_to_merge_arrow_end_gap: float = 0.01,
) -> plt.Figure:
    """Draw a band-split processing layout."""
    setup_plot_style()
    if bands is None:
        bands = make_frequency_bands()
    _validate_font_scale(font_scale)
    n_bins = max(band.end_bin for band in bands)

    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": list(width_ratios)})
    ax_spec, ax_blocks = axes
    ax_spec.set_title(left_title, fontsize=_font_size(title_font_size, font_scale))
    ax_spec.set_xlabel("时间块", fontsize=_font_size(axis_label_font_size, font_scale))
    ax_spec.set_ylabel("频率 bin", fontsize=_font_size(axis_label_font_size, font_scale))
    ax_spec.set_xlim(0, n_frames)
    ax_spec.set_ylim(0, n_bins)
    ax_spec.tick_params(axis="both", labelsize=_font_size(tick_label_font_size, font_scale))
    ax_spec.grid(False)

    legend_handles = []
    for band in bands:
        rect = Rectangle(
            (0, band.start_bin),
            n_frames,
            band.end_bin - band.start_bin,
            facecolor=band.color,
            edgecolor=band_edge_color,
            alpha=band_alpha,
        )
        ax_spec.add_patch(rect)
        legend_handles.append(
            Patch(
                facecolor=band.color,
                edgecolor="0.35",
                label=f"{band.name}: {band.start_bin}-{band.end_bin} bin",
                alpha=band_alpha,
            )
        )
    ax_spec.legend(
        handles=legend_handles,
        title=legend_title,
        loc=legend_loc,
        bbox_to_anchor=legend_bbox_to_anchor,
        frameon=False,
        fontsize=_font_size(legend_font_size, font_scale),
        title_fontsize=_font_size(legend_title_font_size, font_scale),
    )

    ax_blocks.set_axis_off()
    ax_blocks.set_title(right_title, fontsize=_font_size(title_font_size, font_scale))
    y_positions = np.linspace(right_y_top, right_y_bottom, len(bands))
    band_x, band_width = band_block_xy
    local_x, local_width = local_block_xy
    for band, y in zip(bands, y_positions):
        ax_blocks.add_patch(
            Rectangle(
                (band_x, y - block_height / 2),
                band_width,
                block_height,
                facecolor=band.color,
                edgecolor="0.25",
                alpha=0.85,
            )
        )
        ax_blocks.text(
            band_x + band_width / 2,
            y,
            band.name,
            ha="center",
            va="center",
            color=_text_color_for_gray(band.color),
            fontsize=_font_size(block_font_size, font_scale),
        )
        _arrow(
            ax_blocks,
            band_x + band_width + band_to_local_arrow_gap,
            y,
            local_x - band_to_local_arrow_gap,
            y,
            color=arrow_color,
            linewidth=arrow_linewidth,
            mutation_scale=arrow_mutation_scale,
        )
        ax_blocks.add_patch(
            Rectangle((local_x, y - block_height / 2), local_width, block_height, facecolor="0.72", edgecolor="0.25")
        )
        ax_blocks.text(
            local_x + local_width / 2,
            y,
            "局部模型",
            ha="center",
            va="center",
            fontsize=_font_size(block_font_size, font_scale),
        )
        _arrow(
            ax_blocks,
            local_x + local_width + local_to_merge_arrow_start_gap,
            y,
            merge_block_xywh[0] - local_to_merge_arrow_end_gap,
            0.5,
            color=arrow_color,
            linewidth=arrow_linewidth,
            mutation_scale=arrow_mutation_scale,
        )

    merge_x, merge_y, merge_w, merge_h = merge_block_xywh
    ax_blocks.add_patch(Rectangle((merge_x, merge_y), merge_w, merge_h, facecolor="0.45", edgecolor="0.25"))
    ax_blocks.text(
        merge_x + merge_w / 2,
        merge_y + merge_h / 2,
        "融合掩蔽",
        ha="center",
        va="center",
        fontsize=_font_size(block_font_size, font_scale),
        color="white",
    )
    ax_blocks.set_xlim(0, 1)
    ax_blocks.set_ylim(0, 1)
    return _finish(fig, out_path)


def plot_roformer_attention_sketch(
    seq_len: int = 64,
    dim: int = 16,
    out_path: Path | None = None,
    font_scale: float = 1.0,
    figsize: tuple[float, float] = (12, 3.8),
    title_font_size: float = 11,
    axis_label_font_size: float = 10,
    legend_font_size: float = 7,
    node_font_size: float = 10,
    center_text_font_size: float = 8,
    line_width: float = 1.0,
    token_grid_shape: tuple[int, int] = (8, 12),
    token_grid_active_rows: tuple[int, int] = (2, 6),
    token_grid_active_cols: tuple[int, int] = (3, 9),
    token_inactive_face_color: str = "0.94",
    token_active_face_color: str = "0.62",
    token_edge_color: str = "0.35",
    token_linewidth: float = 0.7,
    token_tick_label_font_size: float = 8,
    rope_legend_loc: str = "upper left",
    rope_legend_bbox_to_anchor: tuple[float, float] = (1.02, 1.0),
    q_node_xy: tuple[float, float] = (0.18, 0.5),
    k_upper_xy: tuple[float, float] = (0.50, 0.72),
    k_lower_xy: tuple[float, float] = (0.50, 0.28),
    v_node_xy: tuple[float, float] = (0.82, 0.5),
    center_text_xy: tuple[float, float] = (0.5, 0.5),
    arrow_color: str = "0.25",
    arrow_linewidth: float = 1.1,
    arrow_mutation_scale: float = 10,
) -> plt.Figure:
    """Sketch rotary position encoding and attention over band tokens."""
    setup_plot_style()
    _validate_font_scale(font_scale)
    angles = rotary_embedding_angles(seq_len=seq_len, dim=dim)
    positions = np.arange(seq_len)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].set_title("频带-时间 token", fontsize=_font_size(title_font_size, font_scale))
    row_start, row_end = token_grid_active_rows
    col_start, col_end = token_grid_active_cols
    n_bands, n_blocks = token_grid_shape
    if not (0 <= row_start < row_end <= n_bands):
        raise ValueError("token_grid_active_rows must fall inside token_grid_shape")
    if not (0 <= col_start < col_end <= n_blocks):
        raise ValueError("token_grid_active_cols must fall inside token_grid_shape")
    for band_idx in range(n_bands):
        for block_idx in range(n_blocks):
            is_active = row_start <= band_idx < row_end and col_start <= block_idx < col_end
            axes[0].add_patch(
                Rectangle(
                    (block_idx, band_idx),
                    1,
                    1,
                    facecolor=token_active_face_color if is_active else token_inactive_face_color,
                    edgecolor=token_edge_color,
                    linewidth=token_linewidth,
                )
            )
    axes[0].set_xlim(0, n_blocks)
    axes[0].set_ylim(0, n_bands)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xticks(np.arange(0.5, n_blocks, 1.0))
    axes[0].set_yticks(np.arange(0.5, n_bands, 1.0))
    axes[0].set_xticklabels([str(idx) for idx in range(n_blocks)], fontsize=_font_size(token_tick_label_font_size, font_scale))
    axes[0].set_yticklabels([str(idx) for idx in range(n_bands)], fontsize=_font_size(token_tick_label_font_size, font_scale))
    axes[0].set_xlabel("时间块", fontsize=_font_size(axis_label_font_size, font_scale))
    axes[0].set_ylabel("频带", fontsize=_font_size(axis_label_font_size, font_scale))

    axes[1].set_title("RoPE 角度对", fontsize=_font_size(title_font_size, font_scale))
    line_styles = ("-", "--", "-.", ":")
    for pair in range(min(4, angles.shape[1])):
        axes[1].plot(
            positions,
            np.sin(angles[:, pair]),
            linewidth=line_width,
            color=LINE_GRAYS[pair % len(LINE_GRAYS)],
            linestyle=line_styles[pair % len(line_styles)],
            label=f"维度对 {pair}",
        )
    axes[1].set_xlabel("token 位置", fontsize=_font_size(axis_label_font_size, font_scale))
    axes[1].set_ylabel("sin(角度)", fontsize=_font_size(axis_label_font_size, font_scale))
    axes[1].legend(
        fontsize=_font_size(legend_font_size, font_scale),
        loc=rope_legend_loc,
        bbox_to_anchor=rope_legend_bbox_to_anchor,
        frameon=False,
    )

    axes[2].set_axis_off()
    axes[2].set_title("旋转 Q/K 注意力", fontsize=_font_size(title_font_size, font_scale))
    _attention_node(axes[2], q_node_xy[0], q_node_xy[1], "Q", font_size=_font_size(node_font_size, font_scale))
    _attention_node(axes[2], k_upper_xy[0], k_upper_xy[1], "K", font_size=_font_size(node_font_size, font_scale))
    _attention_node(axes[2], k_lower_xy[0], k_lower_xy[1], "K", font_size=_font_size(node_font_size, font_scale))
    _attention_node(axes[2], v_node_xy[0], v_node_xy[1], "V", font_size=_font_size(node_font_size, font_scale))
    _arrow(
        axes[2],
        q_node_xy[0] + 0.09,
        q_node_xy[1] + 0.02,
        k_upper_xy[0] - 0.09,
        k_upper_xy[1] - 0.03,
        color=arrow_color,
        linewidth=arrow_linewidth,
        mutation_scale=arrow_mutation_scale,
    )
    _arrow(
        axes[2],
        q_node_xy[0] + 0.09,
        q_node_xy[1] - 0.02,
        k_lower_xy[0] - 0.09,
        k_lower_xy[1] + 0.03,
        color=arrow_color,
        linewidth=arrow_linewidth,
        mutation_scale=arrow_mutation_scale,
    )
    _arrow(
        axes[2],
        k_upper_xy[0] + 0.09,
        k_upper_xy[1],
        v_node_xy[0] - 0.08,
        v_node_xy[1] + 0.04,
        color=arrow_color,
        linewidth=arrow_linewidth,
        mutation_scale=arrow_mutation_scale,
    )
    _arrow(
        axes[2],
        k_lower_xy[0] + 0.09,
        k_lower_xy[1],
        v_node_xy[0] - 0.08,
        v_node_xy[1] - 0.04,
        color=arrow_color,
        linewidth=arrow_linewidth,
        mutation_scale=arrow_mutation_scale,
    )
    axes[2].text(
        center_text_xy[0],
        center_text_xy[1],
        "相对\n位置信息",
        ha="center",
        va="center",
        fontsize=_font_size(center_text_font_size, font_scale),
    )
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    return _finish(fig, out_path)


def _normalized_heights(
    values: list[int],
    min_value: float = 0.18,
    max_value: float = 0.36,
) -> list[float]:
    values_arr = np.asarray(values, dtype=np.float64)
    if np.all(values_arr == values_arr[0]):
        return [max_value] * len(values)
    scaled = (values_arr - values_arr.min()) / (values_arr.max() - values_arr.min())
    return (min_value + scaled * (max_value - min_value)).tolist()


def _arrow(
    ax: plt.Axes,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str = "0.25",
    linewidth: float = 1.1,
    mutation_scale: float = 10,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            color=color,
        )
    )


def _skip_connection(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    label: str,
    curvature: float = -0.15,
    label_y_offset: float = 0.03,
    font_size: float = 8.0,
    color: str = "0.35",
    linewidth: float = 1.0,
) -> None:
    con = ConnectionPatch(
        (x1, y),
        (x2, y),
        "data",
        "data",
        arrowstyle="->",
        connectionstyle=f"arc3,rad={curvature}",
        linewidth=linewidth,
        color=color,
    )
    ax.add_artist(con)
    ax.text((x1 + x2) / 2, y + label_y_offset, label, ha="center", va="bottom", fontsize=font_size, color="0.25")


def _attention_node(ax: plt.Axes, x: float, y: float, label: str, font_size: float = 10.0) -> None:
    ax.scatter([x], [y], s=520, color="0.32", edgecolor="0.2", zorder=3)
    ax.text(x, y, label, ha="center", va="center", color="white", fontsize=font_size, fontweight="bold")


def _text_color_for_gray(color: str) -> str:
    try:
        value = float(color)
    except ValueError:
        return "0.10"
    return "white" if value < 0.58 else "0.10"


def _font_size(base_size: float, font_scale: float) -> float:
    return base_size * font_scale


def _validate_font_scale(font_scale: float) -> None:
    if font_scale <= 0:
        raise ValueError("font_scale must be positive")


def _validate_skip_connections(skip_connections: Sequence[SkipConnectionSpec], n_stages: int) -> None:
    for spec in skip_connections:
        if not 0 <= spec.start_stage < n_stages:
            raise ValueError("skip connection start_stage is out of range")
        if not 0 <= spec.end_stage < n_stages:
            raise ValueError("skip connection end_stage is out of range")
        if spec.start_stage >= spec.end_stage:
            raise ValueError("skip connection start_stage must be smaller than end_stage")


def _finish(fig: plt.Figure, out_path: Path | None) -> plt.Figure:
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", dpi=FIGURE_SAVE_DPI)
    return fig
