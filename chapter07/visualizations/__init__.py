"""Architecture visualization helpers for Chapter 7."""

from .architecture_diagrams import (
    FrequencyBand,
    SkipConnectionSpec,
    TensorStage,
    make_frequency_bands,
    plot_band_split_layout,
    plot_roformer_attention_sketch,
    plot_unet_shape_flow,
    rotary_embedding_angles,
    unet_shape_flow,
)

__all__ = [
    "FrequencyBand",
    "SkipConnectionSpec",
    "TensorStage",
    "make_frequency_bands",
    "plot_band_split_layout",
    "plot_roformer_attention_sketch",
    "plot_unet_shape_flow",
    "rotary_embedding_angles",
    "unet_shape_flow",
]
