"""Classic source-separation algorithms for Chapter 7."""

from .hpss import compute_hpss_masks, hpss_separate, median_filter_spectrogram
from .nmf import (
    component_frequency_centroids,
    component_masks,
    nmf_decompose,
    reconstruct_components,
)
from .repet import (
    estimate_period_from_similarity,
    estimate_repeating_background,
    repet_masks,
    repet_separate,
    self_similarity_matrix,
)

__all__ = [
    "component_frequency_centroids",
    "component_masks",
    "compute_hpss_masks",
    "estimate_period_from_similarity",
    "estimate_repeating_background",
    "hpss_separate",
    "median_filter_spectrogram",
    "nmf_decompose",
    "reconstruct_components",
    "repet_masks",
    "repet_separate",
    "self_similarity_matrix",
]
