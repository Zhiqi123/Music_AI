"""NMF helpers for magnitude-spectrogram source-separation demos."""
from __future__ import annotations

import numpy as np


def nmf_decompose(
    mag: np.ndarray,
    n_components: int = 8,
    random_state: int = 0,
    max_iter: int = 800,
) -> tuple[np.ndarray, np.ndarray]:
    """Factorize ``mag`` into nonnegative basis spectra ``W`` and activations ``H``."""
    try:
        from sklearn.decomposition import NMF
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError("nmf_decompose requires scikit-learn") from exc

    mag = _validate_magnitude(mag)
    if n_components < 1:
        raise ValueError("n_components must be positive")
    n_components = min(int(n_components), min(mag.shape))

    model = NMF(
        n_components=n_components,
        init="nndsvda",
        solver="cd",
        beta_loss="frobenius",
        max_iter=max_iter,
        random_state=random_state,
        tol=1e-4,
    )
    W = model.fit_transform(mag)
    H = model.components_
    return W.astype(np.float32), H.astype(np.float32)


def reconstruct_components(W: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Return one magnitude estimate per NMF component."""
    W, H = _validate_factors(W, H)
    return (W.T[:, :, None] * H[:, None, :]).astype(np.float32)


def component_masks(
    W: np.ndarray,
    H: np.ndarray,
    groups: dict[str, list[int]],
    eps: float = 1e-10,
) -> dict[str, np.ndarray]:
    """Convert component groups into soft masks with the same shape as ``V``."""
    components = reconstruct_components(W, H)
    total = np.sum(components, axis=0)
    masks: dict[str, np.ndarray] = {}
    for name, indices in groups.items():
        if not indices:
            raise ValueError(f"group {name!r} must contain at least one component")
        checked = _check_component_indices(indices, components.shape[0])
        estimate = np.sum(components[checked], axis=0)
        masks[name] = np.clip(estimate / (total + eps), 0.0, 1.0).astype(np.float32)
    return masks


def component_frequency_centroids(
    W: np.ndarray,
    sr: int,
    n_fft: int,
    eps: float = 1e-10,
) -> np.ndarray:
    """Estimate each component's spectral centroid in Hz from ``W``."""
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError("W must be 2-D")
    freqs = np.linspace(0.0, sr / 2, W.shape[0])
    weights = np.maximum(W, 0.0)
    return (weights.T @ freqs / (np.sum(weights, axis=0) + eps)).astype(np.float32)


def _validate_magnitude(mag: np.ndarray) -> np.ndarray:
    mag = np.asarray(mag, dtype=np.float64)
    if mag.ndim != 2:
        raise ValueError("mag must be a 2-D magnitude spectrogram")
    if np.any(mag < 0):
        raise ValueError("mag must be nonnegative")
    if not np.any(mag > 0):
        raise ValueError("mag must contain nonzero energy")
    return mag


def _validate_factors(W: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    W = np.asarray(W, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    if W.ndim != 2 or H.ndim != 2:
        raise ValueError("W and H must be 2-D")
    if W.shape[1] != H.shape[0]:
        raise ValueError("W columns must match H rows")
    if np.any(W < 0) or np.any(H < 0):
        raise ValueError("NMF factors must be nonnegative")
    return W, H


def _check_component_indices(indices: list[int], n_components: int) -> list[int]:
    checked = [int(i) for i in indices]
    invalid = [i for i in checked if i < 0 or i >= n_components]
    if invalid:
        raise ValueError(f"component index out of range: {invalid}")
    return checked
