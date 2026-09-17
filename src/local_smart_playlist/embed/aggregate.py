"""Track-level aggregation of window vectors."""


import numpy as np


def _unit(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def mean_vector(window_vecs: np.ndarray) -> np.ndarray:
    """L2-normalized mean over window vectors. Requires >= 1 window."""
    if len(window_vecs) == 0:
        msg = "cannot aggregate zero windows"
        raise ValueError(msg)
    return _unit(window_vecs.mean(axis=0))


def p90_vector(window_vecs: np.ndarray) -> np.ndarray:
    """Per-dim 90th percentile over window vectors, L2-normalized."""
    if len(window_vecs) == 0:
        msg = "cannot aggregate zero windows"
        raise ValueError(msg)
    return _unit(np.percentile(window_vecs, 90, axis=0))


def aggregate(window_vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(mean_vec, p90_vec), both unit-norm."""
    return mean_vector(window_vecs), p90_vector(window_vecs)
