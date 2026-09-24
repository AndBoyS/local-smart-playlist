"""Shared numpy helpers.

Pure numpy — no torch/transformers imports, so everything downstream can use
these without pulling in the model.
"""

import numpy as np
from shape_extensions import IntTuple, IntVar

N = IntVar("N")  # generic row count
D = IntVar("D")  # generic column count


def reshape[Shape: IntTuple](a: np.ndarray, shape: Shape) -> np.ndarray[Shape]:
    """Reshape with the result shape tracked in the type system. Temporary until pyrefly implements"""
    out: np.ndarray[Shape] = a.reshape(shape)
    return out


def l2_normalize(vectors: np.ndarray[[N, D]]) -> np.ndarray[[N, D]]:
    """Row-wise L2 normalization; zero rows stay zero (safe divide by 1)."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)  # pyrefly: ignore[unknown-argument-type]
    return vectors / norms
