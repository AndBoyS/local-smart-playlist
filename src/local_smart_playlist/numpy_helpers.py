"""Shared numpy helpers; pyrefly shape-stub ignores are centralized here.

Pure numpy — no torch/transformers imports, so everything downstream can use
these without pulling in the model.
"""

import numpy as np
from shape_extensions import IntVar

N = IntVar("N")  # generic row count
D = IntVar("D")  # generic column count


def gt(
    values: np.ndarray[[N, D]],
    other: float | np.ndarray[[N, D]] | np.ndarray[[N, 1]],
) -> np.ndarray[[N, D]]:
    """Elementwise strict ``>`` with broadcasting (values vs scalar or per-row).

    Needed until pyrefly shape typing is fixed
    """
    # pyrefly: ignore [bad-assignment, unsupported-operation]
    mask: np.ndarray[[N, D]] = values > other
    return mask


def l2_normalize(vectors: np.ndarray[[N, D]]) -> np.ndarray[[N, D]]:
    """Row-wise L2 normalization; zero rows stay zero (safe divide by 1)."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)  # pyrefly: ignore[unknown-argument-type]
    return vectors / norms
