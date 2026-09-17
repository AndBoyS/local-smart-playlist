"""Batch assembly for MuQ-MuLan feature extraction."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from shape_extensions import IntVar

N = IntVar("N")
W = IntVar("W")

BATCH_WINDOWS = 16


def batches(windows: list[np.ndarray[[W]]], batch_size: int = BATCH_WINDOWS) -> Iterator[list[np.ndarray[[W]]]]:
    for i in range(0, len(windows), batch_size):
        yield windows[i : i + batch_size]


def stack_batch(batch: list[np.ndarray[[W]]]) -> np.ndarray[[N, W]]:
    """Stack equal-length windows into a (batch, window_len) float32 array."""
    if len(batch) == 0:
        msg = "cannot stack zero windows"
        raise ValueError(msg)
    length = len(batch[0])
    if any(len(w) != length for w in batch):
        msg = "windows in a batch must have identical length"
        raise ValueError(msg)
    return np.stack(batch).astype(np.float32)
