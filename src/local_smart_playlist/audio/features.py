"""Batch assembly for MuQ-MuLan feature extraction."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from shape_extensions import IntVar

from local_smart_playlist.type_utils import NonEmptyTuple

N = IntVar("N")
W = IntVar("W")

BATCH_WINDOWS = 16


def batches[T](windows: NonEmptyTuple[T], batch_size: int = BATCH_WINDOWS) -> Iterator[NonEmptyTuple[T]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for i in range(0, len(windows), batch_size):
        yield NonEmptyTuple(windows[i : i + batch_size])


def stack_batch(batch: Sequence[np.ndarray[[W]]]) -> np.ndarray[[N, W]]:
    """Stack equal-length windows into a (batch, window_len) float32 array."""
    if len(batch) == 0:
        msg = "cannot stack zero windows"
        raise ValueError(msg)
    length = len(batch[0])
    if any(len(w) != length for w in batch):
        msg = "windows in a batch must have identical length"
        raise ValueError(msg)
    return np.stack(batch).astype(np.float32)
