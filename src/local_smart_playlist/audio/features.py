"""Batch assembly for CLAP feature extraction."""


from collections.abc import Iterator

import numpy as np

BATCH_WINDOWS = 16


def batches(windows: list[np.ndarray], batch_size: int = BATCH_WINDOWS) -> Iterator[list[np.ndarray]]:
    for i in range(0, len(windows), batch_size):
        yield windows[i : i + batch_size]


def stack_batch(batch: list[np.ndarray]) -> np.ndarray:
    """Stack equal-length windows into a (batch, window_len) float32 array."""
    if len(batch) == 0:
        return np.empty((0, 0), dtype=np.float32)
    length = len(batch[0])
    if any(len(w) != length for w in batch):
        msg = "windows in a batch must have identical length"
        raise ValueError(msg)
    return np.stack(batch).astype(np.float32)
