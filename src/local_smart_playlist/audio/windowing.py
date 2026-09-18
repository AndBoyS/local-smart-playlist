"""Silence trimming and disjoint windowing."""

from __future__ import annotations

import numpy as np
from shape_extensions import IntVar

from local_smart_playlist.audio.decode import TARGET_SR

S = IntVar("S")  # input sample count
F = IntVar("F")  # frame count
W = IntVar("W")  # window length in samples

WINDOW_SECONDS = 10.0
FRAME_SECONDS = 0.1
# Frames with RMS below this are treated as silence. Relative to full scale
# (float32 in [-1, 1]); roughly -50 dBFS.
TRIM_DB = -50.0
MIN_WINDOW_SECONDS = 1.0


def frame_rms(
    samples: np.ndarray[[S]], *, sr: int = TARGET_SR, frame_seconds: float = FRAME_SECONDS
) -> np.ndarray[[F]]:
    """Per-frame RMS over non-overlapping frames of *frame_seconds*."""
    frame_len = int(frame_seconds * sr)
    n_frames = len(samples) // frame_len
    if n_frames == 0:
        return np.array([], dtype=np.float32)
    # ndarray.reshape is `Any` in the shape stubs; pin the shape at the assignment.
    frames: np.ndarray[[F, int]] = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
    squares = frames.astype(np.float64) ** 2
    return np.sqrt(squares.mean(axis=1)).astype(np.float32)


def trim_silence(samples: np.ndarray[[S]], *, sr: int = TARGET_SR, threshold_db: float = TRIM_DB) -> np.ndarray[[S]]:
    """Trim leading/trailing frames whose RMS falls below *threshold_db*."""
    rms = frame_rms(samples, sr=sr)
    if rms.size == 0:
        return np.array([], dtype=np.float32)
    threshold = 10.0 ** (threshold_db / 20.0)
    # ndarray comparison dunders are not in the shape stubs; pin the shape here.
    loud: np.ndarray[[int]] = rms >= threshold
    if not loud.any().item():  # .any() returns 0-d ndarray; stubs lack implicit-bool
        # Pure silence / too quiet: keep everything so the track still indexes
        # with whatever content exists.
        return samples
    frame_len = int(FRAME_SECONDS * sr)
    # np.argmax (module form) is untracked; the method form returns a tracked 0-d array.
    start = int(loud.argmax()) * frame_len
    end = (len(loud) - int(loud[::-1].argmax())) * frame_len
    return samples[start:end]


def slice_windows(
    samples: np.ndarray[[S]], *, sr: int = TARGET_SR, window_seconds: float = WINDOW_SECONDS
) -> list[np.ndarray[[W]]]:
    """Trim silence, then split into disjoint *window_seconds* windows."""
    trimmed = trim_silence(samples, sr=sr)
    window_len = int(window_seconds * sr)
    if len(trimmed) < max(window_len, int(MIN_WINDOW_SECONDS * sr)):
        return []
    n_windows = (len(trimmed) - window_len) // window_len + 1
    return [trimmed[i * window_len : (i + 1) * window_len] for i in range(n_windows)]
