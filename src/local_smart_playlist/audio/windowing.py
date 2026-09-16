"""Silence trimming and disjoint windowing."""

from __future__ import annotations

import numpy as np

from local_smart_playlist.audio.decode import TARGET_SR

WINDOW_SECONDS = 10.0
FRAME_SECONDS = 0.1
# Frames with RMS below this are treated as silence. Relative to full scale
# (float32 in [-1, 1]); roughly -50 dBFS.
TRIM_DB = -50.0
MIN_WINDOW_SECONDS = 1.0


def frame_rms(samples: np.ndarray, *, sr: int = TARGET_SR, frame_seconds: float = FRAME_SECONDS) -> np.ndarray:
    """Per-frame RMS over non-overlapping frames of *frame_seconds*."""
    frame_len = int(frame_seconds * sr)
    n_frames = len(samples) // frame_len
    if n_frames == 0:
        return np.array([], dtype=np.float32)
    frames = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)).astype(np.float32)


def trim_silence(samples: np.ndarray, *, sr: int = TARGET_SR, threshold_db: float = TRIM_DB) -> np.ndarray:
    """Trim leading/trailing frames whose RMS falls below *threshold_db*."""
    rms = frame_rms(samples, sr=sr)
    if rms.size == 0:
        return np.array([], dtype=np.float32)
    threshold = 10.0 ** (threshold_db / 20.0)
    loud = rms >= threshold
    if not loud.any():
        # Pure silence / too quiet: keep everything so the track still indexes
        # with whatever content exists.
        return samples
    frame_len = int(FRAME_SECONDS * sr)
    start = int(np.argmax(loud)) * frame_len  # pyrefly: ignore[unknown-argument-type]
    end = (int(len(loud) - np.argmax(loud[::-1]))) * frame_len  # pyrefly: ignore[unknown-argument-type]
    return samples[start:end]


def slice_windows(samples: np.ndarray, *, sr: int = TARGET_SR, window_seconds: float = WINDOW_SECONDS) -> list[np.ndarray]:
    """Trim silence, then split into disjoint *window_seconds* windows."""
    trimmed = trim_silence(samples, sr=sr)
    window_len = int(window_seconds * sr)
    if len(trimmed) < max(window_len, int(MIN_WINDOW_SECONDS * sr)):
        return []
    n_windows = (len(trimmed) - window_len) // window_len + 1
    return [trimmed[i * window_len : (i + 1) * window_len] for i in range(n_windows)]
