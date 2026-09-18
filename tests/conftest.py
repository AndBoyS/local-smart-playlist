"""Shared fixtures."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest


def _sine(duration_s: float, sr: int, freq: float, amplitude: float) -> np.ndarray:
    t = np.arange(int(sr * duration_s)) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture
def sine() -> Callable[..., np.ndarray]:
    """A sine wave generator: sine(duration_s=12, sr=48000, freq=440, amplitude=0.3)."""

    def make(*, duration_s: float = 12.0, sr: int = 48_000, freq: float = 440.0, amplitude: float = 0.3) -> np.ndarray:
        return _sine(duration_s, sr, freq, amplitude)

    return make


@pytest.fixture
def wav(tmp_path: Any) -> Callable[..., Any]:
    """Write a wav file: wav("a.wav", samples, sr=48000) -> Path."""
    import soundfile as sf

    def write(*, name: str, samples: np.ndarray, sr: int = 48_000) -> Any:
        path = tmp_path / name
        _ = sf.write(str(path), samples, sr)  # pyrefly: ignore[unknown-argument-type]
        return path

    return write
