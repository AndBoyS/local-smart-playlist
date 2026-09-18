"""Windowing + trim tests (no torch)."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf

from local_smart_playlist.audio.decode import TARGET_SR, decode_mono
from local_smart_playlist.audio.windowing import (
    MIN_WINDOW_SECONDS,
    WINDOW_SECONDS,
    frame_rms,
    slice_windows,
    trim_silence,
)

WIN_LEN = int(WINDOW_SECONDS * TARGET_SR)


def test_frame_rms_shape(sine: Callable[..., np.ndarray]) -> None:
    samples = sine(duration_s=5)
    rms = frame_rms(samples)
    assert rms.shape == (50,)


def test_trim_removes_silent_edges(sine: Callable[..., np.ndarray]) -> None:
    pad = np.zeros(2 * TARGET_SR, dtype=np.float32)
    samples = np.concatenate([pad, sine(duration_s=12), pad])
    trimmed = trim_silence(samples)
    assert len(trimmed) == 12 * TARGET_SR


def test_trim_all_silent_keeps_input() -> None:
    samples = np.zeros(4 * TARGET_SR, dtype=np.float32)
    assert len(trim_silence(samples)) == len(samples)


def test_windows_disjoint_10s(sine: Callable[..., np.ndarray]) -> None:
    samples = np.concatenate([np.zeros(TARGET_SR, dtype=np.float32), sine(duration_s=21)])
    windows = slice_windows(samples)
    assert len(windows) == 2
    for w in windows:
        assert len(w) == WIN_LEN
    trimmed = trim_silence(samples)
    assert windows[0].tobytes() == trimmed[0:WIN_LEN].tobytes()
    assert windows[1].tobytes() == trimmed[WIN_LEN : 2 * WIN_LEN].tobytes()


def test_residue_dropped(sine: Callable[..., np.ndarray]) -> None:
    samples = sine(duration_s=14)
    windows = slice_windows(samples)
    assert len(windows) == 1  # 14 s -> one full window, 4 s residue dropped


def test_short_or_silent_gives_no_windows(wav: Callable[..., Path], sine: Callable[..., np.ndarray]) -> None:
    short = decode_mono(wav(name="short.wav", samples=sine(duration_s=0.5)))[0]
    silent = decode_mono(wav(name="silent.wav", samples=np.zeros(3 * TARGET_SR, dtype=np.float32)))[0]
    assert slice_windows(short) == []
    assert slice_windows(silent) == []


def test_decode_resamples_to_48k(tmp_path: Path, sine: Callable[..., np.ndarray]) -> None:
    samples = sine(duration_s=4, sr=22_050)
    path = tmp_path / "resample.wav"
    sf.write(path, samples, 22_050)
    decoded, duration = decode_mono(path)
    assert abs(duration - 4.0) < 0.05
    assert len(decoded) == int(4.0 * TARGET_SR)


def test_decode_unsupported_ext(tmp_path: Path) -> None:
    from local_smart_playlist.audio.decode import DecodeError

    path = tmp_path / "nope.xyz"
    _ = path.write_bytes(b"")
    try:
        _ = decode_mono(path)
    except DecodeError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("expected DecodeError")


def test_min_window_constant() -> None:
    assert MIN_WINDOW_SECONDS < WINDOW_SECONDS
