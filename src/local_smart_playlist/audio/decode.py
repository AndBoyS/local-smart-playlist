"""Decode audio files to mono float32 at 48 kHz."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from shape_extensions import IntVar

N = IntVar("N")

TARGET_SR = 48_000

# Formats soundfile (libsndfile) can decode natively.
SOUNDFILE_EXTS = frozenset({".wav", ".flac", ".ogg", ".oga", ".opus", ".mp3", ".aiff", ".aif"})
# Formats handled by the torchaudio/ffmpeg fallback (AAC/M4A etc.).
TORCHAUDIO_EXTS = frozenset({".m4a", ".mp4", ".aac", ".wma", ".alac"})

AUDIO_EXTS = SOUNDFILE_EXTS | TORCHAUDIO_EXTS


class DecodeError(Exception):
    """Raised when a file cannot be decoded."""


def supports(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTS


def decode_mono(path: Path) -> tuple[np.ndarray[[N]], float]:
    """Decode *path* to mono float32 at TARGET_SR.

    Returns (samples, duration_seconds). Uses soundfile when the format is
    supported, otherwise the torchaudio/ffmpeg backend (m4a/aac).
    """
    ext = path.suffix.lower()
    try:
        if ext in SOUNDFILE_EXTS:
            samples, sr = sf.read(path, dtype="float32", always_2d=True)
        elif ext in TORCHAUDIO_EXTS:
            samples, sr = _decode_torchaudio(path)
        else:
            msg = f"unsupported extension {ext!r}"
            raise DecodeError(msg)
    except (sf.LibsndfileError, RuntimeError, OSError) as first_exc:
        # libsndfile/MPG123 chokes on some valid MP3s (ID3 comment quirks);
        # retry through the ffmpeg CLI before giving up.
        try:
            samples, sr = _decode_ffmpeg_cli(path)
        except DecodeError:
            msg = f"decode failed: {first_exc}"
            raise DecodeError(msg) from first_exc

    mono = samples.mean(axis=1).astype(np.float32)
    if sr != TARGET_SR:
        mono = soxr.resample(mono, sr, TARGET_SR).astype(np.float32)
    duration = len(mono) / TARGET_SR
    return mono, duration


def _decode_torchaudio(path: Path) -> tuple[np.ndarray, int]:
    """Decode via torchaudio (torchcodec backend), falling back to the ffmpeg CLI."""
    try:
        import torchaudio

        tensor, sr = torchaudio.load(path)
        arr = tensor.numpy()
        return arr.T.astype(np.float32), sr
    except Exception:  # noqa: BLE001 — any torchaudio failure falls through to CLI
        return _decode_ffmpeg_cli(path)


def _decode_ffmpeg_cli(path: Path) -> tuple[np.ndarray, int]:
    """Decode to mono f32le via the ffmpeg binary (last resort)."""
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "quiet",
                "-i",
                str(path),
                "-f",
                "f32le",
                "-ac",
                "1",
                "-ar",
                str(TARGET_SR),
                "-",
            ],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        msg = "ffmpeg fallback failed"
        raise DecodeError(msg) from exc

    data = np.frombuffer(proc.stdout, dtype=np.float32)
    if data.size == 0:
        msg = "ffmpeg produced no samples"
        raise DecodeError(msg)
    return data.reshape(-1, 1), TARGET_SR
