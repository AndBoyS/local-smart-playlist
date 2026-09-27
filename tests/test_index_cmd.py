"""`sp index` robustness tests (no model — embed_windows monkeypatched)."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from local_smart_playlist.commands import index_cmd
from local_smart_playlist.index.store import Store
from local_smart_playlist.type_utils import NonEmptyTuple

DIM = 512  # Store's default embed_dim; fake embedder must match


class _FakeEmbedder:
    """MuLanEmbedder stand-in: embed_windows -> ones/DIM rows."""

    def embed_windows(self, windows: NonEmptyTuple[np.ndarray], *, batch_size: int = 16) -> np.ndarray:
        return np.ones((len(windows), DIM), dtype=np.float32) / DIM



@pytest.fixture
def two_wavs(tmp_path: Any, wav: Any, sine: Any) -> None:
    _ = wav(name="good.wav", samples=sine(duration_s=12.0))
    _ = wav(name="bad.wav", samples=sine(duration_s=12.0))


def test_index_survives_non_decode_failure(
    tmp_path: Path, wav: Any, sine: Any, monkeypatch: Any
) -> None:
    """A runtime error mid-track is recorded as a failure; other tracks still index."""
    _ = wav(name="good.wav", samples=sine(duration_s=12.0))
    _ = wav(name="bad.wav", samples=sine(duration_s=12.0))

    real_decode = index_cmd.decode.decode_mono

    def fake_decode(path: Path) -> tuple[np.ndarray, float]:
        if path.name == "bad.wav":
            msg = "simulated embed-time failure"
            raise RuntimeError(msg)
        return real_decode(path)

    monkeypatch.setattr(index_cmd.decode, "decode_mono", fake_decode)
    monkeypatch.setattr(index_cmd, "load_model", _FakeEmbedder)
    monkeypatch.setattr(index_cmd, "cap_torch_threads", lambda: None)

    db = tmp_path / "idx.db"
    index_cmd.IndexArgs.run(root=tmp_path, db=db, rescan=False, quiet=True)

    with Store(db) as store:
        assert store.track_count() == 1
        assert store.failure_count() == 1
        failures = store.list_failures()
        assert failures[0][0] == "bad.wav"
        assert "simulated embed-time failure" in failures[0][1]


def test_index_load_model_fails_fast(
    tmp_path: Path, wav: Any, sine: Any, monkeypatch: Any
) -> None:
    """Model-loading failure aborts the run before any track is recorded."""

    def boom() -> None:
        msg = "no model, no service"
        raise RuntimeError(msg)

    monkeypatch.setattr(index_cmd, "load_model", boom)
    monkeypatch.setattr(index_cmd, "cap_torch_threads", lambda: None)
    _ = wav(name="good.wav", samples=sine(duration_s=12.0))

    db = tmp_path / "idx.db"
    with pytest.raises(RuntimeError, match="no model"):
        index_cmd.IndexArgs.run(root=tmp_path, db=db, rescan=False, quiet=True)

    with Store(db) as store:
        assert store.track_count() == 0
        assert store.failure_count() == 0
