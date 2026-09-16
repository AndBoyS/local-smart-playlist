"""Real CLAP model sanity tests (marked `model`; run with `uv run pytest -m model`)."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from local_smart_playlist.audio.decode import TARGET_SR

pytestmark = pytest.mark.model


def test_model_load_and_dims() -> None:
    from local_smart_playlist.embed.model import EMBED_DIM, load_model

    model, _ = load_model()
    assert model.config.projection_dim == EMBED_DIM


def test_text_embedding() -> None:
    from local_smart_playlist.embed.model import EMBED_DIM, embed_texts

    vecs = embed_texts(["a melancholic song", "an upbeat party song"])
    assert vecs.shape == (2, EMBED_DIM)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    # Different moods should not collapse to the same vector.
    dot = float(cast("Any", vecs[0] @ vecs[1]))
    assert 0.5 < dot < 0.99


def test_audio_embedding(sine: object) -> None:
    from local_smart_playlist.embed.model import EMBED_DIM, embed_windows

    t = np.arange(TARGET_SR * 10) / TARGET_SR
    window = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    vecs = embed_windows([window])
    assert vecs.shape == (1, EMBED_DIM)
    assert np.linalg.norm(vecs[0]) == pytest.approx(1.0, abs=1e-5)
