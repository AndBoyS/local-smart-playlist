"""Real MuQ-MuLan model sanity tests (marked `model`; run with `uv run pytest -m model`)."""

from typing import Any, cast

import numpy as np
import pytest

from local_smart_playlist.audio.decode import TARGET_SR

pytestmark = pytest.mark.model


def test_model_load() -> None:
    from local_smart_playlist.embed.model import EMBED_DIM, load_model

    embedder = load_model()
    _ = embedder.model.eval()
    assert embedder.dim == EMBED_DIM


def test_text_embedding() -> None:
    from local_smart_playlist.embed.model import EMBED_DIM, load_model

    embedder = load_model()
    vecs = embedder.embed_texts(["a melancholic song", "an upbeat party song"])
    assert vecs.shape == (2, EMBED_DIM)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    # Different moods should not collapse to the same vector.
    dot = float(cast("Any", vecs[0] @ vecs[1]))
    assert dot < 0.9


def test_audio_embedding(sine: object) -> None:
    from local_smart_playlist.embed.model import EMBED_DIM, load_model

    t = np.arange(TARGET_SR * 10) / TARGET_SR
    window = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    embedder = load_model()
    vecs = embedder.embed_windows([window])
    assert vecs.shape == (1, EMBED_DIM)
    assert np.linalg.norm(vecs[0]) == pytest.approx(1.0, abs=1e-5)


def test_audio_text_similarity(sine: object) -> None:
    """Cross-modal sanity: matching caption outscores a mismatched caption."""
    from local_smart_playlist.embed.model import load_model

    t = np.arange(TARGET_SR * 10) / TARGET_SR
    window = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    embedder = load_model()
    audio_vec = embedder.embed_windows([window])[0]
    texts = ["a steady electronic sine tone", "a gritty distorted rock guitar riff"]
    text_vecs = embedder.embed_texts(texts)
    sims = text_vecs @ audio_vec
    assert sims.shape == (2,)
    assert float(sims[0]) > float(sims[1])
