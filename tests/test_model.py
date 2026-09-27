"""Real MuQ-MuLan model sanity tests (marked `model`; run with `uv run pytest -m model`)."""

import numpy as np
import pytest

from local_smart_playlist.audio.decode import TARGET_SR
from local_smart_playlist.embed.model import EMBED_DIM, load_model
from local_smart_playlist.type_utils import NonEmptyTuple

pytestmark = pytest.mark.model


def test_model_load() -> None:

    embedder = load_model()
    assert embedder.dim == EMBED_DIM


def test_text_embedding() -> None:

    embedder = load_model()
    vecs = embedder.embed_texts(NonEmptyTuple(("a melancholic song", "an upbeat party song")))
    assert vecs.shape == (2, EMBED_DIM)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    # Different moods should not collapse to the same vector.
    dot = float(vecs[0] @ vecs[1])
    assert dot < 0.9


def test_audio_embedding(sine: object) -> None:

    t = np.arange(TARGET_SR * 10) / TARGET_SR
    window = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    embedder = load_model()
    vecs = embedder.embed_windows(NonEmptyTuple((window,)))
    assert vecs.shape == (1, EMBED_DIM)
    assert np.linalg.norm(vecs[0]) == pytest.approx(1.0, abs=1e-5)


def test_audio_text_similarity(sine: object) -> None:
    """Cross-modal sanity: matching caption outscores a mismatched caption."""

    t = np.arange(TARGET_SR * 10) / TARGET_SR
    window = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    embedder = load_model()
    audio_vecs = embedder.embed_windows(NonEmptyTuple((window,)))
    audio_vec = audio_vecs[0]
    texts = NonEmptyTuple(("a steady electronic sine tone", "a gritty distorted rock guitar riff"))
    text_vecs = embedder.embed_texts(texts)
    sims = text_vecs @ audio_vec
    assert sims.shape == (2,)
    assert float(sims[0]) > float(sims[1])
