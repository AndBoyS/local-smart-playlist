"""Vocab-calibration ranker tests: percentiles, guard fallback, ranking order."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import torch

from local_smart_playlist.embed.model import MuLanEmbedder
from local_smart_playlist.index.store import MetaKey, Store
from local_smart_playlist.query.contrast import baseline_vector, baseline_vector_cached
from local_smart_playlist.query.vocab_cal import (
    MARGIN_TAU,
    rank_by_vocab_calibration,
    vocab_calibration_score,
    vocab_vector_bank,
    vocab_vector_bank_cached,
)
from local_smart_playlist.type_utils import NonEmptyTuple

DIM = 32
VOCAB = NonEmptyTuple(("cap a", "cap b", "cap c", "cap d"))


def basis(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0  # pyrefly: ignore[unsupported-operation]  # numpy shape stubs lack __setitem__ (facebook/pyrefly#4901)
    return v


def fake_embedder(texts: list[str]) -> np.ndarray:
    """Deterministic: text -> hashed basis vector, stable across calls."""
    out = np.empty((len(texts), DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        out[i] = basis(sum(ord(c) for c in text) % DIM)  # pyrefly: ignore[unsupported-operation]  # numpy shape stubs lack __setitem__
    return out


def _torch_model(*, texts: "list[str] | None" = None, wavs: "torch.Tensor | None" = None) -> "torch.Tensor":
    import torch

    assert texts is not None
    return torch.from_numpy(fake_embedder(texts))


def _embedder_for(mapping: dict[str, int]) -> MuLanEmbedder[32]:
    """MuLanEmbedder wrapping the hashed-basis model: text -> basis[mapping[text]]."""

    class _Model:
        def __call__(self, *, texts: "list[str] | None" = None, wavs: "torch.Tensor | None" = None) -> "torch.Tensor":
            import torch

            assert texts is not None
            return torch.from_numpy(np.stack([basis(mapping[t]) for t in texts]))

    return MuLanEmbedder(_Model(), dim=DIM)


def upsert_track(store: Store, rel_path: str, window_vecs: np.ndarray) -> None:
    window_vecs = np.asarray(window_vecs, dtype=np.float32)
    mean = window_vecs.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    mean = (mean / norm if norm > 0 else mean).astype(np.float32)
    store.upsert(
        rel_path=rel_path,
        mean_vec=mean,
        p90_vec=mean,
        n_windows=len(window_vecs),
        duration=60.0,
        title=rel_path,
        model="fake",
        indexed_at="now",
        window_vecs=window_vecs,
    )


def _vocab_vecs(mapping: dict[str, int]) -> np.ndarray:
    return vocab_vector_bank(_embedder_for(mapping), VOCAB)


def test_vocab_bank_is_unit_rows() -> None:
    bank = _vocab_vecs({"cap a": 1, "cap b": 2, "cap c": 3, "cap d": 4})
    assert bank.shape == (4, DIM)
    assert np.allclose(np.linalg.norm(bank, axis=-1), 1.0, atol=1e-5)


def test_vocab_bank_cache_uses_fixed_meta_key(tmp_path: Path) -> None:
    calls: list[int] = []

    class CountingModel:
        def __call__(self, *, texts: "list[str] | None" = None, wavs: "torch.Tensor | None" = None) -> "torch.Tensor":
            import torch

            assert texts is not None
            calls.append(len(texts))
            return torch.from_numpy(fake_embedder(texts))

    model = MuLanEmbedder(CountingModel(), dim=DIM)
    with Store(tmp_path / "cache.db", embed_dim=DIM) as store:
        first = vocab_vector_bank_cached(store, model, vocab=VOCAB)
        encoded = store.get_meta(MetaKey.VOCAB_VECS)
        assert encoded is not None
        assert encoded.startswith("v1\nOpenMuQ/MuQ-MuLan-large\n")

        second = vocab_vector_bank_cached(store, model, vocab=VOCAB)
        assert np.array_equal(first, second)
        assert calls == [len(VOCAB)]

        changed_vocab = NonEmptyTuple(("cap b", "cap a", "cap c", "cap d"))
        _ = vocab_vector_bank_cached(store, model, vocab=changed_vocab)
        assert calls == [len(VOCAB), len(changed_vocab)]


def test_baseline_vector_cache_tracks_model_and_anchor_identity(tmp_path: Path) -> None:
    calls: list[int] = []

    class CountingModel:
        def __call__(self, *, texts: "list[str] | None" = None, wavs: "torch.Tensor | None" = None) -> "torch.Tensor":
            import torch

            assert texts is not None
            calls.append(len(texts))
            return torch.from_numpy(fake_embedder(texts))

    model = MuLanEmbedder(CountingModel(), dim=DIM)
    anchors = NonEmptyTuple(("anchor a", "anchor b"))
    with Store(tmp_path / "baseline-cache.db", embed_dim=DIM) as store:
        first = baseline_vector_cached(store, model, anchors=anchors)
        encoded = store.get_meta(MetaKey.MOOD_BASELINE_VEC)
        assert encoded is not None
        assert encoded.startswith("v1\nOpenMuQ/MuQ-MuLan-large\n")

        encoded_parts = encoded.split("\n", maxsplit=3)
        assert len(encoded_parts) == 4
        store.set_meta(
            MetaKey.MOOD_BASELINE_VEC,
            "\n".join((encoded_parts[0], "other/model", encoded_parts[2], encoded_parts[3])),
        )
        second = baseline_vector_cached(store, model, anchors=anchors)
        assert np.array_equal(first, second)
        assert calls == [len(anchors), len(anchors)]

        _ = baseline_vector_cached(store, model, anchors=anchors)
        assert calls == [len(anchors), len(anchors)]

        changed_anchors = NonEmptyTuple(("anchor b", "anchor a"))
        _ = baseline_vector_cached(store, model, anchors=changed_anchors)
        assert calls == [len(anchors), len(anchors), len(changed_anchors)]

        store.set_meta(MetaKey.MOOD_BASELINE_VEC, "invalid cache")
        _ = baseline_vector_cached(store, model, anchors=changed_anchors)
        assert calls == [len(anchors), len(anchors), len(changed_anchors), len(changed_anchors)]


def test_percentile_counts_captions_beaten() -> None:
    """Window on the query direction beats 3 of 4 vocab captions (one is itself)."""
    q = basis(0)
    bank = _vocab_vecs({"cap a": 0, "cap b": 1, "cap c": 2, "cap d": 3})
    margin_vec = basis(1)
    # window == query direction: sims_v = [1, 0, 0, 0], sim_q = 1 -> beaten 3/4
    on = vocab_calibration_score(np.stack([basis(0)]), query_vec=q, vocab_vecs=bank, margin_vec=margin_vec)
    assert on.score == pytest.approx(0.75, abs=1e-6)
    assert on.coverage == pytest.approx(1.0)  # 0.75 > 0.5
    # off-mood but vocab-covered window: sim_q = 0, sims_v = [0, 1, 0, 0] -> beaten 0/4
    off = vocab_calibration_score(np.stack([basis(1)]), query_vec=q, vocab_vecs=bank, margin_vec=basis(5))
    assert off.score == pytest.approx(0.0, abs=1e-6)


def test_score_averages_window_percentiles() -> None:
    """Two windows: one at 0.75, one at 0.0 -> track score 0.375, coverage 0.5."""
    q = basis(0)
    bank = _vocab_vecs({"cap a": 0, "cap b": 1, "cap c": 2, "cap d": 3})
    result = vocab_calibration_score(np.stack([basis(0), basis(1)]), query_vec=q, vocab_vecs=bank, margin_vec=basis(5))
    assert result.score == pytest.approx(0.375, abs=1e-6)
    assert result.coverage == pytest.approx(0.5)


def test_guard_falls_back_to_margin_sigmoid() -> None:
    """Track far from every vocab caption: score = sigmoid(mean margin / tau)."""
    q = basis(0)
    # vocab orthogonal to the windows: best vocab sim is 0 < VOCAB_COVER_FLOOR
    bank = _vocab_vecs({"cap a": 1, "cap b": 2, "cap c": 3, "cap d": 4})
    margin_vec = basis(5)
    # margin = cos(w, q) - cos(w, margin_vec) = 1.0 -> sigmoid(20) ~ 1
    hot = vocab_calibration_score(np.stack([np.asarray(basis(0))]), query_vec=q, vocab_vecs=bank, margin_vec=margin_vec)
    expected_hot = float(1.0 / (1.0 + np.exp(-1.0 / MARGIN_TAU)))  # pyrefly: ignore[unknown-argument-type]
    assert hot.score == pytest.approx(expected_hot, abs=1e-6)
    assert hot.coverage == pytest.approx(1.0)
    # margin 0 -> neutral 0.5, same scale as percentile space
    neutral = vocab_calibration_score(
        np.stack([np.asarray((basis(0) + basis(5)).astype(np.float32))]),
        query_vec=q,
        vocab_vecs=bank,
        margin_vec=margin_vec,
    )
    # cos(w, q) = cos(w, margin) = 1/sqrt(2) -> margin 0
    assert neutral.score == pytest.approx(0.5, abs=1e-6)


def test_rank_orders_by_calibration_and_breaks_ties() -> None:
    """Sustained fit outranks a peak window; equal scores break by rel path."""
    q = basis(0)
    bank = _vocab_vecs({"cap a": 0, "cap b": 1, "cap c": 2, "cap d": 3})
    margin_vec = basis(1)
    store = Store(Path(":memory:"), embed_dim=DIM)
    upsert_track(store, "a-steady.mp3", np.stack([basis(0)] * 4))
    upsert_track(store, "b-steady.mp3", np.stack([basis(0)] * 4))
    # one perfect window, rest orthogonal: mean percentile (0.75 + 0 + 0 + 0)/4
    upsert_track(store, "peaky.mp3", np.stack([basis(0), basis(9), basis(9), basis(9)]))

    ranked = rank_by_vocab_calibration(store, query_vec=q, vocab_vecs=bank, margin_vec=margin_vec, k=3)
    assert [t.rel_path for t, _ in ranked] == ["a-steady.mp3", "b-steady.mp3", "peaky.mp3"]
    assert ranked[0][1] == pytest.approx(ranked[1][1], abs=1e-6)
    assert ranked[2][1] == pytest.approx(0.1875, abs=1e-6)


def test_rank_respects_k_and_exclude() -> None:
    q = basis(0)
    bank = _vocab_vecs({"cap a": 0, "cap b": 1, "cap c": 2, "cap d": 3})
    margin_vec = basis(1)
    store = Store(Path(":memory:"), embed_dim=DIM)
    upsert_track(store, "a.mp3", np.stack([basis(0)] * 2))
    upsert_track(store, "b.mp3", np.stack([basis(0)] * 2))

    ranked = rank_by_vocab_calibration(
        store, query_vec=q, vocab_vecs=bank, margin_vec=margin_vec, k=1, exclude={"a.mp3"}
    )
    assert [t.rel_path for t, _ in ranked] == ["b.mp3"]


def test_real_embedder_paths_run() -> None:
    """baseline_vector + hashed embedder compose: guard fallback path is reachable."""
    embedder = MuLanEmbedder(_torch_model, dim=DIM)
    bvec = baseline_vector(embedder, anchors=NonEmptyTuple(("x", "y")))
    assert bvec.shape == (DIM,)
