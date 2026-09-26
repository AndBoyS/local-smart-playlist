"""Contrast-ranker tests: margins, sustained scoring, ranking order, tiebreaks."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from local_smart_playlist.embed.model import MuLanEmbedder
from local_smart_playlist.index.store import Store
from local_smart_playlist.query.contrast import (
    MOOD_ANCHORS,
    baseline_vector,
    contrast_score,
    query_vector_contrast,
    rank_by_contrast,
)

if TYPE_CHECKING:
    import torch

DIM = 32


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


def _embedder_for(mapping: dict[str, int]) -> MuLanEmbedder[32]:
    """MuLanEmbedder wrapping the hashed-basis model: text -> basis[mapping[text]]."""

    class _Model:
        def __call__(self, *, texts: "list[str] | None" = None, wavs: "torch.Tensor | None" = None) -> "torch.Tensor":
            import torch

            assert texts is not None
            return torch.from_numpy(np.stack([basis(mapping[t]) for t in texts]))

    return MuLanEmbedder(_Model(), dim=DIM)


def _torch_model(*, texts: "list[str] | None" = None, wavs: "torch.Tensor | None" = None) -> "torch.Tensor":
    import torch

    assert texts is not None
    return torch.from_numpy(fake_embedder(texts))


def _hash_embedder() -> MuLanEmbedder[32]:
    """MuLanEmbedder wrapping :func:`fake_embedder`."""
    return MuLanEmbedder(_torch_model, dim=DIM)


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


def test_mood_bank_covers_broad_moods() -> None:
    """Anchor set is non-trivial: enough breadth for a meaningful baseline."""
    assert len(MOOD_ANCHORS) >= 12
    assert len(set(MOOD_ANCHORS)) == len(MOOD_ANCHORS)


def test_baseline_is_mean_of_anchor_affinities() -> None:
    """Baseline stays unnormalized: window @ baseline == mean of per-anchor cosines."""
    bvec = baseline_vector(_hash_embedder())
    anchor_vecs = np.asarray(fake_embedder(list(MOOD_ANCHORS)), dtype=np.float64)
    anchor_vecs = anchor_vecs / np.linalg.norm(anchor_vecs, axis=-1, keepdims=True)
    anchor_mean = np.asarray(anchor_vecs.mean(axis=0), dtype=np.float64)  # pyrefly: ignore[unknown-argument-type]
    assert np.allclose(bvec, anchor_mean, atol=1e-6)
    # centroid of a diverse bank is shorter than 1 — that is the point
    assert np.linalg.norm(bvec) < 0.9
    qvec = query_vector_contrast("dreamy", _hash_embedder())
    assert np.allclose(np.linalg.norm(qvec), 1.0, atol=1e-5)
    # query embeds exactly as written — no second "mood." variant
    direct = np.asarray(fake_embedder(["dreamy"]), dtype=np.float64)
    direct = direct / float(np.linalg.norm(direct))
    assert np.allclose(qvec, direct, atol=1e-5)


def test_contrast_score_subtracts_baseline() -> None:
    """A window matching the broad-mood baseline as strongly as the query gains nothing."""
    q = basis(0)
    bvec = basis(1)  # single-anchor baseline for exact arithmetic
    pure = contrast_score(np.stack([basis(0)]), query_vec=q, baseline_vec=bvec)
    balanced = contrast_score(np.stack([basis(1)]), query_vec=q, baseline_vec=bvec)

    def window_with_margin(m: float) -> np.ndarray:
        w = (1.0 + m) * basis(0) + basis(1)
        return w.astype(np.float32)[None, :]

    assert pure.median_margin == pytest.approx(1.0, abs=1e-5)
    assert balanced.median_margin == pytest.approx(-1.0, abs=1e-5)  # zero window: no query, full baseline
    one = contrast_score(window_with_margin(0.5), query_vec=q, baseline_vec=bvec)
    assert one.median_margin == pytest.approx(0.5, abs=1e-5)
    assert one.coverage == pytest.approx(1.0)


def test_sustained_fit_beats_peak() -> None:
    """Track fitting throughout outranks track with one matching window."""
    q = basis(0)
    anchor_bank = _embedder_for({"a": 1, "b": 2})  # baseline spans b1/b2, orthogonal to q
    bvec = baseline_vector(anchor_bank, anchors=["a", "b"])

    store = Store(Path(":memory:"), embed_dim=DIM)
    upsert_track(store, "steady.mp3", np.stack([basis(0)] * 4))
    # one perfect window, rest on the baseline: median margin stays negative
    upsert_track(store, "peaky.mp3", np.stack([basis(0), basis(1), basis(1), basis(1)]))

    ranked = rank_by_contrast(store, query_vec=q, baseline_vec=bvec, k=2)
    assert [t.rel_path for t, _ in ranked] == ["steady.mp3", "peaky.mp3"]
    assert ranked[0][1] > 0.0
    assert ranked[1][1] < 0.0


def test_rank_ties_break_by_coverage_then_path() -> None:
    """Equal median margin: higher positive-window coverage first, then rel_path."""
    q = basis(0)
    bvec = basis(1)

    def windows_for(margins: list[float]) -> np.ndarray:
        # margin x == dot with q minus dot with baseline: (x+1)*b0 + b1
        return np.stack([((x + 1.0) * basis(0) + basis(1)).astype(np.float32) for x in margins])

    store = Store(Path(":memory:"), embed_dim=DIM)
    # same median 0.6, coverage 4/5 vs 3/5
    upsert_track(store, "b-wider.mp3", windows_for([0.6, 0.6, 0.6, 0.6, -1.0]))
    upsert_track(store, "a-narrow.mp3", windows_for([0.6, 0.6, 0.6, -1.0, -1.0]))

    ranked = rank_by_contrast(store, query_vec=q, baseline_vec=bvec, k=2)
    assert [t.rel_path for t, _ in ranked] == ["b-wider.mp3", "a-narrow.mp3"]


def test_rank_respects_k_and_exclude() -> None:
    q = basis(0)
    anchor_bank = _embedder_for({"a": 1, "b": 2})
    bvec = baseline_vector(anchor_bank, anchors=["a", "b"])
    store = Store(Path(":memory:"), embed_dim=DIM)
    for i in range(3):
        upsert_track(store, f"t{i}.mp3", np.stack([basis(0)] * 2))
    ranked = rank_by_contrast(store, query_vec=q, baseline_vec=bvec, k=2, exclude={"t0.mp3"})
    rels = [t.rel_path for t, _ in ranked]
    assert rels == ["t1.mp3", "t2.mp3"]


def test_track_without_windows_is_skipped() -> None:
    """Tracks without stored windows must be re-indexed; they do not rank."""
    q = basis(0)
    bvec = basis(1)
    store = Store(Path(":memory:"), embed_dim=DIM)
    store.upsert(
        rel_path="legacy.mp3",
        mean_vec=basis(0),
        p90_vec=basis(0),
        n_windows=0,
        duration=60.0,
        title="legacy",
        model="fake",
        indexed_at="now",
    )
    upsert_track(store, "windowed.mp3", np.stack([basis(0)] * 2))
    ranked = rank_by_contrast(store, query_vec=q, baseline_vec=bvec, k=2)
    assert [t.rel_path for t, _ in ranked] == ["windowed.mp3"]
