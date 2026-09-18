"""Search ranking tests with a fake embedder and hybrid two-stage scoring."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from local_smart_playlist.index.store import Store
from local_smart_playlist.query import prompts
from local_smart_playlist.query.phrases import (
    ensure_documents,
    rank_by_documents,
    track_documents,
    vocab_fingerprint,
)
from local_smart_playlist.query.search import (
    PEAK_WEIGHT,
    query_vector,
    rank_by_similarity,
    rank_hybrid,
)

DIM = 32


def basis(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0  # pyrefly: ignore[unsupported-operation]  # numpy shape stubs lack __setitem__ (facebook/pyrefly#4901)
    return v


def fake_embedder(texts: list[str]) -> np.ndarray:
    """Deterministic: 'sad*' -> e0, 'happy*' -> e1, other -> hashed basis vector."""
    out = np.empty((len(texts), DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        low = text.lower()
        if low.startswith("sad"):
            out[i] = basis(0)  # pyrefly: ignore[unsupported-operation]
        elif low.startswith("happy"):
            out[i] = basis(1)  # pyrefly: ignore[unsupported-operation]
        else:
            v = np.zeros(DIM, dtype=np.float32)
            v[sum(ord(c) for c in low) % DIM] = 1.0  # pyrefly: ignore[unsupported-operation]
            out[i] = v  # pyrefly: ignore[unsupported-operation]
    return out


def upsert_track(store: Store, rel_path: str, window_vecs: np.ndarray) -> None:
    window_vecs = np.asarray(window_vecs, dtype=np.float32)
    mean = window_vecs.mean(axis=0)
    mean = (mean / float(np.linalg.norm(mean))).astype(np.float32)
    store.upsert(
        rel_path=rel_path,
        mean_vec=mean,
        p90_vec=mean,
        n_windows=len(window_vecs),
        duration=60.0,
        title=rel_path,
        model="fake",
        indexed_at="now",
    )
    store.add_windows(rel_path=rel_path, window_vecs=window_vecs)


def store_with(tmp_path: Path) -> Store:
    store = Store(tmp_path / "s.db", embed_dim=DIM)
    noise = np.full(DIM, 0.001, dtype=np.float32)
    for i in range(4):
        vec = basis(i) + noise
        vec = (vec / np.linalg.norm(vec)).astype(np.float32)
        upsert_track(store, f"t{i}.mp3", np.stack([vec] * 3))
    return store


def test_query_vector_direct() -> None:
    v = query_vector("sad", fake_embedder)
    assert np.allclose(v, basis(0))


def test_rank_hybrid_orders_by_similarity(tmp_path: Path) -> None:
    store = store_with(tmp_path)
    ranked = rank_hybrid(store, basis(1), k=4)
    assert ranked[0][0].rel_path == "t1.mp3"
    assert {t.rel_path for t, _ in ranked} == {f"t{i}.mp3" for i in range(4)}
    assert [s for _, s in ranked] == sorted((s for _, s in ranked), reverse=True)


def test_rank_hybrid_respects_k(tmp_path: Path) -> None:
    store = store_with(tmp_path)
    assert len(rank_hybrid(store, basis(1), k=2)) == 2


def test_rank_hybrid_excludes_seed(tmp_path: Path) -> None:
    store = store_with(tmp_path)
    seed = store.get_track("t1.mp3")
    assert seed is not None
    ranked = rank_by_similarity(store, seed.mean_vec, k=4, exclude={seed.rel_path})
    assert all(t.rel_path != "t1.mp3" for t, _ in ranked)
    assert ranked[0][0].rel_path in {"t0.mp3", "t2.mp3", "t3.mp3"}


def test_peak_weighting_lifts_peak_track(tmp_path: Path) -> None:
    """A track with one perfect window outranks a steady track when alpha is high."""
    store = Store(tmp_path / "peak.db", embed_dim=DIM)
    q = basis(0)
    # 'peaked': one perfect match, rest unrelated -> weak mean cos (0.5)
    peaked = np.stack([basis(0), basis(1), basis(1), basis(1)])
    # 'steady': all windows slightly worse than perfect -> strong mean cos (~0.976)
    steady_vec = basis(0) + 0.2 * basis(1)
    steady_vec = (steady_vec / np.linalg.norm(steady_vec)).astype(np.float32)
    steady = np.stack([steady_vec] * 4)
    upsert_track(store, "peaked.mp3", peaked)
    upsert_track(store, "steady.mp3", steady)

    # alpha = PEAK_WEIGHT (0.7): peaked 0.7 + 0.15 = 0.85 vs steady 0.976 -> steady first
    ranked = rank_hybrid(store, q, k=2)
    assert [t.rel_path for t, _ in ranked] == ["steady.mp3", "peaked.mp3"]

    # alpha = 1.0: pure peak -> peaked (1.0) beats steady (0.976)
    ranked_peak = rank_hybrid(store, q, k=2, alpha=1.0)
    assert ranked_peak[0][0].rel_path == "peaked.mp3"


def test_mean_weighting_stabilizes(tmp_path: Path) -> None:
    """alpha = 0 reproduces plain track-mean ranking."""
    store = store_with(tmp_path)
    ranked = rank_hybrid(store, basis(1), k=4, alpha=0.0)
    assert ranked[0][0].rel_path == "t1.mp3"
    _scores = [s for _, s in ranked]
    # steady 3-window tracks: score == mean cos (orthogonal others rank below)
    assert ranked[0][1] > ranked[1][1]


def test_rank_hybrid_legacy_track_without_windows(tmp_path: Path) -> None:
    store = Store(tmp_path / "legacy.db", embed_dim=DIM)
    vec = basis(1)
    store.upsert(
        rel_path="legacy.mp3",
        mean_vec=vec,
        p90_vec=vec,
        n_windows=0,
        duration=60.0,
        title="legacy",
        model="fake",
        indexed_at="now",
    )
    upsert_track(store, "normal.mp3", np.stack([basis(1)] * 3))
    ranked = rank_hybrid(store, basis(1), k=2)
    assert len(ranked) == 2
    assert ranked[0][0].rel_path == "normal.mp3"  # full windows: peak 1.0 beats legacy 1.0 - eps


def test_peak_weight_default() -> None:
    assert PEAK_WEIGHT == 0.7


def test_fallback_prompts() -> None:
    fallback = prompts.fallback_prompts("melancholic")
    assert list(fallback) == ["melancholic"]


def test_expand_mood_raises_on_unreachable(monkeypatch: Any) -> None:
    import httpx

    def boom(url: str, **kwargs: Any) -> Any:
        raise OSError("connection refused")

    monkeypatch.setattr(httpx, "post", boom)
    monkeypatch.setenv("SP_LLM_BASE_URL", "http://127.0.0.1:1")
    with pytest.raises(prompts.LlmError, match="adaptation failed"):
        _ = prompts.adapt_mood("sad")


def test_track_documents_orders_by_similarity() -> None:
    """Documents = nearest vocab phrases to the track mean vector."""
    mean = (basis(0) + 0.5 * basis(1)).astype(np.float32)
    mean = (mean / float(np.linalg.norm(mean))).astype(np.float32)
    docs = track_documents(mean, np.stack([basis(0), basis(1), basis(2)]), top_n=2)
    assert [i for i, _ in docs] == [0, 1]
    assert docs[0][1] > docs[1][1]


def test_track_documents_respects_top_n() -> None:
    docs = track_documents(basis(1), np.stack([basis(0), basis(1)]), top_n=1)
    assert docs == [(1, 1.0)]


def test_ensure_documents_backfills_and_persists(tmp_path: Path) -> None:
    """Backfill extracts documents for every track; fingerprint match keeps existing docs."""
    store = store_with(tmp_path)
    vocab = ["sad song", "happy song", "other song"]
    vocab_vecs = ensure_documents(store, fake_embedder, vocab=vocab)
    docs = store.all_document_indices()
    assert set(docs) == {f"t{i}.mp3" for i in range(4)}
    # every track's mean is a basis vector, so its top document matches exactly
    for i in range(4):
        top = docs[f"t{i}.mp3"][0]
        assert np.allclose(vocab_vecs[top], fake_embedder([vocab[top]]))
    # matching fingerprint: existing documents are left untouched
    store.set_meta("doc_vocab_sha", vocab_fingerprint(vocab))
    store.set_documents("t0.mp3", [(0, 1.0)])
    _ = ensure_documents(store, fake_embedder, vocab=vocab)
    assert store.all_document_indices()["t0.mp3"] == [0]  # untouched


def test_rank_by_documents_orders_by_document_relevance(tmp_path: Path) -> None:
    """Query text ranks tracks by best-matching document: sad track first for 'sad'."""
    store = Store(tmp_path / "doc.db", embed_dim=DIM)
    upsert_track(store, "sad.mp3", np.stack([basis(0)] * 3))
    upsert_track(store, "happy.mp3", np.stack([basis(1)] * 3))
    vocab = ["sad song", "happy song"]
    vocab_vecs = ensure_documents(store, fake_embedder, vocab=vocab, top_n=1)
    ranked = rank_by_documents(store, query_vec=fake_embedder(["sad song"])[0], vocab_vecs=vocab_vecs, k=2)
    assert ranked[0][0].rel_path == "sad.mp3"
    assert ranked[0][1] == 1.0


def test_rank_by_documents_respects_k_and_exclude(tmp_path: Path) -> None:
    store = Store(tmp_path / "doc2.db", embed_dim=DIM)
    for i in range(4):
        upsert_track(store, f"t{i}.mp3", np.stack([basis(i % 2)] * 3))
    vocab_vecs = ensure_documents(store, fake_embedder, vocab=["sad song", "happy song"], top_n=1)
    ranked = rank_by_documents(store, query_vec=basis(0), vocab_vecs=vocab_vecs, k=2, exclude={"t0.mp3"})
    rels = [t.rel_path for t, _ in ranked]
    assert "t0.mp3" not in rels
    assert len(ranked) == 2
