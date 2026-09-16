"""Search ranking tests with a fake embedder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from local_smart_playlist.index.store import Store
from local_smart_playlist.query import prompts
from local_smart_playlist.query.search import query_vector, rank, rank_by_similarity

DIM = 32


def basis(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def fake_embedder(texts: list[str]) -> np.ndarray:
    """Deterministic: 'sad*' -> e0, 'happy*' -> e1, mixed case -> average of letters."""
    out = np.empty((len(texts), DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        low = text.lower()
        if low.startswith("sad"):
            out[i] = basis(0)
        elif low.startswith("happy"):
            out[i] = basis(1)
        else:
            v = np.zeros(DIM, dtype=np.float32)
            v[sum(ord(c) for c in low) % DIM] = 1.0
            out[i] = v
    return out


def store_with(tmp_path: Path) -> Store:
    store = Store(tmp_path / "s.db", embed_dim=DIM)
    noise = np.full(DIM, 0.001, dtype=np.float32)
    for i in range(4):
        vec = basis(i) + noise
        vec = vec / np.linalg.norm(vec)
        store.upsert(rel_path=f"t{i}.mp3", mean_vec=vec, p90_vec=vec, n_windows=1, duration=60.0, title=f"title{i}", model="fake", indexed_at="now")
    return store


def test_query_vector_direct() -> None:
    v = query_vector("sad", fake_embedder)
    assert np.allclose(v, basis(0))


def test_query_vector_llm_averages_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        prompts,
        "expand_mood",
        lambda mood: [  # pyrefly: ignore
            "sad and slow",
            "sad piano line",
            "sad minor key ballad",
        ],
    )
    v = query_vector("sad", fake_embedder, use_llm=True)
    assert np.allclose(v, basis(0))  # all 'sad ...' prompts map to e0


def test_rank_orders_by_similarity(tmp_path: Any) -> None:
    store = store_with(tmp_path)
    rows = rank(store, basis(1), k=4)
    assert rows[0].rel_path == "t1.mp3"
    assert {t.rel_path for t in rows} == {f"t{i}.mp3" for i in range(4)}


def test_rank_respects_k(tmp_path: Any) -> None:
    store = store_with(tmp_path)
    assert len(rank(store, basis(1), k=2)) == 2


def test_rank_excludes_seed(tmp_path: Any) -> None:
    store = store_with(tmp_path)
    seed = store.get_track("t1.mp3")
    assert seed is not None
    rows = rank_by_similarity(store, seed.mean_vec, k=4, exclude={seed.rel_path})
    assert all(t.rel_path != "t1.mp3" for t in rows)
    assert rows[0].rel_path in {"t0.mp3", "t2.mp3", "t3.mp3"}


def test_fallback_prompts() -> None:
    fallback = prompts.fallback_prompts("melancholic")
    _ = list(fallback)
    _ = list(prompts.fallback_prompts("melancholic"))


def test_expand_mood_raises_on_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    def boom(url: str, **kwargs: Any) -> Any:
        raise OSError("connection refused")

    monkeypatch.setattr(httpx, "post", boom)
    monkeypatch.setenv("SP_LLM_BASE_URL", "http://127.0.0.1:1")
    with pytest.raises(prompts.LlmError, match="expansion failed"):
        _ = prompts.expand_mood("sad")
