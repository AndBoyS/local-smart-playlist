"""Search ranking tests with a fake embedder and hybrid two-stage scoring."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from local_smart_playlist.index.store import Store
from local_smart_playlist.query import prompts
from local_smart_playlist.query.search import (
    PEAK_WEIGHT,
    rank_by_similarity,
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
        window_vecs=window_vecs,
    )


def store_with(tmp_path: Path) -> Store:
    store = Store(tmp_path / "s.db", embed_dim=DIM)
    noise = np.full(DIM, 0.001, dtype=np.float32)
    for i in range(4):
        vec = basis(i) + noise
        vec = (vec / np.linalg.norm(vec)).astype(np.float32)
        upsert_track(store, f"t{i}.mp3", np.stack([vec] * 3))
    return store


def test_rank_by_similarity_orders_by_similarity(tmp_path: Path) -> None:
    store = store_with(tmp_path)
    ranked = rank_by_similarity(store, basis(1), k=4)
    assert ranked[0][0].rel_path == "t1.mp3"
    assert {t.rel_path for t, _ in ranked} == {f"t{i}.mp3" for i in range(4)}
    assert [s for _, s in ranked] == sorted((s for _, s in ranked), reverse=True)


def test_rank_by_similarity_respects_k(tmp_path: Path) -> None:
    store = store_with(tmp_path)
    assert len(rank_by_similarity(store, basis(1), k=2)) == 2


def test_rank_by_similarity_excludes_seed(tmp_path: Path) -> None:
    store = store_with(tmp_path)
    seed = store.get_track("t1.mp3")
    assert seed is not None
    assert seed.vectors is not None
    ranked = rank_by_similarity(store, seed.vectors.mean_vec, k=4, exclude={seed.rel_path})
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
    ranked = rank_by_similarity(store, q, k=2)
    assert [t.rel_path for t, _ in ranked] == ["steady.mp3", "peaked.mp3"]

    # alpha = 1.0: pure peak -> peaked (1.0) beats steady (0.976)
    ranked_peak = rank_by_similarity(store, q, k=2, alpha=1.0)
    assert ranked_peak[0][0].rel_path == "peaked.mp3"


def test_mean_weighting_stabilizes(tmp_path: Path) -> None:
    """alpha = 0 reproduces plain track-mean ranking."""
    store = store_with(tmp_path)
    ranked = rank_by_similarity(store, basis(1), k=4, alpha=0.0)
    assert ranked[0][0].rel_path == "t1.mp3"
    _scores = [s for _, s in ranked]
    # steady 3-window tracks: score == mean cos (orthogonal others rank below)
    assert ranked[0][1] > ranked[1][1]


def test_rank_by_similarity_legacy_track_without_windows(tmp_path: Path) -> None:
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
    ranked = rank_by_similarity(store, basis(1), k=2)
    assert len(ranked) == 2
    scores = {track.rel_path: score for track, score in ranked}
    assert scores["normal.mp3"] == pytest.approx(1.0)
    assert scores["legacy.mp3"] == pytest.approx(1.0)


def test_peak_weight_default() -> None:
    assert PEAK_WEIGHT == 0.7


def test_expand_mood_raises_on_unreachable(monkeypatch: Any, caplog: Any) -> None:
    import logging

    import httpx

    def boom(url: str, **kwargs: Any) -> Any:
        raise OSError("connection refused")

    monkeypatch.setattr(httpx, "post", boom)
    monkeypatch.setenv("SP_LLM_BASE_URL", "http://127.0.0.1:1")
    with (
        pytest.raises(prompts.LlmError, match="adaptation failed"),
        caplog.at_level(logging.WARNING, logger="local_smart_playlist.query.prompts"),
    ):
        _ = prompts.adapt_mood("sad")
    assert any("llm adapt failed" in r.message for r in caplog.records)


def test_adapt_prompt_uses_model_caption_examples() -> None:
    prompt = prompts._adapt_prompt()
    vocab = prompts.caption_vocab()
    assert "Readout-caption style examples:" in prompt
    assert vocab[0] in prompt
    assert vocab[-1] in prompt


def test_adapt_mood_logs_correction(monkeypatch: Any, caplog: Any) -> None:
    """Successful correction logs input -> output at INFO."""
    import logging
    from types import SimpleNamespace

    def fake_post(url: str, **kwargs: Any) -> Any:
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "choices": [
                    {
                        "message": {
                            "content": "1. energetic mood, driving beat.\n2. upbeat mood, danceable rhythm.\n3. focused mood, steady tempo.\n4. intense mood, fast pulse."
                        }
                    }
                ]
            },
        )

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    with caplog.at_level(logging.INFO, logger="local_smart_playlist.query.prompts"):
        result = prompts.adapt_mood("sad rainy morning")
    assert result == [
        "energetic mood, driving beat.",
        "upbeat mood, danceable rhythm.",
        "focused mood, steady tempo.",
        "intense mood, fast pulse.",
    ]
    record = next(r for r in caplog.records if "llm adapt:" in r.message)
    assert "sad rainy morning" in record.message
    assert "energetic mood, driving beat." in record.message
