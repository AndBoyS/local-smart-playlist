"""Query embedding and ranking against the track store."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from local_smart_playlist.index.store import Store, TrackRow
from local_smart_playlist.query import prompts


class TextEmbedder(Protocol):
    """Anything that turns text into a unit-norm CLAP vector."""

    def __call__(self, texts: list[str]) -> np.ndarray: ...


def query_vector(mood: str, embedder: TextEmbedder, *, use_llm: bool = False) -> np.ndarray:
    """Embed *mood* (optionally LLM-expanded) into a single unit-norm vector."""
    text_prompts = prompts.expand_mood(mood) if use_llm else prompts.fallback_prompts(mood)
    vecs = embedder(list(text_prompts))
    if vecs.shape[0] == 0:
        msg = "no prompts to embed"
        raise ValueError(msg)
    mean = vecs.mean(axis=0).astype(np.float32)
    norm = float(np.linalg.norm(mean))
    if norm == 0.0:
        msg = "degenerate query embedding"
        raise ValueError(msg)
    return mean / norm


def rank(store: Store, query_vec: np.ndarray, *, k: int, exclude: set[str] | None = None) -> list[TrackRow]:
    """KNN on the track mean vector, minus excluded rel_paths."""
    excluded = exclude if exclude is not None else set()
    hits = store.knn(query_vec=query_vec, k=k + len(excluded))
    rows: list[TrackRow] = []
    seen: set[str] = set()
    for hit in hits:
        if hit.rel_path in excluded or hit.rel_path in seen:
            continue
        track = store.get_track(hit.rel_path)
        if track is not None:
            rows.append(track)
            seen.add(hit.rel_path)
        if len(rows) >= k:
            break
    return rows


def rank_by_similarity(store: Store, seed_vec: np.ndarray, *, k: int, exclude: set[str] | None = None) -> list[TrackRow]:
    """Same KNN path, driven by a seed track vector instead of a text query."""
    return rank(store, seed_vec, k=k, exclude=exclude)
