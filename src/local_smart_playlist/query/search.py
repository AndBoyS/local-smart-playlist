"""Query embedding and ranking against the track store."""


from typing import Protocol

import numpy as np

from local_smart_playlist.index.store import Store, TrackRow
from local_smart_playlist.query import prompts

PEAK_WEIGHT = 0.7  # alpha: peak-window vs track-mean blend
CANDIDATE_POOL = 10  # prefilter fetches k * this many candidates by track mean


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


def rank_hybrid(
    store: Store,
    query_vec: np.ndarray,
    *,
    k: int,
    exclude: set[str] | None = None,
    alpha: float = PEAK_WEIGHT,
) -> list[tuple[TrackRow, float]]:
    """Two-stage ranking: track-mean KNN prefilter, then exact peak-window rescore.

    Score = alpha * max cos(query, window) + (1 - alpha) * cos(query, track mean).
    Legacy tracks without stored windows fall back to their mean cos.
    """
    excluded = exclude if exclude is not None else set()
    candidates = store.knn(query_vec=query_vec, k=k * CANDIDATE_POOL + len(excluded))
    cand_paths = [hit.rel_path for hit in candidates if hit.rel_path not in excluded]
    windows = store.load_windows(cand_paths)

    scored: list[tuple[TrackRow, float]] = []
    for hit in candidates:
        if hit.rel_path in excluded:
            continue
        track = store.get_track(hit.rel_path)
        if track is None:
            continue
        wv = windows.get(hit.rel_path)
        if wv is None:
            score = 1.0 - hit.distance
        else:
            sims = np.asarray(wv @ query_vec, dtype=np.float64)
            peak = float(sims.max())
            mean_vec = np.asarray(wv.mean(axis=0), dtype=np.float64)
            mean_norm = float(np.linalg.norm(mean_vec))
            mean_sim = float(mean_vec @ query_vec) / mean_norm if mean_norm > 0.0 else 0.0
            score = alpha * peak + (1.0 - alpha) * mean_sim
        scored.append((track, score))

    def by_score(entry: tuple[TrackRow, float]) -> float:
        return entry[1]

    scored.sort(key=by_score, reverse=True)
    return scored[:k]


def rank_by_similarity(
    store: Store, seed_vec: np.ndarray, *, k: int, exclude: set[str] | None = None, alpha: float = PEAK_WEIGHT
) -> list[tuple[TrackRow, float]]:
    """Same two-stage path, driven by a seed track vector instead of a text query."""
    return rank_hybrid(store, seed_vec, k=k, exclude=exclude, alpha=alpha)
