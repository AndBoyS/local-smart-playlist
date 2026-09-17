"""Query embedding and ranking against the track store."""


from dataclasses import dataclass
from typing import Protocol, cast

import numpy as np

from local_smart_playlist.index.store import Store, TrackRow
from local_smart_playlist.query import prompts

PEAK_WEIGHT = 0.7  # alpha: peak-window vs track-mean blend
CANDIDATE_POOL = 10  # prefilter fetches k * this many candidates by track mean
KNN_LIMIT = 4096  # sqlite-vec KNN k cap


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


@dataclass(frozen=True)
class _Candidate:
    rel_path: str
    mean_sim: float


def _score_track(
    *,
    rel_path: str,
    mean_sim: float,
    window_vecs: np.ndarray | None,
    query_vec: np.ndarray,
    alpha: float,
) -> float:
    """alpha * peak-window cos + (1 - alpha) * track-mean cos."""
    if window_vecs is None:
        return mean_sim  # legacy track without stored windows
    sims = np.asarray(window_vecs @ query_vec, dtype=np.float64)
    peak = float(sims.max())
    mean_vec = np.asarray(window_vecs.mean(axis=0), dtype=np.float64)
    mean_norm = float(np.linalg.norm(mean_vec))
    mean_sim_exact = float(mean_vec @ query_vec) / mean_norm if mean_norm > 0.0 else 0.0
    return alpha * peak + (1.0 - alpha) * mean_sim_exact


def rank_hybrid(
    store: Store,
    query_vec: np.ndarray,
    *,
    k: int,
    exclude: set[str] | None = None,
    alpha: float = PEAK_WEIGHT,
) -> list[tuple[TrackRow, float]]:
    """Two-stage ranking: track-mean prefilter, then exact peak-window rescore.

    Score = alpha * max cos(query, window) + (1 - alpha) * cos(query, track mean).
    Legacy tracks without stored windows fall back to their mean cos.
    """
    excluded = exclude if exclude is not None else set()
    total = store.track_count()
    want = min(k * CANDIDATE_POOL + len(excluded), total)

    if want > KNN_LIMIT:
        # Full-library ranking: plain scan over mean vectors (numpy matmul),
        # avoiding the sqlite-vec KNN row cap.
        rows = store.all_track_means()
        sims = np.asarray(
            np.stack([v for _, v in rows], dtype=np.float64) @ np.asarray(query_vec, dtype=np.float64),
            dtype=np.float64,
        ).ravel()
        order = np.argsort(-sims)[:want]
        candidates = [
            _Candidate(rel_path=cast("str", rows[i][0]), mean_sim=float(sims[i])) for i in order
        ]
    else:
        knn = store.knn(query_vec=query_vec, k=want)
        candidates = [_Candidate(hit.rel_path, 1.0 - hit.distance) for hit in knn]

    cand_paths = [c.rel_path for c in candidates if c.rel_path not in excluded]
    windows = store.load_windows(cand_paths)

    scored: list[tuple[TrackRow, float]] = []
    for cand in candidates:
        if cand.rel_path in excluded:
            continue
        track = store.get_track(cand.rel_path)
        if track is None:
            continue
        score = _score_track(
            rel_path=cand.rel_path,
            mean_sim=cand.mean_sim,
            window_vecs=windows.get(cand.rel_path),
            query_vec=query_vec,
            alpha=alpha,
        )
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
