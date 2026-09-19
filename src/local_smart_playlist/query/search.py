"""Query embedding and ranking against the track store."""

from dataclasses import dataclass
from typing import cast

import numpy as np
from shape_extensions import IntVar

from local_smart_playlist.index.store import Store, TrackRow

T = IntVar("T")  # total windows across candidate tracks
D = IntVar("D")  # embedding dim

PEAK_WEIGHT = 0.7  # alpha: peak-window vs track-mean blend
CANDIDATE_POOL = 10  # prefilter fetches k * this many candidates by track mean
KNN_LIMIT = 4096  # sqlite-vec KNN k cap


@dataclass(frozen=True)
class _Candidate:
    rel_path: str
    mean_sim: float


def _rescore_flat(
    *,
    paths: list[str],
    big: np.ndarray[[T, D]],
    counts: list[int],
    query_vec: np.ndarray[[D]],
    alpha: float,
) -> dict[str, float]:
    """Batched alpha * peak-window cos + (1 - alpha) * track-mean cos.

    One [total_windows, D] @ q matmul instead of one small matmul per track;
    track blocks are contiguous rows, so per-track peak/mean are slice views.
    """
    if len(paths) == 0:
        return {}
    q32 = np.asarray(query_vec, dtype=np.float32)
    q64 = np.asarray(query_vec, dtype=np.float64)
    starts = np.cumsum([0, *counts[:-1]])
    ends = np.cumsum(counts)

    sims = np.asarray(big @ q32, dtype=np.float64)
    starts = np.cumsum([0, *counts[:-1]]).tolist()
    ends = np.cumsum(counts).tolist()
    scores: dict[str, float] = {}
    for p, s, e in zip(paths, starts, ends, strict=True):
        seg = sims[s:e]
        peak = float(seg.max())
        mean_sim = 0.0
        if alpha < 1.0:
            # track-mean cos == mean of window sims, but the mean-vector norm
            # still needs the window block itself
            mean_vec = big[s:e].mean(axis=0).astype(np.float64)
            mean_norm = float(np.linalg.norm(mean_vec))
            mean_sim = float(mean_vec @ q64) / mean_norm if mean_norm > 0.0 else 0.0
        scores[p] = alpha * peak + (1.0 - alpha) * mean_sim
    return scores


def rank_hybrid(
    store: Store,
    query_vec: np.ndarray[[D]],
    *,
    k: int,
    exclude: set[str] | None = None,
    alpha: float = PEAK_WEIGHT,
) -> list[tuple[TrackRow[D], float]]:
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
        sims = (np.stack([v for _, v in rows]).astype(np.float64) @ np.asarray(query_vec, dtype=np.float64)).ravel()
        order = np.argsort(-sims)[:want]
        candidates = [_Candidate(rel_path=cast("str", rows[i.item()][0]), mean_sim=float(sims[i])) for i in order]
    else:
        knn = store.knn(query_vec=query_vec, k=want)
        candidates = [_Candidate(hit.rel_path, 1.0 - hit.distance) for hit in knn]

    cand_paths = [c.rel_path for c in candidates if c.rel_path not in excluded]
    paths, big, counts = store.load_windows_flat(cand_paths)
    scores = _rescore_flat(paths=paths, big=big, counts=counts, query_vec=query_vec, alpha=alpha)
    tracks = store.get_tracks(cand_paths)

    scored: list[tuple[TrackRow[D], float]] = []
    for cand in candidates:
        track = tracks.get(cand.rel_path)
        if track is None:
            continue
        # legacy track without stored windows falls back to mean cos
        scored.append((track, scores.get(cand.rel_path, cand.mean_sim)))

    def by_score(entry: tuple[TrackRow[D], float]) -> float:
        return entry[1]

    scored.sort(key=by_score, reverse=True)
    return scored[:k]


def rank_by_similarity(
    store: Store, seed_vec: np.ndarray[[D]], *, k: int, exclude: set[str] | None = None, alpha: float = PEAK_WEIGHT
) -> list[tuple[TrackRow[D], float]]:
    """Same two-stage path, driven by a seed track vector instead of a text query."""
    return rank_hybrid(store, seed_vec, k=k, exclude=exclude, alpha=alpha)
