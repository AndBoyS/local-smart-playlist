"""Query embedding and ranking against the track store."""

import numpy as np
from shape_extensions import IntVar

from local_smart_playlist.index.store import Store, TrackData

T = IntVar("T")  # total windows across candidate tracks
D = IntVar("D")  # embedding dim

PEAK_WEIGHT = 0.7  # alpha: peak-window vs track-mean blend
CANDIDATE_POOL = 10  # prefilter fetches k * this many candidates by track mean


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


def rank_by_similarity(
    store: Store,
    query_vec: np.ndarray[[D]],
    *,
    k: int,
    exclude: set[str] | None = None,
    alpha: float = PEAK_WEIGHT,
) -> list[tuple[TrackData[D], float]]:
    """Rank tracks for a query or seed vector with exact peak-window rescoring.

    Score = alpha * max cos(query, window) + (1 - alpha) * cos(query, track mean).
    Legacy tracks without stored windows fall back to their mean cos.
    """
    excluded = exclude if exclude is not None else set()
    total = store.track_count()
    query_amount = min(k * CANDIDATE_POOL + len(excluded), total)
    if query_amount == 0:
        return []

    track_to_mean = store.all_track_means()
    if len(track_to_mean) == 0:
        return []
    paths = list(track_to_mean)
    mean_matrix = np.stack(list(track_to_mean.values())).astype(np.float64)
    sims = (mean_matrix @ np.asarray(query_vec, dtype=np.float64)).ravel()
    order = np.argsort(-sims)[:query_amount]
    path_to_mean_sim = {paths[int(i.item())]: float(sims[i]) for i in order}

    cand_paths = [rel_path for rel_path in path_to_mean_sim if rel_path not in excluded]
    flat = store.load_windows_batched(cand_paths)
    scores = _rescore_flat(
        paths=flat.rel_paths,
        big=flat.vec_matrix,
        counts=flat.window_sizes,
        query_vec=query_vec,
        alpha=alpha,
    )
    tracks = store.get_tracks(cand_paths)

    scored: list[tuple[TrackData[D], float]] = []
    for rel_path, mean_sim in path_to_mean_sim.items():
        track = tracks.get(rel_path)
        if track is None:
            continue
        # legacy track without stored windows falls back to mean cos
        scored.append((track, scores.get(rel_path, mean_sim)))

    def by_score(entry: tuple[TrackData[D], float]) -> float:
        return entry[1]

    scored.sort(key=by_score, reverse=True)
    return scored[:k]
