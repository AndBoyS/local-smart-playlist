"""Margin scoring and the broad-mood anchor bank (guard/eval components).

The additive-margin ranker described here was the production ranker until it
was replaced by the caption-vocab calibration (``query/vocab_cal.py``). Its
pieces remain live:

- **MOOD_ANCHORS / baseline_vector**: the 20 broad-mood anchor bank and its
  unnormalized mean — used as the baseline in the guard fallback of the
  vocab-calibration ranker (tracks with uninformative vocab profiles).
- **query_vector_contrast**: canonical query embedding ({q} + "{q} mood.",
  averaged) — still how text queries become vectors.
- **contrast_score / rank_by_contrast**: the median-margin ranker, kept for
  ``scripts/eval_rank.py`` comparisons and tests only.

Historical rationale (why the margin exists): direct text→audio ranking
buries whole textures (ambient, game OST, sparse piano) because MuQ-MuLan
audio→text cosine scale is timbre-dependent, and a single peak window lets
one short passage admit an off-mood track. The margin subtracts the mean
per-anchor affinity ("music in general") per window, and the track score is
the *median* margin, so most of the track must fit.
"""

from dataclasses import dataclass

import numpy as np
from shape_extensions import IntVar

from local_smart_playlist.index.store import Store, TrackRow
from local_smart_playlist.query.search import TextEmbedder

W = IntVar("W")  # window count
D = IntVar("D")  # embedding dim
B = IntVar("B")  # bank rows (anchors / caption variants)

# Broad mood anchors spanning the widest stylistic range the vocab's caption
# style covers (en + zh). Their mean approximates "music in general"; window
# affinity to it is subtracted from query affinity.
MOOD_ANCHORS: list[str] = [
    "happy mood, major key, bright melody.",
    "sad mood, slow tempo, minor key.",
    "calm mood, slow tempo, soft dynamics.",
    "relaxed mood, mellow groove, warm tone.",
    "energetic mood, fast tempo, driving drums.",
    "dark mood, low drones, dissonant textures.",
    "dreamy mood, reverb pads, hazy atmosphere.",
    "angry mood, aggressive distorted guitars.",
    "tense mood, suspenseful pulsing strings.",
    "nostalgic mood, warm analog synths.",
    "romantic mood, intimate vocals, slow tempo.",
    "欢快情绪，明亮旋律。",
    "悲伤情绪，慢节奏，小调。",
    "平静情绪，慢节奏，柔和音色。",
    "充满活力的情绪，快节奏。",
    "黑暗情绪，低沉氛围。",
    "梦幻情绪，混响氛围。",
    "紧张情绪，悬疑氛围。",
    "怀旧情绪，温暖合成器。",
    "浪漫情绪，亲密氛围。",
]


def _unit(vecs: np.ndarray[[B, D]]) -> np.ndarray[[B, D]]:
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)  # pyrefly: ignore[unknown-argument-type]
    return np.asarray(vecs / safe, dtype=np.float32)


def baseline_vector(embedder: TextEmbedder, *, anchors: list[str] | None = None) -> np.ndarray[[D]]:
    """Unnormalized mean of unit broad-mood anchor embeddings.

    Kept unnormalized so ``window @ baseline`` equals the *mean of the
    window's per-anchor cosines* — its affinity to music-in-general. A unit
    centroid would inflate the baseline by 1/‖centroid‖ (~2–3x for a diverse
    anchor set) and bury gentle textures under the broad baseline.
    """
    vecs = _unit(np.asarray(embedder(list(MOOD_ANCHORS if anchors is None else anchors)), dtype=np.float32))
    return np.asarray(vecs.mean(axis=0), dtype=np.float32)


def query_vector_contrast(mood: str, embedder: TextEmbedder) -> np.ndarray[[D]]:
    """Embed *mood* as caption-style variants, averaged to one unit query vector."""
    variants = [mood, f"{mood} mood."]
    vecs = _unit(np.asarray(embedder(variants), dtype=np.float32))
    return _unit(vecs.mean(axis=0)[None, :])[0]


@dataclass(frozen=True)
class ContrastScore:
    """Sustained-mood fit of one track: median margin plus window coverage."""

    median_margin: float
    coverage: float


def _median(values: np.ndarray[[W]]) -> float:
    """Median without np.median (shape stubs track the module form poorly)."""
    ordered = np.asarray(values, dtype=np.float64).copy()
    n = ordered.shape[0]
    if n == 0:
        return 0.0
    ordered.sort()  # in-place method form; np.sort module form is untracked
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def contrast_score(
    window_vecs: np.ndarray[[W, D]], *, query_vec: np.ndarray[[D]], baseline_vec: np.ndarray[[D]]
) -> ContrastScore:
    """Median query-vs-baseline margin and positive-margin coverage over windows."""
    sims_q = np.asarray(window_vecs @ query_vec, dtype=np.float64).ravel()
    sims_b = np.asarray(window_vecs @ baseline_vec, dtype=np.float64).ravel()
    margins = sims_q - sims_b
    # Python-level count keeps stub-untracked ndarray ops out; n is small (~tens of windows)
    values = margins.tolist()
    coverage = sum(1 for m in values if m > 0.0) / len(values) if values else 0.0
    return ContrastScore(median_margin=_median(margins), coverage=coverage)


def rank_by_contrast(
    store: Store,
    *,
    query_vec: np.ndarray[[D]],
    baseline_vec: np.ndarray[[D]],
    k: int,
    exclude: set[str] | None = None,
) -> list[tuple[TrackRow, float]]:
    """Rank every indexed track by sustained-mood contrast; returns top *k*.

    Score = median over the track's windows of (cos(window, query) −
    cos(window, broad-mood baseline)); ties broken by positive-margin
    coverage. The returned score is the median margin.
    """
    excluded = set() if exclude is None else set(exclude)
    qvec = np.asarray(query_vec, dtype=np.float32).ravel()
    bvec = np.asarray(baseline_vec, dtype=np.float32).ravel()
    windows = store.load_windows(store.track_rel_paths())

    scored: list[tuple[TrackRow, float, float, str]] = []
    for rel_path in store.track_rel_paths():
        if rel_path in excluded:
            continue
        track = store.get_track(rel_path)
        if track is None:
            continue
        window_vecs = windows.get(rel_path)
        if window_vecs is None or window_vecs.shape[0] == 0:
            continue  # track without stored windows must be re-indexed
        result = contrast_score(window_vecs, query_vec=qvec, baseline_vec=bvec)
        scored.append((track, result.median_margin, result.coverage, rel_path))

    scored.sort(key=lambda entry: (-entry[1], -entry[2], entry[3]))
    return [(track, margin) for track, margin, _coverage, _rel in scored[:k]]
