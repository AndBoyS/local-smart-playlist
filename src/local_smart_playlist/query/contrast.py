"""Margin scoring and the broad-mood anchor bank (guard/eval components).

The additive-margin ranker described here was the production ranker until it
was replaced by the caption-vocab calibration (``query/vocab_cal.py``). Its
pieces remain live:

- **MOOD_ANCHORS / baseline_vector**: the 20 broad-mood anchor bank and its
  unnormalized mean — used as the baseline in the guard fallback of the
  vocab-calibration ranker (tracks with uninformative vocab profiles).
- **query_vector_contrast**: unit embedding of the query text — still how
  text queries become vectors.
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

from local_smart_playlist.embed.model import MuLanEmbedder
from local_smart_playlist.index.store import Store, TrackMeta
from local_smart_playlist.numpy_helpers import l2_normalize

W = IntVar("W")  # window count
D = IntVar("D")  # embedding dim

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


def baseline_vector(model: MuLanEmbedder[D], *, anchors: list[str] | None = None) -> np.ndarray[[D]]:
    """Unnormalized mean of unit broad-mood anchor embeddings.

    Kept unnormalized so ``window @ baseline`` equals the *mean of the
    window's per-anchor cosines* — its affinity to music-in-general. A unit
    centroid would inflate the baseline by 1/‖centroid‖ (~2–3x for a diverse
    anchor set) and bury gentle textures under the broad baseline.
    """
    vecs = l2_normalize(model.embed_texts(list(MOOD_ANCHORS if anchors is None else anchors)))
    return vecs.mean(axis=0)


def query_vector_contrast(mood: str, model: MuLanEmbedder[D]) -> np.ndarray[[D]]:
    """Unit embedding of the query text, as written.

    No caption-style rephrasing (`"{q} mood."`): appending "mood" moved
    comma-attribute queries off soft textures (TFS margin 0.120 → 0.040,
    difficulties.md §2), so the query embeds exactly as the user (or the
    LLM pass-through) supplied it.
    """
    vecs = l2_normalize(model.embed_texts([mood]))
    return vecs[0]


@dataclass(frozen=True)
class ContrastScore:
    """Sustained-mood fit of one track: median margin plus window coverage."""

    median_margin: float
    coverage: float


def _median(values: np.ndarray[[W]]) -> float:
    """Median without np.median (shape stubs track the module form poorly)."""
    ordered = np.asarray(values, dtype=np.float64).copy()
    # asarray kept: `values` arrives typed as ndarray but may be a list at call sites
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
    sims_q = (window_vecs @ query_vec).astype(np.float64).ravel()
    sims_b = (window_vecs @ baseline_vec).astype(np.float64).ravel()
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
) -> list[tuple[TrackMeta, float]]:
    """Rank every indexed track by sustained-mood contrast; returns top *k*.

    Score = median over the track's windows of (cos(window, query) −
    cos(window, broad-mood baseline)); ties broken by positive-margin
    coverage. The returned score is the median margin.
    """
    excluded = set() if exclude is None else set(exclude)
    windows = store.load_windows(store.track_rel_paths())

    scored: list[tuple[TrackMeta, float, float, str]] = []
    for track in store.track_meta():
        if track.rel_path in excluded:
            continue
        window_vecs = windows.get(track.rel_path)
        if window_vecs is None or window_vecs.shape[0] == 0:
            continue  # track without stored windows must be re-indexed
        result = contrast_score(window_vecs, query_vec=query_vec, baseline_vec=baseline_vec)
        scored.append((track, result.median_margin, result.coverage, track.rel_path))

    scored.sort(key=lambda entry: (-entry[1], -entry[2], entry[3]))
    return [(track, margin) for track, margin, _coverage, _rel in scored[:k]]
