"""Caption-vocab calibration ranker (lever #5, CAF-Score style).

Per window, the query's affinity is calibrated against the window's own
affinity profile over the caption vocabulary used by `sp describe`: the
score is the percentile of cos(window, query) among that window's
cos(window, caption) values. This is a self-referential baseline — the bar
is the window's own distribution over the readout captions, not 20
hand-picked mood anchors — so the texture-dependent similarity scale
partially cancels: a window counts when the query fits better than a
typical caption *for that window*.

Track score = mean per-window percentile in [0, 1] (sustained mood), ties
broken by coverage (share of windows above 0.5). Semantics: 0.5 = the query
fits as well as a typical caption for the average window of the track.

Guard: tracks whose best vocab affinity is far below the library norm have
uninformative profiles — their top captions barely describe them (measured
on the live index: ~1% of the library below 0.30 max-vocab-sim; chiptune /
breakcore / sparse-minimal clusters). For those tracks the score falls back
to the 20-anchor margin passed through a sigmoid, keeping the [0, 1] scale
with the same neutral point (margin 0 ↔ percentile 0.5).

Ranking is a full-library scan over stored window vectors; the vocab must be
re-embedded per run (215 captions, a few seconds), no re-index needed.
"""

from dataclasses import dataclass

import numpy as np
from shape_extensions import IntVar

from local_smart_playlist.embed.model import MuLanEmbedder
from local_smart_playlist.index.store import Store, TrackMeta
from local_smart_playlist.numpy_helpers import gt, l2_normalize

W = IntVar("W")  # window count
D = IntVar("D")  # embedding dim
V = IntVar("V")  # vocab size

# Cutoff floor in percentile space: 0.5 = query fits as well as a typical
# caption. Scores are absolute (calibrated), so the CLI cutoff is an absolute
# threshold too (min_score, default 0.9); the floor only blocks sub-neutral
# values — nothing below neutral is ever "on-mood".
VOCAB_SCORE_FLOOR = 0.5
# Tracks below this best-vocab-sim have uninformative profiles; measured
# library p1 is 0.294 (see difficulties.md §7).
VOCAB_COVER_FLOOR = 0.30
# Sigmoid temperature for the margin fallback; +0.10 margin → ~0.73.
MARGIN_TAU = 0.05
# Neutral point shared by percentile space (query beats half the captions)
# and sigmoid space (margin 0).
NEUTRAL = 0.5


def vocab_vector_bank(model: MuLanEmbedder[D], vocab: list[str]) -> np.ndarray[[V, D]]:
    """Unit-norm embeddings of the caption vocabulary, one row per caption."""
    vecs = np.asarray(model.embed_texts(vocab), dtype=np.float32)
    return l2_normalize(vecs)


@dataclass(frozen=True)
class VocabCalScore:
    """Calibrated sustained-mood fit: mean percentile plus coverage."""

    score: float
    coverage: float


def vocab_calibration_score(
    window_vecs: np.ndarray[[W, D]],
    *,
    query_vec: np.ndarray[[D]],
    vocab_vecs: np.ndarray[[V, D]],
    margin_vec: np.ndarray[[D]],
) -> VocabCalScore:
    """Percentile of the query affinity within each window's vocab profile.

    Score = mean over windows of (share of vocab captions the query beats),
    or, when the track's best vocab affinity is below ``VOCAB_COVER_FLOOR``,
    a sigmoid of the mean 20-anchor margin (fallback for uninformative
    profiles). Coverage = share of windows above the 0.5 neutral point.
    """
    sims_q = (window_vecs @ query_vec).astype(np.float64)
    sims_v = (window_vecs @ vocab_vecs.T).astype(np.float64)
    best_vocab = float(sims_v.max())
    if best_vocab < VOCAB_COVER_FLOOR:
        sims_m = (window_vecs @ margin_vec).astype(np.float64)
        margins: np.ndarray[[W, 1]] = sims_q[:, None] - sims_m[:, None]
        z = margins.mean() / MARGIN_TAU
        score = float(1.0 / (1.0 + np.exp(-z)))
        coverage = float(gt(margins, 0.0).mean())
    else:
        # pyrefly: ignore [bad-assignment, unsupported-operation]
        beaten: np.ndarray[[W, V]] = sims_q[:, None] > sims_v
        percentiles: np.ndarray[[W, 1]] = beaten.mean(axis=1)[:, None]
        score = float(percentiles.mean())
        coverage = float(gt(percentiles, NEUTRAL).mean())
    return VocabCalScore(score=score, coverage=coverage)


def rank_by_vocab_calibration(
    store: Store,
    *,
    query_vec: np.ndarray[[D]],
    vocab_vecs: np.ndarray[[V, D]],
    margin_vec: np.ndarray[[D]],
    k: int,
    exclude: set[str] | None = None,
) -> list[tuple[TrackMeta, float]]:
    """Rank every indexed track by vocab-calibrated sustained mood; returns top *k*.

    Mirrors :func:`rank_by_contrast`: full scan over stored window vectors,
    ties broken by coverage then rel path. The returned score is the mean
    per-window percentile (or fallback sigmoid margin for guard tracks).
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
        result = vocab_calibration_score(window_vecs, query_vec=query_vec, vocab_vecs=vocab_vecs, margin_vec=margin_vec)
        scored.append((track, result.score, result.coverage, track.rel_path))

    scored.sort(key=lambda entry: (-entry[1], -entry[2], entry[3]))
    return [(track, score) for track, score, _coverage, _rel in scored[:k]]
