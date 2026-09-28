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

Ranking is a full-library scan over stored window vectors; the vocab must be
re-embedded per run (215 captions, a few seconds), no re-index needed.
"""

import base64
import hashlib
from dataclasses import dataclass

import numpy as np
from shape_extensions import IntVar

from local_smart_playlist.audio.features import batches
from local_smart_playlist.embed.model import MODEL_ID, MuLanEmbedder
from local_smart_playlist.index.store import MetaKey, Store, TrackData
from local_smart_playlist.numpy_helpers import l2_normalize, reshape
from local_smart_playlist.type_utils import NonEmptyTuple

W = IntVar("W")  # window count per track
N = IntVar("N")  # total window count in a flat scoring batch
D = IntVar("D")  # embedding dim
V = IntVar("V")  # vocab size

# Cutoff floor in percentile space: 0.5 = query fits as well as a typical
# caption. Scores are absolute (calibrated), so the CLI cutoff is an absolute
# threshold too (min_score, default 0.9); the floor only blocks sub-neutral
# values — nothing below neutral is ever "on-mood".
VOCAB_SCORE_FLOOR = 0.5
CALIBRATION_BATCH_TRACKS = 256
# A window counts toward coverage when it beats more than half of caption prompts.
NEUTRAL = 0.5


def vocab_vector_bank(model: MuLanEmbedder[D], vocab: NonEmptyTuple[str]) -> np.ndarray[[V, D]]:
    """Unit-norm embeddings of the caption vocabulary, one row per caption."""
    embeds = model.embed_texts(vocab)
    return l2_normalize(embeds)


_CACHE_FORMAT = "v1"
_CACHE_PART_COUNT = 4


def _vocab_digest(vocab: NonEmptyTuple[str]) -> str:
    """Digest vocabulary text for cache validation."""
    return hashlib.sha1("\n".join(vocab).encode("utf-8")).hexdigest()


def _encode_bank(vecs: np.ndarray[[V, D]], *, vocab_digest: str) -> str:
    """Encode model, vocab identity, and float32 vectors into meta TEXT value."""
    encoded_vecs = base64.b64encode(np.ascontiguousarray(vecs, dtype=np.float32).tobytes()).decode("ascii")
    return "\n".join((_CACHE_FORMAT, MODEL_ID, vocab_digest, encoded_vecs))


def _decode_bank(*, blob: str, dim: int, n: int, vocab_digest: str) -> np.ndarray[[V, D]] | None:
    """Decode cache value; return None when identity, encoding, or shape mismatches."""
    parts = blob.split("\n", maxsplit=3)
    if len(parts) != _CACHE_PART_COUNT or parts[0] != _CACHE_FORMAT or parts[1] != MODEL_ID or parts[2] != vocab_digest:
        return None
    try:
        arr = np.frombuffer(base64.b64decode(parts[3], validate=True), dtype=np.float32)
    except ValueError:
        return None
    if arr.size != n * dim:
        return None
    return reshape(arr, (n, dim))


def vocab_vector_bank_cached(store: Store, model: MuLanEmbedder[D], *, vocab: NonEmptyTuple[str]) -> np.ndarray[[V, D]]:
    """Unit-norm vocab embeddings, cached under one fixed metadata key.

    Cache value includes format, model, and vocabulary digest; changes silently
    invalidate the prior value. Decoding also validates row count and dimension.
    """
    digest = _vocab_digest(vocab)
    cached = store.get_meta(MetaKey.VOCAB_VECS)
    if cached is not None:
        vecs = _decode_bank(blob=cached, dim=model.dim, n=len(vocab), vocab_digest=digest)
        if vecs is not None:
            return vecs
    vecs = vocab_vector_bank(model, vocab)
    store.set_meta(MetaKey.VOCAB_VECS, _encode_bank(vecs, vocab_digest=digest))
    return vecs


@dataclass(frozen=True)
class VocabCalScore:
    """Calibrated sustained-mood fit: mean percentile plus coverage."""

    score: float
    coverage: float


def vocab_calibration_score(
    flat_vecs: np.ndarray[[N, D]],
    *,
    track_sizes: NonEmptyTuple[int] | None = None,
    query_vec: np.ndarray[[D]],
    vocab_vecs: np.ndarray[[V, D]],
) -> list[VocabCalScore]:
    """Score a flat batch of tracks; ``track_sizes`` partitions its windows.

    Omit ``track_sizes`` to treat all rows as one track. Otherwise, sizes must
    be positive and sum to the number of rows in ``window_vecs``.
    """
    sizes = [flat_vecs.shape[0]] if track_sizes is None else list(track_sizes)
    if any(size <= 0 for size in sizes) or sum(sizes) != flat_vecs.shape[0]:
        raise ValueError("track_sizes must partition window_vecs into positive track sizes")

    sims_q = flat_vecs @ query_vec
    sims_v = flat_vecs @ vocab_vecs.T
    beaten = np.greater(sims_q[:, None], sims_v)
    percentiles = beaten.mean(axis=1)
    track_ids = np.repeat(np.arange(len(sizes)), sizes)
    track_counts = np.asarray(sizes)
    score_sums = np.bincount(track_ids, weights=percentiles, minlength=len(sizes))
    coverage_sums = np.bincount(
        track_ids,
        weights=np.greater(percentiles, NEUTRAL),
        minlength=len(sizes),
    )
    scores: list[float] = (score_sums / track_counts).tolist()
    coverages: list[float] = (coverage_sums / track_counts).tolist()
    results: list[VocabCalScore] = []
    for score, coverage in zip(scores, coverages, strict=True):
        results.append(VocabCalScore(score=score, coverage=coverage))
    return results


def rank_by_vocab_calibration(
    store: Store,
    *,
    query_vec: np.ndarray[[D]],
    vocab_vecs: np.ndarray[[V, D]],
    k: int,
    exclude: set[str] | None = None,
) -> list[tuple[TrackData[D], float]]:
    """Rank every indexed track by vocab-calibrated sustained mood; returns top *k*.

    Full scan over stored window vectors, ties broken by coverage then rel
    path. Every window uses its caption-vocabulary percentile, including
    windows with low affinity to every caption.
    """
    excluded = set() if exclude is None else set(exclude)
    flat_vecs = store.load_windows_flat(store.track_rel_paths())

    path_data = {track.rel_path: track for track in store.track_meta()}
    scored: list[tuple[TrackData[D], float, float, str]] = []
    batch_start = 0
    flat_id_start = 0

    if len(flat_vecs.rel_paths) == 0:
        return []

    paths = NonEmptyTuple(flat_vecs.rel_paths)
    for paths_batch in batches(paths, batch_size=CALIBRATION_BATCH_TRACKS):
        batch_end = batch_start + len(paths_batch)
        sizes_batch = NonEmptyTuple(flat_vecs.window_sizes[batch_start:batch_end])
        flat_id_end = flat_id_start + sum(sizes_batch)
        batch_scores = vocab_calibration_score(
            flat_vecs.vec_matrix[flat_id_start:flat_id_end],
            track_sizes=sizes_batch,
            query_vec=query_vec,
            vocab_vecs=vocab_vecs,
        )
        for path, result in zip(paths_batch, batch_scores, strict=True):
            track = path_data.get(path)
            if path not in excluded and track is not None:
                scored.append((track, result.score, result.coverage, path))
        batch_start = batch_end
        flat_id_start = flat_id_end

    scored.sort(key=lambda entry: (-entry[1], -entry[2], entry[3]))
    return [(track, score) for track, score, _coverage, _rel in scored[:k]]
