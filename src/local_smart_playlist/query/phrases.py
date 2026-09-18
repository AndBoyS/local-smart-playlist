"""Per-track phrase documents and document-retrieval ranking.

Task: "find documents that have relevant text to query". A track's documents
are its top caption-vocab phrases (the `sp describe` readout, extracted from
the track mean vector). `sp play` embeds the query once and ranks tracks by
the best text-to-text cosine between the query and the track's documents.
"""

from __future__ import annotations

import hashlib

import numpy as np
from shape_extensions import IntVar

from local_smart_playlist.index.store import Store, TrackRow
from local_smart_playlist.query import prompts
from local_smart_playlist.query.search import TextEmbedder

V = IntVar("V")
D = IntVar("D")

DOCS_PER_TRACK = 10  # phrases kept per track
DOC_VOCAB_SHA_KEY = "doc_vocab_sha"  # meta key guarding stored documents


def vocab_fingerprint(vocab: list[str]) -> str:
    """Stable fingerprint of the readout vocabulary backing stored documents."""
    return hashlib.sha256("\n".join(vocab).encode("utf-8")).hexdigest()


def track_documents(
    mean_vec: np.ndarray[[D]], vocab_vecs: np.ndarray[[V, D]], *, top_n: int = DOCS_PER_TRACK
) -> list[tuple[int, float]]:
    """(vocab_index, sim) for the *top_n* vocab phrases nearest to a unit track-mean vector."""
    sims = np.asarray(vocab_vecs @ np.asarray(mean_vec, dtype=np.float32), dtype=np.float64).ravel()
    count = min(top_n, vocab_vecs.shape[0])
    order = np.argsort(-sims)[:count]
    return [(int(i), float(sims[int(i)])) for i in order]


def ensure_documents(
    store: Store,
    embedder: TextEmbedder,
    *,
    vocab: list[str] | None = None,
    top_n: int = DOCS_PER_TRACK,
) -> np.ndarray[[V, D]]:
    """Backfill missing per-track phrase documents; returns unit vocab vectors.

    Wipes stored documents when the vocabulary fingerprint changes.
    """
    phrases = prompts.caption_vocab() if vocab is None else list(vocab)
    fingerprint = vocab_fingerprint(phrases)
    if store.get_meta(DOC_VOCAB_SHA_KEY) != fingerprint:
        store.clear_documents()
        store.set_meta(DOC_VOCAB_SHA_KEY, fingerprint)
    vocab_vecs = np.asarray(embedder(phrases), dtype=np.float32)
    have = store.tracks_with_documents()
    for rel_path, mean_vec in store.all_track_means():
        if rel_path not in have:
            store.set_documents(rel_path, track_documents(mean_vec, vocab_vecs, top_n=top_n))
    return vocab_vecs


def _score_of(entry: tuple[TrackRow, float]) -> float:
    return entry[1]


def rank_by_documents(
    store: Store,
    *,
    query_vec: np.ndarray[[D]],
    vocab_vecs: np.ndarray[[V, D]],
    k: int,
    exclude: set[str] | None = None,
) -> list[tuple[TrackRow, float]]:
    """Rank tracks by document relevance: max cos(query, track phrase document)."""
    excluded = set() if exclude is None else set(exclude)
    qvec = np.asarray(query_vec, dtype=np.float64).ravel()
    vocab = np.asarray(vocab_vecs, dtype=np.float64)
    scored: list[tuple[TrackRow, float]] = []
    for rel_path, doc_indices in store.all_document_indices().items():
        if rel_path in excluded:
            continue
        track = store.get_track(rel_path)
        if track is None:
            continue
        rows = vocab[np.asarray(doc_indices, dtype=np.intp)]
        sims = np.asarray(rows @ qvec, dtype=np.float64).ravel()
        scored.append((track, float(sims.max())))
    scored.sort(key=_score_of, reverse=True)
    return scored[:k]
