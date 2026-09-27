"""Track → caption-vocab phrase readout for `sp describe` (diagnostics only).

Playlist ranking never touches the caption vocabulary; this module only
supports the per-track `sp describe` readout: the top caption phrases nearest
to the track's mean vector.
"""

import numpy as np
from shape_extensions import IntVar

from local_smart_playlist.type_utils import NonEmptyTuple

V = IntVar("V")
D = IntVar("D")

DOCS_PER_TRACK = 10  # phrases shown per track


def track_documents(
    mean_vec: np.ndarray[[D]], vocab_vecs: np.ndarray[[V, D]], *, top_n: int = DOCS_PER_TRACK
) -> NonEmptyTuple[tuple[int, float]]:
    """(vocab_index, sim) for the *top_n* vocab phrases nearest to a unit track-mean vector."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    sims = (vocab_vecs @ np.asarray(mean_vec, dtype=np.float32)).astype(np.float64).ravel()
    count = min(top_n, vocab_vecs.shape[0])
    order = np.argsort(-sims)[:count]
    return NonEmptyTuple((int(i), float(sims[int(i)])) for i in order)
