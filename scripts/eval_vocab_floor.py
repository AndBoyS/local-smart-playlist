"""One-off: describe-style caption-vocab sim floor across the library.

For every track: max cos(track mean_vec, caption vocab) — the same metric
`sp describe` prints as its top row. Reports the distribution and the worst
tracks, i.e. edge cases with no good anchor in the vocab (relevant if the
vocab bank replaces/extends the 20 mood anchors).
"""

import numpy as np

from local_smart_playlist.config import default_db_path
from local_smart_playlist.embed.model import MODEL_ID, cap_torch_threads, load_model
from local_smart_playlist.index.store import Store
from local_smart_playlist.query import prompts

WORST_COUNT = 20


def main() -> None:
    cap_torch_threads()
    vocab = prompts.caption_vocab()
    vocab_vecs = np.asarray(load_model().embed_texts(vocab), dtype=np.float32)  # (V, D)
    with Store(default_db_path()) as store:
        store.require_model(MODEL_ID)
        rows = store.all_track_means()
        paths = [r[0] for r in rows]
        means = np.stack([np.asarray(r[1], dtype=np.float32) for r in rows])  # (T, D)
        sims = np.asarray(means @ vocab_vecs.T, dtype=np.float64)  # (T, V)
        best = sims.max(axis=1)
        best_idx = sims.argmax(axis=1)

    order: list[int] = [int(v) for v in np.argsort(best).tolist()]
    qs = np.percentile(best, (0, 1, 5, 10, 25, 50, 75, 95, 100)).tolist()
    print(f"tracks={len(best)}  vocab={len(vocab)}")
    print("max-vocab-sim quantiles min/p1/p5/p10/p25/p50/p75/p95/max:")
    print("  " + " / ".join(f"{v:.3f}" for v in qs))
    print(f"\n{WORST_COUNT} worst-covered tracks (no good anchor in vocab):")
    for i in order[:WORST_COUNT]:
        idx = i
        cap = vocab[best_idx[idx].item()]
        print(f"  {best[idx]:.3f}  {cap[:44]:<44}  {paths[idx][:60]}")


if __name__ == "__main__":
    main()
