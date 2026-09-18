"""Offline eval of the vocab-calibration ranker over the live index.

Read-only. Reports, per probe query: exemplar track scores, library score
quantiles / pass rates at the production cutoff, and a top preview.
"""

import argparse
from pathlib import Path

import numpy as np

from local_smart_playlist.embed.model import MODEL_ID, cap_torch_threads, embed_texts
from local_smart_playlist.index.store import Store, TrackRow
from local_smart_playlist.query.contrast import baseline_vector, query_vector_contrast
from local_smart_playlist.query.prompts import caption_vocab
from local_smart_playlist.query.vocab_cal import (
    VOCAB_COVER_FLOOR,
    VOCAB_SCORE_FLOOR,
    rank_by_vocab_calibration,
    vocab_vector_bank,
)

PROBES: dict[str, list[str]] = {
    "dreamy, melancholic": ["To Far Shores", "The Terminal Show"],
    "calm piano": ["Celeste"],
}
PREVIEW_COUNT = 8


def _matches(track: TrackRow, needle: str) -> bool:
    needle_l = needle.lower()
    return needle_l in track.rel_path.lower() or needle_l in track.title.lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("queries", nargs="*", help="probe queries (default: built-in PROBES)")
    _ = parser.add_argument("--db", type=Path, default=Path(".sp-index/index.db"))
    args = parser.parse_args()

    cap_torch_threads()
    if not args.db.is_file():
        raise SystemExit(f"no index at {args.db}; run `sp index` first")

    probes: dict[str, list[str]] = {q: [] for q in args.queries} if args.queries else PROBES
    vocab = caption_vocab()
    with Store(args.db) as store:
        store.require_model(MODEL_ID)
        vocab_vecs = vocab_vector_bank(embed_texts, vocab)
        print(f"# db={args.db}  vocab={len(vocab)}  cover-floor={VOCAB_COVER_FLOOR}")

        for query, exemplars in probes.items():
            qvec = query_vector_contrast(query, embed_texts)
            bvec = baseline_vector(embed_texts)
            ranked = rank_by_vocab_calibration(
                store, query_vec=qvec, vocab_vecs=vocab_vecs, margin_vec=bvec, k=store.track_count()
            )
            total = len(ranked)
            scores = [s for _, s in ranked]
            arr = np.asarray(scores, dtype=np.float64)
            qs = np.percentile(arr, (50, 75, 90, 95)).tolist()
            cutoff = max(VOCAB_SCORE_FLOOR, 0.9) if total > 0 else 0.0
            passing = sum(1 for s in scores if s >= cutoff)
            print(f"\n== {query!r} — {total} ranked tracks")
            print("   score quantiles 50/75/90/95: " + " / ".join(f"{v:.3f}" for v in qs))
            print(f"   production cutoff {cutoff:.3f} -> {passing} tracks ({passing / total * 100:.1f}%)")
            for needle in exemplars:
                hits = [(i, s) for i, (t, s) in enumerate(ranked) if _matches(t, needle)]
                if len(hits) == 0:
                    print(f"   exemplar {needle!r}: NOT FOUND")
                    continue
                shown = ", ".join(f"rank {i + 1} ({s:.3f})" for i, s in hits[:3])
                print(f"   exemplar {needle!r}: {shown}")
            print("   top preview:")
            for track, score in ranked[:PREVIEW_COUNT]:
                print(f"     {score:+.3f}  {track.title[:40]}")


if __name__ == "__main__":
    main()
