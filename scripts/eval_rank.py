"""Offline ranking eval: probe queries x exemplar tracks over the live index.

Measures where known soft/dense exemplar tracks rank under the current
ranker settings, plus a top-k preview, so ranker experiments (difficulties.md
§1, §6) can be compared without listening. Read-only: never writes the DB or
playlists.

Usage:
    uv run python scripts/eval_rank.py                  # default probes
    uv run python scripts/eval_rank.py "calm piano"     # ad-hoc probe
"""

import argparse
from pathlib import Path

from local_smart_playlist.embed.model import MODEL_ID, MuLanEmbedder, cap_torch_threads, load_model
from local_smart_playlist.index.store import Store, TrackMeta
from local_smart_playlist.query.contrast import (
    baseline_vector,
    query_vector_contrast,
    rank_by_contrast,
)

# Probe query -> case-insensitive substrings of exemplar tracks (rel_path or
# title). Soft/dense exemplars from difficulties.md §1 measurements.
PROBES: dict[str, list[str]] = {
    "dreamy, melancholic": ["To Far Shores", "The Terminal Show"],
    "calm piano": ["Celeste"],
    "dark, aggressive": [],
}
PREVIEW_COUNT = 5
PERCENTILE_SCALE = 100.0


def rank_full(store: Store, model: MuLanEmbedder[512], *, query: str) -> list[tuple[TrackMeta, float]]:
    """Full-library sustained-mood ranking with the production ranker."""
    qvec = query_vector_contrast(query, model)
    bvec = baseline_vector(model)
    return rank_by_contrast(store, query_vec=qvec, baseline_vec=bvec, k=store.track_count())


def report(ranked: list[tuple[TrackMeta, float]], *, query: str, exemplars: list[str]) -> None:
    total = len(ranked)
    print(f"== {query!r} — {total} ranked tracks")
    for needle in exemplars:
        hit = next(
            ((idx, score) for idx, (track, score) in enumerate(ranked) if _matches(track, needle)),
            None,
        )
        if hit is None:
            print(f"   exemplar {needle!r}: NOT IN INDEX (or no stored windows)")
            continue
        rank0, score = hit
        pct = rank0 / total * PERCENTILE_SCALE if total > 0 else 0.0
        print(
            f"   exemplar {needle!r}: rank {rank0 + 1}/{total}"
            f" ({pct:.1f}% from top)  score {score:+.3f}"
        )
    print("   top preview:")
    for track, score in ranked[:PREVIEW_COUNT]:
        print(f"     {score:+.3f}  {track.title}")


def _matches(track: TrackMeta, needle: str) -> bool:
    needle_l = needle.lower()
    return needle_l in track.rel_path.lower() or needle_l in track.title.lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("queries", nargs="*", help="probe queries (default: built-in PROBES)")
    _ = parser.add_argument("--db", type=Path, default=None, help="index DB path")
    args = parser.parse_args()

    cap_torch_threads()
    db_path = args.db if args.db is not None else Path(".sp-index/index.db")
    if not db_path.is_file():
        raise SystemExit(f"no index at {db_path}; run `sp index` first")

    probes: dict[str, list[str]] = {}
    for query in args.queries:
        exemplars: list[str] = []
        probes[query] = exemplars
    if len(probes) == 0:
        probes = PROBES
    with Store(db_path) as store:
        store.require_model(MODEL_ID)
        model = load_model()
        print(f"# db={db_path}")
        for query, exemplars in probes.items():
            ranked = rank_full(store, model, query=query)
            report(ranked, query=query, exemplars=exemplars)


if __name__ == "__main__":
    main()
