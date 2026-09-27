"""`sp play` command."""

import logging
from collections import defaultdict
from pathlib import Path

from shape_extensions import IntVar

from local_smart_playlist.commands import vacuum_worker
from local_smart_playlist.config import default_db_path, default_playlist_dir
from local_smart_playlist.embed.model import MODEL_ID, cap_torch_threads, load_model
from local_smart_playlist.index.store import MetaKey, Store, TrackData
from local_smart_playlist.query.contrast import baseline_vector_cached, query_vector_contrast
from local_smart_playlist.query.playlist import playlist_path, write_playlist
from local_smart_playlist.query.prompts import adapt_mood, caption_vocab
from local_smart_playlist.query.search import rank_by_similarity
from local_smart_playlist.query.vocab_cal import (
    VOCAB_SCORE_FLOOR,
    rank_by_vocab_calibration,
    vocab_vector_bank_cached,
)

logger = logging.getLogger(__name__)

PREVIEW_COUNT = 10
PLAY_DEFAULT_ALPHA = 0.7  # peak-window weight default for --seed-track ranking


def merge_query_rankings[D: IntVar](
    rankings: list[list[tuple[TrackData[D], float]]],
) -> list[tuple[TrackData[D], float]]:
    """Union per-query rankings; keep each track's best score across queries."""
    best_scores: defaultdict[str, float] = defaultdict(lambda: float("-inf"))
    best_tracks: dict[str, TrackData[D]] = {}
    for ranking in rankings:
        for track, score in ranking:
            if score > best_scores[track.rel_path]:
                best_scores[track.rel_path] = score
                best_tracks[track.rel_path] = track
    merged = [(best_tracks[path], score) for path, score in best_scores.items()]
    return sorted(merged, key=lambda item: item[1], reverse=True)


class PlayArgs:
    """Typed via main.py; this module exposes the implementation."""

    @staticmethod
    def run(
        *,
        query: str,
        db: Path | None,
        n: int | None,
        out: str | None,
        llm: bool,
        seed_track: str | None,
        alpha: float,
        min_score: float,
        dry: bool,
    ) -> None:
        cap_torch_threads()
        db_path = db if db is not None else default_db_path()
        if not db_path.is_file():
            raise SystemExit(f"no index at {db_path}; run `sp index` first")

        library_root: str | None = None
        vacuum_recommended = False
        with Store(db_path) as store:
            library_root = store.get_meta(MetaKey.LIBRARY_ROOT)
            store.require_model(MODEL_ID)
            exclude: set[str] = set()
            if seed_track is not None:
                seed = store.get_track(seed_track)
                if seed is None:
                    raise SystemExit(f"seed track not in index: {seed_track}")
                exclude.add(seed.rel_path)
                seed_vectors = seed.vectors
                if seed_vectors is None:
                    raise RuntimeError(f"stored track has no vectors: {seed_track}")
                ranked = rank_by_similarity(
                    store,
                    seed_vectors.mean_vec,
                    k=n if n is not None else store.track_count(),
                    exclude=exclude,
                    alpha=alpha,
                )
            else:
                if alpha != PLAY_DEFAULT_ALPHA:
                    print("note: --alpha ignored (applies only with --seed-track)")
                adapted = [query]
                if llm:
                    adapted = adapt_mood(query)
                    print("LLM queries:")
                    for variant in adapted:
                        print(f"  - {variant}")
                model = load_model(device="cpu", text_only=True)
                bvec = baseline_vector_cached(store, model)
                vocab_vecs = vocab_vector_bank_cached(store, model, vocab=caption_vocab())
                per_query_rankings = [
                    rank_by_vocab_calibration(
                        store,
                        query_vec=query_vector_contrast(variant, model),
                        vocab_vecs=vocab_vecs,
                        margin_vec=bvec,
                        k=store.track_count(),
                        exclude=exclude,
                    )
                    for variant in adapted
                ]
                ranked = merge_query_rankings(per_query_rankings)
            vacuum_recommended = store.vacuum_if_needed(execute=False).recommended

        if vacuum_recommended and vacuum_worker.start_background(db_path):
            logger.info("started background index vacuum; details: %s", f"{db_path}.vacuum.log")

        cutoff = max(VOCAB_SCORE_FLOOR, min_score)
        if len(ranked) > 0:
            # Calibrated score is absolute: 0.5 = query fits as well as a typical
            # caption (neutral), 0.9+ = clearly on-mood. No best-relative cutoff.
            ranked = [(t, s) for t, s in ranked if s >= cutoff]
            if n is not None:
                ranked = ranked[:n]
        if len(ranked) == 0:
            raise SystemExit(
                f"no tracks passed the cutoff (effective {cutoff:.2f} = max({VOCAB_SCORE_FLOOR}, "
                f"--min-score {min_score:.2f})); scores below {VOCAB_SCORE_FLOOR} are below "
                "neutral by construction — lower --min-score toward 0.5 or refine the query"
            )

        root = Path(library_root) if library_root is not None else Path.cwd()
        entries: list[tuple[Path, str, float, float]] = [
            (root / t.rel_path, t.title, t.duration, score) for t, score in ranked
        ]

        if dry is True or out == "-":
            for _path, title, duration, score in entries:
                print(f"{score:6.3f}  {duration:7.1f}s  {title}")
            return

        out_path = Path(out) if out is not None else playlist_path(query, default_playlist_dir(root))
        write_playlist(
            [(str(p), title, dur) for p, title, dur, _ in entries],
            root=root,
            out_path=out_path,
        )
        print(f"wrote {len(entries)} tracks -> {out_path}")
        for _path, title, _dur, score in entries[:PREVIEW_COUNT]:
            print(f"  {score:6.3f}  {title}")
        if len(entries) > PREVIEW_COUNT:
            print(f"  ... +{len(entries) - PREVIEW_COUNT} more")
