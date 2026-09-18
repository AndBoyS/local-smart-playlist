"""`sp play` command."""

from pathlib import Path

from local_smart_playlist.config import default_db_path, default_playlist_dir
from local_smart_playlist.embed.model import MODEL_ID, cap_torch_threads, embed_texts
from local_smart_playlist.index.store import Store
from local_smart_playlist.query.contrast import baseline_vector, query_vector_contrast
from local_smart_playlist.query.playlist import playlist_path, write_playlist
from local_smart_playlist.query.prompts import LlmError, adapt_mood, caption_vocab
from local_smart_playlist.query.search import rank_by_similarity
from local_smart_playlist.query.vocab_cal import (
    VOCAB_SCORE_FLOOR,
    rank_by_vocab_calibration,
    vocab_vector_bank,
)

PREVIEW_COUNT = 10


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
        with Store(db_path) as store:
            library_root = store.get_meta("library_root")
            store.require_model(MODEL_ID)
            exclude: set[str] = set()
            if seed_track is not None:
                seed = store.get_track(seed_track)
                if seed is None:
                    raise SystemExit(f"seed track not in index: {seed_track}")
                exclude.add(seed.rel_path)
                ranked = rank_by_similarity(
                    store, seed.mean_vec, k=n if n is not None else store.track_count(), exclude=exclude, alpha=alpha
                )
            else:
                adapted = query
                if llm:
                    try:
                        adapted = adapt_mood(query)
                    except LlmError:
                        adapted = query  # offline fallback: embed the raw phrase
                qvec = query_vector_contrast(adapted, embed_texts)
                bvec = baseline_vector(embed_texts)
                vocab_vecs = vocab_vector_bank(embed_texts, caption_vocab())
                ranked = rank_by_vocab_calibration(
                    store,
                    query_vec=qvec,
                    vocab_vecs=vocab_vecs,
                    margin_vec=bvec,
                    k=n if n is not None else store.track_count(),
                    exclude=exclude,
                )

        if len(ranked) > 0:
            # Calibrated score is absolute: 0.5 = query fits as well as a typical
            # caption (neutral), 0.9+ = clearly on-mood. No best-relative cutoff.
            cutoff = max(VOCAB_SCORE_FLOOR, min_score)
            ranked = [(t, s) for t, s in ranked if s >= cutoff]
        if len(ranked) == 0:
            raise SystemExit("no tracks passed the score cutoff; lower --min-score")

        root = Path(library_root) if library_root is not None else Path.cwd()
        entries: list[tuple[Path, str, float, float]] = [
            (root / t.rel_path, t.title, t.duration, score) for t, score in ranked
        ]

        if dry is True or out == "-":
            for _path, title, duration, score in entries:
                print(f"{score:6.3f}  {duration:7.1f}s  {title}")  # noqa: T201 — CLI output
            return

        out_path = Path(out) if out is not None else playlist_path(query, default_playlist_dir(root))
        write_playlist(
            [(str(p), title, dur) for p, title, dur, _ in entries],
            root=root,
            out_path=out_path,
        )
        print(f"wrote {len(entries)} tracks -> {out_path}")  # noqa: T201 — CLI output
        for _path, title, _dur, score in entries[:PREVIEW_COUNT]:
            print(f"  {score:6.3f}  {title}")  # noqa: T201 — CLI output
        if len(entries) > PREVIEW_COUNT:
            print(f"  ... +{len(entries) - PREVIEW_COUNT} more")  # noqa: T201 — CLI output
