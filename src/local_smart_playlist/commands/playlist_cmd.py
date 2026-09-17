"""`sp play` command."""

from __future__ import annotations

from pathlib import Path

from local_smart_playlist.config import default_db_path, default_playlist_dir
from local_smart_playlist.embed.model import cap_torch_threads, embed_texts
from local_smart_playlist.index.store import Store
from local_smart_playlist.query.playlist import playlist_path, write_playlist
from local_smart_playlist.query.search import query_vector, rank_by_similarity, rank_hybrid

PREVIEW_COUNT = 10


class PlayArgs:
    """Typed via main.py; this module exposes the implementation."""

    @staticmethod
    def run(
        *,
        query: str,
        db: Path | None,
        n: int,
        out: str | None,
        llm: bool,
        seed_track: str | None,
        alpha: float,
        dry: bool,
    ) -> None:
        cap_torch_threads()
        db_path = db if db is not None else default_db_path()
        if not db_path.is_file():
            raise SystemExit(f"no index at {db_path}; run `sp index` first")

        library_root: str | None = None
        with Store(db_path) as store:
            library_root = store.get_meta("library_root")
            exclude: set[str] = set()
            if seed_track is not None:
                seed = store.get_track(seed_track)
                if seed is None:
                    raise SystemExit(f"seed track not in index: {seed_track}")
                exclude.add(seed.rel_path)
                ranked = rank_by_similarity(store, seed.mean_vec, k=n, exclude=exclude, alpha=alpha)
            else:
                qvec = query_vector(query, embed_texts, use_llm=llm)
                ranked = rank_hybrid(store, qvec, k=n, exclude=exclude, alpha=alpha)

        if len(ranked) == 0:
            raise SystemExit("no indexed tracks matched")

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
