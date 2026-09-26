"""`sp index` command."""

import datetime
from pathlib import Path

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from local_smart_playlist.audio import decode
from local_smart_playlist.audio.windowing import slice_windows
from local_smart_playlist.config import default_db_path
from local_smart_playlist.embed.aggregate import aggregate
from local_smart_playlist.embed.model import MODEL_ID, cap_torch_threads, load_model
from local_smart_playlist.index.library import discover_audio, display_title, relative_posix
from local_smart_playlist.index.store import MetaKey, Store


class IndexArgs:
    """Typed via main.py; this module exposes the implementation."""

    @staticmethod
    def run(*, root: Path, db: Path | None, rescan: bool, quiet: bool) -> None:
        cap_torch_threads()
        db_path = db if db is not None else default_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: PTH110
        root = root.expanduser().resolve()
        if not root.is_dir():
            raise SystemExit(f"not a directory: {root}")

        files = discover_audio(root)
        if len(files) == 0:
            raise SystemExit(f"no audio files under {root}")

        with Store(db_path) as store:
            store.set_meta(MetaKey.LIBRARY_ROOT, str(root))
            store.set_meta(MetaKey.MODEL, MODEL_ID)
            todo_paths = [p for p in files if rescan or not store.has_track(relative_posix(root, p))]

            # Fail fast on model loading (download/device problems) instead of
            # recording every track as a failure below.
            model = load_model()

            with Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                disable=quiet,
            ) as bar:
                task = bar.add_task("indexing", total=len(todo_paths))
                now = datetime.datetime.now().isoformat(timespec="seconds")
                for path in todo_paths:
                    _ = bar.advance(task, 0)
                    rel = relative_posix(root, path)
                    try:
                        samples, duration = decode.decode_mono(path)
                        windows = slice_windows(samples)
                        if len(windows) == 0:
                            store.record_failure(rel_path=rel, error="no usable windows (too short or silent)", now=now)
                            _ = bar.advance(task, 1)
                            continue
                        window_vecs = model.embed_windows(windows)
                        mean_vec, p90_vec = aggregate(window_vecs)
                        store.upsert(
                            rel_path=rel,
                            mean_vec=mean_vec,
                            p90_vec=p90_vec,
                            n_windows=len(windows),
                            duration=duration,
                            title=display_title(root, path),
                            model=MODEL_ID,
                            indexed_at=now,
                            window_vecs=window_vecs,
                        )
                    except decode.DecodeError as exc:
                        store.record_failure(rel_path=rel, error=str(exc), now=now)
                    except Exception as exc:  # noqa: BLE001 — one bad track must not abort a full index run
                        store.record_failure(rel_path=rel, error=str(exc), now=now)
                    _ = bar.advance(task, 1)
            _ = store.prune_missing({relative_posix(root, p) for p in files})

            indexed = store.track_count()
            failed = store.failure_count()
            print(f"indexed {indexed}/{len(files)} tracks, {failed} failures -> {db_path}")  # noqa: T201 — CLI output
