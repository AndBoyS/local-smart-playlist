"""`sp describe` command."""

from pathlib import Path

import numpy as np
from rich import print as rprint
from rich.table import Table

from local_smart_playlist.config import default_db_path
from local_smart_playlist.embed.model import MODEL_ID, cap_torch_threads, embed_texts
from local_smart_playlist.index.store import Store
from local_smart_playlist.query import phrases as phrase_docs
from local_smart_playlist.query import prompts

DEFAULT_TOP_N = 10


def resolve_rel_path(raw: str, root: Path) -> str:
    """Normalize *raw* to a POSIX rel path from the library root.

    Accepts an absolute path, a cwd-relative path inside the root, or an
    already-indexed rel path. Falls back to the input as-is when it does not
    live under the root (get_track will surface a clear error).
    """
    absolute = Path(raw).expanduser()
    if not absolute.is_absolute():
        absolute = Path.cwd() / absolute
    try:
        return absolute.relative_to(root).as_posix()
    except ValueError:
        return Path(raw).expanduser().as_posix()


def rank_captions(mean_vec: np.ndarray, vocab: list[str], *, top_n: int) -> list[tuple[str, float]]:
    """Top captions by cosine similarity between the unit-norm track mean vector and vocab embeddings."""
    docs = phrase_docs.track_documents(mean_vec, embed_texts(vocab), top_n=top_n)
    return [(vocab[i], sim) for i, sim in docs]


class DescribeArgs:
    """Typed via main.py; this module exposes the implementation."""

    @staticmethod
    def run(*, path: str, db: Path | None, n: int) -> None:
        cap_torch_threads()
        db_path = db if db is not None else default_db_path()
        if not db_path.is_file():
            raise SystemExit(f"no index at {db_path}; run `sp index` first")

        with Store(db_path) as store:
            library_root = store.get_meta("library_root")
            store.require_model(MODEL_ID)
            if library_root is None:
                raise SystemExit("index has no library root recorded; re-run `sp index`")
            root = Path(library_root).expanduser()
            if not root.is_absolute():
                root = Path.cwd() / root
            rel_path = resolve_rel_path(path, root)
            track = store.get_track(rel_path)
            if track is None:
                raise SystemExit(f"track not in index: {path}")
            captions = rank_captions(track.mean_vec, prompts.caption_vocab(), top_n=n)

        rprint(f"[bold]{track.title}[/bold]  {track.duration:.1f}s  {track.n_windows} windows  {rel_path}")
        table = Table(title="caption-vocab neighbors")
        table.add_column("sim", justify="right", style="cyan")
        table.add_column("caption")
        for caption, sim in captions:
            table.add_row(f"{sim:.3f}", caption)
        rprint(table)
