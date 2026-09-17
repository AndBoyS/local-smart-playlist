"""`sp status` command."""

from __future__ import annotations

from pathlib import Path

from rich import print as rprint
from rich.table import Table

from local_smart_playlist.config import default_db_path
from local_smart_playlist.embed.model import MODEL_ID
from local_smart_playlist.index.store import Store


class StatusArgs:
    """Typed via main.py; this module exposes the implementation."""

    @staticmethod
    def run(*, db: Path | None) -> None:
        db_path = db if db is not None else default_db_path()
        if not db_path.is_file():
            raise SystemExit(f"no index at {db_path}; run `sp index` first")

        with Store(db_path) as store:
            table = Table(title=f"sp index — {db_path}")
            table.add_column("key", style="cyan")
            table.add_column("value")
            lib = store.get_meta("library_root")
            model_meta = store.get_meta("model")
            schema = store.get_meta("schema_version")
            table.add_row("library root", lib if lib is not None else "—")
            table.add_row("model", model_meta if model_meta is not None else MODEL_ID)
            table.add_row("tracks", str(store.track_count()))
            table.add_row("windows", str(store.window_count()))
            table.add_row("failures", str(store.failure_count()))
            table.add_row("schema", schema if schema is not None else "?")
            rprint(table)

            failures = store.list_failures()
            if len(failures) > 0:
                ftable = Table(title="failures")
                ftable.add_column("path", style="red")
                ftable.add_column("error")
                ftable.add_column("tries", justify="right")
                for rel_path, error, attempts in failures:
                    ftable.add_row(rel_path, error, str(attempts))
                rprint(ftable)
