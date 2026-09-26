"""Detached worker for compacting an idle playlist index."""

import logging
import subprocess
import sys
from pathlib import Path

from local_smart_playlist.index.store import Store

logger = logging.getLogger(__name__)
WORKER_ARGC = 2


def start_background(db_path: Path) -> bool:
    """Launch detached compaction; redirect worker output beside the index DB."""
    log_path = db_path.with_name(f"{db_path.name}.vacuum.log")
    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            _ = subprocess.Popen(
                [sys.executable, Path(__file__).resolve(), db_path.resolve()],
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
            )
    except OSError:
        logger.exception("could not start background vacuum for %s", db_path)
        return False
    return True


def run(db_path: Path) -> None:
    """Recheck thresholds, then let SQLite coordinate any competing DB access."""
    try:
        with Store(db_path) as store:
            result = store.vacuum_if_needed(execute=True)
        if result.performed:
            logger.info(
                "vacuum complete: DB %.1f MiB → %.1f MiB; reclaimed %.1f MiB",
                result.size_before_bytes / (1024 * 1024),
                result.size_after_bytes / (1024 * 1024),
                (result.size_before_bytes - result.size_after_bytes) / (1024 * 1024),
            )
        else:
            logger.info(
                "vacuum skipped: free pages %.1f MiB below thresholds",
                result.free_before_bytes / (1024 * 1024),
            )
    except Exception:
        logger.exception("background vacuum failed for %s", db_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if len(sys.argv) != WORKER_ARGC:
        raise SystemExit("usage: python vacuum_worker.py DB_PATH")
    run(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
