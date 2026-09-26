"""SQLite maintenance threshold and compaction tests."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from local_smart_playlist.commands import vacuum_worker
from local_smart_playlist.index import store as store_module
from local_smart_playlist.index.store import Store, should_vacuum


@pytest.mark.parametrize(
    ("free_bytes", "total_bytes", "expected"),
    [
        (300 * 1024 * 1024, 1024 * 1024 * 1024, True),
        (50 * 1024 * 1024, 200 * 1024 * 1024, True),
        (49 * 1024 * 1024, 196 * 1024 * 1024, False),
        (255 * 1024 * 1024, 1024 * 1024 * 1024, False),
        (300 * 1024 * 1024, 2048 * 1024 * 1024, False),
        (1024 * 1024 * 1024, 0, False),
    ],
)
def test_vacuum_threshold(free_bytes: int, total_bytes: int, expected: bool) -> None:
    assert should_vacuum(free_bytes=free_bytes, total_bytes=total_bytes) is expected


def test_vacuum_if_needed_compacts_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store_module, "MIN_VACUUM_FREE_BYTES", 1)
    monkeypatch.setattr(store_module, "MIN_VACUUM_FREE_FRACTION", 0.0)
    with Store(tmp_path / "vacuum.db", embed_dim=64) as store:
        with store._engine.begin() as connection:
            _ = connection.exec_driver_sql("CREATE TABLE vacuum_test_payload (data BLOB NOT NULL)")
            _ = connection.exec_driver_sql(
                "INSERT INTO vacuum_test_payload(data) VALUES (?)",
                [(b"x" * (64 * 1024),) for _ in range(16)],
            )
            _ = connection.exec_driver_sql("DELETE FROM vacuum_test_payload")
        preview = store.vacuum_if_needed(execute=False)
        assert preview.recommended
        assert not preview.performed
        assert preview.free_before_bytes > 0
        assert preview.size_after_bytes == preview.size_before_bytes

        result = store.vacuum_if_needed(execute=True)

        assert result.recommended
        assert result.performed
        assert result.size_after_bytes < result.size_before_bytes


def test_vacuum_worker_script_runs_directly(tmp_path: Path) -> None:
    worker_path = Path(vacuum_worker.__file__).resolve()
    db_path = tmp_path / "worker.db"
    _ = subprocess.run([sys.executable, str(worker_path), str(db_path)], check=True, timeout=30, capture_output=True)
    assert db_path.is_file()


def test_start_background_vacuum_detaches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str | Path], dict[str, object]]] = []

    def fake_popen(args: list[str | Path], **kwargs: object) -> MagicMock:
        calls.append((args, kwargs))
        return MagicMock()

    monkeypatch.setattr(vacuum_worker.subprocess, "Popen", fake_popen)
    db_path = tmp_path / "index.db"

    assert vacuum_worker.start_background(db_path)
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[1] == Path(vacuum_worker.__file__).resolve()
    assert kwargs["start_new_session"] is True
    assert kwargs["stdin"] is vacuum_worker.subprocess.DEVNULL
    assert (tmp_path / "index.db.vacuum.log").is_file()
