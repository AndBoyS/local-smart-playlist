"""Tests for config helpers."""

from pathlib import Path

from local_smart_playlist.config import default_db_path


def test_default_db_path_is_relative_sp_index() -> None:
    assert default_db_path() == Path(".sp-index") / "index.db"
    assert not default_db_path().is_absolute()
