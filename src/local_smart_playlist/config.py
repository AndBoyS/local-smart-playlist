"""Paths for the index DB and model cache."""

from pathlib import Path


def default_db_path() -> Path:
    """.sp-index/index.db relative to the current working directory."""
    return Path(".sp-index") / "index.db"


def default_playlist_dir(library_root: Path) -> Path:
    """[Playlists] inside the library root so files stay in the synced tree."""
    return library_root / "[Playlists]"
