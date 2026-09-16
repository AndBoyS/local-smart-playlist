"""Paths for the index DB and model cache."""

from __future__ import annotations

import os
from pathlib import Path


def default_db_path() -> Path:
    """$XDG_DATA_HOME/sp/index.db (macOS: ~/Library/Application Support/sp/index.db)."""
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg is not None else Path.home() / "Library" / "Application Support"
    return base / "sp" / "index.db"


def default_playlist_dir(library_root: Path) -> Path:
    """[Playlists] inside the library root so files stay in the synced tree."""
    return library_root / "[Playlists]"
