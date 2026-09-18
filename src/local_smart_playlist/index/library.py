"""File discovery and stable track identity."""

from pathlib import Path

from local_smart_playlist.audio.decode import AUDIO_EXTS


def relative_posix(root: Path, path: Path) -> str:
    """Stable track id: POSIX relative path from *root*."""
    return path.resolve().relative_to(root.resolve()).as_posix()


def display_title(root: Path, path: Path) -> str:
    """Human-readable title: relative path minus extension."""
    return Path(relative_posix(root, path)).with_suffix("").as_posix()


def discover_audio(root: Path) -> list[Path]:
    """All decodable audio files under *root*, sorted by relative path."""
    found = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]

    def sort_key(path: Path) -> str:
        return relative_posix(root, path)

    return sorted(found, key=sort_key)
