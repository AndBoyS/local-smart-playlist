"""m3u8 playlist writing."""

import re
import unicodedata
from pathlib import Path

SLUG_FALLBACK = "playlist"


def slugify(text: str) -> str:
    """Filesystem-safe slug preserving unicode letters, digits, hyphen, underscore."""
    text = text.strip().strip(".")
    text = re.sub(r"[\s]+", "-", text)
    text = re.sub(r"[^0-9A-Za-z\-_.\u00c0-\u024f\u0400-\u04ff\u3100-\u9fff\uac00-\ud7af]", "", text)
    text = re.sub(r"-+", "-", text).strip("-.")
    if text == "":
        return SLUG_FALLBACK
    return text[:80]


def playlist_path(query: str, default_dir: Path) -> Path:
    return default_dir / f"{slugify(query)}.m3u8"


def render_m3u8(tracks: list[tuple[str, str, float]], *, root: Path, out_path: Path) -> str:
    """Render an m3u8 document.

    *tracks* is a list of (abs_or_rel_path, title, duration_seconds). Paths are
    written relative to the playlist file's directory so playlists stay portable
    across synced copies of the library tree. When the playlist lives outside
    the library tree (different root/volume) a relative path is impossible, so
    the track's resolved absolute path is written instead — still a valid m3u8.
    """
    lines = ["#EXTM3U"]
    for track_path, title, duration in tracks:
        try:
            rel = Path(track_path).resolve().relative_to(out_path.parent.resolve(), walk_up=True).as_posix()
        except ValueError:
            rel = Path(track_path).resolve().as_posix()
        lines.append(f"#EXTINF:{duration:.0f},{title}")
        lines.append(rel)
    return "\n".join(lines) + "\n"


def write_playlist(tracks: list[tuple[str, str, float]], *, root: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)  # noqa: PTH110
    _ = out_path.write_text(render_m3u8(tracks, root=root, out_path=out_path), encoding="utf-8")


def ascii_safe(text: str) -> str:
    """NFKC-normalized text for logs."""
    return unicodedata.normalize("NFKC", text)
