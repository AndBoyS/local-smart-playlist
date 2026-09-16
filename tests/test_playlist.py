"""m3u8 playlist writer tests."""

from __future__ import annotations

from pathlib import Path

from local_smart_playlist.query.playlist import (
    SLUG_FALLBACK,
    playlist_path,
    render_m3u8,
    slugify,
    write_playlist,
)


def test_slugify_ascii() -> None:
    assert slugify("Moody Late Night") == "Moody-Late-Night"


def test_slugify_unicode_preserved() -> None:
    assert slugify("печаль и тоска") == "печаль-и-тоска"
    assert slugify("sad ｜ mood") == "sad-mood"  # fullwidth bar dropped


def test_slugify_strips_forbidden() -> None:
    assert slugify('a/b:c*d?"<>|') == "abcd"


def test_slugify_collapses_dashes() -> None:
    assert slugify("a  --  b") == "a-b"


def test_slugify_empty_falls_back() -> None:
    assert slugify("///") == SLUG_FALLBACK
    assert slugify("") == SLUG_FALLBACK


def test_slugify_length_cap() -> None:
    assert len(slugify("x" * 500)) == 80


def test_playlist_path(tmp_path: Path) -> None:
    p = playlist_path("melancholic", tmp_path)
    assert p == tmp_path / "melancholic.m3u8"


def test_render_relative_paths(tmp_path: Path) -> None:
    root = tmp_path / "lib"
    out = tmp_path / "lib" / "[Playlists]" / "q.m3u8"
    tracks = [
        (str(root / "artist/album/song.mp3"), "artist/album/song", 210.4),
        (str(root / "other.flac"), "other", 30),
    ]
    doc = render_m3u8(tracks, root=root, out_path=out)
    assert doc == (
        "#EXTM3U\n"
        "#EXTINF:210,artist/album/song\n"
        "../artist/album/song.mp3\n"
        "#EXTINF:30,other\n"
        "../other.flac\n"
    )


def test_write_playlist_unicode_utf8(tmp_path: Path) -> None:
    root = tmp_path / "lib"
    out = tmp_path / "lib" / "[Playlists]" / "печаль.m3u8"
    write_playlist([(str(root / "Киты - Песня.mp3"), "Киты - Песня", 100.0)], root=root, out_path=out)
    content = out.read_text(encoding="utf-8")
    assert "#EXTM3U" in content
    assert "Киты - Песня.mp3" in content
    # path resolves relative to the playlist dir


def test_write_playlist_overwrites(tmp_path: Path) -> None:
    root = tmp_path / "lib"
    out = playlist_path("q", tmp_path / "lib" / "[Playlists]")
    write_playlist([], root=root, out_path=out)
    first = out.read_text()
    write_playlist([(str(root / "a.mp3"), "a", 10)], root=root, out_path=out)
    assert out.read_text() != first


def test_render_walks_up_outside_subtree(tmp_path: Path) -> None:
    """Playlist outside the library tree still resolves, via walk-up."""
    tracks: list[tuple[str, str, float]] = [(str(tmp_path / "song.mp3"), "song", 10.0)]
    doc = render_m3u8(tracks, root=tmp_path, out_path=tmp_path / "elsewhere" / "sub" / "q.m3u8")
    assert doc == "#EXTM3U\n#EXTINF:10,song\n../../song.mp3\n"
