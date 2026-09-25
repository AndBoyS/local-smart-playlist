"""Playlist command query-union tests."""

from local_smart_playlist.commands.playlist_cmd import merge_query_rankings
from local_smart_playlist.index.store import TrackMeta


def test_merge_query_rankings_keeps_unique_tracks_and_best_score() -> None:
    first = TrackMeta(rel_path="first.mp3", title="first", duration=100.0)
    shared = TrackMeta(rel_path="shared.mp3", title="shared", duration=120.0)
    third = TrackMeta(rel_path="third.mp3", title="third", duration=90.0)

    merged = merge_query_rankings(
        [
            [(first, 0.8), (shared, 0.7)],
            [(shared, 0.9), (third, 0.85)],
        ]
    )

    assert [(track.rel_path, score) for track, score in merged] == [
        ("shared.mp3", 0.9),
        ("third.mp3", 0.85),
        ("first.mp3", 0.8),
    ]
