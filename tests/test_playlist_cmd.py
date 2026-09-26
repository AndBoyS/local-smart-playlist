"""Playlist command query-union tests."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from local_smart_playlist.commands import playlist_cmd
from local_smart_playlist.commands.playlist_cmd import merge_query_rankings
from local_smart_playlist.index.store import TrackMeta
from local_smart_playlist.query.prompts import LlmError


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


def test_play_starts_vacuum_after_closing_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    db.touch()
    store = MagicMock()
    store.__enter__.return_value = store
    store.get_meta.return_value = str(tmp_path)
    store.get_track.return_value = MagicMock(rel_path="seed.mp3", mean_vec=object())
    store.track_count.return_value = 1
    store.vacuum_if_needed.return_value.recommended = True
    track = TrackMeta(rel_path="result.mp3", title="result", duration=60.0)

    monkeypatch.setattr(playlist_cmd, "cap_torch_threads", lambda: None)

    def make_store(_db: Path) -> MagicMock:
        return store

    def fake_rank(*_args: object, **_kwargs: object) -> list[tuple[TrackMeta, float]]:
        return [(track, 0.95)]

    monkeypatch.setattr(playlist_cmd, "Store", make_store)
    monkeypatch.setattr(playlist_cmd, "rank_by_similarity", fake_rank)

    def start_background(_db_path: Path) -> bool:
        assert store.__exit__.called
        return True

    monkeypatch.setattr(playlist_cmd.vacuum_worker, "start_background", start_background)
    playlist_cmd.PlayArgs.run(
        query="seed",
        db=db,
        n=1,
        out=None,
        llm=False,
        seed_track="seed.mp3",
        alpha=playlist_cmd.PLAY_DEFAULT_ALPHA,
        min_score=0.9,
        dry=True,
    )
    store.vacuum_if_needed.assert_called_once_with(execute=False)


def test_play_crashes_when_llm_adaptation_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    db.touch()
    store = MagicMock()

    def fail_adapt(_query: str) -> list[str]:
        raise LlmError("failed")

    def make_store(_db: Path) -> MagicMock:
        return store

    monkeypatch.setattr(playlist_cmd, "cap_torch_threads", lambda: None)
    monkeypatch.setattr(playlist_cmd, "Store", make_store)
    monkeypatch.setattr(playlist_cmd, "adapt_mood", fail_adapt)

    with pytest.raises(LlmError, match="failed"):
        playlist_cmd.PlayArgs.run(
            query="sad",
            db=db,
            n=None,
            out=None,
            llm=True,
            seed_track=None,
            alpha=playlist_cmd.PLAY_DEFAULT_ALPHA,
            min_score=0.9,
            dry=True,
        )
