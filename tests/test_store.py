"""Store tests with a real sqlite-vec, synthetic vectors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from local_smart_playlist.index.store import Store

DIM = 64


def basis_vec(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "idx.db", embed_dim=DIM)


def upsert(store: Store, rel_path: str, i: int) -> None:
    vec = basis_vec(i)
    store.upsert(rel_path=rel_path, mean_vec=vec, p90_vec=vec, n_windows=6, duration=60.0, title=rel_path, model="test", indexed_at="now")


def test_upsert_and_knn(store: Store) -> None:
    for i in range(20):
        upsert(store, f"t{i}.mp3", i)
    hits = store.knn(query_vec=basis_vec(7), k=3)
    assert hits[0].rel_path == "t7.mp3"
    assert hits[0].distance == pytest.approx(0.0, abs=1e-5)
    assert len(hits) == 3
    assert [h.distance for h in hits] == sorted(h.distance for h in hits)


def test_has_track_and_roundtrip(store: Store) -> None:
    upsert(store, "a/b.mp3", 1)
    assert store.has_track("a/b.mp3")
    assert not store.has_track("missing.mp3")
    track = store.get_track("a/b.mp3")
    assert track is not None
    assert track.n_windows == 6
    assert np.allclose(track.mean_vec, basis_vec(1))
    assert store.get_track("missing.mp3") is None


def test_failure_attempts_and_clear_on_upsert(store: Store) -> None:
    store.record_failure(rel_path="bad.mp3", error="boom", now="now")
    store.record_failure(rel_path="bad.mp3", error="boom2", now="now2")
    assert store.list_failures() == [("bad.mp3", "boom2", 2)]
    upsert(store, "bad.mp3", 3)
    assert store.list_failures() == []


def test_prune_missing(store: Store) -> None:
    for i in range(5):
        upsert(store, f"t{i}.mp3", i)
    store.record_failure(rel_path="gone.mp3", error="x", now="now")
    removed = store.prune_missing({"t1.mp3", "t3.mp3"})
    assert removed == 4  # 3 tracks + 1 failure
    assert store.track_count() == 2
    assert store.failure_count() == 0


def test_resumable_scan_semantics(store: Store) -> None:
    upsert(store, "kept.mp3", 0)
    assert store.has_track("kept.mp3")


def test_persistence_and_vec_repopulate(tmp_path: Path) -> None:
    db = tmp_path / "persist.db"
    with Store(db, embed_dim=DIM) as s:
        upsert(s, "x.mp3", 5)
    with Store(db, embed_dim=DIM) as s:
        assert s.track_count() == 1
        hits = s.knn(query_vec=basis_vec(5), k=1)
        assert hits[0].rel_path == "x.mp3"


def test_dim_mismatch_raises(tmp_path: Path) -> None:
    db = tmp_path / "dim.db"
    with Store(db, embed_dim=DIM):
        pass
    with pytest.raises(RuntimeError, match="re-index"), Store(db, embed_dim=DIM + 8):
        pass


def test_schema_version_mismatch(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "schema.db"
    with Store(db, embed_dim=DIM):
        pass
    conn = sqlite3.connect(db)
    _ = conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', '999')")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="schema"), Store(db, embed_dim=DIM):
        pass


def test_knn_limit_exceeds_count(store: Store) -> None:
    for i in range(3):
        upsert(store, f"t{i}.mp3", i)
    hits = store.knn(query_vec=basis_vec(0), k=10)
    assert len(hits) == 3




def test_windows_roundtrip_and_prune(tmp_path: Path) -> None:
    store = Store(tmp_path / "win.db", embed_dim=DIM)
    vec = basis_vec(3)
    store.upsert(
        rel_path="a.mp3",
        mean_vec=vec,
        p90_vec=vec,
        n_windows=2,
        duration=60.0,
        title="a",
        model="test",
        indexed_at="now",
    )
    store.add_windows(rel_path="a.mp3", window_vecs=np.stack([basis_vec(0), basis_vec(1)]))
    assert store.window_count() == 2
    loaded = store.load_windows(["a.mp3"])
    assert np.allclose(loaded["a.mp3"], np.stack([basis_vec(0), basis_vec(1)]))

    # add_windows replaces
    store.add_windows(rel_path="a.mp3", window_vecs=np.stack([basis_vec(5)]))
    assert store.window_count() == 1

    # load_windows skips unknown paths
    assert store.load_windows(["missing.mp3"]) == {}

    # upsert clears stale windows
    store.upsert(
        rel_path="a.mp3",
        mean_vec=vec,
        p90_vec=vec,
        n_windows=1,
        duration=60.0,
        title="a",
        model="test",
        indexed_at="now2",
    )
    assert store.window_count() == 0

    # prune cascades
    store.add_windows(rel_path="a.mp3", window_vecs=np.stack([basis_vec(0)]))
    _removed = store.prune_missing({"b.mp3"})
    assert store.window_count() == 0
