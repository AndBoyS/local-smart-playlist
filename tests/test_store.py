"""Store tests with SQLite and synthetic vectors."""

from pathlib import Path

import numpy as np
import pytest

from local_smart_playlist.index.store import MetaKey, Store

DIM = 64


def basis_vec(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0  # pyrefly: ignore[unsupported-operation]  # numpy shape stubs lack __setitem__ (facebook/pyrefly#4901)
    return v


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "idx.db", embed_dim=DIM)


def upsert(store: Store, rel_path: str, i: int, *, window_vecs: np.ndarray | None = None) -> None:
    vec = basis_vec(i)
    store.upsert(
        rel_path=rel_path,
        mean_vec=vec,
        p90_vec=vec,
        n_windows=6 if window_vecs is None else len(window_vecs),
        duration=60.0,
        title=rel_path,
        model="test",
        indexed_at="now",
        window_vecs=window_vecs,
    )


def test_all_track_means(store: Store) -> None:
    for i in range(20):
        upsert(store, f"t{i}.mp3", i)
    means = store.all_track_means()
    assert len(means) == 20
    assert np.allclose(means["t7.mp3"], basis_vec(7))


def test_has_track_and_roundtrip(store: Store) -> None:
    upsert(store, "a/b.mp3", 1)
    assert store.has_track("a/b.mp3")
    assert not store.has_track("missing.mp3")
    track = store.get_track("a/b.mp3")
    assert track is not None
    assert track.n_windows == 6
    assert track.vectors is not None
    assert np.allclose(track.vectors.mean_vec, basis_vec(1))
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


def test_persistence_and_vector_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "persist.db"
    with Store(db, embed_dim=DIM) as s:
        upsert(s, "x.mp3", 5)
    with Store(db, embed_dim=DIM) as s:
        assert s.track_count() == 1
        assert np.allclose(s.all_track_means()["x.mp3"], basis_vec(5))


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
    _ = conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('SCHEMA_VERSION', '999')")
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="schema"), Store(db, embed_dim=DIM):
        pass


def test_legacy_vocab_cache_meta_keys_are_removed(tmp_path: Path) -> None:
    db = tmp_path / "legacy-cache.db"
    with Store(db, embed_dim=DIM) as store, store._engine.begin() as connection:
        _ = connection.exec_driver_sql(
            "INSERT INTO meta(key, value) VALUES (?, ?)",
            [("vocab_vecs:old-model:old-digest", "old cache")],
        )
        _ = connection.exec_driver_sql("UPDATE meta SET key = 'schema_version' WHERE key = 'SCHEMA_VERSION'")

    with Store(db, embed_dim=DIM) as store:
        with store._engine.connect() as connection:
            legacy_cache_count = connection.exec_driver_sql(
                "SELECT COUNT(*) FROM meta WHERE key LIKE 'vocab_vecs:%'"
            ).scalar_one()
            legacy_key_count = connection.exec_driver_sql(
                "SELECT COUNT(*) FROM meta WHERE key = 'schema_version'"
            ).scalar_one()
            enum_key_count = connection.exec_driver_sql(
                "SELECT COUNT(*) FROM meta WHERE key = 'SCHEMA_VERSION'"
            ).scalar_one()
        assert legacy_cache_count == 0
        assert legacy_key_count == 0
        assert enum_key_count == 1
        assert store.get_meta(MetaKey.SCHEMA_VERSION) == "4"


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
        window_vecs=np.stack([basis_vec(0), basis_vec(1)]),
    )
    assert store.window_count() == 2
    loaded = store.load_windows(["a.mp3"])
    assert np.allclose(loaded["a.mp3"], np.stack([basis_vec(0), basis_vec(1)]))

    # upsert replaces vector payloads atomically with track data
    upsert(store, "a.mp3", 5, window_vecs=np.stack([basis_vec(5)]))
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
    upsert(store, "a.mp3", 0, window_vecs=np.stack([basis_vec(0)]))
    _removed = store.prune_missing({"b.mp3"})
    assert store.window_count() == 0


def test_track_meta_without_vectors(store: Store) -> None:
    """track_meta returns metadata-only records without loading vector BLOBs."""
    upsert(store, "a/b.mp3", 1)
    upsert(store, "c.mp3", 2)
    metas = store.track_meta()
    assert {m.rel_path for m in metas} == {"a/b.mp3", "c.mp3"}
    for m in metas:
        assert m.duration == pytest.approx(60.0)
        assert m.title == m.rel_path
        assert m.vectors is None


def test_migrate_v2_windows_to_blocks(tmp_path: Path) -> None:
    """v2 row-per-window DB upgrades in place to v4 window_blocks."""
    import sqlite3

    path = tmp_path / "v2.db"
    conn = sqlite3.connect(path)
    _ = conn.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE tracks (
            rel_path TEXT PRIMARY KEY,
            mean_vec BLOB NOT NULL,
            p90_vec BLOB NOT NULL,
            n_windows INTEGER NOT NULL,
            duration REAL NOT NULL,
            title TEXT NOT NULL,
            model TEXT NOT NULL,
            indexed_at TEXT NOT NULL
        );
        CREATE TABLE windows (
            rel_path TEXT NOT NULL REFERENCES tracks(rel_path) ON DELETE CASCADE,
            window_idx INTEGER NOT NULL,
            vec BLOB NOT NULL,
            PRIMARY KEY (rel_path, window_idx)
        );
        """
    )
    vec = basis_vec(2)
    _ = conn.execute("INSERT INTO meta VALUES ('schema_version', '2')")
    _ = conn.execute(
        "INSERT INTO tracks VALUES ('a.mp3', ?, ?, 2, 60.0, 'a', 'test', 'now')",
        (np.ascontiguousarray(vec, dtype=np.float32).tobytes(),) * 2,
    )
    _ = conn.executemany(
        "INSERT INTO windows VALUES (?, ?, ?)",
        [
            ("a.mp3", 0, np.ascontiguousarray(basis_vec(0), dtype=np.float32).tobytes()),
            ("a.mp3", 1, np.ascontiguousarray(basis_vec(1), dtype=np.float32).tobytes()),
        ],
    )
    conn.commit()
    conn.close()

    store = Store(path, embed_dim=DIM)  # triggers migration
    assert store.get_meta(MetaKey.SCHEMA_VERSION) == "4"
    assert store.window_count() == 2
    loaded = store.load_windows(["a.mp3"])
    assert np.allclose(loaded["a.mp3"], np.stack([basis_vec(0), basis_vec(1)]))

    # upsert still writes windows after migration
    upsert(store, "a.mp3", 4, window_vecs=np.stack([basis_vec(4)]))
    assert store.window_count() == 1
    # pyrefly: ignore[unknown-argument-type]
    assert np.allclose(store.load_windows(["a.mp3"])["a.mp3"], basis_vec(4).reshape(1, DIM))


def test_migrate_v3_removes_window_delete_cascade(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "v3.db"
    conn = sqlite3.connect(path)
    _ = conn.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE tracks (
            rel_path TEXT PRIMARY KEY,
            mean_vec BLOB NOT NULL,
            p90_vec BLOB NOT NULL,
            n_windows INTEGER NOT NULL,
            duration REAL NOT NULL,
            title TEXT NOT NULL,
            model TEXT NOT NULL,
            indexed_at TEXT NOT NULL
        );
        CREATE TABLE window_blocks (
            rel_path TEXT NOT NULL PRIMARY KEY REFERENCES tracks(rel_path) ON DELETE CASCADE,
            vec BLOB NOT NULL
        );
        """
    )
    vec = basis_vec(0).tobytes()
    _ = conn.execute("INSERT INTO meta VALUES ('schema_version', '3')")
    _ = conn.execute("INSERT INTO tracks VALUES ('a.mp3', ?, ?, 1, 60.0, 'a', 'test', 'now')", (vec, vec))
    _ = conn.execute("INSERT INTO window_blocks VALUES ('a.mp3', ?)", (vec,))
    conn.commit()
    conn.close()

    with Store(path, embed_dim=DIM) as store:
        assert store.get_meta(MetaKey.SCHEMA_VERSION) == "4"
        assert store.window_count() == 1

        check = sqlite3.connect(path)
        _ = check.execute("PRAGMA foreign_keys=ON")
        with pytest.raises(sqlite3.IntegrityError):
            _ = check.execute("DELETE FROM tracks WHERE rel_path = ?", ("a.mp3",))
        check.close()

        assert store.prune_missing(set()) == 1
        assert store.window_count() == 0


def test_load_windows_flat_layout(tmp_path: Path) -> None:
    store = Store(tmp_path / "flat.db", embed_dim=DIM)
    for rel, i in (("b.mp3", 1), ("a.mp3", 0)):
        vec = basis_vec(i)
        store.upsert(
            rel_path=rel,
            mean_vec=vec,
            p90_vec=vec,
            n_windows=2,
            duration=60.0,
            title=rel,
            model="test",
            indexed_at="now",
            window_vecs=np.stack([basis_vec(i), basis_vec(i + 3)]),
        )

    flat = store.load_windows_batched(["a.mp3", "b.mp3", "missing.mp3"])
    assert flat.rel_paths == ["a.mp3", "b.mp3"]
    assert flat.window_sizes == [2, 2]
    assert flat.vec_matrix.shape == (4, DIM)
    assert np.allclose(flat.vec_matrix[:2], np.stack([basis_vec(0), basis_vec(3)]))
    assert np.allclose(flat.vec_matrix[2:], np.stack([basis_vec(1), basis_vec(4)]))

    # empty candidate set
    empty = store.load_windows_batched([])
    assert empty.rel_paths == []
    assert empty.vec_matrix.shape == (0, DIM)
    assert empty.window_sizes == []
    # legacy track without windows
    store.upsert(
        rel_path="legacy.mp3",
        mean_vec=vec,
        p90_vec=vec,
        n_windows=0,
        duration=60.0,
        title="legacy",
        model="test",
        indexed_at="now",
    )
    legacy = store.load_windows_batched(["legacy.mp3"])
    assert legacy.rel_paths == []
    assert legacy.vec_matrix.shape == (0, DIM)
    assert legacy.window_sizes == []
