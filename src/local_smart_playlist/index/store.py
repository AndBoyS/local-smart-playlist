"""SQLite + sqlite-vec track vector store."""

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from os import PathLike
from typing import cast

import numpy as np
import sqlite_vec
from shape_extensions import IntVar

D = IntVar("D")  # embedding dim (fixed per store instance)
W = IntVar("W")  # window count per track
T = IntVar("T")  # total windows across fetched tracks

SCHEMA_VERSION = "3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tracks (
    rel_path TEXT PRIMARY KEY,
    mean_vec BLOB NOT NULL,
    p90_vec BLOB NOT NULL,
    n_windows INTEGER NOT NULL,
    duration REAL NOT NULL,
    title TEXT NOT NULL,
    model TEXT NOT NULL,
    indexed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS window_blocks (
    rel_path TEXT PRIMARY KEY REFERENCES tracks(rel_path) ON DELETE CASCADE,
    vec BLOB NOT NULL  -- concatenated float32 windows, window order
);
CREATE TABLE IF NOT EXISTS failures (
    rel_path TEXT PRIMARY KEY,
    error TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 1,
    last_at TEXT NOT NULL
);
"""


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _windows_blob(window_vecs: np.ndarray[[W, D]]) -> bytes:
    """Concatenated float32 window block, window order (v3 blob format)."""
    return np.ascontiguousarray(window_vecs, dtype=np.float32).tobytes()


def _serialize(vec: np.ndarray[[D]]) -> bytes:
    return sqlite_vec.serialize_float32(np.ascontiguousarray(vec, dtype=np.float32).tolist())


def _deserialize(blob: bytes, dim: int) -> np.ndarray[[D]]:
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.size != dim:
        msg = f"vector size mismatch: {arr.size} != {dim}"
        raise ValueError(msg)
    return arr.copy()


@dataclass(frozen=True)
class TrackMeta:
    """Track identity/metadata without vector payloads (for ranking scans)."""

    rel_path: str
    title: str
    duration: float


@dataclass(frozen=True)
class TrackRow[D: IntVar]:
    """Full track record; vector payloads carry the store's embedding dim."""

    rel_path: str
    mean_vec: np.ndarray[[D]]
    p90_vec: np.ndarray[[D]]
    n_windows: int
    duration: float
    title: str
    model: str


@dataclass(frozen=True)
class KnnHit:
    rel_path: str
    title: str
    distance: float


class Store:
    """Track-level vector store backed by sqlite + sqlite-vec."""

    def __init__(self, db_path: str | bytes | PathLike[str], embed_dim: int = 512) -> None:
        self._embed_dim = embed_dim
        self._conn = sqlite3.connect(db_path)
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._migrate()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _migrate(self) -> None:
        _ = self._conn.executescript(_SCHEMA)
        row = self._conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if row is None:
            self.set_meta("schema_version", SCHEMA_VERSION)
        elif row[0] == "2":
            self._migrate_v2_windows()
        elif row[0] != SCHEMA_VERSION:
            msg = f"index schema v{row[0]} does not match v{SCHEMA_VERSION}; delete the index or pin a release"
            raise RuntimeError(msg)
        self._ensure_vec_table()

    def _migrate_v2_windows(self) -> None:
        """v2 windows table (row per window) -> v3 window_blocks (blob per track)."""
        rows = self._conn.execute(
            "SELECT rel_path, vec FROM windows ORDER BY rel_path, window_idx"
        ).fetchall()
        blocks: list[tuple[str, bytes]] = []
        current_rel: str | None = None
        parts: list[bytes] = []
        for item in rows:
            rel = cast("str", item[0])
            blob = cast("bytes", item[1])
            if rel != current_rel:
                if current_rel is not None:
                    blocks.append((current_rel, b"".join(parts)))
                current_rel = rel
                parts = []
            parts.append(blob)
        if current_rel is not None:
            blocks.append((current_rel, b"".join(parts)))
        del rows, parts
        _ = self._conn.executemany(
            "INSERT OR REPLACE INTO window_blocks(rel_path, vec) VALUES (?, ?)", blocks
        )
        _ = self._conn.execute("DROP TABLE windows")
        self._conn.commit()
        self.set_meta("schema_version", SCHEMA_VERSION)
        self._ensure_vec_table()

    def _ensure_vec_table(self) -> None:
        """Recreate (and repopulate) the vec0 table if missing; hard-fail on dim mismatch."""
        stored_dim = self.get_meta("vec_dim")
        if stored_dim is not None and int(stored_dim) != self._embed_dim:
            msg = f"index has {stored_dim}-dim vectors but {self._embed_dim} requested; delete the index and re-index"
            raise RuntimeError(msg)
        table = self._conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='tracks_vec'").fetchone()
        if table is not None:
            return
        self._rebuild_vec_table()
        # Repopulate from the durable tracks table so the vec index never lags it.
        rows = self._conn.execute("SELECT rel_path, mean_vec FROM tracks").fetchall()
        _ = self._conn.executemany(
            "INSERT OR REPLACE INTO tracks_vec(rel_path, mean_vec) VALUES (?, ?)",
            [(r[0], r[1]) for r in rows],
        )
        self._conn.commit()

    def _rebuild_vec_table(self) -> None:
        _ = self._conn.execute("DROP TABLE IF EXISTS tracks_vec")
        _ = self._conn.execute(
            f"""
            CREATE VIRTUAL TABLE tracks_vec USING vec0(
                rel_path TEXT PRIMARY KEY,
                mean_vec FLOAT[{self._embed_dim}] distance_metric=cosine
            )
            """
        )
        self.set_meta("vec_dim", str(self._embed_dim))

    # -- meta --------------------------------------------------------------

    def set_meta(self, key: str, value: str) -> None:
        _ = self._conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))
        self._conn.commit()

    def require_model(self, model_id: str) -> None:
        """Hard-fail when the index was built with a different embedding model."""
        stored = self.get_meta("model")
        if stored is not None and stored != model_id:
            msg = f"index built with {stored}; requested {model_id} — re-run `sp index` to rebuild"
            raise SystemExit(msg)

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return cast("str", row[0])

    # -- indexing ----------------------------------------------------------

    def has_track(self, rel_path: str) -> bool:
        result: object | None = self._conn.execute("SELECT 1 FROM tracks WHERE rel_path = ?", (rel_path,)).fetchone()
        return result is not None

    def upsert(
        self,
        *,
        rel_path: str,
        mean_vec: np.ndarray[[D]],
        p90_vec: np.ndarray[[D]],
        n_windows: int,
        duration: float,
        title: str,
        model: str,
        indexed_at: str,
    ) -> None:
        _ = self._conn.execute(
            """
            INSERT OR REPLACE INTO tracks(rel_path, mean_vec, p90_vec, n_windows, duration, title, model, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rel_path,
                _serialize(mean_vec),
                _serialize(p90_vec),
                n_windows,
                duration,
                title,
                model,
                indexed_at,
            ),
        )
        _ = self._conn.execute("DELETE FROM tracks_vec WHERE rel_path = ?", (rel_path,))
        _ = self._conn.execute(
            "INSERT INTO tracks_vec(rel_path, mean_vec) VALUES (?, ?)",
            (rel_path, _serialize(mean_vec)),
        )
        _ = self._conn.execute("DELETE FROM window_blocks WHERE rel_path = ?", (rel_path,))
        _ = self._conn.execute("DELETE FROM failures WHERE rel_path = ?", (rel_path,))
        self._conn.commit()

    def add_windows(self, *, rel_path: str, window_vecs: np.ndarray[[W, D]]) -> None:
        """Replace this track's window vectors with the given (n, dim) array."""
        _ = self._conn.execute("DELETE FROM window_blocks WHERE rel_path = ?", (rel_path,))
        _ = self._conn.execute(
            "INSERT OR REPLACE INTO window_blocks(rel_path, vec) VALUES (?, ?)",
            (rel_path, _windows_blob(window_vecs)),
        )
        self._conn.commit()

    def load_windows_flat(
        self, rel_paths: Iterable[str]
    ) -> tuple[list[str], np.ndarray[[T, D]], list[int]]:
        """All windows as one contiguous [total_windows, dim] matrix, row-major.

        Returns (rel_paths, big matrix, per-track window counts);
        big[starts[i] : starts[i] + counts[i]] is track i's window block.
        Each track stores one concatenated BLOB, so this is one row fetch per
        track plus a single copy into the flat matrix.
        """
        ids = list(rel_paths)
        rows: list[tuple[str, bytes]] = []
        chunk_size = 400  # sqlite variable limit headroom
        for start in range(0, len(ids), chunk_size):
            chunk = ids[start : start + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            rows.extend(
                self._conn.execute(
                    f"""
                    SELECT rel_path, vec FROM window_blocks
                    WHERE rel_path IN ({placeholders})
                    ORDER BY rel_path
                    """,
                    chunk,
                ).fetchall()
            )
        stride = 4 * self._embed_dim
        paths: list[str] = []
        counts: list[int] = []
        blocks: list[np.ndarray] = []
        for item in rows:
            rel = item[0]
            blob = item[1]
            n = len(blob) // stride
            assert len(blob) == n * stride, f"corrupt window block for {rel}"
            if n == 0:
                continue  # empty block: no windows stored
            paths.append(rel)
            counts.append(n)
            blocks.append(
                np.frombuffer(blob, dtype=np.float32).reshape(n, self._embed_dim)  # pyrefly: ignore[unknown-argument-type]
            )
        if len(blocks) == 0:
            return [], cast("np.ndarray[[T, D]]", np.empty((0, self._embed_dim), dtype=np.float32)), []
        big = cast(
            "np.ndarray[[T, D]]",
            np.concatenate(blocks),
        )
        return paths, big, counts

    def load_windows(self, rel_paths: Iterable[str]) -> dict[str, np.ndarray[[W, D]]]:
        """All window vectors grouped by rel_path, in window order."""
        paths, big, counts = self.load_windows_flat(rel_paths)
        if len(paths) == 0:
            return {}
        starts = np.cumsum([0, *counts[:-1]]).tolist()
        return {p: big[s : s + n] for p, s, n in zip(paths, starts, counts, strict=True)}

    def record_failure(self, *, rel_path: str, error: str, now: str) -> None:
        row = self._conn.execute("SELECT attempts FROM failures WHERE rel_path = ?", (rel_path,)).fetchone()
        attempts = (cast("int", row[0]) + 1) if row is not None else 1
        _ = self._conn.execute(
            """
            INSERT OR REPLACE INTO failures(rel_path, error, attempts, last_at) VALUES (?, ?, ?, ?)
            """,
            (rel_path, error, attempts, now),
        )
        self._conn.commit()

    def prune_missing(self, valid_rel_paths: Iterable[str]) -> int:
        """Drop tracks/failures no longer on disk; returns number removed."""
        valid = set(valid_rel_paths)
        tracks = self._conn.execute("SELECT rel_path FROM tracks").fetchall()
        stale = [cast("str", r[0]) for r in tracks if cast("str", r[0]) not in valid]
        failures = self._conn.execute("SELECT rel_path FROM failures").fetchall()
        fstale = [cast("str", r[0]) for r in failures if cast("str", r[0]) not in valid]
        _ = self._conn.executemany("DELETE FROM window_blocks WHERE rel_path = ?", [(p,) for p in stale])
        _ = self._conn.executemany("DELETE FROM tracks WHERE rel_path = ?", [(p,) for p in stale])
        _ = self._conn.executemany("DELETE FROM tracks_vec WHERE rel_path = ?", [(p,) for p in stale])
        _ = self._conn.executemany("DELETE FROM failures WHERE rel_path = ?", [(p,) for p in fstale])
        self._conn.commit()
        return len(stale) + len(fstale)

    # -- stats -------------------------------------------------------------

    def track_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM tracks").fetchone()
        return cast("int", row[0])

    def failure_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM failures").fetchone()
        return cast("int", row[0])

    def window_count(self) -> int:
        row = self._conn.execute("SELECT COALESCE(SUM(length(vec)), 0) FROM window_blocks").fetchone()
        return cast("int", row[0]) // (4 * self._embed_dim)

    def list_failures(self, limit: int = 20) -> list[tuple[str, str, int]]:
        rows = self._conn.execute(
            "SELECT rel_path, error, attempts FROM failures ORDER BY rel_path LIMIT ?", (limit,)
        ).fetchall()
        return [(cast("str", r[0]), cast("str", r[1]), cast("int", r[2])) for r in rows]

    # -- search ------------------------------------------------------------

    def all_track_means(self) -> list[tuple[str, np.ndarray[[D]]]]:
        """Every (rel_path, mean_vec) — full scan for over-KNN-limit ranking."""
        rows = self._conn.execute("SELECT rel_path, mean_vec FROM tracks").fetchall()
        return [(cast("str", r[0]), _deserialize(cast("bytes", r[1]), self._embed_dim)) for r in rows]

    def track_rel_paths(self) -> set[str]:
        """Every indexed rel_path."""
        rows = self._conn.execute("SELECT rel_path FROM tracks").fetchall()
        return {cast("str", r[0]) for r in rows}

    def knn(self, *, query_vec: np.ndarray[[D]], k: int) -> list[KnnHit]:
        """Nearest tracks by cosine distance on the mean vector."""
        rows = self._conn.execute(
            """
            SELECT v.rel_path, t.title, v.distance
            FROM tracks_vec v
            JOIN tracks t USING (rel_path)
            WHERE v.mean_vec MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (_serialize(query_vec), k),
        ).fetchall()
        return [
            KnnHit(
                rel_path=cast("str", r[0]),
                title=cast("str", r[1]),
                distance=cast("float", r[2]),
            )
            for r in rows
        ]

    def track_meta(self) -> list[TrackMeta]:
        """Every (rel_path, title, duration) — one query, no BLOB deserialization."""
        rows = self._conn.execute("SELECT rel_path, title, duration FROM tracks").fetchall()
        return [
            TrackMeta(
                rel_path=cast("str", r[0]),
                title=cast("str", r[1]),
                duration=cast("float", r[2]),
            )
            for r in rows
        ]

    def get_tracks(self, rel_paths: Iterable[str]) -> dict[str, TrackRow[D]]:
        """Full track records for many rel_paths, one query per chunk."""
        ids = list(rel_paths)
        out: dict[str, TrackRow[D]] = {}
        chunk_size = 400  # sqlite variable limit headroom
        for start in range(0, len(ids), chunk_size):
            chunk = ids[start : start + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            rows = self._conn.execute(
                f"""
                SELECT rel_path, mean_vec, p90_vec, n_windows, duration, title, model
                FROM tracks WHERE rel_path IN ({placeholders})
                """,
                chunk,
            ).fetchall()
            for row in rows:
                out[cast("str", row[0])] = TrackRow(
                    rel_path=cast("str", row[0]),
                    mean_vec=_deserialize(cast("bytes", row[1]), self._embed_dim),
                    p90_vec=_deserialize(cast("bytes", row[2]), self._embed_dim),
                    n_windows=cast("int", row[3]),
                    duration=cast("float", row[4]),
                    title=cast("str", row[5]),
                    model=cast("str", row[6]),
                )
        return out

    def get_track(self, rel_path: str) -> TrackRow[D] | None:
        row = self._conn.execute(
            "SELECT rel_path, mean_vec, p90_vec, n_windows, duration, title, model FROM tracks WHERE rel_path = ?",
            (rel_path,),
        ).fetchone()
        if row is None:
            return None
        return TrackRow(
            rel_path=cast("str", row[0]),
            mean_vec=_deserialize(cast("bytes", row[1]), self._embed_dim),
            p90_vec=_deserialize(cast("bytes", row[2]), self._embed_dim),
            n_windows=cast("int", row[3]),
            duration=cast("float", row[4]),
            title=cast("str", row[5]),
            model=cast("str", row[6]),
        )
