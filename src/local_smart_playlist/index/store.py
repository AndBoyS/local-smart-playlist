"""SQLAlchemy ORM-backed SQLite track and vector store."""

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from os import PathLike, fsdecode
from typing import Self

import numpy as np
from shape_extensions import IntVar
from sqlalchemy import (
    Enum,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    Text,
    create_engine,
    delete,
    event,
    func,
    select,
)
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from local_smart_playlist.numpy_helpers import reshape
from local_smart_playlist.type_utils import verify_type

D = IntVar("D")  # embedding dim (fixed per store instance)
W = IntVar("W")  # window count per track
T = IntVar("T")  # total windows across fetched tracks


@dataclass(frozen=True)
class BatchedVectors[N: IntVar, D: IntVar]:
    rel_paths: list[str]
    vec_matrix: np.ndarray[[N, D]]
    window_sizes: list[int]


SCHEMA_VERSION = "4"


class MetaKey(StrEnum):
    """Fixed metadata keys stored in the index."""

    SCHEMA_VERSION = "schema_version"
    VEC_DIM = "vec_dim"
    MODEL = "model"
    LIBRARY_ROOT = "library_root"
    VOCAB_VECS = "vocab_vecs"


MIN_VACUUM_FREE_BYTES = 50 * 1024 * 1024
MIN_VACUUM_FREE_FRACTION = 0.25


def should_vacuum(*, free_bytes: int, total_bytes: int) -> bool:
    """Vacuum only when free space is both substantial and a large share of the DB."""
    return (
        total_bytes > 0 and free_bytes >= MIN_VACUUM_FREE_BYTES and free_bytes / total_bytes >= MIN_VACUUM_FREE_FRACTION
    )


class TableRegistry(DeclarativeBase):
    pass


class MetaOrm(TableRegistry):
    """Key-value metadata stored with the index."""

    __tablename__ = "meta"

    key: Mapped[MetaKey] = mapped_column(Enum(MetaKey, native_enum=False), primary_key=True)
    value: Mapped[str] = mapped_column(Text)


class TrackOrm(TableRegistry):
    """Persisted track metadata and vector blobs."""

    __tablename__ = "tracks"

    rel_path: Mapped[str] = mapped_column(Text, primary_key=True)
    mean_vec: Mapped[bytes] = mapped_column(LargeBinary)
    p90_vec: Mapped[bytes] = mapped_column(LargeBinary)
    n_windows: Mapped[int] = mapped_column(Integer)
    duration: Mapped[float] = mapped_column(Float)
    title: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(Text)
    indexed_at: Mapped[str] = mapped_column(Text)


@dataclass(frozen=True)
class TrackVectors[D: IntVar]:
    mean_vec: np.ndarray[[D]]
    p90_vec: np.ndarray[[D]]


@dataclass(frozen=True)
class TrackData[D: IntVar]:
    """Track metadata, with vector payloads loaded only when requested."""

    rel_path: str
    title: str
    duration: float
    n_windows: int | None = None
    model: str | None = None
    vectors: TrackVectors[D] | None = None


class EmbedOrm(TableRegistry):
    __tablename__ = "window_blocks"

    rel_path: Mapped[str] = mapped_column(
        Text,
        ForeignKey(TrackOrm.rel_path),
        primary_key=True,
    )
    vec: Mapped[bytes] = mapped_column(LargeBinary)


class FailureOrm(TableRegistry):
    """Most recent indexing failure for one track."""

    __tablename__ = "failures"

    rel_path: Mapped[str] = mapped_column(Text, primary_key=True)
    error: Mapped[str] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    last_at: Mapped[str] = mapped_column(Text)


def serialize_vec(vec: np.ndarray) -> bytes:
    return np.ascontiguousarray(vec, dtype=np.float32).tobytes()


def deserialize_vec(blob: bytes, dim: int) -> np.ndarray[[D]]:
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.size != dim:
        msg = f"vector size mismatch: {arr.size} != {dim}"
        raise ValueError(msg)
    return arr.copy()


@dataclass(frozen=True)
class VacuumResult:
    recommended: bool
    performed: bool
    free_before_bytes: int
    size_before_bytes: int
    size_after_bytes: int


class Store:
    """Track-level vector store backed by SQLAlchemy ORM and SQLite."""

    def __init__(self, db_path: str | bytes | PathLike[str], embed_dim: int = 512) -> None:
        self.embed_dim = embed_dim
        database = fsdecode(db_path)
        self._engine = create_engine(URL.create("sqlite", database=database))
        _ = event.listen(self._engine, "connect", self._enable_fkeys_constrain_check)
        self._migrate()

    @staticmethod
    def _enable_fkeys_constrain_check(dbapi_connection: sqlite3.Connection, _: object) -> None:
        _ = dbapi_connection.execute("PRAGMA foreign_keys=ON")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self._engine.dispose()

    def _migrate(self) -> None:
        _ = TableRegistry.metadata.create_all(self._engine)
        with self._engine.begin() as connection:
            _ = connection.exec_driver_sql("DELETE FROM meta WHERE key LIKE 'vocab_vecs:%'")
            _ = connection.exec_driver_sql(
                "UPDATE meta SET key = ? WHERE key = ?",
                [(key.name, key.value) for key in MetaKey],
            )
        version = self.get_meta(MetaKey.SCHEMA_VERSION)
        if version is None:
            self.set_meta(MetaKey.SCHEMA_VERSION, SCHEMA_VERSION)
        else:
            if version == "2":
                self._migrate_v2_windows()
                version = "3"
            if version == "3":
                self._migrate_v3_window_fk()
                version = SCHEMA_VERSION
            if version != SCHEMA_VERSION:
                msg = f"index schema v{version} does not match v{SCHEMA_VERSION}; delete the index or pin a release"
                raise RuntimeError(msg)
        self._ensure_vec_dim()

    def _migrate_v2_windows(self) -> None:
        """v2 windows table (row per window) -> v3 window_blocks (blob per track)."""
        with self._engine.begin() as connection:
            result = connection.exec_driver_sql("SELECT rel_path, vec FROM windows ORDER BY rel_path, window_idx")
            rows = result.all()
            blocks: list[tuple[str, bytes]] = []
            current_rel: str | None = None
            parts: list[bytes] = []
            for rel, blob in rows:
                rel_path = verify_type(rel, str)
                part = verify_type(blob, bytes)
                if rel_path != current_rel:
                    if current_rel is not None:
                        blocks.append((current_rel, b"".join(parts)))
                    current_rel = rel_path
                    parts = []
                parts.append(part)
            if current_rel is not None:
                blocks.append((current_rel, b"".join(parts)))
            if len(blocks) > 0:
                _ = connection.exec_driver_sql(
                    "INSERT OR REPLACE INTO window_blocks(rel_path, vec) VALUES (?, ?)", blocks
                )
            _ = connection.exec_driver_sql("DROP TABLE windows")
        self.set_meta(MetaKey.SCHEMA_VERSION, "3")

    def _migrate_v3_window_fk(self) -> None:
        """Remove database-level cascade; prune_missing deletes child rows explicitly."""
        with self._engine.begin() as connection:
            _ = connection.exec_driver_sql(
                """
                CREATE TABLE window_blocks_v4 (
                    rel_path TEXT NOT NULL PRIMARY KEY,
                    vec BLOB NOT NULL,
                    FOREIGN KEY (rel_path) REFERENCES tracks (rel_path)
                )
                """
            )
            _ = connection.exec_driver_sql(
                "INSERT INTO window_blocks_v4 (rel_path, vec) SELECT rel_path, vec FROM window_blocks"
            )
            _ = connection.exec_driver_sql("DROP TABLE window_blocks")
            _ = connection.exec_driver_sql("ALTER TABLE window_blocks_v4 RENAME TO window_blocks")
        self.set_meta(MetaKey.SCHEMA_VERSION, SCHEMA_VERSION)

    def _ensure_vec_dim(self) -> None:
        """Validate stored vector dimension, initializing it for new indexes."""
        stored_dim = self.get_meta(MetaKey.VEC_DIM)
        if stored_dim is not None and int(stored_dim) != self.embed_dim:
            msg = f"index has {stored_dim}-dim vectors but {self.embed_dim} requested; delete the index and re-index"
            raise RuntimeError(msg)
        if stored_dim is None:
            self.set_meta(MetaKey.VEC_DIM, str(self.embed_dim))

    def set_meta(self, key: MetaKey, value: str) -> None:
        with Session(self._engine) as session, session.begin():
            row = session.get(MetaOrm, key)
            if row is None:
                session.add(MetaOrm(key=key, value=value))
            else:
                row.value = value

    def get_meta(self, key: MetaKey) -> str | None:
        with Session(self._engine) as session:
            row = session.get(MetaOrm, key)
            return row.value if row is not None else None

    def require_model(self, model_id: str) -> None:
        """Hard-fail when the index was built with a different embedding model."""
        stored = self.get_meta(MetaKey.MODEL)
        if stored is not None and stored != model_id:
            msg = f"index built with {stored}; requested {model_id} — re-run `sp index` to rebuild"
            raise SystemExit(msg)

    def has_track(self, rel_path: str) -> bool:
        with Session(self._engine) as session:
            return session.get(TrackOrm, rel_path) is not None

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
        window_vecs: np.ndarray[[W, D]] | None = None,
    ) -> None:
        with Session(self._engine) as session, session.begin():
            track = session.get(TrackOrm, rel_path)
            if track is None:
                track = TrackOrm(rel_path=rel_path)
                session.add(track)
            track.mean_vec = serialize_vec(mean_vec)
            track.p90_vec = serialize_vec(p90_vec)
            track.n_windows = n_windows
            track.duration = duration
            track.title = title
            track.model = model
            track.indexed_at = indexed_at
            _ = session.execute(delete(EmbedOrm).where(EmbedOrm.rel_path == rel_path))
            if window_vecs is not None:
                session.add(EmbedOrm(rel_path=rel_path, vec=serialize_vec(window_vecs)))
            _ = session.execute(delete(FailureOrm).where(FailureOrm.rel_path == rel_path))

    def load_windows_batched(self, rel_paths: Iterable[str]) -> BatchedVectors[T, D]:
        """All windows as one contiguous [total_windows, dim] matrix, row-major.

        Returns rel_paths, vec_matrix, and per-track window_sizes;
        vec_matrix[starts[i] : starts[i] + window_sizes[i]] is track i's window block.
        Each track stores one concatenated BLOB, so this is one row fetch per
        track plus a single copy into the flat matrix.
        """
        ids = list(rel_paths)
        rows: list[tuple[str, bytes]] = []
        chunk_size = 400  # sqlite variable limit headroom
        with Session(self._engine) as session:
            for start in range(0, len(ids), chunk_size):
                chunk = ids[start : start + chunk_size]
                rows.extend(
                    (rel_path, vec)
                    for rel_path, vec in session.execute(
                        select(EmbedOrm.rel_path, EmbedOrm.vec)
                        .where(EmbedOrm.rel_path.in_(chunk))
                        .order_by(EmbedOrm.rel_path)
                    ).all()
                )
        stride = 4 * self.embed_dim
        paths: list[str] = []
        counts: list[int] = []
        blocks: list[np.ndarray[[T, D]]] = []
        for rel_path, blob in rows:
            n = len(blob) // stride
            assert len(blob) == n * stride, f"corrupt window block for {rel_path}"
            if n == 0:
                continue  # empty block: no windows stored
            paths.append(rel_path)
            counts.append(n)
            blocks.append(reshape(np.frombuffer(blob, dtype=np.float32), (n, self.embed_dim)))
        if len(blocks) == 0:
            return BatchedVectors(
                rel_paths=[],
                vec_matrix=np.empty((len(blocks), self.embed_dim), dtype=np.float32),
                window_sizes=[],
            )
        vecs = np.concatenate(blocks)
        return BatchedVectors(rel_paths=paths, vec_matrix=vecs, window_sizes=counts)

    def load_windows(self, rel_paths: Iterable[str]) -> dict[str, np.ndarray[[W, D]]]:
        """All window vectors grouped by rel_path, in window order."""
        batch = self.load_windows_batched(rel_paths)
        if len(batch.rel_paths) == 0:
            return {}
        starts: list[int] = np.cumsum([0, *batch.window_sizes[:-1]]).tolist()
        return {
            path: batch.vec_matrix[start : start + count]
            for path, start, count in zip(batch.rel_paths, starts, batch.window_sizes, strict=True)
        }

    def record_failure(self, *, rel_path: str, error: str, now: str) -> None:
        with Session(self._engine) as session, session.begin():
            row = session.get(FailureOrm, rel_path)
            if row is None:
                session.add(FailureOrm(rel_path=rel_path, error=error, attempts=1, last_at=now))
            else:
                row.error = error
                row.attempts += 1
                row.last_at = now

    def prune_missing(self, valid_rel_paths: Iterable[str]) -> int:
        """Drop tracks/failures no longer on disk; returns number removed."""
        valid = set(valid_rel_paths)
        with Session(self._engine) as session, session.begin():
            track_paths = session.scalars(select(TrackOrm.rel_path)).all()
            failure_paths = session.scalars(select(FailureOrm.rel_path)).all()
            stale = [path for path in track_paths if path not in valid]
            stale_failures = [path for path in failure_paths if path not in valid]
            if len(stale) > 0:
                _ = session.execute(delete(EmbedOrm).where(EmbedOrm.rel_path.in_(stale)))
                _ = session.execute(delete(TrackOrm).where(TrackOrm.rel_path.in_(stale)))
            if len(stale_failures) > 0:
                _ = session.execute(delete(FailureOrm).where(FailureOrm.rel_path.in_(stale_failures)))
        return len(stale) + len(stale_failures)

    def vacuum_if_needed(self, *, execute: bool) -> VacuumResult:
        """Check free-page thresholds, optionally compact, and report DB sizes."""
        with self._engine.connect() as connection:
            page_size = verify_type(connection.exec_driver_sql("PRAGMA page_size").scalar_one(), int)
            page_count = verify_type(connection.exec_driver_sql("PRAGMA page_count").scalar_one(), int)
            freelist_count = verify_type(connection.exec_driver_sql("PRAGMA freelist_count").scalar_one(), int)
        free_before = freelist_count * page_size
        size_before = page_count * page_size
        recommended = should_vacuum(free_bytes=free_before, total_bytes=size_before)
        if not recommended or not execute:
            return VacuumResult(
                recommended=recommended,
                performed=False,
                free_before_bytes=free_before,
                size_before_bytes=size_before,
                size_after_bytes=size_before,
            )

        with self._engine.connect() as connection:
            _ = connection.exec_driver_sql("VACUUM")
            page_count_after = verify_type(connection.exec_driver_sql("PRAGMA page_count").scalar_one(), int)

        size_after = page_count_after * page_size
        return VacuumResult(
            recommended=recommended,
            performed=True,
            free_before_bytes=free_before,
            size_before_bytes=size_before,
            size_after_bytes=size_after,
        )

    def track_count(self) -> int:
        with Session(self._engine) as session:
            return verify_type(session.scalar(select(func.count()).select_from(TrackOrm)), int)

    def failure_count(self) -> int:
        with Session(self._engine) as session:
            return verify_type(session.scalar(select(func.count()).select_from(FailureOrm)), int)

    def window_count(self) -> int:
        with Session(self._engine) as session:
            byte_count = verify_type(
                session.scalar(select(func.coalesce(func.sum(func.length(EmbedOrm.vec)), 0))),
                int,
            )
        return byte_count // (4 * self.embed_dim)

    def list_failures(self, limit: int = 20) -> list[tuple[str, str, int]]:
        with Session(self._engine) as session:
            rows = session.execute(
                select(FailureOrm.rel_path, FailureOrm.error, FailureOrm.attempts)
                .order_by(FailureOrm.rel_path)
                .limit(limit)
            ).all()
        return [(rel_path, error, attempts) for rel_path, error, attempts in rows]

    def all_track_means(self) -> dict[str, np.ndarray[[D]]]:
        """Every rel_path mapped to mean_vec for exact NumPy ranking."""
        with Session(self._engine) as session:
            rows = session.execute(select(TrackOrm.rel_path, TrackOrm.mean_vec)).all()
        return {rel_path: deserialize_vec(mean_vec, self.embed_dim) for rel_path, mean_vec in rows}

    def track_rel_paths(self) -> set[str]:
        """Every rel_path."""
        with Session(self._engine) as session:
            return set(session.scalars(select(TrackOrm.rel_path)).all())

    def track_meta(self) -> list[TrackData[D]]:
        """Metadata-only track records — one query, no BLOB deserialization."""
        with Session(self._engine) as session:
            rows = session.execute(select(TrackOrm.rel_path, TrackOrm.title, TrackOrm.duration)).all()
        return [TrackData(rel_path=path, title=title, duration=duration) for path, title, duration in rows]

    def get_tracks(self, rel_paths: Iterable[str]) -> dict[str, TrackData[D]]:
        """Full track records for many rel_paths, one query per chunk."""
        ids = list(rel_paths)
        out: dict[str, TrackData[D]] = {}
        chunk_size = 400  # sqlite variable limit headroom
        with Session(self._engine) as session:
            for start in range(0, len(ids), chunk_size):
                chunk = ids[start : start + chunk_size]
                for track in session.scalars(select(TrackOrm).where(TrackOrm.rel_path.in_(chunk))).all():
                    out[track.rel_path] = self.to_track_data(track)
        return out

    def get_track(self, rel_path: str) -> TrackData[D] | None:
        with Session(self._engine) as session:
            track = session.get(TrackOrm, rel_path)
            return None if track is None else self.to_track_data(track)

    def to_track_data(self, track: TrackOrm) -> TrackData[D]:
        """Decode ORM track blobs into metadata plus optional vector payloads."""
        return TrackData(
            rel_path=track.rel_path,
            title=track.title,
            duration=track.duration,
            n_windows=track.n_windows,
            model=track.model,
            vectors=TrackVectors(
                mean_vec=deserialize_vec(track.mean_vec, self.embed_dim),
                p90_vec=deserialize_vec(track.p90_vec, self.embed_dim),
            ),
        )
