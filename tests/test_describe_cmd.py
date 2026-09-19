"""`sp describe` tests: caption ranking, path resolution, CLI arg errors."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from local_smart_playlist.commands import describe_cmd
from local_smart_playlist.embed.model import MuLanEmbedder

if TYPE_CHECKING:
    import torch
from local_smart_playlist.index.store import Store
from local_smart_playlist.query import prompts

DIM = 8


def basis(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0  # pyrefly: ignore[unsupported-operation]  # numpy shape stubs lack __setitem__ (facebook/pyrefly#4901)
    return v


class _FakeTextModel:
    """MuQMuLan stand-in: 'caption A' -> e0, 'caption B' -> e1, 'caption C' -> e2, other -> hashed basis."""

    def __call__(
        self, *, texts: "list[str] | None" = None, wavs: "torch.Tensor | None" = None
    ) -> "torch.Tensor":
        """Deterministic basis vectors."""
        assert texts is not None
        out = np.empty((len(texts), DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            if text == "caption A":
                out[i] = basis(0)  # pyrefly: ignore[unsupported-operation]
            elif text == "caption B":
                out[i] = basis(1)  # pyrefly: ignore[unsupported-operation]
            elif text == "caption C":
                out[i] = basis(2)  # pyrefly: ignore[unsupported-operation]
            else:
                v = np.zeros(DIM, dtype=np.float32)
                v[sum(ord(c) for c in text) % DIM] = 1.0  # pyrefly: ignore[unsupported-operation]
                out[i] = v  # pyrefly: ignore[unsupported-operation]
        import torch

        return torch.from_numpy(out)


def fake_embedder() -> MuLanEmbedder[8]:
    """MuLanEmbedder bound to the fake model; dim matches the fake store's embed_dim."""
    return MuLanEmbedder(_FakeTextModel(), dim=DIM)


def fake_store(db: Path) -> Store:
    """Store stub with a small embed dim matching the fake embedder."""
    return Store(db, embed_dim=DIM)


def upsert_track(store: Store, rel_path: str, mean_vec: np.ndarray) -> None:
    store.upsert(
        rel_path=rel_path,
        mean_vec=np.asarray(mean_vec, dtype=np.float32),
        p90_vec=np.asarray(mean_vec, dtype=np.float32),
        n_windows=3,
        duration=60.0,
        title=rel_path,
        model="fake",
        indexed_at="now",
    )


def test_rank_captions_orders_by_similarity() -> None:
    vocab = ["caption C", "caption B", "caption A"]
    mean = basis(0) + 0.5 * basis(1)
    mean = (mean / float(np.linalg.norm(mean))).astype(np.float32)
    ranked = describe_cmd.rank_captions(fake_embedder(), mean_vec=mean, vocab=vocab, top_n=2)
    assert ranked[0] == ("caption A", pytest.approx(0.8944, abs=1e-3))
    assert ranked[1][0] == "caption B"


def test_rank_captions_respects_top_n() -> None:
    vocab = ["caption A", "caption B"]
    ranked = describe_cmd.rank_captions(fake_embedder(), mean_vec=basis(1), vocab=vocab, top_n=1)
    assert ranked == [("caption B", 1.0)]


def test_resolve_rel_path_absolute_inside_root(tmp_path: Path) -> None:
    rel = describe_cmd.resolve_rel_path(str(tmp_path / "a" / "song.flac"), tmp_path)
    assert rel == "a/song.flac"


def test_resolve_rel_path_cwd_relative(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    rel = describe_cmd.resolve_rel_path("a/song.flac", tmp_path)
    assert rel == "a/song.flac"


def test_resolve_rel_path_already_relative_falls_back(tmp_path: Path) -> None:
    rel = describe_cmd.resolve_rel_path("a/song.flac", tmp_path / "other-root")
    assert rel == "a/song.flac"


def test_run_missing_index(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no index at"):
        describe_cmd.DescribeArgs.run(path="x.flac", db=tmp_path / "nope.db", n=10)


def test_run_no_library_root(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    with Store(db):
        pass
    with pytest.raises(SystemExit, match="no library root"):
        describe_cmd.DescribeArgs.run(path="x.flac", db=db, n=10)


def test_run_track_not_in_index(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(describe_cmd, "load_model", fake_embedder)
    monkeypatch.setattr(describe_cmd, "Store", fake_store)
    root = tmp_path / "lib"
    root.mkdir()
    db = tmp_path / "index.db"
    with Store(db, embed_dim=DIM) as store:
        store.set_meta("library_root", str(root))
        upsert_track(store, "other.flac", basis(0))
    with pytest.raises(SystemExit, match="track not in index"):
        describe_cmd.DescribeArgs.run(path=str(root / "missing.flac"), db=db, n=10)


def test_run_describes_track(monkeypatch: Any, tmp_path: Path, capsys: Any) -> None:
    monkeypatch.setattr(describe_cmd, "load_model", fake_embedder)
    monkeypatch.setattr(describe_cmd, "Store", fake_store)
    root = tmp_path / "lib"
    root.mkdir()
    db = tmp_path / "index.db"
    with Store(db, embed_dim=DIM) as store:
        store.set_meta("library_root", str(root))
        upsert_track(store, "song.flac", basis(0))

    monkeypatch.setattr(prompts, "caption_vocab", lambda: ["caption A", "caption B"])
    describe_cmd.DescribeArgs.run(path=str(root / "song.flac"), db=db, n=2)
    out = capsys.readouterr().out
    assert "song.flac" in out
    assert "caption A" in out


def test_caption_vocab_strips_blanks() -> None:
    vocab = prompts.caption_vocab()
    assert len(vocab) >= 1
    assert all(line == line.strip() for line in vocab)
