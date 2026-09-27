"""Cache selection tests for pretrained model components."""

import json
from pathlib import Path

import pytest

from local_smart_playlist.embed import model as embed_model


def test_hub_offline_requires_cached_nested_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = tmp_path / "mulan"
    snapshot.mkdir()
    _ = (snapshot / "config.json").write_text(
        json.dumps({"audio_model": {"name": "audio-model"}, "text_model": {"name": "text-model"}}),
        encoding="utf-8",
    )
    cached_models: set[str] = {"text-model"}

    def snapshot_for(model_id: str) -> Path | None:
        return tmp_path / model_id if model_id in cached_models else None

    monkeypatch.setattr(embed_model, "_snapshot_dir", snapshot_for)

    assert embed_model._hub_submodels_cached(snapshot, text_only=True)
    assert not embed_model._hub_submodels_cached(snapshot, text_only=False)

    cached_models.add("audio-model")
    assert embed_model._hub_submodels_cached(snapshot, text_only=False)
