"""MuQ-MuLan model loading and window/text embedding."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Protocol, cast, overload

import numpy as np
import soxr
import torch
from shape_extensions import Int, IntTuple, IntVar
from torch import nn

from local_smart_playlist.audio.features import batches
from local_smart_playlist.numpy_helpers import l2_normalize

W = IntVar("W")  # samples per window at 48 kHz
W24 = IntVar("W24")  # samples per window after resample to 24 kHz

MODEL_ID = "OpenMuQ/MuQ-MuLan-large"
EMBED_DIM = 512
MODEL_SR = 24_000  # MuQ-MuLan strictly requires 24 kHz input


def pick_device() -> str:
    """Best available torch device: MPS (Apple GPU), else CPU."""
    from torch.backends import mps

    if mps.is_available():
        return "mps"
    return "cpu"


def _to_24k(windows: list[np.ndarray[[W]]]) -> list[np.ndarray[[W24]]]:
    """Resample 48 kHz windows to the model's 24 kHz input rate (one shot per window)."""
    return [np.asarray(soxr.resample(w, 48_000, MODEL_SR), dtype=np.float32) for w in windows]


class MuLanModel[D: IntVar](Protocol):
    """Forward protocol for the MuQ-MuLan module: shaped embedding output."""

    @overload
    def __call__[B3: IntVar](self, *, wavs: torch.Tensor) -> torch.Tensor[[B3, D]]: ...
    @overload
    def __call__[N3: IntVar](self, *, texts: list[str]) -> torch.Tensor[[N3, D]]: ...
    def __call__[B3: IntVar, N3: IntVar](
        self, *, texts: list[str] | None = None, wavs: torch.Tensor | None = None
    ) -> torch.Tensor:
        pass


def torch_to_numpy[Shape: IntTuple](tensor: torch.Tensor[Shape]) -> np.ndarray[Shape]:
    """Detach a shaped torch tensor to a numpy array, preserving type shape."""
    raw: np.ndarray[Shape] = tensor.cpu().numpy()  # pyrefly: ignore[bad-argument-type]
    return raw


class MuLanEmbedder[D: IntVar]:
    """MuQ-MuLan model bound to its embedding dimension.

    Load once with :func:`load_model` and pass the embedder around
    explicitly — no hidden global state. ``self.dim`` carries the embedding
    width into the type system: the embed methods return
    ``np.ndarray[[N, D]]`` with D fixed by the loaded model.
    """

    def __init__(self, model: MuLanModel[D], *, dim: Int[D]) -> None:
        """*model* is the loaded MuQMuLan forward callable (torch boundary)."""
        self.model = model
        self.dim = dim

    def embed_texts[N2: IntVar](self, texts: list[str]) -> np.ndarray[[N2, D]]:
        """Embed text prompts; returns (n_texts, dim), L2-normalized."""
        if len(texts) == 0:
            return np.empty((len(texts), self.dim), dtype=np.float32)
        import torch

        with torch.no_grad():
            raw = torch_to_numpy(self.model(texts=list(texts)))
        return l2_normalize(raw)

    def embed_windows[W2: IntVar, N2: IntVar](
        self, windows: list[np.ndarray[[W2]]], *, batch_size: int = 16
    ) -> np.ndarray[[N2, D]]:
        """Embed disjoint windows; returns (n_windows, dim), L2-normalized."""
        if len(windows) == 0:
            return np.empty((len(windows), self.dim), dtype=np.float32)
        import torch

        device = pick_device()
        out = np.empty((len(windows), self.dim), dtype=np.float32)
        with torch.no_grad():
            for i, batch in enumerate(batches(windows, batch_size=batch_size)):
                wavs_24k = _to_24k(batch)
                wavs = torch.tensor(np.stack(wavs_24k), dtype=torch.float32).to(device)
                raw = torch_to_numpy(self.model(wavs=wavs))
                # numpy shape stubs lack __setitem__ (facebook/pyrefly#4901); slice-assign is valid at runtime.
                out[i * batch_size : i * batch_size + len(batch)] = raw  # pyrefly: ignore[unsupported-operation]

        return l2_normalize(out)


def _hub_cache_dir() -> Path:
    """HF cache dir, resolved the way huggingface_hub resolves it — without importing it.

    Mirrors ``huggingface_hub/constants.py``: HF_HUB_CACHE → legacy
    HUGGINGFACE_HUB_CACHE → $HF_HOME/hub → $XDG_CACHE_HOME (default
    ``~/.cache``) + huggingface + /hub. Importing the constants module first
    would freeze the ``HF_HUB_OFFLINE`` constant before load_model sets it.
    """
    import os

    hub_cache = os.environ.get("HF_HUB_CACHE")
    if hub_cache is None:
        hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache is None:
        hf_home = os.environ.get("HF_HOME")
        if hf_home is None:
            xdg = os.environ.get("XDG_CACHE_HOME")
            cache_root = Path(xdg) if xdg is not None else Path.home() / ".cache"
            hf_home = str(cache_root / "huggingface")
        hub_cache = str(Path(hf_home) / "hub")
    return Path(hub_cache).expanduser()


def _snapshot_dir(model_id: str) -> Path | None:
    """Local snapshot dir for *model_id*, or None when not (plausibly) cached.

    Pure filesystem check: importing huggingface_hub first would freeze its
    ``HF_HUB_OFFLINE`` constant (read at import time) before load_model can
    set it, re-enabling hub metadata checks in the tokenizer/sub-model loads.
    """

    repo_dir = _hub_cache_dir() / ("models--" + model_id.replace("/", "--"))
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    ref = repo_dir / "refs" / "main"
    candidates: list[Path] = []
    if ref.is_file():
        candidates.append(snapshots / ref.read_text(encoding="utf-8").strip())
    else:
        candidates.extend(s for s in snapshots.iterdir() if s.is_dir())
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate
    return None


def _load_muq_safetensors(snapshot: Path) -> nn.Module:
    """Build MuQMuLan directly and assign mmap'd safetensors weights (zero-copy).

    Bypasses ``MuQMuLan.from_pretrained``: its safetensors path re-copies the
    full 2.5 GB state dict into freshly initialized parameters (``copy_``),
    which costs several seconds. ``load_state_dict(assign=True)`` swaps the
    mmap-backed tensors in directly; ``.to(device)`` later still copies when
    the GPU device is in play. Requires ``model.safetensors`` in the snapshot
    (written once by :func:`_convert_checkpoint`).
    """
    from muq import MuQMuLan  # pyrefly: ignore[implicit-reexport]
    from safetensors.torch import load_file

    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    model: nn.Module = MuQMuLan(config=config)
    state = load_file(str(snapshot / "model.safetensors"))
    _ = model.load_state_dict(state, strict=False, assign=True)
    return model.eval()


def _load_muq_safetensors_text_only(snapshot: Path) -> nn.Module:
    """MuQ-MuLan text tower only: audio tower replaced by an inert stub.

    Text-only commands (play/describe) never embed audio, so skipping the
    audio transformer avoids its construction and ~2 GB of weights
    (construct+assign: 0.1 s vs 3.8 s, latents bit-identical). Mirrors
    ``create_MuLan_from_config``'s text half; audio weights stay unassigned
    (``audio_to_latents`` remains random and unused — calling this model
    with ``wavs=`` is undefined behavior by design). Also requires
    ``model.safetensors``.
    """
    from types import SimpleNamespace

    from muq import MuQMuLan  # pyrefly: ignore[implicit-reexport]
    from muq.muq_mulan.models.mulan import MuLanModel
    from muq.muq_mulan.models.text import TextTransformerPretrained
    from safetensors.torch import load_file

    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    text_transformer = TextTransformerPretrained(
        model_name=cast("str", config["text_model"]["name"]),
        model_dim=cast("int | None", config["text_model"]["model_dim"]),
        **cast("dict[str, Any]", config["text_transformer"]),
        frozen_pretrained=False,
    )
    audio_stub = SimpleNamespace(dim=cast("int", config["audio_transformer"]["dim"]), depth=0)
    mulan = MuLanModel(
        audio_transformer=cast("Any", audio_stub),
        text_transformer=cast("Any", text_transformer),
        **cast("dict[str, Any]", config["mulan"]),
    )
    model = MuQMuLan.__new__(MuQMuLan)
    nn.Module.__init__(model)
    model.config = config
    model.mulan = mulan
    model.sr = config["mulan"]["sr"]
    model.clip_secs = config["mulan"]["clip_secs"]

    state = load_file(str(snapshot / "model.safetensors"))
    state = {key: tensor for key, tensor in state.items() if not key.startswith("mulan.audio")}
    _ = model.load_state_dict(state, strict=False, assign=True)
    return model.eval()


def _convert_checkpoint(snapshot: Path) -> None:
    """One-time ``pytorch_model.bin → model.safetensors`` conversion, then delete the original.

    Atomic: write to a temp file, rename, and only then unlink the pickle —
    a failed or interrupted conversion leaves the cache untouched. The
    pickle is recoverable from the Hub, but a corrupt safetensors would
    otherwise be silently masked by a valid fallback.
    """
    from safetensors.torch import save_file

    source = snapshot / "pytorch_model.bin"
    target = snapshot / "model.safetensors"
    tmp = target.with_suffix(".safetensors.tmp")
    state: dict[str, torch.Tensor] = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    clean = {key: tensor.contiguous() for key, tensor in state.items()}
    save_file(clean, str(tmp), metadata={"format": "pt"})
    _ = tmp.replace(target)
    source.unlink()


def _from_hub() -> nn.Module:
    """``MuQMuLan.from_pretrained`` against the local cache; online download on cache miss."""
    from muq import MuQMuLan  # pyrefly: ignore[implicit-reexport]

    try:
        return MuQMuLan.from_pretrained(MODEL_ID, local_files_only=True).eval()
    except Exception:  # noqa: BLE001 — incomplete cache: fall back to the online download path
        _ = os.environ.pop("HF_HUB_OFFLINE", None)
        return MuQMuLan.from_pretrained(MODEL_ID).eval()


def _load_weights(snapshot: Path | None, *, text_only: bool = False) -> nn.Module:
    """Load MuQ-MuLan weights: safetensors fast path, else ``from_pretrained``.

    Pickle-only snapshot: convert to safetensors after loading (original
    deleted after a clean write), so every later run takes the fast path.
    """
    if snapshot is not None and (snapshot / "model.safetensors").is_file():
        try:
            loader = _load_muq_safetensors_text_only if text_only else _load_muq_safetensors
            return loader(snapshot)
        # corrupt/mismatched safetensors: hub loading recovers
        except Exception:
            logging.getLogger(__name__).warning("safetensors load failed; using from_pretrained", exc_info=True)
    model = _from_hub()
    if snapshot is not None:
        try:
            _convert_checkpoint(snapshot)
        except Exception:
            logging.getLogger(__name__).debug("checkpoint conversion skipped", exc_info=True)
    return model


def load_model(*, device: str | None = None, text_only: bool = False) -> MuLanEmbedder[512]:
    """Load MuQ-MuLan once per run; local HF cache first, downloads on first use only.

    Every run against a converted snapshot takes the fast path: mmap'd
    safetensors weights assigned in place, no pickle parse, no 2.5 GB copy.
    ``text_only=True`` skips the audio tower entirely (play/describe never
    embed audio) — construction and memory drop accordingly. When the
    snapshot is cached, ``HF_HUB_OFFLINE`` is set before the muq import so
    the tokenizer and sub-model loads skip hub metadata checks. Pass
    ``device="cpu"`` for text-only paths (embedding large audio batches is
    the only case where the GPU device wins).
    """
    snapshot = _snapshot_dir(MODEL_ID)
    if snapshot is not None:
        _ = os.environ.setdefault("HF_HUB_OFFLINE", "1")
    model = _load_weights(snapshot, text_only=text_only)
    _ = model.to(device if device is not None else pick_device())
    return MuLanEmbedder(model, dim=EMBED_DIM)


def cap_torch_threads(count: int = 2) -> None:
    """Cap CPU threads; heavy compute runs on the GPU device instead."""
    import torch

    torch.set_num_threads(count)
