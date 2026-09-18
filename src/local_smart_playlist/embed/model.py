"""MuQ-MuLan model loading and window/text embedding."""

from typing import Any, cast

import numpy as np
import soxr
from shape_extensions import Int, IntVar

from local_smart_playlist.audio.features import batches

W = IntVar("W")  # samples per window at 48 kHz
W24 = IntVar("W24")  # samples per window after resample to 24 kHz
N = IntVar("N")  # window count / row count

MODEL_ID = "OpenMuQ/MuQ-MuLan-large"
EMBED_DIM = 512
MODEL_SR = 24_000  # MuQ-MuLan strictly requires 24 kHz input


def pick_device() -> str:
    """Best available torch device: MPS (Apple GPU), else CPU."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _unit[N: IntVar, D: IntVar](vectors: np.ndarray[[N, D]]) -> np.ndarray[[N, D]]:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)  # pyrefly: ignore[unknown-argument-type]
    return vectors / safe


def _to_24k(windows: list[np.ndarray[[W]]]) -> list[np.ndarray[[W24]]]:
    """Resample 48 kHz windows to the model's 24 kHz input rate (one shot per window)."""
    return [np.asarray(soxr.resample(w, 48_000, MODEL_SR), dtype=np.float32) for w in windows]


class MuLanEmbedder[D: IntVar]:
    """MuQ-MuLan model bound to its embedding dimension.

    Load once with :func:`load_model` and pass the embedder around
    explicitly — no hidden global state. ``self.dim`` carries the embedding
    width into the type system: the embed methods return
    ``np.ndarray[[N, D]]`` with D fixed by the loaded model.
    """

    def __init__(self, model: Any, *, dim: Int[D]) -> None:
        """*model* is the loaded MuQMuLan module (torch boundary — stays untyped)."""
        self.model: Any = model
        self.dim = dim

    def embed_texts[N2: IntVar](self, texts: list[str]) -> np.ndarray[[N2, D]]:
        """Embed text prompts; returns (n_texts, dim), L2-normalized."""
        if len(texts) == 0:
            return np.empty((len(texts), self.dim), dtype=np.float32)
        import torch

        with torch.no_grad():
            embeds: torch.Tensor = self.model(texts=list(texts))
            raw: np.ndarray[[N2, D]] = embeds.cpu().numpy()
        return _unit(np.asarray(raw, dtype=np.float32))

    def embed_windows(self, windows: list[np.ndarray[[W]]], *, batch_size: int = 16) -> np.ndarray[[N, D]]:
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
                embeds = self.model(wavs=wavs)
                raw: Any = cast("Any", embeds).cpu().numpy()
                # numpy shape stubs lack __setitem__ (facebook/pyrefly#4901); slice-assign is valid at runtime.
                out[i * batch_size : i * batch_size + len(batch)] = (  # pyrefly: ignore[unsupported-operation]
                    np.asarray(raw, dtype=np.float32)
                )
        return _unit(out)


def load_model() -> MuLanEmbedder[512]:
    """Load MuQ-MuLan once per run (downloads the checkpoint from the HF cache on first use)."""
    from muq import MuQMuLan  # pyrefly: ignore[implicit-reexport]

    model = MuQMuLan.from_pretrained(MODEL_ID)
    _ = model.eval()
    _ = model.to(pick_device())
    return MuLanEmbedder(model, dim=EMBED_DIM)


def cap_torch_threads(count: int = 2) -> None:
    """Cap CPU threads; heavy compute runs on the GPU device instead."""
    import torch

    torch.set_num_threads(count)
