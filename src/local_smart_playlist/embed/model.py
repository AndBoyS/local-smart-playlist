"""MuQ-MuLan model loading and window/text embedding."""

from typing import Protocol, overload

import numpy as np
import soxr
import torch
from shape_extensions import Int, IntTuple, IntVar

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
