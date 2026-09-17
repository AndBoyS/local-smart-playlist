"""Lazy MuQ-MuLan model loading and window embedding."""

from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import soxr

from local_smart_playlist.audio.features import batches

if TYPE_CHECKING:
    from muq import MuQMuLan  # pyrefly: ignore[implicit-reexport]

MODEL_ID = "OpenMuQ/MuQ-MuLan-large"
EMBED_DIM = 512
MODEL_SR = 24_000  # MuQ-MuLan strictly requires 24 kHz input


@lru_cache(maxsize=1)
def load_model() -> "MuQMuLan":
    """Load MuQ-MuLan once (downloads the checkpoint from the HF cache on first use)."""
    from muq import MuQMuLan  # pyrefly: ignore[implicit-reexport]

    model = MuQMuLan.from_pretrained(MODEL_ID)
    _ = model.eval()
    _ = model.to(pick_device())
    return model


def pick_device() -> str:
    """Best available torch device: MPS (Apple GPU), else CPU."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)  # pyrefly: ignore
    return np.asarray(vectors / safe, dtype=np.float32)


def _to_24k(windows: list[np.ndarray]) -> list[np.ndarray]:
    """Resample 48 kHz windows to the model's 24 kHz input rate (one shot per window)."""
    return [np.asarray(soxr.resample(w, 48_000, MODEL_SR), dtype=np.float32) for w in windows]


def embed_windows(windows: list[np.ndarray], *, batch_size: int = 16) -> np.ndarray:
    """Embed disjoint windows; returns (n_windows, EMBED_DIM), L2-normalized."""
    if len(windows) == 0:
        return np.empty((0, EMBED_DIM), dtype=np.float32)
    import torch

    model = load_model()
    device = pick_device()
    out = np.empty((len(windows), EMBED_DIM), dtype=np.float32)
    with torch.no_grad():
        for i, batch in enumerate(batches(windows, batch_size=batch_size)):
            wavs_24k = _to_24k(batch)
            wavs = torch.tensor(np.stack(wavs_24k), dtype=torch.float32).to(device)
            embeds = model(wavs=wavs)
            raw: Any = cast("Any", embeds).cpu().numpy()  # pyrefly: ignore
            # numpy shape stubs lack __setitem__ (facebook/pyrefly#4901); slice-assign is valid at runtime.
            out[i * batch_size : i * batch_size + len(batch)] = np.asarray(  # pyrefly: ignore[unsupported-operation]
                raw, dtype=np.float32
            )
    return _unit(out)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed text prompts; returns (n_texts, EMBED_DIM), L2-normalized."""
    if len(texts) == 0:
        return np.empty((0, EMBED_DIM), dtype=np.float32)
    import torch

    model = load_model()
    with torch.no_grad():
        embeds = model(texts=list(texts))
        raw: Any = cast("Any", embeds).cpu().numpy()  # pyrefly: ignore
    return _unit(np.asarray(raw, dtype=np.float32))


def cap_torch_threads(count: int = 2) -> None:
    """Cap CPU threads; heavy compute runs on the GPU device instead."""
    import torch

    torch.set_num_threads(count)
