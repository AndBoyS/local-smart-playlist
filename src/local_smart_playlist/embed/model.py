"""Lazy CLAP model loading and window embedding."""


from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from local_smart_playlist.audio.features import batches

if TYPE_CHECKING:
    from transformers import ClapModel, ClapProcessor

MODEL_ID = "laion/clap-htsat-fused"
EMBED_DIM = 512


@lru_cache(maxsize=1)
def load_model() -> "tuple[ClapModel, ClapProcessor]":
    """Load CLAP model + processor once (downloads to the HF cache on first use)."""
    from transformers import AutoProcessor, ClapModel

    model = ClapModel.from_pretrained(MODEL_ID)
    _ = model.eval()
    device = pick_device()
    model = model.to(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


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


def _feature_tensor(output: object) -> Any:
    """transformers 5.x get_*_features returns a model output; extract the 512-d tensor."""
    import torch

    if isinstance(output, torch.Tensor):
        return output
    pooled = cast("Any", output).pooler_output  # pyrefly: ignore
    if not isinstance(pooled, torch.Tensor):
        msg = f"unexpected get_*_features output type {type(output).__name__}"
        raise TypeError(msg)
    return pooled


def embed_windows(windows: list[np.ndarray], *, batch_size: int = 16) -> np.ndarray:
    """Embed disjoint windows; returns (n_windows, EMBED_DIM), L2-normalized."""
    if len(windows) == 0:
        return np.empty((0, EMBED_DIM), dtype=np.float32)
    import torch

    model, processor = load_model()
    device = pick_device()
    out = np.empty((len(windows), EMBED_DIM), dtype=np.float32)
    with torch.no_grad():
        for i, batch in enumerate(batches(windows, batch_size=batch_size)):
            inputs = processor(
                audio=batch,
                sampling_rate=48_000,  # pyrefly: ignore
                return_tensors="pt",  # pyrefly: ignore
                padding="max_length",  # pyrefly: ignore
            )
            inputs = {k: v.to(device) for k, v in cast("Any", inputs).items()}
            features = model.get_audio_features(**inputs)  # pyrefly: ignore
            raw = cast("Any", _feature_tensor(features).cpu().numpy())  # pyrefly: ignore  # pyrefly: ignore
            out[i * batch_size : i * batch_size + len(batch)] = np.asarray(raw, dtype=np.float32)
    return _unit(out)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed text prompts; returns (n_texts, EMBED_DIM), L2-normalized."""
    if len(texts) == 0:
        return np.empty((0, EMBED_DIM), dtype=np.float32)
    import torch

    model, processor = load_model()
    device = pick_device()
    with torch.no_grad():
        inputs = processor(
            text=list(texts),
            return_tensors="pt",  # pyrefly: ignore
            padding=True,  # pyrefly: ignore
        )
        inputs = {k: v.to(device) for k, v in cast("Any", inputs).items()}
        features = model.get_text_features(**inputs)  # pyrefly: ignore
        raw = cast("Any", _feature_tensor(features).cpu().numpy())  # pyrefly: ignore
    return _unit(np.asarray(raw, dtype=np.float32))


def cap_torch_threads(count: int = 2) -> None:
    """Cap CPU threads; heavy compute runs on the GPU device instead."""
    import torch

    torch.set_num_threads(count)
