"""Mood word → LLM caption expansion (OpenAI-compatible endpoint)."""

import os
from collections.abc import Sequence
from importlib import resources
from typing import Any, cast

DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"  # OpenCode Go subscription
DEFAULT_MODEL = "deepseek-v4-flash"


def _load_vocab() -> str:
    """The readout vocabulary — captions in the embedding model's text-caption style."""
    return resources.files("local_smart_playlist.data").joinpath("caption_vocab.txt").read_text(encoding="utf-8")


def caption_vocab() -> list[str]:
    """Readout vocabulary lines, blank-stripped, `#` comments dropped."""
    return [
        line.strip() for line in _load_vocab().splitlines() if line.strip() != "" and not line.lstrip().startswith("#")
    ]


_ADAPT_PROMPT = (
    "You rewrite a user's music mood query as ONE attribute caption in the "
    "format used to train a music-text embedding model (MuQ-MuLan): "
    "comma-separated lowercase attributes — mood, genre, instrument, tempo, "
    "vocals — ending with a period. Example input 'sad rainy morning' → "
    "output 'melancholic mood, slow tempo, sparse piano.'.\n\n"
    "Translate non-English text, fix typos, normalize phrasing. Express every "
    "meaning element of the input as an attribute; never add styles, "
    "instruments or moods the input does not imply. Keep it one short line, "
    "no quotes, no commentary, nothing but the caption."
)


class LlmError(Exception):
    """Raised when the LLM endpoint is unreachable or returns unusable output."""


def _settings() -> tuple[str, str, str]:
    base_url = os.environ.get("SP_LLM_BASE_URL", DEFAULT_BASE_URL)
    model = os.environ.get("SP_LLM_MODEL", DEFAULT_MODEL)
    sp_key = os.environ.get("SP_LLM_API_KEY")
    oc_key = os.environ.get("OPENCODE_API_KEY")
    if sp_key is not None:
        api_key = sp_key
    elif oc_key is not None:
        api_key = oc_key
    else:
        api_key = ""
    return base_url, model, api_key


def adapt_mood(mood: str) -> str:
    """Correct *mood* into one embedding-friendly phrase; raises LlmError on failure."""
    import uuid

    import httpx

    base_url, model, api_key = _settings()
    headers = {"x-opencode-session": uuid.uuid4().hex}
    if api_key != "":
        headers["Authorization"] = f"Bearer {api_key}"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _ADAPT_PROMPT},
            {"role": "user", "content": f"Query: {mood}"},
        ],
        "temperature": 0.2,
    }
    try:
        resp = httpx.post(f"{base_url.rstrip('/')}/chat/completions", json=body, headers=headers, timeout=60.0)
        _ = resp.raise_for_status()
        payload = cast("Any", resp.json())
        content = payload["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001 — surface any endpoint failure as LlmError
        msg = f"LLM adaptation failed: {exc}"
        raise LlmError(msg) from exc

    text = cast("str", content)
    lines = [line.strip().lstrip("-*• ").strip() for line in text.splitlines()]
    lines = [line for line in lines if line != ""]
    if len(lines) == 0:
        raise LlmError("LLM adaptation returned no text")
    return lines[0]


def fallback_prompts(mood: str) -> Sequence[str]:
    """Deterministic offline expansion used when --llm is off."""
    return [mood]
