"""Mood word → LLM caption expansion (OpenAI-compatible endpoint)."""

import os
from collections.abc import Sequence
from importlib import resources
from typing import Any, cast

DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"  # OpenCode Go subscription
DEFAULT_MODEL = "deepseek-v4-flash"


def _load_vocab() -> str:
    """The readout vocabulary — captions mined from CLAP's training distribution."""
    return resources.files("local_smart_playlist.data").joinpath("caption_vocab.txt").read_text(encoding="utf-8")


_SYSTEM_PROMPT = (
    "You expand a mood word or short mood phrase into candidate descriptions of "
    "how music matching it could sound. Your output is fed to an audio-text "
    "embedding model (CLAP), so every line should be a caption in the style of "
    "the reference vocabulary below.\n\n"
    "Reference vocabulary (captions in the model's own language):\n"
    "---\n"
    "{vocab}\n"
    "---\n\n"
    "For the given mood: first output the 8 lines from the reference vocabulary "
    "that best match the mood, verbatim. Then write 8-12 NEW captions in the "
    "same style (full sentences or label-bags, tempo + instrumentation + vocals "
    "+ mood + production) covering as many readings as possible: every genre, "
    "era, instrumentation and energy level the phrase could plausibly describe. "
    "Aim for 18-20 lines total, one per line, no numbering, no commentary."
)


def _system_prompt() -> str:
    return _SYSTEM_PROMPT.format(vocab=_load_vocab())

MAX_PROMPTS = 20
MIN_PROMPTS = 5
MIN_CAPTION_LEN = 8


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


def expand_mood(mood: str) -> list[str]:
    """Ask the LLM for up to 20 caption-style prompts covering many readings of *mood*."""
    import uuid

    import httpx

    base_url, model, api_key = _settings()
    headers = {"x-opencode-session": uuid.uuid4().hex}
    if api_key != "":
        headers["Authorization"] = f"Bearer {api_key}"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": f"Mood: {mood}"},
        ],
        "temperature": 0.9,
    }
    try:
        resp = httpx.post(f"{base_url.rstrip('/')}/chat/completions", json=body, headers=headers, timeout=30.0)
        _ = resp.raise_for_status()
        payload = cast("Any", resp.json())
        content = payload["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001 — surface any endpoint failure as LlmError
        msg = f"LLM expansion failed: {exc}"
        raise LlmError(msg) from exc

    text = cast("str", content)
    prompts = [line.strip().lstrip("-*• ").strip() for line in text.splitlines()]
    prompts = [p for p in prompts if len(p) >= MIN_CAPTION_LEN][:MAX_PROMPTS]
    if len(prompts) < MIN_PROMPTS:
        msg = f"LLM returned only {len(prompts)} usable captions"
        raise LlmError(msg)
    return prompts


def fallback_prompts(mood: str) -> Sequence[str]:
    """Deterministic offline expansion used when --llm is off."""
    return [mood]
