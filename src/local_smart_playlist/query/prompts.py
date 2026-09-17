"""Mood word → LLM caption expansion (OpenAI-compatible endpoint)."""


import os
from collections.abc import Sequence
from typing import Any, cast

DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "llama3.2"

_SYSTEM_PROMPT = (
    "You expand a mood word or short mood phrase into candidate descriptions of "
    "how the matching music would sound. Reply with 5-10 lines, one description "
    "per line, no numbering, no extra commentary. Each line is a vivid caption in "
    "the style of a music tagger, describing tempo, instrumentation, vocals, mood "
    'and production, e.g. "a slow melancholic song with soft vocals, sparse piano '
    'and a minor key".'
)

MAX_PROMPTS = 10
MIN_PROMPTS = 5
MIN_CAPTION_LEN = 8


class LlmError(Exception):
    """Raised when the LLM endpoint is unreachable or returns unusable output."""


def _settings() -> tuple[str, str, str]:
    base_url = os.environ.get("SP_LLM_BASE_URL", DEFAULT_BASE_URL)
    model = os.environ.get("SP_LLM_MODEL", DEFAULT_MODEL)
    return base_url, model, os.environ.get("SP_LLM_API_KEY", "")


def expand_mood(mood: str) -> list[str]:
    """Ask the LLM for 5-10 caption-style prompts describing *mood* music."""
    import httpx

    base_url, model, api_key = _settings()
    headers = {}
    if api_key != "":
        headers["Authorization"] = f"Bearer {api_key}"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Mood: {mood}"},
        ],
        "temperature": 0.7,
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
