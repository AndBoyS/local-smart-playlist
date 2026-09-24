"""Mood query → LLM correction (OpenAI-compatible endpoint).

Free text becomes a comma-attribute caption; existing comma-attribute lists
pass through unchanged (typos/translation only)
"""

import logging
import os
import time
from importlib import resources

logger = logging.getLogger(__name__)

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
    "You rewrite a user's music mood query for a music-text embedding model "
    "(MuQ-MuLan), trained on comma-separated lowercase attribute captions.\n\n"
    "If the input is already a comma-separated attribute list (e.g. "
    "'dreamy, melancholic'), return it UNCHANGED: translate non-English text "
    "and fix typos only. Never append 'mood' or a period, never reword, "
    "reorder or expand it — adding words measurably worsens retrieval.\n\n"
    "If the input is free text (a sentence, a scene, a vibe), express every "
    "meaning element of it as comma-separated lowercase attributes — mood, "
    "genre, instrument, tempo, vocals — ending with a period. Example input "
    "'sad rainy morning' → output 'rainy, melancholic, morning.'. Never add "
    "styles, instruments or moods the input does not imply.\n\n"
    "Keep the output one short line, no quotes, no commentary, nothing but "
    "the rewritten query."
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
        started = time.perf_counter()
        resp = httpx.post(f"{base_url.rstrip('/')}/chat/completions", json=body, headers=headers, timeout=60.0)
        _ = resp.raise_for_status()
        payload = resp.json()
        content = payload["choices"][0]["message"]["content"]
        assert isinstance(content, str)
    except Exception as exc:  # noqa: BLE001 — surface any endpoint failure as LlmError
        logger.warning("llm adapt failed for %r via %s/%s: %s", mood, base_url, model, exc)
        msg = f"LLM adaptation failed: {exc}"
        raise LlmError(msg) from exc

    lines = [line.strip().lstrip("-*• ").strip() for line in content.splitlines()]
    lines = [line for line in lines if line != ""]
    if len(lines) == 0:
        logger.warning("llm adapt returned no text for %r via %s/%s", mood, base_url, model)
        raise LlmError("LLM adaptation returned no text")
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    logger.info("llm adapt: %r -> %r (%s, %.0f ms)", mood, lines[0], model, elapsed_ms)
    return lines[0]
