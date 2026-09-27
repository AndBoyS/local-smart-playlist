"""Mood query → LLM correction (OpenAI-compatible endpoint).

Free text becomes a small set of comma-attribute query variants; existing
comma-attribute lists pass through unchanged (typos/translation only).
"""

import logging
import os
import re
import time

from local_smart_playlist.const import CAPTION_VOCAB_PATH
from local_smart_playlist.type_utils import NonEmptyTuple

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"  # OpenCode Go subscription
DEFAULT_MODEL = "deepseek-v4-flash"


def caption_vocab() -> NonEmptyTuple[str]:
    """Readout vocabulary lines, blank-stripped, `#` comments dropped."""
    vocab_text = CAPTION_VOCAB_PATH.read_text(encoding="utf-8")
    return NonEmptyTuple(
        line.strip() for line in vocab_text.splitlines() if line.strip() != "" and not line.lstrip().startswith("#")
    )


ADAPT_PROMPT = (
    "You adapt a user's music request for MuQ-MuLan, a music-text embedding "
    "model whose captions use short, comma-separated lowercase attributes.\n\n"
    "For an input that is already a comma-separated attribute list (e.g. "
    "'dreamy, melancholic'), return exactly one line, unchanged except for "
    "translation or typo fixes. Never expand or reorder it.\n\n"
    "For other inputs, infer the musical qualities reasonably implied by "
    "the user's intent, then write distinct, faithful query variants. Include "
    "as many as are meaningfully different; there is no fixed count. Each "
    "variant must be a concise, comma-separated attribute "
    "caption ending with a period. Use only attributes implied by the request; "
    "do not invent a specific genre, instrument, or vocal style. Make variants "
    "usefully different, not paraphrases. Example: 'music to train' can imply "
    "energetic, driving, or steady rhythmic music, but not one specific genre.\n\n"
    "Output one query per line, no bullets, numbering, quotes, or commentary."
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


def _adapt_prompt() -> str:
    """Add evenly spaced readout-caption examples as a model-specific style reference."""
    vocab = caption_vocab()
    count = min(12, len(vocab))
    if count == 0:
        return ADAPT_PROMPT
    if count == 1:
        examples = vocab
    else:
        indices = [i * (len(vocab) - 1) // (count - 1) for i in range(count)]
        examples = [vocab[i] for i in indices]
    return f"{ADAPT_PROMPT}\n\nReadout-caption style examples:\n" + "\n".join(examples)


def adapt_mood(mood: str) -> list[str]:
    """Adapt *mood* into embedding-friendly query variants; raises LlmError on failure."""
    import uuid

    import httpx

    base_url, model, api_key = _settings()
    headers = {"x-opencode-session": uuid.uuid4().hex}
    if api_key != "":
        headers["Authorization"] = f"Bearer {api_key}"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _adapt_prompt()},
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
    except Exception as exc:
        logger.warning("llm adapt failed for %r via %s/%s: %s", mood, base_url, model, exc)
        msg = f"LLM adaptation failed: {exc}"
        raise LlmError(msg) from exc

    lines = [re.sub(r"^(?:[-*•]\s*|\d+[.)]\s*)", "", line.strip()).strip("`\"'") for line in content.splitlines()]
    lines = list(dict.fromkeys(line for line in lines if line != ""))
    if len(lines) == 0:
        logger.warning("llm adapt returned no text for %r via %s/%s", mood, base_url, model)
        raise LlmError("LLM adaptation returned no text")
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    logger.info("llm adapt: %r -> %r (%s, %.0f ms)", mood, lines, model, elapsed_ms)
    return lines
