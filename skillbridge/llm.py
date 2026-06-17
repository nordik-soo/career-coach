"""Anthropic client wrapper.

Defaults to Haiku 4.5 (`claude-haiku-4-5-20251001`) — Anthropic's cheap+fast
tier. Falls back to Sonnet on overload. Uses prompt caching on the system
prompt so repeated calls within ~5 minutes pay ~90% less for system tokens.

Hard rules enforced by the design:
- LLM extracts JSON; we validate before persisting.
- LLM never writes to Postgres directly.
- LLM never produces final match scores.
- LLM only narrates numbers that came from a tool/DB result.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    ANTHROPIC_API_KEY,
    LLM_ENABLED,
    LLM_FALLBACK_MODEL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TIMEOUT_SECONDS,
)

log = logging.getLogger(__name__)


_client: anthropic.Anthropic | None = None


def is_enabled() -> bool:
    return LLM_ENABLED


def _client_get() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY.startswith("PLACEHOLDER"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY missing or placeholder. "
                "Set LLM_ENABLED=false to use rule-based extractors only."
            )
        _client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY,
            timeout=LLM_TIMEOUT_SECONDS,
        )
    return _client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def call(
    system: str,
    user: str,
    *,
    max_tokens: int | None = None,
    model: str | None = None,
    cache_system: bool = True,
) -> str:
    """Single text completion. Returns assistant text or empty string."""
    if not LLM_ENABLED:
        return ""
    client = _client_get()
    chosen = model or LLM_MODEL
    tokens = max_tokens or LLM_MAX_TOKENS
    system_blocks: list[dict[str, Any]]
    if cache_system:
        system_blocks = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    else:
        system_blocks = [{"type": "text", "text": system}]
    try:
        resp = client.messages.create(
            model=chosen,
            max_tokens=tokens,
            system=system_blocks,
            messages=[{"role": "user", "content": user}],
        )
    except anthropic.APIStatusError as e:
        if e.status_code in (429, 529, 503) and chosen != LLM_FALLBACK_MODEL:
            log.warning("Model %s overloaded; falling back to %s", chosen, LLM_FALLBACK_MODEL)
            return call(system, user, max_tokens=tokens, model=LLM_FALLBACK_MODEL, cache_system=cache_system)
        raise
    if not resp.content:
        return ""
    return resp.content[0].text.strip()


def call_json(
    system: str,
    user: str,
    *,
    max_tokens: int | None = None,
) -> dict | None:
    """Same as call() but expects the model to return valid JSON.

    Returns a dict (or list wrapped) or None if parse fails. We don't retry on
    parse failures — caller decides whether to fall back to rule-based logic.
    """
    if not LLM_ENABLED:
        return None
    text = call(system, user, max_tokens=max_tokens)
    if not text:
        return None
    # Strip code fences if the model added them despite instructions.
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.warning("LLM returned non-JSON (%s): %s", e, text[:200])
        return None
