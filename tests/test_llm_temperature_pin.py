"""Pin temperature=0 on the shared LLM helper (skillbridge.llm.call).

Locked 2026-07-02. The shared call() helper is used by:
  - resume facts extraction (skillbridge/resume/extract.py)
  - chat evidence extractor (skillbridge/chat/extractor.py)
  - LLM skill extractor (skillbridge/extract/llm_based.py)
  - planner JSON
  - normal responder prose
  - v2 outcome responder prose
  - recommender responder prose
  - coach tiers responder prose

Anthropic's default temperature is 1.0. Same PDF re-uploaded across
sessions was producing different extracted skill lists ("accounts
receivable management" -> "accounts receivable aging"), which cascaded
into different Layer C adjacent-NOC surfaces because retrieve_candidates
keys off the extracted skills.

Test asserts client.messages.create receives temperature=0 unconditionally.
If someone silently removes the kwarg, the extraction drift returns.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.nodb


def test_call_passes_temperature_zero_to_messages_create(monkeypatch):
    """The shared call() helper MUST pass temperature=0 to
    client.messages.create. This is the load-bearing determinism
    pin for the entire shared-helper path."""
    from skillbridge import llm as _llm

    captured: dict = {}

    class _FakeResp:
        content = [type("Block", (), {"text": "ok"})()]

    class _FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeResp()

    class _FakeClient:
        messages = _FakeMessages()

    monkeypatch.setattr(_llm, "LLM_ENABLED", True)
    monkeypatch.setattr(_llm, "_client_get", lambda: _FakeClient())

    _llm.call("system prompt", "user text", max_tokens=100)

    assert "temperature" in captured, (
        "call() no longer passes temperature to messages.create -- the "
        "shared-helper stability pin is broken."
    )
    assert captured["temperature"] == 0, (
        f"temperature must be exactly 0 for stability, got {captured['temperature']!r}."
    )


def test_call_temperature_zero_survives_model_fallback(monkeypatch):
    """When the primary model returns 429/529/503, call() falls back to
    LLM_FALLBACK_MODEL via a recursive call. Temperature=0 must still
    apply on the retry path."""
    import anthropic
    from skillbridge import llm as _llm

    captured_temps: list = []
    call_count = {"n": 0}

    class _FakeResp:
        content = [type("Block", (), {"text": "ok"})()]

    class _FakeMessages:
        def create(self, **kwargs):
            captured_temps.append(kwargs.get("temperature"))
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First call: simulate overload -> triggers fallback
                raise anthropic.APIStatusError(
                    "overloaded",
                    response=type("R", (), {"status_code": 529, "headers": {}})(),
                    body=None,
                )
            return _FakeResp()

    class _FakeClient:
        messages = _FakeMessages()

    monkeypatch.setattr(_llm, "LLM_ENABLED", True)
    monkeypatch.setattr(_llm, "_client_get", lambda: _FakeClient())
    monkeypatch.setattr(_llm, "LLM_MODEL", "primary")
    monkeypatch.setattr(_llm, "LLM_FALLBACK_MODEL", "fallback")

    _llm.call("system", "user", max_tokens=50)

    # Both the primary attempt AND the fallback retry must have
    # temperature=0.
    assert call_count["n"] == 2, "Fallback path was not exercised."
    assert all(t == 0 for t in captured_temps), (
        f"Not all attempts had temperature=0: {captured_temps}"
    )


def test_call_json_inherits_temperature_zero(monkeypatch):
    """call_json is a thin wrapper around call() -- verifying it also
    ends up passing temperature=0 to messages.create."""
    from skillbridge import llm as _llm

    captured: dict = {}

    class _FakeResp:
        content = [type("Block", (), {"text": '{"skills": []}'})()]

    class _FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeResp()

    class _FakeClient:
        messages = _FakeMessages()

    monkeypatch.setattr(_llm, "LLM_ENABLED", True)
    monkeypatch.setattr(_llm, "_client_get", lambda: _FakeClient())

    result = _llm.call_json("system", "user")
    assert result == {"skills": []}
    assert captured.get("temperature") == 0
