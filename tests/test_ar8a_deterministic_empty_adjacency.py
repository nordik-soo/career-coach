"""AR-8a tests: deterministic empty-adjacency early-return in
`compose_response_v2`.

Live observation (2026-06-10): on a `recommend_adjacent_roles` turn
with `recommendations: []`, the LLM was called and improvised
"The one role we had isn't quite the fit you're looking for right
now..." -- narrating around zero data and inventing the "one role"
claim. The deterministic fallback already handles the empty case;
the fix is to short-circuit to it BEFORE the LLM ever sees the
empty payload.

Contract (per AR-8a sign-off):
  - Emptiness is defined the SAME way the fallback defines valid
    recommendations (`_valid_adjacent_recommendations`): no payload,
    payload not a dict, recommendations not a list, no list entry
    is a dict with a non-empty string title.
  - On emptiness: skip LLM call AND policy check entirely; return
    the deterministic fallback output verbatim.
  - On at least one valid entry: normal LLM path runs (the gate
    does NOT fire).
  - Applies ONLY to `recommend_adjacent_roles` -- other moves keep
    their existing paths.
"""
from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.nodb


from skillbridge.chat.arbiter import ArbiterDecision
from skillbridge.chat.responder import (
    ResponderV2Input,
    _recommend_adjacent_roles_fallback_v2,
    _valid_adjacent_recommendations,
    compose_response_v2,
)
from skillbridge.chat.url_views import (
    build_sanitized_responder_view_v2 as _v_v2,
)


# =========================================================================
# Helpers
# =========================================================================
def _decision(move: str) -> ArbiterDecision:
    return ArbiterDecision(
        final_move=move,
        reason_code="x",
        tone="brief_confident",
        arbiter_action="handler_synthesized_adjacent_recommendations",
        ask_slot=None,
    )


def _input(move: str, **payloads) -> ResponderV2Input:
    return ResponderV2Input(
        user_message="what other roles?",
        decision=_decision(move),
        results=[],
        training_by_job={},
        next_skill=(None, 0),
        band_signal="none",
        requires_consent=True,
        target_role_text="warehouse worker",
        resume_facts=None,
        conversation_context=None,
        **payloads,
    )


class _LLMSpy:
    """Fail-fast spy: any call records and (optionally) raises."""

    def __init__(self, *, must_not_run: bool, return_text: str = "stub-llm"):
        self.must_not_run = must_not_run
        self.return_text = return_text
        self.calls: list[tuple[Any, ...]] = []

    def __call__(self, system: str, user: str, max_tokens: int = 500) -> str:
        self.calls.append((system, user, max_tokens))
        if self.must_not_run:
            pytest.fail("LLM `call` MUST NOT run on this code path")
        return self.return_text


class _PolicySpy:
    def __init__(self, *, must_not_run: bool, return_value: bool = True):
        self.must_not_run = must_not_run
        self.return_value = return_value
        self.calls: list[tuple[Any, ...]] = []

    def __call__(self, reply: str, inp: ResponderV2Input, view=None) -> bool:
        # sub-step 4 signature change: _policy_ok_v2 now takes (reply, inp, view).
        self.calls.append((reply, inp))
        if self.must_not_run:
            pytest.fail("`_policy_ok_v2` MUST NOT run on this code path")
        return self.return_value


# =========================================================================
# _valid_adjacent_recommendations unit semantics
# =========================================================================
@pytest.mark.parametrize("payload, expected_count", [
    (None,                                              0),
    ({},                                                0),
    ({"recommendations": None},                         0),
    ({"recommendations": "not a list"},                 0),
    ({"recommendations": 42},                           0),
    ({"recommendations": []},                           0),
    ({"recommendations": [None]},                       0),
    ({"recommendations": ["string-not-dict"]},          0),
    ({"recommendations": [{}]},                         0),
    ({"recommendations": [{"title": None}]},            0),
    ({"recommendations": [{"title": ""}]},              0),
    ({"recommendations": [{"title": "   "}]},           0),
    ({"recommendations": [{"title": 123}]},             0),
    ({"recommendations": [{"title": "Welder"}]},        1),
    ({"recommendations": [
        {"title": "Welder"},
        {},
        {"title": "Forklift Op"},
    ]},                                                 2),
])
def test_valid_adjacent_recommendations_pins_validity_rule(payload, expected_count):
    """Validity rule MUST be: dict payload, list recommendations, each
    entry a dict with a non-empty string title. Anything else drops."""
    assert len(_valid_adjacent_recommendations(payload)) == expected_count


# =========================================================================
# Early-return: empty/missing/malformed -> deterministic fallback,
# LLM and policy NEVER called
# =========================================================================
@pytest.mark.parametrize("payload", [
    None,
    {},
    {"recommendations": None},
    {"recommendations": "not a list"},
    {"recommendations": []},
    {"recommendations": [{}]},                  # malformed: no title
    {"recommendations": [{"title": ""}]},       # malformed: empty title
    {"recommendations": [{"title": "  "}]},     # malformed: whitespace title
    {"recommendations": [{"title": 123}]},      # malformed: non-string title
    {"recommendations": ["not-a-dict"]},        # malformed: not a dict
])
def test_empty_payload_bypasses_llm_and_policy(monkeypatch, payload):
    """For every empty-shape payload, compose_response_v2 MUST:
       (a) return the deterministic fallback's verbatim text;
       (b) never call the LLM;
       (c) never call the policy gate.
    """
    from skillbridge.chat import responder

    llm_spy = _LLMSpy(must_not_run=True)
    policy_spy = _PolicySpy(must_not_run=True)
    monkeypatch.setattr(responder, "is_enabled", lambda: True)
    monkeypatch.setattr(responder, "call", llm_spy)
    monkeypatch.setattr(responder, "_policy_ok_v2", policy_spy)

    inp = _input("recommend_adjacent_roles",
                 adjacent_recommendations_payload=payload)
    reply = compose_response_v2(inp)

    # Same string the fallback returns on empty: identity check is the
    # strongest pin (rules out any "looks empty but went through LLM"
    # masquerade).
    expected = _recommend_adjacent_roles_fallback_v2(inp, _v_v2(inp))
    assert reply == expected
    assert llm_spy.calls == []
    assert policy_spy.calls == []


def test_empty_payload_returns_locked_empty_result_line(monkeypatch):
    """Beyond the identity check above, pin the exact wording of the
    empty-result line so a future fallback rewrite has to update this
    test rather than silently regress live messaging."""
    from skillbridge.chat import responder

    monkeypatch.setattr(responder, "is_enabled", lambda: True)
    monkeypatch.setattr(responder, "call",
                        _LLMSpy(must_not_run=True))
    monkeypatch.setattr(responder, "_policy_ok_v2",
                        _PolicySpy(must_not_run=True))

    inp = _input("recommend_adjacent_roles",
                 adjacent_recommendations_payload={"recommendations": []})
    reply = compose_response_v2(inp)
    assert "From today's Sault Ste. Marie postings" in reply
    assert "I'm not seeing other roles" in reply
    # Forbidden-vocabulary hold-the-line on the empty path: even though
    # the policy gate isn't called, the deterministic text MUST NOT
    # contain candidate / fit framing.
    lower = reply.lower()
    for forbidden in ("good fit", "perfect fit", "strong candidate",
                      "you qualify", "ideal for you"):
        assert forbidden not in lower


# =========================================================================
# Non-empty: LLM path runs as before
# =========================================================================
def test_non_empty_recommendations_calls_llm_and_policy(monkeypatch):
    """At least one valid recommendation -> normal LLM path.
    Early-return MUST NOT fire."""
    from skillbridge.chat import responder

    llm_spy = _LLMSpy(must_not_run=False, return_text="LLM-narrated reply")
    policy_spy = _PolicySpy(must_not_run=False, return_value=True)
    monkeypatch.setattr(responder, "is_enabled", lambda: True)
    monkeypatch.setattr(responder, "call", llm_spy)
    monkeypatch.setattr(responder, "_policy_ok_v2", policy_spy)

    inp = _input("recommend_adjacent_roles",
                 adjacent_recommendations_payload={
                     "recommendations": [
                         {"title": "Forklift Operator",
                          "employer": "ACME",
                          "evidence_summary": "3 of 5"},
                     ],
                     "total_retrieved": 5,
                 })
    reply = compose_response_v2(inp)
    assert reply == "LLM-narrated reply"
    assert len(llm_spy.calls) == 1
    assert len(policy_spy.calls) == 1


def test_mixed_valid_and_malformed_takes_llm_path(monkeypatch):
    """A list with at least one valid entry (and any number of
    malformed siblings) is NOT empty; LLM path runs. The fallback,
    if reached, would still filter the malformed entries."""
    from skillbridge.chat import responder

    llm_spy = _LLMSpy(must_not_run=False, return_text="LLM-narrated")
    policy_spy = _PolicySpy(must_not_run=False, return_value=True)
    monkeypatch.setattr(responder, "is_enabled", lambda: True)
    monkeypatch.setattr(responder, "call", llm_spy)
    monkeypatch.setattr(responder, "_policy_ok_v2", policy_spy)

    inp = _input("recommend_adjacent_roles",
                 adjacent_recommendations_payload={
                     "recommendations": [
                         {},                                # invalid
                         {"title": ""},                     # invalid
                         {"title": "Maintenance Tech"},     # valid
                     ],
                     "total_retrieved": 3,
                 })
    reply = compose_response_v2(inp)
    assert reply == "LLM-narrated"
    assert len(llm_spy.calls) == 1


# =========================================================================
# Scope: only applies to recommend_adjacent_roles
# =========================================================================
@pytest.mark.parametrize("move", [
    "present_matches",
    "present_no_match",
    "present_near_miss",
    "describe_adjacent_role",
    "explain_gap",
    "ask_one_clarifying_question",
    "redirect_scope",
    "acknowledge_and_continue",
    "offer_refinement",
    "explain_remaining_gaps",
])
def test_non_adjacency_moves_unaffected_by_ar8a_guard(monkeypatch, move):
    """The early-return MUST be scoped to `recommend_adjacent_roles`.
    Other moves keep their existing path -- LLM call runs even when no
    adjacency payload is present (because they don't use that payload)."""
    from skillbridge.chat import responder

    llm_spy = _LLMSpy(must_not_run=False, return_text="other-move-reply")
    policy_spy = _PolicySpy(must_not_run=False, return_value=True)
    monkeypatch.setattr(responder, "is_enabled", lambda: True)
    monkeypatch.setattr(responder, "call", llm_spy)
    monkeypatch.setattr(responder, "_policy_ok_v2", policy_spy)

    # Some moves have their own short-circuits (e.g. clarification);
    # we only need to assert that AR-8a's guard doesn't fire. We do
    # that by passing no adjacency payload at all and observing that
    # the LLM path was either reached OR a non-AR-8a deterministic
    # path was hit. AR-8a's deterministic empty-recs fallback would
    # produce a string containing "From today's Sault Ste. Marie
    # postings, I'm not seeing other roles" -- if we don't see that
    # exact wording for these non-adjacency moves, AR-8a stayed out
    # of their way.
    inp = _input(move)
    reply = compose_response_v2(inp)
    assert "I'm not seeing other roles where your current skills" not in reply
