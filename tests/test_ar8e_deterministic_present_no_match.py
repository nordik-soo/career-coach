"""AR-8e tests: deterministic `present_no_match` in `compose_response_v2`.

Live observation (2026-06-10):
  WARNING skillbridge.chat.responder:
    policy v2: reply names ungrounded training provider
    'sault community career centre' (not in this turn's TRAINING block)
  WARNING skillbridge.chat.responder:
    Responder v2 reply failed policy check; falling back

Diagnosis:
  - OUTCOME_RESPONDER_PROMPT instructs the LLM to suggest Sault
    Community Career Centre on present_no_match turns.
  - `_policy_ok_v2`'s `_check_ungrounded_provider` rejects any SCCC
    mention NOT present in this turn's TRAINING block.
  - TRAINING is always empty on present_no_match.
  - Result: every present_no_match turn pays the LLM call + policy
    rejection + fallback. The fallback `_present_no_match_fallback_v2`
    itself names SCCC, so the user sees SCCC anyway -- via the
    deterministic path, not the LLM.

AR-8e fix: skip the LLM entirely on present_no_match. The
deterministic fallback is the contract; the LLM call was pure
waste plus a policy-warning log line.

Contract:
  - `compose_response_v2` early-returns to `_present_no_match_fallback_v2`
    on present_no_match BEFORE `is_enabled()`.
  - LLM `call` and `_policy_ok_v2` are NEVER invoked.
  - Behavior applies ONLY to present_no_match -- other moves still
    take the LLM path.
"""
from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.nodb


from skillbridge.chat.arbiter import ArbiterDecision
from skillbridge.chat.responder import (
    ResponderV2Input,
    _present_no_match_fallback_v2,
    compose_response_v2,
)
from skillbridge.chat.url_views import (
    build_sanitized_responder_view_v2 as _v_v2,
)


# =========================================================================
# Helpers
# =========================================================================
def _decision(move: str, **kw) -> ArbiterDecision:
    return ArbiterDecision(
        final_move=move,
        reason_code=kw.get("reason_code", "x"),
        tone=kw.get("tone", "honest_redirect"),
        arbiter_action=kw.get("arbiter_action", "resolved_to_no_match"),
        ask_slot=kw.get("ask_slot"),
        caps_applied=kw.get("caps_applied", ()),
    )


def _input(move: str, **overrides) -> ResponderV2Input:
    defaults = dict(
        user_message="any other jobs?",
        decision=_decision(move),
        results=[],
        training_by_job={},
        next_skill=(None, 0),
        band_signal="none",
        requires_consent=True,
        target_role_text="software developer",
        resume_facts=None,
        conversation_context=None,
    )
    defaults.update(overrides)
    return ResponderV2Input(**defaults)


class _LLMSpy:
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
# present_no_match -> deterministic fallback verbatim, LLM/policy not called
# =========================================================================
def test_present_no_match_bypasses_llm_and_policy(monkeypatch) -> None:
    """The whole point of AR-8e: on present_no_match,
    `compose_response_v2` MUST early-return to
    `_present_no_match_fallback_v2` without ever invoking the LLM
    or the policy gate."""
    from skillbridge.chat import responder

    llm_spy = _LLMSpy(must_not_run=True)
    policy_spy = _PolicySpy(must_not_run=True)
    monkeypatch.setattr(responder, "is_enabled", lambda: True)
    monkeypatch.setattr(responder, "call", llm_spy)
    monkeypatch.setattr(responder, "_policy_ok_v2", policy_spy)

    inp = _input("present_no_match")
    reply = compose_response_v2(inp)
    expected = _present_no_match_fallback_v2(inp)
    assert reply == expected
    assert llm_spy.calls == []
    assert policy_spy.calls == []


def test_present_no_match_reply_names_sccc_verbatim(monkeypatch) -> None:
    """Slice 6 (2026-06-29) UPDATE: locked Option-1 text after live
    verify caught the matching engine's no-match LLM repeatedly
    making false claims ('I checked related roles' when recommender
    never ran) and hollow offers ('want training directions?' with
    no consume hook).

    Locked text:
      target set:    'I don't see any {target} postings in Sault
                      Ste. Marie today.'
      target unset:  'I don't see matching postings in Sault Ste.
                      Marie today.'
      + ' The Sault Community Career Centre has access to more
         sources and can flag openings as they come up.'

    No training offer. No related-role claim. No 'do you want?'
    dead-end. No editorial market panorama."""
    from skillbridge.chat import responder

    monkeypatch.setattr(responder, "is_enabled", lambda: True)
    monkeypatch.setattr(responder, "call",
                        _LLMSpy(must_not_run=True))
    monkeypatch.setattr(responder, "_policy_ok_v2",
                        _PolicySpy(must_not_run=True))

    inp = _input("present_no_match")
    reply = compose_response_v2(inp)
    # Target-conditional absence statement.
    assert "Sault Ste. Marie today" in reply
    # SCCC referral.
    assert (
        "Sault Community Career Centre has access to more sources"
        in reply
    )
    # Old offending claims absent.
    assert "training direction" not in reply.lower()
    assert "related roles" not in reply.lower()
    assert "do you want" not in reply.lower()
    # Old editorial padding absent.
    assert "37 active" not in reply
    assert "mostly in" not in reply.lower()


def test_present_no_match_uses_target_role_text_when_present(
    monkeypatch,
) -> None:
    """Slice 6: when target_role_text is set, the absence sentence
    uses it verbatim."""
    from skillbridge.chat import responder

    monkeypatch.setattr(responder, "is_enabled", lambda: True)
    monkeypatch.setattr(responder, "call",
                        _LLMSpy(must_not_run=True))
    monkeypatch.setattr(responder, "_policy_ok_v2",
                        _PolicySpy(must_not_run=True))

    inp = _input("present_no_match")  # default helper sets a target
    reply = compose_response_v2(inp)
    # The reply should NAME the target role.
    assert (
        "I don't see any" in reply
        and "postings in Sault Ste. Marie today" in reply
    )


def test_present_no_match_drops_next_skill_hint(monkeypatch) -> None:
    """Slice 6 (2026-06-29) UPDATE: the locked Option-1 text is
    MINIMAL -- just absence + SCCC referral. The next_skill engine
    hint is no longer woven in (was part of the older editorial
    padding that contributed nothing to the user's question and
    added drift surface)."""
    from skillbridge.chat import responder

    monkeypatch.setattr(responder, "is_enabled", lambda: True)
    monkeypatch.setattr(responder, "call",
                        _LLMSpy(must_not_run=True))
    monkeypatch.setattr(responder, "_policy_ok_v2",
                        _PolicySpy(must_not_run=True))

    inp = _input("present_no_match", next_skill=("forklift operation", 4))
    reply = compose_response_v2(inp)
    # next_skill no longer surfaced in the locked text.
    assert "forklift operation" not in reply
    assert "around 4 more" not in reply
    # SCCC referral still present.
    assert "Sault Community Career Centre" in reply


def test_present_no_match_bypass_works_even_when_llm_disabled(monkeypatch) -> None:
    """The AR-8e early-return runs BEFORE the `is_enabled()` check,
    so it applies regardless of LLM state. This pins the locking
    placement."""
    from skillbridge.chat import responder

    monkeypatch.setattr(responder, "is_enabled", lambda: False)
    monkeypatch.setattr(responder, "call",
                        _LLMSpy(must_not_run=True))
    monkeypatch.setattr(responder, "_policy_ok_v2",
                        _PolicySpy(must_not_run=True))

    inp = _input("present_no_match")
    reply = compose_response_v2(inp)
    assert "Sault Community Career Centre" in reply
    # Identity check against the fallback's output.
    assert reply == _present_no_match_fallback_v2(inp)


# =========================================================================
# Scope: only present_no_match short-circuits; other moves run LLM
# =========================================================================
@pytest.mark.parametrize("move", [
    "present_matches",
    "present_near_miss",
    "explain_gap",
    "explain_remaining_gaps",
    "ask_one_clarifying_question",
    "redirect_scope",
    "acknowledge_and_continue",
    "offer_refinement",
])
def test_non_no_match_moves_unaffected_by_ar8e_guard(monkeypatch, move) -> None:
    """The AR-8e early-return MUST be scoped to present_no_match.
    Other moves keep their existing LLM-narrated path (or their own
    specific early-returns from earlier slices)."""
    from skillbridge.chat import responder

    llm_spy = _LLMSpy(must_not_run=False, return_text="LLM-narrated")
    policy_spy = _PolicySpy(must_not_run=False, return_value=True)
    monkeypatch.setattr(responder, "is_enabled", lambda: True)
    monkeypatch.setattr(responder, "call", llm_spy)
    monkeypatch.setattr(responder, "_policy_ok_v2", policy_spy)

    inp = _input(move)
    reply = compose_response_v2(inp)
    # The LLM-narrated string must NOT be the deterministic
    # present_no_match wording.
    assert "I don't see one in today's Sault Ste. Marie postings" not in reply
