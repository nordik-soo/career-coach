"""Rule 3 wording slice tests.

Pins the contract for "user said training_request without naming a
credential":
  1. Router Rule 3 emits the DISTINCT reason `training_request_no_entity`
  2. PlannerDecision schema accepts `ask_one_clarifying_question +
     training_request_no_entity + skills_text` and rejects mismatched
     pairings
  3. `compose_response_v2` returns the EXACT locked training-discovery
     question for this reason code -- LLM is NOT called
  4. Planner + engine remain skipped on this turn
  5. The regular `insufficient_profile_evidence + skills_text` path
     still produces the role-aware skills-intake prompt (no regression)
"""
from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("LLM_ENABLED", "false")

import pytest
from pydantic import ValidationError

from skillbridge.chat import handler, responder
from skillbridge.chat.arbiter import ArbiterDecision
from skillbridge.chat.message_understanding import understand_message
from skillbridge.chat.planner import PlannerDecision
from skillbridge.chat.responder import (
    ResponderV2Input,
    _TRAINING_REQUEST_NO_ENTITY_QUESTION,
    compose_response_v2,
)
from skillbridge.chat.url_views import (
    build_sanitized_responder_view_v2 as _v_v2,
)
from skillbridge.chat.routing import route_from_understanding

pytestmark = pytest.mark.nodb


# ============================================================================
# 1. Router emits the new reason
# ============================================================================
@pytest.mark.parametrize("msg", [
    "what training do I need",
    "any course you recommend",
    "where can I get training",
    "do you have any certification options",
])
def test_router_rule_3_emits_training_request_no_entity(msg):
    """Locked: Rule 3 (training_request HIGH, no registry entity)
    emits `training_request_no_entity`, NOT the generic
    `insufficient_profile_evidence` that overlapped with intake."""
    u = understand_message(user_message=msg, registry_gaps_in_message=[])
    decision, trace = route_from_understanding(u, truth={})
    assert decision is not None
    assert decision.move == "ask_one_clarifying_question"
    assert decision.ask_slot == "skills_text"
    assert decision.reason_code == "training_request_no_entity"
    assert trace.rule_fired == "rule_3_training_no_entity"
    assert trace.planner_skipped is True


# ============================================================================
# 2. Planner schema accepts the new move/reason/slot pairing
# ============================================================================
def test_planner_schema_accepts_training_request_no_entity():
    """The new reason MUST be valid for ask_one_clarifying_question
    so the planner CAN emit it (for parity with the router's deterministic
    rule)."""
    d = PlannerDecision.model_validate({
        "move":        "ask_one_clarifying_question",
        "reason_code": "training_request_no_entity",
        "ask_slot":    "skills_text",
        "tone":        "warm_supportive",
    })
    assert d.reason_code == "training_request_no_entity"
    assert d.move == "ask_one_clarifying_question"


def test_planner_schema_rejects_training_request_no_entity_with_wrong_move():
    """The cross-field invariant in `_VALID_REASON_BY_MOVE` should
    reject `training_request_no_entity` paired with any move OTHER
    than ask_one_clarifying_question."""
    for forbidden_move in (
        "proceed_to_match", "explain_gap", "offer_refinement",
        "redirect_scope", "acknowledge_and_continue",
    ):
        with pytest.raises(ValidationError):
            PlannerDecision.model_validate({
                "move":        forbidden_move,
                "reason_code": "training_request_no_entity",
                # ask_slot only required for ask_one_clarifying_question
                "ask_slot":    None,
                "tone":        "warm_supportive",
            })


def test_planner_schema_rejects_training_request_no_entity_without_ask_slot():
    """ask_one_clarifying_question requires a non-null ask_slot."""
    with pytest.raises(ValidationError):
        PlannerDecision.model_validate({
            "move":        "ask_one_clarifying_question",
            "reason_code": "training_request_no_entity",
            "ask_slot":    None,
            "tone":        "warm_supportive",
        })


# ============================================================================
# 2b. Round-26 R2 fix -- reason MUST pair with skills_text
# ============================================================================
@pytest.mark.parametrize("wrong_slot", [
    "target_role_text",
    "experience_text",
    "work_type_preference",
    "shift_preference",
    "education_text",
])
def test_planner_schema_rejects_training_request_no_entity_with_wrong_slot(wrong_slot):
    """Round-26 R2: `training_request_no_entity` is semantically the
    training-discovery question, which lives on skills_text. Any other
    slot would mean the responder rendered the training prompt while
    the planner claimed it was asking for, e.g., target_role -- a
    routing-vs-rendering contract break."""
    with pytest.raises(ValidationError) as exc_info:
        PlannerDecision.model_validate({
            "move":        "ask_one_clarifying_question",
            "reason_code": "training_request_no_entity",
            "ask_slot":    wrong_slot,
            "tone":        "warm_supportive",
        })
    # The error must mention either ask_slot or skills_text so it's
    # obvious WHY the pairing was rejected.
    msg = str(exc_info.value)
    assert "skills_text" in msg or "ask_slot" in msg, (
        f"Validation error should mention the slot mismatch; got {msg!r}"
    )


def test_planner_schema_only_training_request_pairs_with_skills_text_binding():
    """Sanity: regular reasons that target skills_text (like
    `insufficient_profile_evidence`) are NOT bound to a single slot --
    they can still pair with `experience_text` or others, as before."""
    # insufficient_profile_evidence + skills_text is the canonical
    # pairing but not the only one for that reason; the binding rule
    # MUST be specific to training_request_no_entity.
    d = PlannerDecision.model_validate({
        "move":        "ask_one_clarifying_question",
        "reason_code": "insufficient_profile_evidence",
        "ask_slot":    "experience_text",
        "tone":        "warm_supportive",
    })
    assert d.reason_code == "insufficient_profile_evidence"
    assert d.ask_slot == "experience_text"


# ============================================================================
# 2c. Round-26 R2 fix -- planner prompt grounding branches asking_about_gap
#     by registry_gaps_in_message presence
# ============================================================================
def test_planner_prompt_documents_training_no_entity_branch():
    """Round-26 R2: the planner's grounding rule 3 MUST branch on
    registry_gaps_in_message so that an asking_about_gap turn with NO
    registry gap is routed to training_request_no_entity, not to
    explain_gap. Without this branch the planner LLM falls back to
    explain_gap and the new reason is unreachable via the LLM path."""
    from skillbridge.chat.planner import PLANNER_SYSTEM_PROMPT
    # The reason code is mentioned by name
    assert "training_request_no_entity" in PLANNER_SYSTEM_PROMPT, (
        "Planner prompt must name `training_request_no_entity` so the "
        "LLM has the code in its vocabulary."
    )
    # The branch is documented in the grounding section
    p = PLANNER_SYSTEM_PROMPT.lower()
    assert "registry_gaps_in_message" in p
    # The locked behavior: empty registry_gaps + asking_about_gap ->
    # training_request_no_entity. Looser check is fine -- we just
    # need the documentation lines together.
    assert "empty" in p
    assert "non-empty" in p


def test_planner_prompt_still_routes_named_credential_to_explain_gap():
    """The pre-existing behavior for asking_about_gap with a known
    credential mention MUST still route to explain_gap."""
    from skillbridge.chat.planner import PLANNER_SYSTEM_PROMPT
    p = PLANNER_SYSTEM_PROMPT.lower()
    # The grounding doc for rule 3 still names explain_gap as the
    # outcome for the named-credential branch.
    assert "explain_gap" in p


def test_planner_prompt_grounding_rules_are_still_ordered_first_win():
    """The expanded rule 3 must NOT break the "earlier rules win"
    ordering -- scope (rule 1) and resume-failed (rule 2) MUST still
    fire BEFORE the training-no-entity branch."""
    from skillbridge.chat.planner import PLANNER_SYSTEM_PROMPT
    # The "Apply these rules IN ORDER -- earlier rules win" instruction
    # must still be present.
    assert "earlier rules win" in PLANNER_SYSTEM_PROMPT


# ============================================================================
# 3. compose_response_v2 deterministically renders the locked text
# ============================================================================
def _decision_rule3() -> ArbiterDecision:
    return ArbiterDecision(
        final_move="ask_one_clarifying_question",
        reason_code="training_request_no_entity",
        tone="warm_supportive",
        arbiter_action="passed_planner_through",
        ask_slot="skills_text",
    )


def _input(decision: ArbiterDecision) -> ResponderV2Input:
    return ResponderV2Input(
        user_message="any course you recommend",
        decision=decision,
        results=[],
        training_by_job={},
        next_skill=(None, 0),
        band_signal="none",
        requires_consent=True,
        target_role_text=None,
        resume_facts=None,
        conversation_context=None,
        near_miss_payload=None,
        remaining_gaps_payload=None,
        clarification_payload=None,
    )


def test_compose_response_v2_emits_exact_locked_phrasing():
    """The locked text from design decision #4 must appear VERBATIM:
        'Sure -- what skill or certificate do you want training for?
         For example Excel, WHMIS, forklift, Class G, or 310T.'
    """
    out = compose_response_v2(_input(_decision_rule3()))
    assert out == (
        "Sure -- what skill or certificate do you want training for? "
        "For example Excel, WHMIS, forklift, Class G, or 310T."
    )
    # Pin the constant identity so any future text drift is caught.
    assert out == _TRAINING_REQUEST_NO_ENTITY_QUESTION


def test_compose_response_v2_skips_llm_for_training_request_no_entity(monkeypatch):
    """The reason exists specifically to avoid LLM cost/variation.
    Pin that the model is NOT called on this turn even when enabled."""
    monkeypatch.setattr(responder, "is_enabled", lambda: True)
    monkeypatch.setattr(
        responder, "call",
        lambda *a, **kw: pytest.fail(
            "LLM must NOT be called on training_request_no_entity"
        ),
    )
    monkeypatch.setattr(
        responder, "_policy_ok_v2",
        lambda *a, **kw: pytest.fail(
            "Policy sweep must NOT run on training_request_no_entity"
        ),
    )
    out = compose_response_v2(_input(_decision_rule3()))
    assert out == _TRAINING_REQUEST_NO_ENTITY_QUESTION


def test_fallback_reply_v2_defense_in_depth_uses_locked_phrasing(monkeypatch):
    """If compose_response_v2's early-return ever regresses, the
    same string MUST still be emitted via `_fallback_reply_v2`."""
    monkeypatch.setattr(responder, "is_enabled", lambda: False)
    out = compose_response_v2(_input(_decision_rule3()))
    assert out == _TRAINING_REQUEST_NO_ENTITY_QUESTION


# ============================================================================
# 4. Planner + engine remain skipped (router commits, no planner LLM)
# ============================================================================
def test_router_skips_planner_on_rule_3():
    """Trace MUST report planner_skipped=True -- the router decided
    deterministically and the LLM is not called."""
    u = understand_message(
        user_message="any course you recommend", registry_gaps_in_message=[],
    )
    decision, trace = route_from_understanding(u, truth={})
    assert decision is not None
    assert trace.planner_skipped is True


# ============================================================================
# 5. Regression: existing insufficient_profile_evidence + skills_text path
#    still emits the role-aware skills-intake question
# ============================================================================
def test_regular_skills_text_ask_still_uses_intake_phrasing():
    """Sanity: an `ask_one_clarifying_question + skills_text +
    insufficient_profile_evidence` turn (the original intake path) MUST
    still produce the role-aware skills-intake question, NOT the
    new training-discovery one."""
    intake_decision = ArbiterDecision(
        final_move="ask_one_clarifying_question",
        reason_code="insufficient_profile_evidence",
        tone="warm_supportive",
        arbiter_action="passed_planner_through",
        ask_slot="skills_text",
    )
    out = compose_response_v2(_input(intake_decision))
    # Must NOT be the training-discovery line
    assert out != _TRAINING_REQUEST_NO_ENTITY_QUESTION
    # And must NOT contain the training-specific list of examples
    assert "skill or certificate do you want training for" not in out


# ============================================================================
# 6. End-to-end through _try_v2_path: planner + engine NEVER called
# ============================================================================
def test_end_to_end_rule_3_skips_planner_and_engine(monkeypatch):
    """The router's deterministic decision short-circuits the planner
    LLM and the engine. Neither should be invoked on this turn."""
    from skillbridge.chat import truth_summary as ts_mod
    from skillbridge.session.staging import StagedProfile

    monkeypatch.setattr(handler, "CHAT_ORCHESTRATOR", "v2")

    real_build = ts_mod.build_truth_summary

    def fake_build(*, staged, user_message, **kw):
        r = real_build(staged=staged, user_message=user_message, **kw)
        object.__setattr__(r, "scope_violations_detected", [])
        return r
    monkeypatch.setattr(handler, "build_truth_summary", fake_build)

    planner_calls: list[Any] = []

    def fake_planner(truth_json):
        planner_calls.append(truth_json)
        # The router should have decided; planner is the fallback.
        return None
    monkeypatch.setattr(handler, "plan_next_move", fake_planner)

    def fail_engine(staged, top=20):
        pytest.fail(
            "Engine MUST NOT run on a Rule 3 turn -- the router's "
            "decision short-circuits before engine."
        )
    monkeypatch.setattr(
        handler.match_engine, "compute_matches_in_memory", fail_engine,
    )

    class _Store:
        def new_session(self): return "x"
        def load(self, s): return None
        def save(self, s): return s.session_id or "x"
        def delete(self, s): pass

    staged = StagedProfile.new("rule3-e2e")
    staged.message_count = 2
    staged.target_role_text = "automotive technician"
    # MESSAGE_UNDERSTANDING_ENABLED must be on for the router rule to
    # fire. Test it explicitly.
    monkeypatch.setattr(handler, "MESSAGE_UNDERSTANDING_ENABLED", True)

    response = handler._try_v2_path(
        staged=staged,
        message="any course you recommend",
        uploaded_file=False, resume_info=None, store=_Store(),
    )
    assert response is not None
    assert response["final_move"] == "ask_one_clarifying_question"
    assert response["reply"] == _TRAINING_REQUEST_NO_ENTITY_QUESTION
    assert planner_calls == [], (
        "Planner was called on a Rule 3 turn -- the router's "
        "deterministic decision should have short-circuited it. "
        f"planner_calls={len(planner_calls)}"
    )
