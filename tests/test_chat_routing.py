"""Unit tests for the deterministic router (chat orchestration v2.1).

Three concerns:

  1. Each of the 7 priority rules fires correctly on its canonical input.
  2. Priority order is enforced: when multiple rules could fire, the
     higher-priority one wins.
  3. Architectural promise (locked in design doc): the planner LLM
     MUST NOT be called for HIGH-confidence scope_violation or
     training-with-entity cases. The router returning a PlannerDecision
     for these is the contract; the handler-level integration test in
     Slice B step 5 mocks plan_next_move to fail loudly if it's called.

The handler integration (Slice B step 5) is tested separately in
test_chat_handler*.py; here we test the router in isolation.
"""
from __future__ import annotations

import os

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from skillbridge.chat.message_understanding import (
    DetectedEntity,
    MessageUnderstanding,
    understand_message,
)
from skillbridge.chat.routing import (
    RouterTrace,
    route_from_understanding,
)


pytestmark = pytest.mark.nodb


# ===========================================================================
# Helpers
# ===========================================================================
def _truth_ready() -> dict:
    """Truth-summary stub satisfying Rule 4's preconditions."""
    return {"enough_to_match": True, "usable_evidence_present": True}


def _truth_not_ready() -> dict:
    """Truth-summary stub failing Rule 4's preconditions (one false is enough)."""
    return {"enough_to_match": False, "usable_evidence_present": True}


def _u(msg: str, gaps: list[str] | None = None) -> MessageUnderstanding:
    return understand_message(user_message=msg, registry_gaps_in_message=gaps)


# ===========================================================================
# Rule 1: scope_violation -> redirect_scope, planner SKIPPED
# ===========================================================================
@pytest.mark.parametrize("msg,expected_reason_code", [
    ("Can I apply for PR while looking for work?",      "scope_violation_immigration"),
    ("canadian average wage",                           "scope_violation_wages"),
    ("jobs in toronto please",                          "scope_violation_non_ssm"),
])
def test_rule_1_scope_violation_emits_redirect_scope_and_skips_planner(
    msg, expected_reason_code,
):
    u = _u(msg)
    decision, trace = route_from_understanding(u, truth={})
    assert decision is not None
    assert decision.move == "redirect_scope"
    assert decision.reason_code == expected_reason_code
    assert decision.tone == "honest_redirect"
    assert decision.ask_slot is None
    assert trace.planner_skipped is True
    assert trace.rule_fired == "rule_1_scope_violation"


def test_rule_1_defensive_skip_when_no_scope_entity_attached():
    """Belt-and-suspenders: if understand_message classifies as
    scope_violation but no scope_keyword entity is attached, the router
    refuses to invent a reason_code and falls through to the planner.
    The existing arbiter scope-override safety net handles the truth-
    detected case."""
    broken = MessageUnderstanding(
        primary_intent="scope_violation",
        confidence="high",
        entities=(),  # bug: classification without entity
        reason="synthetic test case",
    )
    decision, trace = route_from_understanding(broken, truth={})
    assert decision is None
    assert trace.planner_skipped is False
    assert trace.rule_fired == "rule_1_skipped_no_scope_entity"


# ===========================================================================
# Rule 2: training_request + registry_gap -> explain_gap, planner SKIPPED
# ===========================================================================
@pytest.mark.parametrize("msg,gaps", [
    ("where can I do course for learning Excel",   ["Microsoft Excel"]),
    ("how can I get my Class G driver's licence",  ["Class G licence"]),
    ("online Excel course",                        ["Microsoft Excel"]),
    ("how do I get my 310T",                       ["310T truck and coach technician"]),
    ("any forklift training near me",              ["forklift certification"]),
])
def test_rule_2_training_with_entity_emits_explain_gap_and_skips_planner(
    msg, gaps,
):
    u = _u(msg, gaps=gaps)
    # Pre-condition: understand_message must have classified this as
    # training_request HIGH with at least one registry_gap.
    assert u.primary_intent == "training_request"
    assert u.confidence == "high"
    assert u.has_entity_type("registry_gap")

    decision, trace = route_from_understanding(u, truth={})
    assert decision is not None
    assert decision.move == "explain_gap"
    assert decision.reason_code == "credential_gap_present"
    assert decision.tone == "warm_supportive"
    assert decision.ask_slot is None
    assert trace.planner_skipped is True
    assert trace.rule_fired == "rule_2_training_with_entity"


# ===========================================================================
# Rule 3: training_request without entity
#   -> ask_one_clarifying_question(skills_text), planner SKIPPED
# ===========================================================================
@pytest.mark.parametrize("msg", [
    "what training do I need",
    "any course you recommend",
    "where can I get training",
    "do you have any certification options",
])
def test_rule_3_training_without_entity_asks_clarifying_question(msg):
    u = _u(msg)
    # Pre-condition: training_request HIGH with NO registry_gap entity.
    assert u.primary_intent == "training_request"
    assert u.confidence == "high"
    assert not u.has_entity_type("registry_gap")

    decision, trace = route_from_understanding(u, truth={})
    assert decision is not None
    assert decision.move == "ask_one_clarifying_question"
    assert decision.ask_slot == "skills_text"
    # Rule 3 wording slice: distinct reason code so the responder can
    # emit a training-discovery question instead of the regular skills
    # intake prompt. Previously emitted `insufficient_profile_evidence`.
    assert decision.reason_code == "training_request_no_entity"
    assert decision.tone == "warm_supportive"
    assert trace.planner_skipped is True
    assert trace.rule_fired == "rule_3_training_no_entity"


# ===========================================================================
# Rule 4: job_search + truth ready -> proceed_to_match, planner SKIPPED
# ===========================================================================
@pytest.mark.parametrize("msg", [
    "just show me jobs already",
    "show me what you got",
    "show me jobs",
    "go ahead and search",
])
def test_rule_4_job_search_with_truth_ready_proceeds_to_match(msg):
    u = _u(msg)
    assert u.primary_intent == "job_search"
    assert u.confidence == "high"
    decision, trace = route_from_understanding(u, truth=_truth_ready())
    assert decision is not None
    assert decision.move == "proceed_to_match"
    assert decision.reason_code == "user_explicitly_asked_to_match"
    assert decision.tone == "brief_confident"
    assert decision.ask_slot is None
    assert trace.planner_skipped is True
    assert trace.rule_fired == "rule_4_job_search_truth_ready"


# ===========================================================================
# Rule 6: job_search + truth NOT ready -> planner consulted
# ===========================================================================
@pytest.mark.parametrize("truth_state", [
    {"enough_to_match": False, "usable_evidence_present": True},
    {"enough_to_match": True,  "usable_evidence_present": False},
    {"enough_to_match": False, "usable_evidence_present": False},
    {},  # missing keys are treated as falsy -> planner
])
def test_rule_6_job_search_without_truth_ready_falls_to_planner(truth_state):
    u = _u("just show me jobs already")
    assert u.primary_intent == "job_search"
    decision, trace = route_from_understanding(u, truth=truth_state)
    assert decision is None
    assert trace.planner_skipped is False
    assert trace.rule_fired == "rule_6_job_search_truth_not_ready"


# ===========================================================================
# Rule 5: registry_gap entity present without training intent (MEDIUM)
#   -> planner consulted
# ===========================================================================
def test_rule_5_entity_without_training_intent_falls_to_planner():
    """The skill-claim case 'I have Excel, find me jobs' - even when
    Excel is detected as a gap (e.g. via canonical 'Microsoft Excel'
    via some upstream path), the router must NOT route to training.
    Hand to the planner with the understanding's MEDIUM context.

    We construct the MessageUnderstanding directly so the test
    doesn't depend on registry-internals filtering 'Excel' bare alias.
    """
    u = MessageUnderstanding(
        primary_intent="ambiguous",
        confidence="medium",
        entities=(DetectedEntity(
            type="registry_gap",
            canonical_name="Microsoft Excel",
            matched_text="Microsoft Excel",
            source="registry_alias",
        ),),
        reason="entity present without training intent",
    )
    decision, trace = route_from_understanding(u, truth={})
    assert decision is None
    assert trace.planner_skipped is False
    assert trace.rule_fired == "rule_5_entity_without_training_intent"


# ===========================================================================
# Rule 7: default / low-signal -> planner consulted
# ===========================================================================
@pytest.mark.parametrize("msg", [
    "hello there",
    "thanks for your help",
    "the weather is nice today",
    "interesting",
])
def test_rule_7_default_falls_to_planner(msg):
    u = _u(msg)
    decision, trace = route_from_understanding(u, truth={})
    assert decision is None
    assert trace.planner_skipped is False
    assert trace.rule_fired == "rule_7_default_planner"


# ===========================================================================
# Conversational signals (decline / correction / confirmation) -> planner.
# These are MEDIUM in understand_message; the router has no rule for
# them yet, so they fall through to Rule 7.
# ===========================================================================
@pytest.mark.parametrize("msg", [
    "no thanks",
    "actually, I meant electrician",
    "yes that sounds right",
])
def test_conversational_signals_fall_to_planner(msg):
    u = _u(msg)
    decision, trace = route_from_understanding(u, truth={})
    assert decision is None
    assert trace.planner_skipped is False
    # Either rule_5 (if entities attached) or rule_7 (default) is
    # acceptable. The invariant is: planner consulted, not skipped.


# ===========================================================================
# Priority order: when multiple rules could fire, earlier wins.
# ===========================================================================
def test_priority_scope_beats_training_when_both_signals_present():
    """'can I get a forklift course for PR?' -- scope (Rule 1) wins
    over training (Rule 2). Design doc acknowledges this risk:
    redirect_scope is correct here; the forklift question can resume
    on the next turn."""
    u = _u("can I get a forklift course for PR")
    decision, trace = route_from_understanding(u, truth={})
    assert decision is not None
    assert decision.move == "redirect_scope"
    assert trace.rule_fired == "rule_1_scope_violation"


def test_priority_training_with_entity_beats_job_search_signal():
    """Hypothetical case: 'show me Excel courses' -- has impatient-ish
    surface but training+entity is the intended classification. The
    actual classification depends on understand_message's priority
    order; the router test pins that whatever HIGH classification
    understand_message yields, the router handles it without bouncing
    to the planner."""
    u = _u("show me Excel courses", gaps=["Microsoft Excel"])
    # We expect training_request to win in understand_message:
    assert u.primary_intent in ("training_request", "job_search")
    decision, trace = route_from_understanding(u, truth=_truth_ready())
    assert decision is not None
    assert trace.planner_skipped is True


# ===========================================================================
# Live-bug regression cases (the failures that triggered this refactor).
# Each one MUST end with decision != None and planner_skipped=True so
# the LLM cannot overreach.
# ===========================================================================
def test_live_bug_class_g_routes_to_explain_gap_without_planner():
    u = _u("how can I get my Class G driver's licence",
           gaps=["Class G licence"])
    decision, trace = route_from_understanding(u, truth={})
    assert decision is not None and decision.move == "explain_gap"
    assert trace.planner_skipped is True


def test_live_bug_excel_course_routes_to_explain_gap_without_planner():
    u = _u("online Excel course", gaps=["Microsoft Excel"])
    decision, trace = route_from_understanding(u, truth={})
    assert decision is not None and decision.move == "explain_gap"
    assert trace.planner_skipped is True


def test_live_bug_pr_question_routes_to_redirect_without_planner():
    u = _u("Can I apply for PR while looking?")
    decision, trace = route_from_understanding(u, truth={})
    assert decision is not None and decision.move == "redirect_scope"
    assert decision.reason_code == "scope_violation_immigration"
    assert trace.planner_skipped is True


# ===========================================================================
# RouterTrace shape: every router call returns a trace with the same
# fields populated regardless of which rule fires. This is what the
# handler logs.
# ===========================================================================
def test_router_trace_is_always_populated():
    for u in [
        _u("hello"),
        _u("Can I apply for PR"),
        _u("Excel course", gaps=["Microsoft Excel"]),
    ]:
        decision, trace = route_from_understanding(u, truth={})
        assert isinstance(trace, RouterTrace)
        assert trace.rule_fired  # non-empty
        assert trace.understanding_intent
        assert trace.understanding_confidence in ("high", "medium", "low")
        # planner_skipped is True iff a decision was emitted.
        assert (decision is not None) == trace.planner_skipped
        # entity_canonical_names is always a tuple (empty when no entities).
        # Slice B review feedback: live debug logs need to show WHICH
        # entity the router saw, so this field MUST be populated.
        assert isinstance(trace.entity_canonical_names, tuple)


def test_router_trace_carries_entity_names_for_live_debug_logs():
    """Slice B review patch: handler log includes router_trace.entity_canonical_names
    so a live failure ('why did the router pick rule_2 here?') is
    answerable from the log alone, no transcript replay needed.

    This test pins the per-rule shape so a future refactor that
    accidentally drops the field surfaces fast.
    """
    # Rule 2: training + Excel entity -> entity_canonical_names == ("Microsoft Excel",)
    u = _u("online Excel course", gaps=["Microsoft Excel"])
    _, trace = route_from_understanding(u, truth={})
    assert trace.rule_fired == "rule_2_training_with_entity"
    assert trace.entity_canonical_names == ("Microsoft Excel",)

    # Rule 1: scope -> entity_canonical_names == ("immigration",)
    u = _u("can I apply for PR while looking")
    _, trace = route_from_understanding(u, truth={})
    assert trace.rule_fired == "rule_1_scope_violation"
    assert trace.entity_canonical_names == ("immigration",)

    # Rule 7 (default): no entities -> empty tuple
    u = _u("hello there")
    _, trace = route_from_understanding(u, truth={})
    assert trace.rule_fired == "rule_7_default_planner"
    assert trace.entity_canonical_names == ()


# ===========================================================================
# Returned PlannerDecision must be a valid PlannerDecision (passes
# Pydantic validation, frozen, etc). If the router synthesizes an
# invalid PlannerDecision, that's a bug; the arbiter would crash
# downstream. Easier to catch here.
# ===========================================================================
def test_router_decision_passes_pydantic_validation():
    from skillbridge.chat.planner import PlannerDecision

    for u, truth in [
        (_u("Can I apply for PR"),                            {}),
        (_u("Excel course",        gaps=["Microsoft Excel"]), {}),
        (_u("where can I get training"),                       {}),
        (_u("just show me jobs already"),                     _truth_ready()),
    ]:
        decision, _ = route_from_understanding(u, truth=truth)
        assert decision is not None
        assert isinstance(decision, PlannerDecision)
        # Frozen: attempting to set raises (just confirms the contract).
        with pytest.raises(Exception):
            decision.move = "acknowledge_and_continue"  # type: ignore[misc]
