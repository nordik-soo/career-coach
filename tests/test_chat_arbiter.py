"""Unit tests for chat orchestration v2 slice 4 -- the two-pass arbiter.

Covers `validate_planner_intent` (pass 1) and `resolve_match_outcome`
(pass 2) as separate pure functions, plus the cross-cutting invariants
from design doc §6 and the Slice 4 review tightenings.

No DB, no engine, no LLM. The `nodb` marker keeps the conftest
TRUNCATE off. Pass 2 receives engine RESULTS (match_count, caps_applied)
directly, so we don't need a fake engine.
"""
from __future__ import annotations

import os
from itertools import product
from typing import get_args

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from skillbridge.chat.arbiter import (
    ARBITER_REASON_DUPLICATE_ASK,
    ARBITER_REASON_FALLBACK,
    ARBITER_REASON_MATCHES_FOUND,
    ARBITER_REASON_MATCHES_WITH_CAPS,
    ARBITER_REASON_NEAR_MISS,
    ARBITER_REASON_NO_MATCHES,
    ARBITER_REASON_SCOPE_OVERRIDE,
    ArbiterDecision,
    OutcomeMove,
    RunEngine,
    _is_slot_strongly_filled,
    _next_unfilled_priority_slot,
    _pick_ask_reason_and_slot,
    _scope_reason_code,
    resolve_match_outcome,
    validate_planner_intent,
)
from skillbridge.chat.planner import PlannerDecision, PlannerMove

pytestmark = pytest.mark.nodb


# ===========================================================================
# Helpers -- build minimal planner decisions and truth dicts
# ===========================================================================
def _planner(
    move: str = "proceed_to_match",
    reason_code: str = "resume_skills_sufficient",
    ask_slot: str | None = None,
    tone: str = "brief_confident",
) -> PlannerDecision:
    """Build a valid PlannerDecision. Defaults yield a happy-path
    proceed_to_match. Tests override fields as needed."""
    return PlannerDecision.model_validate({
        "move": move,
        "reason_code": reason_code,
        "ask_slot": ask_slot,
        "tone": tone,
    })


def _truth(**overrides) -> dict:
    """Minimal truth_summary-shaped dict. Defaults yield a happy
    proceed-ready profile: enough_to_match, usable evidence, no scope
    violations, specific target role, nothing filled with weak proxies."""
    base = {
        "user_message": "show me jobs",
        "enough_to_match": True,
        "enough_to_match_reason": "resume_skills_sufficient",
        "usable_evidence_present": True,
        "scope_violations_detected": [],
        "target_role_text": "warehouse worker",
        "target_role_specificity": "specific",
        "resume_parse_quality": "full",
        "filled_slots": ["target_role_text", "skills_text"],
        "user_intent_signal": "impatient_proceed",
        "match_count": 0,
        "caps_applied": [],
    }
    base.update(overrides)
    return base


# ===========================================================================
# Pass 1 -- Rule 1: planner returned None -> fallback_to_legacy
# ===========================================================================
def test_pass1_returns_fallback_when_planner_is_none():
    """Legacy fallback signal -- handler substitutes intake_state.decide()."""
    result = validate_planner_intent(None, _truth())
    assert isinstance(result, ArbiterDecision)
    assert result.arbiter_action == "fallback_to_legacy"
    assert result.reason_code == ARBITER_REASON_FALLBACK


def test_pass1_fallback_never_returns_match_outcomes():
    """Even on fallback, final_move must not be a match-outcome
    (those are Pass-2-only)."""
    result = validate_planner_intent(None, _truth())
    assert isinstance(result, ArbiterDecision)
    assert result.final_move not in {"present_matches", "present_no_match"}


# ===========================================================================
# Pass 1 -- Rule 2: scope override (wins over any planner move)
# ===========================================================================
@pytest.mark.parametrize("planner_move,planner_reason,planner_ask_slot", [
    ("acknowledge_and_continue", "user_confirmed", None),
    ("proceed_to_match", "resume_skills_sufficient", None),
    ("ask_one_clarifying_question", "target_role_unclear", "target_role_text"),
    ("explain_gap", "credential_gap_present", None),
    ("offer_refinement", "narrow_request", None),
    ("redirect_scope", "scope_violation_immigration", None),
])
def test_pass1_scope_violation_overrides_any_planner_move(
    planner_move, planner_reason, planner_ask_slot,
):
    """Scope wins precedence. Even when the planner says proceed_to_match
    on an otherwise valid truth summary, scope_violations_detected
    forces redirect_scope. This is the case the user emphasized:
    a scoped-out user asking impatiently must NOT run the engine."""
    decision = _planner(
        move=planner_move,
        reason_code=planner_reason,
        ask_slot=planner_ask_slot,
    )
    truth = _truth(scope_violations_detected=["immigration"])
    result = validate_planner_intent(decision, truth)
    assert isinstance(result, ArbiterDecision)
    assert result.final_move == "redirect_scope"
    assert result.arbiter_action == "overrode_to_redirect"
    assert result.tone == "honest_redirect"


def test_pass1_scope_override_does_NOT_run_engine_even_when_proceed_requested():
    """Critical case from the user's review note: planner says
    proceed_to_match, truth has BOTH enough_to_match==true AND
    scope_violations_detected non-empty. Scope must win -- engine
    must NOT be run. Pass 1 returns a terminal redirect, not RunEngine."""
    decision = _planner(move="proceed_to_match")
    truth = _truth(
        enough_to_match=True,
        usable_evidence_present=True,
        scope_violations_detected=["national_wages"],
    )
    result = validate_planner_intent(decision, truth)
    assert isinstance(result, ArbiterDecision)  # NOT RunEngine
    assert result.final_move == "redirect_scope"


@pytest.mark.parametrize("violation_tag,expected_reason", [
    ("immigration", "scope_violation_immigration"),
    ("express_entry", "scope_violation_immigration"),
    ("national_wages", "scope_violation_wages"),
    ("statcan", "scope_violation_wages"),
    ("salary_outside_ssm", "scope_violation_wages"),
    ("non_ssm_city", "scope_violation_non_ssm"),
    ("toronto", "scope_violation_non_ssm"),
    ("off_topic", "scope_violation_off_topic"),
    ("unknown_thing", ARBITER_REASON_SCOPE_OVERRIDE),  # safe fallback
])
def test_scope_reason_code_maps_violation_tags(violation_tag, expected_reason):
    truth = _truth(scope_violations_detected=[violation_tag])
    assert _scope_reason_code(truth) == expected_reason


# ===========================================================================
# Pass 1 -- Rule 3: proceed_to_match independent re-check (the BIG one)
# ===========================================================================
def test_pass1_proceed_overridden_when_usable_evidence_present_false():
    """LLM proposes, backend disposes. Planner says proceed; truth says
    we have no usable evidence -- arbiter overrides to ask."""
    decision = _planner(move="proceed_to_match")
    truth = _truth(
        enough_to_match=True,        # contradictory state -- defense in depth
        usable_evidence_present=False,
    )
    result = validate_planner_intent(decision, truth)
    assert isinstance(result, ArbiterDecision)
    assert result.final_move == "ask_one_clarifying_question"
    assert result.arbiter_action == "overrode_to_ask"
    assert result.reason_code == "resume_failed_need_chat_skills"
    assert result.ask_slot == "skills_text"


def test_pass1_proceed_overridden_when_enough_to_match_false():
    """Even with usable evidence, if enough_to_match is false the
    engine shouldn't run."""
    decision = _planner(move="proceed_to_match")
    truth = _truth(
        enough_to_match=False,
        enough_to_match_reason="missing_target",
        usable_evidence_present=True,
        target_role_text=None,
        target_role_specificity="none",
    )
    result = validate_planner_intent(decision, truth)
    assert isinstance(result, ArbiterDecision)
    assert result.final_move == "ask_one_clarifying_question"
    assert result.arbiter_action == "overrode_to_ask"
    assert result.reason_code == "target_role_unclear"
    assert result.ask_slot == "target_role_text"


def test_pass1_proceed_clears_returns_run_engine_signal():
    """Happy path: planner says proceed, truth supports it -> RunEngine.
    Pass 1 does NOT itself emit present_matches/present_no_match; that's
    Pass 2's job after the engine runs."""
    decision = _planner(
        move="proceed_to_match",
        reason_code="user_explicitly_asked_to_match",
        tone="brief_confident",
    )
    result = validate_planner_intent(decision, _truth())
    assert isinstance(result, RunEngine)
    assert result.planner_reason_code == "user_explicitly_asked_to_match"
    assert result.planner_tone == "brief_confident"


def test_pass1_proceed_clears_preserves_each_planner_tone():
    """Tone must be carried into pass 2 verbatim so the planner's
    intent shapes the eventual present_matches output."""
    from skillbridge.chat.planner import Tone
    for tone in get_args(Tone):
        decision = _planner(move="proceed_to_match", tone=tone)
        result = validate_planner_intent(decision, _truth())
        assert isinstance(result, RunEngine)
        assert result.planner_tone == tone


# Rule 3 precedence within itself: usable_evidence_present is checked
# BEFORE enough_to_match so the "failed scan" message is more specific
# than the generic "insufficient evidence" one.
def test_pass1_usable_evidence_check_takes_precedence_over_enough_check():
    decision = _planner(move="proceed_to_match")
    truth = _truth(
        enough_to_match=False,
        usable_evidence_present=False,  # both false
        resume_parse_quality="failed",
    )
    result = validate_planner_intent(decision, truth)
    assert isinstance(result, ArbiterDecision)
    # Specific reason wins (resume_failed_need_chat_skills, not
    # the generic enough_to_match-derived reason)
    assert result.reason_code == "resume_failed_need_chat_skills"
    assert result.ask_slot == "skills_text"


# ===========================================================================
# Pass 1 -- Rule 4: duplicate-ask reroute (narrow scope)
# ===========================================================================
def test_pass1_rule4_reroutes_to_run_engine_when_slot_strongly_filled_and_enough():
    """Slot is strongly filled (target_role_text + specific) AND
    enough_to_match AND usable_evidence_present -> drop to
    proceed_to_match (RunEngine signal). Both signals required after
    the post-Slice-8 hardening."""
    decision = _planner(
        move="ask_one_clarifying_question",
        reason_code="target_role_unclear",
        ask_slot="target_role_text",
        tone="warm_supportive",
    )
    truth = _truth(
        filled_slots=["target_role_text", "skills_text"],
        target_role_text="warehouse worker",
        target_role_specificity="specific",
        enough_to_match=True,
        usable_evidence_present=True,   # explicit, post-hardening
    )
    result = validate_planner_intent(decision, truth)
    assert isinstance(result, RunEngine)
    # Tone preserved from planner
    assert result.planner_tone == "warm_supportive"


def test_pass1_rule4_does_NOT_reroute_to_engine_when_evidence_missing(monkeypatch):
    """Post-Slice-8 live-test regression. The exact bug shape: planner
    asks for a slot that's strongly filled (e.g. target_role_text
    set to a specific job), `enough_to_match` is set true (contradictory
    but possible if truth_summary drifts), BUT usable_evidence_present
    is false (no resume + < 3 chat skills). Pre-hardening this hit
    Rule 4's RunEngine path and the engine ran on a thin profile.
    Now both signals must be true; falls through to next-slot ask."""
    decision = _planner(
        move="ask_one_clarifying_question",
        reason_code="target_role_unclear",
        ask_slot="target_role_text",
        tone="warm_supportive",
    )
    truth = _truth(
        filled_slots=["target_role_text"],
        target_role_text="truck and coach technician",
        target_role_specificity="specific",
        # Contradictory state (defense-in-depth scenario):
        enough_to_match=True,
        usable_evidence_present=False,  # the new guard
    )
    result = validate_planner_intent(decision, truth)
    # MUST NOT return RunEngine
    assert not isinstance(result, RunEngine), (
        "Rule 4 must not reroute to the engine when usable_evidence_present "
        "is False, regardless of enough_to_match. This catches the live-test "
        "bug where a user supplied only a target role and the engine ran "
        "anyway on title-match alone."
    )
    assert isinstance(result, ArbiterDecision)
    assert result.final_move == "ask_one_clarifying_question"
    # Should pick a different unfilled slot (the user's profile lacks evidence,
    # so we ask for skills/experience next).
    assert result.ask_slot != "target_role_text"
    assert result.arbiter_action == "overrode_to_ask"


def test_pass1_rule4_does_NOT_reroute_when_only_target_role_filled_no_evidence():
    """The live-test scenario reduced to the smallest possible case:
    user has set target_role only (specific), no resume, no chat skills.
    truth_summary computes:
        target_role_specificity = "specific"
        resume_parse_quality    = "no_resume"
        chat_skill_count        = 0
        usable_evidence_present = False
        enough_to_match         = False
    Planner asks for skills. Even though target_role IS strongly filled,
    the arbiter must not reroute to the engine -- no evidence to match
    against."""
    decision = _planner(
        move="ask_one_clarifying_question",
        reason_code="insufficient_profile_evidence",
        ask_slot="target_role_text",   # would trigger reroute path
        tone="warm_supportive",
    )
    truth = _truth(
        filled_slots=["target_role_text"],
        target_role_text="truck and coach technician",
        target_role_specificity="specific",
        enough_to_match=False,           # realistic state
        usable_evidence_present=False,   # realistic state
        resume_parse_quality="no_resume",
    )
    result = validate_planner_intent(decision, truth)
    assert not isinstance(result, RunEngine), (
        "Engine must not run on a profile with only a target role and "
        "no skill/resume evidence. This is the no-hidden-matching rule."
    )


def test_pass1_rule4_reroutes_to_next_slot_when_strongly_filled_but_not_enough():
    """Slot strongly filled but enough_to_match=false -> ask next
    canonical priority slot, NOT the duplicate one."""
    decision = _planner(
        move="ask_one_clarifying_question",
        reason_code="target_role_unclear",
        ask_slot="target_role_text",
    )
    truth = _truth(
        filled_slots=["target_role_text"],
        target_role_text="warehouse worker",
        target_role_specificity="specific",
        enough_to_match=False,
        usable_evidence_present=True,
    )
    result = validate_planner_intent(decision, truth)
    assert isinstance(result, ArbiterDecision)
    assert result.final_move == "ask_one_clarifying_question"
    assert result.arbiter_action == "overrode_to_ask"
    assert result.ask_slot == "skills_text"  # next canonical after target_role
    assert result.reason_code == ARBITER_REASON_DUPLICATE_ASK


# THE KEY USER REVIEW CASE: weak fill should NOT reroute.
def test_pass1_rule4_does_NOT_reroute_when_target_role_is_filled_but_vague():
    """Slice 4 review tightening: target_role='same role' is technically
    filled but weak. The planner's question must go through. Catches
    the case the reviewer flagged."""
    decision = _planner(
        move="ask_one_clarifying_question",
        reason_code="target_role_unclear",
        ask_slot="target_role_text",
        tone="warm_supportive",
    )
    truth = _truth(
        filled_slots=["target_role_text"],
        target_role_text="same role",
        target_role_specificity="vague",   # WEAK fill
        enough_to_match=False,
    )
    result = validate_planner_intent(decision, truth)
    assert isinstance(result, ArbiterDecision)
    assert result.arbiter_action == "passed_planner_through"
    assert result.final_move == "ask_one_clarifying_question"
    assert result.ask_slot == "target_role_text"


def test_pass1_rule4_does_NOT_reroute_when_other_slot_filled_without_usability_proxy():
    """Conservative default: slots without a usability proxy (skills,
    experience, etc.) default to weak fill -> planner's question goes
    through. Letting the planner ask again is cheaper than silently
    skipping a needed question."""
    decision = _planner(
        move="ask_one_clarifying_question",
        reason_code="insufficient_profile_evidence",
        ask_slot="skills_text",
        tone="warm_supportive",
    )
    truth = _truth(
        filled_slots=["target_role_text", "skills_text"],
        target_role_specificity="specific",
        enough_to_match=False,
    )
    result = validate_planner_intent(decision, truth)
    assert isinstance(result, ArbiterDecision)
    assert result.arbiter_action == "passed_planner_through"
    assert result.ask_slot == "skills_text"


def test_is_slot_strongly_filled_returns_false_when_slot_not_in_filled():
    truth = _truth(filled_slots=[], target_role_specificity="specific")
    assert not _is_slot_strongly_filled("target_role_text", truth)


def test_is_slot_strongly_filled_target_role_requires_specificity_specific():
    truth = _truth(
        filled_slots=["target_role_text"],
        target_role_specificity="vague",
    )
    assert not _is_slot_strongly_filled("target_role_text", truth)

    truth_specific = _truth(
        filled_slots=["target_role_text"],
        target_role_specificity="specific",
    )
    assert _is_slot_strongly_filled("target_role_text", truth_specific)


@pytest.mark.parametrize("slot", [
    "skills_text", "experience_text", "work_type_preference",
    "shift_preference", "education_text",
])
def test_is_slot_strongly_filled_other_slots_default_to_weak(slot):
    """Slice 4 review tightening: any slot without a usability proxy
    defaults to weak fill. Add per-slot checks here as truth_summary
    exposes them."""
    truth = _truth(filled_slots=[slot])
    assert not _is_slot_strongly_filled(slot, truth)


# ===========================================================================
# Pass 1 -- Rule 6: passes other planner moves through unchanged
# ===========================================================================
@pytest.mark.parametrize("move,reason_code,ask_slot,tone", [
    ("acknowledge_and_continue", "user_confirmed", None, "brief_confident"),
    ("explain_gap", "credential_gap_present", None, "honest_redirect"),
    ("offer_refinement", "narrow_request", None, "brief_confident"),
    # redirect_scope no longer fits this parametrize since the new
    # Rule 3 (planner-overreach override) needs scope_violations
    # populated to allow passthrough. Tested separately below.
])
def test_pass1_passes_other_moves_through(move, reason_code, ask_slot, tone):
    """No override applies -> planner's move + reason + tone + ask_slot
    pass through unchanged."""
    decision = _planner(
        move=move, reason_code=reason_code, ask_slot=ask_slot, tone=tone,
    )
    result = validate_planner_intent(decision, _truth())
    assert isinstance(result, ArbiterDecision)
    assert result.arbiter_action == "passed_planner_through"
    assert result.final_move == move
    assert result.reason_code == reason_code
    assert result.ask_slot == ask_slot
    assert result.tone == tone


def test_pass1_redirect_scope_passes_through_when_scope_violations_present():
    """The passthrough case for redirect_scope: scope IS actually
    violated. Note that Rule 2 fires BEFORE the passthrough -- this
    is technically tested by scope-override tests above, but pinning
    the explicit passthrough shape here completes the matrix."""
    decision = _planner(
        move="redirect_scope",
        reason_code="scope_violation_immigration",
        tone="honest_redirect",
    )
    truth = _truth(scope_violations_detected=["immigration"])
    result = validate_planner_intent(decision, truth)
    assert isinstance(result, ArbiterDecision)
    assert result.final_move == "redirect_scope"
    # Rule 2 fires first when scope is non-empty -- arbiter_action is
    # overrode_to_redirect even though the planner picked the same
    # move. Architecturally this is correct (the arbiter independently
    # confirmed the scope concern, didn't just trust the planner).
    assert result.arbiter_action == "overrode_to_redirect"


# =========================================================================
# Post-cold-session-fix: planner-overreach override on invented
# redirect_scope (Rule 3)
# =========================================================================
# Live test exposed: cold-session user asks "how can I get my Class G
# driver's licence?" -- Haiku invented a scope violation and emitted
# redirect_scope even though scope_violations_detected was empty AND
# intent was asking_about_gap. Rule 3 routes the user correctly
# based on actual intent rather than the planner's hallucinated reason.
def test_pass1_overrides_redirect_scope_to_explain_gap_when_intent_is_gap():
    """The exact bug shape from the live test: planner emits
    redirect_scope on a cold-session credential question; truth shows
    scope_violations empty and intent=asking_about_gap. Arbiter must
    override to explain_gap."""
    decision = _planner(
        move="redirect_scope",
        reason_code="scope_violation_off_topic",
        tone="honest_redirect",
    )
    truth = _truth(
        scope_violations_detected=[],            # planner went off-prompt
        user_intent_signal="asking_about_gap",
        target_role_text=None,
        target_role_specificity="none",
    )
    result = validate_planner_intent(decision, truth)
    assert isinstance(result, ArbiterDecision)
    assert result.final_move == "explain_gap"
    assert result.arbiter_action == "overrode_to_explain_gap"
    assert result.reason_code == "credential_gap_present"


def test_pass1_overrides_redirect_scope_to_ask_when_intent_is_not_gap():
    """Same bug shape but intent is NOT asking_about_gap. Safest move
    is a clarifying question -- don't reject the user on a fabricated
    scope concern, but also don't force-route to explain_gap if the
    user didn't actually ask about a gap."""
    decision = _planner(
        move="redirect_scope",
        reason_code="scope_violation_off_topic",
        tone="honest_redirect",
    )
    truth = _truth(
        scope_violations_detected=[],
        user_intent_signal="neutral",
        target_role_text=None,
        target_role_specificity="none",
    )
    result = validate_planner_intent(decision, truth)
    assert isinstance(result, ArbiterDecision)
    assert result.final_move == "ask_one_clarifying_question"
    assert result.arbiter_action == "overrode_to_ask"


def test_pass1_redirect_scope_with_real_scope_violation_still_redirects():
    """Inverse case: planner emits redirect_scope AND scope IS
    actually violated. Rule 2 fires first; result is redirect_scope
    (no override to explain_gap even if intent happens to be
    asking_about_gap, because scope wins precedence)."""
    decision = _planner(
        move="redirect_scope",
        reason_code="scope_violation_immigration",
        tone="honest_redirect",
    )
    truth = _truth(
        scope_violations_detected=["immigration"],   # REAL violation
        user_intent_signal="asking_about_gap",
    )
    result = validate_planner_intent(decision, truth)
    assert isinstance(result, ArbiterDecision)
    assert result.final_move == "redirect_scope"
    # Rule 2 (scope override) wins over Rule 3 (planner-overreach)
    assert result.arbiter_action == "overrode_to_redirect"


def test_pass1_invented_redirect_scope_override_preserves_planner_tone_on_gap():
    """When the override routes to explain_gap, the planner's tone
    carries through. (Planner picked honest_redirect for its
    redirect; but the override is going to a gap explanation, so the
    tone needs to suit that. Implementation choice: preserve
    planner_tone if set, else warm_supportive default.)"""
    decision = _planner(
        move="redirect_scope",
        reason_code="scope_violation_off_topic",
        tone="warm_supportive",
    )
    truth = _truth(
        scope_violations_detected=[],
        user_intent_signal="asking_about_gap",
    )
    result = validate_planner_intent(decision, truth)
    assert result.tone == "warm_supportive"


def test_pass1_passes_ask_through_when_slot_not_filled():
    """Planner asks for an unfilled slot -> question goes through."""
    decision = _planner(
        move="ask_one_clarifying_question",
        reason_code="target_role_unclear",
        ask_slot="target_role_text",
        tone="warm_supportive",
    )
    truth = _truth(
        filled_slots=[],  # target_role_text NOT filled
        target_role_text=None,
        target_role_specificity="none",
        enough_to_match=False,
    )
    result = validate_planner_intent(decision, truth)
    assert isinstance(result, ArbiterDecision)
    assert result.arbiter_action == "passed_planner_through"
    assert result.ask_slot == "target_role_text"


# ===========================================================================
# Pass 1 INVARIANT (Slice 4 review): Pass 1 NEVER returns present_matches
# or present_no_match, no matter what inputs it sees
# ===========================================================================
# Exhaustively enumerate the relevant input dimensions and assert
# that no combination produces an outcome-move final_move. This is the
# whole architectural boundary of the slice.
def test_pass1_never_emits_match_outcome_moves_exhaustive():
    """Slice 4 review invariant: Pass 1 must NEVER return final_move
    in {present_matches, present_no_match}. Those moves are reachable
    ONLY via Pass 2 (after the engine has actually run).

    If this test ever fails, the architecture has broken: someone
    added a code path in Pass 1 that emits an outcome move, which
    means the engine is being skipped. That's the bug Slice 4 exists
    to prevent."""
    forbidden = {"present_matches", "present_no_match", "confirm_resume_summary"}

    moves = list(get_args(PlannerMove))  # all valid planner moves
    enough_options = (True, False)
    usable_options = (True, False)
    scope_options = ([], ["immigration"], ["national_wages"], ["off_topic"])
    target_specificity = ("specific", "vague", "none")
    filled_options = (
        [],
        ["target_role_text"],
        ["target_role_text", "skills_text"],
        ["target_role_text", "skills_text", "experience_text"],
    )

    checked = 0
    for move, enough, usable, scope, spec, filled in product(
        moves, enough_options, usable_options, scope_options,
        target_specificity, filled_options,
    ):
        # Build a planner decision -- some combos require ask_slot,
        # respect the schema by setting it for ask moves only.
        try:
            ask_slot = "target_role_text" if move == "ask_one_clarifying_question" else None
            reason_code = _reason_for(move)
            decision = _planner(
                move=move, reason_code=reason_code,
                ask_slot=ask_slot, tone="warm_supportive",
            )
        except Exception:
            continue
        truth = _truth(
            enough_to_match=enough,
            usable_evidence_present=usable,
            scope_violations_detected=scope,
            target_role_specificity=spec,
            target_role_text="warehouse worker" if spec == "specific" else None,
            filled_slots=filled,
        )
        result = validate_planner_intent(decision, truth)
        if isinstance(result, ArbiterDecision):
            assert result.final_move not in forbidden, (
                f"PASS 1 INVARIANT VIOLATED: Pass 1 returned "
                f"final_move={result.final_move!r} for inputs "
                f"(move={move}, enough_to_match={enough}, "
                f"usable_evidence_present={usable}, scope={scope}, "
                f"specificity={spec}, filled={filled}). "
                f"Pass 1 must NEVER emit match-outcome moves; only "
                f"Pass 2 (after engine) can. arbiter_action="
                f"{result.arbiter_action!r}, notes={result.notes!r}"
            )
        # RunEngine is fine here -- it's the signal to invoke pass 2
        checked += 1

    # Also explicitly test the None path
    result = validate_planner_intent(None, _truth())
    assert isinstance(result, ArbiterDecision)
    assert result.final_move not in forbidden
    checked += 1

    # Sanity: we actually exercised a lot of combinations, not a no-op
    assert checked > 200, (
        f"Invariant test only exercised {checked} combinations -- "
        f"check the enumeration; expected several hundred."
    )


def _reason_for(move: str) -> str:
    """Pick any valid reason_code for the given planner move so we
    can construct a decision in the invariant test."""
    return {
        "acknowledge_and_continue": "user_confirmed",
        "proceed_to_match": "resume_skills_sufficient",
        "ask_one_clarifying_question": "target_role_unclear",
        "explain_gap": "credential_gap_present",
        "offer_refinement": "narrow_request",
        "redirect_scope": "scope_violation_immigration",
    }[move]


def test_pass1_source_does_not_construct_match_outcome_moves():
    """Belt-and-suspenders structural check: the source of
    `validate_planner_intent` must not literally contain the strings
    'present_matches' or 'present_no_match' as final_move values.

    The exhaustive test above runs the function; this one reads the
    source. Either alone catches drift; both together catch drift
    even when the input enumeration misses a case."""
    import inspect
    src = inspect.getsource(validate_planner_intent)
    assert 'final_move="present_matches"' not in src, (
        "Pass 1 source constructs present_matches directly -- only "
        "Pass 2 (after engine) is allowed to."
    )
    assert 'final_move="present_no_match"' not in src, (
        "Pass 1 source constructs present_no_match directly -- only "
        "Pass 2 (after engine) is allowed to."
    )
    assert 'final_move="confirm_resume_summary"' not in src, (
        "Pass 1 source constructs confirm_resume_summary directly -- "
        "only the resume_upload gate (gates.py) is allowed to emit this."
    )


# ===========================================================================
# Pass 2 -- resolve_match_outcome
# ===========================================================================
def test_pass2_zero_matches_returns_present_no_match_with_honest_redirect():
    """No matches: force honest_redirect tone. brief_confident here
    would read as flippant. This is the ONE place Pass 2 overrides
    the planner's tone."""
    d = resolve_match_outcome(
        match_count=0,
        caps_applied=(),
        planner_reason_code="user_explicitly_asked_to_match",
        planner_tone="brief_confident",  # overridden
    )
    assert d.final_move == "present_no_match"
    assert d.reason_code == ARBITER_REASON_NO_MATCHES
    assert d.tone == "honest_redirect"
    assert d.arbiter_action == "resolved_to_no_match"
    assert d.caps_applied == ()


def test_pass2_matches_with_caps_preserves_planner_tone():
    """Slice 4 review tightening: caps must NOT force honest_redirect
    universally. Preserve the planner's tone; surface caps as a
    separate field. The responder narrates the cap honestly within
    whatever tone was chosen (warm/brief/etc.)."""
    d = resolve_match_outcome(
        match_count=3,
        caps_applied=("credential_310T_missing", "experience_under_2_years"),
        planner_reason_code="user_explicitly_asked_to_match",
        planner_tone="warm_supportive",   # preserved, NOT overridden
    )
    assert d.final_move == "present_matches"
    assert d.reason_code == ARBITER_REASON_MATCHES_WITH_CAPS
    assert d.tone == "warm_supportive"   # preserved!
    assert d.arbiter_action == "resolved_to_matches"
    assert d.caps_applied == (
        "credential_310T_missing", "experience_under_2_years",
    )


def test_pass2_matches_no_caps_preserves_planner_tone():
    """Same tone-preservation rule on the clean-matches path."""
    d = resolve_match_outcome(
        match_count=5,
        caps_applied=(),
        planner_reason_code="resume_skills_sufficient",
        planner_tone="brief_confident",
    )
    assert d.final_move == "present_matches"
    assert d.reason_code == ARBITER_REASON_MATCHES_FOUND
    assert d.tone == "brief_confident"
    assert d.caps_applied == ()


@pytest.mark.parametrize("planner_tone", [
    "brief_confident", "warm_supportive", "honest_redirect", "excited_share",
])
def test_pass2_preserves_every_planner_tone_on_matches(planner_tone):
    """Every Tone enum value must be preservable through Pass 2 on
    the matches path. If we ever add a tone, this test exercises it
    automatically."""
    d = resolve_match_outcome(
        match_count=2, caps_applied=(),
        planner_reason_code="resume_skills_sufficient",
        planner_tone=planner_tone,
    )
    assert d.tone == planner_tone


def test_pass2_uses_caps_with_caps_reason_when_caps_present():
    """The arbiter chooses its reason based on what HAPPENED, not the
    planner's pre-engine guess. matches_with_caps vs matches_found."""
    with_caps = resolve_match_outcome(
        match_count=1, caps_applied=("foo",),
        planner_reason_code="user_explicitly_asked_to_match",
        planner_tone="brief_confident",
    )
    no_caps = resolve_match_outcome(
        match_count=1, caps_applied=(),
        planner_reason_code="user_explicitly_asked_to_match",
        planner_tone="brief_confident",
    )
    assert with_caps.reason_code == ARBITER_REASON_MATCHES_WITH_CAPS
    assert no_caps.reason_code == ARBITER_REASON_MATCHES_FOUND


def test_pass2_caps_applied_is_immutable_tuple():
    """caps_applied is stored as a tuple so downstream consumers can't
    mutate the decision. ArbiterDecision is frozen too."""
    d = resolve_match_outcome(
        match_count=1, caps_applied=("foo",),
        planner_reason_code="user_explicitly_asked_to_match",
        planner_tone="brief_confident",
    )
    assert isinstance(d.caps_applied, tuple)
    with pytest.raises((AttributeError, TypeError, Exception)):
        # Both the dataclass freeze and the tuple immutability protect this
        d.caps_applied = ("bar",)  # type: ignore[misc]


# ===========================================================================
# Pass 2 -- present_near_miss outcome (Slice N, 2026-06-05)
#
# Behavior contract:
#   match_count == 0 AND near_miss_candidates non-empty
#     -> present_near_miss + warm_supportive + new reason code
#
#   match_count == 0 AND no near_miss_candidates (default)
#     -> existing present_no_match path (legacy callers unchanged)
#
#   match_count > 0 (with or without near_miss_candidates)
#     -> existing present_matches path (matches always win)
#
# These tests pin each of those branches plus the rollback property:
# an existing caller that doesn't pass the new arg gets byte-identical
# behavior to before Slice N-3.
# ===========================================================================
def test_pass2_near_miss_emits_present_near_miss_outcome():
    """Match count zero + non-empty near-miss list -> the new outcome."""
    d = resolve_match_outcome(
        match_count=0,
        near_miss_candidates=("synthetic-truck-tech-match",),
        planner_reason_code="user_explicitly_asked_to_match",
        planner_tone="brief_confident",   # overridden -> warm_supportive
    )
    assert d.final_move == "present_near_miss"
    assert d.reason_code == ARBITER_REASON_NEAR_MISS
    assert d.reason_code == "title_match_with_major_gaps"  # literal pin
    assert d.tone == "warm_supportive"   # locked design tone
    assert d.arbiter_action == "resolved_to_near_miss"
    assert d.caps_applied == ()


def test_pass2_near_miss_tone_does_not_inherit_planner_tone():
    """The near-miss outcome forces warm_supportive regardless of the
    planner tone passed in. honest_redirect would read as 'I can't
    help'; brief_confident would read as flippant given the gap."""
    for planner_tone in ("brief_confident", "honest_redirect",
                         "warm_supportive", "excited_share"):
        d = resolve_match_outcome(
            match_count=0,
            near_miss_candidates=("x",),
            planner_tone=planner_tone,
        )
        assert d.tone == "warm_supportive", (
            f"near-miss tone must be warm_supportive regardless of "
            f"planner_tone={planner_tone!r}; got {d.tone!r}"
        )


def test_pass2_empty_near_miss_candidates_falls_through_to_present_no_match():
    """Rollback property: passing `near_miss_candidates=[]` (or `()`)
    is identical to not passing the arg at all. Legacy no-match
    callers that haven't been updated yet keep working unchanged."""
    legacy = resolve_match_outcome(
        match_count=0,
        planner_reason_code="user_explicitly_asked_to_match",
        planner_tone="brief_confident",
    )
    explicit_empty = resolve_match_outcome(
        match_count=0,
        near_miss_candidates=(),
        planner_reason_code="user_explicitly_asked_to_match",
        planner_tone="brief_confident",
    )
    explicit_empty_list = resolve_match_outcome(
        match_count=0,
        near_miss_candidates=[],
        planner_reason_code="user_explicitly_asked_to_match",
        planner_tone="brief_confident",
    )
    # All three must produce identical ArbiterDecisions.
    assert legacy == explicit_empty == explicit_empty_list
    assert legacy.final_move == "present_no_match"
    assert legacy.arbiter_action == "resolved_to_no_match"


def test_pass2_present_matches_wins_over_near_miss_when_count_positive():
    """If match_count > 0 the matches path runs even if
    near_miss_candidates is also non-empty. This shouldn't happen in
    production (the handler only computes near-miss when no matches),
    but the precedence rule is locked: positive match_count always
    means present_matches."""
    d = resolve_match_outcome(
        match_count=3,
        near_miss_candidates=("synthetic-A", "synthetic-B"),
        planner_reason_code="user_explicitly_asked_to_match",
        planner_tone="warm_supportive",
    )
    assert d.final_move == "present_matches"
    assert d.arbiter_action == "resolved_to_matches"


def test_pass2_near_miss_decision_is_frozen():
    """ArbiterDecision is frozen. Confirm the near-miss path produces
    an equally immutable instance -- a future contributor can't
    quietly mutate the tone or reason on a downstream code path."""
    d = resolve_match_outcome(
        match_count=0, near_miss_candidates=("x",),
    )
    with pytest.raises((AttributeError, TypeError, Exception)):
        d.tone = "honest_redirect"   # type: ignore[misc]


def test_pass2_near_miss_accepts_arbitrary_sequence_types():
    """The arbiter doesn't import MatchResult; it only checks
    truthiness on `near_miss_candidates`. Tuple, list, generator
    (materialized) all qualify."""
    for nm in [
        ("placeholder",),
        ["placeholder"],
        [object()],
        ("a", "b", "c"),
    ]:
        d = resolve_match_outcome(match_count=0, near_miss_candidates=nm)
        assert d.final_move == "present_near_miss", (
            f"sequence type {type(nm).__name__} must trigger near-miss; "
            f"got {d.final_move!r}"
        )


# ===========================================================================
# Pass 2 INVARIANT: Pass 2 NEVER returns Pass-1-only moves
# ===========================================================================
def test_pass2_only_emits_match_outcome_moves():
    """Symmetric invariant: Pass 2 ONLY emits present_matches,
    present_no_match, or present_near_miss (Slice N). It must never
    emit ask/explain/refine/etc. Those come from Pass 1.

    Slice N (2026-06-05): added `present_near_miss` to the allowed
    set + iterates over near_miss_candidates variants so the
    invariant covers all three pass-2 outputs."""
    allowed_pass2_moves = {
        "present_matches", "present_no_match", "present_near_miss",
    }
    for count in (0, 1, 5, 100):
        for caps in ((), ("c1",), ("c1", "c2")):
            for nm in ((), ("synthetic-near-miss-candidate",)):
                d = resolve_match_outcome(
                    match_count=count, caps_applied=caps,
                    near_miss_candidates=nm,
                    planner_reason_code="user_explicitly_asked_to_match",
                    planner_tone="brief_confident",
                )
                assert d.final_move in allowed_pass2_moves, (
                    f"Pass 2 returned non-outcome move {d.final_move!r} "
                    f"for match_count={count}, caps={caps}, nm={nm}"
                )


# ===========================================================================
# Internal helpers
# ===========================================================================
@pytest.mark.parametrize("truth_kwargs,expected", [
    # No target role -> ask for it
    ({"target_role_text": None, "target_role_specificity": "none"},
     ("target_role_unclear", "target_role_text")),
    # Vague target role -> ask for it
    ({"target_role_text": "any job", "target_role_specificity": "vague"},
     ("target_role_unclear", "target_role_text")),
    # Specific target + failed resume -> ask for chat skills
    ({"target_role_text": "electrician", "target_role_specificity": "specific",
      "resume_parse_quality": "failed"},
     ("resume_failed_need_chat_skills", "skills_text")),
    # Specific target + good resume but enough_to_match_reason==no_usable_evidence
    ({"target_role_text": "electrician", "target_role_specificity": "specific",
      "enough_to_match_reason": "no_usable_evidence"},
     ("resume_failed_need_chat_skills", "skills_text")),
    # Default: insufficient evidence -> ask for skills
    ({"target_role_text": "electrician", "target_role_specificity": "specific",
      "resume_parse_quality": "full"},
     ("insufficient_profile_evidence", "skills_text")),
])
def test_pick_ask_reason_and_slot(truth_kwargs, expected):
    truth = _truth(**truth_kwargs)
    assert _pick_ask_reason_and_slot(truth) == expected


def test_next_unfilled_priority_slot_returns_first_canonical():
    truth = _truth(filled_slots=[])
    assert _next_unfilled_priority_slot(truth) == "target_role_text"


def test_next_unfilled_priority_slot_skips_filled():
    truth = _truth(filled_slots=["target_role_text", "skills_text"])
    assert _next_unfilled_priority_slot(truth) == "experience_text"


def test_next_unfilled_priority_slot_returns_none_when_all_filled():
    truth = _truth(filled_slots=[
        "target_role_text", "skills_text", "experience_text",
        "work_type_preference", "shift_preference", "education_text",
    ])
    assert _next_unfilled_priority_slot(truth) is None


# ===========================================================================
# Structural: every OutcomeMove value is reachable through some code path
# ===========================================================================
def test_every_outcome_move_is_reachable_through_some_path():
    """Every value in OutcomeMove must be producible by at least one
    code path. An unreachable outcome is dead enum noise.

    Most outcomes come from the arbiter (pass 1 passthrough or pass 2
    resolution). One outcome -- `confirm_resume_summary` -- is
    GATE-emitted only (the resume_upload gate in gates.py). We
    exercise it via the gate to prove the union of paths covers the
    enum.
    """
    reachable: set[str] = set()

    # Pass 2 -> present_matches + present_no_match + present_near_miss
    d = resolve_match_outcome(match_count=0)
    reachable.add(d.final_move)
    d = resolve_match_outcome(match_count=1, planner_reason_code="x", planner_tone="brief_confident")
    reachable.add(d.final_move)
    # Slice N: near_miss is reachable when match_count==0 AND
    # near_miss_candidates non-empty. Use a synthetic placeholder --
    # the arbiter only checks truthiness.
    d = resolve_match_outcome(
        match_count=0, near_miss_candidates=("synthetic-candidate",),
    )
    reachable.add(d.final_move)
    # AR-9.feat.coach-tiers CP2 step 2: present_tiered_matches is
    # reachable when the handler supplies tiered_evidence_available=True
    # alongside a positive match_count.
    d = resolve_match_outcome(match_count=1, tiered_evidence_available=True)
    reachable.add(d.final_move)

    # Pass 1 passthrough (or override) for each non-proceed planner move.
    # redirect_scope needs scope_violations populated so Rule 3
    # (planner-overreach override) doesn't fire and route it away.
    for move in ("acknowledge_and_continue", "ask_one_clarifying_question",
                 "explain_gap", "offer_refinement", "redirect_scope"):
        decision = _planner(
            move=move,
            reason_code=_reason_for(move),
            ask_slot="target_role_text" if move == "ask_one_clarifying_question" else None,
        )
        truth = _truth(filled_slots=[])
        if move == "redirect_scope":
            # Real scope violation present -> redirect passes through
            # (or Rule 2 fires; either way redirect_scope is the result)
            truth = _truth(
                filled_slots=[], scope_violations_detected=["immigration"],
            )
        if move == "ask_one_clarifying_question":
            # Search-first override (2026-06-16 evening): when truth says
            # enough_to_match + usable_evidence + alignment, the arbiter
            # now forces RunEngine on an ask. To exercise the passthrough
            # path, give a truth that does NOT meet engine-run conditions
            # — a cold profile with no skills/experience yet.
            truth = _truth(
                filled_slots=[],
                enough_to_match=False,
                enough_to_match_reason="insufficient_skill_evidence",
                usable_evidence_present=False,
            )
        r = validate_planner_intent(decision, truth)
        if isinstance(r, ArbiterDecision):
            reachable.add(r.final_move)

    # GATE-emitted: confirm_resume_summary comes from the resume_upload
    # gate, not from the arbiter. Exercise via the gate.
    from skillbridge.chat import gates as gates_module
    gate_decision = gates_module._is_resume_upload_gate(uploaded_file=True)
    assert gate_decision is not None
    reachable.add(gate_decision.final_move)

    # R-3 (remaining-gaps iteration): handler-synthesized -- neither
    # arbiter Pass 1 nor Pass 2 can produce these. Exercise via the
    # synthesis helpers in handler.py to prove the union of paths
    # covers the enum. (A separate invariant test below checks that
    # the arbiter CANNOT produce explain_remaining_gaps.)
    from skillbridge.chat import handler as handler_module
    from skillbridge.chat.remaining_gaps import (
        CredentialClaim, RemainingGapsIntent,
    )
    from skillbridge.session.staging import StagedProfile
    _sp = StagedProfile.new("test-arbiter")
    _intent = RemainingGapsIntent(
        kind="subtract",
        current_turn_claims=(CredentialClaim(canonical="X", mode="claimed"),),
    )
    _decision = handler_module._synthesize_remaining_gaps_decision(
        _sp, _intent, retracted=False,
    )
    reachable.add(_decision.final_move)

    # AR-1c (adjacent-recommendations design v12): two more
    # handler-synthesized outcomes that neither arbiter Pass 1 nor
    # Pass 2 produce. The pure synthesis factories live in
    # match/adjacent.py; AR-6 wires the production dispatch. We
    # exercise them here so the reachability invariant holds at the
    # AR-1c commit even though no production caller exists yet.
    from skillbridge.match.adjacent import (
        _synthesize_describe_adjacent_role_decision,
        _synthesize_recommend_adjacent_roles_decision,
    )
    reachable.add(_synthesize_recommend_adjacent_roles_decision().final_move)
    reachable.add(_synthesize_describe_adjacent_role_decision().final_move)

    expected = set(get_args(OutcomeMove))
    missing = expected - reachable
    assert not missing, (
        f"OutcomeMove values not reachable through any path: "
        f"{missing}. Either remove the value from OutcomeMove, add "
        f"an arbiter code path that emits it, or wire it through a gate "
        f"or handler-synthesis helper."
    )
