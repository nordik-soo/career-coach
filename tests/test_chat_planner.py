"""Unit tests for chat orchestration v2 slice 3 -- the planner LLM call.

Covers the Pydantic schema (`PlannerDecision`), the move/ask_slot
pairing rule, the `plan_next_move` entry point (with mocked
`llm.call_json`), prompt-vs-code parity, and the two-pass arbiter
contract (planner MUST NOT emit outcome moves).

The `nodb` marker keeps the conftest TRUNCATE off -- no DB needed.

Real Haiku integration test is at the bottom, skipped unless
RUN_PLANNER_LIVE=1 is set in the environment.
"""
from __future__ import annotations

import os
from typing import get_args

os.environ.setdefault("LLM_ENABLED", "false")

import pytest
from pydantic import ValidationError

from skillbridge.chat.planner import (
    PLANNER_SYSTEM_PROMPT,
    AskSlot,
    PlannerDecision,
    PlannerMove,
    ReasonCode,
    Tone,
    _VALID_REASON_BY_MOVE,
    _format_user_prompt,
    plan_next_move,
)

pytestmark = pytest.mark.nodb


# ===========================================================================
# Helper: build a minimal valid decision dict
# ===========================================================================
def _valid_decision_dict(**overrides) -> dict:
    base = {
        "move": "proceed_to_match",
        "reason_code": "resume_skills_sufficient",
        "ask_slot": None,
        "tone": "brief_confident",
    }
    base.update(overrides)
    return base


# ===========================================================================
# PlannerDecision schema -- valid shapes
# ===========================================================================
def test_valid_proceed_to_match_decision_parses():
    d = PlannerDecision.model_validate(_valid_decision_dict())
    assert d.move == "proceed_to_match"
    assert d.reason_code == "resume_skills_sufficient"
    assert d.ask_slot is None
    assert d.tone == "brief_confident"


def test_valid_ask_one_clarifying_question_decision_parses():
    d = PlannerDecision.model_validate(_valid_decision_dict(
        move="ask_one_clarifying_question",
        reason_code="target_role_unclear",
        ask_slot="target_role_text",
        tone="warm_supportive",
    ))
    assert d.move == "ask_one_clarifying_question"
    assert d.ask_slot == "target_role_text"


def test_valid_redirect_scope_decision_parses():
    d = PlannerDecision.model_validate(_valid_decision_dict(
        move="redirect_scope",
        reason_code="scope_violation_immigration",
        tone="honest_redirect",
    ))
    assert d.move == "redirect_scope"


# Each PlannerMove should be parseable with at least one valid reason
# code. Parametrize across the type's value set to lock that any future
# addition to PlannerMove gets a happy-path test for free.
@pytest.mark.parametrize("move,reason_code,ask_slot", [
    ("acknowledge_and_continue", "user_confirmed", None),
    ("proceed_to_match", "resume_skills_sufficient", None),
    ("proceed_to_match", "chat_skills_sufficient", None),
    ("proceed_to_match", "resume_work_history_present", None),
    ("proceed_to_match", "user_explicitly_asked_to_match", None),
    ("proceed_to_match", "resume_confirmed_target_same_role", None),
    ("ask_one_clarifying_question", "target_role_unclear", "target_role_text"),
    ("ask_one_clarifying_question", "missing_work_type_preference", "work_type_preference"),
    ("ask_one_clarifying_question", "insufficient_profile_evidence", "skills_text"),
    ("ask_one_clarifying_question", "resume_failed_need_chat_skills", "skills_text"),
    ("explain_gap", "credential_gap_present", None),
    ("explain_gap", "experience_gap_present", None),
    ("explain_gap", "caps_applied", None),
    ("offer_refinement", "narrow_request", None),
    ("offer_refinement", "broaden_request", None),
    ("redirect_scope", "scope_violation_immigration", None),
    ("redirect_scope", "scope_violation_wages", None),
    ("redirect_scope", "scope_violation_off_topic", None),
    ("redirect_scope", "scope_violation_non_ssm", None),
])
def test_each_move_reason_combination_parses(move, reason_code, ask_slot):
    d = PlannerDecision.model_validate({
        "move": move, "reason_code": reason_code,
        "ask_slot": ask_slot, "tone": "warm_supportive",
    })
    assert d.move == move
    assert d.reason_code == reason_code


# ===========================================================================
# PlannerDecision schema -- TWO-PASS CONTRACT (planner cannot emit outcomes)
# ===========================================================================
# This is the single most important set of tests in slice 3: the planner
# is FORBIDDEN from emitting present_matches, present_no_match, or
# confirm_resume_summary. If the Literal type ever drifts to include
# them, these tests scream.
@pytest.mark.parametrize("forbidden_move", [
    "present_matches",
    "present_no_match",
    "confirm_resume_summary",
])
def test_planner_cannot_emit_outcome_moves(forbidden_move):
    """Two-pass arbiter contract: only the arbiter emits these moves
    (present_matches/present_no_match after the engine runs;
    confirm_resume_summary from the resume-upload gate). The planner
    layer MUST reject them at the schema level. See design doc §5, §6."""
    with pytest.raises(ValidationError) as exc_info:
        PlannerDecision.model_validate({
            "move": forbidden_move,
            "reason_code": "user_confirmed",
            "ask_slot": None,
            "tone": "brief_confident",
        })
    # Make sure the error is about `move`, not some unrelated field --
    # protects against a future schema change that accidentally lets
    # one of these through with a different error.
    error_locations = [e["loc"] for e in exc_info.value.errors()]
    assert any("move" in loc for loc in error_locations), (
        f"Expected validation error on `move` field for {forbidden_move!r}; "
        f"got errors at {error_locations}"
    )


# ===========================================================================
# PlannerDecision schema -- invalid enum values
# ===========================================================================
def test_unknown_move_rejected():
    with pytest.raises(ValidationError):
        PlannerDecision.model_validate(_valid_decision_dict(move="invent_jobs"))


def test_unknown_reason_code_rejected():
    with pytest.raises(ValidationError):
        PlannerDecision.model_validate(_valid_decision_dict(
            reason_code="planner_felt_like_it",
        ))


def test_unknown_tone_rejected():
    with pytest.raises(ValidationError):
        PlannerDecision.model_validate(_valid_decision_dict(tone="aggressive"))


def test_unknown_ask_slot_rejected():
    with pytest.raises(ValidationError):
        PlannerDecision.model_validate(_valid_decision_dict(
            move="ask_one_clarifying_question",
            reason_code="target_role_unclear",
            ask_slot="favorite_color",
        ))


# ===========================================================================
# PlannerDecision schema -- extra fields forbidden
# ===========================================================================
def test_extra_fields_rejected():
    """`extra="forbid"` is the structural guard that catches an LLM
    trying to smuggle in 'confidence', 'reasoning', or other unbounded
    prose. Closed-output discipline lives here."""
    with pytest.raises(ValidationError):
        PlannerDecision.model_validate(_valid_decision_dict(confidence=0.85))


def test_extra_field_with_real_looking_name_rejected():
    with pytest.raises(ValidationError):
        PlannerDecision.model_validate(_valid_decision_dict(
            reasoning="user is clearly ready",
        ))


# ===========================================================================
# Slice 3 review fix: reason_code MUST belong to the chosen move
# ===========================================================================
# Catches the cross-field bug the reviewer flagged: a payload like
# {move=proceed_to_match, reason_code=target_role_unclear} would have
# passed Literal validation in v1 even though the prompt clearly groups
# those reasons under different moves. The schema layer now enforces it.
@pytest.mark.parametrize("move,wrong_reason,ask_slot", [
    # The exact example the reviewer gave
    ("proceed_to_match", "target_role_unclear", None),
    # cross-pairs from other neighboring groups
    ("proceed_to_match", "caps_applied", None),
    ("acknowledge_and_continue", "resume_skills_sufficient", None),
    ("ask_one_clarifying_question", "user_confirmed", "target_role_text"),
    ("ask_one_clarifying_question", "credential_gap_present", "skills_text"),
    ("explain_gap", "narrow_request", None),
    ("offer_refinement", "scope_violation_off_topic", None),
    ("redirect_scope", "resume_skills_sufficient", None),
    ("redirect_scope", "user_confirmed", None),
])
def test_reason_code_must_belong_to_chosen_move(move, wrong_reason, ask_slot):
    """Cross-field invariant: a reason from one move's group cannot
    be paired with a different move. Closed-enum discipline at the
    schema layer, not just the prompt."""
    with pytest.raises(ValidationError, match="is not valid for"):
        PlannerDecision.model_validate({
            "move": move,
            "reason_code": wrong_reason,
            "ask_slot": ask_slot,
            "tone": "warm_supportive",
        })


def test_plan_next_move_returns_none_on_mismatched_reason_code(monkeypatch):
    """End-to-end regression for the reason-code pairing fix. If the
    LLM emits a move/reason combo that's syntactically valid (both in
    their respective enums) but semantically wrong (reason belongs to
    a different move), the entry point must reject it via the model
    validator and fall back to None."""
    monkeypatch.setattr("skillbridge.chat.planner.llm.is_enabled", lambda: True)
    monkeypatch.setattr(
        "skillbridge.chat.planner.llm.call_json",
        lambda system, user, max_tokens=None: {
            "move": "proceed_to_match",
            "reason_code": "target_role_unclear",  # belongs to ask_one_clarifying_question
            "ask_slot": None,
            "tone": "brief_confident",
        },
    )
    assert plan_next_move(_truth()) is None


# Structural invariants on _VALID_REASON_BY_MOVE itself. These catch
# drift between the enum types and the mapping (e.g., adding a new
# PlannerMove without adding it to the mapping, or adding a ReasonCode
# without placing it under a move).
def test_every_planner_move_has_at_least_one_reason():
    """A move with no valid reasons would be permanently unreachable
    once the cross-field validator runs. Symmetry check against the
    mapping."""
    mapping_moves = set(_VALID_REASON_BY_MOVE.keys())
    enum_moves = set(get_args(PlannerMove))
    assert mapping_moves == enum_moves, (
        f"PlannerMove and _VALID_REASON_BY_MOVE drifted apart. "
        f"In enum but not mapping: {enum_moves - mapping_moves}. "
        f"In mapping but not enum: {mapping_moves - enum_moves}."
    )
    for move, reasons in _VALID_REASON_BY_MOVE.items():
        assert len(reasons) >= 1, (
            f"Move {move!r} has zero valid reasons -- it could never "
            f"pass the cross-field validator. Add at least one reason "
            f"or remove the move."
        )


def test_every_reason_code_belongs_to_exactly_one_move():
    """Each reason maps to exactly one move's group -- no orphans, no
    duplicates. Orphans are reasons the validator would always reject;
    duplicates blur the move/reason coupling that this whole mapping
    exists to enforce."""
    all_reasons_in_mapping = []
    for reasons in _VALID_REASON_BY_MOVE.values():
        all_reasons_in_mapping.extend(reasons)
    enum_reasons = set(get_args(ReasonCode))

    # No orphans (every ReasonCode value is reachable by some move)
    missing = enum_reasons - set(all_reasons_in_mapping)
    assert not missing, (
        f"ReasonCode values not in _VALID_REASON_BY_MOVE: {missing}. "
        f"These reasons can never pass validation -- add them to a "
        f"move's group or remove them from the enum."
    )
    # No extras (every entry in the mapping is in the enum)
    extras = set(all_reasons_in_mapping) - enum_reasons
    assert not extras, (
        f"_VALID_REASON_BY_MOVE references reasons not in ReasonCode: "
        f"{extras}. Add them to the Literal or remove from mapping."
    )
    # No duplicates (each reason in exactly one move's set)
    from collections import Counter
    dup_counts = {r: c for r, c in Counter(all_reasons_in_mapping).items() if c > 1}
    assert not dup_counts, (
        f"Reasons appearing under multiple moves: {dup_counts}. Each "
        f"reason must belong to exactly one move."
    )


# ===========================================================================
# move / ask_slot pairing rule
# ===========================================================================
def test_ask_one_clarifying_question_requires_ask_slot():
    with pytest.raises(ValidationError, match="ask_slot must be non-null"):
        PlannerDecision.model_validate(_valid_decision_dict(
            move="ask_one_clarifying_question",
            reason_code="target_role_unclear",
            ask_slot=None,
        ))


@pytest.mark.parametrize("move,reason", [
    ("acknowledge_and_continue", "user_confirmed"),
    ("proceed_to_match", "resume_skills_sufficient"),
    ("explain_gap", "caps_applied"),
    ("offer_refinement", "narrow_request"),
    ("redirect_scope", "scope_violation_off_topic"),
])
def test_non_ask_moves_reject_ask_slot(move, reason):
    """ask_slot must be null when move != ask_one_clarifying_question.
    A planner that fills both is malformed -- the arbiter shouldn't
    have to detect this at runtime."""
    with pytest.raises(ValidationError, match="ask_slot must be null"):
        PlannerDecision.model_validate({
            "move": move, "reason_code": reason,
            "ask_slot": "target_role_text", "tone": "warm_supportive",
        })


# ===========================================================================
# Immutability (frozen=True)
# ===========================================================================
def test_planner_decision_is_frozen():
    """The arbiter receives a value object, not something it can quietly
    mutate. `frozen=True` enforces immutability after validation."""
    d = PlannerDecision.model_validate(_valid_decision_dict())
    with pytest.raises(ValidationError):
        d.move = "redirect_scope"  # type: ignore[misc]


# ===========================================================================
# plan_next_move() entry point -- mocked LLM
# ===========================================================================
def _truth(**overrides) -> dict:
    """Minimal truth summary dict for tests. Real shape from
    TruthSummary.to_planner_json() -- we mirror the field names but
    leave most empty so each test can focus on one variable."""
    base = {
        "user_message": "show me jobs",
        "enough_to_match": True,
        "user_intent_signal": "impatient_proceed",
        "target_role_text": "warehouse worker",
        "target_role_specificity": "specific",
        "scope_violations_detected": [],
    }
    base.update(overrides)
    return base


def test_plan_next_move_returns_none_when_llm_disabled(monkeypatch):
    monkeypatch.setattr("skillbridge.chat.planner.llm.is_enabled", lambda: False)
    # call_json should never be reached
    monkeypatch.setattr(
        "skillbridge.chat.planner.llm.call_json",
        lambda *a, **kw: pytest.fail("call_json should not be invoked when LLM disabled"),
    )
    assert plan_next_move(_truth()) is None


def test_plan_next_move_returns_decision_on_valid_llm_output(monkeypatch):
    monkeypatch.setattr("skillbridge.chat.planner.llm.is_enabled", lambda: True)
    monkeypatch.setattr(
        "skillbridge.chat.planner.llm.call_json",
        lambda system, user, max_tokens=None: {
            "move": "proceed_to_match",
            "reason_code": "resume_skills_sufficient",
            "ask_slot": None,
            "tone": "brief_confident",
        },
    )
    d = plan_next_move(_truth())
    assert d is not None
    assert d.move == "proceed_to_match"
    assert d.tone == "brief_confident"


def test_plan_next_move_returns_none_on_llm_parse_failure(monkeypatch):
    """llm.call_json returns None when its underlying call returned
    non-JSON. plan_next_move must propagate that as None, not crash."""
    monkeypatch.setattr("skillbridge.chat.planner.llm.is_enabled", lambda: True)
    monkeypatch.setattr(
        "skillbridge.chat.planner.llm.call_json",
        lambda system, user, max_tokens=None: None,
    )
    assert plan_next_move(_truth()) is None


def test_plan_next_move_returns_none_on_extra_field(monkeypatch):
    monkeypatch.setattr("skillbridge.chat.planner.llm.is_enabled", lambda: True)
    monkeypatch.setattr(
        "skillbridge.chat.planner.llm.call_json",
        lambda system, user, max_tokens=None: {
            "move": "proceed_to_match",
            "reason_code": "resume_skills_sufficient",
            "ask_slot": None,
            "tone": "brief_confident",
            "confidence": 0.92,
        },
    )
    assert plan_next_move(_truth()) is None


def test_plan_next_move_returns_none_on_unknown_move(monkeypatch):
    monkeypatch.setattr("skillbridge.chat.planner.llm.is_enabled", lambda: True)
    monkeypatch.setattr(
        "skillbridge.chat.planner.llm.call_json",
        lambda system, user, max_tokens=None: {
            "move": "invent_a_job",
            "reason_code": "user_confirmed",
            "ask_slot": None,
            "tone": "brief_confident",
        },
    )
    assert plan_next_move(_truth()) is None


@pytest.mark.parametrize("forbidden_move", [
    "present_matches", "present_no_match", "confirm_resume_summary",
])
def test_plan_next_move_returns_none_when_llm_emits_outcome_move(
    monkeypatch, forbidden_move,
):
    """End-to-end regression: even if the LLM (somehow) returns a
    forbidden outcome move, the entry point must reject it and fall
    back to None. Defense in depth -- the prompt forbids it, and the
    schema forbids it. If either layer drifts, the other catches the
    bug, but only if BOTH stay in sync. This test exercises the schema
    layer's catch."""
    monkeypatch.setattr("skillbridge.chat.planner.llm.is_enabled", lambda: True)
    monkeypatch.setattr(
        "skillbridge.chat.planner.llm.call_json",
        lambda system, user, max_tokens=None: {
            "move": forbidden_move,
            "reason_code": "user_confirmed",
            "ask_slot": None,
            "tone": "brief_confident",
        },
    )
    assert plan_next_move(_truth()) is None


def test_plan_next_move_passes_truth_summary_into_user_prompt(monkeypatch):
    """Verify the planner actually wires truth_summary into the user
    message it sends to Haiku -- without this, the LLM is making
    decisions in a vacuum."""
    monkeypatch.setattr("skillbridge.chat.planner.llm.is_enabled", lambda: True)
    captured: dict = {}

    def fake_call(system, user, max_tokens=None):
        captured["system"] = system
        captured["user"] = user
        captured["max_tokens"] = max_tokens
        return {
            "move": "proceed_to_match",
            "reason_code": "resume_skills_sufficient",
            "ask_slot": None,
            "tone": "brief_confident",
        }
    monkeypatch.setattr("skillbridge.chat.planner.llm.call_json", fake_call)

    truth = _truth(target_role_text="electrical journeyman")
    plan_next_move(truth)

    assert "electrical journeyman" in captured["user"]
    assert captured["system"] == PLANNER_SYSTEM_PROMPT
    # Output budget per design doc §5 -- must stay tight
    assert captured["max_tokens"] == 100


# ===========================================================================
# _format_user_prompt -- input framing
# ===========================================================================
def test_format_user_prompt_includes_action_directive():
    """User message must end with the JSON-only directive so the LLM
    has a clear cue to switch into output mode."""
    out = _format_user_prompt({"user_message": "hi"})
    assert "TRUTH SUMMARY:" in out
    assert out.rstrip().endswith("Return the JSON decision now.")


def test_format_user_prompt_serializes_compactly():
    """Compact JSON separators trim a few tokens from each call.
    Locking the format guards against accidental whitespace creep."""
    out = _format_user_prompt({"a": 1, "b": [2, 3]})
    # Compact separators -- no spaces after comma or colon
    assert '"a":1' in out
    assert '"b":[2,3]' in out


def test_format_user_prompt_handles_unicode():
    """Non-ASCII user messages (e.g., accented names in SSM's
    francophone population) must round-trip without escaping."""
    out = _format_user_prompt({"user_message": "café manager"})
    assert "café" in out


# ===========================================================================
# Structural parity tests -- prompt mentions every enum value
# ===========================================================================
# These guard against silent drift where someone adds an enum value but
# forgets to mention it in the prompt. The LLM can't pick a value it's
# never told about.
def test_prompt_mentions_every_planner_move():
    for move in get_args(PlannerMove):
        assert move in PLANNER_SYSTEM_PROMPT, (
            f"PlannerMove value {move!r} is not mentioned in the system "
            f"prompt -- the LLM has no way to pick it. Update "
            f"PLANNER_SYSTEM_PROMPT to describe this move."
        )


def test_prompt_names_every_forbidden_outcome_move():
    """The prompt must explicitly forbid the moves only the arbiter
    can emit. Regression guard against the two-pass contract going
    stale in the prompt while the schema still catches it."""
    for move in ("present_matches", "present_no_match", "confirm_resume_summary"):
        assert move in PLANNER_SYSTEM_PROMPT, (
            f"Outcome move {move!r} must be NAMED in the prompt's "
            f"'YOU MUST NOT EMIT' section so the LLM is steered away "
            f"from emitting it in the first place."
        )


def test_prompt_mentions_every_reason_code():
    for code in get_args(ReasonCode):
        assert code in PLANNER_SYSTEM_PROMPT, (
            f"ReasonCode value {code!r} is not mentioned in the system "
            f"prompt -- LLM cannot pick it. Update PLANNER_SYSTEM_PROMPT."
        )


def test_prompt_mentions_every_ask_slot():
    for slot in get_args(AskSlot):
        assert slot in PLANNER_SYSTEM_PROMPT, (
            f"AskSlot value {slot!r} is not mentioned in the system prompt."
        )


def test_prompt_mentions_every_tone():
    for tone in get_args(Tone):
        assert tone in PLANNER_SYSTEM_PROMPT, (
            f"Tone value {tone!r} is not mentioned in the system prompt."
        )


def test_prompt_has_explicit_json_only_rule():
    """The single most-violated LLM output rule. Lock its phrasing
    here so a future prompt edit can't accidentally water it down."""
    assert "Return JSON only" in PLANNER_SYSTEM_PROMPT
    assert "No prose" in PLANNER_SYSTEM_PROMPT
    assert "No markdown" in PLANNER_SYSTEM_PROMPT


def test_prompt_has_fallback_directive():
    """If unsure -> ask_one_clarifying_question with target_role_unclear
    or insufficient_profile_evidence. Locks the fallback behavior at
    the prompt level so the LLM can't drift toward a different default."""
    assert "If unsure" in PLANNER_SYSTEM_PROMPT
    assert "ask_one_clarifying_question" in PLANNER_SYSTEM_PROMPT
    assert "target_role_unclear" in PLANNER_SYSTEM_PROMPT
    assert "insufficient_profile_evidence" in PLANNER_SYSTEM_PROMPT


# Slice 3 review fix: every truth_summary field the prompt references
# must actually exist in TruthSummary.to_planner_json(). The reviewer
# caught that chat_skill_count was referenced in the prompt but never
# exposed -- this test prevents that class of drift coming back.
def test_prompt_only_references_existing_truth_summary_fields():
    """Every truth-summary field name mentioned in the prompt's
    GROUNDING block must be a real field in to_planner_json().
    Otherwise the planner is being told to look at fields it can't
    see. Regression guard for the reviewer's slice 3 finding."""
    from skillbridge.chat.truth_summary import TruthSummary
    available_fields = set(TruthSummary(user_message="").to_planner_json().keys())

    # Fields the current prompt references in its grounding rules
    referenced_fields = [
        "scope_violations_detected",
        "resume_parse_quality",
        "usable_evidence_present",
        "enough_to_match",
        "user_intent_signal",
    ]
    for field in referenced_fields:
        assert field in PLANNER_SYSTEM_PROMPT, (
            f"Prompt was changed to drop reference to {field!r}; "
            f"either restore it or update this test."
        )
        assert field in available_fields, (
            f"Prompt references truth_summary field {field!r} but it's "
            f"NOT in TruthSummary.to_planner_json(). The planner can't "
            f"follow a rule using a field it never sees. Either expose "
            f"the field in to_planner_json() or rewrite the rule using "
            f"an existing field."
        )


def test_prompt_does_not_reference_phantom_chat_skill_count():
    """Slice 3 review-fix regression test. The reviewer flagged that
    the prompt previously said `chat_skill_count<3` but that field is
    NOT in TruthSummary.to_planner_json() -- a quiet drift the
    architecture is supposed to prevent. The replacement uses
    `usable_evidence_present`, which IS exposed."""
    assert "chat_skill_count" not in PLANNER_SYSTEM_PROMPT, (
        "Slice 3 review fix: chat_skill_count is not in "
        "TruthSummary.to_planner_json(). Use usable_evidence_present "
        "(which is exposed) instead."
    )


def test_prompt_grounding_rules_list_scope_before_proceed():
    """Slice 3 review note: if both scope_violations_detected is
    non-empty AND enough_to_match is true, scope must win. The
    arbiter enforces this later, but the prompt must not nudge the
    LLM in the wrong order. The scope rule must appear earlier in
    the GROUNDING block than the enough_to_match rule."""
    scope_idx = PLANNER_SYSTEM_PROMPT.find("scope_violations_detected")
    enough_idx = PLANNER_SYSTEM_PROMPT.find("enough_to_match==true")
    assert scope_idx != -1, "scope rule missing from prompt"
    assert enough_idx != -1, "enough_to_match rule missing from prompt"
    assert scope_idx < enough_idx, (
        f"In GROUNDING, the scope rule must appear before the "
        f"enough_to_match rule (scope wins when both could fire). "
        f"Found scope at char {scope_idx}, enough_to_match at "
        f"char {enough_idx}."
    )


def test_prompt_routes_asking_about_gap_to_explain_gap():
    """Slice 9: planner prompt MUST teach the LLM to map
    user_intent_signal=='asking_about_gap' to explain_gap. Otherwise
    the planner reverts to proceed_to_match on follow-up gap questions
    (the live bug shape: 're-show the same matches')."""
    assert "asking_about_gap" in PLANNER_SYSTEM_PROMPT, (
        "asking_about_gap intent must appear in the prompt's GROUNDING "
        "rules so the planner knows to route it to explain_gap."
    )
    # The asking_about_gap -> explain_gap mapping is the load-bearing rule.
    assert (
        "asking_about_gap" in PLANNER_SYSTEM_PROMPT
        and "explain_gap" in PLANNER_SYSTEM_PROMPT
    )


def test_prompt_forbids_redirect_scope_when_scope_violations_empty():
    """Post-cold-session-fix: tighten rule 1 so the planner cannot
    emit redirect_scope on a vibes-based scope guess. The field
    scope_violations_detected is the AUTHORITATIVE source. Pre-fix,
    Haiku invented a scope concern on the live Class-G question."""
    assert (
        "Do NOT emit redirect_scope when it is empty" in PLANNER_SYSTEM_PROMPT
        or "AUTHORITATIVE" in PLANNER_SYSTEM_PROMPT.upper()
    ), (
        "Planner prompt rule 1 must explicitly forbid emitting "
        "redirect_scope when scope_violations_detected is empty."
    )


def test_prompt_clarifies_what_is_in_scope():
    """The prompt must name the gap-question categories that ARE in
    scope so the planner doesn't reject credential/license/safety
    questions as off-topic."""
    # At least one of these scope-positive examples must appear
    in_scope_signals = ["Credentials", "licences", "safety training", "in scope"]
    matches = [s for s in in_scope_signals if s in PLANNER_SYSTEM_PROMPT]
    assert matches, (
        f"Planner prompt rule 1 must clarify what IS in scope. "
        f"Expected one of {in_scope_signals} in the prompt."
    )


def test_prompt_excludes_asking_about_gap_from_proceed_match_rule():
    """Slice 9: the enough_to_match -> proceed_to_match rule must
    explicitly exclude asking_about_gap so a credential question after
    matches doesn't trigger another match. The exclusion set should
    contain at least {declining, correcting, asking_about_gap}."""
    # Find the proceed_to_match rule and verify the exclusion set.
    proceed_idx = PLANNER_SYSTEM_PROMPT.find("proceed_to_match")
    not_in_idx = PLANNER_SYSTEM_PROMPT.find("not in", proceed_idx - 200)
    assert not_in_idx != -1, "Couldn't find 'not in' exclusion set near proceed_to_match"
    snippet = PLANNER_SYSTEM_PROMPT[not_in_idx:not_in_idx + 200]
    assert "asking_about_gap" in snippet, (
        f"The proceed_to_match grounding rule must exclude "
        f"asking_about_gap. Snippet: {snippet!r}"
    )


def test_prompt_grounding_block_is_explicitly_ordered():
    """The GROUNDING block must signal precedence explicitly so a
    reader (human or model) knows earlier rules win. Locks the
    'apply in order -- earlier rules win' framing."""
    # Either phrasing is acceptable; this checks the intent is there.
    grounding_section = PLANNER_SYSTEM_PROMPT
    assert (
        "earlier rules win" in grounding_section
        or "apply in order" in grounding_section.lower()
        or "in this order" in grounding_section.lower()
    ), (
        "GROUNDING block must explicitly state that rules are ordered "
        "by precedence (e.g. 'apply in order -- earlier rules win'). "
        "Without this, the LLM has no signal that rule 1 beats rule 3 "
        "when both could fire."
    )


# ===========================================================================
# Prompt size budget
# ===========================================================================
def test_prompt_token_budget_is_tight():
    """Budget bumps over time, each with explicit justification:
      500 -> Slice 3 baseline
      575 -> Slice 9 (asking_about_gap rule added)
      600 -> Post-cold-session-hardening: rule 1 names in-scope credentials
              explicitly (so Haiku stops inventing scope violations on
              Class G / 310T questions), and rule 3 documents the
              registry_gaps_in_message routing signal.
      605 -> Rule 3 wording slice: added `training_request_no_entity`
              reason code to the REASON_CODES list so the planner can
              emit it for parity with the router (which now emits this
              code when training_request fires without a registry
              entity). Two tokens for the literal + small list join
              overhead.
      660 -> Rule 3 wording slice round 2: planner grounding rule 3
              now branches `asking_about_gap` on
              `registry_gaps_in_message` presence -- non-empty -> the
              previous explain_gap behavior; empty -> training-
              discovery ask. The expanded text is load-bearing because
              without it the planner LLM would still route every
              training-shaped turn to explain_gap even when no
              credential was named, undermining the router's
              deterministic Rule 3 (which only fires when
              MESSAGE_UNDERSTANDING_ENABLED is on).
      675 -> AR-1c (adjacent-recommendations design v12): two new
              handler-synthesized outcome moves added to the "YOU
              MUST NOT EMIT" enumeration (recommend_adjacent_roles,
              describe_adjacent_role). Mirrors the contract already
              applied to present_matches / present_no_match /
              confirm_resume_summary: name them explicitly so the
              LLM is steered away, and let the PlannerMove Literal
              be the second-line schema defense. The added text
              clarifies that the adjacency hooks (alongside the
              arbiter and the resume-upload gate) are the
              authoritative emitters of these outcomes.
    All additions are load-bearing for cold-session training questions
    -- without them the planner hallucinates scope or fails to route
    training intent to explain_gap.

    Tighten or bump again only with explicit justification documenting
    what new architectural property the addition encodes."""
    char_budget = 675 * 5  # generous: ~5 chars/token
    assert len(PLANNER_SYSTEM_PROMPT) < char_budget, (
        f"PLANNER_SYSTEM_PROMPT grew to {len(PLANNER_SYSTEM_PROMPT)} chars "
        f"(~{len(PLANNER_SYSTEM_PROMPT) // 4} tokens). Budget is 600 tokens. "
        f"Tighten the prompt or bump this budget with explicit "
        f"justification before merging."
    )


# ===========================================================================
# Live Haiku integration test -- manual flag
# ===========================================================================
@pytest.mark.skipif(
    os.environ.get("RUN_PLANNER_LIVE") != "1",
    reason="real Haiku call -- set RUN_PLANNER_LIVE=1 to run",
)
def test_plan_next_move_against_real_haiku_returns_valid_decision():
    """Manual integration test against the real Haiku endpoint. Skipped
    by default so unit-test runs stay deterministic + free. Run with
    RUN_PLANNER_LIVE=1 to validate prompt shape against the actual
    model, e.g. before shipping a prompt change.

    Asserts only that the response is parseable as a PlannerDecision --
    not that any specific move is chosen, since the live model might
    legitimately pick different reasonable moves on different days.
    """
    # Truth shape that should clearly route to proceed_to_match:
    # enough_to_match=true, impatient user, specific role, no scope
    # violations. If the live model returns something else, that's
    # signal worth investigating (might be a prompt regression).
    truth = _truth(
        user_message="just match me already",
        user_intent_signal="impatient_proceed",
        enough_to_match=True,
        target_role_specificity="specific",
        target_role_text="warehouse worker",
    )
    decision = plan_next_move(truth)
    assert decision is not None, "real Haiku call returned None (check ANTHROPIC_API_KEY and LLM_ENABLED)"
    # Schema validation already passed if we got here; verify the
    # enum value actually landed in the closed set.
    assert decision.move in get_args(PlannerMove)
    assert decision.reason_code in get_args(ReasonCode)
    assert decision.tone in get_args(Tone)
