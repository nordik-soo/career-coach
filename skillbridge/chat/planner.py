"""Chat orchestration v2 -- the planner LLM layer.

Slice 3. See docs/chat-orchestration-v2-design.md sections 5 + 6.1.

The planner is a Haiku call. It consumes a deterministic truth_summary
dict (produced by slice 1's `TruthSummary.to_planner_json()`) and
returns a JSON decision with four fields, all closed enums:

    {move, reason_code, ask_slot, tone}

The planner emits INTENT moves only. Outcome moves like
`present_matches` and `present_no_match` are produced exclusively by
the arbiter (slice 4) after running the engine -- the planner is not
allowed to skip the engine and jump to a result state. The
`confirm_resume_summary` outcome is the resume-upload gate's job, not
the planner's. The system prompt names all three forbidden moves in
its "YOU MUST NOT EMIT" block, and the `PlannerMove` Literal excludes
them at the type level; either layer alone would catch a drift, but
defense in depth catches drift earlier.

Fallback contract: any failure (LLM disabled, JSON parse error,
Pydantic validation error, API timeout, overload) returns None. The
arbiter (slice 4) treats None as a fallback signal and routes through
legacy `intake_state.decide()`. System degrades to today's UX, not
to a crash.

Cost / latency: see design doc §5. Prompt < 500 tokens, output < 100
tokens, no chain-of-thought, system prompt is cached. Target latency
< 400ms; marginal cost per turn ~$0.0001 on Haiku.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from skillbridge import llm

log = logging.getLogger(__name__)


# =========================================================================
# Closed enums -- INTENT moves only
# =========================================================================
# These three moves are DELIBERATELY excluded from PlannerMove:
#
#   - present_matches / present_no_match: only the arbiter emits these,
#     after running the engine in pass 2 (design doc §6.2). Letting the
#     planner emit them would let it skip the engine and invent a
#     result state.
#   - confirm_resume_summary: only the resume-upload gate emits this.
#     The planner never sees a turn where this is the right answer
#     because the gate short-circuits before the planner runs.
#
# If a future contributor adds one of these to PlannerMove, both the
# schema-level type check AND the prompt's "YOU MUST NOT EMIT" block
# would need to drift together for the bug to surface. The structural
# tests in test_chat_planner.py guard against silent drift.
PlannerMove = Literal[
    "acknowledge_and_continue",
    "proceed_to_match",
    "ask_one_clarifying_question",
    "explain_gap",
    "offer_refinement",
    "redirect_scope",
]


# Reason codes pin *why* the planner picked the move. Closed enum so
# transcript tests and the arbiter can switch on the value. Adding a
# new reason requires also adding it to the prompt -- the parity test
# in test_chat_planner.py enforces this.
ReasonCode = Literal[
    # acknowledge_and_continue
    "user_confirmed",
    # proceed_to_match
    "resume_skills_sufficient",
    "chat_skills_sufficient",
    "resume_work_history_present",
    "user_explicitly_asked_to_match",
    "resume_confirmed_target_same_role",
    # ask_one_clarifying_question
    "target_role_unclear",
    "missing_work_type_preference",
    "insufficient_profile_evidence",
    "resume_failed_need_chat_skills",
    # Router Rule 3: user expressed training_request intent but did NOT
    # name a registry credential. Semantically distinct from
    # `insufficient_profile_evidence` (which is "intake hasn't gathered
    # enough yet to match"). The responder renders a deterministic
    # training-discovery question for this code rather than the
    # role-aware skills-intake prompt.
    "training_request_no_entity",
    # explain_gap
    "credential_gap_present",
    "experience_gap_present",
    "caps_applied",
    # offer_refinement
    "narrow_request",
    "broaden_request",
    # redirect_scope
    "scope_violation_immigration",
    "scope_violation_wages",
    "scope_violation_off_topic",
    "scope_violation_non_ssm",
]


# Slot names match the staged-profile slot vocabulary (intake_priority).
# Only the slots intake_priority asks about are listed -- "passive"
# slots like preferred_location are intentionally excluded so the
# planner can't ask for them.
AskSlot = Literal[
    "target_role_text",
    "skills_text",
    "experience_text",
    "work_type_preference",
    "shift_preference",
    "education_text",
]


Tone = Literal[
    "brief_confident",
    "warm_supportive",
    "honest_redirect",
    "excited_share",
]


# =========================================================================
# Cross-field invariant: reason codes must belong to the chosen move
# =========================================================================
# The prompt's REASON_CODES block already groups reasons under their
# parent move, but a prompt is advisory -- the LLM could still pair a
# move with a reason from a different group. The schema enforces it:
# `move == "proceed_to_match"` with `reason_code == "target_role_unclear"`
# is rejected at validation time and the planner falls back to None.
#
# Adding a new reason code requires adding it here AND mentioning it
# in the prompt. The structural parity tests in test_chat_planner.py
# enforce the round-trip: every ReasonCode value must appear in
# exactly one move's set, and every PlannerMove must have at least
# one valid reason.
_VALID_REASON_BY_MOVE: dict[str, frozenset[str]] = {
    "acknowledge_and_continue": frozenset({
        "user_confirmed",
    }),
    "proceed_to_match": frozenset({
        "resume_skills_sufficient",
        "chat_skills_sufficient",
        "resume_work_history_present",
        "user_explicitly_asked_to_match",
        "resume_confirmed_target_same_role",
    }),
    "ask_one_clarifying_question": frozenset({
        "target_role_unclear",
        "missing_work_type_preference",
        "insufficient_profile_evidence",
        "resume_failed_need_chat_skills",
        "training_request_no_entity",
    }),
    "explain_gap": frozenset({
        "credential_gap_present",
        "experience_gap_present",
        "caps_applied",
    }),
    "offer_refinement": frozenset({
        "narrow_request",
        "broaden_request",
    }),
    "redirect_scope": frozenset({
        "scope_violation_immigration",
        "scope_violation_wages",
        "scope_violation_off_topic",
        "scope_violation_non_ssm",
    }),
}


# =========================================================================
# Pydantic model -- single source of truth for planner output shape
# =========================================================================
class PlannerDecision(BaseModel):
    """The planner's per-turn output.

    `extra="forbid"` rejects any fields outside the schema -- this is
    the structural guard that catches an LLM trying to smuggle in
    "confidence", "reasoning", or other unbounded prose.

    `frozen=True` makes the decision immutable once validated; the
    arbiter receives a value object, not something it can quietly
    mutate.

    The model validator enforces the move/ask_slot pairing rule so
    downstream callers don't need to defensively check it.
    """
    model_config = ConfigDict(extra="forbid", frozen=True)

    move: PlannerMove
    reason_code: ReasonCode
    ask_slot: AskSlot | None = None
    tone: Tone

    @model_validator(mode="after")
    def _enforce_cross_field_invariants(self) -> "PlannerDecision":
        """Two cross-field rules the Literal checks can't catch on their own.

        1. ask_slot pairing: must be set iff move ==
           ask_one_clarifying_question. "If you said you're asking,
           you must say what you're asking about; if you said you're
           not asking, don't sneak in a slot anyway."

        2. Reason-code grouping: reason_code must belong to the chosen
           move's reason set (see `_VALID_REASON_BY_MOVE`). Prevents
           silently mismatched pairs like
           {move=proceed_to_match, reason_code=target_role_unclear}
           which would pass Literal validation but contradict the
           grouped prompt taxonomy.
        """
        # ---- ask_slot pairing ----
        if self.move == "ask_one_clarifying_question":
            if self.ask_slot is None:
                raise ValueError(
                    "ask_slot must be non-null when "
                    "move == 'ask_one_clarifying_question'"
                )
        elif self.ask_slot is not None:
            raise ValueError(
                f"ask_slot must be null when move == {self.move!r} "
                f"(got ask_slot={self.ask_slot!r})"
            )

        # ---- reason_code grouping ----
        allowed = _VALID_REASON_BY_MOVE.get(self.move, frozenset())
        if self.reason_code not in allowed:
            raise ValueError(
                f"reason_code {self.reason_code!r} is not valid for "
                f"move {self.move!r}. Allowed for this move: "
                f"{sorted(allowed)}"
            )

        # ---- reason_code -> ask_slot binding ----
        # Rule 3 wording slice (round 26): some reason codes are
        # semantically tied to a SPECIFIC ask_slot. The responder
        # branches deterministically on reason_code, but if the slot
        # were mismatched the routing-vs-rendering pair would silently
        # disagree. Enforce the binding at schema time.
        required_slot = _REQUIRED_ASK_SLOT_BY_REASON.get(self.reason_code)
        if required_slot is not None and self.ask_slot != required_slot:
            raise ValueError(
                f"reason_code {self.reason_code!r} requires "
                f"ask_slot=={required_slot!r}; got ask_slot="
                f"{self.ask_slot!r}"
            )

        return self


# Reason codes whose semantics REQUIRE a specific ask_slot. The schema
# validator (above) rejects mismatched pairings so the router-vs-responder
# contract stays tight. Keep small; only add when the reason itself is
# only meaningful with that exact slot.
_REQUIRED_ASK_SLOT_BY_REASON: dict[str, str] = {
    # Rule 3 wording slice: the training-discovery question lives on the
    # skills_text slot. A different slot here would mean the responder
    # rendered "what skill do you want training for?" but the planner
    # claimed it was asking for, e.g., target_role_text -- a contract
    # break.
    "training_request_no_entity": "skills_text",
}


# =========================================================================
# System prompt -- cached by Anthropic prompt caching (5-min TTL)
# =========================================================================
# Token budget: < 500 system tokens per design doc §5. Anthropic prompt
# caching makes repeated calls within ~5 minutes pay ~90% less for
# these system tokens; the per-turn user message (truth_summary JSON)
# is the only un-cached portion.
PLANNER_SYSTEM_PROMPT = """\
You are SkillBridge SSM's chat planner. SkillBridge serves Sault Ste. Marie only.
Decide the next conversational MOVE from a deterministic truth summary. A
downstream engine -- not you -- decides whether matches exist.

OUTPUT
Return JSON only. One object. No prose. No markdown. No code fences. No
explanations, confidence, or notes. No fields outside the schema; unknown
fields are rejected.

Schema: {"move": M, "reason_code": R, "ask_slot": S|null, "tone": T}
- ask_slot MUST be set when M == "ask_one_clarifying_question"; null otherwise.
- If unsure: M="ask_one_clarifying_question", R="target_role_unclear" (no
  target role yet) or "insufficient_profile_evidence" (target known, no
  usable skills/work).

MOVES (M): acknowledge_and_continue | proceed_to_match |
ask_one_clarifying_question | explain_gap | offer_refinement | redirect_scope

YOU MUST NOT EMIT: present_matches, present_no_match, confirm_resume_summary,
recommend_adjacent_roles, describe_adjacent_role. The arbiter, the
resume-upload gate, and the adjacency hooks emit those; if you emit any of
them your output is rejected.

REASON_CODES (R), grouped by move:
- acknowledge_and_continue: user_confirmed
- proceed_to_match: resume_skills_sufficient, chat_skills_sufficient,
  resume_work_history_present, user_explicitly_asked_to_match,
  resume_confirmed_target_same_role
- ask_one_clarifying_question: target_role_unclear,
  missing_work_type_preference, insufficient_profile_evidence,
  resume_failed_need_chat_skills, training_request_no_entity
- explain_gap: credential_gap_present, experience_gap_present, caps_applied
- offer_refinement: narrow_request, broaden_request
- redirect_scope: scope_violation_immigration, scope_violation_wages,
  scope_violation_off_topic, scope_violation_non_ssm

ASK_SLOTS (S): target_role_text, skills_text, experience_text,
work_type_preference, shift_preference, education_text

TONES (T): brief_confident, warm_supportive, honest_redirect, excited_share

GROUNDING (deterministic facts in the truth summary; do not second-guess).
Apply these rules IN ORDER -- earlier rules win when both could fire:
1. scope_violations_detected non-empty -> redirect_scope. This field
   is the ONLY authoritative source. Do NOT emit redirect_scope when
   it is empty. Credentials, licences, safety training (WHMIS, 310T,
   Class G, forklift), and job-readiness questions are IN scope.
2. resume_parse_quality=="failed" AND usable_evidence_present==false ->
   ask_one_clarifying_question + resume_failed_need_chat_skills.
3. user_intent_signal=="asking_about_gap" -> branch on
   registry_gaps_in_message:
     * non-empty (user named a known credential) -> explain_gap.
     * empty (user asked about training generically) ->
       ask_one_clarifying_question + training_request_no_entity +
       skills_text. This is the training-discovery question; the
       responder renders a fixed prompt asking which skill or
       certificate they want training for.
   (registry_gaps alone WITHOUT asking_about_gap may be a skill claim;
   don't override on it.)
4. enough_to_match==true AND user_intent_signal not in
   {declining,correcting,asking_about_gap} -> proceed_to_match.
5. Never invent jobs, statistics, training, or URLs. You output four enum
   fields only; there is nowhere to put invented content.

Respond with the JSON object only.
"""


# =========================================================================
# Public entry point
# =========================================================================
def plan_next_move(
    truth_summary_json: dict[str, Any],
) -> PlannerDecision | None:
    """Ask Haiku for the next move given a deterministic truth summary.

    Args:
        truth_summary_json: dict produced by `TruthSummary.to_planner_json()`.
            The planner module owns the prompt-format contract; the
            truth_summary module doesn't need to know about LLM framing.

    Returns:
        A validated `PlannerDecision` on success, or `None` on any
        failure (LLM disabled, JSON parse error, schema validation
        error, API timeout, overload after retries). The arbiter
        treats `None` as a fallback signal and routes through legacy
        `intake_state.decide()`.

    Side effects:
        Logs at DEBUG when fallback is taken silently, WARNING when
        the LLM returned something we had to reject (so prompt drift
        is visible in logs over time).
    """
    if not llm.is_enabled():
        log.debug("planner: llm disabled, returning None")
        return None

    user_msg = _format_user_prompt(truth_summary_json)

    raw = llm.call_json(
        PLANNER_SYSTEM_PROMPT,
        user_msg,
        max_tokens=100,
    )
    if raw is None:
        log.debug("planner: llm.call_json returned None (JSON parse failed)")
        return None

    try:
        return PlannerDecision.model_validate(raw)
    except ValidationError as e:
        # Log the raw payload so drift is visible. Don't log the full
        # ValidationError stringification -- Pydantic v2's error
        # messages can be verbose; the first error.errors() entry is
        # enough to diagnose.
        first_err = e.errors()[0] if e.errors() else {}
        log.warning(
            "planner: schema validation failed (%s at %s); raw=%s",
            first_err.get("msg", "unknown"),
            first_err.get("loc", "?"),
            raw,
        )
        return None


def _format_user_prompt(truth_summary_json: dict[str, Any]) -> str:
    """Format the per-turn input. Keep it minimal: the system prompt
    carries the schema and rules; the user message carries only the
    truth summary plus a one-line action prompt. Compact JSON
    separators trim a few tokens from each call."""
    return (
        "TRUTH SUMMARY:\n"
        + json.dumps(truth_summary_json, ensure_ascii=False, separators=(",", ":"))
        + "\n\nReturn the JSON decision now."
    )
