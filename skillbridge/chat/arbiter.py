"""Chat orchestration v2 -- the deterministic arbiter.

Slice 4. See docs/chat-orchestration-v2-design.md section 6.

The arbiter has the final say on what the responder writes. It treats
planner output as ADVICE, not authority -- same defense-in-depth
pattern as matching v2's max-wins rule. The architectural contract is
"LLM proposes, backend disposes."

Two passes, both PURE FUNCTIONS (zero I/O in this module):

  Pass 1: validate_planner_intent(decision, truth) -> ArbiterDecision | RunEngine
    Independently re-checks the truth summary. Catches: planner==None,
    scope violations, unsafe proceed_to_match (when enough_to_match or
    usable_evidence_present is false), duplicate clarifying questions
    on STRONGLY-filled slots. Returns either a terminal
    ArbiterDecision or a RUN_ENGINE signal.

  Pass 2: resolve_match_outcome(...) -> ArbiterDecision
    Pure function over engine RESULTS (not engine itself). Resolves
    match_count + near_miss_candidates -> present_matches /
    present_no_match / present_near_miss (Slice N). Preserves planner
    tone on matches; surfaces caps as a separate field so the responder
    narrates them honestly within the planner's tone. Forces
    honest_redirect on the no-match path and warm_supportive on the
    near-miss path.

The handler (Slice 6) chains them with the actual engine call between.
Keeping the engine OUT of this module keeps the dangerous part
explicit at the call site -- the handler is the only place where
"LLM proposes, backend disposes" is observable end-to-end.

Pass 1 invariant: Pass 1 NEVER returns a Pass-2-only outcome --
present_matches, present_no_match, or present_near_miss. Those are
reachable ONLY through Pass 2. A structural test in
test_chat_arbiter.py enumerates inputs and verifies this exhaustively.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Literal

from skillbridge.chat.planner import PlannerDecision


# =========================================================================
# Outcome moves -- what the responder narrates
# =========================================================================
OutcomeMove = Literal[
    "acknowledge_and_continue",
    "ask_one_clarifying_question",
    "explain_gap",
    "offer_refinement",
    "redirect_scope",
    # Pass-2-only outcomes -- only the arbiter can produce these,
    # and only AFTER the engine has run.
    "present_matches",
    "present_no_match",
    # AR-9.feat.coach-tiers CP2: present_matches with the three-tier
    # coach response shape (Apply today / Worth a try / Sideways move).
    # The arbiter emits this ONLY when the handler has already built
    # TieredEvidence and passes `tiered_evidence_available=True` to
    # resolve_match_outcome. The arbiter does NOT infer tier
    # availability from match_count, caps, or near-miss signals.
    # Legacy ACTION_PRESENT_MATCHES action mapping preserved so the
    # session snapshot lifecycle and analytics consumers see this as
    # a present-matches continuation.
    "present_tiered_matches",
    # Slice N (2026-06-05): the "role exists but candidate is far from
    # ready" outcome. Fires when match_count == 0 AND the handler
    # supplied a non-empty `near_miss_candidates` list (pre-filtered
    # by `filter_near_miss_candidates`). The responder narrates this
    # as skill-gap analysis -- NOT "no jobs." See
    # docs/near-miss-gap-analysis-design.md.
    "present_near_miss",
    # GATE-only outcome. The resume_upload gate (gates.py) emits this
    # so the responder narrates the parsed-resume facts via the
    # existing RESUME_FACTS context. The arbiter NEVER emits this --
    # neither pass 1 nor pass 2. Tests exhaustively assert that.
    "confirm_resume_summary",
    # R-3 (remaining-gaps iteration): handler-synthesized only. Emitted
    # when the detection layer returns kind="subtract" or kind="retract"
    # (see docs/remaining-gaps-design.md §8). The arbiter NEVER emits
    # this -- neither pass 1 nor pass 2; the synthesis happens at the
    # handler level BEFORE either pass runs. A tests-as-invariants
    # check pins this contract.
    "explain_remaining_gaps",
    # AR-1c (adjacent recommendations): handler-synthesized only. The
    # planner and the standard direct-match engine NEVER produce these
    # moves; they're emitted by the dedicated adjacency hooks added in
    # AR-6 (gated on Redis-mode sessions via _adjacency_enabled). See
    # docs/adjacent-recommendations-design.md.
    #   - recommend_adjacent_roles: surfaces up to three different-role
    #     recommendations the user is not credentially blocked from.
    #   - describe_adjacent_role: re-fetches the live job by id and
    #     renders the chosen item from last_adjacent_snapshot when the
    #     user resolves an ordinal reference ("the second one").
    "recommend_adjacent_roles",
    "describe_adjacent_role",
]


# What the arbiter did -- logged on every decision for transcript tests
# and operational visibility.
ArbiterAction = Literal[
    "passed_planner_through",
    "overrode_to_redirect",
    "overrode_to_ask",
    "overrode_to_proceed",       # rule-4 reroute toward engine
    # Post-cold-session-fix: planner emitted redirect_scope but
    # truth.scope_violations_detected was empty (Haiku invented a
    # scope concern). When user intent is asking_about_gap, the
    # arbiter overrides to explain_gap so legitimate training
    # questions ("how do I get my Class G?") reach the recommender.
    "overrode_to_explain_gap",
    "fallback_to_legacy",
    "resolved_to_matches",       # pass 2
    "resolved_to_no_match",      # pass 2
    # Slice N (2026-06-05): pass 2 emits this when match_count==0 AND
    # near_miss_candidates is non-empty -- the role exists locally but
    # the user has major gaps. Distinct from resolved_to_no_match so
    # transcript tests and logs can tell them apart.
    "resolved_to_near_miss",     # pass 2
    # AR-9.feat.coach-tiers CP2 step 2: pass 2 emits this when the
    # handler supplied `tiered_evidence_available=True` AND match_count
    # is positive. Distinct from `resolved_to_matches` so transcript
    # tests can tell the legacy present_matches surface from the new
    # three-tier coach surface. The arbiter does NOT infer tier
    # availability; the handler is the authority.
    "resolved_to_tiered_matches",  # pass 2

    # Set by the handler when a deterministic gate (gates.py) fires
    # and the handler synthesizes an ArbiterDecision from the gate's
    # GateDecision so the responder can be called uniformly. The
    # arbiter itself never produces this value -- it's a handler-level
    # bridge between gates and the responder input shape.
    "gate_fired",
    # R-3 (remaining-gaps iteration): handler-synthesized when the
    # detection layer in skillbridge.chat.remaining_gaps returns a
    # truthy RemainingGapsIntent. Two variants so transcript tests can
    # tell the add/subtract path from the retraction path from the
    # confirmation-clarification path:
    #   - "handler_synthesized_remaining_gaps" -- kind="subtract" or
    #     kind="retract"; final_move == "explain_remaining_gaps"
    #   - "handler_synthesized_clarification"  -- kind="confirm" or
    #     kind="bootstrap"; final_move == "ask_one_clarifying_question"
    "handler_synthesized_remaining_gaps",
    "handler_synthesized_clarification",
    # AR-1c (adjacent-recommendations design v12): handler-synthesized
    # when AR-6 wires the dispatch.
    #   - "handler_synthesized_adjacent_recommendations" -- the
    #     adjacency engine surfaced up to three different-role
    #     recommendations; final_move == "recommend_adjacent_roles".
    #   - "handler_synthesized_adjacent_description" -- the user
    #     resolved an ordinal reference against last_adjacent_snapshot;
    #     final_move == "describe_adjacent_role".
    # The arbiter NEVER produces these -- the pure synthesis factories
    # live in match/adjacent.py and are exercised by the OutcomeMove
    # reachability invariant in test_chat_arbiter.py.
    "handler_synthesized_adjacent_recommendations",
    "handler_synthesized_adjacent_description",
]


# =========================================================================
# Arbiter-specific reason codes (beyond planner's ReasonCode enum)
# =========================================================================
# These are emitted when the arbiter overrides or resolves, NOT passed
# through from the planner. Documented as module constants so transcript
# tests can assert on specific strings and grep keeps things visible.
ARBITER_REASON_SCOPE_OVERRIDE = "scope_override"
ARBITER_REASON_DUPLICATE_ASK = "duplicate_ask_override"
ARBITER_REASON_FALLBACK = "fallback_to_legacy"
ARBITER_REASON_MATCHES_FOUND = "matches_found"
ARBITER_REASON_MATCHES_WITH_CAPS = "matches_found_with_caps"
# AR-9.feat.coach-tiers CP2 step 2: distinct reason codes for the
# tiered-matches surface so transcript tests can distinguish the
# legacy present_matches path from the three-tier coach path.
ARBITER_REASON_TIERED_MATCHES_FOUND = "tiered_matches_found"
ARBITER_REASON_TIERED_MATCHES_WITH_CAPS = "tiered_matches_found_with_caps"
ARBITER_REASON_NO_MATCHES = "zero_matches_in_dataset"
# Slice N reason code: the engine returned no presentable matches BUT
# at least one low-band candidate qualified as a near-miss (title or
# NOC match against the user's specific target). The responder uses
# this to enter the "role exists, here are the blockers" narration.
ARBITER_REASON_NEAR_MISS = "title_match_with_major_gaps"
# R-3 (remaining-gaps iteration): two distinct reason codes so transcript
# tests + telemetry can distinguish "user added a hypothetical/claim"
# from "user walked one back." Both terminate at the same final_move
# (`explain_remaining_gaps`); see docs/remaining-gaps-design.md §8.
ARBITER_REASON_REMAINING_GAPS = "remaining_credential_gaps_after_assumption"
ARBITER_REASON_REMAINING_GAPS_RETRACTED = "remaining_credential_gaps_after_retraction"
# Synthesized reason codes for the clarification path (kind="confirm"
# or kind="bootstrap"). The responder's clarification renderer picks
# the right template by inspecting `inp.clarification_payload`; these
# reasons are for transcript tests + telemetry.
ARBITER_REASON_CONFIRM_CREDENTIAL = "confirm_credential_completion"
ARBITER_REASON_BOOTSTRAP_MATCH = "bootstrap_match_request"
# AR-1c (adjacent recommendations): two reason codes so transcript
# tests + telemetry can distinguish the two handler-synthesized
# outcomes. The synthesis helpers live in match/adjacent.py and are
# DEAD CODE until AR-6 wires the dispatch.
ARBITER_REASON_ADJACENT_RECOMMENDATIONS = "adjacent_recommendations_user_requested"
ARBITER_REASON_ADJACENT_DESCRIPTION = "adjacent_ordinal_followup"


# Slice C (2026-06-18): readiness reasons that are SAFE to suppress
# the fresh-intake-on-target-change override against. Both come from
# objective resume parsing -- they're not collected for a specific
# target, so they don't go stale on a target change. Chat-collected
# reasons (chat_skills_sufficient, user_explicitly_asked_to_match,
# skills_only_explicit_request) are NOT in this set; the override
# still fires for them because chat evidence CAN be stale-target.
_RESUME_BASED_READINESS_REASONS: frozenset[str] = frozenset({
    "resume_skills_sufficient",
    "resume_work_history_present",
})


# =========================================================================
# Decision shapes
# =========================================================================
@dataclass(frozen=True)
class ArbiterDecision:
    """The terminal output reaching the responder.

    Frozen so downstream consumers can't quietly mutate it.

    Field conventions:
      - final_move: an `OutcomeMove` value
      - reason_code: either a planner ReasonCode (when arbiter_action ==
        passed_planner_through) or one of the ARBITER_REASON_* constants
      - tone: a Tone value -- planner-preserved on most paths
      - caps_applied: surfaces matching-engine caps to the responder
        WITHOUT forcing a tone change (Slice 4 review tightening)
      - notes: human-readable explanation for transcript tests / logs
    """
    final_move: str
    reason_code: str
    tone: str
    arbiter_action: str
    ask_slot: str | None = None
    caps_applied: tuple[str, ...] = ()
    notes: str | None = None


@dataclass(frozen=True)
class RunEngine:
    """Sentinel returned by Pass 1 when proceed_to_match has been
    independently verified by the truth summary. The caller MUST run
    the match engine and call resolve_match_outcome() with the result.

    Carries the planner's reason_code + tone forward so Pass 2 can
    preserve them on present_matches outcomes (Slice 4 review).
    """
    planner_reason_code: str
    planner_tone: str


# =========================================================================
# Pass 1: validate planner intent
# =========================================================================
def validate_planner_intent(
    decision: PlannerDecision | None,
    truth: dict[str, Any],
) -> ArbiterDecision | RunEngine:
    """Pass 1 of the two-pass arbiter. Pure function -- no I/O.

    Rule order (earlier rules win when both could fire):

        1. decision is None
           -> fallback_to_legacy (ArbiterDecision with action set;
              caller substitutes legacy intake_state.decide())

        2. truth.scope_violations_detected non-empty
           -> redirect_scope (overrides ANY planner move including
              proceed_to_match; scope wins precedence)

        3. planner.move == redirect_scope AND
           truth.scope_violations_detected is EMPTY
           -> planner went off-prompt (invented a scope concern).
              Route based on intent: asking_about_gap -> explain_gap;
              otherwise -> ask_one_clarifying_question. The field
              scope_violations_detected is the AUTHORITATIVE source
              for scope; the planner is not allowed to invent one.

        4. planner.move == proceed_to_match AND (
               truth.usable_evidence_present is false
               OR truth.enough_to_match is false
           )
           -> override to ask_one_clarifying_question
              This is the "LLM proposes, backend disposes" rule. We
              re-check usable_evidence_present AND enough_to_match
              independently even though enough_to_match currently
              implies usable_evidence_present -- defense in depth in
              case truth_summary's composition rule ever changes.

        5. planner.move == ask_one_clarifying_question AND
           planner.ask_slot is STRONGLY filled in truth
           -> reroute (drop to proceed if enough_to_match, else next slot)
              Narrowly scoped per Slice 4 review: only target_role_text
              has a usability proxy today; other slots default to
              "weak fill" so the planner's question goes through.

        5. planner.move == proceed_to_match AND checks 3 cleared
           -> RunEngine (caller runs engine, calls Pass 2)

        6. otherwise
           -> pass planner through unchanged

    Returns:
        ArbiterDecision when the move is fully resolved without engine,
        RunEngine when the caller must run the engine + call Pass 2.

    Invariant (tested exhaustively):
        Pass 1 NEVER returns final_move == present_matches or
        present_no_match. Those outcomes can only come from Pass 2.
    """
    # ---- Rule 1: planner returned nothing (LLM disabled, parse fail, etc.) ----
    if decision is None:
        return ArbiterDecision(
            # final_move is a placeholder -- the handler sees
            # arbiter_action == fallback_to_legacy and substitutes the
            # legacy intake_state.decide() result before reaching the
            # responder.
            final_move="ask_one_clarifying_question",
            reason_code=ARBITER_REASON_FALLBACK,
            tone="warm_supportive",
            arbiter_action="fallback_to_legacy",
            ask_slot=None,
            notes=(
                "planner returned None; caller substitutes legacy "
                "intake_state.decide()"
            ),
        )

    # ---- Rule 2: scope override (wins over any planner move) ----
    if truth.get("scope_violations_detected"):
        return ArbiterDecision(
            final_move="redirect_scope",
            reason_code=_scope_reason_code(truth),
            tone="honest_redirect",
            arbiter_action="overrode_to_redirect",
            notes=(
                f"scope_violations_detected={truth.get('scope_violations_detected')}; "
                f"planner emitted {decision.move!r}, overridden"
            ),
        )

    # ---- Rule 3: planner emitted redirect_scope WITHOUT a real scope
    # violation. Haiku has been observed to invent scope concerns on
    # cold-session questions like "how do I get my Class G?" -- the
    # field scope_violations_detected is the AUTHORITATIVE source for
    # scope concerns and was empty here, so the planner went off-prompt.
    # Route based on actual user intent rather than rejecting a
    # legitimate training question.
    if (
        decision.move == "redirect_scope"
        and not truth.get("scope_violations_detected")
    ):
        if truth.get("user_intent_signal") == "asking_about_gap":
            return ArbiterDecision(
                final_move="explain_gap",
                reason_code="credential_gap_present",
                tone=decision.tone or "warm_supportive",
                arbiter_action="overrode_to_explain_gap",
                ask_slot=None,
                notes=(
                    "planner emitted redirect_scope but "
                    "scope_violations_detected is empty; intent is "
                    "asking_about_gap, overriding to explain_gap. "
                    "Training / credentials / licences are in scope."
                ),
            )
        # Intent isn't a gap question -- safest route is a clarifying
        # question rather than rejecting the user with a fabricated
        # redirect.
        reason, slot = _pick_ask_reason_and_slot(truth)
        return ArbiterDecision(
            final_move="ask_one_clarifying_question",
            reason_code=reason,
            tone="warm_supportive",
            arbiter_action="overrode_to_ask",
            ask_slot=slot,
            notes=(
                "planner emitted redirect_scope but "
                "scope_violations_detected is empty AND intent is not "
                "asking_about_gap; defaulting to a clarifying question "
                "rather than rejecting the user on a fabricated scope concern."
            ),
        )

    # ---- Rule 4: proceed_to_match independent re-check ----
    # The most important arbiter rule. Even if the planner says go, we
    # require the truth summary to independently support both signals.
    if decision.move == "proceed_to_match":
        # usable_evidence_present check first -- more specific error msg
        if not truth.get("usable_evidence_present"):
            return ArbiterDecision(
                final_move="ask_one_clarifying_question",
                reason_code="resume_failed_need_chat_skills",
                tone="warm_supportive",
                arbiter_action="overrode_to_ask",
                ask_slot="skills_text",
                notes=(
                    "planner said proceed but truth.usable_evidence_present=false; "
                    "no usable skill evidence to run the engine on"
                ),
            )
        if not truth.get("enough_to_match"):
            reason, slot = _pick_ask_reason_and_slot(truth)
            return ArbiterDecision(
                final_move="ask_one_clarifying_question",
                reason_code=reason,
                tone="warm_supportive",
                arbiter_action="overrode_to_ask",
                ask_slot=slot,
                notes=(
                    "planner said proceed but truth.enough_to_match=false; "
                    f"reason={truth.get('enough_to_match_reason')!r}"
                ),
            )
        # Fresh-intake-on-target-change pillar (2026-06-15): the user
        # may have plenty of skill / experience evidence (enough_to_match
        # = True) but it was collected for a PRIOR target_role_text. The
        # locked design says the engine must not run on stale-target
        # evidence — re-ask for the misaligned slot first.
        #
        # Slice C (2026-06-18): suppress the fresh-intake override when
        # readiness came from RESUME-based evidence. Resume skills and
        # resume work history are OBJECTIVE facts from the CV -- they
        # were not "collected for the prior target," so they're not
        # stale on a target change. Asking the user to re-state skills
        # they already gave us in their resume is what caused the live
        # bug in James's session (Turn 2: target=admin assistant, resume
        # uploaded, system asked for skills_text instead of running
        # engine). Chat-collected skills remain protected by this
        # override -- those CAN be stale-target.
        if not truth.get("target_alignment_ok", True):
            if truth.get(
                "enough_to_match_reason"
            ) in _RESUME_BASED_READINESS_REASONS:
                pass  # fall through to RunEngine below
            else:
                misaligned_slot = truth.get(
                    "target_alignment_first_misaligned_slot"
                ) or "skills_text"
                return ArbiterDecision(
                    final_move="ask_one_clarifying_question",
                    reason_code="target_changed_need_fresh_intake",
                    tone="warm_supportive",
                    arbiter_action="overrode_to_ask",
                    ask_slot=misaligned_slot,
                    notes=(
                        "planner said proceed but truth.target_alignment_ok=false; "
                        f"asking for {misaligned_slot!r} against the new "
                        f"target_role_text={truth.get('target_role_text')!r}"
                    ),
                )
        # Cleared -- the handler runs the engine and calls Pass 2.
        return RunEngine(
            planner_reason_code=decision.reason_code,
            planner_tone=decision.tone,
        )

    # ---- Rule 5: ask-clarifying-question overrides ----
    if decision.move == "ask_one_clarifying_question":
        # Search-first override (2026-06-16 evening, user-signed-off):
        # when the planner asks for ANOTHER clarifying slot but the
        # truth summary has already cleared enough_to_match +
        # usable_evidence_present + target_alignment_ok, drop to
        # RunEngine. The user already has enough evidence to score
        # against; collecting one more preference slot (work_type,
        # shift, education, etc.) before EVER showing matches makes
        # the system feel chatty. The user's model is: search first,
        # iterate on results. The arbiter is the right place to enforce
        # this because it's the deterministic safety net that already
        # overrides the planner in three other rules. Filtering by
        # work_type / shift / etc. is a NARROWING step that happens
        # AFTER first-search, not a prerequisite to it.
        if (
            truth.get("enough_to_match")
            and truth.get("usable_evidence_present")
            and truth.get("target_alignment_ok", True)
        ):
            return RunEngine(
                planner_reason_code="user_explicitly_asked_to_match",
                planner_tone=decision.tone,
            )

        # Skills-first ask reroute (2026-06-16 evening, user-signed-off):
        # when the user has not provided enough chat-side skill claims
        # (enough_to_match=False because the chat_skills_sufficient
        # branch hasn't cleared), and the planner asks for a
        # PREFERENCE slot (work_type, shift, education, etc.) instead
        # of skills, force the ask to `skills_text`. The user's
        # product rule: never ask a preference question before we
        # have real skills to score against.
        #
        # Trigger: planner asked a preference slot AND truth says
        # we're not ready to match. Override to skills_text.
        # The planner LLM has been seen to leap to work_type after
        # one fallback-filled skills_text turn (where the fallback
        # filled the slot with non-skills experience prose, leaving
        # chat_skill_count low). The arbiter is the right place to
        # enforce skills-first because it's the deterministic safety
        # net that already overrides the planner in three other rules.
        _PREFERENCE_SLOTS = {
            "work_type_preference", "shift_preference",
            "education_text", "preferred_location",
            "availability_text", "salary_expectation_text",
            "transportation_text", "language_preferences",
        }
        if (
            decision.ask_slot in _PREFERENCE_SLOTS
            and not truth.get("enough_to_match")
        ):
            return ArbiterDecision(
                final_move="ask_one_clarifying_question",
                reason_code=ARBITER_REASON_DUPLICATE_ASK,
                tone=decision.tone,
                arbiter_action="overrode_to_ask",
                ask_slot="skills_text",
                notes=(
                    f"planner asked preference slot {decision.ask_slot!r} "
                    f"before enough_to_match cleared; rerouted to "
                    f"'skills_text' (skills-first rule, 2026-06-16)"
                ),
            )

        # Truth doesn't support engine-run yet. Handle the duplicate-ask
        # reroute case: planner asked for a slot that's already filled
        # → reroute to the next unfilled intake-priority slot.
        slot = decision.ask_slot
        if slot is not None and _is_slot_strongly_filled(slot, truth):
            next_slot = _next_unfilled_priority_slot(truth)
            if next_slot:
                return ArbiterDecision(
                    final_move="ask_one_clarifying_question",
                    reason_code=ARBITER_REASON_DUPLICATE_ASK,
                    tone=decision.tone,
                    arbiter_action="overrode_to_ask",
                    ask_slot=next_slot,
                    notes=(
                        f"planner asked {slot!r} but it's strongly filled; "
                        f"rerouted to {next_slot!r}"
                    ),
                )
            # Nothing else to ask AND not enough_to_match -- legacy fallback.
            return ArbiterDecision(
                final_move="ask_one_clarifying_question",
                reason_code=ARBITER_REASON_FALLBACK,
                tone="warm_supportive",
                arbiter_action="fallback_to_legacy",
                ask_slot=None,
                notes=(
                    "duplicate-ask hit, no unfilled priority slot, "
                    "not enough_to_match -- legacy intake_state.decide() will run"
                ),
            )

    # ---- Rule 6: pass planner through unchanged ----
    return ArbiterDecision(
        final_move=decision.move,
        reason_code=decision.reason_code,
        tone=decision.tone,
        arbiter_action="passed_planner_through",
        ask_slot=decision.ask_slot,
    )


# =========================================================================
# Pass 2: resolve match outcome
# =========================================================================
def resolve_match_outcome(
    *,
    match_count: int,
    caps_applied: tuple[str, ...] = (),
    near_miss_candidates: Sequence[Any] = (),
    planner_reason_code: str = "user_explicitly_asked_to_match",
    planner_tone: str = "brief_confident",
    tiered_evidence_available: bool = False,
) -> ArbiterDecision:
    """Pass 2 of the two-pass arbiter. Pure function.

    Takes engine RESULTS (not the engine itself) and resolves to a
    terminal outcome. The handler runs the engine and feeds the count
    + caps + near-miss candidates in.

    Tone preservation rules (Slice 4 review tightening):

      - match_count > 0: preserve planner_tone. If caps are present,
        they surface as a separate `caps_applied` field on the
        decision; the responder narrates the cap honestly within the
        planner's chosen tone (warm_supportive, brief_confident, etc.)
        rather than being force-flattened to honest_redirect.

      - match_count == 0 AND near_miss_candidates non-empty: force
        tone to warm_supportive (Slice N lock). The responder narrates
        skill-gap analysis -- honest about the gap, optimistic about
        the path. NOT honest_redirect, which reads as "I can't help."

      - match_count == 0 AND no near_miss_candidates: force tone to
        honest_redirect. No-match is a genuinely hard moment to land;
        brief_confident reads as flippant when the user was expecting
        results.

    The reason_code on the resulting decision reflects what HAPPENED
    (matches_found / matches_with_caps / near_miss / zero_matches),
    not the planner's pre-engine guess. Transcript tests assert on
    these arbiter-coined codes.

    `near_miss_candidates` is typed as `Sequence[Any]` rather than
    `list[MatchResult]` to keep the arbiter from importing the match
    module. The arbiter only needs truthiness here; the responder
    (which DOES know MatchResult) consumes the actual list via the
    handler.

    AR-9.feat.coach-tiers CP2 step 2 — tiered_evidence_available:
      The handler is the SOLE authority for whether the three-tier
      coach surface fires. When the handler has built a TieredEvidence
      with at least one populated tier, it passes
      `tiered_evidence_available=True`; this routes a positive
      match_count to `present_tiered_matches` (preserving planner_tone
      and caps_applied). The arbiter does NOT infer tier availability
      from match_count or near-miss signals — and notably NOT from
      `len(strong)`-style heuristics, since the arbiter has no view
      of strong/stretch/adjacent buckets. When this flag is False
      (default), the function's pre-CP2 behaviour is preserved
      byte-stable.

      `tiered_evidence_available=True` is honored on both positive
      and zero match_count, so a Sideways-only surface is reachable
      when the handler's proactive adjacency pass yields records but
      the engine returned no Strong/Stretch. `present_near_miss`
      precedence is preserved (a near-miss is the user's current
      focus; tier evidence does not override it).

      CP2 step 6.1 (2026-06-14): the prior block that re-checked
      `tiered_evidence_available` below `match_count == 0` is
      removed — one branch is sufficient and the second was
      unreachable after the reorder.
    """
    if match_count == 0 and near_miss_candidates:
        # Slice N (2026-06-05): the role exists locally but candidate
        # is far from ready. Distinct from present_no_match so the
        # responder enters skill-gap-analysis narration instead of
        # the SCCC-referral no-match shape. Near-miss precedence is
        # intact in CP2 step 6.1: tier evidence does not override a
        # near-miss because the near-miss is the user's stated focus.
        return ArbiterDecision(
            final_move="present_near_miss",
            reason_code=ARBITER_REASON_NEAR_MISS,
            tone="warm_supportive",
            arbiter_action="resolved_to_near_miss",
        )
    # AR-9.feat.coach-tiers CP2 step 2 + step 6.1: tiered-matches
    # dispatch. The handler is the SOLE authority for whether the
    # three-tier surface fires. When it supplies
    # `tiered_evidence_available=True`, route to
    # present_tiered_matches — independent of `match_count`. This
    # makes the Sideways-only surface reachable (match_count == 0
    # with a populated Adjacent tier). caps_applied flows through
    # identically so the responder narrates caps within the
    # three-tier shape.
    if tiered_evidence_available:
        if caps_applied:
            return ArbiterDecision(
                final_move="present_tiered_matches",
                reason_code=ARBITER_REASON_TIERED_MATCHES_WITH_CAPS,
                tone=planner_tone,
                arbiter_action="resolved_to_tiered_matches",
                caps_applied=tuple(caps_applied),
            )
        return ArbiterDecision(
            final_move="present_tiered_matches",
            reason_code=ARBITER_REASON_TIERED_MATCHES_FOUND,
            tone=planner_tone,
            arbiter_action="resolved_to_tiered_matches",
        )
    if match_count == 0:
        return ArbiterDecision(
            final_move="present_no_match",
            reason_code=ARBITER_REASON_NO_MATCHES,
            tone="honest_redirect",
            arbiter_action="resolved_to_no_match",
        )
    if caps_applied:
        return ArbiterDecision(
            final_move="present_matches",
            reason_code=ARBITER_REASON_MATCHES_WITH_CAPS,
            tone=planner_tone,  # preserved -- caps narrated via caps_applied field
            arbiter_action="resolved_to_matches",
            caps_applied=tuple(caps_applied),
        )
    return ArbiterDecision(
        final_move="present_matches",
        reason_code=ARBITER_REASON_MATCHES_FOUND,
        tone=planner_tone,  # preserved
        arbiter_action="resolved_to_matches",
    )


# =========================================================================
# Internal helpers (pure)
# =========================================================================
def _scope_reason_code(truth: dict[str, Any]) -> str:
    """Map detected scope-violation tags to a planner-flavored reason
    code. Defaults to ARBITER_REASON_SCOPE_OVERRIDE when no specific
    code matches -- so a future violation tag we don't recognize still
    produces a coherent override rather than crashing."""
    violations = truth.get("scope_violations_detected") or []
    if not violations:
        return ARBITER_REASON_SCOPE_OVERRIDE
    first = str(violations[0]).lower()
    # Order matters: check longer/more-specific tags first.
    if "immigration" in first or "pr " in first or "express" in first:
        return "scope_violation_immigration"
    if "wage" in first or "salary" in first or "statcan" in first or "national" in first:
        return "scope_violation_wages"
    if "non_ssm" in first or "outside" in first or "toronto" in first:
        return "scope_violation_non_ssm"
    if "off_topic" in first or "off-topic" in first:
        return "scope_violation_off_topic"
    return ARBITER_REASON_SCOPE_OVERRIDE


def _pick_ask_reason_and_slot(truth: dict[str, Any]) -> tuple[str, str]:
    """When pass 1 overrides proceed_to_match to ask, pick the right
    (reason_code, ask_slot) pair from the truth signals.

    Mirrors the planner's reason taxonomy so the responder sees a
    familiar reason code regardless of which layer chose the ask.
    """
    if (
        not truth.get("target_role_text")
        or truth.get("target_role_specificity") != "specific"
    ):
        return ("target_role_unclear", "target_role_text")
    if (
        truth.get("resume_parse_quality") == "failed"
        or truth.get("enough_to_match_reason") == "no_usable_evidence"
    ):
        return ("resume_failed_need_chat_skills", "skills_text")
    return ("insufficient_profile_evidence", "skills_text")


# Canonical fallback order when we need to pick the next slot to ask.
# Mirrors the existing intake_priority "other" / generic priority.
# Slice 6 may swap this for intake_priority.pick_slots_to_ask().
_CANONICAL_SLOT_ORDER = (
    "target_role_text",
    "skills_text",
    "experience_text",
    "work_type_preference",
    "shift_preference",
    "education_text",
)


def _next_unfilled_priority_slot(truth: dict[str, Any]) -> str | None:
    """Pick the next intake-priority slot to ask about. Returns the
    first canonical slot not in `filled_slots`, or None if everything
    is filled (rare -- usually means we should be matching by now)."""
    filled = set(truth.get("filled_slots") or [])
    for slot in _CANONICAL_SLOT_ORDER:
        if slot not in filled:
            return slot
    return None


def _is_slot_strongly_filled(slot: str, truth: dict[str, Any]) -> bool:
    """Slice 4 review tightening: a slot can be technically filled but
    weak ("same role", "any job"). Don't reroute away from the
    planner's question unless we have STRONG evidence the slot has
    usable content.

    Only `target_role_text` has a usability proxy in truth_summary
    today, via `target_role_specificity == "specific"`. Other slots
    default to False (treat as weak): the cost of asking again is
    lower than the cost of silently skipping a needed question.

    Add per-slot usability checks here as truth_summary surfaces them.
    """
    if slot not in (truth.get("filled_slots") or []):
        return False
    if slot == "target_role_text":
        return truth.get("target_role_specificity") == "specific"
    # Conservative default for slots without a usability proxy.
    return False
