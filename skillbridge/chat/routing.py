"""Deterministic message router for chat orchestration v2.1.

See docs/message-understanding-design.md for the full spec.

The router takes a `MessageUnderstanding` (from `understand_message`)
plus the `TruthSummary` (already computed by the handler) and decides
whether to emit a `PlannerDecision` deterministically -- bypassing the
LLM planner -- or hand off to the planner as before.

Return contract:
  * `PlannerDecision`  -> router decided; handler MUST skip the planner
                          LLM call and pass this decision straight into
                          arbiter pass 1.
  * `None`             -> router declined; handler MUST call the planner
                          as today. Reason logged so transcript tests
                          can see why.

The router returns a `PlannerDecision` (not a separate `RouterDecision`
type) because that IS the contract the downstream arbiter and responder
already consume. Introducing a parallel type would force every consumer
to learn a second shape for no behavioral benefit. The function name
communicates the source of the decision; the type matches the pipeline.

Architectural promise (locked in the design doc, 2026-06-04):

    The planner LLM MUST NOT run for high-confidence scope_violation or
    training-with-entity cases. This module enforces that contract.

Tests in `test_chat_routing.py` assert it by mocking `plan_next_move`
to fail loudly if the handler ever calls it on a HIGH-confidence case.

Priority order (exact mirror of the design doc, rules 1-7):

  1. scope_violation                        (HIGH)  -> redirect_scope
  2. training_request + registry_gap        (HIGH)  -> explain_gap
  3. training_request without registry_gap  (HIGH)  -> ask_one_clarifying_question
  4. job_search + truth_ready               (HIGH)  -> proceed_to_match
  5. registry_gap entity without training   (MEDIUM) -> None (planner)
  6. job_search + truth NOT ready           (MEDIUM) -> None (planner)
  7. ambiguous / low signal                 (LOW)   -> None (planner)

Rules 5-7 currently return `None` (planner consulted as today). The
"understanding as hint" pass-through is reserved for a future slice;
the contract for now is "skip planner on HIGH, otherwise no change."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from skillbridge.chat.message_understanding import MessageUnderstanding
from skillbridge.chat.planner import PlannerDecision


# ---------------------------------------------------------------------------
# Scope category -> ReasonCode mapping
# ---------------------------------------------------------------------------
# `understand_message` tags scope entities with one of these canonical
# names (see `_detect_scope_entities`). The planner ReasonCode enum has
# matching scope codes; this dict is the one place the mapping lives.
# If a new scope category is added to understanding, this dict is the
# breakage point -- a missing key triggers a KeyError loudly rather
# than a silent default. That's intentional.
_SCOPE_REASON_BY_CATEGORY: dict[str, str] = {
    "immigration":   "scope_violation_immigration",
    "national_wages": "scope_violation_wages",
    "non_ssm_city":  "scope_violation_non_ssm",
}


# ---------------------------------------------------------------------------
# Router decision metadata
# ---------------------------------------------------------------------------
# This is what we hand back so the handler can log _which_ rule fired
# without having to re-derive it from the PlannerDecision. The decision
# itself goes straight to the arbiter; this is operational metadata.
@dataclass(frozen=True)
class RouterTrace:
    """Why the router decided what it decided. Carried alongside the
    PlannerDecision for logs and transcript tests, NOT for runtime
    behavior. The arbiter ignores this.

    `entity_canonical_names` is included so live-debug logs can show
    WHICH entity the router saw without re-deriving from the
    MessageUnderstanding. Added per Slice B review feedback: when a
    live test failed it was impossible to tell from the log line
    whether the router skipped the planner because Excel was detected
    or because Class G was -- the entity names answer that directly.
    """
    rule_fired: str          # e.g. "rule_1_scope_violation", "rule_7_default_planner"
    planner_skipped: bool
    understanding_intent: str
    understanding_confidence: str
    reason: str              # human-readable, from MessageUnderstanding.reason
    entity_canonical_names: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def route_from_understanding(
    understanding: MessageUnderstanding,
    truth: dict[str, Any],
) -> tuple[PlannerDecision | None, RouterTrace]:
    """Apply the 7 priority rules to the message understanding.

    Inputs:
      - understanding: result of `understand_message` for this turn.
      - truth: the truth summary as a planner-JSON dict (the same shape
        `plan_next_move` would receive). Used by Rule 4 to gate
        `proceed_to_match` on `enough_to_match` AND `usable_evidence_present`.

    Returns a `(decision, trace)` tuple:
      - decision: PlannerDecision the handler should hand to the arbiter
        (planner LLM is SKIPPED), OR None meaning the handler MUST call
        the planner as today.
      - trace: RouterTrace for logging. Always returned, even when
        decision is None, so transcript tests can assert which rule fired.

    Pure function. No I/O, no LLM, no truth-summary mutation. Safe to
    call from sync or async contexts.
    """
    intent = understanding.primary_intent
    confidence = understanding.confidence
    # Captured once and stamped onto every RouterTrace below. Used by
    # the handler log line so live debugging can see exactly which
    # entities the router saw (e.g. "rule_2 with Microsoft Excel" vs
    # "rule_2 with Class G licence") without re-deriving from the
    # MessageUnderstanding.
    entity_names: tuple[str, ...] = tuple(
        e.canonical_name for e in understanding.entities
    )

    # =======================================================================
    # Rule 1 (HIGH): scope_violation -> redirect_scope, planner SKIPPED
    # =======================================================================
    # Scope categories are exclusive in `_detect_scope_entities` (one per
    # category found, first match wins). We pick the FIRST scope_keyword
    # entity to derive the reason_code; multi-category messages get the
    # highest-priority hit (immigration > national_wages > non_ssm_city
    # by detection order).
    if intent == "scope_violation" and confidence == "high":
        scope_entity = next(
            (e for e in understanding.entities if e.type == "scope_keyword"),
            None,
        )
        if scope_entity is None:
            # Defensive: scope_violation classification without a
            # scope_keyword entity is a bug in understand_message itself.
            # Don't try to repair -- fall through to the planner, log
            # the inconsistency, and let the existing arbiter scope
            # override (Rule 2 today) catch the truth-detected case.
            return None, RouterTrace(
                rule_fired="rule_1_skipped_no_scope_entity",
                planner_skipped=False,
                understanding_intent=intent,
                understanding_confidence=confidence,
                reason=(
                    "scope_violation classified but no scope_keyword "
                    "entity attached; falling through to planner"
                ),
                entity_canonical_names=entity_names,
            )
        reason_code = _SCOPE_REASON_BY_CATEGORY[scope_entity.canonical_name]
        decision = PlannerDecision(
            move="redirect_scope",
            reason_code=reason_code,  # type: ignore[arg-type]
            ask_slot=None,
            tone="honest_redirect",
        )
        return decision, RouterTrace(
            rule_fired="rule_1_scope_violation",
            planner_skipped=True,
            understanding_intent=intent,
            understanding_confidence=confidence,
            reason=understanding.reason,
            entity_canonical_names=entity_names,
        )

    # =======================================================================
    # Rule 2 (HIGH): training_request + registry_gap -> explain_gap
    # =======================================================================
    # The locked architectural promise: planner does NOT run on this case.
    # Multi-entity: design decision is "first/strongest only"; the
    # responder narrates one and offers the next. Carrying all entities
    # into the planner decision isn't needed -- the recommender resolves
    # gaps from `discovered_registry_gaps` via the handler's existing
    # `_registry_training_for_gap` call.
    if (
        intent == "training_request"
        and confidence == "high"
        and understanding.has_entity_type("registry_gap")
    ):
        decision = PlannerDecision(
            move="explain_gap",
            reason_code="credential_gap_present",
            ask_slot=None,
            tone="warm_supportive",
        )
        return decision, RouterTrace(
            rule_fired="rule_2_training_with_entity",
            planner_skipped=True,
            understanding_intent=intent,
            understanding_confidence=confidence,
            reason=understanding.reason,
            entity_canonical_names=entity_names,
        )

    # =======================================================================
    # Rule 3 (HIGH): training_request without registry_gap
    #   -> ask_one_clarifying_question(skills_text)
    # =======================================================================
    # User wants training but didn't name a specific credential. The
    # responder owns the NEW phrasing locked in design decision #4:
    #
    #   "Sure -- what skill or certificate do you want training for?
    #    For example Excel, WHMIS, forklift, Class G, or 310T."
    #
    # Round-25 Rule-3 wording slice: emit the DISTINCT reason code
    # `training_request_no_entity` so the responder can branch on it
    # deterministically. Previously this path emitted
    # `insufficient_profile_evidence`, which made the responder fall
    # into the regular skills-intake question ("what software have you
    # worked with?") -- wrong frame for someone who asked about
    # training. The new reason code is in `ReasonCode` and
    # `_VALID_REASON_BY_MOVE["ask_one_clarifying_question"]`.
    if (
        intent == "training_request"
        and confidence == "high"
        and not understanding.has_entity_type("registry_gap")
    ):
        decision = PlannerDecision(
            move="ask_one_clarifying_question",
            reason_code="training_request_no_entity",
            ask_slot="skills_text",
            tone="warm_supportive",
        )
        return decision, RouterTrace(
            rule_fired="rule_3_training_no_entity",
            planner_skipped=True,
            understanding_intent=intent,
            understanding_confidence=confidence,
            reason=understanding.reason,
            entity_canonical_names=entity_names,
        )

    # =======================================================================
    # Rule 4 (HIGH): job_search + truth ready -> proceed_to_match
    # =======================================================================
    # Two truth-summary conditions MUST both hold before the router
    # commits to proceed_to_match:
    #   * enough_to_match: the truth summary has a target role + minimum
    #     evidence (resume parsed OR sufficient chat skills).
    #   * usable_evidence_present: at least one resume_chunk or
    #     conversational skill with sufficient quality to seed the engine.
    # If either is missing, fall through to Rule 6 (planner with the
    # existing ask logic).
    #
    # Defense-in-depth: even if the router emits proceed_to_match here,
    # arbiter Rule 4 ("proceed_to_match + truth not ready -> ask")
    # acts as a final safety net. The router's job is to AVOID needing
    # the safety net, not to remove it.
    if intent == "job_search" and confidence == "high":
        ready = bool(truth.get("enough_to_match")) and bool(
            truth.get("usable_evidence_present"),
        )
        if ready:
            decision = PlannerDecision(
                move="proceed_to_match",
                reason_code="user_explicitly_asked_to_match",
                ask_slot=None,
                tone="brief_confident",
            )
            return decision, RouterTrace(
                rule_fired="rule_4_job_search_truth_ready",
                planner_skipped=True,
                understanding_intent=intent,
                understanding_confidence=confidence,
                reason=understanding.reason,
                entity_canonical_names=entity_names,
            )
        # ---- Rule 6 (MEDIUM): job_search but truth not ready ----
        return None, RouterTrace(
            rule_fired="rule_6_job_search_truth_not_ready",
            planner_skipped=False,
            understanding_intent=intent,
            understanding_confidence="medium",
            reason=(
                f"{understanding.reason}; truth not ready "
                f"(enough_to_match={truth.get('enough_to_match')!r}, "
                f"usable_evidence_present={truth.get('usable_evidence_present')!r})"
            ),
            entity_canonical_names=entity_names,
        )

    # =======================================================================
    # Rule 5 (MEDIUM): registry_gap entity without training intent
    # =======================================================================
    # Detect this via the MEDIUM signal `understand_message` emits when
    # an entity is present but no training-action word fires (e.g. "I
    # have Excel, find me jobs"). Hand to the planner -- it knows the
    # last_assistant_move and can decide whether to treat this as a
    # skill claim or a soft training mention.
    if confidence == "medium" and understanding.has_entity_type("registry_gap"):
        return None, RouterTrace(
            rule_fired="rule_5_entity_without_training_intent",
            planner_skipped=False,
            understanding_intent=intent,
            understanding_confidence=confidence,
            reason=understanding.reason,
            entity_canonical_names=entity_names,
        )

    # =======================================================================
    # Rule 7 (LOW / default): planner handles
    # =======================================================================
    # Everything else -- conversational signals, ambiguous reference,
    # multi-turn context. The planner has the conversation memory and
    # last_assistant_move that the router doesn't. Nothing changes from
    # today's behavior.
    return None, RouterTrace(
        rule_fired="rule_7_default_planner",
        planner_skipped=False,
        understanding_intent=intent,
        understanding_confidence=confidence,
        reason=understanding.reason or "no router rule applied; planner handles",
        entity_canonical_names=entity_names,
    )
