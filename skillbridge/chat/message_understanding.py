"""Deterministic message understanding for chat orchestration v2.1.

See docs/message-understanding-design.md for the full spec. Short version:

  user_message + staged + registry
       │
       ▼
  understand_message(...) -> MessageUnderstanding
       │
       ▼
  router (next slice) decides whether to emit a decision deterministically
  or hand off to the planner.

This module is DEAD CODE until Slice B wires it into the handler. Built
in isolation so its tests prove correctness before integration. The
`MESSAGE_UNDERSTANDING_ENABLED` flag (added in Slice B) is the rollout
toggle.

Architectural promise: the planner does NOT run for high-confidence
scope-violation or training-with-entity cases. The router (Slice B)
enforces this; this module surfaces the signals it needs.

What this module is responsible for:
  - Classifying the user's current message into a `PrimaryIntent`
  - Surfacing detected entities (scope keywords, registry training gaps)
  - Assigning a confidence level (high/medium/low)
  - Providing a human-readable `reason` for logs and transcript tests

What this module is NOT responsible for:
  - Profile/match-readiness state (that's TruthSummary)
  - Picking a chat move (that's the router + arbiter)
  - Picking a tone (that's the planner / responder)
  - Multi-turn conversational memory beyond the current message
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from skillbridge.chat.truth_summary import (
    _ASKING_ABOUT_GAP_PATTERNS,
    _CONFIRMING_PATTERNS,
    _CORRECTING_PATTERNS,
    _DECLINING_PATTERNS,
    _IMPATIENT_PATTERNS,
    _REDIRECTING_PATTERNS,
    _SCOPE_IMMIGRATION_PATTERNS,
    _SCOPE_NATIONAL_WAGES_PATTERNS,
    _SCOPE_NON_LOCAL_CITY_PATTERNS,
    _TRAINING_ACTION_WORDS,
    _has_training_action_words,
    _NON_LOCAL_CITY_RE,
)


# ============================================================================
# Closed enums (Literal types -- single source of truth)
# ============================================================================
PrimaryIntent = Literal[
    "scope_violation",
    "training_request",
    "gap_explanation",
    "job_search",
    "confirmation",
    "decline",
    "correction",
    "ambiguous",
]

Confidence = Literal["high", "medium", "low"]

EntityType = Literal[
    "registry_gap",
    "scope_keyword",
]


# ============================================================================
# Data shapes
# ============================================================================
@dataclass(frozen=True)
class DetectedEntity:
    """One concrete thing the user named in their message.

    `matched_text` is the actual substring the regex/alias matched against;
    useful for logs + transcript tests so we can see exactly what fired.
    `source` describes which classifier produced this entity, for
    debugging cross-classifier conflicts.
    """
    type: str                # EntityType value
    canonical_name: str
    matched_text: str
    source: str


@dataclass(frozen=True)
class MessageUnderstanding:
    """Structured output of `understand_message`. Read-only contract
    consumed by the router in Slice B.

    Field conventions:
      - primary_intent: one of PrimaryIntent. "ambiguous" means "router
        should NOT decide; let the planner handle it."
      - confidence: high/medium/low. Only `high` should cause the router
        to skip the planner. Medium passes the understanding as a hint;
        low falls through entirely to the planner.
      - entities: tuple in first-seen order, deduped by canonical_name.
      - reason: human-readable explanation. Examples:
          'matched _IMPATIENT_PATTERNS regex'
          'scope keyword "PR" present'
          'registry entity "Microsoft Excel" + training action word "course"'
    """
    primary_intent: str             # PrimaryIntent value
    confidence: str                 # Confidence value
    entities: tuple[DetectedEntity, ...] = ()
    reason: str = ""

    def has_entity_type(self, entity_type: str) -> bool:
        """True when any detected entity has the given EntityType."""
        return any(e.type == entity_type for e in self.entities)

    def registry_gap_canonical_names(self) -> tuple[str, ...]:
        """Convenience: canonical names of all detected registry_gap
        entities, in first-seen order."""
        return tuple(
            e.canonical_name for e in self.entities if e.type == "registry_gap"
        )


# ============================================================================
# Lower-level signal helpers
# ============================================================================
def _detect_scope_entities(message: str) -> list[DetectedEntity]:
    """Scan the message for scope-violation keywords. Returns one
    DetectedEntity per scope category that fires, deduped. Order:
    immigration > national_wages > non_ssm_city (priority within
    scope), but in practice each category is independent."""
    if not message:
        return []
    lowered = message.lower()
    found: list[DetectedEntity] = []
    seen: set[str] = set()

    for category, patterns in (
        ("immigration", _SCOPE_IMMIGRATION_PATTERNS),
        ("national_wages", _SCOPE_NATIONAL_WAGES_PATTERNS),
        ("non_ssm_city", _SCOPE_NON_LOCAL_CITY_PATTERNS),
    ):
        if category in seen:
            continue
        for pat in patterns:
            m = re.search(pat, lowered)
            if m:
                seen.add(category)
                found.append(DetectedEntity(
                    type="scope_keyword",
                    canonical_name=category,
                    matched_text=m.group(0),
                    source=f"scope_pattern:{category}",
                ))
                break
    return found


def _matches_any_pattern(message: str, patterns: tuple[str, ...]) -> str | None:
    """Return the first matching pattern (raw regex string) or None.
    Used so `reason` strings can name which pattern fired."""
    if not message:
        return None
    lowered = message.lower()
    for pat in patterns:
        if re.search(pat, lowered):
            return pat
    return None


# ============================================================================
# Public entry point
# ============================================================================
def understand_message(
    *,
    user_message: str,
    registry_gaps_in_message: list[str] | None = None,
) -> MessageUnderstanding:
    """Classify the user's current message into a structured
    understanding the router can route on.

    Inputs:
      - user_message: the user's text this turn. Empty / None handled.
      - registry_gaps_in_message: canonical gap names already discovered
        by `registry.find_gaps_in_message(user_message)`. Passed in
        rather than re-scanned here so the handler's existing scan
        (used for both this module and the training recommender) is
        the single source of truth.

    Returns a frozen MessageUnderstanding. Empty / None input produces
    `primary_intent="ambiguous", confidence="low"` (NOT an error -- "no
    message" is a valid no-signal state).

    Raises TypeError when `registry_gaps_in_message` contains any
    non-string value (e.g. a `Gap` dataclass instance leaked through
    from `registry.find_gaps_in_message`). This is intentional: silent
    coercion would let integration bugs hide. Callers must normalize:
        [g.canonical_name for g in registry.find_gaps_in_message(msg)]

    Priority order (mirrors the design doc's locked router priorities):
      1. scope_violation        (HIGH)
      2. training_request + registry_gap (HIGH)
      3. training_request without entity (HIGH)
      4. job_search via impatient_proceed (HIGH)  -- truth-readiness
         not checked here; the router pairs this with TruthSummary.
      5. confirmation / decline / correction (MEDIUM each)
      6. registry_gap without training intent (MEDIUM)
      7. asking_question / neutral (LOW)
    """
    msg = (user_message or "").strip()
    if not msg:
        return MessageUnderstanding(
            primary_intent="ambiguous",
            confidence="low",
            entities=(),
            reason="empty message",
        )

    # ---- 1. Scope (highest priority) ----
    scope_entities = _detect_scope_entities(msg)
    if scope_entities:
        return MessageUnderstanding(
            primary_intent="scope_violation",
            confidence="high",
            entities=tuple(scope_entities),
            reason=(
                f"scope keyword detected: "
                f"{scope_entities[0].canonical_name}/{scope_entities[0].matched_text!r}"
            ),
        )

    # Build entity list for registry gaps once -- referenced by multiple
    # rules below. The contract is strict: callers must pass canonical
    # name strings, not Gap dataclass instances. `registry.find_gaps_in_message`
    # returns `list[Gap]`; the handler/router layer is responsible for the
    # `[g.canonical_name for g in gaps]` conversion. Silently coercing here
    # would hide integration bugs (the type mismatch found in Slice A review),
    # so non-strings raise TypeError.
    registry_entities: list[DetectedEntity] = []
    seen_canonicals: set[str] = set()
    for canonical in (registry_gaps_in_message or []):
        if not isinstance(canonical, str):
            raise TypeError(
                "registry_gaps_in_message must contain str canonical names, "
                f"got {type(canonical).__name__}. Convert Gap objects with "
                "[g.canonical_name for g in registry.find_gaps_in_message(msg)] "
                "in the caller."
            )
        name = canonical.strip()
        if not name or name in seen_canonicals:
            continue
        seen_canonicals.add(name)
        registry_entities.append(DetectedEntity(
            type="registry_gap",
            canonical_name=name,
            matched_text=name,
            source="registry_alias",
        ))

    # ---- 2. Training request with registry entity (HIGH) ----
    asking_gap_pattern = _matches_any_pattern(msg, _ASKING_ABOUT_GAP_PATTERNS)
    has_training_action = _has_training_action_words(msg)
    has_registry_entity = bool(registry_entities)

    # Two paths that count as a training request:
    #   (a) explicit asking-about-gap regex match
    #   (b) entity + training-action word together (covers
    #       "online Excel course" which the regex layer can miss)
    is_training_request = bool(asking_gap_pattern) or (
        has_registry_entity and has_training_action
    )

    if is_training_request and has_registry_entity:
        return MessageUnderstanding(
            primary_intent="training_request",
            confidence="high",
            entities=tuple(registry_entities),
            reason=(
                "training intent ("
                + (
                    f"asking_gap_pattern={asking_gap_pattern!r}"
                    if asking_gap_pattern
                    else "registry_entity + training_action_word"
                )
                + f") + registry_entity={registry_entities[0].canonical_name!r}"
            ),
        )

    # ---- 3. Training request without entity (HIGH) ----
    # User asked about training but didn't name a specific credential.
    # Router will emit ask_one_clarifying_question with the NEW
    # "what skill or certificate?" phrasing.
    if is_training_request and not has_registry_entity:
        return MessageUnderstanding(
            primary_intent="training_request",
            confidence="high",
            entities=(),
            reason=(
                "training intent without specific entity "
                f"(asking_gap_pattern={asking_gap_pattern!r}, "
                f"training_action_words_present={has_training_action})"
            ),
        )

    # ---- 4. Job search via impatient_proceed (HIGH at this layer;
    #         truth-readiness gate happens in the router) ----
    impatient_pattern = _matches_any_pattern(msg, _IMPATIENT_PATTERNS)
    if impatient_pattern:
        return MessageUnderstanding(
            primary_intent="job_search",
            confidence="high",
            entities=tuple(registry_entities),  # may be empty; informational
            reason=f"impatient_proceed pattern matched: {impatient_pattern!r}",
        )

    # ---- 5. Conversational signals (declining/correcting/confirming) ----
    # These are MEDIUM at most -- multi-turn context matters and the
    # planner sees last_assistant_move.
    if _matches_any_pattern(msg, _DECLINING_PATTERNS):
        return MessageUnderstanding(
            primary_intent="decline",
            confidence="medium",
            entities=tuple(registry_entities),
            reason="declining pattern matched",
        )
    if _matches_any_pattern(msg, _CORRECTING_PATTERNS):
        return MessageUnderstanding(
            primary_intent="correction",
            confidence="medium",
            entities=tuple(registry_entities),
            reason="correcting pattern matched",
        )
    # redirecting is treated as ambiguous at MEDIUM -- planner knows context
    if _matches_any_pattern(msg, _REDIRECTING_PATTERNS):
        return MessageUnderstanding(
            primary_intent="ambiguous",
            confidence="medium",
            entities=tuple(registry_entities),
            reason="redirecting pattern matched; planner consulted",
        )
    if _matches_any_pattern(msg, _CONFIRMING_PATTERNS):
        return MessageUnderstanding(
            primary_intent="confirmation",
            confidence="medium",
            entities=tuple(registry_entities),
            reason="confirming pattern matched",
        )

    # ---- 6. Registry entity present without training intent (MEDIUM) ----
    # Could be skill claim ("I have Excel") or soft training mention.
    # Planner consulted with entity as context.
    if has_registry_entity:
        return MessageUnderstanding(
            primary_intent="ambiguous",
            confidence="medium",
            entities=tuple(registry_entities),
            reason=(
                f"registry entity {registry_entities[0].canonical_name!r} "
                f"present without training intent; planner consulted "
                f"(could be skill claim or implicit training mention)"
            ),
        )

    # ---- 7. Default: ambiguous, low confidence ----
    return MessageUnderstanding(
        primary_intent="ambiguous",
        confidence="low",
        entities=(),
        reason="no strong classification signal; planner handles",
    )
