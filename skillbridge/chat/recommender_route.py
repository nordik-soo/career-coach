"""Recommender route decision module (locked 2026-06-22).

Pure-function router: given the hard pattern signal, the career
intent (from the LLM classifier), and substrate state, decide which
engine path should fire this turn.

This module:
  - imports no engines
  - calls no LLM
  - does no DB access
  - mutates no state
  - touches no responder / prompts

Its single output is a frozen `RecommenderRouteVerdict` that the
handler will act on in step 3 (separate work item). Until step 3
wires it into `handle_anonymous`, this module is inert in production.

See [[project-recommender-peer-engine-locked]] for the design lock
and [[project-recommender-step2-router-spec]] (this work item) for
the verdict-shape rationale.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from skillbridge.chat.gap_evidence import RecommenderMode
from skillbridge.chat.recommender_intent import CareerIntent
from skillbridge.chat.truth_summary import UserIntentSignal


# Substrate threshold -- locked at N=5 per memo section 11 item 1
# (matches the existing `_should_offer_resume_upload` chat_skill_count
# gate).
_MIN_CHAT_SKILLS_FOR_SUBSTRATE: int = 5


RouteAction = Literal[
    "matching_engine",     # fall into existing matching flow
    "recommender_layer",   # dispatch a named recommender mode
    "ask_substrate",       # tell handler substrate is missing
    "out_of_scope_canned", # emit canned redirect; no engine
    "default",             # fall through to existing planner
]


@dataclass(frozen=True, slots=True)
class RecommenderRouteVerdict:
    """The router's single output. Handler acts on `action` first; the
    other fields are populated per action type.

    Field contracts:
      - `recommender_mode`: set iff action == "recommender_layer".
        One of the three RecommenderMode values from gap_evidence.
      - `voice_hint`: set iff action == "recommender_layer". Carries
        the original CareerIntent so the responder can branch voice
        for Layer B (skill_gap vs training_recommendation).
      - `missing`: set iff action == "ask_substrate". Subset of
        {"target", "skills"} naming what the substrate gate found
        absent.
      - `deferred_intent`: set iff action == "ask_substrate". The
        intent the router was about to fire; handler stashes this so
        the next turn (after substrate fills) can re-evaluate.
      - `reason`: telemetry-grade short string for logging. Always
        present.
    """
    action: RouteAction
    recommender_mode: RecommenderMode | None = None
    voice_hint: CareerIntent | None = None
    missing: tuple[str, ...] = field(default_factory=tuple)
    deferred_intent: CareerIntent | None = None
    reason: str = ""


# Pattern signals that win over career_intent. Anything else
# (asking_about_gap, asking_question, neutral) falls through to the
# career_intent branch per the locked decision.
_PATTERN_TO_MATCHING: frozenset[UserIntentSignal] = frozenset({
    "impatient_proceed",
})
_PATTERN_TO_DEFAULT: frozenset[UserIntentSignal] = frozenset({
    "declining",
    "confirming",
    "correcting",
    "redirecting",
})


def _check_substrate(
    *,
    target_noc: str | None,
    chat_skill_count: int,
    has_resume: bool,
) -> tuple[bool, tuple[str, ...]]:
    """Return (substrate_ok, missing). Missing is a stable-ordered
    tuple of {"target", "skills"} in that canonical order so tests
    can assert without sorting."""
    target_ok = (
        isinstance(target_noc, str)
        and len(target_noc) == 5
        and target_noc.isdigit()
    )
    skills_ok = chat_skill_count >= _MIN_CHAT_SKILLS_FOR_SUBSTRATE or has_resume

    missing: list[str] = []
    if not target_ok:
        missing.append("target")
    if not skills_ok:
        missing.append("skills")
    return (not missing, tuple(missing))


# Locked mapping from career_intent to recommender_mode. Layer B is
# the destination for both `local_skill_gap` and
# `training_recommendation`; the voice_hint carries the original
# intent so the responder can split voice.
_INTENT_TO_MODE: dict[CareerIntent, RecommenderMode] = {
    "local_skill_gap": "local_gap_coach",
    "training_recommendation": "local_gap_coach",
    "noc_standard_comparison": "target_noc_standard",
    "career_exploration": "adjacent_noc_standard",
}


def route_recommender(
    *,
    pattern_intent: UserIntentSignal,
    career_intent: CareerIntent,
    target_noc: str | None,
    chat_skill_count: int,
    has_resume: bool,
) -> RecommenderRouteVerdict:
    """Decide which engine path should fire this turn.

    Pure function. No side effects. Returns a frozen verdict the
    handler acts on in step 3.

    Decision precedence (locked):
      1. Hard pattern signals override career_intent.
      2. Career intent dispatch.
      3. Substrate-gated recommender intents check target+skills
         before firing the engine; missing substrate yields
         `ask_substrate` with `deferred_intent` set.
    """
    # ---- 1. Hard pattern signals win ----
    if pattern_intent in _PATTERN_TO_MATCHING:
        return RecommenderRouteVerdict(
            action="matching_engine",
            reason="hard_proceed_signal",
        )
    if pattern_intent in _PATTERN_TO_DEFAULT:
        # Bare yes/no/correct/redirect -- not engine selectors. The
        # handler's existing consume hooks + planner handle these.
        return RecommenderRouteVerdict(
            action="default",
            reason=f"pattern_signal:{pattern_intent}",
        )

    # asking_about_gap, asking_question, neutral -> fall through

    # ---- 2. Career intent dispatch ----
    if career_intent == "job_matching":
        return RecommenderRouteVerdict(
            action="matching_engine",
            reason="career_intent_job_matching",
        )
    if career_intent == "application_help_out_of_scope":
        return RecommenderRouteVerdict(
            action="out_of_scope_canned",
            reason="oos_application_help",
        )
    if career_intent == "unclear":
        return RecommenderRouteVerdict(
            action="default",
            reason="classifier_unclear",
        )

    # ---- 3. Substrate-gated recommender intents ----
    mode = _INTENT_TO_MODE.get(career_intent)
    if mode is None:
        # Defensive: a new CareerIntent value added without router
        # update lands here. Treat as unclear rather than misroute.
        return RecommenderRouteVerdict(
            action="default",
            reason=f"unknown_career_intent:{career_intent}",
        )

    substrate_ok, missing = _check_substrate(
        target_noc=target_noc,
        chat_skill_count=chat_skill_count,
        has_resume=has_resume,
    )
    if not substrate_ok:
        return RecommenderRouteVerdict(
            action="ask_substrate",
            missing=missing,
            deferred_intent=career_intent,
            reason=f"substrate_missing:{','.join(missing)}",
        )

    return RecommenderRouteVerdict(
        action="recommender_layer",
        recommender_mode=mode,
        voice_hint=career_intent,
        reason=f"career_intent_{career_intent}",
    )
