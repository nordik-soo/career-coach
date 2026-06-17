"""Intake state machine for guided chat.

The state machine lives in the BACKEND, not the LLM. The LLM only fills
slots (extractor) and narrates the next move (responder). The decision of
what to do next is deterministic Python.

States:
  anonymous_chat           — first turn, nothing extracted yet.
  intake_collecting        — extractor is filling slots; we are asking
                             follow-up questions to reach band readiness.
  intake_ready_for_consent — enough slots filled; we can offer matches
                             with consent OR run a preview match.
  matching                 — matches computed (in-memory or persisted).
  recommendation_review    — user has been shown matches; we can refine
                             or update profile.
  profile_update_loop      — post-consent path; same loop but persisted.
  resume_review            — Sprint 1: the user just uploaded a resume.
                             We narrate the parsed facts and ask for
                             corrections before continuing to the regular
                             intake / matching loop. The handler sets
                             this state directly after a successful
                             upload; decide() doesn't transition into
                             it on its own.

Transitions are driven by:
  - profile completeness (chat.intake_priority.completeness_band)
  - explicit user actions (consent grant happens in routes/profiles.py
    and pushes us into profile_update_loop)
  - off-topic messages keep state unchanged
"""
from __future__ import annotations

from dataclasses import dataclass

from skillbridge.chat.intake_priority import (
    MIN_SLOTS_FOR_MATCH,
    completeness_band,
)
from skillbridge.session.staging import StagedProfile

# State string constants — strings (not enum) because StagedProfile
# serializes to JSON via the session store.
STATE_ANONYMOUS_CHAT          = "anonymous_chat"
STATE_INTAKE_COLLECTING       = "intake_collecting"
STATE_INTAKE_READY_FOR_CONSENT = "intake_ready_for_consent"
STATE_MATCHING                 = "matching"
STATE_RECOMMENDATION_REVIEW    = "recommendation_review"
STATE_PROFILE_UPDATE_LOOP      = "profile_update_loop"
STATE_RESUME_REVIEW            = "resume_review"   # Sprint 1

VALID_STATES: frozenset[str] = frozenset({
    STATE_ANONYMOUS_CHAT,
    STATE_INTAKE_COLLECTING,
    STATE_INTAKE_READY_FOR_CONSENT,
    STATE_MATCHING,
    STATE_RECOMMENDATION_REVIEW,
    STATE_PROFILE_UPDATE_LOOP,
    STATE_RESUME_REVIEW,
})


# NEXT_ACTION constants — passed to the responder so it knows what to say.
ACTION_ASK_QUESTIONS         = "ASK_QUESTIONS"
ACTION_CONFIRM_READY         = "CONFIRM_READY"
ACTION_PRESENT_MATCHES       = "PRESENT_MATCHES"
ACTION_REDIRECT              = "REDIRECT"               # off-topic
ACTION_ACKNOWLEDGE_AND_WAIT  = "ACKNOWLEDGE_AND_WAIT"   # decline / "thanks"
ACTION_PRESENT_RESUME_FACTS  = "PRESENT_RESUME_FACTS"   # Sprint 1: confirmation turn after upload


@dataclass
class Decision:
    next_state: str
    action: str
    ask_slots: list[str]          # slots the responder may ask about (max 3)
    show_matches: bool            # whether to compute & include matches in payload
    redirect_reason: str | None = None


def decide(
    staged: StagedProfile,
    *,
    off_topic: bool,
    extracted_anything: bool,
    declined_this_turn: list[str],
    authenticated: bool,
) -> Decision:
    """Return the next decision given the current staged profile.

    The handler is responsible for calling extractor.extract() first and
    merging non-declined fields into staged BEFORE calling decide().
    """
    current = staged.intake_state if staged.intake_state in VALID_STATES else STATE_ANONYMOUS_CHAT

    # Hard rule across every branch: ONE question per turn. The state
    # machine still ranks N possible slots in priority order, but the
    # responder only ever asks the single highest-priority one. Users
    # overwhelmingly prefer a natural back-and-forth over a multi-question
    # form. If we ever need a closely-related pair (e.g. "frontend or
    # backend?" + "full-time or part-time?"), we'll add an explicit pair
    # mechanism — not relax the cap.

    # ---- Off-topic short-circuit ----------------------------------------
    if off_topic and not declined_this_turn:
        return Decision(
            next_state=current,
            action=ACTION_REDIRECT,
            ask_slots=_pick_ask_slots(staged, max_n=1),
            show_matches=False,
            redirect_reason="off_topic",
        )

    # ---- Decline-only turn ----------------------------------------------
    if declined_this_turn and not extracted_anything:
        return Decision(
            next_state=current if current != STATE_ANONYMOUS_CHAT else STATE_INTAKE_COLLECTING,
            action=ACTION_ACKNOWLEDGE_AND_WAIT,
            ask_slots=_pick_ask_slots(staged, max_n=1),
            show_matches=False,
        )

    # ---- Decide state from completeness ---------------------------------
    band = completeness_band(staged)

    if authenticated:
        if band == "ready":
            return Decision(
                next_state=STATE_PROFILE_UPDATE_LOOP,
                action=ACTION_PRESENT_MATCHES,
                ask_slots=_pick_ask_slots(staged, max_n=1),
                show_matches=True,
            )
        return Decision(
            next_state=STATE_PROFILE_UPDATE_LOOP,
            action=ACTION_ASK_QUESTIONS,
            ask_slots=_pick_ask_slots(staged, max_n=1),
            show_matches=False,
        )

    if band == "ready":
        return Decision(
            next_state=STATE_INTAKE_READY_FOR_CONSENT,
            action=ACTION_PRESENT_MATCHES,
            ask_slots=_pick_ask_slots(staged, max_n=1),
            show_matches=True,
        )

    if band == "minimum" and current in {STATE_INTAKE_READY_FOR_CONSENT, STATE_RECOMMENDATION_REVIEW}:
        return Decision(
            next_state=STATE_INTAKE_READY_FOR_CONSENT,
            action=ACTION_PRESENT_MATCHES,
            ask_slots=_pick_ask_slots(staged, max_n=1),
            show_matches=True,
        )

    if band == "minimum":
        # If there's still a primary slot to ask about, confirm + ask one.
        # If we've exhausted primary slots, skip the confirm dance and
        # show matches now — no point asking about location/salary just
        # to fill space.
        ask = _pick_ask_slots(staged, max_n=1)
        if ask:
            return Decision(
                next_state=STATE_INTAKE_COLLECTING,
                action=ACTION_CONFIRM_READY,
                ask_slots=ask,
                show_matches=False,
            )
        return Decision(
            next_state=STATE_INTAKE_READY_FOR_CONSENT,
            action=ACTION_PRESENT_MATCHES,
            ask_slots=[],
            show_matches=True,
        )

    # band == "low"
    return Decision(
        next_state=STATE_INTAKE_COLLECTING,
        action=ACTION_ASK_QUESTIONS,
        ask_slots=_pick_ask_slots(staged, max_n=1),
        show_matches=False,
    )


def _pick_ask_slots(staged: StagedProfile, *, max_n: int) -> list[str]:
    """Delegate to intake_priority — kept here so callers only import one module."""
    from skillbridge.chat.intake_priority import pick_slots_to_ask
    return pick_slots_to_ask(staged, max_n=max_n)


# Backwards-compat re-export so callers don't need two imports.
__all__ = [
    "STATE_ANONYMOUS_CHAT", "STATE_INTAKE_COLLECTING",
    "STATE_INTAKE_READY_FOR_CONSENT", "STATE_MATCHING",
    "STATE_RECOMMENDATION_REVIEW", "STATE_PROFILE_UPDATE_LOOP",
    "STATE_RESUME_REVIEW",
    "VALID_STATES",
    "ACTION_ASK_QUESTIONS", "ACTION_CONFIRM_READY",
    "ACTION_PRESENT_MATCHES", "ACTION_REDIRECT", "ACTION_ACKNOWLEDGE_AND_WAIT",
    "ACTION_PRESENT_RESUME_FACTS",
    "Decision", "decide",
    "MIN_SLOTS_FOR_MATCH",
]
