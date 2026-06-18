"""Slice B (2026-06-18): DerivedTurnState helper.

A pure projection of (TruthSummary, StagedProfile) into a single
state object every layer can read instead of re-deriving from
scratch. NO consumers yet -- this slice only adds the helper and
its tests. Slices A2 and C wire planner/arbiter/handler to use it.

Design constraints (locked with Nazmul 2026-06-18):
  - Frozen dataclass with slots. No methods.
  - Pure function -- no side effects, no mutation of inputs.
  - engine_readiness derives from truth.enough_to_match + target_status.
    NO duplicated logic with `_compute_enough_to_match` -- the helper
    is a projection of the existing truth, not a re-implementation.
  - No `intent_resolves_pending` field. Each pending flag resolves by
    different semantics (registry entity for training_topic, canonical
    snapshot for credential, dedicated classifier for adjacent_search).
    Baking a generic boolean here would be premature and wrong. Slice
    A2 introduces a separate `classify_pending_response(...)` helper
    that reads the per-flag semantics correctly.
  - pending_count is a cheap proxy that still answers "is the
    situation ambiguous?" without claiming resolution logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from skillbridge.chat.truth_summary import (
    TargetRoleSpecificity,
    TruthSummary,
    UserIntentSignal,
)
from skillbridge.session.staging import StagedProfile

EvidenceStatus = Literal[
    "none",              # no resume parsed, 0 chat-claimed skills
    "thin_chat",         # 1-2 chat skills, no usable resume
    "rich_chat",         # 3+ chat skills, no usable resume
    "resume_only",       # resume parsed (not failed), 0-2 chat skills
    "resume_plus_chat",  # resume parsed AND 3+ chat skills
]

EngineReadiness = Literal[
    "ready_with_target",            # specific target + enough_to_match
    "ready_skills_only_explicit",   # missing/vague target + enough_to_match
                                    # (means user passed the impatient-intent
                                    # gate from A1)
    "not_ready_missing_target",     # missing/vague target + !enough_to_match
    "not_ready_missing_evidence",   # specific target + !enough_to_match
]


@dataclass(frozen=True, slots=True)
class DerivedTurnState:
    """Single-source-of-truth state object for one turn.

    A pure projection of TruthSummary + StagedProfile. No layer reads
    this in Slice B -- the helper exists so Slices A2 and C have a
    stable shape to consume.
    """
    target_status: TargetRoleSpecificity
    evidence_status: EvidenceStatus
    pending_flags_active: frozenset[str]
    pending_count: int
    user_intent: UserIntentSignal
    engine_readiness: EngineReadiness


# Resume parse qualities that count as "usable resume evidence". Mirrors
# the same guard in `_compute_enough_to_match`'s `usable_evidence_present`
# computation (truth_summary.py:486-488).
_USABLE_RESUME_QUALITIES: frozenset[str] = frozenset({
    "full", "skills_only", "work_only", "partial",
})


def _count_chat_skills(staged: StagedProfile) -> int:
    """Same formula as truth_summary.py:814 and handler.py:3693. Kept
    inside the helper so callers don't have to thread it through."""
    return sum(1 for s in staged.skills if s.source != "resume")


def _classify_evidence(
    *, resume_usable: bool, chat_skill_count: int,
) -> EvidenceStatus:
    if resume_usable and chat_skill_count >= 3:
        return "resume_plus_chat"
    if resume_usable:
        return "resume_only"
    if chat_skill_count >= 3:
        return "rich_chat"
    if chat_skill_count >= 1:
        return "thin_chat"
    return "none"


def _collect_pending_flags(staged: StagedProfile) -> frozenset[str]:
    flags = set()
    if staged.pending_credential_confirmation is not None:
        flags.add("credential_confirmation")
    if staged.pending_adjacent_offer:
        flags.add("adjacent_offer")
    if staged.pending_training_topic:
        flags.add("training_topic")
    if staged.pending_adjacent_search_offer:
        flags.add("adjacent_search_offer")
    return frozenset(flags)


def _project_engine_readiness(
    *, enough_to_match: bool, target_status: TargetRoleSpecificity,
) -> EngineReadiness:
    """Pure projection of (enough_to_match, target_status) -> readiness
    label. By construction, this stays in sync with whatever rule
    `_compute_enough_to_match` uses -- it reads the boolean output
    rather than re-implementing the logic."""
    if enough_to_match:
        if target_status == "specific":
            return "ready_with_target"
        return "ready_skills_only_explicit"
    if target_status == "specific":
        return "not_ready_missing_evidence"
    return "not_ready_missing_target"


def derive_turn_state(
    *,
    truth: TruthSummary,
    staged: StagedProfile,
) -> DerivedTurnState:
    """Pure helper. Returns a frozen DerivedTurnState built by
    projecting `truth` + `staged`. NO mutation of inputs, NO I/O,
    safe to call multiple times in the same turn."""
    chat_skill_count = _count_chat_skills(staged)
    resume_usable = truth.resume_parse_quality in _USABLE_RESUME_QUALITIES
    evidence_status = _classify_evidence(
        resume_usable=resume_usable,
        chat_skill_count=chat_skill_count,
    )
    pending_flags = _collect_pending_flags(staged)
    engine_readiness = _project_engine_readiness(
        enough_to_match=truth.enough_to_match,
        target_status=truth.target_role_specificity,
    )
    return DerivedTurnState(
        target_status=truth.target_role_specificity,
        evidence_status=evidence_status,
        pending_flags_active=pending_flags,
        pending_count=len(pending_flags),
        user_intent=truth.user_intent_signal,
        engine_readiness=engine_readiness,
    )
