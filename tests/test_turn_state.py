"""Tests for skillbridge.chat.turn_state.derive_turn_state.

Slice B (2026-06-18): pure projection helper. No layer consumes
this yet -- these tests pin the shape so Slices A2 and C can rely
on it.

Coverage:
  - target_status mapping (3 cases)
  - evidence_status mapping (5 cases)
  - pending_flags_active build + pending_count (6 cases)
  - engine_readiness derivation (4 base + 2 edge)
  - DerivedTurnState frozen / immutable
  - Pure: idempotent, no mutation of inputs
"""
from __future__ import annotations

import pytest

from skillbridge.chat.truth_summary import (
    ResumeFactsSummary,
    TruthSummary,
)
from skillbridge.chat.turn_state import (
    DerivedTurnState,
    derive_turn_state,
)
from skillbridge.session.staging import StagedProfile, StagedSkill

# Pure-logic tests; opt out of conftest._clean_db's DB truncate
# (mirrors tests/test_truth_summary.py:36).
pytestmark = pytest.mark.nodb


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------
def _staged(
    *,
    chat_skill_count: int = 0,
    pending_credential: bool = False,
    pending_adjacent_offer: bool = False,
    pending_training_topic: bool = False,
    pending_adjacent_search_offer: bool = False,
) -> StagedProfile:
    sp = StagedProfile(
        session_id="t",
        created_at="2026-06-18T00:00:00+00:00",
        last_active_at="2026-06-18T00:00:00+00:00",
    )
    sp.skills = [
        StagedSkill(skill_name=f"skill_{i}", source="chat")
        for i in range(chat_skill_count)
    ]
    if pending_credential:
        sp.pending_credential_confirmation = {
            "canonical": "x", "action": "add",
        }
    sp.pending_adjacent_offer = pending_adjacent_offer
    sp.pending_training_topic = pending_training_topic
    sp.pending_adjacent_search_offer = pending_adjacent_search_offer
    return sp


def _truth(
    *,
    target_role_specificity: str = "none",
    resume_parse_quality: str = "no_resume",
    enough_to_match: bool = False,
    user_intent_signal: str = "neutral",
) -> TruthSummary:
    return TruthSummary(
        user_message="",
        target_role_specificity=target_role_specificity,  # type: ignore[arg-type]
        resume_parse_quality=resume_parse_quality,  # type: ignore[arg-type]
        resume_facts_summary=ResumeFactsSummary(),
        enough_to_match=enough_to_match,
        user_intent_signal=user_intent_signal,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# target_status -- direct projection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("specificity", ["none", "vague", "specific"])
def test_target_status_is_a_direct_projection_of_truth(specificity):
    state = derive_turn_state(
        truth=_truth(target_role_specificity=specificity),
        staged=_staged(),
    )
    assert state.target_status == specificity


# ---------------------------------------------------------------------------
# evidence_status -- classification table
# ---------------------------------------------------------------------------
def test_evidence_none_when_no_resume_and_no_chat_skills():
    state = derive_turn_state(
        truth=_truth(resume_parse_quality="no_resume"),
        staged=_staged(chat_skill_count=0),
    )
    assert state.evidence_status == "none"


def test_evidence_thin_chat_when_one_or_two_chat_skills_no_resume():
    state = derive_turn_state(
        truth=_truth(resume_parse_quality="no_resume"),
        staged=_staged(chat_skill_count=2),
    )
    assert state.evidence_status == "thin_chat"


def test_evidence_rich_chat_when_three_or_more_chat_skills_no_resume():
    state = derive_turn_state(
        truth=_truth(resume_parse_quality="no_resume"),
        staged=_staged(chat_skill_count=3),
    )
    assert state.evidence_status == "rich_chat"


def test_evidence_resume_only_when_resume_usable_and_few_chat_skills():
    state = derive_turn_state(
        truth=_truth(resume_parse_quality="full"),
        staged=_staged(chat_skill_count=2),
    )
    assert state.evidence_status == "resume_only"


def test_evidence_resume_plus_chat_when_both_present():
    state = derive_turn_state(
        truth=_truth(resume_parse_quality="full"),
        staged=_staged(chat_skill_count=5),
    )
    assert state.evidence_status == "resume_plus_chat"


def test_evidence_failed_resume_treated_as_no_resume():
    """A failed resume parse should NOT count as usable evidence."""
    state = derive_turn_state(
        truth=_truth(resume_parse_quality="failed"),
        staged=_staged(chat_skill_count=0),
    )
    assert state.evidence_status == "none"


def test_evidence_skills_only_resume_quality_is_usable():
    state = derive_turn_state(
        truth=_truth(resume_parse_quality="skills_only"),
        staged=_staged(chat_skill_count=0),
    )
    assert state.evidence_status == "resume_only"


# ---------------------------------------------------------------------------
# pending_flags_active + pending_count
# ---------------------------------------------------------------------------
def test_pending_flags_empty_when_none_set():
    state = derive_turn_state(truth=_truth(), staged=_staged())
    assert state.pending_flags_active == frozenset()
    assert state.pending_count == 0


def test_pending_credential_confirmation_detected():
    state = derive_turn_state(
        truth=_truth(),
        staged=_staged(pending_credential=True),
    )
    assert state.pending_flags_active == frozenset({"credential_confirmation"})
    assert state.pending_count == 1


def test_pending_adjacent_offer_detected():
    state = derive_turn_state(
        truth=_truth(),
        staged=_staged(pending_adjacent_offer=True),
    )
    assert state.pending_flags_active == frozenset({"adjacent_offer"})
    assert state.pending_count == 1


def test_pending_training_topic_detected():
    state = derive_turn_state(
        truth=_truth(),
        staged=_staged(pending_training_topic=True),
    )
    assert state.pending_flags_active == frozenset({"training_topic"})
    assert state.pending_count == 1


def test_pending_adjacent_search_offer_detected():
    state = derive_turn_state(
        truth=_truth(),
        staged=_staged(pending_adjacent_search_offer=True),
    )
    assert state.pending_flags_active == frozenset({"adjacent_search_offer"})
    assert state.pending_count == 1


def test_pending_count_reflects_multiple_active_flags():
    """The ambiguous-yes case: two flags both expecting yes/no input.
    Slice A2 will use pending_count > 1 to ASK rather than guess."""
    state = derive_turn_state(
        truth=_truth(),
        staged=_staged(
            pending_adjacent_offer=True,
            pending_adjacent_search_offer=True,
        ),
    )
    assert state.pending_flags_active == frozenset(
        {"adjacent_offer", "adjacent_search_offer"}
    )
    assert state.pending_count == 2


def test_pending_count_with_three_active_flags():
    state = derive_turn_state(
        truth=_truth(),
        staged=_staged(
            pending_credential=True,
            pending_adjacent_offer=True,
            pending_training_topic=True,
        ),
    )
    assert state.pending_count == 3


# ---------------------------------------------------------------------------
# engine_readiness -- projection of (enough_to_match, target_status)
# ---------------------------------------------------------------------------
def test_engine_readiness_ready_with_target():
    state = derive_turn_state(
        truth=_truth(
            target_role_specificity="specific",
            enough_to_match=True,
        ),
        staged=_staged(),
    )
    assert state.engine_readiness == "ready_with_target"


def test_engine_readiness_ready_skills_only_explicit_when_target_missing():
    """enough_to_match=True + target=none means A1's
    `skills_only_explicit_request` rule fired."""
    state = derive_turn_state(
        truth=_truth(
            target_role_specificity="none",
            enough_to_match=True,
            user_intent_signal="impatient_proceed",
        ),
        staged=_staged(chat_skill_count=5),
    )
    assert state.engine_readiness == "ready_skills_only_explicit"


def test_engine_readiness_ready_skills_only_explicit_when_target_vague():
    state = derive_turn_state(
        truth=_truth(
            target_role_specificity="vague",
            enough_to_match=True,
            user_intent_signal="impatient_proceed",
        ),
        staged=_staged(chat_skill_count=5),
    )
    assert state.engine_readiness == "ready_skills_only_explicit"


def test_engine_readiness_not_ready_missing_target():
    state = derive_turn_state(
        truth=_truth(
            target_role_specificity="none",
            enough_to_match=False,
        ),
        staged=_staged(),
    )
    assert state.engine_readiness == "not_ready_missing_target"


def test_engine_readiness_not_ready_missing_target_when_vague():
    state = derive_turn_state(
        truth=_truth(
            target_role_specificity="vague",
            enough_to_match=False,
        ),
        staged=_staged(),
    )
    assert state.engine_readiness == "not_ready_missing_target"


def test_engine_readiness_not_ready_missing_evidence():
    state = derive_turn_state(
        truth=_truth(
            target_role_specificity="specific",
            enough_to_match=False,
        ),
        staged=_staged(),
    )
    assert state.engine_readiness == "not_ready_missing_evidence"


# ---------------------------------------------------------------------------
# user_intent -- direct projection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("intent", [
    "neutral", "confirming", "declining", "impatient_proceed",
    "asking_question", "asking_about_gap", "correcting", "redirecting",
])
def test_user_intent_is_a_direct_projection(intent):
    state = derive_turn_state(
        truth=_truth(user_intent_signal=intent),
        staged=_staged(),
    )
    assert state.user_intent == intent


# ---------------------------------------------------------------------------
# Frozen / immutability
# ---------------------------------------------------------------------------
def test_derived_turn_state_is_frozen():
    state = derive_turn_state(truth=_truth(), staged=_staged())
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        state.target_status = "specific"  # type: ignore[misc]


def test_derived_turn_state_uses_slots():
    """Slot-only dataclass should reject arbitrary attribute writes."""
    state = derive_turn_state(truth=_truth(), staged=_staged())
    with pytest.raises(AttributeError):
        object.__setattr__(state, "extra_field", "x")


# ---------------------------------------------------------------------------
# Purity -- idempotent, no mutation
# ---------------------------------------------------------------------------
def test_derive_turn_state_is_pure_and_idempotent():
    """Calling twice with the same inputs returns equal results, and
    neither call mutates staged or truth."""
    truth = _truth(
        target_role_specificity="specific",
        resume_parse_quality="full",
        enough_to_match=True,
    )
    staged = _staged(chat_skill_count=3, pending_adjacent_offer=True)

    truth_target_before = truth.target_role_specificity
    staged_skills_before = len(staged.skills)
    staged_pending_before = staged.pending_adjacent_offer

    a = derive_turn_state(truth=truth, staged=staged)
    b = derive_turn_state(truth=truth, staged=staged)

    assert a == b
    assert truth.target_role_specificity == truth_target_before
    assert len(staged.skills) == staged_skills_before
    assert staged.pending_adjacent_offer == staged_pending_before


# ---------------------------------------------------------------------------
# No-consumer guarantee for Slice B
# ---------------------------------------------------------------------------
def test_derive_turn_state_is_not_yet_consumed_by_production_layers():
    """Slice B adds the helper but no production layer reads it yet.
    Slices A2 and C will wire planner/arbiter/handler to consume
    DerivedTurnState. Until then this guard catches accidental early
    coupling.

    If this test fails because a legitimate consumer was added, delete
    the test in the same diff that introduces the consumer.
    """
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    skillbridge_dir = repo_root / "skillbridge"

    # Use git grep-equivalent via Python -- no subprocess on grep itself
    # because we want this test to run on any platform without ripgrep
    # installed.
    sentinel = "derive_turn_state"
    hits: list[str] = []
    for path in skillbridge_dir.rglob("*.py"):
        if path.name == "turn_state.py":
            continue  # defines it
        text = path.read_text(encoding="utf-8")
        if sentinel in text:
            hits.append(str(path.relative_to(repo_root)))

    assert not hits, (
        "Slice B contract: no production layer should import or call "
        f"derive_turn_state yet. Found references in: {hits}. "
        "If you intentionally wired a consumer in Slices A2/C, delete "
        "this test in the same diff that adds the consumer."
    )

    # Silence flake8 about unused import
    _ = sys, subprocess
