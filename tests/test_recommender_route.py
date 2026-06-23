"""Tests for `route_recommender` -- pure function, no LLM, no DB.

See [[project-recommender-peer-engine-locked]] for the design lock.
The router's contract: given (pattern_intent, career_intent,
target_noc, chat_skill_count, has_resume), produce a verdict the
handler acts on. Step 3 wires the verdict into handle_anonymous.
"""
from __future__ import annotations

import pytest

from skillbridge.chat.recommender_route import (
    RecommenderRouteVerdict,
    route_recommender,
)

pytestmark = pytest.mark.nodb


# Convenience -- realistic substrate state for happy-path tests.
_FULL_SUBSTRATE: dict = dict(
    target_noc="14200",
    chat_skill_count=8,
    has_resume=True,
)


# ===========================================================================
# Pattern-signal precedence
# ===========================================================================
@pytest.mark.parametrize("career_intent", [
    "job_matching",
    "local_skill_gap",
    "training_recommendation",
    "noc_standard_comparison",
    "career_exploration",
    "application_help_out_of_scope",
    "unclear",
])
def test_impatient_proceed_always_wins_to_matching(career_intent):
    """`impatient_proceed` (show jobs, match now) routes to matching
    regardless of career_intent or substrate."""
    v = route_recommender(
        pattern_intent="impatient_proceed",
        career_intent=career_intent,
        **_FULL_SUBSTRATE,
    )
    assert v.action == "matching_engine"
    assert v.reason == "hard_proceed_signal"


@pytest.mark.parametrize("pattern_intent", [
    "declining", "confirming", "correcting", "redirecting",
])
def test_consent_and_state_change_signals_fall_to_default(pattern_intent):
    """Bare yes/no/correct/redirect are NOT engine selectors. They
    route to default; handler's existing consume hooks + planner
    handle them."""
    v = route_recommender(
        pattern_intent=pattern_intent,
        career_intent="local_skill_gap",
        **_FULL_SUBSTRATE,
    )
    assert v.action == "default"
    assert v.reason.startswith("pattern_signal:")


@pytest.mark.parametrize("pattern_intent", [
    "asking_about_gap", "asking_question", "neutral",
])
def test_non_override_patterns_fall_through_to_career_intent(pattern_intent):
    """asking_about_gap, asking_question, neutral are NOT pattern
    overrides. Router uses career_intent to decide."""
    # With job_matching career_intent, these should route to matching.
    v = route_recommender(
        pattern_intent=pattern_intent,
        career_intent="job_matching",
        **_FULL_SUBSTRATE,
    )
    assert v.action == "matching_engine"
    assert v.reason == "career_intent_job_matching"


# ===========================================================================
# Career intent dispatch -- non-substrate intents
# ===========================================================================
def test_job_matching_intent_routes_to_matching():
    v = route_recommender(
        pattern_intent="neutral",
        career_intent="job_matching",
        target_noc=None, chat_skill_count=0, has_resume=False,
    )
    # Job-matching does NOT need substrate to enter the engine; the
    # matching engine has its own intake flow for thin profiles.
    assert v.action == "matching_engine"


def test_application_help_routes_to_canned_redirect():
    v = route_recommender(
        pattern_intent="neutral",
        career_intent="application_help_out_of_scope",
        **_FULL_SUBSTRATE,
    )
    assert v.action == "out_of_scope_canned"
    assert v.reason == "oos_application_help"
    # Wording is NOT emitted by the router -- handler step 3 owns it.
    assert not hasattr(v, "canned_message") or getattr(v, "canned_message", None) is None


def test_unclear_intent_routes_to_default():
    v = route_recommender(
        pattern_intent="neutral",
        career_intent="unclear",
        **_FULL_SUBSTRATE,
    )
    assert v.action == "default"
    assert v.reason == "classifier_unclear"


# ===========================================================================
# Substrate-satisfied recommender paths
# ===========================================================================
@pytest.mark.parametrize("career_intent, expected_mode", [
    ("local_skill_gap", "local_gap_coach"),
    ("training_recommendation", "local_gap_coach"),
    ("noc_standard_comparison", "target_noc_standard"),
    ("career_exploration", "adjacent_noc_standard"),
])
def test_recommender_intent_with_substrate_routes_to_layer(
    career_intent, expected_mode,
):
    v = route_recommender(
        pattern_intent="neutral",
        career_intent=career_intent,
        **_FULL_SUBSTRATE,
    )
    assert v.action == "recommender_layer"
    assert v.recommender_mode == expected_mode
    assert v.voice_hint == career_intent
    assert v.reason == f"career_intent_{career_intent}"


def test_voice_hint_differentiates_layer_b_intents():
    """Both local_skill_gap and training_recommendation map to
    local_gap_coach mode, but voice_hint must differ so the responder
    can switch voice in step 3."""
    skill_gap = route_recommender(
        pattern_intent="neutral",
        career_intent="local_skill_gap",
        **_FULL_SUBSTRATE,
    )
    training = route_recommender(
        pattern_intent="neutral",
        career_intent="training_recommendation",
        **_FULL_SUBSTRATE,
    )
    assert skill_gap.recommender_mode == training.recommender_mode == "local_gap_coach"
    assert skill_gap.voice_hint == "local_skill_gap"
    assert training.voice_hint == "training_recommendation"


# ===========================================================================
# Substrate-missing paths
# ===========================================================================
@pytest.mark.parametrize("career_intent", [
    "local_skill_gap",
    "training_recommendation",
    "noc_standard_comparison",
    "career_exploration",
])
def test_recommender_intent_no_target_asks_substrate(career_intent):
    v = route_recommender(
        pattern_intent="neutral",
        career_intent=career_intent,
        target_noc=None,
        chat_skill_count=8,
        has_resume=True,
    )
    assert v.action == "ask_substrate"
    assert v.missing == ("target",)
    assert v.deferred_intent == career_intent
    assert v.reason == "substrate_missing:target"


@pytest.mark.parametrize("career_intent", [
    "local_skill_gap",
    "training_recommendation",
    "noc_standard_comparison",
    "career_exploration",
])
def test_recommender_intent_no_skills_asks_substrate(career_intent):
    v = route_recommender(
        pattern_intent="neutral",
        career_intent=career_intent,
        target_noc="14200",
        chat_skill_count=0,
        has_resume=False,
    )
    assert v.action == "ask_substrate"
    assert v.missing == ("skills",)
    assert v.deferred_intent == career_intent
    assert v.reason == "substrate_missing:skills"


def test_recommender_intent_both_missing_lists_both():
    v = route_recommender(
        pattern_intent="neutral",
        career_intent="local_skill_gap",
        target_noc=None,
        chat_skill_count=0,
        has_resume=False,
    )
    assert v.action == "ask_substrate"
    # Locked canonical order: target before skills.
    assert v.missing == ("target", "skills")
    assert v.deferred_intent == "local_skill_gap"


# ===========================================================================
# Substrate edge cases (NOC format)
# ===========================================================================
@pytest.mark.parametrize("bad_noc", [
    "",           # empty
    "13",         # too short
    "13110abc",   # not all digits
    "ABC12",      # not digits
    "131100",     # too long
    None,         # explicit None
])
def test_invalid_noc_format_counts_as_no_target(bad_noc):
    v = route_recommender(
        pattern_intent="neutral",
        career_intent="local_skill_gap",
        target_noc=bad_noc,
        chat_skill_count=8,
        has_resume=True,
    )
    assert v.action == "ask_substrate"
    assert "target" in v.missing


def test_valid_5_digit_noc_passes():
    v = route_recommender(
        pattern_intent="neutral",
        career_intent="local_skill_gap",
        target_noc="13110",
        chat_skill_count=8,
        has_resume=True,
    )
    assert v.action == "recommender_layer"


# ===========================================================================
# Substrate edge cases (skills threshold)
# ===========================================================================
def test_resume_alone_satisfies_skills():
    """has_resume=True covers the skills floor even when chat_skill_count is 0."""
    v = route_recommender(
        pattern_intent="neutral",
        career_intent="local_skill_gap",
        target_noc="14200",
        chat_skill_count=0,
        has_resume=True,
    )
    assert v.action == "recommender_layer"


def test_chat_skills_at_threshold_satisfies_substrate():
    """chat_skill_count == 5 is sufficient (N=5 floor is inclusive)."""
    v = route_recommender(
        pattern_intent="neutral",
        career_intent="local_skill_gap",
        target_noc="14200",
        chat_skill_count=5,
        has_resume=False,
    )
    assert v.action == "recommender_layer"


def test_chat_skills_below_threshold_no_resume_misses():
    v = route_recommender(
        pattern_intent="neutral",
        career_intent="local_skill_gap",
        target_noc="14200",
        chat_skill_count=4,
        has_resume=False,
    )
    assert v.action == "ask_substrate"
    assert v.missing == ("skills",)


def test_chat_skills_below_threshold_with_resume_passes():
    """Resume is a substitute for chat_skill_count >= 5."""
    v = route_recommender(
        pattern_intent="neutral",
        career_intent="local_skill_gap",
        target_noc="14200",
        chat_skill_count=3,
        has_resume=True,
    )
    assert v.action == "recommender_layer"


def test_no_target_no_skills_returns_both_in_canonical_order():
    v = route_recommender(
        pattern_intent="neutral",
        career_intent="noc_standard_comparison",
        target_noc=None,
        chat_skill_count=2,
        has_resume=False,
    )
    assert v.missing == ("target", "skills")  # target first, then skills


# ===========================================================================
# Defensive: action carries the right adjacent fields
# ===========================================================================
def test_matching_engine_action_has_no_recommender_mode():
    v = route_recommender(
        pattern_intent="impatient_proceed",
        career_intent="local_skill_gap",
        **_FULL_SUBSTRATE,
    )
    assert v.recommender_mode is None
    assert v.voice_hint is None
    assert v.missing == ()
    assert v.deferred_intent is None


def test_default_action_has_no_recommender_mode():
    v = route_recommender(
        pattern_intent="neutral",
        career_intent="unclear",
        **_FULL_SUBSTRATE,
    )
    assert v.recommender_mode is None
    assert v.voice_hint is None
    assert v.missing == ()
    assert v.deferred_intent is None


def test_out_of_scope_action_has_no_recommender_mode():
    v = route_recommender(
        pattern_intent="neutral",
        career_intent="application_help_out_of_scope",
        **_FULL_SUBSTRATE,
    )
    assert v.recommender_mode is None
    assert v.voice_hint is None
    assert v.missing == ()
    assert v.deferred_intent is None


def test_ask_substrate_action_has_no_recommender_mode():
    v = route_recommender(
        pattern_intent="neutral",
        career_intent="local_skill_gap",
        target_noc=None,
        chat_skill_count=0,
        has_resume=False,
    )
    assert v.action == "ask_substrate"
    assert v.recommender_mode is None
    assert v.voice_hint is None
    assert v.deferred_intent == "local_skill_gap"


def test_recommender_layer_action_has_no_substrate_missing():
    v = route_recommender(
        pattern_intent="neutral",
        career_intent="career_exploration",
        **_FULL_SUBSTRATE,
    )
    assert v.action == "recommender_layer"
    assert v.missing == ()
    assert v.deferred_intent is None


# ===========================================================================
# Verdict invariants -- frozen, slotted, hashable
# ===========================================================================
def test_verdict_is_frozen():
    v = route_recommender(
        pattern_intent="neutral",
        career_intent="unclear",
        **_FULL_SUBSTRATE,
    )
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        v.action = "matching_engine"  # type: ignore[misc]


def test_verdict_equality():
    """Two identical verdicts should compare equal (dataclass default)."""
    v1 = route_recommender(
        pattern_intent="neutral", career_intent="job_matching",
        **_FULL_SUBSTRATE,
    )
    v2 = route_recommender(
        pattern_intent="neutral", career_intent="job_matching",
        **_FULL_SUBSTRATE,
    )
    assert v1 == v2


# ===========================================================================
# No state mutation
# ===========================================================================
def test_router_does_not_mutate_inputs():
    """Defensive: pure function. None of the inputs are mutated."""
    inputs = dict(
        pattern_intent="neutral",
        career_intent="local_skill_gap",
        target_noc="14200",
        chat_skill_count=8,
        has_resume=True,
    )
    snapshot = dict(inputs)
    route_recommender(**inputs)
    assert inputs == snapshot
