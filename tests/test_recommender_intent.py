"""Tests for the LLM-based career intent classifier
(locked 2026-06-22). See [[project-recommender-peer-engine-locked]]
for the design context.

These tests mock the LLM call by monkeypatching `_call_classifier_llm`.
There is NO live LLM round-trip in this suite -- the classifier's
contract with the LLM (tool_use, schema, fallback) is tested
indirectly via the seam.
"""
from __future__ import annotations

import pytest

from skillbridge.chat import recommender_intent as ri

pytestmark = pytest.mark.nodb


@pytest.fixture(autouse=True)
def _clear_intent_cache():
    """Each test starts with an empty cache. Otherwise tests order
    can leak cached results across cases."""
    ri.reset_cache()
    yield
    ri.reset_cache()


# ---------------------------------------------------------------------------
# Defensive / empty-input handling
# ---------------------------------------------------------------------------
def test_blank_message_returns_unclear_without_calling_llm(monkeypatch):
    """Blank input bypasses the LLM entirely. The classifier MUST NOT
    call the API on empty input."""
    call_count = 0

    def fake_call(**kwargs):
        nonlocal call_count
        call_count += 1
        return "job_matching"

    monkeypatch.setattr(ri, "_call_classifier_llm", fake_call)
    monkeypatch.setattr(ri, "LLM_ENABLED", True)

    assert ri.classify_career_intent(message="") == "unclear"
    assert ri.classify_career_intent(message="   ") == "unclear"
    assert ri.classify_career_intent(message="\t\n  ") == "unclear"
    assert call_count == 0


def test_none_message_returns_unclear(monkeypatch):
    monkeypatch.setattr(ri, "LLM_ENABLED", True)
    assert ri.classify_career_intent(message=None) == "unclear"


def test_non_string_message_returns_unclear(monkeypatch):
    monkeypatch.setattr(ri, "LLM_ENABLED", True)
    # Defensive contract: non-string inputs are silently coerced to unclear,
    # not raised. Production code should never pass these but the
    # classifier degrades gracefully.
    assert ri.classify_career_intent(message=123) == "unclear"  # type: ignore[arg-type]
    assert ri.classify_career_intent(message=[]) == "unclear"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# LLM disabled path
# ---------------------------------------------------------------------------
def test_llm_disabled_returns_unclear(monkeypatch):
    """When the system runs without LLM (LLM_ENABLED=False), the
    classifier returns unclear and never attempts an API call."""
    call_count = 0

    def fake_call(**kwargs):
        nonlocal call_count
        call_count += 1
        return "job_matching"

    monkeypatch.setattr(ri, "LLM_ENABLED", False)
    monkeypatch.setattr(ri, "_call_classifier_llm", fake_call)

    assert ri.classify_career_intent(message="show me admin jobs") == "unclear"
    assert call_count == 0


def test_llm_disabled_result_is_cached(monkeypatch):
    """Disabled-LLM returns unclear but ALSO caches that result so
    repeated calls don't re-evaluate the disabled path."""
    monkeypatch.setattr(ri, "LLM_ENABLED", False)
    ri.classify_career_intent(message="show me admin jobs")
    assert ri.cache_size() == 1


# ---------------------------------------------------------------------------
# Happy path -- each enum value
# ---------------------------------------------------------------------------
def _stub_llm(monkeypatch, returns: str):
    """Patch the LLM seam to return the given intent."""
    monkeypatch.setattr(ri, "LLM_ENABLED", True)

    def fake_call(**kwargs):
        return returns

    monkeypatch.setattr(ri, "_call_classifier_llm", fake_call)


@pytest.mark.parametrize("intent", [
    "job_matching",
    "local_skill_gap",
    "training_recommendation",
    "noc_standard_comparison",
    "career_exploration",
    "application_help_out_of_scope",
    "unclear",
])
def test_each_enum_value_is_accepted_when_llm_returns_it(monkeypatch, intent):
    """All seven enum values must round-trip through the classifier
    without being downgraded to unclear."""
    _stub_llm(monkeypatch, intent)
    assert ri.classify_career_intent(message="some user message") == intent


# ---------------------------------------------------------------------------
# Invalid LLM output -- defense in depth on top of tool_use enum
# ---------------------------------------------------------------------------
def test_invalid_enum_value_falls_back_to_unclear(monkeypatch):
    """Even though tool_use enforces the enum schema, the classifier
    validates the returned value in code as defense-in-depth."""
    _stub_llm(monkeypatch, "not_a_real_intent")
    assert ri.classify_career_intent(message="show me jobs") == "unclear"


def test_empty_string_from_llm_falls_back_to_unclear(monkeypatch):
    _stub_llm(monkeypatch, "")
    assert ri.classify_career_intent(message="show me jobs") == "unclear"


def test_llm_exception_falls_back_to_unclear(monkeypatch):
    """Network errors, auth errors, anything thrown by the API client
    must be caught and degraded to unclear without breaking the caller."""
    monkeypatch.setattr(ri, "LLM_ENABLED", True)

    def boom(**kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(ri, "_call_classifier_llm", boom)
    assert ri.classify_career_intent(message="show me jobs") == "unclear"


# ---------------------------------------------------------------------------
# Cache behavior
# ---------------------------------------------------------------------------
def test_cache_hits_on_identical_input(monkeypatch):
    """Same (message, pending, target, last_move) tuple -> one LLM call."""
    monkeypatch.setattr(ri, "LLM_ENABLED", True)
    call_count = 0

    def fake_call(**kwargs):
        nonlocal call_count
        call_count += 1
        return "job_matching"

    monkeypatch.setattr(ri, "_call_classifier_llm", fake_call)

    ri.classify_career_intent(message="show me jobs")
    ri.classify_career_intent(message="show me jobs")
    ri.classify_career_intent(message="show me jobs")
    assert call_count == 1


def test_cache_differs_by_message(monkeypatch):
    monkeypatch.setattr(ri, "LLM_ENABLED", True)
    call_count = 0

    def fake_call(**kwargs):
        nonlocal call_count
        call_count += 1
        return "job_matching"

    monkeypatch.setattr(ri, "_call_classifier_llm", fake_call)

    ri.classify_career_intent(message="show me jobs")
    ri.classify_career_intent(message="show me admin jobs")
    assert call_count == 2


def test_cache_differs_by_pending_recommender_offer(monkeypatch):
    """Same surface message, different pending state -> two cache entries.
    'yes' after a local_gap_coach offer vs 'yes' after no pending offer
    can legitimately classify differently."""
    monkeypatch.setattr(ri, "LLM_ENABLED", True)
    call_count = 0

    def fake_call(**kwargs):
        nonlocal call_count
        call_count += 1
        return "unclear"

    monkeypatch.setattr(ri, "_call_classifier_llm", fake_call)

    ri.classify_career_intent(message="yes", pending_recommender_offer=None)
    ri.classify_career_intent(
        message="yes", pending_recommender_offer="local_gap_coach",
    )
    assert call_count == 2


def test_cache_differs_by_target_role_text(monkeypatch):
    monkeypatch.setattr(ri, "LLM_ENABLED", True)
    call_count = 0

    def fake_call(**kwargs):
        nonlocal call_count
        call_count += 1
        return "unclear"

    monkeypatch.setattr(ri, "_call_classifier_llm", fake_call)

    ri.classify_career_intent(message="what should I improve?",
                              target_role_text=None)
    ri.classify_career_intent(message="what should I improve?",
                              target_role_text="administrative assistant")
    assert call_count == 2


def test_cache_differs_by_last_assistant_move(monkeypatch):
    monkeypatch.setattr(ri, "LLM_ENABLED", True)
    call_count = 0

    def fake_call(**kwargs):
        nonlocal call_count
        call_count += 1
        return "unclear"

    monkeypatch.setattr(ri, "_call_classifier_llm", fake_call)

    ri.classify_career_intent(message="yes", last_assistant_move=None)
    ri.classify_career_intent(
        message="yes", last_assistant_move="present_tiered_matches",
    )
    assert call_count == 2


def test_cache_key_full_combination(monkeypatch):
    """Verify all four context fields participate in the cache key."""
    monkeypatch.setattr(ri, "LLM_ENABLED", True)
    call_count = 0

    def fake_call(**kwargs):
        nonlocal call_count
        call_count += 1
        return "unclear"

    monkeypatch.setattr(ri, "_call_classifier_llm", fake_call)

    # Four distinct context combinations on the same message.
    common = dict(message="hi")
    ri.classify_career_intent(
        **common, pending_recommender_offer=None,
        target_role_text=None, last_assistant_move=None,
    )
    ri.classify_career_intent(
        **common, pending_recommender_offer="local_gap_coach",
        target_role_text=None, last_assistant_move=None,
    )
    ri.classify_career_intent(
        **common, pending_recommender_offer=None,
        target_role_text="admin", last_assistant_move=None,
    )
    ri.classify_career_intent(
        **common, pending_recommender_offer=None,
        target_role_text=None, last_assistant_move="present_tiered_matches",
    )
    assert call_count == 4
    assert ri.cache_size() == 4


def test_message_is_normalized_for_cache(monkeypatch):
    """Leading/trailing whitespace must NOT create a cache miss --
    the classifier strips before lookup."""
    monkeypatch.setattr(ri, "LLM_ENABLED", True)
    call_count = 0

    def fake_call(**kwargs):
        nonlocal call_count
        call_count += 1
        return "job_matching"

    monkeypatch.setattr(ri, "_call_classifier_llm", fake_call)

    ri.classify_career_intent(message="show me jobs")
    ri.classify_career_intent(message="  show me jobs  ")
    ri.classify_career_intent(message="\tshow me jobs\n")
    assert call_count == 1


# ---------------------------------------------------------------------------
# Context flows into the LLM call
# ---------------------------------------------------------------------------
def test_context_fields_are_passed_to_llm(monkeypatch):
    """The LLM seam must receive all four context fields verbatim
    (or null when caller passed None)."""
    monkeypatch.setattr(ri, "LLM_ENABLED", True)
    received: dict = {}

    def fake_call(**kwargs):
        received.update(kwargs)
        return "local_skill_gap"

    monkeypatch.setattr(ri, "_call_classifier_llm", fake_call)

    ri.classify_career_intent(
        message="suggest me to improve",
        pending_recommender_offer="local_gap_coach",
        target_role_text="admin assistant",
        last_assistant_move="present_tiered_matches",
    )
    assert received["message"] == "suggest me to improve"
    assert received["pending_recommender_offer"] == "local_gap_coach"
    assert received["target_role_text"] == "admin assistant"
    assert received["last_assistant_move"] == "present_tiered_matches"


# ---------------------------------------------------------------------------
# User-block serialization
# ---------------------------------------------------------------------------
def test_user_block_serializes_all_four_fields():
    """The serialized user_block must include all four fields by name
    so the LLM can read them, with 'null' for unset values."""
    block = ri._build_user_block(
        message="what other careers fit me",
        pending_recommender_offer=None,
        target_role_text="admin",
        last_assistant_move=None,
    )
    assert "USER_MESSAGE: what other careers fit me" in block
    assert "PENDING_RECOMMENDER_OFFER: null" in block
    assert "TARGET_ROLE_TEXT: admin" in block
    assert "LAST_ASSISTANT_MOVE: null" in block


def test_user_block_lists_fields_on_separate_lines():
    """Each context field is on its own line so the LLM parses them
    deterministically."""
    block = ri._build_user_block(
        message="x", pending_recommender_offer=None,
        target_role_text=None, last_assistant_move=None,
    )
    assert block.count("\n") == 3  # 4 lines, 3 separators


# ---------------------------------------------------------------------------
# Tool schema invariants
# ---------------------------------------------------------------------------
def test_tool_schema_enum_matches_literal():
    """The tool schema's enum constraint must list exactly the seven
    intent values. Any drift means the tool can return values the
    Python Literal doesn't include."""
    schema_enum = set(ri._TOOL_SCHEMA["input_schema"]["properties"]["intent"]["enum"])
    assert schema_enum == ri._VALID_INTENTS


def test_tool_schema_intent_is_required():
    schema = ri._TOOL_SCHEMA["input_schema"]
    assert schema["required"] == ["intent"]
    assert schema["additionalProperties"] is False


def test_tool_name_constant_matches_schema():
    """_TOOL_NAME and the schema's name field must agree (used by the
    tool_choice param in the API call)."""
    assert ri._TOOL_NAME == ri._TOOL_SCHEMA["name"]


# ---------------------------------------------------------------------------
# Reset cache helper
# ---------------------------------------------------------------------------
def test_reset_cache_clears_entries(monkeypatch):
    monkeypatch.setattr(ri, "LLM_ENABLED", True)

    def fake_call(**kwargs):
        return "job_matching"

    monkeypatch.setattr(ri, "_call_classifier_llm", fake_call)

    ri.classify_career_intent(message="show me jobs")
    ri.classify_career_intent(message="show me admin")
    assert ri.cache_size() == 2
    ri.reset_cache()
    assert ri.cache_size() == 0
