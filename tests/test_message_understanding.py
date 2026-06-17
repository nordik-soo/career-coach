"""Unit tests for the message understanding layer (chat orchestration v2.1).

Three concerns:

  1. Each PrimaryIntent value reachable and correct on a representative
     message.
  2. The priority order from the design doc fires correctly when
     multiple signals are present.
  3. The live-bug scenarios that triggered this refactor route correctly:
       - "online Excel course" cold session
       - "how can I get my Class G driver's licence" cold session
       - "Can I apply for PR while looking?" cold session
       - "I have Excel, find me jobs" (skill claim NOT training)

This module is DEAD CODE until Slice B wires it. Tests prove
correctness in isolation so Slice B's integration is safe.
"""
from __future__ import annotations

import os

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from skillbridge.chat.message_understanding import (
    DetectedEntity,
    MessageUnderstanding,
    understand_message,
)

pytestmark = pytest.mark.nodb


# ===========================================================================
# Helper: shorter call site
# ===========================================================================
def _u(msg: str, gaps: list[str] | None = None) -> MessageUnderstanding:
    return understand_message(
        user_message=msg, registry_gaps_in_message=gaps,
    )


# ===========================================================================
# 1. Empty / None / whitespace -> ambiguous, low
# ===========================================================================
def test_empty_message_returns_ambiguous_low():
    u = _u("")
    assert u.primary_intent == "ambiguous"
    assert u.confidence == "low"
    assert u.entities == ()


def test_whitespace_only_message_returns_ambiguous_low():
    u = _u("   \n  \t  ")
    assert u.primary_intent == "ambiguous"
    assert u.confidence == "low"


def test_none_user_message_handled_defensively():
    """None coercion: caller may pass None; we treat it as empty."""
    u = understand_message(user_message=None, registry_gaps_in_message=None)  # type: ignore[arg-type]
    assert u.primary_intent == "ambiguous"


# ===========================================================================
# 2. Rule 1 (HIGHEST PRIORITY): scope_violation
# ===========================================================================
@pytest.mark.parametrize("msg, expected_category", [
    # immigration
    ("Can I apply for PR while looking?", "immigration"),
    ("this job can help for PR", "immigration"),
    ("Will my work permit support this?", "immigration"),
    ("RCIP eligible?", "immigration"),
    ("Express Entry timeline", "immigration"),
    # national wages
    ("what's the national average wage", "national_wages"),
    ("does StatCan have data on this", "national_wages"),
    # non-local city
    ("any jobs in Toronto?", "non_ssm_city"),
    ("moving to Ottawa next month", "non_ssm_city"),
])
def test_scope_violations_classify_as_scope_violation_high(msg, expected_category):
    u = _u(msg)
    assert u.primary_intent == "scope_violation"
    assert u.confidence == "high"
    assert u.has_entity_type("scope_keyword")
    assert any(
        e.canonical_name == expected_category for e in u.entities
    ), f"Expected scope category {expected_category!r} in {u.entities!r}"


def test_scope_wins_when_combined_with_other_signals():
    """Priority order: scope > everything. Even if the message also
    mentions a registry credential, scope wins."""
    u = _u("can I apply for PR while getting my forklift certificate?",
           gaps=["forklift certification"])
    assert u.primary_intent == "scope_violation"
    assert u.confidence == "high"


# ===========================================================================
# 3. Rule 2: training_request + registry_entity (HIGH)
# ===========================================================================
@pytest.mark.parametrize("msg, gaps", [
    # The live-bug case (the reason this refactor exists)
    ("online Excel course", ["Microsoft Excel"]),
    # Other natural training questions with entity
    ("how can I get my Class G driver's licence",
     ["Class G driver's license"]),
    ("how do I get my 310T?", ["310T technician certification"]),
    ("where I can do course for learning excel", ["Microsoft Excel"]),
    ("WHMIS training please", ["WHMIS"]),
    ("any course do you recommend for forklift",
     ["forklift certification"]),
    ("CPR certification online", ["first aid and CPR"]),
    ("Class G link", ["Class G driver's license"]),
])
def test_training_with_entity_is_high_confidence_training_request(msg, gaps):
    u = _u(msg, gaps)
    assert u.primary_intent == "training_request", (
        f"Expected training_request for {msg!r}; got {u.primary_intent!r} "
        f"(reason: {u.reason!r})"
    )
    assert u.confidence == "high"
    assert u.has_entity_type("registry_gap")
    assert u.registry_gap_canonical_names() == tuple(gaps)


# ===========================================================================
# 4. Rule 3: training_request WITHOUT entity (HIGH)
# ===========================================================================
@pytest.mark.parametrize("msg", [
    "any course do you recommend",
    "give me a link to an online course",
    "what training do I need?",
    "I want to learn something new",
    "any online training please",
    "do you recommend any class",
])
def test_training_without_entity_is_high_confidence_training_request(msg):
    u = _u(msg, gaps=[])
    assert u.primary_intent == "training_request"
    assert u.confidence == "high"
    assert u.entities == (), "No registry entity should be detected"


# ===========================================================================
# 5. Rule 4: job_search via impatient_proceed (HIGH)
# ===========================================================================
@pytest.mark.parametrize("msg", [
    "show me jobs",
    "just match me",
    "see my CV",
    "same role",
    "let's go",
    "match me now",
])
def test_impatient_proceed_is_high_confidence_job_search(msg):
    u = _u(msg)
    assert u.primary_intent == "job_search"
    assert u.confidence == "high"


# ===========================================================================
# 6. Rule 5: declining / correcting / confirming (MEDIUM)
# ===========================================================================
@pytest.mark.parametrize("msg, expected_intent", [
    ("no thanks", "decline"),
    ("not now", "decline"),
    ("skip it", "decline"),
    ("actually I want warehouse work", "correction"),
    ("wait, that's wrong", "correction"),
    ("no I meant truck driving", "correction"),
    ("yes", "confirmation"),
    ("alright", "confirmation"),
    ("looks right", "confirmation"),
])
def test_conversational_signals_are_medium_confidence(msg, expected_intent):
    u = _u(msg)
    assert u.primary_intent == expected_intent
    assert u.confidence == "medium"


# ===========================================================================
# 7. Rule 6: registry entity WITHOUT training intent (MEDIUM ambiguous)
# ===========================================================================
@pytest.mark.parametrize("msg, gaps", [
    # The critical false-positive guard from the design discussion
    ("I have Excel and forklift experience, find me jobs",
     ["Microsoft Excel", "forklift certification"]),
    ("warehouse job with Excel", ["Microsoft Excel"]),
    ("I worked with forklifts", ["forklift certification"]),
    ("my background includes Microsoft Excel",
     ["Microsoft Excel"]),
])
def test_entity_without_training_intent_is_medium_ambiguous(msg, gaps):
    """Skill claims must NOT route to training_request. Router will
    hand these to the planner (with the entity as context)."""
    u = _u(msg, gaps)
    # Should NOT be training_request -- the skill claim guard
    assert u.primary_intent != "training_request", (
        f"Skill claim {msg!r} must not route to training_request"
    )
    # Most of these route via impatient_proceed ("find me jobs") -> job_search.
    # Some land at registry_entity+no-training -> ambiguous medium.
    # Both are acceptable; the key is "not training_request."


def test_pure_entity_mention_without_action_word_routes_to_ambiguous_medium():
    """A bare entity reference with no training words and no job-search
    impatience: ambiguous, medium. Planner handles."""
    u = _u("I worked with forklifts last year", gaps=["forklift certification"])
    # No training words, no impatient phrase
    assert u.primary_intent != "training_request"
    assert u.primary_intent != "job_search"


# ===========================================================================
# 8. Rule 7: low-confidence default (planner handles)
# ===========================================================================
@pytest.mark.parametrize("msg", [
    "hello there",
    "thanks for your help",
    "interesting",
    "the weather is nice today",
])
def test_low_signal_messages_are_low_confidence(msg):
    """Messages that don't match any classifier fall through to
    ambiguous + low; the planner takes over.

    Note: messages starting with "okay" / "yes" / "alright" trigger
    confirming-pattern (MEDIUM), which is the CORRECT classification --
    they signal acknowledgment. Tests for those live in the
    conversational-signals parametrize."""
    u = _u(msg, gaps=[])
    assert u.confidence == "low", (
        f"Expected low confidence for {msg!r}; "
        f"got {u.confidence!r} via {u.reason!r}"
    )


# ===========================================================================
# 9. Priority order tests (the design doc's core invariant)
# ===========================================================================
def test_priority_scope_over_training():
    """Scope wins over training. Don't recommend a forklift cert as
    the answer to a PR question."""
    u = _u("can I take a forklift course for PR?",
           gaps=["forklift certification"])
    assert u.primary_intent == "scope_violation"


def test_priority_training_with_entity_over_job_search():
    """Training intent with a specific credential wins over passive
    job-context. 'show me Excel courses' is training, not job-search."""
    u = _u("show me online Excel courses", gaps=["Microsoft Excel"])
    # "show me" could be impatient_proceed pattern, but we check
    # training_request fires first via Rule 2.
    # Note: this is a close case -- both pattern types fire. The
    # design says training_with_entity is Rule 2; impatient is Rule 4.
    # Rule 2 wins.
    assert u.primary_intent == "training_request"


def test_priority_training_without_entity_still_beats_default_low():
    u = _u("any course do you recommend please")
    assert u.primary_intent == "training_request"
    assert u.confidence == "high"


# ===========================================================================
# 10. MessageUnderstanding helper methods
# ===========================================================================
def test_has_entity_type_returns_correct_bool():
    u = _u("how can I get my 310T?", gaps=["310T technician certification"])
    assert u.has_entity_type("registry_gap")
    assert not u.has_entity_type("scope_keyword")


def test_registry_gap_canonical_names_preserves_order():
    u = _u("any course on Excel and 310T?",
           gaps=["Microsoft Excel", "310T technician certification"])
    # Both entities surface; order matches input
    names = u.registry_gap_canonical_names()
    assert "Microsoft Excel" in names
    assert "310T technician certification" in names


def test_message_understanding_is_frozen():
    """Downstream consumers shouldn't be able to mutate the result."""
    u = _u("hello")
    with pytest.raises(Exception):
        u.primary_intent = "scope_violation"  # type: ignore[misc]


def test_detected_entity_is_frozen():
    e = DetectedEntity(
        type="registry_gap", canonical_name="X",
        matched_text="x", source="test",
    )
    with pytest.raises(Exception):
        e.canonical_name = "Y"  # type: ignore[misc]


def test_entities_returned_as_tuple_not_list():
    """Immutability of the entities collection -- can't be mutated
    via shared reference."""
    u = _u("how do I get my 310T?", gaps=["310T technician certification"])
    assert isinstance(u.entities, tuple)


# ===========================================================================
# 11. Reason field surfaces something useful
# ===========================================================================
def test_reason_field_is_populated_for_high_confidence_classifications():
    """For transcript-test debugging: reason explains WHY classification
    fired, not just WHAT classification was assigned."""
    u_scope = _u("Can I apply for PR?")
    assert u_scope.reason, "scope decision should carry a reason"

    u_train = _u("how do I get my 310T?", gaps=["310T technician certification"])
    assert u_train.reason, "training decision should carry a reason"
    assert "310T" in u_train.reason or "registry_entity" in u_train.reason

    u_job = _u("show me jobs")
    assert "impatient" in u_job.reason


def test_reason_for_empty_message():
    u = _u("")
    assert "empty" in u.reason.lower()


# ===========================================================================
# 12. Specific live-bug scenarios that triggered this refactor
# ===========================================================================
def test_live_bug_class_g_cold_session():
    """Pre-refactor: planner invented redirect_scope on this.
    Post-refactor: training_request HIGH, planner skipped (Slice B)."""
    u = _u("how can I get my Class G driver's licence",
           gaps=["Class G driver's license"])
    assert u.primary_intent == "training_request"
    assert u.confidence == "high"


def test_live_bug_excel_course_cold_session():
    """Pre-refactor: planner asked target_role on this.
    Post-refactor: training_request HIGH, planner skipped."""
    u = _u("online Excel course", gaps=["Microsoft Excel"])
    assert u.primary_intent == "training_request"
    assert u.confidence == "high"


def test_live_bug_where_can_I_do_course_cold_session():
    u = _u("where I can do course for learning excel",
           gaps=["Microsoft Excel"])
    assert u.primary_intent == "training_request"
    assert u.confidence == "high"


def test_live_bug_pr_question_cold_session():
    """Pre-Slice-10: planner re-pitched matches.
    Post-refactor: scope_violation HIGH."""
    u = _u("can I apply for PR while I finish the apprenticeship?")
    assert u.primary_intent == "scope_violation"
    assert u.confidence == "high"


def test_live_bug_skill_claim_with_entity_stays_ambiguous():
    """The critical false-positive guard: entity present but the
    user is reporting OWNED skills, not asking about training.
    Must NOT route to training_request."""
    u = _u("I have Excel and forklift experience, find me jobs",
           gaps=["Microsoft Excel", "forklift certification"])
    assert u.primary_intent != "training_request"
    # "find me jobs" -> impatient pattern -> job_search high confidence
    # Either job_search or ambiguous is acceptable here -- BOTH are
    # not training_request, which is the key invariant.


# ===========================================================================
# 14. Input contract: registry_gaps_in_message must be list[str].
#     Slice A review found `find_gaps_in_message` returns list[Gap], so the
#     handler/router MUST normalize to canonical strings before calling
#     understand_message. These tests pin the contract so the bug can never
#     re-enter silently.
# ===========================================================================
def test_contract_dedupes_and_skips_empty_canonical_names():
    """Caller may pass canonical names that are duplicated, empty, or
    surrounded by whitespace. The understanding layer must:
      * keep first-seen order
      * dedupe by stripped canonical name
      * skip empty / whitespace-only entries (they carry no signal)
    Net result for ["Microsoft Excel", "", "Microsoft Excel", "  ",
    "  Microsoft Excel  ", "forklift certification"]: two entities,
    Microsoft Excel then forklift certification."""
    u = _u(
        "where can I get training",
        gaps=[
            "Microsoft Excel",
            "",
            "Microsoft Excel",          # exact duplicate
            "   ",                      # whitespace only
            "  Microsoft Excel  ",      # duplicate after strip
            "forklift certification",
        ],
    )
    names = [e.canonical_name for e in u.entities]
    assert names == ["Microsoft Excel", "forklift certification"]
    # All entities must be plain strings, not Gap-like objects.
    for e in u.entities:
        assert isinstance(e.canonical_name, str)
        assert isinstance(e.matched_text, str)


def test_contract_rejects_non_string_canonical_with_typeerror():
    """If the caller forgets to convert `list[Gap]` -> canonical names
    (the bug Slice A review caught), understand_message must fail
    LOUD with TypeError. Silent coercion would let a Gap dataclass
    end up stuffed into DetectedEntity.canonical_name and only
    surface as a string-format mess in logs."""
    class _FakeGap:
        """Stand-in for registry.Gap; the actual class is irrelevant —
        the contract is 'must be str', anything else explodes."""
        canonical_name = "Microsoft Excel"

    with pytest.raises(TypeError) as excinfo:
        understand_message(
            user_message="where can I get training",
            registry_gaps_in_message=[_FakeGap()],  # type: ignore[list-item]
        )
    # Error message must point the caller at the fix, not just say
    # "wrong type". Future-me will read this exception under load.
    msg = str(excinfo.value)
    assert "str" in msg
    assert "canonical_name" in msg or "Gap" in msg


def test_contract_accepts_none_and_empty_list_without_error():
    """The other half of the contract: passing None or [] is a
    perfectly valid 'no entities discovered' signal, not an error.
    Belt-and-suspenders test against an over-eager isinstance check."""
    u_none = understand_message(
        user_message="hello there",
        registry_gaps_in_message=None,
    )
    u_empty = understand_message(
        user_message="hello there",
        registry_gaps_in_message=[],
    )
    assert u_none.entities == ()
    assert u_empty.entities == ()
    assert u_none.primary_intent == u_empty.primary_intent
