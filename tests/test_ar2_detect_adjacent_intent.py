"""AR-2 tests: adjacency intent detector + anchor classifier.

Covers (per docs/adjacent-recommendations-design.md):
  - `detect_adjacent_intent`: pure terminal-state classifier
    - Explicit different-role phrasings -> AdjacentIntent.
    - Soft-offer affirmative replies -> AdjacentIntent
      (only when pending_offer=True; locked affirmative set).
    - Same-role gap phrasings -> None (R-3 owns them).
    - Ambiguous / unrelated -> None.
    - NeedsEvidenceIntent when the phrasing matched but the user
      has no usable evidence.
    - Defense in depth: substring leakage rejected at word
      boundaries.
  - `is_non_generic_transferable`: anchor classifier for AR-3
    - Generic skills (communication, leadership, etc.) rejected.
    - Customer service NOT generic (v4 lock).
    - Credentials rejected.
    - Off-source / low-confidence / malformed-numeric rejected.

All AR-2 modules are dead code; no production caller dispatches into
them until AR-6. These tests exercise the pure functions directly.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.nodb


from skillbridge.chat.adjacent_intent import (
    AdjacentIntent,
    NeedsEvidenceIntent,
    _AFFIRMATIVE_REPLIES,
    _normalize,
    detect_adjacent_intent,
)
from skillbridge.match.adjacent import (
    _GENERIC_SKILL_CANONICALS,
    is_non_generic_transferable,
)
from skillbridge.session.staging import StagedProfile, StagedSkill


# =========================================================================
# detect_adjacent_intent -- happy path
# =========================================================================
def _staged() -> StagedProfile:
    return StagedProfile.new("sess-1")


@pytest.mark.parametrize("phrase", [
    "what other jobs are there?",
    "What other roles could I look at?",
    "what other positions",
    "any other roles?",
    "any other jobs",
    "any other openings I should know about",
    "different roles I could try",
    "other postings worth considering",
    "show me other jobs",
    "show me other roles",
    "what am I close to in other roles?",
    "anything else I could do",
    "anything else I can do",
    "roles like this one",
    "any role like that",
])
def test_explicit_phrasing_with_evidence_returns_adjacent_intent(phrase) -> None:
    """Each clear different-role phrase must produce AdjacentIntent
    when the user has usable evidence."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert isinstance(result, AdjacentIntent), f"phrase={phrase!r} returned {result!r}"
    assert result.trigger == "user_explicit"


# =========================================================================
# detect_adjacent_intent -- NeedsEvidenceIntent
# =========================================================================
def test_explicit_phrase_without_evidence_returns_needs_evidence() -> None:
    result = detect_adjacent_intent(
        "what other jobs are there?",
        _staged(),
        user_has_evidence=False,
        pending_offer=False,
    )
    assert isinstance(result, NeedsEvidenceIntent)
    assert result.trigger == "user_explicit"


def test_affirmative_after_offer_without_evidence_returns_needs_evidence() -> None:
    """Even when the user accepts the soft offer, no evidence -> we
    can't recommend roles. Handler asks for skills instead."""
    result = detect_adjacent_intent(
        "yes please",
        _staged(),
        user_has_evidence=False,
        pending_offer=True,
    )
    assert isinstance(result, NeedsEvidenceIntent)
    assert result.trigger == "soft_offer_accepted"


# =========================================================================
# detect_adjacent_intent -- soft offer affirmative path
# =========================================================================
@pytest.mark.parametrize("reply", sorted(_AFFIRMATIVE_REPLIES))
def test_affirmative_after_offer_with_evidence_returns_adjacent_intent(reply) -> None:
    """Each member of the locked affirmative set, with pending_offer
    and evidence, produces AdjacentIntent."""
    result = detect_adjacent_intent(
        reply, _staged(),
        user_has_evidence=True,
        pending_offer=True,
    )
    assert isinstance(result, AdjacentIntent), f"reply={reply!r} returned {result!r}"
    assert result.trigger == "soft_offer_accepted"


def test_affirmative_normalization_strips_surrounding_punctuation() -> None:
    """'yes.' / 'YES!' / '  yes  ' should all normalize to 'yes'."""
    for reply in ["yes.", "YES!", "  yes  ", "yes,", "Yes"]:
        result = detect_adjacent_intent(
            reply, _staged(),
            user_has_evidence=True,
            pending_offer=True,
        )
        assert isinstance(result, AdjacentIntent), f"reply={reply!r} returned {result!r}"


@pytest.mark.parametrize("ambiguous", [
    "i guess?",
    "i guess so",
    "maybe",
    "hmm",
    "perhaps",
    "i don't know",
    "ok i guess why not",
    "if you must",
])
def test_ambiguous_after_offer_returns_none(ambiguous) -> None:
    """v11 QF lock: ambiguous post-offer replies fall through to None
    so the planner emits a clarification, not adjacency."""
    result = detect_adjacent_intent(
        ambiguous, _staged(),
        user_has_evidence=True,
        pending_offer=True,
    )
    assert result is None, f"ambiguous={ambiguous!r} returned {result!r}"


def test_affirmative_without_pending_offer_returns_none() -> None:
    """'yes' alone, with no prior soft offer, is meaningless. No
    adjacency."""
    result = detect_adjacent_intent(
        "yes", _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert result is None


# =========================================================================
# detect_adjacent_intent -- R-3 precedence + bare phrases
# =========================================================================
@pytest.mark.parametrize("phrase", [
    "what else do i need for this job",
    "what else for this role?",
    "what am i still missing?",
    "what do i need to add",
    "what do i need to do next",
    "what am i missing for this job?",
])
def test_same_role_gap_phrasings_return_none(phrase) -> None:
    """v11 trigger precedence: same-role gap phrasings belong to R-3
    (remaining-gaps). Adjacency must return None so R-3's dispatch
    handles them."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert result is None, f"phrase={phrase!r} returned {result!r}"


@pytest.mark.parametrize("phrase", [
    "what else?",
    "what now?",
    "okay",
    "alright",
    "tell me more",
    "hi",
    "hello",
])
def test_bare_phrases_return_none_when_no_offer_pending(phrase) -> None:
    """v11 lock: bare 'what else?' / 'okay' / etc. with no pending
    offer falls through to the planner."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert result is None, f"phrase={phrase!r} returned {result!r}"


# =========================================================================
# detect_adjacent_intent -- defense in depth
# =========================================================================
@pytest.mark.parametrize("phrase", [
    # Word-boundary safety: "other things" shouldn't match "other
    # jobs/roles/positions/openings".
    "what other things should i bring",
    "other ways i could improve",
    # "what else" alone shouldn't fire even with the word 'else'.
    "what else have you got",
    # "what other" alone (without jobs/roles/etc.) shouldn't fire.
    "what other reasons might that be",
])
def test_non_role_phrasings_return_none(phrase) -> None:
    """Conservative pattern set: the detector returns None on
    ambiguous adjacency-adjacent phrasings so the planner handles
    them."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert result is None, f"phrase={phrase!r} returned {result!r}"


def test_empty_and_whitespace_input_returns_none() -> None:
    for msg in ["", "   ", "\t\n", None]:
        result = detect_adjacent_intent(
            msg, _staged(),
            user_has_evidence=True,
            pending_offer=False,
        )
        assert result is None, f"msg={msg!r} returned {result!r}"


# =========================================================================
# Negation / decline guard (AR-2 round-2)
# =========================================================================
@pytest.mark.parametrize("phrase", [
    "I don't want other jobs",
    "I don't want other roles",
    "i do not want other jobs",
    "don't show me other roles",
    "do not show me other postings",
    "don't recommend other jobs",
    "no other jobs please",
    "no other roles please",
    "no other postings, thanks",
    "I am not looking for different roles",
    "I'm not looking for other jobs",
    "I'm not interested in other roles",
    "i am not interested in other postings",
    "not now",
    "maybe later",
    "no thanks",
    "No.",
    "no thank you",
])
def test_negation_decline_returns_none(phrase) -> None:
    """Explicit refusals must NOT fire AdjacentIntent even when they
    contain trigger tokens like 'other jobs' / 'other roles'."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert result is None, f"phrase={phrase!r} returned {result!r}"


def test_negation_blocks_even_pending_offer_path() -> None:
    """A user who replies 'no thanks' to the soft offer must NOT
    accidentally fire AdjacentIntent."""
    result = detect_adjacent_intent(
        "no thanks", _staged(),
        user_has_evidence=True,
        pending_offer=True,
    )
    assert result is None


# =========================================================================
# Cross-detector test -- R-3 must NOT steal AR-2's explicit phrasings
# =========================================================================
class _FakeRegistry:
    """Minimal registry stub for R-3 detection.  Mirrors the pattern at
    test_remaining_gaps_detection.py:_FakeRegistry."""

    def lookup(self, query):
        return None


def _r3_snapshot_with_credential_gap() -> dict:
    return {
        "captured_at_turn": 1,
        "lead_job": {
            "job_id": "job-x",
            "title": "Truck Tech",
            "employer": "ACME",
            "credential_gaps": [
                {"display": "Class G License", "canonical": "class g license"},
            ],
            "core_skill_gaps": [],
        },
        "other_jobs_meta": [],
    }


@pytest.mark.parametrize("phrase", [
    "what other jobs?",
    "what other roles?",
    "what other positions are available",
    "any other jobs?",
    "any other roles?",
    "anything else I could do?",
    "anything else I can do",
    "show me other jobs",
    "show me other roles",
    "different roles",
    "different jobs",
    "other postings",
    "roles like this one",
])
def test_r3_yields_to_adjacency_on_different_role_requests(phrase) -> None:
    """R-3's `_GENERIC_REMAINING` pattern would otherwise match these
    different-role discovery phrasings as same-role gap requests.
    The `_DIFFERENT_ROLE_REQUEST` exclusion in remaining_gaps.py keeps
    them in adjacency's lane.

    Asserts: `detect_remaining_gaps_intent` returns None on these
    messages even when a snapshot with credential gaps is present
    (which would normally trigger an R-3 subtract response)."""
    from skillbridge.chat.remaining_gaps import detect_remaining_gaps_intent

    intent = detect_remaining_gaps_intent(
        phrase,
        _r3_snapshot_with_credential_gap(),
        _FakeRegistry(),
        accumulated_credentials=[],
        pending_confirmation=None,
        last_discussed_canonical=None,
    )
    assert intent is None, (
        f"R-3 should yield to adjacency on {phrase!r} but returned "
        f"{intent!r}. Either _DIFFERENT_ROLE_REQUEST in remaining_gaps "
        f"is too narrow or _GENERIC_REMAINING is matching despite the "
        f"exclusion."
    )


def test_r3_still_fires_on_same_role_gap_phrasings() -> None:
    """Sanity: the `_DIFFERENT_ROLE_REQUEST` exclusion does NOT make
    R-3 a no-op on its own territory. Generic 'what else?' against a
    populated snapshot still triggers R-3's subtract path."""
    from skillbridge.chat.remaining_gaps import detect_remaining_gaps_intent

    intent = detect_remaining_gaps_intent(
        "what else?",
        _r3_snapshot_with_credential_gap(),
        _FakeRegistry(),
        accumulated_credentials=[],
        pending_confirmation=None,
        last_discussed_canonical=None,
    )
    assert intent is not None, "R-3 should still fire on bare 'what else?'"


# =========================================================================
# Compound semantics (AR-2 round-3) -- same-role anchor + nuanced negation
# =========================================================================
@pytest.mark.parametrize("phrase", [
    "anything else I could do for this job?",
    "anything else I can do with this role",
    "what other steps for this position",
    "anything else for this one",
    "what else could I do for the current job",
])
def test_same_role_anchor_yields_to_r3_in_adjacency_detector(phrase) -> None:
    """The same-role anchor ("for this job") overrides the explicit
    adjacency phrasing. AR-2 returns None so R-3 owns the message."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert result is None, f"phrase={phrase!r} returned {result!r}"


def test_same_role_anchor_routes_to_r3_in_cross_detector() -> None:
    """End-to-end: R-3's STEP 7 exclusion respects the same-role
    anchor. 'anything else I could do for this job?' must produce an
    R-3 intent (subtract or bootstrap) -- NOT yield to adjacency."""
    from skillbridge.chat.remaining_gaps import detect_remaining_gaps_intent

    intent = detect_remaining_gaps_intent(
        "anything else I could do for this job?",
        _r3_snapshot_with_credential_gap(),
        _FakeRegistry(),
        accumulated_credentials=[],
        pending_confirmation=None,
        last_discussed_canonical=None,
    )
    assert intent is not None, (
        "R-3 should fire on 'anything else I could do for this job?' -- "
        "the same-role anchor overrides the different-role exclusion."
    )


# --- Pure-decline + adjacency clause ---
@pytest.mark.parametrize("phrase", [
    "no, show me other jobs",
    "no, what other roles are there?",
    "no. show me other postings.",
    "no! any other jobs?",
])
def test_leading_pure_decline_blocks_adjacency_clause(phrase) -> None:
    """A leading bare 'no' (followed by punctuation) seals the intent
    as a refusal even if an adjacency-shaped clause follows. The
    leading 'no' is the user's primary signal -- the handler should
    not auto-route to adjacency on a contradictory utterance."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert result is None, f"phrase={phrase!r} returned {result!r}"


# --- Object-scoped decline (other roles are the object) ---
@pytest.mark.parametrize("phrase", [
    "I don't think other jobs are right for me",
    "I don't think other roles are a good fit",
    "I do not think other positions are right",
    "I'm not looking for other jobs",
    "I am not looking for different roles",
    "I don't want other jobs",
    "no other roles please",
    "don't recommend other jobs",
    "don't show me any other postings",
])
def test_other_role_object_decline_blocks_adjacency(phrase) -> None:
    """When OTHER roles are the explicit object of the negation, the
    detector must NOT fire AdjacentIntent."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert result is None, f"phrase={phrase!r} returned {result!r}"


# --- Current-role decline + adjacency pivot ---
@pytest.mark.parametrize("phrase", [
    # Object-style "don't want / like / need / care for"
    "I don't want this job, show me other roles",
    "I don't want this job — what other roles are there?",
    "I don't want this position. Show me other jobs.",
    "I don't like this role; any other jobs available?",
    "I don't want the current job, show me different roles",
    # Object-style "not interested in" / "am not interested in"
    "I'm not interested in this job, show me other roles",
    "I am not interested in this role, what other jobs are there?",
    "I am not interested in the current role; what other jobs are there?",
    # Object-style "don't see myself in"
    "I don't see myself in this role; any other jobs?",
    "I do not see myself in the current position. Other roles?",
    # Subject-style "this role isn't for me / a good fit / right for me"
    "This role isn't for me, show me other jobs",
    "This job is not for me; any other roles?",
    "The current role isn't a good fit -- what other jobs?",
    "this role is not right for me; show me other postings",
])
def test_current_role_decline_plus_adjacency_pivot_fires(phrase) -> None:
    """Compound semantics: when the user explicitly rejects THIS role
    (NOT 'other roles') AND asks for different roles in the same
    message, the explicit pivot wins -- AdjacentIntent fires.

    Covers both the original "don't want/like" object-style and the
    natural-language extensions:
        - "not interested in <role>"
        - "don't see myself in <role>"
        - subject-style "<role> isn't for me"
    """
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert isinstance(result, AdjacentIntent), (
        f"phrase={phrase!r} returned {result!r}; should be AdjacentIntent"
    )
    assert result.trigger == "user_explicit"


# --- Person-subject form ("I'm not a fit for this role") ---
@pytest.mark.parametrize("phrase", [
    "I'm not a fit for this role, show me other jobs",
    "I am not a fit for this position, what other roles are there?",
    "I'm not a good fit for the current role; any other jobs?",
    "I am not a good fit for this job. Show me other postings.",
])
def test_person_subject_fit_decline_pivot_fires(phrase) -> None:
    """The user negates THEIR OWN fit for the role -- 'I'm not a fit
    for this role' -- coupled with an other-role request. AR-2 must
    return AdjacentIntent."""
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=True,
        pending_offer=False,
    )
    assert isinstance(result, AdjacentIntent), (
        f"phrase={phrase!r} returned {result!r}; should be AdjacentIntent"
    )
    assert result.trigger == "user_explicit"


# --- Smart-apostrophe normalization ---
@pytest.mark.parametrize("phrase", [
    "I’m not a fit for this role, show me other jobs",
    "I’m not interested in this job; any other roles?",
    "I’m not looking for other jobs",  # object-scoped decline w/ smart '
    "I’m not a good fit for the current role; what other jobs?",
])
def test_smart_apostrophe_normalized(phrase) -> None:
    """Mobile autocorrect emits U+2019 (’) where users type `'`.
    The detector must fold smart apostrophes to straight ones before
    pattern matching so the same intent classifications apply."""
    user_has_evidence = True
    pending_offer = False
    result = detect_adjacent_intent(
        phrase, _staged(),
        user_has_evidence=user_has_evidence,
        pending_offer=pending_offer,
    )
    # The smart-apostrophe-folded equivalent of each phrase:
    folded = phrase.replace("’", "'")
    expected = detect_adjacent_intent(
        folded, _staged(),
        user_has_evidence=user_has_evidence,
        pending_offer=pending_offer,
    )
    assert type(result) is type(expected), (
        f"smart-apostrophe phrase {phrase!r} returned {result!r} but "
        f"straight-apostrophe equivalent {folded!r} returned {expected!r}"
    )


# --- R-3 cross-detector yields on current-role-decline pivots ---
@pytest.mark.parametrize("phrase", [
    "I am not interested in this role, what other jobs are there?",
    "I am not interested in the current role; what other jobs are there?",
    "I don't see myself in this role; any other jobs?",
    "I do not see myself in the current position. Any other roles?",
    "This role isn't for me, show me other jobs",
    "This job is not for me; any other roles?",
    # Person-subject form -- R-3 must also yield to adjacency here.
    "I'm not a fit for this role, show me other jobs",
    "I am not a good fit for the current position, what other jobs?",
    # Smart-apostrophe variant exercises the AR-2 fold in R-3.
    "I’m not a fit for this role; any other jobs?",
])
def test_r3_yields_on_current_role_decline_pivot(phrase) -> None:
    """End-to-end: R-3 must yield to adjacency when the message
    couples a current-role decline with an other-role request --
    even though the message contains a same-role anchor ('this role').
    The anchor lives INSIDE a current-role decline, so the user is
    pivoting away."""
    from skillbridge.chat.remaining_gaps import detect_remaining_gaps_intent

    intent = detect_remaining_gaps_intent(
        phrase,
        _r3_snapshot_with_credential_gap(),
        _FakeRegistry(),
        accumulated_credentials=[],
        pending_confirmation=None,
        last_discussed_canonical=None,
    )
    assert intent is None, (
        f"R-3 should yield to adjacency on {phrase!r} -- the same-role "
        f"anchor is inside a current-role decline. Got {intent!r}."
    )


def test_detector_is_pure_no_staged_mutation() -> None:
    """The detector must not mutate the staged profile -- it's pure."""
    sp = _staged()
    sp.skills = [
        StagedSkill(skill_name="welding", source="resume", confidence=0.7),
    ]
    sp.target_role_text = "welder"

    snapshot_before = (
        list(sp.skills), sp.target_role_text, sp.pending_adjacent_offer,
        sp.last_adjacent_snapshot,
    )
    _ = detect_adjacent_intent(
        "what other jobs?", sp,
        user_has_evidence=True, pending_offer=False,
    )
    snapshot_after = (
        list(sp.skills), sp.target_role_text, sp.pending_adjacent_offer,
        sp.last_adjacent_snapshot,
    )
    assert snapshot_before == snapshot_after


# =========================================================================
# _normalize helper
# =========================================================================
def test_normalize_lowercases_and_strips() -> None:
    assert _normalize("YES PLEASE!") == "yes please"
    assert _normalize("  Sure.  ") == "sure"
    assert _normalize("ok, ") == "ok"
    assert _normalize("yes\t please\n") == "yes please"


def test_normalize_empty_inputs() -> None:
    assert _normalize("") == ""
    assert _normalize(None) == ""
    assert _normalize("   ") == ""


# =========================================================================
# is_non_generic_transferable
# =========================================================================
def test_concrete_skill_passes_anchor_classifier() -> None:
    skill = StagedSkill(skill_name="welding", source="resume", confidence=0.7)
    assert is_non_generic_transferable(skill) is True


def test_customer_service_passes_anchor_classifier() -> None:
    """v4 lock: 'customer service' is CONCRETE evidence (SCCC retail /
    hospitality), NOT generic. Belongs to the anchor set."""
    skill = StagedSkill(skill_name="customer service", source="resume", confidence=0.7)
    assert is_non_generic_transferable(skill) is True


@pytest.mark.parametrize("generic", sorted(_GENERIC_SKILL_CANONICALS))
def test_generic_skill_rejected(generic) -> None:
    """Every member of the generic-skill set MUST be rejected."""
    skill = StagedSkill(skill_name=generic, source="resume", confidence=0.7)
    assert is_non_generic_transferable(skill) is False, (
        f"Generic skill {generic!r} unexpectedly passed the classifier."
    )


def test_generic_skill_case_variants_rejected() -> None:
    """Case variants canonicalize to the same key; the classifier
    catches them."""
    for variant in ["Communication", "TEAMWORK", "Leadership Skills"]:
        skill = StagedSkill(skill_name=variant, source="resume", confidence=0.7)
        assert is_non_generic_transferable(skill) is False, (
            f"Case variant {variant!r} slipped past the generic filter."
        )


def test_credential_rejected() -> None:
    skill = StagedSkill(skill_name="Class G License", source="resume", confidence=0.9)
    assert is_non_generic_transferable(skill) is False


def test_off_source_rejected() -> None:
    skill = StagedSkill(skill_name="welding", source="fallback", confidence=0.9)
    assert is_non_generic_transferable(skill) is False


def test_low_confidence_rejected() -> None:
    skill = StagedSkill(skill_name="welding", source="resume", confidence=0.5)
    assert is_non_generic_transferable(skill) is False


def test_at_confidence_floor_passes() -> None:
    skill = StagedSkill(skill_name="welding", source="resume", confidence=0.6)
    assert is_non_generic_transferable(skill) is True


def test_malformed_numeric_confidence_rejected() -> None:
    """The same _is_valid_normalized_score guard used by
    has_usable_skill_evidence: NaN / inf / booleans / out-of-range
    confidences are rejected."""
    for bad in [float("nan"), float("inf"), float("-inf"), True, 1.5, -0.1]:
        skill = StagedSkill(skill_name="welding", source="resume", confidence=bad)  # type: ignore[arg-type]
        assert is_non_generic_transferable(skill) is False, (
            f"confidence={bad!r} slipped past the validator."
        )


def test_empty_canonical_rejected() -> None:
    skill = StagedSkill(skill_name="", source="resume", confidence=0.7)
    assert is_non_generic_transferable(skill) is False


def test_non_staged_skill_rejected() -> None:
    """Defensive: a non-StagedSkill object must not crash."""
    assert is_non_generic_transferable({"skill_name": "welding"}) is False  # type: ignore[arg-type]
    assert is_non_generic_transferable(None) is False  # type: ignore[arg-type]


# =========================================================================
# AR-1c activation audit covers the no-production-caller invariant.
# AR-6a legitimately wires `detect_adjacent_intent` into
# `handler._try_adjacency_dispatch`; the per-name dead-helpers audit
# in test_ar1c_parity_and_activation.py tracks which names remain
# dead at each AR-6 sub-step boundary.
# =========================================================================
