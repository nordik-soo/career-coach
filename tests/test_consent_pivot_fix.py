"""Unit tests for the pivot-intent short-circuit in the consent classifier
(slice 1 follow-up, 2026-07-03).

The bug: `_classify_pattern_2_reply` used to return "yes" on messages
like "yes, but show me jobs" because `_classify_intent` classifies the
whole message as one intent (impatient_proceed) and the whole-message
intent won the verdict. Downstream, this caused the Pattern 2 blanking
hook (handler.py:3675) to suppress the direct-target job tiers -- the
user asked for jobs and got related-roles-only.

Same swallowing hit all three consume paths because
`_classify_recommender_consent` wraps `_classify_pattern_2_reply`.

Path A fix (locked with lead 2026-07-03): keep the output enum closed
at yes|no|other. Add a pre-check that returns "other" when the message
names an explicit engine-level pivot, EVEN IF a yes/no signal is also
present. Downstream "other" semantics: consume returns None, message
falls through to router, router's Step 1.3 pivot-clear closes the
stale flag when the router resolves the explicit intent.

DB-free, no LLM.
"""
from __future__ import annotations

import pytest

from skillbridge.chat.handler import (
    _classify_pattern_2_reply,
    _classify_recommender_consent,
    _has_pivot_intent,
)


pytestmark = pytest.mark.nodb


# ---------------------------------------------------------------- must become other


@pytest.mark.parametrize("message", [
    # User's locked list of phrasings that must classify as "other".
    "yes, but show me jobs",
    "yes but show me jobs",
    "yes but also match me",
    "yes and show admin jobs",
    "show me jobs",
    "no thanks show jobs",
    "actually show me jobs",
    "match me instead",
    "what training should I take",
    "compare me to NOC standard",
    # Slice-1 scenario 5's exact string (was safe today via "instead"
    # falling to neutral; assert it stays safe after the fix).
    "show me admin jobs instead",
    # Case-variation coverage.
    "YES, BUT SHOW ME JOBS",
    "Yes but Show me jobs",
])
def test_pivot_phrasings_classify_as_other(message):
    assert _classify_pattern_2_reply(message) == "other", (
        f"pivot phrasing must classify as other: {message!r}"
    )


# ---------------------------------------------------------------- must remain yes


@pytest.mark.parametrize("message", [
    # User's locked "must-remain-yes" list.
    "yes",
    "yes please",
    "yes go ahead",
    "sure",
    "sounds good",
    "ok",
    # Bonus regressions covering the existing Pattern 2 test suite's
    # yes-list -- must NOT regress after this change.
    "alright",
    "looks right",
    "looks good",
    "that's right",
    "yep",
    "yeah",
    "okay",
    "go ahead",
    "let's go",
    "yes. go ahead",   # the 2026-06-17 live-repro that broke v1
    "yes, go ahead",
])
def test_legit_yes_still_classifies_as_yes(message):
    assert _classify_pattern_2_reply(message) == "yes", (
        f"legit yes must still classify as yes: {message!r}"
    )


# ---------------------------------------------------------------- must remain no


@pytest.mark.parametrize("message", [
    # User's locked "must-remain-no" list.
    "no thanks",
    "not now",
    "skip it",
    # Bonus regressions covering the existing Pattern 2 no-list --
    # must NOT regress after this change.
    "no thank you",
    "not today",
    "not interested",
    "skip",
    "nope",
    "I don't want",
    "i do not want",
])
def test_legit_no_still_classifies_as_no(message):
    assert _classify_pattern_2_reply(message) == "no", (
        f"legit no must still classify as no: {message!r}"
    )


# ---------------------------------------------------------------- pivot detector


class TestHasPivotIntent:
    """Direct unit tests on the pivot-detector helper."""

    @pytest.mark.parametrize("message", [
        "show me jobs",
        "show me admin jobs",
        "show admin jobs",
        "find me a job",
        "find me some work",
        "get me a job",
        "match me",
        "match me to programming",
        "what training should I take",
        "what course do I need",
        "what certifications would help",
        "what should I learn",
        "what should I improve",
        "compare me to NOC",
        "compare to the standard",
        "NOC standard",
        "what else can I do",
        "other careers",
    ])
    def test_positive_matches(self, message):
        assert _has_pivot_intent(message) is True

    @pytest.mark.parametrize("message", [
        # Legit yes replies -- must NOT match as pivots.
        "yes",
        "yes please",
        "yes go ahead",
        "sure",
        "sounds good",
        "ok",
        "yes look at related roles",  # Pattern 2's own offer wording
        "sure show me those",         # ambiguous, but no engine noun
        # Legit no replies.
        "no thanks",
        "not now",
        "skip it",
        # Other neutral-but-not-pivot messages.
        "tell me more",
        "the second one",
        "what about the first",
        "how do I get my Class G?",
    ])
    def test_negative_matches(self, message):
        assert _has_pivot_intent(message) is False

    @pytest.mark.parametrize("bad", [None, 123, 3.14, ["show me jobs"]])
    def test_non_string_inputs_return_false(self, bad):
        assert _has_pivot_intent(bad) is False

    def test_empty_and_whitespace_return_false(self):
        assert _has_pivot_intent("") is False
        assert _has_pivot_intent("   ") is False


# ---------------------------------------------------------------- wrapper inheritance


class TestRecommenderConsentInheritsPivotFix:
    """`_classify_recommender_consent` wraps `_classify_pattern_2_reply`.
    The pivot fix must therefore propagate to the recommender consume
    + drilldown consume paths automatically -- no per-site changes."""

    @pytest.mark.parametrize("message", [
        "yes, but show me jobs",
        "yes but also match me",
        "show me jobs",
        "what training should I take",
        "match me instead",
    ])
    def test_pivot_phrases_classify_as_other_via_wrapper(self, message):
        assert _classify_recommender_consent(message) == "other"

    @pytest.mark.parametrize("message", [
        "yes",
        "yes go ahead",
        "sure",
        "yes. go ahead",
    ])
    def test_legit_yes_still_yes_via_wrapper(self, message):
        assert _classify_recommender_consent(message) == "yes"


# ---------------------------------------------------------------- specific bug repros


class TestSpecificBugRepros:
    """Concrete failure modes from the 2026-07-03 preflight, pinned as
    named tests so a regression here is obvious in CI output."""

    def test_pattern_2_yes_with_pivot_no_longer_hijacks(self):
        """The primary bug: 'yes, but show me jobs' used to return 'yes',
        which triggered the Pattern 2 blanking hook at handler.py:3675,
        clearing direct-target tiers and forcing the surface to
        related-roles-only. Fix returns 'other' -> consume returns None
        -> router picks up -> Step 1.3 pivot-clear closes the stale flag."""
        assert _classify_pattern_2_reply("yes, but show me jobs") == "other"

    def test_no_with_pivot_no_longer_hijacks(self):
        """Adjacent bug surfaced by the preflight: 'no thanks show jobs'
        used to return 'yes' (impatient_proceed pattern won over the
        earlier 'no thanks'). The user is declining AND pivoting;
        classifying as 'other' preserves both signals via fallthrough
        instead of firing the yes-consent blanking hook."""
        assert _classify_pattern_2_reply("no thanks show jobs") == "other"

    def test_recommender_consume_pivot_no_longer_dispatches_stale_chain(
        self,
    ):
        """The bigger finding: `_classify_recommender_consent` wraps the
        same classifier, so 'yes but also match me' during a live
        recommender chain used to fire the yes-dispatch branch of
        _dispatch_recommender_consume -- user gets Layer B/C output
        instead of their requested matches. Fix: returns 'other' via
        the wrapper -> consume returns None -> router pivot-clear
        catches the stale offer."""
        assert _classify_recommender_consent("yes but also match me") == "other"
