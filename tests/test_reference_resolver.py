"""Unit tests for the deterministic reference resolver (Step 2.1).

DB-free, no LLM. Pure input/output tests on `resolve_reference`.

Locked rule set (see reference_resolver module docstring):
  1. no surface           -> no_reference
  2. single + pronoun     -> resolved
  3. ordinal in range     -> resolved
  4. full-label substring -> resolved (unique) / clarification (multi)
  5. multi + pronoun-only -> clarification
  6. otherwise            -> no_reference

Explicitly out of scope for the deterministic layer (tested as
`no_reference` here so the LLM fallback in step 2.3 picks them up):
  - prefix / partial name matching
  - fuzzy / Levenshtein near-misses
"""
from __future__ import annotations

import pytest

from skillbridge.chat.conversation_frame import SurfaceItem
from skillbridge.chat.reference_resolver import (
    ResolveOutcome,
    resolve_reference,
)


pytestmark = pytest.mark.nodb


def _role(label: str, noc: str, ordinal: int) -> SurfaceItem:
    return SurfaceItem(kind="role", label=label, id=noc, ordinal=ordinal)


def _job(label: str, job_id: str | None, ordinal: int) -> SurfaceItem:
    return SurfaceItem(kind="job", label=label, id=job_id, ordinal=ordinal)


_TWO_ROLES = (
    _role("Administrative assistant", "13110", 1),
    _role("Accounting clerk", "14200", 2),
)

_THREE_ROLES = (
    _role("Administrative assistant", "13110", 1),
    _role("Accounting clerk", "14200", 2),
    _role("Data entry clerk", "14400", 3),
)

_ONE_ROLE = (_role("Administrative assistant", "13110", 1),)


# ---------------------------------------------------------------- empty / defensive


class TestEmptyAndDefensive:
    def test_empty_surface_no_reference(self):
        out = resolve_reference("match me to that role", ())
        assert out.status == "no_reference"
        assert out.item is None
        assert out.reason == "no_surface"

    def test_empty_message_no_reference(self):
        out = resolve_reference("", _TWO_ROLES)
        assert out.status == "no_reference"
        assert out.reason == "empty_message"

    def test_whitespace_message_no_reference(self):
        out = resolve_reference("   \t\n  ", _TWO_ROLES)
        assert out.status == "no_reference"
        assert out.reason == "empty_message"

    @pytest.mark.parametrize("bad", [None, 123, 3.14, ["match"], {}])
    def test_non_string_no_reference(self, bad):
        out = resolve_reference(bad, _TWO_ROLES)
        assert out.status == "no_reference"
        assert out.reason == "non_string_message"


# ---------------------------------------------------------------- ordinals


class TestOrdinalWord:
    @pytest.mark.parametrize("message,index", [
        ("the first one", 1),
        ("first", 1),
        ("show me the second", 2),
        ("the third role", 3),
        ("second one please", 2),
    ])
    def test_word_ordinals_resolve(self, message, index):
        out = resolve_reference(message, _THREE_ROLES)
        assert out.status == "resolved"
        assert out.item is _THREE_ROLES[index - 1]
        assert out.reason == "ordinal"


class TestOrdinalDigit:
    @pytest.mark.parametrize("message,index", [
        ("1", 1),
        ("2", 2),
        ("the 1st", 1),
        ("the 2nd one", 2),
        ("3rd", 3),
        ("#2", 2),
    ])
    def test_digit_ordinals_resolve(self, message, index):
        out = resolve_reference(message, _THREE_ROLES)
        assert out.status == "resolved"
        assert out.item is _THREE_ROLES[index - 1]
        assert out.reason == "ordinal"


class TestOrdinalEdgeCases:
    def test_out_of_range_ordinal_returns_no_reference(self):
        """Locked: out-of-range ordinal is no_reference, not clarification.
        Caller decides how to respond; resolver doesn't invent copy."""
        out = resolve_reference("the fifth one", _TWO_ROLES)
        assert out.status == "no_reference"
        assert out.reason == "ordinal_out_of_range"

    def test_word_beats_digit_when_both_present(self):
        """Deterministic tie-break: word ordinal wins over stray digit."""
        out = resolve_reference("the first (option 2)", _THREE_ROLES)
        assert out.status == "resolved"
        assert out.item is _THREE_ROLES[0]

    def test_secondary_does_not_match_second(self):
        """Word boundary discipline: 'secondary' must not fire the
        'second' ordinal pattern."""
        out = resolve_reference("secondary school", _THREE_ROLES)
        # No ordinal, no pronoun, no label match -> no_reference.
        assert out.status == "no_reference"


# ---------------------------------------------------------------- pronouns


class TestSingleItemPronoun:
    @pytest.mark.parametrize("message", [
        "that",
        "it",
        "that one",
        "that role",
        "this one",
        "this role",
        "tell me about that",
        "match me to it",
    ])
    def test_single_item_with_pronoun_resolves(self, message):
        out = resolve_reference(message, _ONE_ROLE)
        assert out.status == "resolved"
        assert out.item is _ONE_ROLE[0]
        assert out.reason == "single_item_pronoun"


class TestMultiItemPronounOnly:
    @pytest.mark.parametrize("message", [
        "that",
        "it",
        "that one",
        "that role",
        "match me to it",
        "tell me about that",
    ])
    def test_multi_item_pronoun_only_asks_clarification(self, message):
        out = resolve_reference(message, _THREE_ROLES)
        assert out.status == "clarification"
        assert out.item is None
        assert out.reason == "multi_item_pronoun"


# ---------------------------------------------------------------- label match


class TestLabelMatchUnique:
    @pytest.mark.parametrize("message", [
        "Administrative assistant",
        "match me to Administrative assistant",
        "show me jobs for administrative assistant",
        "administrative assistant please",
        # Case-insensitive.
        "ADMINISTRATIVE ASSISTANT",
        "administrative Assistant",
    ])
    def test_full_label_substring_resolves_uniquely(self, message):
        out = resolve_reference(message, _THREE_ROLES)
        assert out.status == "resolved"
        assert out.item.label == "Administrative assistant"
        assert out.reason == "label_match_unique"

    def test_label_match_wins_over_pronoun(self):
        """When a label match is present, resolver returns that item
        even if the message also contains a pronoun. Explicit name
        beats vague pronoun."""
        out = resolve_reference(
            "match me to that Accounting clerk", _THREE_ROLES,
        )
        assert out.status == "resolved"
        assert out.item.label == "Accounting clerk"
        assert out.reason == "label_match_unique"


class TestLabelMatchAmbiguous:
    def test_multiple_labels_in_message_ask_clarification(self):
        """User named two roles at once. Not a resolvable reference --
        ask which."""
        out = resolve_reference(
            "should I go for Administrative assistant or Accounting clerk?",
            _THREE_ROLES,
        )
        assert out.status == "clarification"
        assert out.item is None
        assert out.reason == "label_match_ambiguous"


class TestPartialAndFuzzyOutOfScope:
    """Locked: no prefix, no fuzzy in the deterministic layer. These
    tests pin the boundary so a future 'let's just add prefix'
    change trips loudly. LLM fallback (step 2.3) will pick these up."""

    @pytest.mark.parametrize("message", [
        # Common near-miss / abbreviation the user might type.
        "admin secretary",       # not a label of any item
        "admin assistant",       # prefix of "Administrative assistant"
        "accounting",            # prefix of "Accounting clerk"
        "administrative",        # prefix of "Administrative assistant"
    ])
    def test_partial_and_near_miss_are_no_reference(self, message):
        out = resolve_reference(message, _THREE_ROLES)
        # Deterministic layer says nothing -- LLM fallback (step 2.3)
        # handles near-misses. This test guards the boundary.
        assert out.status == "no_reference"


# ---------------------------------------------------------------- structural


class TestFrozenResult:
    def test_outcome_is_frozen(self):
        out = resolve_reference("first", _ONE_ROLE)
        with pytest.raises((AttributeError, TypeError)):
            out.status = "clarification"  # type: ignore[misc]


class TestJobSurfaceStillResolves:
    """The resolver is kind-agnostic -- it can resolve references
    against any SurfaceItem tuple, including a matching-engine
    'matches' surface. Kind-based routing (role -> handoff, job ->
    different behavior) lives at the handoff helper (step 2.5), not
    here."""

    def test_job_kind_ordinal_resolves(self):
        jobs = (
            _job("Truck driver at Acme", "j1", 1),
            _job("Delivery driver at Sault Co", "j2", 2),
        )
        out = resolve_reference("the second one", jobs)
        assert out.status == "resolved"
        assert out.item.kind == "job"
        assert out.item.id == "j2"


# ---------------------------------------------------------------- malformed items


class TestMalformedSurfaceEntries:
    def test_blank_label_item_never_matches_by_substring(self):
        """A malformed surface entry with an empty label should never
        substring-match against any user message."""
        items = (
            _role("", "13110", 1),
            _role("Accounting clerk", "14200", 2),
        )
        out = resolve_reference("show me jobs", items)
        # No pronoun, no ordinal, no label match on non-blank label.
        assert out.status == "no_reference"

    def test_ordinal_still_works_when_earlier_item_has_blank_label(self):
        items = (
            _role("", "13110", 1),
            _role("Accounting clerk", "14200", 2),
        )
        out = resolve_reference("the second", items)
        assert out.status == "resolved"
        assert out.item.label == "Accounting clerk"
