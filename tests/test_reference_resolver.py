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
    build_clarification_prompt,
    llm_cache_size,
    reset_llm_cache,
    resolve_reference,
    resolve_reference_via_llm,
    resolve_reference_with_fallback,
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


# ---------------------------------------------------------------- clarification prompt


class TestClarificationPromptRendering:
    """Locked shape (Step 2.2, 2026-07-03):
      - Em dash (U+2014) between lead-in and list.
      - Oxford comma before the final 'or' for 3+ item lists.
      - 2-item list uses plain 'A or B' (no serial comma).
      - Blank labels filtered defensively.
    """

    def test_two_items_uses_or_without_serial_comma(self):
        prompt = build_clarification_prompt(_TWO_ROLES)
        assert prompt == (
            "Which one do you mean — "
            "Administrative assistant or Accounting clerk?"
        )

    def test_three_items_uses_oxford_comma_before_final_or(self):
        prompt = build_clarification_prompt(_THREE_ROLES)
        assert prompt == (
            "Which one do you mean — "
            "Administrative assistant, "
            "Accounting clerk, "
            "or Data entry clerk?"
        )

    def test_four_items_uses_oxford_comma(self):
        items = _THREE_ROLES + (_role("Bookkeeper", "12200", 4),)
        prompt = build_clarification_prompt(items)
        assert prompt == (
            "Which one do you mean — "
            "Administrative assistant, "
            "Accounting clerk, "
            "Data entry clerk, "
            "or Bookkeeper?"
        )

    def test_em_dash_is_unicode_u2014(self):
        prompt = build_clarification_prompt(_TWO_ROLES)
        assert "—" in prompt   # em dash
        assert " - " not in prompt  # not a plain hyphen with spaces
        assert "--" not in prompt   # not a double hyphen

    def test_blank_labels_filtered_defensively(self):
        items = (
            _role("Administrative assistant", "13110", 1),
            _role("", "14200", 2),
            _role("   ", "14400", 3),  # whitespace-only
            _role("Data entry clerk", "14500", 4),
        )
        prompt = build_clarification_prompt(items)
        # Only two labels remained after filtering; uses plain "or"
        # form since 2 usable labels.
        assert prompt == (
            "Which one do you mean — "
            "Administrative assistant or Data entry clerk?"
        )

    def test_single_item_after_filter_returns_empty(self):
        """The resolver never emits clarification with < 2 items, so
        this is defensive. Return empty string so a caller can treat
        it as 'nothing to ask; fall through'."""
        items = (
            _role("Administrative assistant", "13110", 1),
            _role("", "14200", 2),
        )
        assert build_clarification_prompt(items) == ""

    def test_all_blank_items_returns_empty(self):
        items = (
            _role("", "13110", 1),
            _role("  ", "14200", 2),
        )
        assert build_clarification_prompt(items) == ""

    def test_empty_input_returns_empty(self):
        assert build_clarification_prompt(()) == ""

    def test_labels_are_preserved_verbatim(self):
        """No lowercasing / no title-casing / no rewording. Coach
        voice already lives upstream; this helper only formats."""
        items = (
            _role("ADMIN. ASST.", "13110", 1),
            _role("Data-entry clerk (I)", "14400", 2),
        )
        prompt = build_clarification_prompt(items)
        assert "ADMIN. ASST." in prompt
        assert "Data-entry clerk (I)" in prompt

    def test_prompt_ends_with_question_mark(self):
        assert build_clarification_prompt(_TWO_ROLES).endswith("?")
        assert build_clarification_prompt(_THREE_ROLES).endswith("?")

    def test_job_kind_items_render_the_same(self):
        """Rendering is kind-agnostic. Job-kind clarification is
        deferred from a product-behavior standpoint (Step 2.5), but
        the formatter itself doesn't discriminate."""
        jobs = (
            _job("Truck driver", "j1", 1),
            _job("Delivery driver", "j2", 2),
        )
        prompt = build_clarification_prompt(jobs)
        assert prompt == (
            "Which one do you mean — Truck driver or Delivery driver?"
        )


# ---------------------------------------------------------------- LLM fallback (Step 2.3)


class TestLLMFallbackDefensive:
    """Defensive short-circuits — none of these should hit the LLM."""

    def setup_method(self):
        reset_llm_cache()

    def test_empty_surface_returns_no_reference_without_llm_call(
        self, monkeypatch,
    ):
        def _fail(*a, **k):
            raise AssertionError("LLM should not be called on empty surface")
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver._call_reference_llm", _fail,
        )
        out = resolve_reference_via_llm(message="anything", surface_items=())
        assert out.status == "no_reference"
        assert out.reason == "no_surface"

    def test_empty_message_returns_no_reference_without_llm_call(
        self, monkeypatch,
    ):
        def _fail(*a, **k):
            raise AssertionError("LLM should not be called on empty message")
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver._call_reference_llm", _fail,
        )
        out = resolve_reference_via_llm(
            message="   ", surface_items=_TWO_ROLES,
        )
        assert out.status == "no_reference"
        assert out.reason == "empty_message"

    @pytest.mark.parametrize("bad", [None, 123, ["not a string"]])
    def test_non_string_returns_no_reference_without_llm_call(
        self, bad, monkeypatch,
    ):
        def _fail(*a, **k):
            raise AssertionError("LLM should not be called on non-string")
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver._call_reference_llm", _fail,
        )
        out = resolve_reference_via_llm(
            message=bad, surface_items=_TWO_ROLES,
        )
        assert out.status == "no_reference"
        assert out.reason == "non_string_message"

    def test_llm_disabled_returns_no_reference_without_llm_call(
        self, monkeypatch,
    ):
        def _fail(*a, **k):
            raise AssertionError("LLM should not be called when disabled")
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver._call_reference_llm", _fail,
        )
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver.LLM_ENABLED", False,
        )
        out = resolve_reference_via_llm(
            message="admin secretary", surface_items=_TWO_ROLES,
        )
        assert out.status == "no_reference"
        assert out.reason == "llm_disabled"


class TestLLMFallbackSelectionInterpretation:
    """LLM returned a valid selection — verify each enum value maps
    correctly back to a ResolveOutcome."""

    def setup_method(self):
        reset_llm_cache()

    def _stub_returning(self, monkeypatch, value: str):
        def _fake(*a, **k):
            return value
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver._call_reference_llm", _fake,
        )
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver.LLM_ENABLED", True,
        )

    def test_item_1_resolves_to_first_item(self, monkeypatch):
        self._stub_returning(monkeypatch, "item_1")
        out = resolve_reference_via_llm(
            message="admin", surface_items=_THREE_ROLES,
        )
        assert out.status == "resolved"
        assert out.item is _THREE_ROLES[0]
        assert out.reason == "llm_selected"

    def test_item_2_resolves_to_second_item(self, monkeypatch):
        self._stub_returning(monkeypatch, "item_2")
        out = resolve_reference_via_llm(
            message="accounting stuff", surface_items=_THREE_ROLES,
        )
        assert out.status == "resolved"
        assert out.item is _THREE_ROLES[1]

    def test_item_n_at_edge_of_surface(self, monkeypatch):
        self._stub_returning(monkeypatch, "item_3")
        out = resolve_reference_via_llm(
            message="data entry work", surface_items=_THREE_ROLES,
        )
        assert out.status == "resolved"
        assert out.item is _THREE_ROLES[2]

    def test_clarification_maps_to_clarification_status(self, monkeypatch):
        self._stub_returning(monkeypatch, "clarification")
        out = resolve_reference_via_llm(
            message="admin", surface_items=_THREE_ROLES,
        )
        assert out.status == "clarification"
        assert out.item is None
        assert out.reason == "llm_clarification"

    def test_no_match_maps_to_no_reference(self, monkeypatch):
        self._stub_returning(monkeypatch, "no_match")
        out = resolve_reference_via_llm(
            message="tell me about the weather", surface_items=_THREE_ROLES,
        )
        assert out.status == "no_reference"
        assert out.item is None
        assert out.reason == "llm_no_match"

    def test_out_of_range_item_coerces_to_no_reference(self, monkeypatch):
        """Defensive: LLM returned item_5 but surface has only 3 items."""
        self._stub_returning(monkeypatch, "item_5")
        out = resolve_reference_via_llm(
            message="fifth?", surface_items=_THREE_ROLES,
        )
        assert out.status == "no_reference"
        assert out.reason == "llm_out_of_range"

    def test_unknown_selection_coerces_to_no_reference(self, monkeypatch):
        """Defensive: schema constrains the enum, but if the model
        somehow returns junk (never enforced client-side), coerce to
        no_reference rather than crash."""
        self._stub_returning(monkeypatch, "some_garbage_value")
        out = resolve_reference_via_llm(
            message="?", surface_items=_THREE_ROLES,
        )
        assert out.status == "no_reference"
        assert out.reason == "llm_invalid"


class TestLLMFallbackExceptions:
    """Failure modes — API errors, exceptions, etc."""

    def setup_method(self):
        reset_llm_cache()

    def test_llm_exception_returns_no_reference(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("simulated API failure")
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver._call_reference_llm", _boom,
        )
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver.LLM_ENABLED", True,
        )
        out = resolve_reference_via_llm(
            message="admin secretary", surface_items=_TWO_ROLES,
        )
        assert out.status == "no_reference"
        assert out.reason == "llm_error"

    def test_llm_error_is_cached_to_prevent_retry_storms(self, monkeypatch):
        """Match the intent classifier's behavior: cache the failure
        result. Prevents retry storms on a permanently broken LLM."""
        call_count = {"n": 0}
        def _boom(*a, **k):
            call_count["n"] += 1
            raise RuntimeError("simulated API failure")
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver._call_reference_llm", _boom,
        )
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver.LLM_ENABLED", True,
        )
        out1 = resolve_reference_via_llm(
            message="admin secretary", surface_items=_TWO_ROLES,
        )
        out2 = resolve_reference_via_llm(
            message="admin secretary", surface_items=_TWO_ROLES,
        )
        assert call_count["n"] == 1, "second call should hit the cache"
        assert out1 == out2


class TestLLMFallbackCaching:
    """Cache identity discipline: full-identity key catches surface
    changes; message normalization catches trivial whitespace/case
    variation."""

    def setup_method(self):
        reset_llm_cache()

    def test_cache_hit_avoids_second_llm_call(self, monkeypatch):
        call_count = {"n": 0}
        def _fake(*a, **k):
            call_count["n"] += 1
            return "item_1"
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver._call_reference_llm", _fake,
        )
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver.LLM_ENABLED", True,
        )
        _ = resolve_reference_via_llm(
            message="admin secretary", surface_items=_TWO_ROLES,
        )
        _ = resolve_reference_via_llm(
            message="admin secretary", surface_items=_TWO_ROLES,
        )
        assert call_count["n"] == 1
        assert llm_cache_size() == 1

    def test_message_normalization_strips_and_lowers_in_cache_key(
        self, monkeypatch,
    ):
        """Same message with different whitespace / case hits the cache."""
        call_count = {"n": 0}
        def _fake(*a, **k):
            call_count["n"] += 1
            return "item_1"
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver._call_reference_llm", _fake,
        )
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver.LLM_ENABLED", True,
        )
        _ = resolve_reference_via_llm(
            message="admin secretary", surface_items=_TWO_ROLES,
        )
        _ = resolve_reference_via_llm(
            message="  Admin SECRETARY  ", surface_items=_TWO_ROLES,
        )
        assert call_count["n"] == 1

    def test_different_surface_kind_gets_distinct_cache_entry(
        self, monkeypatch,
    ):
        """Full-identity key: same label but different kind
        (role vs job) must not share a cache entry."""
        call_count = {"n": 0}
        def _fake(*a, **k):
            call_count["n"] += 1
            return "item_1"
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver._call_reference_llm", _fake,
        )
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver.LLM_ENABLED", True,
        )
        roles = (_role("Data entry clerk", "14400", 1),)
        jobs = (_job("Data entry clerk", "j1", 1),)
        _ = resolve_reference_via_llm(
            message="that one", surface_items=roles,
        )
        _ = resolve_reference_via_llm(
            message="that one", surface_items=jobs,
        )
        assert call_count["n"] == 2, (
            "kind change (role vs job) must produce distinct cache entries "
            "even when labels match"
        )
        assert llm_cache_size() == 2

    def test_different_surface_id_gets_distinct_cache_entry(
        self, monkeypatch,
    ):
        """Full-identity key: same label and kind but different id
        (different NOC) must not share a cache entry."""
        call_count = {"n": 0}
        def _fake(*a, **k):
            call_count["n"] += 1
            return "item_1"
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver._call_reference_llm", _fake,
        )
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver.LLM_ENABLED", True,
        )
        surface_a = (_role("Bookkeeper", "12200", 1),)
        surface_b = (_role("Bookkeeper", "12201", 1),)
        _ = resolve_reference_via_llm(
            message="that one", surface_items=surface_a,
        )
        _ = resolve_reference_via_llm(
            message="that one", surface_items=surface_b,
        )
        assert call_count["n"] == 2
        assert llm_cache_size() == 2

    def test_surface_order_change_gets_distinct_cache_entry(
        self, monkeypatch,
    ):
        """Reordering items renumbers the enum (item_1 now points to a
        different item), so ordering must be part of the cache key."""
        call_count = {"n": 0}
        def _fake(*a, **k):
            call_count["n"] += 1
            return "item_1"
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver._call_reference_llm", _fake,
        )
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver.LLM_ENABLED", True,
        )
        forward = _TWO_ROLES
        reversed_ = tuple(reversed(_TWO_ROLES))
        _ = resolve_reference_via_llm(
            message="the first", surface_items=forward,
        )
        _ = resolve_reference_via_llm(
            message="the first", surface_items=reversed_,
        )
        assert call_count["n"] == 2


class TestLLMFallbackNoLabelNormalization:
    """Locked: labels/ids pass through untouched (no lowercasing,
    no stripping) so malformed surfaces stay visible."""

    def setup_method(self):
        reset_llm_cache()

    def test_label_whitespace_change_is_distinct_cache_entry(
        self, monkeypatch,
    ):
        """A label with trailing whitespace vs one without produces
        different cache entries — malformed surfaces must not be
        silently normalized away."""
        call_count = {"n": 0}
        def _fake(*a, **k):
            call_count["n"] += 1
            return "item_1"
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver._call_reference_llm", _fake,
        )
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver.LLM_ENABLED", True,
        )
        clean = (_role("Bookkeeper", "12200", 1),)
        dirty = (_role("Bookkeeper ", "12200", 1),)  # trailing space
        _ = resolve_reference_via_llm(
            message="that one", surface_items=clean,
        )
        _ = resolve_reference_via_llm(
            message="that one", surface_items=dirty,
        )
        assert call_count["n"] == 2, (
            "labels are not normalized in the cache key; a malformed "
            "label should not collide with its clean equivalent"
        )


# ---------------------------------------------------------------- composition (Step 2.4)


class TestComposedResolverShortCircuits:
    """When deterministic has a definitive answer, LLM must not be
    called. Verifies short-circuit rules 1-4 of the locked
    composition."""

    def setup_method(self):
        reset_llm_cache()

    def _fail_llm(self, monkeypatch):
        """Install a stub that raises if the LLM is invoked."""
        def _boom(*a, **k):
            raise AssertionError("LLM must not be called on this path")
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver._call_reference_llm", _boom,
        )
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver.LLM_ENABLED", True,
        )

    def test_deterministic_ordinal_hit_skips_llm(self, monkeypatch):
        self._fail_llm(monkeypatch)
        out = resolve_reference_with_fallback(
            message="the first", surface_items=_THREE_ROLES,
        )
        assert out.status == "resolved"
        assert out.reason == "ordinal"

    def test_deterministic_label_hit_skips_llm(self, monkeypatch):
        self._fail_llm(monkeypatch)
        out = resolve_reference_with_fallback(
            message="Administrative assistant", surface_items=_THREE_ROLES,
        )
        assert out.status == "resolved"
        assert out.reason == "label_match_unique"

    def test_deterministic_pronoun_on_single_item_skips_llm(
        self, monkeypatch,
    ):
        self._fail_llm(monkeypatch)
        out = resolve_reference_with_fallback(
            message="that", surface_items=_ONE_ROLE,
        )
        assert out.status == "resolved"
        assert out.reason == "single_item_pronoun"

    def test_deterministic_multi_item_pronoun_clarification_skips_llm(
        self, monkeypatch,
    ):
        """Deterministic returned clarification -- LLM must not
        override it. The deterministic layer's clarification is a
        definitive answer, not a "we don't know"."""
        self._fail_llm(monkeypatch)
        out = resolve_reference_with_fallback(
            message="that one", surface_items=_THREE_ROLES,
        )
        assert out.status == "clarification"
        assert out.reason == "multi_item_pronoun"

    def test_deterministic_label_match_ambiguous_skips_llm(self, monkeypatch):
        self._fail_llm(monkeypatch)
        out = resolve_reference_with_fallback(
            message="Administrative assistant or Accounting clerk?",
            surface_items=_THREE_ROLES,
        )
        assert out.status == "clarification"
        assert out.reason == "label_match_ambiguous"


class TestComposedResolverFallthrough:
    """When deterministic returns no_reference AND surface is
    non-empty AND llm_enabled=True, LLM must be called and its
    outcome returned."""

    def setup_method(self):
        reset_llm_cache()

    def _stub_llm(self, monkeypatch, value: str):
        call_count = {"n": 0}
        def _fake(*a, **k):
            call_count["n"] += 1
            return value
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver._call_reference_llm", _fake,
        )
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver.LLM_ENABLED", True,
        )
        return call_count

    def test_deterministic_no_signal_falls_through_to_llm(self, monkeypatch):
        """The primary fallthrough case: message doesn't hit any
        deterministic rule but the surface has items and the caller
        allowed LLM."""
        calls = self._stub_llm(monkeypatch, "item_1")
        out = resolve_reference_with_fallback(
            message="admin secretary",  # near-miss for Administrative assistant
            surface_items=_THREE_ROLES,
        )
        assert calls["n"] == 1
        assert out.status == "resolved"
        assert out.item is _THREE_ROLES[0]
        assert out.reason == "llm_selected"

    def test_llm_clarification_propagates(self, monkeypatch):
        """LLM returned clarification -- composed helper propagates
        the LLM's reason verbatim, distinct from the deterministic
        clarification reason."""
        calls = self._stub_llm(monkeypatch, "clarification")
        out = resolve_reference_with_fallback(
            message="admin something", surface_items=_THREE_ROLES,
        )
        assert calls["n"] == 1
        assert out.status == "clarification"
        assert out.reason == "llm_clarification"

    def test_llm_no_match_propagates(self, monkeypatch):
        calls = self._stub_llm(monkeypatch, "no_match")
        out = resolve_reference_with_fallback(
            message="tell me about the weather", surface_items=_THREE_ROLES,
        )
        assert calls["n"] == 1
        assert out.status == "no_reference"
        assert out.reason == "llm_no_match"


class TestComposedResolverCallerDisabledFallback:
    """llm_enabled=False must preserve the deterministic outcome and
    reason verbatim -- no invented "suppressed" state."""

    def setup_method(self):
        reset_llm_cache()

    def _fail_llm(self, monkeypatch):
        def _boom(*a, **k):
            raise AssertionError("LLM must not be called when llm_enabled=False")
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver._call_reference_llm", _boom,
        )
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver.LLM_ENABLED", True,
        )

    def test_deterministic_no_signal_preserved_when_llm_disabled(
        self, monkeypatch,
    ):
        """The load-bearing telemetry test: caller passed
        llm_enabled=False, deterministic said no_signal. Reason must
        stay `no_signal` -- NOT rewritten to `llm_suppressed` or
        similar."""
        self._fail_llm(monkeypatch)
        out = resolve_reference_with_fallback(
            message="admin secretary",
            surface_items=_THREE_ROLES,
            llm_enabled=False,
        )
        assert out.status == "no_reference"
        assert out.reason == "no_signal", (
            "deterministic reason must be preserved verbatim when caller "
            "disabled fallback -- do not invent a suppressed state"
        )

    def test_deterministic_resolved_preserved_when_llm_disabled(
        self, monkeypatch,
    ):
        """llm_enabled=False on a deterministic-hit path is a no-op."""
        self._fail_llm(monkeypatch)
        out = resolve_reference_with_fallback(
            message="the first",
            surface_items=_THREE_ROLES,
            llm_enabled=False,
        )
        assert out.status == "resolved"
        assert out.reason == "ordinal"

    def test_deterministic_clarification_preserved_when_llm_disabled(
        self, monkeypatch,
    ):
        self._fail_llm(monkeypatch)
        out = resolve_reference_with_fallback(
            message="that one",
            surface_items=_THREE_ROLES,
            llm_enabled=False,
        )
        assert out.status == "clarification"
        assert out.reason == "multi_item_pronoun"


class TestComposedResolverEmptySurface:
    """Empty surface -- both deterministic and LLM would short-circuit
    on this. The composed helper must skip the LLM call outright."""

    def setup_method(self):
        reset_llm_cache()

    def test_empty_surface_skips_llm_call(self, monkeypatch):
        def _fail(*a, **k):
            raise AssertionError(
                "LLM must not be called with empty surface -- both layers "
                "short-circuit on no_surface anyway; skipping saves the call"
            )
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver._call_reference_llm", _fail,
        )
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver.LLM_ENABLED", True,
        )
        out = resolve_reference_with_fallback(
            message="admin secretary", surface_items=(),
        )
        assert out.status == "no_reference"
        assert out.reason == "no_surface"


class TestComposedResolverTelemetryReasonSignals:
    """Locked telemetry contract: the composed helper preserves three
    distinct "no LLM answer" signals downstream can grep on:
      1. deterministic `reason` when caller disabled fallback
      2. `llm_disabled` when LLM was runtime-disabled inside Step 2.3
      3. specific LLM failure reasons when the fallback ran

    This class pins that these three states are visibly distinct in
    ResolveOutcome.reason."""

    def setup_method(self):
        reset_llm_cache()

    def test_caller_disabled_signal(self, monkeypatch):
        """Caller signal: llm_enabled=False."""
        # LLM not called -- so monkeypatching it is defensive only.
        def _fail(*a, **k):
            raise AssertionError("LLM must not be called")
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver._call_reference_llm", _fail,
        )
        out = resolve_reference_with_fallback(
            message="admin secretary",
            surface_items=_THREE_ROLES,
            llm_enabled=False,
        )
        assert out.reason == "no_signal"  # deterministic reason preserved

    def test_runtime_disabled_signal(self, monkeypatch):
        """Config signal: LLM_ENABLED=False at runtime, caller allowed
        fallback. Step 2.3 short-circuits with llm_disabled; composed
        helper propagates it verbatim."""
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver.LLM_ENABLED", False,
        )
        def _fail(*a, **k):
            raise AssertionError(
                "LLM must not be called when LLM_ENABLED=False"
            )
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver._call_reference_llm", _fail,
        )
        out = resolve_reference_with_fallback(
            message="admin secretary",
            surface_items=_THREE_ROLES,
            # llm_enabled=True (default) -- caller allowed fallback
        )
        assert out.reason == "llm_disabled", (
            "config-level disable is a different signal from caller-level "
            "disable -- Step 2.3's reason must propagate verbatim"
        )

    def test_llm_failure_signal_propagates(self, monkeypatch):
        """LLM ran and hit an API failure. `llm_error` is a third,
        distinct signal from the two disabled cases above."""
        def _boom(*a, **k):
            raise RuntimeError("simulated failure")
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver._call_reference_llm", _boom,
        )
        monkeypatch.setattr(
            "skillbridge.chat.reference_resolver.LLM_ENABLED", True,
        )
        out = resolve_reference_with_fallback(
            message="admin secretary", surface_items=_THREE_ROLES,
        )
        assert out.reason == "llm_error"
