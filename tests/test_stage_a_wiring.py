"""Integration tests for Step 2.6 Stage A wiring
(matching-handoff hijack inside _consume_drilldown_selection).

DB-free, no LLM, no HTTP. Monkeypatched store; deterministic resolver
path only (LLM_ENABLED=False so the composed resolver stops after the
deterministic pass).

Locked wiring under test:
  1. pivot intent absent -> Stage A is bypassed; drilldown resolver
     runs unchanged. Bare "the second one" still selects the second
     adjacent NOC and dispatches drilldown.
  2. pivot intent present + resolver resolves role -> handoff fires,
     cascade clears the recommender surface + pending offer, function
     returns None (main flow will run matching next), telemetry emits
     path=drilldown_handoff_matching with resolution_outcome=deterministic.
  3. pivot intent present + no reference resolvable -> Stage A falls
     through to the existing drilldown resolver (bare "show me jobs"
     scenario).
  4. pivot intent present + clarification -> clarification prompt
     emitted with resolution_outcome=clarification_asked.
"""
from __future__ import annotations

import logging

import pytest

from skillbridge.chat import handler
from skillbridge.chat.handler import _consume_drilldown_selection
from skillbridge.session.staging import StagedProfile


pytestmark = pytest.mark.nodb


class _FakeStore:
    """Minimal in-memory session store; save() returns the session id."""

    def save(self, staged: StagedProfile) -> str:
        return staged.session_id


def _new_staged_with_drilldown_pending() -> StagedProfile:
    """Fresh staged with the exact preconditions for
    _consume_drilldown_selection to run: pending_recommender_offer set
    to the drilldown-select value, and a two-role surface populated."""
    s = StagedProfile.new(session_id="test-stage-a-uuid-0001")
    s.pending_recommender_offer = "adjacent_role_drilldown_select"
    s.last_recommender_adjacent_surface = (
        {"noc_code": "13110", "title": "Administrative assistant"},
        {"noc_code": "14200", "title": "Accounting clerk"},
    )
    s.last_recommender_adjacent_surface_at_turn = 5
    return s


# ---------------------------------------------------------------- scenario 1


class TestBareOrdinalStillMeansDrilldown:
    """The most important non-regression: without pivot intent, the
    existing drilldown resolver runs unchanged. 'the second one'
    keeps meaning drilldown-select."""

    def test_bare_ordinal_dispatches_drilldown_not_handoff(
        self, monkeypatch,
    ):
        staged = _new_staged_with_drilldown_pending()
        store = _FakeStore()

        # Stub the drilldown dispatcher so we detect that it fires.
        called = {}
        def _fake_dispatch(*, staged, noc_code, role_title, store,
                           resume_info, user_message):
            called["noc_code"] = noc_code
            called["role_title"] = role_title
            return {
                "reply": "drilldown fired",
                "profile_id": None,
                "session_id": staged.session_id,
                "intake_state": staged.intake_state,
                "asked_slots": [],
                "next_action": "present_matches",
                "recommended_jobs": [],
                "next_skill_suggestion": None,
                "resume_info": resume_info,
                "requires_consent": True,
            }
        monkeypatch.setattr(handler, "_dispatch_role_drilldown", _fake_dispatch)

        # Stage A must NOT run its resolver on this path -- assert by
        # patching the composed resolver to raise if invoked.
        from skillbridge.chat import reference_resolver
        def _boom(*a, **k):
            raise AssertionError(
                "Stage A resolver ran without pivot intent -- pivot gate broke"
            )
        monkeypatch.setattr(
            reference_resolver, "resolve_reference_with_fallback", _boom,
        )

        result = _consume_drilldown_selection(
            staged=staged,
            user_message="the second one",
            store=store,
            resume_info=None,
        )
        # Drilldown fired on the second role (14200 accounting clerk).
        assert result is not None
        assert called.get("noc_code") == "14200"
        assert called.get("role_title") == "Accounting clerk"
        # Recommender state UNCHANGED -- drilldown keeps pending +
        # surface so the user can pick another.
        assert staged.pending_recommender_offer == (
            "adjacent_role_drilldown_select"
        )
        assert len(staged.last_recommender_adjacent_surface) == 2


# ---------------------------------------------------------------- scenario 2


class TestPivotWithResolvableRoleHandsOff:
    """Pivot intent + resolvable role -> handoff fires, target set,
    cascade clears recommender state, return None so matching flow
    handles the turn."""

    def test_match_me_to_the_second_one_hands_off_to_matching(
        self, monkeypatch, caplog,
    ):
        staged = _new_staged_with_drilldown_pending()
        store = _FakeStore()

        # Deterministic resolver will hit on ordinal "second" -> role
        # kind (Administrative assistant is item 1, Accounting clerk
        # is item 2). No LLM path needed; leave LLM disabled.
        from skillbridge.chat import reference_resolver
        monkeypatch.setattr(reference_resolver, "LLM_ENABLED", False)

        with caplog.at_level(
            logging.INFO, logger="skillbridge.chat.frame_telemetry"
        ):
            result = _consume_drilldown_selection(
                staged=staged,
                user_message="match me to the second one",
                store=store,
                resume_info=None,
            )

        # Handoff fires -> None returned -> caller falls through to
        # matching flow.
        assert result is None
        # Target written to Accounting clerk (item 2 in the surface).
        assert staged.target_role_text == "Accounting clerk"
        # NOC 14200 was a 5-digit NOC 2021 code -> target_noc set.
        assert staged.target_noc == "14200"
        # Cascade cleared the recommender state.
        assert staged.pending_recommender_offer is None
        assert staged.last_recommender_adjacent_surface == ()
        assert staged.last_recommender_adjacent_surface_at_turn is None
        # Telemetry emitted with the locked path label and outcome.
        rec = next(
            r for r in caplog.records
            if "frame_telemetry" in r.getMessage()
            and "drilldown_handoff_matching" in r.getMessage()
        )
        assert "resolution_outcome=deterministic" in rec.getMessage()

    def test_match_me_to_admin_assistant_by_label_hands_off(
        self, monkeypatch, caplog,
    ):
        """Full-label deterministic hit -> handoff to item 1."""
        staged = _new_staged_with_drilldown_pending()
        store = _FakeStore()
        from skillbridge.chat import reference_resolver
        monkeypatch.setattr(reference_resolver, "LLM_ENABLED", False)

        with caplog.at_level(
            logging.INFO, logger="skillbridge.chat.frame_telemetry"
        ):
            result = _consume_drilldown_selection(
                staged=staged,
                user_message="match me to Administrative assistant",
                store=store,
                resume_info=None,
            )
        assert result is None
        assert staged.target_role_text == "Administrative assistant"
        assert staged.target_noc == "13110"


# ---------------------------------------------------------------- scenario 3


class TestPivotWithoutReferenceFallsThrough:
    """Pivot present but nothing on the surface matches -> Stage A
    falls through to the existing drilldown resolver. Bare 'show me
    jobs' has no reference to resolve."""

    def test_show_me_jobs_falls_through_to_existing_drilldown(
        self, monkeypatch,
    ):
        staged = _new_staged_with_drilldown_pending()
        store = _FakeStore()
        from skillbridge.chat import reference_resolver
        monkeypatch.setattr(reference_resolver, "LLM_ENABLED", False)

        # Assert we reach the existing drilldown resolver.
        from skillbridge.chat import recommender_assembly
        called = {"resolve": False}
        original = recommender_assembly.resolve_drilldown_selection
        def _wrapper(user_message, surface):
            called["resolve"] = True
            return original(user_message, surface)
        monkeypatch.setattr(
            recommender_assembly,
            "resolve_drilldown_selection",
            _wrapper,
        )

        result = _consume_drilldown_selection(
            staged=staged,
            user_message="show me jobs",  # pivot but no reference
            store=store,
            resume_info=None,
        )
        assert called["resolve"], (
            "expected fallthrough to existing drilldown resolver when "
            "pivot intent is present but no reference resolves"
        )
        # Handoff did NOT fire (this is the load-bearing assertion for
        # Stage A: pivot present, but nothing on the surface resolved).
        assert staged.target_role_text is None
        # Downstream behavior belongs to pre-existing logic: existing
        # drilldown resolver misses -> consent classifier fires ->
        # Path A pivot short-circuit returns "other" -> existing
        # consent="other" branch clears pending state and returns None
        # so the main router takes over on "show me jobs" (matching
        # engine). That downstream chain is Path A + slice 1 territory,
        # not Stage A. What matters for this test: Stage A DID NOT
        # hand off. Which it didn't (no target write).
        assert result is None


# ---------------------------------------------------------------- scenario 4


class TestPivotWithAmbiguousReferenceAsksClarification:
    """Pivot + ambiguous reference (deterministic resolver returns
    clarification) -> Stage A emits the clarification prompt and
    returns a response dict."""

    def test_ambiguous_pivot_yields_clarification_prompt(
        self, monkeypatch, caplog,
    ):
        staged = _new_staged_with_drilldown_pending()
        store = _FakeStore()
        from skillbridge.chat import reference_resolver
        monkeypatch.setattr(reference_resolver, "LLM_ENABLED", False)

        with caplog.at_level(
            logging.INFO, logger="skillbridge.chat.frame_telemetry"
        ):
            # This message names BOTH labels -> deterministic
            # label_match_ambiguous -> clarification status.
            result = _consume_drilldown_selection(
                staged=staged,
                user_message=(
                    "match me to Administrative assistant "
                    "or Accounting clerk?"
                ),
                store=store,
                resume_info=None,
            )
        assert result is not None
        # Locked clarification format from Step 2.2.
        assert "Which one do you mean" in result["reply"]
        assert "Administrative assistant" in result["reply"]
        assert "Accounting clerk" in result["reply"]
        # Handoff did NOT fire.
        assert staged.target_role_text is None
        # Pending drilldown offer + surface preserved so the user can
        # answer the clarification.
        assert staged.pending_recommender_offer == (
            "adjacent_role_drilldown_select"
        )
        assert len(staged.last_recommender_adjacent_surface) == 2
        # Telemetry emitted with the clarification path + outcome.
        rec = next(
            r for r in caplog.records
            if "frame_telemetry" in r.getMessage()
            and "drilldown_handoff_clarification" in r.getMessage()
        )
        assert "resolution_outcome=clarification_asked" in rec.getMessage()
