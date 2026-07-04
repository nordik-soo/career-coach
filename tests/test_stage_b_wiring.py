"""Integration tests for Step 2.6 Stage B wiring
(matching-handoff pre-classifier inside _maybe_route_recommender_from_intent).

DB-free, no LLM, no HTTP. Monkeypatched store; deterministic resolver
path only (LLM_ENABLED=False so the composed resolver stops after the
deterministic pass); LLM_ENABLED=False for the intent classifier too
so its fall-through path returns "unclear" without a real API call.

Locked wiring under test:
  1. pivot intent absent -> Stage B is bypassed; existing classifier/
     router flow runs unchanged. Composed resolver stubbed to raise
     if invoked -- pivot gate holding is what makes this pass.
  2. pivot intent present + resolver resolves role -> handoff fires,
     cascade clears the recommender surface + pending offer, function
     returns None (main flow will run matching next), telemetry emits
     path=route_handoff_matching with resolution_outcome=deterministic.
  3. pivot intent present + no reference resolvable -> Stage B falls
     through to the classifier/router (bare "show me jobs" scenario).
  4. pivot intent present + clarification -> clarification response
     via _emit_canned_response with resolution_outcome=clarification_asked.
  5. pivot intent + resolved-but-job-kind -> Stage B does NOT hand
     off; falls through to classifier. Verifies kind guard.
"""
from __future__ import annotations

import logging

import pytest

from skillbridge.chat import handler
from skillbridge.chat.handler import _maybe_route_recommender_from_intent
from skillbridge.session.staging import StagedProfile


pytestmark = pytest.mark.nodb


class _FakeStore:
    """Minimal in-memory session store; save() returns the session id."""

    def save(self, staged: StagedProfile) -> str:
        return staged.session_id


def _new_staged_with_recommender_surface() -> StagedProfile:
    """Fresh staged with a two-role recommender surface visible via
    derive_frame, but NO pending_recommender_offer (this is the
    non-drilldown case -- the surface was left over from a prior
    Layer C render that has since resolved). Also no target_role_text
    so the router's target_noc resolution branch skips its DB call."""
    s = StagedProfile.new(session_id="test-stage-b-uuid-0001")
    s.last_recommender_adjacent_surface = (
        {"noc_code": "13110", "title": "Administrative assistant"},
        {"noc_code": "14200", "title": "Accounting clerk"},
    )
    s.last_recommender_adjacent_surface_at_turn = 5
    # target_role_text stays None so `if not target_noc and
    # staged.target_role_text` guard in _maybe_route_recommender_from_intent
    # short-circuits and doesn't invoke resolve_title_to_noc (DB call).
    return s


# ---------------------------------------------------------------- scenario 1


class TestNoPivotBypassesStageB:
    """The load-bearing non-regression test: without pivot intent,
    Stage B is entirely bypassed. Existing classifier + router chain
    runs untouched."""

    def test_no_pivot_bypasses_stage_b_resolver(self, monkeypatch):
        staged = _new_staged_with_recommender_surface()
        store = _FakeStore()

        # If Stage B's resolver runs, this stub raises -- the pivot
        # gate is what prevents that.
        from skillbridge.chat import reference_resolver
        def _boom(*a, **k):
            raise AssertionError(
                "Stage B resolver ran without pivot intent -- gate broke"
            )
        monkeypatch.setattr(
            reference_resolver, "resolve_reference_with_fallback", _boom,
        )
        # LLM classifier disabled so it short-circuits to "unclear"
        # without a real API call.
        from skillbridge.chat import recommender_intent
        monkeypatch.setattr(recommender_intent, "LLM_ENABLED", False)

        result = _maybe_route_recommender_from_intent(
            staged=staged,
            message="hello there",  # no pivot verb, no reference
            store=store,
        )
        # Stage B never touched target_role_text.
        assert staged.target_role_text is None
        # Result depends on classifier+router. career_intent="unclear"
        # + pattern_intent="neutral" -> router verdict = default ->
        # returns None. What matters: Stage B didn't intercept.
        assert result is None


# ---------------------------------------------------------------- scenario 2


class TestPivotWithResolvableRoleHandsOff:
    """Pivot + resolvable role -> handoff, target set, cascade clears
    recommender surface, return None so matching flow handles the turn."""

    def test_match_me_to_second_one_hands_off_from_router(
        self, monkeypatch, caplog,
    ):
        staged = _new_staged_with_recommender_surface()
        store = _FakeStore()

        # Deterministic path only.
        from skillbridge.chat import reference_resolver
        monkeypatch.setattr(reference_resolver, "LLM_ENABLED", False)

        with caplog.at_level(
            logging.INFO, logger="skillbridge.chat.frame_telemetry"
        ):
            result = _maybe_route_recommender_from_intent(
                staged=staged,
                message="match me to the second one",
                store=store,
            )

        # Handoff fires -> None returned -> caller falls through to
        # matching flow with the new target.
        assert result is None
        assert staged.target_role_text == "Accounting clerk"
        assert staged.target_noc == "14200"
        # Cascade cleared the recommender state (surface + anchor).
        assert staged.last_recommender_adjacent_surface == ()
        assert staged.last_recommender_adjacent_surface_at_turn is None
        # Telemetry emitted with the locked router-path label + outcome.
        rec = next(
            r for r in caplog.records
            if "frame_telemetry" in r.getMessage()
            and "route_handoff_matching" in r.getMessage()
        )
        assert "resolution_outcome=deterministic" in rec.getMessage()

    def test_match_me_to_admin_assistant_hands_off_via_label(
        self, monkeypatch, caplog,
    ):
        """Full-label deterministic hit -> handoff to item 1."""
        staged = _new_staged_with_recommender_surface()
        store = _FakeStore()
        from skillbridge.chat import reference_resolver
        monkeypatch.setattr(reference_resolver, "LLM_ENABLED", False)

        with caplog.at_level(
            logging.INFO, logger="skillbridge.chat.frame_telemetry"
        ):
            result = _maybe_route_recommender_from_intent(
                staged=staged,
                message="match me to Administrative assistant",
                store=store,
            )
        assert result is None
        assert staged.target_role_text == "Administrative assistant"
        assert staged.target_noc == "13110"


# ---------------------------------------------------------------- scenario 3


class TestPivotWithoutReferenceFallsThrough:
    """Pivot present but nothing on the surface matches -> Stage B
    falls through to the classifier/router chain."""

    def test_show_me_jobs_falls_through_to_classifier(self, monkeypatch):
        staged = _new_staged_with_recommender_surface()
        store = _FakeStore()

        # Deterministic-only + classifier disabled so no LLM calls
        # anywhere. Classifier will return "unclear"; verdict = default;
        # returns None.
        from skillbridge.chat import reference_resolver
        monkeypatch.setattr(reference_resolver, "LLM_ENABLED", False)
        from skillbridge.chat import recommender_intent
        monkeypatch.setattr(recommender_intent, "LLM_ENABLED", False)

        result = _maybe_route_recommender_from_intent(
            staged=staged,
            message="show me jobs",  # pivot verb, no reference
            store=store,
        )
        # Stage B did NOT hand off (nothing resolved to a role).
        assert staged.target_role_text is None
        # "show me jobs" is Path A's impatient_proceed signal --
        # pattern_intent -> _PATTERN_TO_MATCHING -> router verdict
        # matching_engine -> returns None (matching flow handles).
        assert result is None
        # Downstream note: with router verdict == matching_engine and
        # a live recommender surface, Step 1.3's aggressive pivot-clear
        # fires and clears the recommender surface + anchor. That is
        # CORRECT slice-1 behavior (user pivoted to matching; stale
        # recommender state must not hijack the next confirming reply).
        # Stage B's contract was: don't hand off + fall through
        # cleanly, which it did. Any surface clearing downstream of
        # Stage B belongs to Step 1.3, not Stage B.
        assert staged.last_recommender_adjacent_surface == ()


# ---------------------------------------------------------------- scenario 4


class TestPivotWithAmbiguousReferenceAsksClarification:
    """Pivot + ambiguous reference (deterministic returns clarification)
    -> Stage B emits the clarification prompt via
    _emit_canned_response and returns the response dict."""

    def test_ambiguous_pivot_yields_clarification_response(
        self, monkeypatch, caplog,
    ):
        staged = _new_staged_with_recommender_surface()
        store = _FakeStore()
        from skillbridge.chat import reference_resolver
        monkeypatch.setattr(reference_resolver, "LLM_ENABLED", False)

        with caplog.at_level(
            logging.INFO, logger="skillbridge.chat.frame_telemetry"
        ):
            # Message names BOTH labels -> deterministic
            # label_match_ambiguous -> clarification.
            result = _maybe_route_recommender_from_intent(
                staged=staged,
                message=(
                    "match me to Administrative assistant "
                    "or Accounting clerk?"
                ),
                store=store,
            )
        assert result is not None
        # Locked Step 2.2 format.
        assert "Which one do you mean" in result["reply"]
        assert "Administrative assistant" in result["reply"]
        assert "Accounting clerk" in result["reply"]
        # Handoff did NOT fire.
        assert staged.target_role_text is None
        # Recommender surface preserved so the user can answer.
        assert len(staged.last_recommender_adjacent_surface) == 2
        # Telemetry.
        rec = next(
            r for r in caplog.records
            if "frame_telemetry" in r.getMessage()
            and "route_handoff_clarification" in r.getMessage()
        )
        assert "resolution_outcome=clarification_asked" in rec.getMessage()


# ---------------------------------------------------------------- scenario 5


class TestPivotWithJobKindDoesNotHandOff:
    """The kind guard: resolved-but-job-kind means Stage B falls
    through to the classifier rather than raise (handoff helper is
    role-only and would ValueError)."""

    def test_pivot_with_matching_surface_job_kind_falls_through(
        self, monkeypatch,
    ):
        # Set up a MATCHING surface (job kind) instead of the
        # recommender's role surface.
        staged = StagedProfile.new(session_id="test-stage-b-jobkind")
        staged.last_presented_job_titles = [
            "Truck driver at Acme",
            "Delivery driver at Sault Co",
        ]
        staged.last_presented_at_turn = 5

        store = _FakeStore()
        from skillbridge.chat import reference_resolver
        monkeypatch.setattr(reference_resolver, "LLM_ENABLED", False)
        from skillbridge.chat import recommender_intent
        monkeypatch.setattr(recommender_intent, "LLM_ENABLED", False)

        # Pivot + ordinal ("the first") -> deterministic resolves item
        # from the matching surface -> kind is "job" -> Stage B guard
        # skips handoff and falls through.
        result = _maybe_route_recommender_from_intent(
            staged=staged,
            message="match me to the first",
            store=store,
        )
        # Stage B did NOT hand off (kind was job, not role).
        assert staged.target_role_text is None
        # Matching titles surface untouched.
        assert len(staged.last_presented_job_titles) == 2
        # Result depends on classifier fallthrough; None expected under
        # unclear + default.
        assert result is None
