"""Slice 2 integration tests — synthetic multi-turn scenarios.

DB-free, no LLM (except one scenario that stubs the internal LLM
tool call), no real engine, no real Layer C evidence pipeline. Each
scenario simulates a prior turn's output by seeding StagedProfile
directly, then exercises the current turn's wiring (Stage A or B).

Style follows tests/test_slice1_integration.py: one test class per
scenario, load-bearing assertions inline with comments explaining the
production behavior the assertion protects.

Locked scope (2026-07-04): Step 2.7 is an orchestration proof for
slice 2's cross-turn memory contract, not a full-stack production
gate. The full-stack gate is live testing before push/release, per
the standing discipline. Explicitly out of scope here:
  - handle_anonymous full turn flow (extractor/planner/session save)
  - real Layer C evidence build (needs DB + OaSIS + embeddings)
  - cookie session round-trip
  - real Anthropic classifier or fallback model

What this file DOES prove:
  - Turn-1 surface stamping survives to turn-2 resolver invocation
  - Turn-2 handoff cascade cleans up recommender state before matching
  - Kind guard (role-only handoff) propagates through both wiring stages
  - LLM fallback composition works when deterministic misses
  - Clarification preserves surface for the following turn
  - Non-regression: pivot gate holds so bare ordinals keep drilldown
"""
from __future__ import annotations

import logging

import pytest

from skillbridge.chat import handler
from skillbridge.chat.handler import (
    _consume_drilldown_selection,
    _maybe_route_recommender_from_intent,
)
from skillbridge.session.staging import StagedProfile


pytestmark = pytest.mark.nodb


# ---------------------------------------------------------------- shared helpers


class _FakeStore:
    """Minimal in-memory session store; save() returns the session id.
    Matches the fake used in the slice-1 integration file so downstream
    code that calls store.save(staged) works without a real Redis or
    cookie backend."""

    def save(self, staged: StagedProfile) -> str:
        return staged.session_id


def _after_turn_1_recommender_rendered(
    *,
    pending_drilldown: bool,
    at_turn: int = 3,
) -> StagedProfile:
    """Simulate the state StagedProfile would carry AFTER a prior turn
    where the recommender rendered its adjacent NOC surface.

    Two variants gated by `pending_drilldown`:
      True  -> Layer C just rendered the surface + set pending drilldown
               (this turn will hit _consume_drilldown_selection / Stage A)
      False -> Layer C surface was rendered on a prior turn but the
               drilldown chain has since ended (or was pivoted past)
               (this turn will hit _maybe_route_recommender_from_intent /
                Stage B)

    Locked minimum seeded state (mirrors what Layer C's real
    dispatchers set in production, without invoking them):
      - last_recommender_adjacent_surface (2 role items)
      - last_recommender_adjacent_surface_at_turn (Step 1.2 anchor)
      - pending_recommender_offer (only when pending_drilldown=True)

    target_role_text stays None so the router's target_noc resolution
    branch (in _maybe_route_recommender_from_intent) short-circuits
    without invoking resolve_title_to_noc (a DB call).
    """
    s = StagedProfile.new(session_id="test-slice2-integration-uuid-0001")
    s.last_recommender_adjacent_surface = (
        {"noc_code": "13110", "title": "Administrative assistant"},
        {"noc_code": "14200", "title": "Accounting clerk"},
    )
    s.last_recommender_adjacent_surface_at_turn = at_turn
    if pending_drilldown:
        s.pending_recommender_offer = "adjacent_role_drilldown_select"
    return s


def _after_turn_1_matching_rendered(*, at_turn: int = 3) -> StagedProfile:
    """Simulate the state StagedProfile would carry AFTER a prior turn
    where the matching engine rendered its top-N job titles (no
    recommender turn). Used by the kind-guard scenario to force the
    frame's latest_surface_items to be job kind instead of role."""
    s = StagedProfile.new(session_id="test-slice2-integration-jobkind")
    s.last_presented_job_titles = [
        "Truck driver at Acme",
        "Delivery driver at Sault Co",
    ]
    s.last_presented_at_turn = at_turn
    return s


def _find_telemetry(caplog, path_substring: str) -> logging.LogRecord:
    """Grep caplog for the frame_telemetry record naming the given
    path label. Raises StopIteration if absent (which is what we want
    on assertion — the test fails loudly)."""
    return next(
        r for r in caplog.records
        if "frame_telemetry" in r.getMessage()
        and path_substring in r.getMessage()
    )


# ---------------------------------------------------------------- scenario 1


class TestScenario1_DrilldownDeterministicOrdinal:
    """Primary product flow:

    Turn 1 (simulated): Layer C rendered two adjacent NOC roles and
      set pending_recommender_offer = adjacent_role_drilldown_select.
    Turn 2: user says "match me to the second one" -- pivot verb +
      ordinal reference. Deterministic resolver hits ordinal on the
      surface. Stage A hands off to matching against item 2.

    Cross-turn memory contract: the surface stamped at turn 1 must be
    visible to the resolver at turn 2 (via derive_frame), the handoff
    cascade must clean it up on the target write, and the return value
    must be None so handle_anonymous falls through to matching."""

    def test(self, monkeypatch, caplog):
        # Turn 1 state:
        staged = _after_turn_1_recommender_rendered(pending_drilldown=True)
        assert staged.last_recommender_adjacent_surface_at_turn == 3
        # LLM off across the board (deterministic path only).
        from skillbridge.chat import reference_resolver
        monkeypatch.setattr(reference_resolver, "LLM_ENABLED", False)

        # Turn 2 message:
        with caplog.at_level(
            logging.INFO, logger="skillbridge.chat.frame_telemetry"
        ):
            result = _consume_drilldown_selection(
                staged=staged,
                user_message="match me to the second one",
                store=_FakeStore(),
                resume_info=None,
            )

        # Handoff fires -> None returned -> caller falls through to
        # the matching flow with the new target.
        assert result is None
        # Target written to item 2 of turn-1's surface.
        assert staged.target_role_text == "Accounting clerk"
        assert staged.target_noc == "14200"
        # Cross-turn cleanup: cascade wiped the recommender state so
        # a future confirming reply cannot hijack it.
        assert staged.last_recommender_adjacent_surface == ()
        assert staged.last_recommender_adjacent_surface_at_turn is None
        assert staged.pending_recommender_offer is None
        # Telemetry: Stage A's drilldown_handoff_matching path with
        # the deterministic-layer outcome.
        rec = _find_telemetry(caplog, "drilldown_handoff_matching")
        assert "resolution_outcome=deterministic" in rec.getMessage()


# ---------------------------------------------------------------- scenario 2


class TestScenario2_BareOrdinalStaysDrilldown:
    """Non-regression: without pivot intent, the existing drilldown
    resolver runs unchanged. `the second one` alone must still mean
    "select item 2 for drilldown", not "handoff to matching for item 2".

    This is the LOAD-BEARING regression guard for the Stage A pivot
    gate. If the gate breaks, this test asserts drilldown fired via a
    stubbed dispatcher; if Stage A wrongly intercepted, no dispatcher
    call would be recorded and the assertion trips."""

    def test(self, monkeypatch):
        # Turn 1 state:
        staged = _after_turn_1_recommender_rendered(pending_drilldown=True)

        # Turn 2 shim: stub _dispatch_role_drilldown so we detect that
        # the drilldown flow (not handoff) fired.
        called = {}
        def _fake_dispatch(
            *, staged, noc_code, role_title, store, resume_info,
            user_message,
        ):
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

        # Also raise if Stage A's composed resolver runs -- pivot gate
        # holding is what makes this pass.
        from skillbridge.chat import reference_resolver
        def _boom(*a, **k):
            raise AssertionError(
                "Stage A resolver ran on a bare ordinal -- pivot gate broke"
            )
        monkeypatch.setattr(
            reference_resolver, "resolve_reference_with_fallback", _boom,
        )

        result = _consume_drilldown_selection(
            staged=staged,
            user_message="the second one",  # no pivot verb
            store=_FakeStore(),
            resume_info=None,
        )
        # Drilldown fired on the correct item.
        assert result is not None
        assert called.get("noc_code") == "14200"
        assert called.get("role_title") == "Accounting clerk"
        # Turn 1's recommender state preserved so the user can pick
        # another item (drilldown does not clear on hit).
        assert len(staged.last_recommender_adjacent_surface) == 2
        assert staged.pending_recommender_offer == (
            "adjacent_role_drilldown_select"
        )
        # No handoff occurred: target still None.
        assert staged.target_role_text is None


# ---------------------------------------------------------------- scenario 3


class TestScenario3_RouterPreCheckHandoff:
    """Stage B path: no drilldown pending, but a Layer C surface is
    still visible from a prior turn. The user pivots and names a role.

    Turn 1 (simulated): Layer C rendered adjacent roles on some prior
      turn; the drilldown chain has since ended (pending offer
      cleared).
    Turn 2: user says "match me to Administrative assistant" -- Stage B
      resolves via deterministic label match and hands off."""

    def test(self, monkeypatch, caplog):
        # Turn 1 state:
        staged = _after_turn_1_recommender_rendered(pending_drilldown=False)
        assert staged.pending_recommender_offer is None
        assert staged.last_recommender_adjacent_surface_at_turn == 3

        from skillbridge.chat import reference_resolver
        monkeypatch.setattr(reference_resolver, "LLM_ENABLED", False)

        with caplog.at_level(
            logging.INFO, logger="skillbridge.chat.frame_telemetry"
        ):
            result = _maybe_route_recommender_from_intent(
                staged=staged,
                message="match me to Administrative assistant",
                store=_FakeStore(),
            )

        assert result is None
        assert staged.target_role_text == "Administrative assistant"
        assert staged.target_noc == "13110"
        assert staged.last_recommender_adjacent_surface == ()
        assert staged.last_recommender_adjacent_surface_at_turn is None
        # Telemetry: Stage B's router-path label.
        rec = _find_telemetry(caplog, "route_handoff_matching")
        assert "resolution_outcome=deterministic" in rec.getMessage()


# ---------------------------------------------------------------- scenario 4


class TestScenario4_ClarificationRoundTrip:
    """Cross-turn contract for clarification: turn 2's ambiguous
    message produces a clarification response, but the surface stays
    alive so turn 3 can answer it.

    Turn 1 (simulated): Layer C rendered two roles + set pending
      drilldown.
    Turn 2: user says "match me to Administrative assistant or
      Accounting clerk?" -- deterministic label_match_ambiguous ->
      Stage A emits the clarification prompt.

    Load-bearing: the recommender surface + pending offer must survive
    turn 2 so turn 3 has the state to resolve against. Same for the
    surface anchor -- without it, turn 3's derive_frame would show the
    surface as unanchored."""

    def test(self, monkeypatch, caplog):
        # Turn 1 state:
        staged = _after_turn_1_recommender_rendered(pending_drilldown=True)

        from skillbridge.chat import reference_resolver
        monkeypatch.setattr(reference_resolver, "LLM_ENABLED", False)

        with caplog.at_level(
            logging.INFO, logger="skillbridge.chat.frame_telemetry"
        ):
            result = _consume_drilldown_selection(
                staged=staged,
                user_message=(
                    "match me to Administrative assistant "
                    "or Accounting clerk?"
                ),
                store=_FakeStore(),
                resume_info=None,
            )

        assert result is not None
        # Step 2.2 locked format.
        assert "Which one do you mean" in result["reply"]
        assert "Administrative assistant" in result["reply"]
        assert "Accounting clerk" in result["reply"]
        # No handoff on this turn.
        assert staged.target_role_text is None
        # Cross-turn contract: surface + anchor + pending offer
        # PRESERVED so turn 3 can resolve against the same items.
        assert len(staged.last_recommender_adjacent_surface) == 2
        assert staged.last_recommender_adjacent_surface_at_turn == 3
        assert staged.pending_recommender_offer == (
            "adjacent_role_drilldown_select"
        )
        rec = _find_telemetry(caplog, "drilldown_handoff_clarification")
        assert "resolution_outcome=clarification_asked" in rec.getMessage()


# ---------------------------------------------------------------- scenario 5


class TestScenario5_LLMFallbackNearMiss:
    """Composed resolver composition proof: deterministic misses on a
    near-miss phrasing, LLM fallback picks up and returns item_1.

    Monkeypatched at the INTERNAL _call_reference_llm level (not at
    resolve_reference_with_fallback) so the composition logic
    (deterministic-first-then-LLM) actually runs. Only the Anthropic
    API round-trip is short-circuited.

    Turn 1 (simulated): Layer C rendered surface.
    Turn 2: user says "match me to admin secretary" (near-miss for
      "Administrative assistant"). Deterministic no_signal -> LLM stub
      returns item_1 -> composed helper returns resolved with
      reason=llm_selected -> Stage B hands off with
      resolution_outcome=llm_fallback."""

    def test(self, monkeypatch, caplog):
        # Turn 1 state:
        staged = _after_turn_1_recommender_rendered(pending_drilldown=False)

        # LLM enabled for the resolver; stub the internal call so the
        # composition (deterministic -> LLM) runs live, but the actual
        # Anthropic tool_use round-trip is short-circuited.
        from skillbridge.chat import reference_resolver
        monkeypatch.setattr(reference_resolver, "LLM_ENABLED", True)
        def _fake_llm_call(message, surface_items):
            # Sanity: verify composition passed the near-miss message
            # + the seeded surface into the LLM stub.
            assert "admin secretary" in message.lower()
            assert len(surface_items) == 2
            return "item_1"
        monkeypatch.setattr(
            reference_resolver, "_call_reference_llm", _fake_llm_call,
        )

        with caplog.at_level(
            logging.INFO, logger="skillbridge.chat.frame_telemetry"
        ):
            result = _maybe_route_recommender_from_intent(
                staged=staged,
                message="match me to admin secretary",
                store=_FakeStore(),
            )

        # Handoff to item 1 (Administrative assistant, NOC 13110).
        assert result is None
        assert staged.target_role_text == "Administrative assistant"
        assert staged.target_noc == "13110"
        # Telemetry proves the LLM path fired (deterministic would set
        # resolution_outcome=deterministic).
        rec = _find_telemetry(caplog, "route_handoff_matching")
        assert "resolution_outcome=llm_fallback" in rec.getMessage()


# ---------------------------------------------------------------- scenario 6


class TestScenario6_KindGuardJobFallthrough:
    """Kind guard end-to-end via Stage B: the frame's latest surface
    is matching titles (job kind, from a prior matching turn). User
    says "match me to the first" -- deterministic resolves item 1
    successfully, but its kind is "job" not "role". Stage B's kind
    check skips the handoff (Step 2.5's handoff helper is role-only
    and would raise ValueError on job kind).

    Cross-turn contract: the matching titles surface must survive
    unchanged when the kind guard trips. Handoff must not fire."""

    def test(self, monkeypatch):
        # Turn 1 state: matching titles rendered, no recommender turn.
        staged = _after_turn_1_matching_rendered()

        from skillbridge.chat import reference_resolver
        monkeypatch.setattr(reference_resolver, "LLM_ENABLED", False)
        from skillbridge.chat import recommender_intent
        monkeypatch.setattr(recommender_intent, "LLM_ENABLED", False)

        # Also raise if handoff helper is invoked -- kind guard
        # holding is what makes this pass.
        from skillbridge.chat import reference_handoff
        def _boom(*a, **k):
            raise AssertionError(
                "handoff invoked despite job kind -- Stage B kind guard broke"
            )
        monkeypatch.setattr(
            reference_handoff,
            "handoff_recommender_to_matching",
            _boom,
        )

        result = _maybe_route_recommender_from_intent(
            staged=staged,
            message="match me to the first",
            store=_FakeStore(),
        )
        # No handoff, target untouched.
        assert staged.target_role_text is None
        assert staged.target_noc is None
        # Matching titles surface still visible for the next turn.
        assert staged.last_presented_job_titles == [
            "Truck driver at Acme",
            "Delivery driver at Sault Co",
        ]
        assert staged.last_presented_at_turn == 3
        # Downstream classifier + router chain then runs (returns None
        # under LLM-disabled unclear + default verdict). What matters
        # here: Stage B fell through cleanly without corrupting state.
        assert result is None
