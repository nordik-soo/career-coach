"""Unit tests for _clear_recommender_state_on_pivot (Step 1.3).

DB-free, no LLM, no store. Constructs StagedProfile + RecommenderRouteVerdict
instances directly and asserts the helper's side effects on staged
plus the tuple of cleared field names it returns.

Locked contracts under test:
  - Pivot verdicts (matching_engine / out_of_scope_canned / mode-switch
    recommender_layer) clear pending_recommender_offer +
    last_recommender_adjacent_surface + last_recommender_adjacent_surface_at_turn.
  - Non-pivot verdicts (recommender_layer same mode / ask_substrate /
    default) never clear.
  - pending_adjacent_search_offer is NOT touched by this helper -- it is
    consumed unconditionally at handler entry (see slice-1 follow-up).
"""
from __future__ import annotations

import pytest

from skillbridge.chat.handler import _clear_recommender_state_on_pivot
from skillbridge.chat.recommender_route import RecommenderRouteVerdict
from skillbridge.session.staging import StagedProfile


pytestmark = pytest.mark.nodb


def _staged_with_pending_local_gap_coach() -> StagedProfile:
    """Staged profile with a live local_gap_coach pending offer +
    adjacent surface (with anchor). Baseline fixture for pivot tests."""
    s = StagedProfile.new(session_id="test-session-uuid-0001")
    s.pending_recommender_offer = "local_gap_coach"
    s.last_recommender_adjacent_surface = (
        {"noc_code": "13110", "title": "Administrative assistant"},
        {"noc_code": "14200", "title": "Accounting clerk"},
    )
    s.last_recommender_adjacent_surface_at_turn = 5
    return s


# ---------------------------------------------------------------- pivot verdicts


class TestPivotVerdictsClear:
    def test_matching_engine_verdict_clears_both_state_groups(self):
        s = _staged_with_pending_local_gap_coach()
        v = RecommenderRouteVerdict(
            action="matching_engine",
            reason="career_intent_job_matching",
        )
        cleared = _clear_recommender_state_on_pivot(s, v)
        assert set(cleared) == {
            "pending_recommender_offer",
            "last_recommender_adjacent_surface",
        }
        assert s.pending_recommender_offer is None
        assert s.last_recommender_adjacent_surface == ()
        assert s.last_recommender_adjacent_surface_at_turn is None

    def test_out_of_scope_verdict_clears_both_state_groups(self):
        s = _staged_with_pending_local_gap_coach()
        v = RecommenderRouteVerdict(
            action="out_of_scope_canned",
            reason="oos_application_help",
        )
        cleared = _clear_recommender_state_on_pivot(s, v)
        assert set(cleared) == {
            "pending_recommender_offer",
            "last_recommender_adjacent_surface",
        }
        assert s.pending_recommender_offer is None
        assert s.last_recommender_adjacent_surface == ()
        assert s.last_recommender_adjacent_surface_at_turn is None

    def test_recommender_layer_mode_switch_clears(self):
        """Pending is local_gap_coach; router routes to a DIFFERENT
        mode (target_noc_standard). Pivot -> clear the stale offer +
        surface so the dispatch downstream lands cleanly."""
        s = _staged_with_pending_local_gap_coach()
        v = RecommenderRouteVerdict(
            action="recommender_layer",
            recommender_mode="target_noc_standard",
            voice_hint="noc_standard_comparison",
            reason="career_intent_noc_standard_comparison",
        )
        cleared = _clear_recommender_state_on_pivot(s, v)
        assert "pending_recommender_offer" in cleared
        assert "last_recommender_adjacent_surface" in cleared
        assert s.pending_recommender_offer is None
        assert s.last_recommender_adjacent_surface == ()
        assert s.last_recommender_adjacent_surface_at_turn is None

    def test_recommender_layer_switch_to_adjacent_noc_standard_clears(self):
        """Second concrete mode-switch case: local_gap_coach ->
        adjacent_noc_standard. Same clear semantics."""
        s = _staged_with_pending_local_gap_coach()
        v = RecommenderRouteVerdict(
            action="recommender_layer",
            recommender_mode="adjacent_noc_standard",
            voice_hint="career_exploration",
            reason="career_intent_career_exploration",
        )
        cleared = _clear_recommender_state_on_pivot(s, v)
        assert "pending_recommender_offer" in cleared
        assert s.pending_recommender_offer is None


# ---------------------------------------------------------------- non-pivot verdicts


class TestNonPivotVerdictsPreserve:
    def test_recommender_layer_same_mode_is_chain_continuation(self):
        """Pending is local_gap_coach; router also chose local_gap_coach.
        This is a chain continuation, NOT a pivot. State survives."""
        s = _staged_with_pending_local_gap_coach()
        v = RecommenderRouteVerdict(
            action="recommender_layer",
            recommender_mode="local_gap_coach",
            voice_hint="local_skill_gap",
            reason="career_intent_local_skill_gap",
        )
        cleared = _clear_recommender_state_on_pivot(s, v)
        assert cleared == ()
        assert s.pending_recommender_offer == "local_gap_coach"
        assert len(s.last_recommender_adjacent_surface) == 2
        assert s.last_recommender_adjacent_surface_at_turn == 5

    def test_ask_substrate_preserves_pending_state(self):
        """Substrate ask means user hasn't rejected -- just needs to
        provide missing info. The pending offer must survive to fire
        again when substrate fills."""
        s = _staged_with_pending_local_gap_coach()
        v = RecommenderRouteVerdict(
            action="ask_substrate",
            missing=("target",),
            deferred_intent="local_skill_gap",
            reason="substrate_missing:target",
        )
        cleared = _clear_recommender_state_on_pivot(s, v)
        assert cleared == ()
        assert s.pending_recommender_offer == "local_gap_coach"
        assert len(s.last_recommender_adjacent_surface) == 2
        assert s.last_recommender_adjacent_surface_at_turn == 5

    def test_default_preserves_pending_state(self):
        """Default = classifier unclear or pattern signal that lets
        the existing planner handle. Not an explicit new direction."""
        s = _staged_with_pending_local_gap_coach()
        v = RecommenderRouteVerdict(
            action="default",
            reason="classifier_unclear",
        )
        cleared = _clear_recommender_state_on_pivot(s, v)
        assert cleared == ()
        assert s.pending_recommender_offer == "local_gap_coach"
        assert s.last_recommender_adjacent_surface_at_turn == 5


# ---------------------------------------------------------------- edge cases


class TestEdgeCases:
    def test_no_pending_offer_matching_engine_is_no_op(self):
        """Empty session with matching_engine verdict -- there's
        nothing pending to clear. Returns empty tuple."""
        s = StagedProfile.new(session_id="test-session-empty")
        v = RecommenderRouteVerdict(action="matching_engine", reason="test")
        cleared = _clear_recommender_state_on_pivot(s, v)
        assert cleared == ()
        assert s.pending_recommender_offer is None
        assert s.last_recommender_adjacent_surface == ()

    def test_pending_offer_but_no_surface_clears_pending_only(self):
        """User was mid-recommender chain (pending set) but Layer C
        hasn't been rendered this session (no surface). Pivot clears
        only what was live."""
        s = StagedProfile.new(session_id="test-session-partial")
        s.pending_recommender_offer = "local_gap_coach"
        v = RecommenderRouteVerdict(action="matching_engine", reason="test")
        cleared = _clear_recommender_state_on_pivot(s, v)
        assert cleared == ("pending_recommender_offer",)
        assert s.pending_recommender_offer is None

    def test_recommender_layer_mode_switch_with_no_pending_no_op(self):
        """recommender_layer verdict with a mode but no pending offer
        to switch away from is not a pivot -- there's nothing stale
        to clear. Returns empty tuple."""
        s = StagedProfile.new(session_id="test-session-empty")
        v = RecommenderRouteVerdict(
            action="recommender_layer",
            recommender_mode="local_gap_coach",
            voice_hint="local_skill_gap",
            reason="test",
        )
        cleared = _clear_recommender_state_on_pivot(s, v)
        assert cleared == ()

    def test_recommender_layer_mode_none_never_pivots(self):
        """Defensive: a recommender_layer verdict with recommender_mode
        = None is malformed upstream. The helper treats it as
        non-pivot rather than crash-or-clear."""
        s = _staged_with_pending_local_gap_coach()
        v = RecommenderRouteVerdict(
            action="recommender_layer",
            recommender_mode=None,
            reason="test",
        )
        cleared = _clear_recommender_state_on_pivot(s, v)
        assert cleared == ()
        assert s.pending_recommender_offer == "local_gap_coach"

    def test_pattern_2_flag_never_touched(self):
        """Explicit contract: this helper does NOT clear
        pending_adjacent_search_offer even on a pivot verdict. The
        Pattern 2 flag has its own upstream consume site; a
        router-level clear here would be dead code (see helper
        docstring + slice-1 follow-up)."""
        s = _staged_with_pending_local_gap_coach()
        s.pending_adjacent_search_offer = True
        v = RecommenderRouteVerdict(
            action="matching_engine",
            reason="test",
        )
        _ = _clear_recommender_state_on_pivot(s, v)
        # Recommender state cleared, Pattern 2 flag preserved.
        assert s.pending_recommender_offer is None
        assert s.pending_adjacent_search_offer is True


# ---------------------------------------------------------------- purity


class TestPurity:
    def test_helper_does_not_touch_unrelated_fields(self):
        s = _staged_with_pending_local_gap_coach()
        # Fill in a few unrelated fields to verify they survive.
        s.target_role_text = "accounting clerk"
        s.target_noc = "14200"
        s.pending_credential_confirmation = {
            "canonical": "class_g", "action": "add",
        }
        s.pending_adjacent_search_offer = True
        s.last_presented_job_titles = ["AP Clerk"]
        s.last_presented_at_turn = 3
        v = RecommenderRouteVerdict(action="matching_engine", reason="test")
        _ = _clear_recommender_state_on_pivot(s, v)
        # Recommender state cleared.
        assert s.pending_recommender_offer is None
        assert s.last_recommender_adjacent_surface == ()
        # Everything else untouched.
        assert s.target_role_text == "accounting clerk"
        assert s.target_noc == "14200"
        assert s.pending_credential_confirmation is not None
        assert s.pending_adjacent_search_offer is True
        assert s.last_presented_job_titles == ["AP Clerk"]
        assert s.last_presented_at_turn == 3
