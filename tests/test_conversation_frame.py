"""Unit tests for derive_frame (Step 1.1 of memory/routing refactor).

DB-free, no LLM, no session store. Constructs StagedProfile instances
directly and asserts the derived frame shape.

Locked contracts under test:
  - Pending precedence: credential > recommender > adjacent_search > adjacent_offer.
  - Surface item normalization to {kind, label, id, ordinal}.
  - Purity: derive_frame never mutates staged.
  - Frozen result: ConversationFrame and SurfaceItem cannot be mutated.
"""
from __future__ import annotations

import pytest

from skillbridge.chat.conversation_frame import (
    ConversationFrame,
    SurfaceItem,
    derive_frame,
)
from skillbridge.session.staging import StagedProfile


pytestmark = pytest.mark.nodb


def _new_staged() -> StagedProfile:
    """Fresh staged profile with defaults. Tests set fields as needed."""
    return StagedProfile.new(session_id="test-session-uuid-0001")


# ---------------------------------------------------------------- empty


class TestEmptyProfile:
    def test_empty_profile_has_none_everywhere(self):
        staged = _new_staged()
        frame = derive_frame(staged)
        assert isinstance(frame, ConversationFrame)
        assert frame.active_target_role is None
        assert frame.active_target_noc is None
        assert frame.active_pending_offer is None
        assert frame.latest_surface_type == "none"
        assert frame.latest_surface_at_turn is None
        assert frame.latest_surface_items == ()
        assert frame.last_engine_used == "none"
        assert frame.available_referents == ()


# ---------------------------------------------------------------- target


class TestTargetPassthrough:
    def test_target_role_and_noc_pass_through(self):
        staged = _new_staged()
        staged.target_role_text = "accounting clerk"
        staged.target_noc = "14200"
        frame = derive_frame(staged)
        assert frame.active_target_role == "accounting clerk"
        assert frame.active_target_noc == "14200"

    def test_empty_string_target_role_reads_as_none(self):
        staged = _new_staged()
        # Direct __dict__ write bypasses the __setattr__ clearing side
        # effects; we're specifically testing the frame's or-None
        # normalization of an empty string.
        staged.__dict__["target_role_text"] = ""
        frame = derive_frame(staged)
        assert frame.active_target_role is None


# ---------------------------------------------------------------- pending precedence


class TestPendingPrecedence:
    """Locked precedence: credential > recommender > adjacent_search > adjacent_offer."""

    def test_credential_wins_over_all_others(self):
        staged = _new_staged()
        staged.pending_credential_confirmation = {
            "canonical": "class_g_licence",
            "action": "add",
        }
        staged.pending_recommender_offer = "local_gap_coach"
        staged.pending_adjacent_search_offer = True
        staged.pending_adjacent_offer = True
        frame = derive_frame(staged)
        assert frame.active_pending_offer == "credential_confirmation"

    def test_recommender_wins_over_adjacent_offers(self):
        staged = _new_staged()
        staged.pending_recommender_offer = "target_noc_standard"
        staged.pending_adjacent_search_offer = True
        staged.pending_adjacent_offer = True
        frame = derive_frame(staged)
        assert frame.active_pending_offer == "recommender:target_noc_standard"

    def test_adjacent_search_wins_over_adjacent_offer(self):
        staged = _new_staged()
        staged.pending_adjacent_search_offer = True
        staged.pending_adjacent_offer = True
        frame = derive_frame(staged)
        assert frame.active_pending_offer == "adjacent_search"

    def test_adjacent_offer_alone(self):
        staged = _new_staged()
        staged.pending_adjacent_offer = True
        frame = derive_frame(staged)
        assert frame.active_pending_offer == "adjacent_offer"

    def test_recommender_offer_carries_mode(self):
        staged = _new_staged()
        staged.pending_recommender_offer = "adjacent_noc_standard"
        frame = derive_frame(staged)
        assert frame.active_pending_offer == "recommender:adjacent_noc_standard"

    def test_recommender_drilldown_mode_carried(self):
        staged = _new_staged()
        staged.pending_recommender_offer = "adjacent_role_drilldown_select"
        frame = derive_frame(staged)
        assert (
            frame.active_pending_offer
            == "recommender:adjacent_role_drilldown_select"
        )

    def test_no_pending_flags_returns_none(self):
        staged = _new_staged()
        frame = derive_frame(staged)
        assert frame.active_pending_offer is None


# ---------------------------------------------------------------- surfaces


class TestRecommenderSurface:
    def test_recommender_role_surface(self):
        staged = _new_staged()
        staged.last_recommender_adjacent_surface = (
            {"noc_code": "13110", "title": "Administrative assistant"},
            {"noc_code": "14200", "title": "Accounting clerk"},
        )
        frame = derive_frame(staged)
        assert frame.latest_surface_type == "adjacent_recs"
        assert len(frame.latest_surface_items) == 2
        first = frame.latest_surface_items[0]
        assert first.kind == "role"
        assert first.label == "Administrative assistant"
        assert first.id == "13110"
        assert first.ordinal == 1
        assert frame.latest_surface_items[1].ordinal == 2
        assert frame.available_referents == (
            "Administrative assistant",
            "Accounting clerk",
        )
        assert frame.last_engine_used == "recommender"

    def test_recommender_surface_skips_malformed_entries(self):
        staged = _new_staged()
        staged.last_recommender_adjacent_surface = (
            {"noc_code": "13110", "title": "Administrative assistant"},
            "not a dict",  # type: ignore[list-item]
            {"noc_code": "14200"},  # missing title
            {"noc_code": "10000", "title": "   "},  # blank title
            {"noc_code": "22222", "title": "Roofer"},
        )
        frame = derive_frame(staged)
        # Two valid entries; malformed skipped.
        assert len(frame.latest_surface_items) == 2
        assert frame.latest_surface_items[0].label == "Administrative assistant"
        assert frame.latest_surface_items[1].label == "Roofer"
        # Ordinal reflects source index, not filtered index. Source index
        # of "Roofer" is 4 (0-based) so ordinal is 5.
        assert frame.latest_surface_items[0].ordinal == 1
        assert frame.latest_surface_items[1].ordinal == 5

    def test_recommender_surface_missing_noc_id_becomes_none(self):
        staged = _new_staged()
        staged.last_recommender_adjacent_surface = (
            {"noc_code": "", "title": "Untitled role"},
            {"title": "No code field"},
        )
        frame = derive_frame(staged)
        assert len(frame.latest_surface_items) == 2
        assert frame.latest_surface_items[0].id is None
        assert frame.latest_surface_items[1].id is None


class TestMatchingSurface:
    def test_matching_job_titles_surface(self):
        staged = _new_staged()
        staged.last_presented_job_titles = ["Truck driver", "Delivery driver"]
        frame = derive_frame(staged)
        assert frame.latest_surface_type == "matches"
        assert len(frame.latest_surface_items) == 2
        first = frame.latest_surface_items[0]
        assert first.kind == "job"
        assert first.label == "Truck driver"
        assert first.id is None  # last_presented_job_titles has no id
        assert first.ordinal == 1
        assert frame.last_engine_used == "matching"

    def test_matching_snapshot_only_marks_engine(self):
        """last_match_snapshot alone (no titles) still counts as
        matching for last_engine_used, though it does not populate
        latest_surface_items (that path reads titles / adjacent items)."""
        staged = _new_staged()
        staged.last_match_snapshot = {"jobs": []}
        frame = derive_frame(staged)
        assert frame.last_engine_used == "matching"
        assert frame.latest_surface_type == "none"
        assert frame.latest_surface_items == ()

    def test_matching_titles_skip_blank_entries(self):
        staged = _new_staged()
        staged.last_presented_job_titles = ["Truck driver", "", "  ", "Cook"]
        frame = derive_frame(staged)
        assert len(frame.latest_surface_items) == 2
        assert frame.latest_surface_items[0].label == "Truck driver"
        assert frame.latest_surface_items[1].label == "Cook"
        # Source-index ordinals: Cook is at position 3 -> ordinal 4.
        assert frame.latest_surface_items[1].ordinal == 4


class TestAdjacentSnapshotSurface:
    def test_adjacent_snapshot_populates_surface_items_and_anchor(self):
        staged = _new_staged()
        staged.last_adjacent_snapshot = {
            "created_message_count": 5,
            "items": [
                {"job_id": "j1", "title": "Warehouse associate"},
                {"job_id": "j2", "title": "Order picker"},
            ],
        }
        frame = derive_frame(staged)
        assert frame.latest_surface_type == "matches"
        assert frame.latest_surface_at_turn == 5
        assert frame.latest_surface_items[0].kind == "job"
        assert frame.latest_surface_items[0].id == "j1"
        assert frame.latest_surface_items[1].id == "j2"

    def test_adjacent_snapshot_missing_created_yields_null_anchor(self):
        staged = _new_staged()
        staged.last_adjacent_snapshot = {
            "items": [{"job_id": "j1", "title": "Warehouse associate"}],
        }
        frame = derive_frame(staged)
        assert frame.latest_surface_type == "matches"
        assert frame.latest_surface_at_turn is None

    def test_adjacent_snapshot_alone_counts_as_matching_engine(self):
        """last_adjacent_snapshot is the matching engine's sideways_move
        tier output -- an adjacent-JOB surface -- so a session with only
        this snapshot must report last_engine_used == 'matching', not
        'none'. Caught in live review before Step 1.1 commit."""
        staged = _new_staged()
        staged.last_adjacent_snapshot = {
            "created_message_count": 3,
            "items": [{"job_id": "j1", "title": "Warehouse associate"}],
        }
        frame = derive_frame(staged)
        assert frame.last_engine_used == "matching"

    def test_adjacent_snapshot_with_anchor_beats_unanchored_recommender_surface(
        self,
    ):
        """Both engines have produced output. Adjacent snapshot carries
        its own created_message_count anchor (3); recommender surface
        has no anchor. Post-Step-1.2 rule: anchored side wins over the
        unanchored side, so matching."""
        staged = _new_staged()
        staged.last_adjacent_snapshot = {
            "created_message_count": 3,
            "items": [{"job_id": "j1", "title": "Warehouse associate"}],
        }
        staged.last_recommender_adjacent_surface = (
            {"noc_code": "13110", "title": "Administrative assistant"},
        )
        frame = derive_frame(staged)
        assert frame.last_engine_used == "matching"

    def test_both_engines_unanchored_stopgap_picks_recommender(self):
        """Legacy path: neither side carries an anchor (pre-Step-1.2
        session or malformed cookie). The static stopgap picks
        recommender. Regression guard against the stopgap silently
        changing behavior."""
        staged = _new_staged()
        # last_match_snapshot is a matching signal but has no anchor
        # field of its own; last_presented_at_turn stays None.
        staged.last_match_snapshot = {"jobs": []}
        staged.last_recommender_adjacent_surface = (
            {"noc_code": "13110", "title": "Administrative assistant"},
        )
        # No anchors set on either side.
        frame = derive_frame(staged)
        assert frame.last_engine_used == "recommender"


class TestSurfacePrecedence:
    def test_recommender_surface_wins_over_matching_stopgap(self):
        """Step 1.1 stopgap: static precedence puts recommender surface
        ahead of matching when message_count anchors are absent.
        Step 1.2 replaces with message_count ordering; the public shape
        does not change."""
        staged = _new_staged()
        staged.last_presented_job_titles = ["Truck driver"]
        staged.last_recommender_adjacent_surface = (
            {"noc_code": "13110", "title": "Administrative assistant"},
        )
        frame = derive_frame(staged)
        assert frame.latest_surface_type == "adjacent_recs"
        assert frame.latest_surface_items[0].label == "Administrative assistant"
        assert frame.last_engine_used == "recommender"

    def test_adjacent_snapshot_anchor_beats_recommender_without_anchor(self):
        """When one surface has an anchor and another does not, the
        anchored one wins. Simulates a scenario where Layer C rendered
        earlier (no anchor field until Step 1.2) and adjacent snapshot
        rendered more recently (carries its own created_message_count)."""
        staged = _new_staged()
        staged.last_recommender_adjacent_surface = (
            {"noc_code": "13110", "title": "Administrative assistant"},
        )
        staged.last_adjacent_snapshot = {
            "created_message_count": 7,
            "items": [{"job_id": "j1", "title": "Order picker"}],
        }
        frame = derive_frame(staged)
        assert frame.latest_surface_type == "matches"
        assert frame.latest_surface_at_turn == 7
        assert frame.latest_surface_items[0].id == "j1"


# ---------------------------------------------------------------- structural


class TestFrozen:
    def test_frame_cannot_be_mutated(self):
        staged = _new_staged()
        frame = derive_frame(staged)
        with pytest.raises((AttributeError, TypeError)):
            frame.active_target_role = "something"  # type: ignore[misc]

    def test_surface_item_cannot_be_mutated(self):
        item = SurfaceItem(kind="job", label="x", id=None, ordinal=1)
        with pytest.raises((AttributeError, TypeError)):
            item.label = "y"  # type: ignore[misc]


class TestPurity:
    def test_derive_does_not_mutate_staged(self):
        staged = _new_staged()
        staged.target_role_text = "software developer"
        staged.pending_recommender_offer = "local_gap_coach"
        staged.last_presented_job_titles = ["Junior Developer", "Analyst"]
        # Snapshot pre-call state.
        pre_target = staged.target_role_text
        pre_offer = staged.pending_recommender_offer
        pre_titles = list(staged.last_presented_job_titles)
        pre_message_count = staged.message_count
        _ = derive_frame(staged)
        assert staged.target_role_text == pre_target
        assert staged.pending_recommender_offer == pre_offer
        assert list(staged.last_presented_job_titles) == pre_titles
        assert staged.message_count == pre_message_count


# ---------------------------------------------------------------- integration


# ---------------------------------------------------------------- anchors (Step 1.2)


class TestAnchorFields:
    """Step 1.2 landed message_count anchors on StagedProfile. These
    tests exercise the deterministic ordering paths that supersede the
    slice-1 static tiebreaker."""

    def test_anchor_fields_default_to_none(self):
        staged = _new_staged()
        assert staged.last_recommender_adjacent_surface_at_turn is None
        assert staged.last_presented_at_turn is None

    def test_recommender_anchor_beats_matching_anchor_when_higher(self):
        """Recommender rendered later (turn 8) than matching (turn 5).
        Deterministic ordering picks recommender even though matching
        is second in the static precedence list."""
        staged = _new_staged()
        staged.last_presented_job_titles = ["Truck driver"]
        staged.last_presented_at_turn = 5
        staged.last_recommender_adjacent_surface = (
            {"noc_code": "13110", "title": "Administrative assistant"},
        )
        staged.last_recommender_adjacent_surface_at_turn = 8
        frame = derive_frame(staged)
        assert frame.latest_surface_type == "adjacent_recs"
        assert frame.latest_surface_at_turn == 8
        assert frame.last_engine_used == "recommender"

    def test_matching_anchor_beats_recommender_anchor_when_higher(self):
        """Matching rendered later (turn 9) than recommender (turn 4).
        Deterministic ordering picks matching even though recommender
        is first in the static precedence list."""
        staged = _new_staged()
        staged.last_recommender_adjacent_surface = (
            {"noc_code": "13110", "title": "Administrative assistant"},
        )
        staged.last_recommender_adjacent_surface_at_turn = 4
        staged.last_presented_job_titles = ["Truck driver"]
        staged.last_presented_at_turn = 9
        frame = derive_frame(staged)
        assert frame.latest_surface_type == "matches"
        assert frame.latest_surface_at_turn == 9
        assert frame.latest_surface_items[0].label == "Truck driver"
        assert frame.last_engine_used == "matching"

    def test_tied_anchors_fall_back_to_static_precedence(self):
        """Same anchor value -> static tiebreaker (recommender wins)."""
        staged = _new_staged()
        staged.last_presented_job_titles = ["Truck driver"]
        staged.last_presented_at_turn = 7
        staged.last_recommender_adjacent_surface = (
            {"noc_code": "13110", "title": "Administrative assistant"},
        )
        staged.last_recommender_adjacent_surface_at_turn = 7
        frame = derive_frame(staged)
        assert frame.latest_surface_type == "adjacent_recs"
        assert frame.last_engine_used == "recommender"

    def test_anchor_zero_is_valid_and_wins_over_none(self):
        """message_count starts at 0; 0 is a legitimate anchor value.
        An anchored side (turn=0) beats an unanchored side."""
        staged = _new_staged()
        # Unanchored matching surface (legacy blob).
        staged.last_presented_job_titles = ["Truck driver"]
        # Anchored recommender surface at turn 0.
        staged.last_recommender_adjacent_surface = (
            {"noc_code": "13110", "title": "Administrative assistant"},
        )
        staged.last_recommender_adjacent_surface_at_turn = 0
        frame = derive_frame(staged)
        assert frame.latest_surface_type == "adjacent_recs"
        assert frame.latest_surface_at_turn == 0
        assert frame.last_engine_used == "recommender"

    def test_matching_engine_uses_adjacent_snapshot_created_count(self):
        """The adjacent-JOB snapshot carries its own created_message_count.
        The engine derivation must fold it into matching's anchor when
        picking the winner."""
        staged = _new_staged()
        staged.last_recommender_adjacent_surface = (
            {"noc_code": "13110", "title": "Administrative assistant"},
        )
        staged.last_recommender_adjacent_surface_at_turn = 3
        staged.last_adjacent_snapshot = {
            "created_message_count": 6,
            "items": [{"job_id": "j1", "title": "Order picker"}],
        }
        frame = derive_frame(staged)
        # Both surfaces present; matching's anchor (6) > recommender's (3).
        assert frame.last_engine_used == "matching"


class TestTargetChangeClearsRecommenderAnchor:
    """Step 1.2 adds the anchor to the __setattr__ clear list for
    target_role_text changes, matching the surface it points to."""

    def test_target_change_clears_recommender_anchor(self):
        staged = _new_staged()
        staged.target_role_text = "accounting clerk"
        staged.last_recommender_adjacent_surface = (
            {"noc_code": "13110", "title": "Administrative assistant"},
        )
        staged.last_recommender_adjacent_surface_at_turn = 5
        # Sanity: surface + anchor both live.
        assert staged.last_recommender_adjacent_surface
        assert staged.last_recommender_adjacent_surface_at_turn == 5
        # Target switch.
        staged.target_role_text = "warehouse worker"
        # Both surface and anchor cleared together.
        assert staged.last_recommender_adjacent_surface == ()
        assert staged.last_recommender_adjacent_surface_at_turn is None

    def test_target_change_does_not_clear_matching_anchor(self):
        """last_presented_* fields survive a target switch by existing
        design (per-turn fallback that can be reused across a target
        change); the matching anchor follows the same lifecycle."""
        staged = _new_staged()
        staged.target_role_text = "accounting clerk"
        staged.last_presented_job_titles = ["AP Clerk"]
        staged.last_presented_at_turn = 5
        staged.target_role_text = "warehouse worker"
        # Anchor + titles both survive.
        assert staged.last_presented_job_titles == ["AP Clerk"]
        assert staged.last_presented_at_turn == 5


class TestCookieRoundTrip:
    """Verify anchors survive to_json / from_json cleanly and that
    None defaults minify losslessly in cookie mode."""

    def test_none_anchors_minify_and_restore(self):
        staged = _new_staged()
        # Both anchors at default None.
        raw = staged.to_json(redact_for_cookie=True)
        # Lossless minification drops the keys.
        assert "last_recommender_adjacent_surface_at_turn" not in raw
        assert "last_presented_at_turn" not in raw
        # Round trip restores the defaults.
        restored = StagedProfile.from_json(raw)
        assert restored.last_recommender_adjacent_surface_at_turn is None
        assert restored.last_presented_at_turn is None

    def test_int_anchors_survive_round_trip(self):
        staged = _new_staged()
        staged.last_presented_job_titles = ["AP Clerk"]
        staged.last_presented_at_turn = 12
        staged.last_recommender_adjacent_surface = (
            {"noc_code": "13110", "title": "Administrative assistant"},
        )
        staged.last_recommender_adjacent_surface_at_turn = 15
        raw = staged.to_json(redact_for_cookie=True)
        restored = StagedProfile.from_json(raw)
        assert restored.last_presented_at_turn == 12
        assert restored.last_recommender_adjacent_surface_at_turn == 15

    def test_zero_anchor_survives_round_trip(self):
        """0 is a valid anchor (message_count starts there); must not
        be minified as if it were None."""
        staged = _new_staged()
        staged.last_presented_job_titles = ["AP Clerk"]
        staged.last_presented_at_turn = 0
        raw = staged.to_json(redact_for_cookie=True)
        assert "last_presented_at_turn" in raw
        restored = StagedProfile.from_json(raw)
        assert restored.last_presented_at_turn == 0

    def test_forged_string_anchor_coerces_to_none(self):
        """Defensive deserialize: a forged cookie carrying a string in
        an int|None slot must coerce to None, not blow up the
        dataclass."""
        # Build a minimal valid cookie then tamper.
        import json
        staged = _new_staged()
        data = json.loads(staged.to_json(redact_for_cookie=True))
        data["last_presented_at_turn"] = "not-a-number"
        data["last_recommender_adjacent_surface_at_turn"] = True  # bool sneaks past int check
        forged = json.dumps(data)
        restored = StagedProfile.from_json(forged)
        assert restored.last_presented_at_turn is None
        assert restored.last_recommender_adjacent_surface_at_turn is None

    def test_negative_anchor_coerces_to_none(self):
        """Anchors are non-negative by contract (message_count only
        grows). A negative int is malformed and must coerce to None."""
        import json
        staged = _new_staged()
        data = json.loads(staged.to_json(redact_for_cookie=True))
        data["last_presented_at_turn"] = -5
        forged = json.dumps(data)
        restored = StagedProfile.from_json(forged)
        assert restored.last_presented_at_turn is None


class TestFullyPopulatedProfile:
    def test_typical_mid_recommender_flow(self):
        """A realistic session: target set, matching ran, recommender
        Layer C also ran and left the surface + pending offer."""
        staged = _new_staged()
        staged.target_role_text = "accounting clerk"
        staged.target_noc = "14200"
        staged.last_presented_job_titles = ["AP Clerk", "Bookkeeper"]
        staged.last_match_snapshot = {"jobs": [{"title": "AP Clerk"}]}
        staged.last_recommender_adjacent_surface = (
            {"noc_code": "13110", "title": "Administrative assistant"},
            {"noc_code": "14400", "title": "Data entry clerk"},
        )
        staged.pending_recommender_offer = "adjacent_role_drilldown_select"
        frame = derive_frame(staged)
        assert frame.active_target_role == "accounting clerk"
        assert frame.active_target_noc == "14200"
        assert (
            frame.active_pending_offer
            == "recommender:adjacent_role_drilldown_select"
        )
        # Recommender surface wins under slice-1 stopgap.
        assert frame.latest_surface_type == "adjacent_recs"
        assert len(frame.latest_surface_items) == 2
        assert frame.latest_surface_items[0].kind == "role"
        assert frame.last_engine_used == "recommender"
        assert frame.available_referents == (
            "Administrative assistant",
            "Data entry clerk",
        )
