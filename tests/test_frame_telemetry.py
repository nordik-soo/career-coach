"""Unit tests for the frame_telemetry emitter (Step 1.4).

DB-free, no LLM, no session store. Constructs StagedProfile instances
directly, calls the emitter, captures the log record via caplog, and
asserts the locked format + field values + privacy invariants.

Locked contracts under test:
  - One INFO log line per emit call with the frame_telemetry marker.
  - All locked fields present: session, path, msg_count, pattern_intent,
    career_intent, router_action, router_reason, pending_before,
    pending_after, active_target_noc, latest_surface,
    latest_surface_at_turn, last_engine.
  - `none` sentinel for unset optional fields (no missing keys, no
    empty values).
  - Privacy: user_message text and target_role_text never appear in
    the emitted record.
  - snapshot_pending_flags returns ALL live pending flags in
    deterministic precedence order.
  - emit_frame_telemetry never raises even if the frame derivation
    somehow fails.
"""
from __future__ import annotations

import logging
import re

import pytest

from skillbridge.chat.frame_telemetry import (
    emit_frame_telemetry,
    snapshot_pending_flags,
)
from skillbridge.session.staging import StagedProfile


pytestmark = pytest.mark.nodb


def _new_staged() -> StagedProfile:
    return StagedProfile.new(session_id="test-session-uuid-0001")


def _find_record(caplog) -> logging.LogRecord:
    """Return the single frame_telemetry record emitted; assert exactly one."""
    hits = [
        r for r in caplog.records
        if "frame_telemetry" in r.getMessage()
    ]
    assert len(hits) == 1, (
        f"expected exactly one frame_telemetry record; got {len(hits)}"
    )
    return hits[0]


def _kv(msg: str) -> dict[str, str]:
    """Parse the key=value tokens out of a frame_telemetry log message."""
    # The log format is space-separated key=value pairs after the
    # "frame_telemetry" marker. Values are never-empty tokens (either
    # a literal, an int, or the `none` sentinel), so a simple regex
    # split is enough.
    pairs = re.findall(r"(\w+)=(\S+)", msg)
    return dict(pairs)


# ---------------------------------------------------------------- snapshot


class TestSnapshotPendingFlags:
    def test_empty_profile_no_flags(self):
        s = _new_staged()
        assert snapshot_pending_flags(s) == ()

    def test_all_flags_set_returns_precedence_ordered(self):
        s = _new_staged()
        s.pending_credential_confirmation = {
            "canonical": "class_g", "action": "add",
        }
        s.pending_recommender_offer = "local_gap_coach"
        s.pending_adjacent_search_offer = True
        s.pending_adjacent_offer = True
        flags = snapshot_pending_flags(s)
        assert flags == (
            "credential_confirmation",
            "recommender:local_gap_coach",
            "adjacent_search",
            "adjacent_offer",
        )

    def test_recommender_flag_carries_mode(self):
        s = _new_staged()
        s.pending_recommender_offer = "adjacent_role_drilldown_select"
        assert snapshot_pending_flags(s) == (
            "recommender:adjacent_role_drilldown_select",
        )


# ---------------------------------------------------------------- emit format


class TestEmitFormat:
    def test_emit_produces_one_info_record_with_all_locked_fields(
        self, caplog
    ):
        s = _new_staged()
        s.target_noc = "14200"
        s.pending_recommender_offer = "local_gap_coach"
        with caplog.at_level(
            logging.INFO, logger="skillbridge.chat.frame_telemetry"
        ):
            emit_frame_telemetry(
                staged=s,
                path="unit_test",
                pending_before=("recommender:local_gap_coach",),
                pattern_intent="neutral",
                career_intent="local_skill_gap",
                router_action="recommender_layer",
                router_reason="career_intent_local_skill_gap",
            )
        rec = _find_record(caplog)
        assert rec.levelno == logging.INFO
        kv = _kv(rec.getMessage())
        # Every locked field is present.
        for key in (
            "session", "path", "msg_count", "pattern_intent",
            "career_intent", "router_action", "router_reason",
            "pending_before", "pending_after", "active_target_noc",
            "latest_surface", "latest_surface_at_turn", "last_engine",
        ):
            assert key in kv, f"locked field missing: {key}"

    def test_locked_field_values_from_typical_state(self, caplog):
        s = _new_staged()
        s.target_noc = "14200"
        s.pending_recommender_offer = "local_gap_coach"
        with caplog.at_level(
            logging.INFO, logger="skillbridge.chat.frame_telemetry"
        ):
            emit_frame_telemetry(
                staged=s,
                path="unit_test",
                pending_before=("recommender:local_gap_coach",),
                pattern_intent="neutral",
                career_intent="local_skill_gap",
                router_action="recommender_layer",
                router_reason="career_intent_local_skill_gap",
            )
        kv = _kv(_find_record(caplog).getMessage())
        assert kv["session"] == "test-ses"  # 8-char prefix
        assert kv["path"] == "unit_test"
        assert kv["pattern_intent"] == "neutral"
        assert kv["career_intent"] == "local_skill_gap"
        assert kv["router_action"] == "recommender_layer"
        assert kv["router_reason"] == "career_intent_local_skill_gap"
        assert kv["pending_before"] == "recommender:local_gap_coach"
        assert kv["pending_after"] == "recommender:local_gap_coach"
        assert kv["active_target_noc"] == "14200"
        assert kv["latest_surface"] == "none"
        assert kv["latest_surface_at_turn"] == "none"
        assert kv["last_engine"] == "none"

    def test_multiple_pending_flags_pipe_joined(self, caplog):
        s = _new_staged()
        s.pending_credential_confirmation = {
            "canonical": "class_g", "action": "add",
        }
        s.pending_recommender_offer = "local_gap_coach"
        with caplog.at_level(
            logging.INFO, logger="skillbridge.chat.frame_telemetry"
        ):
            emit_frame_telemetry(
                staged=s,
                path="unit_test",
                pending_before=snapshot_pending_flags(s),
            )
        kv = _kv(_find_record(caplog).getMessage())
        assert kv["pending_before"] == (
            "credential_confirmation|recommender:local_gap_coach"
        )


class TestNoneSentinel:
    def test_all_optional_fields_render_none_when_absent(self, caplog):
        s = _new_staged()
        with caplog.at_level(
            logging.INFO, logger="skillbridge.chat.frame_telemetry"
        ):
            emit_frame_telemetry(
                staged=s,
                path="unit_test",
                pending_before=(),
                # pattern_intent / career_intent / router_action /
                # router_reason all default to None (consume-hook path).
            )
        kv = _kv(_find_record(caplog).getMessage())
        assert kv["pattern_intent"] == "none"
        assert kv["career_intent"] == "none"
        assert kv["router_action"] == "none"
        assert kv["router_reason"] == "none"
        assert kv["pending_before"] == "none"
        assert kv["pending_after"] == "none"
        assert kv["active_target_noc"] == "none"
        assert kv["latest_surface"] == "none"
        assert kv["latest_surface_at_turn"] == "none"
        assert kv["last_engine"] == "none"


class TestBeforeAfterDiff:
    def test_pending_cleared_between_before_and_after_visible_in_log(
        self, caplog
    ):
        """Simulates a pivot-clear scenario: before snapshot captures
        the pending offer, then downstream code clears it, then
        telemetry emits. The line should show the diff."""
        s = _new_staged()
        s.pending_recommender_offer = "local_gap_coach"
        before = snapshot_pending_flags(s)
        # Simulate the pivot-clear.
        s.pending_recommender_offer = None
        with caplog.at_level(
            logging.INFO, logger="skillbridge.chat.frame_telemetry"
        ):
            emit_frame_telemetry(
                staged=s,
                path="unit_test",
                pending_before=before,
            )
        kv = _kv(_find_record(caplog).getMessage())
        assert kv["pending_before"] == "recommender:local_gap_coach"
        assert kv["pending_after"] == "none"


class TestSurfaceAndEngineFields:
    def test_recommender_surface_reflected(self, caplog):
        s = _new_staged()
        s.last_recommender_adjacent_surface = (
            {"noc_code": "13110", "title": "Administrative assistant"},
        )
        s.last_recommender_adjacent_surface_at_turn = 5
        with caplog.at_level(
            logging.INFO, logger="skillbridge.chat.frame_telemetry"
        ):
            emit_frame_telemetry(
                staged=s,
                path="unit_test",
                pending_before=(),
            )
        kv = _kv(_find_record(caplog).getMessage())
        assert kv["latest_surface"] == "adjacent_recs"
        assert kv["latest_surface_at_turn"] == "5"
        assert kv["last_engine"] == "recommender"

    def test_matching_surface_reflected(self, caplog):
        s = _new_staged()
        s.last_presented_job_titles = ["AP Clerk", "Bookkeeper"]
        s.last_presented_at_turn = 3
        with caplog.at_level(
            logging.INFO, logger="skillbridge.chat.frame_telemetry"
        ):
            emit_frame_telemetry(
                staged=s,
                path="unit_test",
                pending_before=(),
            )
        kv = _kv(_find_record(caplog).getMessage())
        assert kv["latest_surface"] == "matches"
        assert kv["latest_surface_at_turn"] == "3"
        assert kv["last_engine"] == "matching"


# ---------------------------------------------------------------- privacy


class TestPrivacyInvariant:
    def test_target_role_text_never_appears_in_log(self, caplog):
        """Locked privacy contract: free-text fields never emit.
        target_role_text can carry PII (someone types 'my brother is a
        nurse in Toronto') and must not touch the telemetry surface."""
        s = _new_staged()
        s.target_role_text = "ACCOUNTING_CLERK_WITH_SENSITIVE_TEXT"
        s.target_noc = "14200"
        with caplog.at_level(
            logging.INFO, logger="skillbridge.chat.frame_telemetry"
        ):
            emit_frame_telemetry(
                staged=s,
                path="unit_test",
                pending_before=(),
            )
        msg = _find_record(caplog).getMessage()
        assert "ACCOUNTING_CLERK_WITH_SENSITIVE_TEXT" not in msg
        # NOC code IS allowed (it's a stable enum).
        assert "14200" in msg

    def test_user_message_never_appears_in_log(self, caplog):
        """The emitter has no user_message parameter -- this is a
        structural privacy invariant. Sanity-checked here so a future
        refactor that adds one to the signature fails this test."""
        import inspect
        sig = inspect.signature(emit_frame_telemetry)
        assert "user_message" not in sig.parameters
        assert "message" not in sig.parameters


# ---------------------------------------------------------------- robustness


class TestNoRaise:
    def test_missing_session_id_falls_back_to_none_token(self, caplog):
        """Defensive: a StagedProfile with an empty/None session_id
        must not crash the emitter."""
        s = _new_staged()
        # Direct __dict__ write to bypass any dataclass invariant.
        s.__dict__["session_id"] = ""
        with caplog.at_level(
            logging.INFO, logger="skillbridge.chat.frame_telemetry"
        ):
            emit_frame_telemetry(
                staged=s,
                path="unit_test",
                pending_before=(),
            )
        # A record was still emitted; session field falls back to none.
        kv = _kv(_find_record(caplog).getMessage())
        assert kv["session"] == "none"

    def test_emit_swallows_derivation_failure(self, caplog, monkeypatch):
        """If the frame derivation raises (shouldn't; it's defensive),
        the emitter must log the failure and continue -- telemetry is
        best-effort observability, not a blocking side effect. Any
        exception here would crash chat turns."""
        from skillbridge.chat import frame_telemetry as ft
        def _boom(_staged):
            raise RuntimeError("simulated derive_frame failure")
        monkeypatch.setattr(ft, "derive_frame", _boom)
        s = _new_staged()
        # Must not raise.
        with caplog.at_level(
            logging.INFO, logger="skillbridge.chat.frame_telemetry"
        ):
            emit_frame_telemetry(
                staged=s,
                path="unit_test",
                pending_before=(),
            )
        # The exception was swallowed and logged.
        assert any(
            "emit_failed" in r.getMessage()
            for r in caplog.records
        )
