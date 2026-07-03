"""Integration tests for slice 1 (memory/routing refactor).

Scope: orchestration confidence -- prove Step 1.3 pivot-clear and
Step 1.4 telemetry work together across the router + consume paths.
NOT another broad emitter unit suite (Step 1.4 owns that).

DB-free, no LLM, no HTTP. Classifiers and dispatch builders are
monkeypatched to canned returns; state assertions and caplog captures
verify the actual orchestration.

Five scenarios (locked with lead 2026-07-03):
  1. Pending recommender + explicit matching pivot
       -- pivot-clear fires, router falls through to matching flow,
          no frame_telemetry emitted (matching_engine fallthrough is
          deliberately not instrumented).
  2. Pending recommender + same-mode career intent
       -- chain continuation: no pivot-clear, dispatch fires,
          frame_telemetry emitted with path=route_recommender_layer.
  3. Consent yes on target_noc_standard
       -- consume path emits frame_telemetry on a consume_yes_* path.
  4. Drilldown resolver hit
       -- selector maps user message to a NOC, dispatch fires,
          frame_telemetry emitted with path=drilldown_resolver_hit.
  5. Consent "other" fallthrough → router pivot-clear
       -- consume returns None with flag still set; router picks up,
          classifier says job_matching, Step 1.3 clears the stale
          flag, no frame_telemetry emitted for the matching fallthrough.
"""
from __future__ import annotations

import logging
import re

import pytest

from skillbridge.chat import handler as h
from skillbridge.session.staging import StagedProfile, StagedSkill


pytestmark = pytest.mark.nodb


# ---------------------------------------------------------------- fakes


class _FakeStore:
    """Minimal in-memory session store. save() returns the staged
    session_id verbatim so tests can assert on it. load() returns
    None; new_session() returns a canned string. Enough to satisfy the
    _emit_canned_response / _dispatch_recommender_consume / drilldown
    consume call surface."""

    def __init__(self) -> None:
        self.saves: list[StagedProfile] = []

    def save(self, staged: StagedProfile) -> str:
        self.saves.append(staged)
        return staged.session_id

    def load(self, session_id: str | None):  # noqa: ARG002
        return None

    def new_session(self) -> str:
        return "fake-new-session-id"


def _new_staged() -> StagedProfile:
    return StagedProfile.new(session_id="test-session-uuid-0001")


def _new_staged_with_substrate() -> StagedProfile:
    """Staged with target_noc + 5 chat skills (satisfies substrate gate
    without touching resume_facts_json)."""
    s = _new_staged()
    s.target_role_text = "accounting clerk"
    s.target_noc = "14200"
    s.skills_text = "a, b, c, d, e"
    for name in ("a", "b", "c", "d", "e"):
        s.skills.append(StagedSkill(skill_name=name))
    return s


def _messages(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records]


def _grep(msgs: list[str], substr: str) -> list[str]:
    return [m for m in msgs if substr in m]


def _kv(msg: str) -> dict[str, str]:
    return dict(re.findall(r"(\w+)=(\S+)", msg))


# ---------------------------------------------------------------- scenario 1


def test_scenario_1_pending_recommender_matching_pivot(monkeypatch, caplog):
    """Pending offer + user explicitly asks for matches → router
    pivot-clears, falls through to matching (return None), and does
    NOT emit frame_telemetry (matching_engine fallthrough is
    deliberately not instrumented)."""
    staged = _new_staged_with_substrate()
    staged.pending_recommender_offer = "local_gap_coach"
    staged.last_recommender_adjacent_surface = (
        {"noc_code": "13110", "title": "Administrative assistant"},
    )
    staged.last_recommender_adjacent_surface_at_turn = 3

    monkeypatch.setattr(
        "skillbridge.chat.recommender_intent.classify_career_intent",
        lambda **_kw: "job_matching",
    )
    store = _FakeStore()

    with caplog.at_level(logging.INFO):
        result = h._maybe_route_recommender_from_intent(
            staged=staged,
            message="show me admin jobs",
            store=store,
        )

    # Router fell through -- matching flow handles the turn.
    assert result is None
    # Step 1.3 pivot-clear fired.
    assert staged.pending_recommender_offer is None
    assert staged.last_recommender_adjacent_surface == ()
    assert staged.last_recommender_adjacent_surface_at_turn is None
    # Log evidence.
    msgs = _messages(caplog)
    pivot_lines = _grep(msgs, "pivot_cleared=")
    assert len(pivot_lines) == 1
    assert "pending_recommender_offer" in pivot_lines[0]
    assert "last_recommender_adjacent_surface" in pivot_lines[0]
    assert "action=matching_engine" in pivot_lines[0]
    # No frame_telemetry: matching_engine fallthrough is not instrumented.
    assert _grep(msgs, "frame_telemetry") == []


# ---------------------------------------------------------------- scenario 2


def test_scenario_2_pending_recommender_same_mode_chain_continuation(
    monkeypatch, caplog,
):
    """Pending local_gap_coach + user asks a local_skill_gap question
    → same mode → chain continuation, NOT a pivot. Dispatch fires,
    frame_telemetry emits on route_recommender_layer path, no
    pivot_cleared log."""
    staged = _new_staged_with_substrate()
    staged.pending_recommender_offer = "local_gap_coach"
    staged.last_recommender_adjacent_surface = (
        {"noc_code": "13110", "title": "Administrative assistant"},
    )
    staged.last_recommender_adjacent_surface_at_turn = 3

    monkeypatch.setattr(
        "skillbridge.chat.recommender_intent.classify_career_intent",
        lambda **_kw: "local_skill_gap",
    )
    # Dispatch is heavy (engine + evidence + LLM). Patch to canned dict.
    dispatch_calls: list[dict] = []

    def _fake_dispatch(**kwargs):
        dispatch_calls.append(kwargs)
        return {"reply": "canned dispatch reply", "session_id": "fake"}

    monkeypatch.setattr(h, "_dispatch_recommender_from_intent", _fake_dispatch)
    store = _FakeStore()

    with caplog.at_level(logging.INFO):
        result = h._maybe_route_recommender_from_intent(
            staged=staged,
            message="what should I improve",
            store=store,
        )

    # Dispatch was called.
    assert len(dispatch_calls) == 1
    assert dispatch_calls[0]["mode"] == "local_gap_coach"
    assert result == {"reply": "canned dispatch reply", "session_id": "fake"}
    # State untouched (chain continuation, not pivot).
    assert staged.pending_recommender_offer == "local_gap_coach"
    assert len(staged.last_recommender_adjacent_surface) == 1
    assert staged.last_recommender_adjacent_surface_at_turn == 3
    # Log evidence.
    msgs = _messages(caplog)
    assert _grep(msgs, "pivot_cleared=") == []
    tele_lines = _grep(msgs, "frame_telemetry")
    assert len(tele_lines) == 1
    kv = _kv(tele_lines[0])
    assert kv["path"] == "route_recommender_layer"
    assert kv["router_action"] == "recommender_layer"
    assert kv["career_intent"] == "local_skill_gap"
    # pending_before was captured BEFORE the router ran.
    assert kv["pending_before"] == "recommender:local_gap_coach"
    assert kv["pending_after"] == "recommender:local_gap_coach"


# ---------------------------------------------------------------- scenario 3


def test_scenario_3_consent_yes_emits_consume_yes_telemetry(
    monkeypatch, caplog,
):
    """Pending target_noc_standard + user says yes → consume yes
    branch fires, evidence builds, compose runs (canned), and
    frame_telemetry emits on the consume_yes_dispatch path."""
    staged = _new_staged_with_substrate()
    staged.pending_recommender_offer = "target_noc_standard"

    # Patch the evidence builder to a non-empty stub so the yes branch
    # reaches the main dispatch return (not the empty-evidence branch).
    from skillbridge.chat.gap_evidence import (
        GapEvidence,
        RecommenderEvidence,
    )

    def _fake_target_noc_evidence(**_kw):
        return RecommenderEvidence(
            mode="target_noc_standard",
            evidence=(
                GapEvidence(
                    layer="target_noc_standard",
                    source_id="14200",
                    source_label="accounting clerk",
                    skill_id="skill-canonical",
                    skill_name="Reading Comprehension",
                    blocker=False,
                    importance=3.5,
                    source="reference.noc_skill",
                ),
            ),
            training=(),
        )

    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly."
        "build_recommender_evidence_target_noc_standard",
        _fake_target_noc_evidence,
    )
    monkeypatch.setattr(
        h, "compose_response_v2",
        lambda _inp: "canned composed reply",
    )
    store = _FakeStore()

    with caplog.at_level(logging.INFO):
        result = h._dispatch_recommender_consume(
            staged=staged,
            user_message="yes",
            store=store,
            resume_info=None,
        )

    # A response dict came back.
    assert result is not None
    assert result["reply"] == "canned composed reply"
    # Consume-yes dispatch telemetry.
    msgs = _messages(caplog)
    tele_lines = _grep(msgs, "frame_telemetry")
    assert len(tele_lines) == 1
    kv = _kv(tele_lines[0])
    assert kv["path"] == "consume_yes_dispatch"
    # Router-side fields absent (consume path doesn't run the router).
    assert kv["pattern_intent"] == "none"
    assert kv["router_action"] == "none"


# ---------------------------------------------------------------- scenario 4


def test_scenario_4_drilldown_resolver_hit(monkeypatch, caplog):
    """pending = adjacent_role_drilldown_select + surface with two
    roles + user says 'the first one' → resolver hits, dispatch
    fires, frame_telemetry emitted on drilldown_resolver_hit path."""
    staged = _new_staged()
    staged.pending_recommender_offer = "adjacent_role_drilldown_select"
    staged.last_recommender_adjacent_surface = (
        {"noc_code": "13110", "title": "Administrative assistant"},
        {"noc_code": "14200", "title": "Accounting clerk"},
    )
    staged.last_recommender_adjacent_surface_at_turn = 5

    # Resolver: pick the first surface entry.
    def _fake_resolve(_msg, surface):
        return surface[0] if surface else None

    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly.resolve_drilldown_selection",
        _fake_resolve,
    )
    # Dispatch: return canned dict (real dispatch is heavy).
    dispatch_calls: list[dict] = []

    def _fake_drilldown(**kwargs):
        dispatch_calls.append(kwargs)
        return {"reply": "canned drilldown reply", "session_id": "fake"}

    monkeypatch.setattr(h, "_dispatch_role_drilldown", _fake_drilldown)
    store = _FakeStore()

    with caplog.at_level(logging.INFO):
        result = h._dispatch_recommender_consume(
            staged=staged,
            user_message="the first one",
            store=store,
            resume_info=None,
        )

    # Dispatch was called with the surfaced NOC.
    assert len(dispatch_calls) == 1
    assert dispatch_calls[0]["noc_code"] == "13110"
    assert dispatch_calls[0]["role_title"] == "Administrative assistant"
    assert result == {"reply": "canned drilldown reply", "session_id": "fake"}
    # Telemetry emitted BEFORE dispatch call.
    msgs = _messages(caplog)
    tele_lines = _grep(msgs, "frame_telemetry")
    assert len(tele_lines) == 1
    kv = _kv(tele_lines[0])
    assert kv["path"] == "drilldown_resolver_hit"
    # Pending state was live at snapshot time (dispatch hasn't
    # mutated it).
    assert "recommender:adjacent_role_drilldown_select" in kv["pending_before"]


# ---------------------------------------------------------------- scenario 5


def test_scenario_5_consent_other_fallthrough_router_pivot_clears(
    monkeypatch, caplog,
):
    """The real-world handoff test: pending recommender + user
    message that's neither yes nor no. Consume classifies as 'other',
    returns None, and (per its documented contract) leaves the flag
    set. Then the router runs, classifier says job_matching, and
    Step 1.3's pivot-clear finally closes the stale offer. Verifies
    the two pieces cooperate across function boundaries."""
    staged = _new_staged_with_substrate()
    staged.pending_recommender_offer = "local_gap_coach"
    staged.last_recommender_adjacent_surface = (
        {"noc_code": "13110", "title": "Administrative assistant"},
    )
    staged.last_recommender_adjacent_surface_at_turn = 4

    monkeypatch.setattr(
        "skillbridge.chat.recommender_intent.classify_career_intent",
        lambda **_kw: "job_matching",
    )
    store = _FakeStore()
    message = "show me admin jobs instead"

    with caplog.at_level(logging.INFO):
        # 1) Consume runs first. Classifies as "other" (message is not
        # yes/no). Per its documented contract at handler.py:625-627,
        # returns None WITHOUT clearing the flag.
        consume_result = h._dispatch_recommender_consume(
            staged=staged,
            user_message=message,
            store=store,
            resume_info=None,
        )
        assert consume_result is None
        # Contract check: flag still set after consume=other.
        assert staged.pending_recommender_offer == "local_gap_coach"
        assert len(staged.last_recommender_adjacent_surface) == 1

        # 2) Router runs next (mirrors handle_anonymous ordering).
        router_result = h._maybe_route_recommender_from_intent(
            staged=staged,
            message=message,
            store=store,
        )

    # Router fell through (matching flow handles).
    assert router_result is None
    # Pivot-clear finally fired.
    assert staged.pending_recommender_offer is None
    assert staged.last_recommender_adjacent_surface == ()
    assert staged.last_recommender_adjacent_surface_at_turn is None
    # Log evidence.
    msgs = _messages(caplog)
    # Consume returned None on "other" -- no frame_telemetry from it.
    # Router matching_engine fallthrough -- no frame_telemetry from it either.
    assert _grep(msgs, "frame_telemetry") == []
    # But pivot-clear log fired once, from the router.
    pivot_lines = _grep(msgs, "pivot_cleared=")
    assert len(pivot_lines) == 1
    assert "pending_recommender_offer" in pivot_lines[0]
    assert "last_recommender_adjacent_surface" in pivot_lines[0]
    assert "action=matching_engine" in pivot_lines[0]
