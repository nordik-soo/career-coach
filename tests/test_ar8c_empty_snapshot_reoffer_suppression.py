"""AR-8c tests: empty-snapshot reoffer suppression in
`_maybe_append_soft_offer`.

Live observation (2026-06-10): on a credential-capped match → soft
offer turn, the user said "other role" → the engine ran and returned
0 recommendations → the responder showed the empty-result narration
→ the next no-match turn re-attached the soft-offer line. That reads
as nagging: the user just walked the adjacency path and got nothing.

Contract (round-2 lifecycle reconciliation):
  - `_last_adjacency_was_empty(staged)` returns True iff
    `last_adjacent_snapshot.items` is the empty list AND
    `created_message_count == staged.message_count - 1` (K=1 window).
  - `_try_v2_path` captures this predicate BEFORE the AR-6c
    lifecycle clear (`present_matches` / `present_near_miss` clear
    `last_adjacent_snapshot = None`). The captured bool is passed
    to `_maybe_append_soft_offer` as `prior_empty_adjacency`.
  - `_maybe_append_soft_offer` uses the flag as the source of truth
    (does NOT re-read the snapshot, which the lifecycle would have
    cleared on `present_matches`).
  - When True, the function returns the reply unchanged.
  - Defensive: bool-as-int rejected, malformed snapshot rejected.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.nodb


from skillbridge.chat.arbiter import ArbiterDecision
from skillbridge.chat.handler import (
    _last_adjacency_was_empty,
    _maybe_append_soft_offer,
)
from skillbridge.match.adjacent import _SOFT_OFFER_LINE
from skillbridge.match.engine import MATCH
from skillbridge.session.staging import StagedProfile, StagedSkill


# =========================================================================
# Helpers
# =========================================================================
def _staged_with_evidence(*, message_count: int = 5) -> StagedProfile:
    sp = StagedProfile.new("sess-1")
    sp.message_count = message_count
    sp.skills = [
        StagedSkill(skill_name="welding", source="resume", confidence=0.8),
        StagedSkill(skill_name="blueprint reading", source="resume", confidence=0.8),
        StagedSkill(skill_name="forklift operation", source="resume", confidence=0.8),
    ]
    return sp


def _decision(move: str, **kw) -> ArbiterDecision:
    return ArbiterDecision(
        final_move=move,
        reason_code=kw.get("reason_code", "x"),
        tone=kw.get("tone", "brief_confident"),
        arbiter_action=kw.get("arbiter_action", "passed_planner_through"),
        ask_slot=kw.get("ask_slot"),
        caps_applied=kw.get("caps_applied", ()),
    )


def _credential_capped_lead_result() -> dict:
    return {
        "score_explanation": {
            "score_components": {
                "score_pre_caps": MATCH.band_good + 0.01,
            },
            "caps_applied": ["band_capped_by_credential"],
        },
    }


# =========================================================================
# _last_adjacency_was_empty unit semantics
# =========================================================================
@pytest.mark.parametrize("snapshot, message_count, expected", [
    # Empty items, K=1 -> True
    ({"created_message_count": 4, "items": []}, 5, True),
    # Empty items, K=0 (same turn) -> False (suppression only applies AFTER)
    ({"created_message_count": 5, "items": []}, 5, False),
    # Empty items, K=2 -> False (no longer fresh)
    ({"created_message_count": 3, "items": []}, 5, False),
    # Empty items, K=3 -> False
    ({"created_message_count": 2, "items": []}, 5, False),
    # Non-empty items, K=1 -> False (snapshot has results)
    ({"created_message_count": 4, "items": [{"title": "Welder"}]}, 5, False),
    # Snapshot missing -> False
    (None, 5, False),
    # Snapshot not a dict -> False
    ("not a dict", 5, False),
    ([], 5, False),
    # items not a list -> False
    ({"created_message_count": 4, "items": "not a list"}, 5, False),
    ({"created_message_count": 4, "items": None}, 5, False),
    ({"created_message_count": 4}, 5, False),  # missing items
    # created_message_count missing -> False
    ({"items": []}, 5, False),
    # created_message_count not int -> False
    ({"created_message_count": "4", "items": []}, 5, False),
    ({"created_message_count": 4.0, "items": []}, 5, False),
    ({"created_message_count": None, "items": []}, 5, False),
    # bool-as-int guard: True/False would compare as 1/0
    ({"created_message_count": True, "items": []}, 2, False),
    ({"created_message_count": False, "items": []}, 1, False),
])
def test_last_adjacency_was_empty_pins_predicate(snapshot, message_count, expected):
    sp = StagedProfile.new("sess-x")
    sp.message_count = message_count
    sp.last_adjacent_snapshot = snapshot
    assert _last_adjacency_was_empty(sp) is expected


# =========================================================================
# Helper-level: `prior_empty_adjacency` flag is the source of truth
# =========================================================================
def test_helper_uses_prior_empty_adjacency_flag_not_snapshot(monkeypatch):
    """The helper MUST consult the passed flag, not re-read the
    snapshot. Set snapshot=None (post-lifecycle-clear shape) and
    pass `prior_empty_adjacency=True` -- suppression must still
    fire. This pins that the lifecycle-clear race condition is
    closed by passing the captured flag rather than letting the
    helper peek."""
    from skillbridge.match import adjacent

    monkeypatch.setattr(adjacent, "_adjacency_enabled", lambda: True)

    sp = _staged_with_evidence(message_count=5)
    sp.last_adjacent_snapshot = None   # cleared by lifecycle
    reply = _maybe_append_soft_offer(
        reply="here are your matches.",
        staged=sp,
        final=_decision("present_matches"),
        results=[_credential_capped_lead_result()],
        pending_offer=False,
        prior_empty_adjacency=True,
    )
    assert _SOFT_OFFER_LINE not in reply
    assert sp.pending_adjacent_offer is False


def test_helper_with_flag_false_appends_offer_regardless_of_snapshot(monkeypatch):
    """Symmetric: when `prior_empty_adjacency=False` is passed (the
    caller decided the user's prior adjacency was NOT empty), the
    helper appends the offer even if the staged snapshot looks
    "empty" in a defensive sense. The caller's captured state
    wins."""
    from skillbridge.match import adjacent

    monkeypatch.setattr(adjacent, "_adjacency_enabled", lambda: True)

    sp = _staged_with_evidence(message_count=5)
    # Set a snapshot that WOULD have triggered the predicate if read.
    sp.last_adjacent_snapshot = {"created_message_count": 4, "items": []}
    reply = _maybe_append_soft_offer(
        reply="I don't see one in today's postings.",
        staged=sp,
        final=_decision("present_no_match"),
        results=[],
        pending_offer=False,
        prior_empty_adjacency=False,
    )
    assert reply.endswith(_SOFT_OFFER_LINE)
    assert sp.pending_adjacent_offer is True


def test_helper_defaults_prior_empty_adjacency_to_false(monkeypatch):
    """Backward compatibility: existing test callers that don't pass
    `prior_empty_adjacency` rely on the default. Default MUST be
    False so the offer fires as before."""
    from skillbridge.match import adjacent

    monkeypatch.setattr(adjacent, "_adjacency_enabled", lambda: True)

    sp = _staged_with_evidence(message_count=5)
    reply = _maybe_append_soft_offer(
        reply="I don't see one in today's postings.",
        staged=sp,
        final=_decision("present_no_match"),
        results=[],
        pending_offer=False,
        # prior_empty_adjacency omitted -- relies on default
    )
    assert reply.endswith(_SOFT_OFFER_LINE)


# =========================================================================
# Interaction: AR-6b pending_offer suppression still wins
# =========================================================================
def test_pending_offer_suppression_still_wins_over_ar8c_flag(monkeypatch):
    """When pending_offer is True at entry, the AR-6b reoffer
    suppression fires first. AR-8c's flag is unreached but the
    outcome (no offer) is the same."""
    from skillbridge.match import adjacent

    monkeypatch.setattr(adjacent, "_adjacency_enabled", lambda: True)

    sp = _staged_with_evidence(message_count=5)
    reply = _maybe_append_soft_offer(
        reply="I don't see one in today's postings.",
        staged=sp,
        final=_decision("present_no_match"),
        results=[],
        pending_offer=True,
        prior_empty_adjacency=True,
    )
    assert _SOFT_OFFER_LINE not in reply
    assert sp.pending_adjacent_offer is False


# =========================================================================
# Integration: _try_v2_path captures pre-lifecycle-clear state and
# threads it through to the soft-offer hook on present_matches
# =========================================================================
class _FakeStore:
    def __init__(self):
        self.held: dict[str, StagedProfile] = {}

    def new_session(self) -> str:
        return "sess-1"

    def load(self, session_id):
        return self.held.get(session_id)

    def save(self, staged) -> str:
        self.held[staged.session_id] = staged
        return staged.session_id

    def delete(self, session_id):
        self.held.pop(session_id, None)


def test_try_v2_path_captures_empty_adjacency_before_present_matches_clear(
    monkeypatch,
):
    """Integration: the reviewer-found lifecycle race.

    Setup: staged has an empty adjacency snapshot from the prior
    turn (created_message_count = message_count - 1). The current
    turn lands on credential-capped `present_matches`, which clears
    `last_adjacent_snapshot = None` mid-flow. The soft-offer hook
    runs AFTER that clear.

    If `_maybe_append_soft_offer` peeked at the snapshot, it would
    see None (cleared) and append the soft offer -- exactly the
    nagging behavior AR-8c was meant to prevent. With the captured
    `prior_empty_adjacency` flag passed through, suppression still
    fires.

    This test would have failed before the round-2 fix (the
    AR-6c clear happens at handler.py:~1461; the soft-offer call
    at handler.py:~1500)."""
    from skillbridge.chat import handler
    from skillbridge.match import adjacent

    monkeypatch.setattr(handler, "CHAT_ORCHESTRATOR", "v2")
    monkeypatch.setattr(adjacent, "_adjacency_enabled", lambda: True)
    # Disable router (uses Haiku); skip planner; we'll fabricate
    # the engine output directly.
    monkeypatch.setattr(handler, "MESSAGE_UNDERSTANDING_ENABLED", False)
    monkeypatch.setattr(handler, "plan_next_move", lambda truth_json: None)
    # Detect intent must not fire -- the user message is a plain
    # follow-up, not adjacency.
    monkeypatch.setattr(
        "skillbridge.chat.adjacent_followup.resolve_adjacent_followup",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "skillbridge.chat.adjacent_intent.detect_adjacent_intent",
        lambda **kw: None,
    )
    # No remaining-gaps dispatch.
    monkeypatch.setattr(
        handler, "_run_remaining_gaps_dispatch",
        lambda staged, msg: (None, 0),
    )

    # Stub the engine to return a credential-capped lead result.
    fake_lead = {
        "job_id": "j1", "title": "310S Technician",
        "employer": "ACME Auto", "location": "Sault Ste. Marie, ON",
        "score": 0.65, "band": "good",
        "score_explanation": {
            "score_components": {"score_pre_caps": MATCH.band_good + 0.01},
            "caps_applied": ["band_capped_by_credential"],
        },
    }
    monkeypatch.setattr(
        "skillbridge.match.engine.compute_matches_in_memory",
        lambda staged, top=20: [fake_lead],
    )
    monkeypatch.setattr(
        handler, "_build_results_block",
        lambda matches: ([fake_lead], "good"),
    )
    monkeypatch.setattr(handler, "_collect_caps_applied", lambda r: ["band_capped_by_credential"])
    monkeypatch.setattr(handler, "_attach_training", lambda r: {})
    monkeypatch.setattr(
        "skillbridge.match.engine.next_skill_to_unlock_in_memory",
        lambda staged: (None, 0),
    )
    monkeypatch.setattr(
        handler, "_compute_near_miss",
        lambda **kw: ([], None),
    )
    # Force arbiter to return present_matches (the clear path).
    monkeypatch.setattr(
        handler, "resolve_match_outcome",
        lambda **kw: ArbiterDecision(
            final_move="present_matches",
            reason_code="capped",
            tone="brief_confident",
            arbiter_action="resolved_to_present_matches",
            caps_applied=("band_capped_by_credential",),
        ),
    )
    # Arbiter pass-1: clear the engine to run.
    from skillbridge.chat.arbiter import RunEngine
    monkeypatch.setattr(
        handler, "validate_planner_intent",
        lambda decision, truth_json: RunEngine(
            planner_reason_code="ok", planner_tone="brief_confident",
        ),
    )
    # Stub the responder so we can inspect the final reply.
    monkeypatch.setattr(
        handler, "compose_response_v2",
        lambda inp: "Here are your matches.",
    )

    sp = _staged_with_evidence(message_count=5)
    sp.target_role_text = "automotive technician"
    sp.intake_state = "intake_collecting"
    # EMPTY adjacency snapshot from prior turn -- K=1.
    sp.last_adjacent_snapshot = {
        "created_message_count": 4,
        "items": [],
    }

    response = handler._try_v2_path(
        staged=sp, message="anything",
        uploaded_file=False, resume_info=None, store=_FakeStore(),
        pending_adjacent_offer=False,
    )
    assert response is not None
    assert response.get("final_move") == "present_matches"
    # Snapshot was cleared by the AR-6c lifecycle.
    assert sp.last_adjacent_snapshot is None
    # But the SOFT OFFER must NOT have been re-attached, because
    # `_try_v2_path` captured the empty-adjacency state BEFORE the
    # clear and threaded it through.
    assert _SOFT_OFFER_LINE not in response["reply"]
    # And the `pending_adjacent_offer` flag must NOT have been set.
    assert sp.pending_adjacent_offer is False
