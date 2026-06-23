"""Slice 5 step 4 (2026-06-19) -- recommender chain end-to-end pins.

Exercises the handler-level consume dispatch with detector + LLM
stubs. Focus is on:
  - consent classification at each chain step routes to the right
    branch (yes/no/other);
  - the chain ADVANCES correctly through local_gap_coach ->
    target_noc_standard -> adjacent_noc_standard -> None;
  - last_adjacent_nocs is cleared exactly when the chain ENDS
    (adjacent_noc_standard) or the user declines anywhere;
  - target_role_text change resets BOTH pending_recommender_offer and
    last_adjacent_nocs via the StagedProfile lifecycle (already pinned
    in test_ar1a_state_slots, repeated here for chain context).

DB / LLM are stubbed: detector helpers are monkeypatched to return
controlled GapEvidence shapes; the responder LLM is disabled so
compose_response_v2 falls through to render_recommender_fallback.
That makes the chain orchestration the focus, not LLM output
contents.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest

from skillbridge.chat.gap_evidence import (
    GapEvidence,
    RecommenderEvidence,
)
from skillbridge.chat.handler import (
    _RECOMMENDER_NEXT_MODE,
    _VALID_RECOMMENDER_MODES,
    _classify_recommender_consent,
    _dispatch_recommender_consume,
)
from skillbridge.session.staging import StagedProfile

pytestmark = pytest.mark.nodb


class _StubStore:
    """In-memory session store stub: save() returns the same session id."""
    def __init__(self):
        self.saved: dict[str, StagedProfile] = {}

    def save(self, staged: StagedProfile) -> str:
        self.saved[staged.session_id] = staged
        return staged.session_id


def _make_staged(
    *,
    target_role: str = "accounting clerk",
    target_noc: str = "14200",
    pending: str | None = None,
    last_adjacent_nocs: tuple[str, ...] = (),
) -> StagedProfile:
    sp = StagedProfile.new("sess-test")
    sp.target_role_text = target_role
    sp.target_noc = target_noc
    if pending is not None:
        sp.pending_recommender_offer = pending
    if last_adjacent_nocs:
        sp.last_adjacent_nocs = last_adjacent_nocs
    return sp


# ===========================================================================
# Module-level invariants
# ===========================================================================
def test_valid_modes_are_locked_set():
    assert _VALID_RECOMMENDER_MODES == frozenset({
        "local_gap_coach",
        "target_noc_standard",
        "adjacent_noc_standard",
    })


def test_chain_advances_through_the_three_modes_and_ends():
    assert _RECOMMENDER_NEXT_MODE["local_gap_coach"] == "target_noc_standard"
    assert _RECOMMENDER_NEXT_MODE["target_noc_standard"] == "adjacent_noc_standard"
    assert _RECOMMENDER_NEXT_MODE["adjacent_noc_standard"] is None


# ===========================================================================
# Consent classifier wrapper
# ===========================================================================
@pytest.mark.parametrize("reply, expected", [
    ("yes", "yes"),
    ("yes please", "yes"),
    ("go ahead", "yes"),
    ("yes. go ahead", "yes"),  # the live repro that broke v1 classifier
    ("no thanks", "no"),
    ("not interested", "no"),
    ("skip it", "no"),
    ("what about something else?", "other"),
    ("actually wait", "other"),
])
def test_classify_recommender_consent_passthrough(reply, expected):
    """The recommender consent helper is a thin wrap of _classify_pattern_2_reply.
    Same semantics, same vocabulary, same yes/no/other outputs."""
    assert _classify_recommender_consent(reply) == expected


# ===========================================================================
# No-consent / no-flag path
# ===========================================================================
def test_dispatch_returns_none_when_no_pending_flag():
    """Without a pending flag, the dispatcher is a no-op -- caller
    falls through to normal flow."""
    sp = _make_staged(pending=None)
    store = _StubStore()
    result = _dispatch_recommender_consume(
        staged=sp,
        user_message="yes",
        store=store,
        resume_info=None,
    )
    assert result is None


def test_dispatch_clears_forged_invalid_mode_value():
    """Defensive: a forged or stale flag value (not in
    _VALID_RECOMMENDER_MODES) clears safely and the caller falls
    through to normal flow."""
    sp = _make_staged()
    # Bypass the validator by direct dict mutation -- mimics a
    # malformed cookie load.
    sp.__dict__["pending_recommender_offer"] = "unknown_mode"
    store = _StubStore()
    result = _dispatch_recommender_consume(
        staged=sp,
        user_message="yes",
        store=store,
        resume_info=None,
    )
    assert result is None
    assert sp.pending_recommender_offer is None


# ===========================================================================
# Consent = "other" -> leave flag + fall through
# ===========================================================================
def test_dispatch_other_leaves_flag_and_returns_none():
    """On consent='other' the flag stays set so the user can come
    back to it on a subsequent yes/no. Caller routes the message
    through normal flow."""
    sp = _make_staged(pending="local_gap_coach")
    store = _StubStore()
    result = _dispatch_recommender_consume(
        staged=sp,
        user_message="actually wait, can you remind me what you found?",
        store=store,
        resume_info=None,
    )
    assert result is None
    assert sp.pending_recommender_offer == "local_gap_coach"


# ===========================================================================
# Consent = "no" -> render acknowledgment + clear + return
# ===========================================================================
def test_dispatch_no_clears_flag_and_last_adjacent_nocs():
    sp = _make_staged(
        pending="local_gap_coach",
        last_adjacent_nocs=("13110", "13100"),
    )
    store = _StubStore()
    result = _dispatch_recommender_consume(
        staged=sp,
        user_message="no thanks",
        store=store,
        resume_info=None,
    )
    assert result is not None
    assert "reply" in result
    assert sp.pending_recommender_offer is None
    assert sp.last_adjacent_nocs == ()


def test_dispatch_no_at_target_mode_clears_flag():
    """A 'no' anywhere in the chain ends it."""
    sp = _make_staged(
        pending="target_noc_standard",
        last_adjacent_nocs=("13110",),
    )
    store = _StubStore()
    result = _dispatch_recommender_consume(
        staged=sp, user_message="no thanks", store=store,
        resume_info=None,
    )
    assert result is not None
    assert sp.pending_recommender_offer is None
    assert sp.last_adjacent_nocs == ()


def test_dispatch_no_at_adjacent_mode_clears_flag():
    sp = _make_staged(
        pending="adjacent_noc_standard",
        last_adjacent_nocs=("13110",),
    )
    store = _StubStore()
    result = _dispatch_recommender_consume(
        staged=sp, user_message="no thanks", store=store,
        resume_info=None,
    )
    assert result is not None
    assert sp.pending_recommender_offer is None
    assert sp.last_adjacent_nocs == ()


# ===========================================================================
# Consent = "yes" -> chain advance
# ===========================================================================
def test_yes_at_target_mode_advances_to_adjacent(monkeypatch):
    """When the user yes-consents at target_noc_standard, the chain
    advances to adjacent_noc_standard, last_adjacent_nocs is preserved
    (the chain is not yet ended), and the responder dispatch is
    invoked."""
    sp = _make_staged(
        pending="target_noc_standard",
        last_adjacent_nocs=("13110", "13100"),
    )
    store = _StubStore()
    # Stub the Layer A detector (called via assembly helper).
    monkeypatch.setattr(
        "skillbridge.chat.gap_evidence._fetch_noc_skill_rows",
        lambda noc: [
            {"skill_id": "F.01", "skill_name": "Reading Comprehension",
             "importance": 4.5, "noc_title": "Accounting clerk"},
        ],
    )
    # Stub the LLM as disabled so the deterministic fallback renders.
    monkeypatch.setattr(
        "skillbridge.chat.responder.is_enabled", lambda: False,
    )
    result = _dispatch_recommender_consume(
        staged=sp, user_message="yes please", store=store,
        resume_info=None,
    )
    assert result is not None
    assert result["reply"]
    assert sp.pending_recommender_offer == "adjacent_noc_standard"
    # last_adjacent_nocs persists -- chain not yet ended.
    assert sp.last_adjacent_nocs == ("13110", "13100")


def test_yes_at_adjacent_mode_ends_chain_and_clears_last_adjacent(monkeypatch):
    """When the user yes-consents at adjacent_noc_standard, the chain
    ENDS HERE: pending_recommender_offer is set to None and the
    last_adjacent_nocs cache is cleared."""
    sp = _make_staged(
        pending="adjacent_noc_standard",
        last_adjacent_nocs=("13110",),
    )
    store = _StubStore()
    monkeypatch.setattr(
        "skillbridge.chat.gap_evidence._fetch_noc_skill_rows",
        lambda noc: [
            {"skill_id": "F.01", "skill_name": "Writing",
             "importance": 4.5, "noc_title": "Admin assistant"},
        ],
    )
    monkeypatch.setattr(
        "skillbridge.chat.responder.is_enabled", lambda: False,
    )
    result = _dispatch_recommender_consume(
        staged=sp, user_message="yes", store=store, resume_info=None,
    )
    assert result is not None
    assert sp.pending_recommender_offer is None  # chain ends
    assert sp.last_adjacent_nocs == ()  # cache cleared


def test_yes_at_local_gap_coach_advances_to_target(monkeypatch):
    """When the user yes-consents at local_gap_coach (initial offer),
    the chain advances to target_noc_standard. Engine + CP4 are
    stubbed."""
    sp = _make_staged(
        pending="local_gap_coach",
        last_adjacent_nocs=("13110",),
    )
    store = _StubStore()
    # Stub engine + CP4 calls to skip the real pipeline.
    from skillbridge.match.engine import MatchResult

    monkeypatch.setattr(
        "skillbridge.match.engine.compute_matches_in_memory",
        lambda staged, top=5: [],
    )
    monkeypatch.setattr(
        "skillbridge.chat.development_plan.compute_primary_gap_name",
        lambda **kwargs: None,  # CP4 returns no primary -- evidence empty
    )
    monkeypatch.setattr(
        "skillbridge.chat.responder.is_enabled", lambda: False,
    )
    result = _dispatch_recommender_consume(
        staged=sp, user_message="yes", store=store, resume_info=None,
    )
    assert result is not None
    # Chain advances even when evidence is empty (the fallback narrates
    # honestly and the chain still moves forward).
    assert sp.pending_recommender_offer == "target_noc_standard"
    # last_adjacent_nocs preserved across the chain.
    assert sp.last_adjacent_nocs == ("13110",)


# ===========================================================================
# Lifecycle: target change clears chain state
# ===========================================================================
def test_target_change_clears_pending_recommender_offer_and_adjacent_nocs():
    """The StagedProfile __setattr__ override resets BOTH chain fields
    when target_role_text changes. Repeated here for chain context."""
    sp = _make_staged(
        target_role="accounting clerk",
        pending="target_noc_standard",
        last_adjacent_nocs=("13110",),
    )
    sp.target_role_text = "truck driver"
    assert sp.pending_recommender_offer is None
    assert sp.last_adjacent_nocs == ()
