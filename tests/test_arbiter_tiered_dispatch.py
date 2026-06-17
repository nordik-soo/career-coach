"""AR-9.feat.coach-tiers CP2 step 2 — arbiter tiered-matches dispatch.

Pins:
  - `resolve_match_outcome` does NOT infer tier availability from
    match_count, caps, or near-miss signals — the handler supplies
    `tiered_evidence_available` explicitly;
  - When `tiered_evidence_available=True` AND match_count > 0,
    the arbiter emits `present_tiered_matches` (preserving
    planner_tone and caps_applied);
  - When `tiered_evidence_available=False` (default), behaviour is
    byte-stable with the pre-CP2 arbiter;
  - `tiered_evidence_available=True` combined with match_count == 0
    falls through to the existing present_no_match / present_near_miss
    paths — the flag has no effect there (defense in depth against
    handler-side contract violations);
  - The new outcome maps to legacy ACTION_PRESENT_MATCHES;
  - Distinct reason codes (`tiered_matches_found` and
    `tiered_matches_found_with_caps`) for transcript-test
    disambiguation from the legacy present_matches surface;
  - Distinct arbiter_action (`resolved_to_tiered_matches`).
"""
from __future__ import annotations

import pytest

from skillbridge.chat.arbiter import (
    ARBITER_REASON_MATCHES_FOUND,
    ARBITER_REASON_MATCHES_WITH_CAPS,
    ARBITER_REASON_NEAR_MISS,
    ARBITER_REASON_NO_MATCHES,
    ARBITER_REASON_TIERED_MATCHES_FOUND,
    ARBITER_REASON_TIERED_MATCHES_WITH_CAPS,
    resolve_match_outcome,
)
from skillbridge.chat import intake_state

pytestmark = pytest.mark.nodb


# =========================================================================
# Default behaviour (tiered_evidence_available not supplied) is
# byte-stable with the pre-CP2 arbiter.
# =========================================================================
def test_default_kwarg_preserves_legacy_present_matches():
    d = resolve_match_outcome(match_count=3)
    assert d.final_move == "present_matches"
    assert d.arbiter_action == "resolved_to_matches"
    assert d.reason_code == ARBITER_REASON_MATCHES_FOUND


def test_default_kwarg_preserves_legacy_present_matches_with_caps():
    d = resolve_match_outcome(match_count=3, caps_applied=("band_cap",))
    assert d.final_move == "present_matches"
    assert d.arbiter_action == "resolved_to_matches"
    assert d.reason_code == ARBITER_REASON_MATCHES_WITH_CAPS
    assert d.caps_applied == ("band_cap",)


def test_explicit_false_is_equivalent_to_default():
    """Explicit `tiered_evidence_available=False` produces a decision
    indistinguishable from the default-kwarg call. No behaviour drift."""
    a = resolve_match_outcome(match_count=2)
    b = resolve_match_outcome(match_count=2, tiered_evidence_available=False)
    assert a == b


# =========================================================================
# Tiered dispatch — handler signal routes positive match_count to
# the new surface.
# =========================================================================
def test_tiered_dispatch_emits_present_tiered_matches():
    d = resolve_match_outcome(
        match_count=5, tiered_evidence_available=True,
    )
    assert d.final_move == "present_tiered_matches"
    assert d.arbiter_action == "resolved_to_tiered_matches"
    assert d.reason_code == ARBITER_REASON_TIERED_MATCHES_FOUND


def test_tiered_dispatch_preserves_planner_tone():
    d = resolve_match_outcome(
        match_count=2,
        planner_tone="warm_supportive",
        tiered_evidence_available=True,
    )
    assert d.tone == "warm_supportive"


def test_tiered_dispatch_preserves_caps_applied():
    d = resolve_match_outcome(
        match_count=2,
        caps_applied=("band_cap", "near_miss_cap"),
        tiered_evidence_available=True,
    )
    assert d.final_move == "present_tiered_matches"
    assert d.reason_code == ARBITER_REASON_TIERED_MATCHES_WITH_CAPS
    assert d.caps_applied == ("band_cap", "near_miss_cap")
    assert d.arbiter_action == "resolved_to_tiered_matches"


def test_tiered_dispatch_without_caps_uses_clean_reason():
    d = resolve_match_outcome(
        match_count=2, tiered_evidence_available=True,
    )
    assert d.reason_code == ARBITER_REASON_TIERED_MATCHES_FOUND
    assert d.caps_applied == ()


# =========================================================================
# Handler signal precedence vs match_count == 0 paths (CP2 step 6.1).
# =========================================================================
def test_tiered_flag_overrides_no_match_when_no_near_miss():
    """CP2 step 6.1: when the handler's proactive adjacency build
    finds Sideways records under match_count == 0 (no Strong/Stretch),
    `tiered_evidence_available=True` must route to
    `present_tiered_matches` instead of `present_no_match`.

    The legacy invariant ("flag has no effect on no_match") was
    explicitly reversed by the user on 2026-06-14 to make the
    Sideways-only surface reachable."""
    d = resolve_match_outcome(
        match_count=0, tiered_evidence_available=True,
    )
    assert d.final_move == "present_tiered_matches"
    assert d.arbiter_action == "resolved_to_tiered_matches"
    assert d.reason_code == ARBITER_REASON_TIERED_MATCHES_FOUND


def test_tiered_flag_does_not_override_near_miss_path():
    """Near-miss precedence is preserved: a near-miss is the user's
    current focus and overrides tier-evidence dispatch even when
    `tiered_evidence_available=True`."""
    d = resolve_match_outcome(
        match_count=0,
        near_miss_candidates=("synthetic-candidate",),
        tiered_evidence_available=True,
    )
    assert d.final_move == "present_near_miss"
    assert d.arbiter_action == "resolved_to_near_miss"
    assert d.reason_code == ARBITER_REASON_NEAR_MISS


def test_no_match_unchanged_when_flag_not_set():
    """Default-kwarg path must preserve the legacy no_match outcome
    so callers that have not opted into tiered dispatch still get
    the expected fallback."""
    d = resolve_match_outcome(match_count=0)
    assert d.final_move == "present_no_match"
    assert d.arbiter_action == "resolved_to_no_match"
    assert d.reason_code == ARBITER_REASON_NO_MATCHES


# =========================================================================
# Arbiter does NOT infer tier availability.
# =========================================================================
def test_arbiter_signature_does_not_take_strong_count():
    """The arbiter must remain blind to strong/stretch/adjacent
    bucket counts. The handler is the sole authority."""
    import inspect
    sig = inspect.signature(resolve_match_outcome)
    forbidden_names = {
        "strong_count", "len_strong", "stretch_count", "adjacent_count",
        "tier_counts", "tiered_evidence",
    }
    assert forbidden_names.isdisjoint(sig.parameters.keys()), (
        f"resolve_match_outcome must not infer tier availability — "
        f"the handler supplies tiered_evidence_available explicitly. "
        f"Found forbidden param(s): "
        f"{forbidden_names & set(sig.parameters.keys())}"
    )


def test_arbiter_does_not_import_tier_evidence_types():
    """The arbiter must not import `tiered_evidence` types (TieredEvidence,
    StrongMatch, etc.). The tiered_evidence_available flag is a primitive
    bool; the layering stays clean."""
    import skillbridge.chat.arbiter as arb_mod
    forbidden_attrs = {
        "TieredEvidence", "StrongMatch", "StretchMatch", "AdjacentJob",
        "build_tiered_evidence", "tiered_evidence",
    }
    found = forbidden_attrs & set(vars(arb_mod).keys())
    assert not found, (
        f"arbiter.py must not import tier-evidence types; "
        f"found: {found}"
    )


# =========================================================================
# Legacy action mapping (signed-off pin)
# =========================================================================
def test_present_tiered_matches_maps_to_legacy_present_matches_action():
    """Signed-off pin: the new move maps to ACTION_PRESENT_MATCHES so
    the session-snapshot lifecycle and downstream analytics consumers
    see it as a present-matches continuation, not a topic-change ask."""
    from skillbridge.chat.handler import _final_move_to_legacy_action
    assert (
        _final_move_to_legacy_action("present_tiered_matches")
        == intake_state.ACTION_PRESENT_MATCHES
    )


def test_legacy_action_mapping_includes_present_tiered_matches_key():
    from skillbridge.chat.handler import _FINAL_MOVE_TO_LEGACY_ACTION
    assert "present_tiered_matches" in _FINAL_MOVE_TO_LEGACY_ACTION
    assert (
        _FINAL_MOVE_TO_LEGACY_ACTION["present_tiered_matches"]
        == intake_state.ACTION_PRESENT_MATCHES
    )


# =========================================================================
# OutcomeMove enum membership
# =========================================================================
def test_present_tiered_matches_is_a_valid_outcome_move():
    from typing import get_args
    from skillbridge.chat.arbiter import OutcomeMove
    assert "present_tiered_matches" in set(get_args(OutcomeMove))


def test_resolved_to_tiered_matches_is_a_valid_arbiter_action():
    from typing import get_args
    from skillbridge.chat.arbiter import ArbiterAction
    assert "resolved_to_tiered_matches" in set(get_args(ArbiterAction))


# =========================================================================
# Reason-code distinctness (transcript-test disambiguation)
# =========================================================================
def test_tiered_reason_codes_are_distinct_from_legacy():
    assert ARBITER_REASON_TIERED_MATCHES_FOUND != ARBITER_REASON_MATCHES_FOUND
    assert (
        ARBITER_REASON_TIERED_MATCHES_WITH_CAPS
        != ARBITER_REASON_MATCHES_WITH_CAPS
    )


def test_tiered_reason_codes_have_distinguishing_prefix():
    """Both new codes carry the `tiered_matches` prefix so log greps
    and dashboards can identify the new surface."""
    assert ARBITER_REASON_TIERED_MATCHES_FOUND.startswith("tiered_matches")
    assert ARBITER_REASON_TIERED_MATCHES_WITH_CAPS.startswith("tiered_matches")
