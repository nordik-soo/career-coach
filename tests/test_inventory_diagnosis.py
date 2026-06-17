"""CP4-pre-1 inventory-diagnosis gate — isolated tests against the
LOCKED contract: six outcomes, six rules, deterministic.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from skillbridge.chat.inventory_diagnosis import (
    InventoryDiagnosis,
    Outcome,
    diagnose,
)

pytestmark = pytest.mark.nodb


def _match(
    *,
    job_id: str = "j1",
    match_band: str = "stretch",
    match_eligible: bool = True,
    required_missing: list[str] | None = None,
    noc_code: str | None = "14200",
):
    """Minimal MatchResult-shaped duck for diagnose() inputs."""
    return SimpleNamespace(
        job_id=job_id,
        match_band=match_band,
        match_eligible=match_eligible,
        noc_code=noc_code,
        score_explanation={"required_missing": list(required_missing or [])},
    )


# --- Rule 0: engine completion precondition -----------------------------

def test_engine_failed_returns_market_data_unavailable():
    out = diagnose(
        enough_to_match=True, usable_evidence_present=True,
        engine_completed=False, snapshot_usable=True,
        direct_match_results=[], skill_adjacent_results=[],
        target_posting_count=10,
    )
    assert out.outcome == "MARKET_DATA_UNAVAILABLE"
    assert out.reason_code == "market_data_unavailable_engine_failed"


# --- Rule 1: insufficient user evidence ---------------------------------

def test_enough_to_match_false_returns_undetermined():
    out = diagnose(
        enough_to_match=False, usable_evidence_present=True,
        engine_completed=True, snapshot_usable=True,
        direct_match_results=[], skill_adjacent_results=[],
        target_posting_count=None,
    )
    assert out.outcome == "UNDETERMINED"
    assert out.reason_code == "insufficient_user_evidence"


def test_usable_evidence_false_returns_undetermined():
    out = diagnose(
        enough_to_match=True, usable_evidence_present=False,
        engine_completed=True, snapshot_usable=True,
        direct_match_results=[], skill_adjacent_results=[],
        target_posting_count=None,
    )
    assert out.outcome == "UNDETERMINED"


# --- Rule 2: market data unusable ---------------------------------------

def test_snapshot_unusable_returns_market_data_unavailable():
    out = diagnose(
        enough_to_match=True, usable_evidence_present=True,
        engine_completed=True, snapshot_usable=False,
        direct_match_results=[_match(match_band="strong", required_missing=[])],
        skill_adjacent_results=[],
        target_posting_count=10,
    )
    assert out.outcome == "MARKET_DATA_UNAVAILABLE"
    assert out.reason_code == "market_data_unavailable_snapshot"


# --- Rule 3: Apply-today match ------------------------------------------

def test_strong_match_no_missing_returns_ready_to_apply():
    out = diagnose(
        enough_to_match=True, usable_evidence_present=True,
        engine_completed=True, snapshot_usable=True,
        direct_match_results=[
            _match(match_band="strong", required_missing=[]),
        ],
        skill_adjacent_results=[],
        target_posting_count=5,
    )
    assert out.outcome == "READY_TO_APPLY"
    assert "ready_job_ids" in out.supporting_evidence


def test_good_match_no_missing_returns_ready_to_apply():
    out = diagnose(
        enough_to_match=True, usable_evidence_present=True,
        engine_completed=True, snapshot_usable=True,
        direct_match_results=[
            _match(match_band="good", required_missing=[]),
        ],
        skill_adjacent_results=[],
        target_posting_count=5,
    )
    assert out.outcome == "READY_TO_APPLY"


def test_strong_match_with_missing_does_not_return_ready_to_apply():
    out = diagnose(
        enough_to_match=True, usable_evidence_present=True,
        engine_completed=True, snapshot_usable=True,
        direct_match_results=[
            _match(match_band="strong", required_missing=["bookkeeping"]),
        ],
        skill_adjacent_results=[],
        target_posting_count=5,
    )
    # Falls through to PREPARATION_GAP (target exists, gap present)
    assert out.outcome == "PREPARATION_GAP"


def test_ineligible_match_does_not_return_ready_to_apply():
    out = diagnose(
        enough_to_match=True, usable_evidence_present=True,
        engine_completed=True, snapshot_usable=True,
        direct_match_results=[
            _match(match_band="strong", required_missing=[],
                   match_eligible=False),
        ],
        skill_adjacent_results=[],
        target_posting_count=5,
    )
    # Ineligible doesn't count as ready; with target_posting_count>0,
    # but no eligible record with required_missing, falls to no preparation gap
    # because there are no records with required_missing != [].
    # However target postings exist → still PREPARATION_GAP.
    assert out.outcome == "PREPARATION_GAP"


# --- Rule 4: target postings exist + gap --------------------------------

def test_target_postings_exist_returns_preparation_gap():
    out = diagnose(
        enough_to_match=True, usable_evidence_present=True,
        engine_completed=True, snapshot_usable=True,
        direct_match_results=[
            _match(match_band="stretch", required_missing=["bookkeeping"]),
        ],
        skill_adjacent_results=[],
        target_posting_count=3,
    )
    assert out.outcome == "PREPARATION_GAP"
    assert out.reason_code == "target_exists_user_underqualified"
    assert out.supporting_evidence["target_posting_count"] == 3
    assert out.supporting_evidence["gap_record_count"] == 1


def test_preparation_gap_does_not_inspect_actionability():
    """A credential gap with no mapped training still produces
    PREPARATION_GAP. Diagnosis identifies the problem; CP4 decides if
    a credible solution exists."""
    out = diagnose(
        enough_to_match=True, usable_evidence_present=True,
        engine_completed=True, snapshot_usable=True,
        direct_match_results=[
            _match(
                match_band="stretch",
                required_missing=["Class A licence"],
            ),
        ],
        skill_adjacent_results=[],
        target_posting_count=2,
    )
    assert out.outcome == "PREPARATION_GAP"


def test_market_existence_independent_of_user_fit():
    """The central correction: a user with WEAK matches against a
    populous target NOC must NOT be diagnosed as inventory absence.
    target_posting_count comes from the independent market query, not
    from user-evaluated results."""
    # User has no eligible matches at all, but the market has 12
    # postings in the target NOC.
    out = diagnose(
        enough_to_match=True, usable_evidence_present=True,
        engine_completed=True, snapshot_usable=True,
        direct_match_results=[
            _match(
                match_band="low",
                match_eligible=True,
                required_missing=["accounts payable"],
            ),
        ],
        skill_adjacent_results=[],
        target_posting_count=12,
    )
    assert out.outcome == "PREPARATION_GAP"
    # NEVER NO_OPPORTUNITY_FOUND when target_posting_count > 0.


# --- Rule 5: skill-adjacent jobs ----------------------------------------

def test_skill_adjacent_returns_sideways_when_no_target_match():
    out = diagnose(
        enough_to_match=True, usable_evidence_present=True,
        engine_completed=True, snapshot_usable=True,
        direct_match_results=[],
        skill_adjacent_results=["adj1", "adj2"],
        target_posting_count=0,
    )
    assert out.outcome == "SKILL_ADJACENT_AVAILABLE"
    assert out.supporting_evidence["skill_adjacent_count"] == 2


def test_skill_adjacent_skipped_when_target_postings_exist():
    """Target postings existing (rule 4) takes precedence over skill
    adjacency (rule 5)."""
    out = diagnose(
        enough_to_match=True, usable_evidence_present=True,
        engine_completed=True, snapshot_usable=True,
        direct_match_results=[
            _match(match_band="stretch", required_missing=["x"]),
        ],
        skill_adjacent_results=["adj1"],
        target_posting_count=1,
    )
    assert out.outcome == "PREPARATION_GAP"


# --- Rule 6: nothing found ---------------------------------------------

def test_nothing_found_returns_no_opportunity_found():
    out = diagnose(
        enough_to_match=True, usable_evidence_present=True,
        engine_completed=True, snapshot_usable=True,
        direct_match_results=[],
        skill_adjacent_results=[],
        target_posting_count=0,
    )
    assert out.outcome == "NO_OPPORTUNITY_FOUND"
    assert out.reason_code == "direct_and_skill_empty"


def test_nothing_found_with_unresolved_target():
    """When target_posting_count is None (vague/unresolved target),
    rule 4 is unreachable. Falls through to rule 5/6."""
    out = diagnose(
        enough_to_match=True, usable_evidence_present=True,
        engine_completed=True, snapshot_usable=True,
        direct_match_results=[],
        skill_adjacent_results=[],
        target_posting_count=None,
    )
    assert out.outcome == "NO_OPPORTUNITY_FOUND"


# --- Pillar authorization integrity -------------------------------------

def test_undetermined_authorizes_intake_clarification_only():
    out = diagnose(
        enough_to_match=False, usable_evidence_present=False,
        engine_completed=True, snapshot_usable=True,
        direct_match_results=[], skill_adjacent_results=[],
        target_posting_count=None,
    )
    assert out.pillars_authorized == frozenset({"intake_clarification"})


def test_preparation_gap_authorizes_cp4_primary_only():
    out = diagnose(
        enough_to_match=True, usable_evidence_present=True,
        engine_completed=True, snapshot_usable=True,
        direct_match_results=[
            _match(match_band="stretch", required_missing=["x"]),
        ],
        skill_adjacent_results=[],
        target_posting_count=3,
    )
    assert out.pillars_authorized == frozenset({"CP4_primary"})


def test_ready_to_apply_authorizes_cp4_secondary_admissible():
    out = diagnose(
        enough_to_match=True, usable_evidence_present=True,
        engine_completed=True, snapshot_usable=True,
        direct_match_results=[_match(match_band="strong", required_missing=[])],
        skill_adjacent_results=[],
        target_posting_count=1,
    )
    assert out.pillars_authorized == frozenset({"CP4_secondary_admissible"})


def test_skill_adjacent_authorizes_adjacency_surface_only():
    out = diagnose(
        enough_to_match=True, usable_evidence_present=True,
        engine_completed=True, snapshot_usable=True,
        direct_match_results=[], skill_adjacent_results=["x"],
        target_posting_count=0,
    )
    assert out.pillars_authorized == frozenset({
        "adjacency_recommendation_surface",
    })


def test_no_opportunity_authorizes_inventory_coaching():
    out = diagnose(
        enough_to_match=True, usable_evidence_present=True,
        engine_completed=True, snapshot_usable=True,
        direct_match_results=[], skill_adjacent_results=[],
        target_posting_count=0,
    )
    assert out.pillars_authorized == frozenset({"inventory_coaching"})


# --- Determinism + totality ---------------------------------------------

def test_function_is_total_and_deterministic():
    inputs = dict(
        enough_to_match=True, usable_evidence_present=True,
        engine_completed=True, snapshot_usable=True,
        direct_match_results=[
            _match(match_band="stretch", required_missing=["x"]),
        ],
        skill_adjacent_results=[],
        target_posting_count=4,
    )
    a = diagnose(**inputs)
    b = diagnose(**inputs)
    assert a == b
    assert a.outcome == "PREPARATION_GAP"
