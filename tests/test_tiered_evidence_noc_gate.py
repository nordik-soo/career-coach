"""Bug B regression tests — same-NOC-family gate on direct tiers.

The v5 coach-tiers contract originally left Apply-today and Worth-a-try
NOC-blind. When a user's target had ZERO local postings (e.g.
"truck driver" in SSM today), the engine returned top jobs by skill
match — typically the user's PRIOR target's matches (accounting jobs
when their skills bled from a previous turn). Those jobs surfaced in
the Worth-a-try tier, the responder LLM honestly contrasted them
("this is what came back but it's not what you wanted"), and the user
saw an accounting recommendation for a truck-driver query.

CP4's diagnose contract already returned NO_OPPORTUNITY_FOUND in that
exact configuration. The v5 contract was RELOCKED on 2026-06-15 with
a same-NOC-family gate so the user-facing surface aligns: when
target_noc is resolved, only jobs sharing the 4-digit NOC prefix
("minor group") reach Apply-today / Worth-a-try. Out-of-family jobs
that pass the adjacency strict-AND gate still surface in Sideways-move.

These tests pin the gate's behaviour.
"""
from __future__ import annotations

import os

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from skillbridge.chat.tiered_evidence import (
    _in_target_noc_family,
    build_tiered_evidence,
)
from skillbridge.match.engine import MatchResult

pytestmark = pytest.mark.nodb


# ============================================================================
# §1 — _in_target_noc_family unit cases
# ============================================================================
@pytest.mark.parametrize("target,result,expected", [
    # Same minor group — admit
    ("73300", "73300", True),
    ("73300", "73301", True),
    ("14200", "14200", True),
    # Different minor group — reject
    ("73300", "14200", False),
    ("73300", "75201", False),  # truck driver vs delivery driver: different families
    # 5-digit target, 4-digit result (legacy) — first 4 chars match
    ("73300", "7330", True),    # 4-char result, first-4-chars equal → admit
    # Target unresolved — admit everything (NOC-blind fallback)
    (None,    "14200", True),
    ("",      "14200", True),
    ("12",    "14200", True),
    # Result NOC malformed when target resolved — reject (fail-closed)
    ("73300", None,    False),
    ("73300", "",      False),
    ("73300", "73",    False),
    ("73300", 73300,   False),   # non-string
])
def test_in_target_noc_family(target, result, expected):
    assert _in_target_noc_family(result, target) is expected


# ============================================================================
# §2 — direct-tier filtering helpers
# ============================================================================
def _result(
    job_id: str, noc_code: str | None, *,
    band: str = "stretch",
    required_missing: list[str] | None = None,
    match_eligible: bool = True,
) -> MatchResult:
    """Minimal MatchResult for tier-admission tests."""
    return MatchResult(
        job_id=job_id,
        profile_id="test-profile",
        title=f"Job {job_id}",
        employer=f"Employer {job_id}",
        url=None,
        location="Sault Ste. Marie",
        match_score=0.5,
        match_band=band,
        match_eligible=match_eligible,
        ineligibility_reason=None,
        matched_skills=[],
        missing_skills=list(required_missing or ()),
        matched_skill_ids=[],
        missing_skill_ids=[],
        required_skills_count=len(required_missing or ()),
        credential_warning=None,
        posted_date=None,
        noc_code=noc_code,
        score_explanation={
            "required_missing": list(required_missing or ()),
            "credential_gap_skills": [],
            "required_match_stages": [],
            "preferred_match_stages": [],
            "score_components": {},
            "caps_applied": [],
        },
    )


# ============================================================================
# §3 — Worth-a-try gate
# ============================================================================
def test_worth_a_try_drops_out_of_family_job():
    """Bug B repro: target=truck driver (NOC 73300), accounting job
    (NOC 14200) scored high on skills. Pre-fix it surfaced in
    Worth-a-try. Gate must drop it."""
    accounting = _result("acc-1", "14200", required_missing=["payroll processing"])
    te = build_tiered_evidence(
        results=[accounting],
        accepted_adjacent=[],
        user_rows=[], user_skill_ids=set(),
        user_skill_names=set(), user_skill_names_canon=set(),
        target_noc="73300",
    )
    assert te.worth_a_try == ()
    assert te.apply_today == ()


def test_worth_a_try_admits_same_family_job():
    """Truck driver job (NOC 73301) with target 73300 — same 4-digit
    minor group — must surface in Worth-a-try. Uses a non-credential
    gap (`route planning experience`) so the training-required arm of
    `_is_worth_a_try` is vacuously satisfied and the test isolates
    NOC-family-gate behaviour."""
    truck = _result("truck-1", "73301", required_missing=["route planning experience"])
    te = build_tiered_evidence(
        results=[truck],
        accepted_adjacent=[],
        user_rows=[], user_skill_ids=set(),
        user_skill_names=set(), user_skill_names_canon=set(),
        target_noc="73300",
    )
    assert len(te.worth_a_try) == 1
    assert te.worth_a_try[0].job_id == "truck-1"


# ============================================================================
# §4 — Apply-today gate
# ============================================================================
def test_apply_today_drops_out_of_family_job():
    """Strong-band, no required-missing — would normally land in Apply
    today, but out-of-family must drop."""
    finance = _result("fin-1", "14200", band="strong", required_missing=[])
    te = build_tiered_evidence(
        results=[finance],
        accepted_adjacent=[],
        user_rows=[], user_skill_ids=set(),
        user_skill_names=set(), user_skill_names_canon=set(),
        target_noc="73300",
    )
    assert te.apply_today == ()


def test_apply_today_admits_same_family_job():
    truck = _result("truck-2", "73300", band="strong", required_missing=[])
    te = build_tiered_evidence(
        results=[truck],
        accepted_adjacent=[],
        user_rows=[], user_skill_ids=set(),
        user_skill_names=set(), user_skill_names_canon=set(),
        target_noc="73300",
    )
    assert len(te.apply_today) == 1
    assert te.apply_today[0].job_id == "truck-2"


# ============================================================================
# §5 — target unresolved → existing NOC-blind behaviour preserved
# ============================================================================
def test_target_noc_none_preserves_noc_blind_behavior():
    """When target_noc is None (target unresolved), the gate must be a
    no-op. v5 NOC-blind admission applies — any band-eligible job
    surfaces regardless of NOC."""
    job = _result("any-1", "14200", required_missing=["something"])
    te = build_tiered_evidence(
        results=[job],
        accepted_adjacent=[],
        user_rows=[], user_skill_ids=set(),
        user_skill_names=set(), user_skill_names_canon=set(),
        target_noc=None,
        training_by_job={"any-1": [{"provider": "Test", "title": "Course", "url": "https://example.com"}]},
    )
    # NOC-blind: job is admitted (would have been pre-fix behavior)
    assert len(te.worth_a_try) == 1


# ============================================================================
# §6 — mixed family — the Bug B repro shape
# ============================================================================
def test_mixed_family_only_in_family_surfaces():
    """Exact Bug B shape: engine returns accounting (out-of-family) AND
    one truck driver (in-family) job, both technically tier-eligible.
    Only the truck driver job surfaces."""
    # Use non-credential gaps so `_is_worth_a_try`'s training-required
    # arm is vacuously satisfied — isolates NOC-family-gate behaviour.
    in_fam = _result("truck-3", "73300", required_missing=["route planning experience"])
    out_fam_1 = _result("acc-2", "14200", required_missing=["payroll processing"])
    out_fam_2 = _result("retail-1", "65201", required_missing=["pos experience"])
    te = build_tiered_evidence(
        results=[out_fam_1, in_fam, out_fam_2],  # engine order; in_fam in middle
        accepted_adjacent=[],
        user_rows=[], user_skill_ids=set(),
        user_skill_names=set(), user_skill_names_canon=set(),
        target_noc="73300",
    )
    surfaced_ids = {m.job_id for m in te.worth_a_try}
    assert surfaced_ids == {"truck-3"}


# ============================================================================
# §7 — empty target NOC, all in dataset → nothing surfaces
# ============================================================================
def test_target_with_no_in_family_jobs_returns_empty_direct_tiers():
    """The actual Bug B scenario: target truck driver, dataset has only
    accounting and retail jobs scoring high on user's skill bleed.
    Both direct tiers must be empty — alignment with CP4's
    NO_OPPORTUNITY_FOUND."""
    accountings = [
        _result(f"acc-{i}", "14200", required_missing=["something"])
        for i in range(5)
    ]
    te = build_tiered_evidence(
        results=accountings,
        accepted_adjacent=[],
        user_rows=[], user_skill_ids=set(),
        user_skill_names=set(), user_skill_names_canon=set(),
        target_noc="73300",
        training_by_job={
            f"acc-{i}": [{"provider": "T", "title": "X", "url": "https://example.com/x"}]
            for i in range(5)
        },
    )
    assert te.apply_today == ()
    assert te.worth_a_try == ()
