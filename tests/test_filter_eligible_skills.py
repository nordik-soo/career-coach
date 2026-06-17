"""Unit tests for Sprint 5 slice 4b -- credential carve-out in
`_filter_eligible_skills`.

The carve-out exists because v1.2.0 LLM extraction frequently ranks
credentials lower than core duties (rank 9 for 'Class G driver's
license' on the truck/coach posting). Without the carve-out, the
top-N cutoff drops credentials from the matching set, and the
credential cap then silently never fires.

These tests are pure-function -- no DB, no Anthropic. The `nodb` marker
keeps the conftest TRUNCATE off.
"""
from __future__ import annotations

import os

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from config import MATCH
from skillbridge.match.engine import _filter_eligible_skills

pytestmark = pytest.mark.nodb


def _s(name: str, *, rank: int = 1, conf: float = 0.95,
       skill_type: str = "required") -> dict:
    """Build a job-skill row in the shape _fetch_job_skills returns."""
    return {
        "skill_id": None,
        "skill_name": name,
        "confidence": conf,
        "importance_rank": rank,
        "skill_type": skill_type,
    }


# ---------------------------------------------------------------------------
# Baseline: top-N still works for non-credential skills.
# ---------------------------------------------------------------------------
def test_top_n_keeps_first_n_when_no_credentials():
    """With no credentials in play, behavior matches the pre-Step-4 cutoff."""
    skills = [_s(f"skill {i}", rank=i) for i in range(1, 20)]
    kept = _filter_eligible_skills(skills)
    assert len(kept) == MATCH.top_n_required_skills
    # Order preserved by rank
    assert [s["skill_name"] for s in kept] == [
        f"skill {i}" for i in range(1, MATCH.top_n_required_skills + 1)
    ]


def test_low_confidence_skills_still_dropped():
    """The confidence threshold applies first, before the carve-out."""
    skills = [
        _s("welding", rank=1, conf=0.9),
        _s("low conf skill", rank=2, conf=0.1),
        _s("class g license", rank=9, conf=0.05),   # credential but very low conf
    ]
    kept = _filter_eligible_skills(skills)
    names = [s["skill_name"] for s in kept]
    assert "welding" in names
    assert "low conf skill" not in names
    # Credentials don't get a free pass on confidence -- they have to clear
    # the same bar everyone else does.
    assert "class g license" not in names


# ---------------------------------------------------------------------------
# Carve-out: credentials survive even when ranked outside top-N.
# ---------------------------------------------------------------------------
def test_credential_at_rank_beyond_top_n_survives():
    """The exact failure mode the carve-out targets: 'Class G driver's
    license' at rank 9 on a job with 12+ other higher-ranked skills."""
    skills = [
        _s(f"core duty {i}", rank=i) for i in range(1, 13)
    ] + [
        _s("Class G driver's license", rank=9),
    ]
    # Tweak: put the credential at rank 14 so it's clearly past top-12.
    skills = [_s(f"core duty {i}", rank=i) for i in range(1, 14)]
    skills.append(_s("Class G driver's license", rank=14))

    kept = _filter_eligible_skills(skills)
    names = [s["skill_name"] for s in kept]
    assert "Class G driver's license" in names, (
        "credential at rank 14 should survive carve-out even though "
        "top-12 cutoff would normally drop it"
    )


@pytest.mark.parametrize("credential_name", [
    "Class G driver's license",
    "Class G licence",
    "WHMIS",
    "WHMIS 2015",
    "First Aid certification",
    "CPR-C",
    "310T technician certification",
    "forklift ticket",
    "Z endorsement",
    "air brake permit",
])
def test_various_credential_phrasings_all_survive(credential_name):
    """The keyword set should catch every common credential phrasing
    Haiku produces on SCCC postings."""
    # Pad with 15 unrelated skills so the credential is well past top-12.
    skills = [_s(f"core duty {i}", rank=i) for i in range(1, 16)]
    skills.append(_s(credential_name, rank=20))
    kept = _filter_eligible_skills(skills)
    assert credential_name in [s["skill_name"] for s in kept]


def test_credential_inside_top_n_not_duplicated():
    """If a credential is already in top-N by its own rank, the carve-out
    must not append a second copy."""
    skills = [
        _s("Class G driver's license", rank=1),   # already top-1
    ] + [_s(f"core duty {i}", rank=i) for i in range(2, 15)]
    kept = _filter_eligible_skills(skills)
    names = [s["skill_name"] for s in kept]
    assert names.count("Class G driver's license") == 1


def test_multiple_credentials_outside_top_n_all_survive():
    """If a JD lists several credentials below top-N, all of them are
    carved in -- credentials aren't capped to one."""
    skills = [_s(f"core duty {i}", rank=i) for i in range(1, 13)]
    skills.append(_s("Class G driver's license", rank=14))
    skills.append(_s("WHMIS 2015", rank=15))
    skills.append(_s("First Aid certification", rank=16))
    kept = _filter_eligible_skills(skills)
    names = [s["skill_name"] for s in kept]
    assert "Class G driver's license" in names
    assert "WHMIS 2015" in names
    assert "First Aid certification" in names


def test_non_credential_skill_at_rank_15_still_filtered_out():
    """Carve-out applies ONLY to credentials -- a non-credential skill
    past top-N still gets dropped (otherwise top-N has no meaning)."""
    skills = [_s(f"core duty {i}", rank=i) for i in range(1, 16)]
    kept = _filter_eligible_skills(skills)
    names = [s["skill_name"] for s in kept]
    assert "core duty 15" not in names, (
        "non-credential at rank 15 should be dropped by top-12 cutoff"
    )


def test_credential_carve_out_preserves_other_skill_metadata():
    """The carved-in credential carries its skill_type, confidence,
    and importance_rank through unchanged."""
    skills = [_s(f"core duty {i}", rank=i) for i in range(1, 13)]
    skills.append(_s("Class G driver's license", rank=14,
                     conf=0.98, skill_type="required"))
    kept = _filter_eligible_skills(skills)
    cred = next(s for s in kept if s["skill_name"] == "Class G driver's license")
    assert cred["importance_rank"] == 14
    assert cred["confidence"] == 0.98
    assert cred["skill_type"] == "required"
