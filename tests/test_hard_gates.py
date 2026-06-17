"""Unit tests for Sprint 5 step 4 -- hard gates.

Three gates demote a would-be 'strong'/'good' band to 'stretch':

  1. Credential cap        -- missing licence/cert (e.g. Class G).
  2. No-experience floor   -- empty experience_text.
  3. Work-type mismatch    -- full-time user vs part-time-only job (or v.v.).

These tests call `_score_one_job` directly with a hand-crafted minimal
job dict and profile dict. They avoid the DB by setting
`target_role_text=None` and `noc_code=None`, which short-circuits the
`_regulated()` lookup -- the only DB call inside the scoring path.
"""
from __future__ import annotations

import os

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from config import MATCH
from skillbridge.match import engine

pytestmark = pytest.mark.nodb


# ---------------------------------------------------------------------------
# Builders -- keep each test self-contained.
# ---------------------------------------------------------------------------
def _make_job(
    *,
    title: str = "Customer Service Representative",
    employment_type: str | None = None,
) -> dict:
    """Minimal job dict. noc_code=None keeps _regulated() out of the DB."""
    return {
        "job_id": "job-test",
        "title": title,
        "description": "",
        "employer": "Test Employer",
        "url": "https://example.test/job",
        "location": "Sault Ste. Marie, ON",
        "region_code": "3557011",
        "posted_date": None,
        "employment_type": employment_type,
        "noc_code": None,
    }


def _make_profile(
    *,
    experience_text: str | None = "5 years of relevant experience",
    work_type_preference: str | None = None,
) -> dict:
    """Minimal profile dict. target_role_text=None keeps _regulated() out
    of the DB and also disables the title-match override."""
    return {
        "profile_id": "profile-test",
        "preferred_location": "Sault Ste. Marie",
        "target_role_text": None,
        "work_type_preference": work_type_preference,
        "shift_preference": None,
        "experience_text": experience_text,
    }


def _make_skill(name: str, *, skill_type: str | None = "required") -> dict:
    """A single JD-side skill row, shaped like _fetch_job_skills returns."""
    return {
        "skill_id": None,
        "skill_name": name,
        "confidence": 0.95,
        "importance_rank": 1,
        "skill_type": skill_type,
    }


def _score(
    *,
    job_skills: list[dict],
    user_skill_names: set[str],
    job: dict | None = None,
    profile: dict | None = None,
):
    job = job or _make_job()
    profile = profile or _make_profile()
    return engine._score_one_job(
        job=job,
        job_skills=job_skills,
        user_skill_ids=set(),
        user_skill_names=user_skill_names,
        profile=profile,
    )


# ---------------------------------------------------------------------------
# 1. Credential cap -- existing behavior + the max() cleanup
# ---------------------------------------------------------------------------
def test_credential_cap_demotes_and_floors_score_to_stretch():
    """Missing 'class g license' caps a would-be strong match.

    Also verifies the Sprint 5 step 4 cleanup: the score is floored to
    exactly `MATCH.band_stretch + 0.01`, not derived from a stray
    `max(x, x)` expression.
    """
    skills = [
        _make_skill("welding"),
        _make_skill("driving"),
        _make_skill("class g license"),
    ]
    # User matches welding + driving but is missing the licence -- credential
    # cap should fire and demote.
    result = _score(job_skills=skills, user_skill_names={"welding", "driving"})
    assert result is not None
    assert result.match_band == "stretch"
    assert result.match_score == round(MATCH.band_stretch + 0.01, 3)
    assert result.score_explanation["band_capped_by_credential"] is True
    assert "class g license" in [
        s.lower() for s in result.score_explanation["credential_gap_skills"]
    ]


def test_credential_cap_does_not_fire_without_licence_miss():
    """No licence in missing -> no credential cap. Sanity check."""
    skills = [
        _make_skill("welding"),
        _make_skill("hand tools"),
        _make_skill("safety procedures"),
    ]
    result = _score(
        job_skills=skills,
        user_skill_names={"welding", "hand tools", "safety procedures"},
    )
    assert result.score_explanation.get("band_capped_by_credential") is not True


# ---------------------------------------------------------------------------
# 2. No-experience floor
# ---------------------------------------------------------------------------
def test_no_experience_floor_caps_strong_match_to_stretch():
    """All required matched but no experience_text -> floor demotes to stretch."""
    skills = [
        _make_skill("customer service"),
        _make_skill("phone communication"),
        _make_skill("payment processing"),
    ]
    result = _score(
        job_skills=skills,
        user_skill_names={"customer service", "phone communication",
                          "payment processing"},
        profile=_make_profile(experience_text=None),
    )
    assert result.match_band == "stretch"
    assert result.match_score == round(MATCH.band_stretch + 0.01, 3)
    assert result.score_explanation["band_capped_by_no_experience"] is True


def test_no_experience_floor_with_empty_string():
    """Whitespace-only experience_text is treated the same as None."""
    skills = [_make_skill("customer service"), _make_skill("phone communication"),
              _make_skill("payment processing")]
    result = _score(
        job_skills=skills,
        user_skill_names={"customer service", "phone communication",
                          "payment processing"},
        profile=_make_profile(experience_text="   "),
    )
    assert result.score_explanation["band_capped_by_no_experience"] is True


def test_no_experience_floor_silent_when_experience_present():
    skills = [_make_skill("customer service"), _make_skill("phone communication"),
              _make_skill("payment processing")]
    result = _score(
        job_skills=skills,
        user_skill_names={"customer service", "phone communication",
                          "payment processing"},
        profile=_make_profile(experience_text="3 years at retail"),
    )
    assert result.score_explanation.get("band_capped_by_no_experience") is not True
    assert result.match_band in ("good", "strong")


# ---------------------------------------------------------------------------
# 3. Work-type mismatch cap
# ---------------------------------------------------------------------------
def test_work_type_cap_full_time_user_vs_part_time_job():
    skills = [_make_skill("customer service"), _make_skill("phone communication"),
              _make_skill("payment processing")]
    result = _score(
        job_skills=skills,
        user_skill_names={"customer service", "phone communication",
                          "payment processing"},
        job=_make_job(employment_type="part-time"),
        profile=_make_profile(work_type_preference="full-time"),
    )
    assert result.match_band == "stretch"
    assert result.score_explanation["band_capped_by_work_type_mismatch"] is True
    assert result.score_explanation["work_type_user"] == "full-time"
    assert result.score_explanation["work_type_job"] == "part-time"


def test_work_type_cap_part_time_user_vs_full_time_job():
    skills = [_make_skill("customer service"), _make_skill("phone communication"),
              _make_skill("payment processing")]
    result = _score(
        job_skills=skills,
        user_skill_names={"customer service", "phone communication",
                          "payment processing"},
        job=_make_job(employment_type="full-time"),
        profile=_make_profile(work_type_preference="part-time"),
    )
    assert result.match_band == "stretch"
    assert result.score_explanation["band_capped_by_work_type_mismatch"] is True


def test_work_type_cap_silent_when_user_says_flexible():
    """User explicitly opted out of work-type preference -> no cap."""
    skills = [_make_skill("customer service"), _make_skill("phone communication"),
              _make_skill("payment processing")]
    result = _score(
        job_skills=skills,
        user_skill_names={"customer service", "phone communication",
                          "payment processing"},
        job=_make_job(employment_type="part-time"),
        profile=_make_profile(work_type_preference="flexible"),
    )
    assert result.score_explanation.get("band_capped_by_work_type_mismatch") is not True


def test_work_type_cap_silent_when_job_employment_type_unknown():
    """Many SCCC postings omit employment_type. No signal -> no cap."""
    skills = [_make_skill("customer service"), _make_skill("phone communication"),
              _make_skill("payment processing")]
    result = _score(
        job_skills=skills,
        user_skill_names={"customer service", "phone communication",
                          "payment processing"},
        job=_make_job(employment_type=None),
        profile=_make_profile(work_type_preference="full-time"),
    )
    assert result.score_explanation.get("band_capped_by_work_type_mismatch") is not True


def test_work_type_cap_silent_when_compatible_match():
    """Full-time user, full-time job -> no cap (sanity)."""
    skills = [_make_skill("customer service"), _make_skill("phone communication"),
              _make_skill("payment processing")]
    result = _score(
        job_skills=skills,
        user_skill_names={"customer service", "phone communication",
                          "payment processing"},
        job=_make_job(employment_type="full-time"),
        profile=_make_profile(work_type_preference="full-time"),
    )
    assert result.score_explanation.get("band_capped_by_work_type_mismatch") is not True


# ---------------------------------------------------------------------------
# Cap stacking: multiple caps can fire simultaneously, all flags persist,
# band ends at stretch (the strictest), score floors to band_stretch + 0.01.
# ---------------------------------------------------------------------------
def test_multiple_caps_stack_flags_band_stays_stretch():
    """User missing a licence AND no experience AND wrong work type ->
    all three flags fire; band caps to stretch only once."""
    skills = [
        _make_skill("welding"),
        _make_skill("driving"),
        _make_skill("class g license"),
    ]
    result = _score(
        job_skills=skills,
        user_skill_names={"welding", "driving"},   # missing class g license
        job=_make_job(employment_type="part-time"),
        profile=_make_profile(
            experience_text=None,
            work_type_preference="full-time",
        ),
    )
    se = result.score_explanation
    assert result.match_band == "stretch"
    assert result.match_score == round(MATCH.band_stretch + 0.01, 3)
    assert se["band_capped_by_credential"] is True
    assert se["band_capped_by_no_experience"] is True
    assert se["band_capped_by_work_type_mismatch"] is True


# ---------------------------------------------------------------------------
# Direct-title early-return path: must apply the same hard gates.
# A job with fewer than min_required_skills_for_eligibility (default 3)
# but a strong title match takes a separate scoring path. Without the
# Step 4 review fix, that path returned 'good'/'stretch' without ever
# consulting the caps -- letting users typing an exact title bypass the
# honesty gates a normal-path match would hit.
# ---------------------------------------------------------------------------
def test_direct_title_match_applies_no_experience_floor(monkeypatch):
    """User types exact title, zero experience -> floor fires even on
    the early-return path. monkeypatch _regulated to keep this nodb."""
    monkeypatch.setattr(engine, "_regulated", lambda *a, **k: None)

    # Two skills -> below the default min_required_skills_for_eligibility
    # (3) -> direct-title early-return path is taken.
    skills = [_make_skill("communication"), _make_skill("customer focus")]
    job = _make_job(title="Front Desk Agent")
    # target_role matches title closely so _direct_title_match_score returns
    # a non-None score and the early-return path fires.
    profile = _make_profile(experience_text=None)
    profile["target_role_text"] = "Front Desk Agent"

    result = _score(
        job_skills=skills,
        user_skill_names={"communication", "customer focus"},
        job=job,
        profile=profile,
    )
    assert result is not None
    assert result.score_explanation["band_capped_by_no_experience"] is True
    # Without the cap, sim>=92 would have produced 'good' (score 0.61).
    assert result.match_band == "stretch"


def test_direct_title_match_applies_credential_cap(monkeypatch):
    """User types exact title, missing required licence -> credential cap
    fires on the early-return path too."""
    monkeypatch.setattr(engine, "_regulated", lambda *a, **k: None)
    # Two skills, one of which is a licence the user doesn't have.
    skills = [_make_skill("driving"), _make_skill("class g license")]
    job = _make_job(title="Truck Driver")
    profile = _make_profile(experience_text="3 years as a delivery driver")
    profile["target_role_text"] = "Truck Driver"

    result = _score(
        job_skills=skills,
        user_skill_names={"driving"},   # missing class g license
        job=job,
        profile=profile,
    )
    assert result.score_explanation["band_capped_by_credential"] is True
    assert result.match_band == "stretch"


def test_direct_title_match_applies_work_type_cap(monkeypatch):
    """User types exact title but explicitly wants full-time and job is
    part-time only -> work-type cap fires on the early-return path."""
    monkeypatch.setattr(engine, "_regulated", lambda *a, **k: None)
    skills = [_make_skill("communication"), _make_skill("teamwork")]
    job = _make_job(title="Front Desk Agent", employment_type="part-time")
    profile = _make_profile(
        experience_text="3 years at retail",
        work_type_preference="full-time",
    )
    profile["target_role_text"] = "Front Desk Agent"

    result = _score(
        job_skills=skills,
        user_skill_names={"communication", "teamwork"},
        job=job,
        profile=profile,
    )
    se = result.score_explanation
    assert se["band_capped_by_work_type_mismatch"] is True
    assert se["work_type_user"] == "full-time"
    assert se["work_type_job"] == "part-time"
    assert result.match_band == "stretch"


def test_direct_title_match_no_caps_when_all_clear(monkeypatch):
    """Sanity: when no cap conditions hold, the early-return path returns
    the title-match band unchanged."""
    monkeypatch.setattr(engine, "_regulated", lambda *a, **k: None)
    skills = [_make_skill("communication"), _make_skill("teamwork")]
    job = _make_job(title="Front Desk Agent", employment_type="full-time")
    profile = _make_profile(
        experience_text="3 years at a hotel front desk",
        work_type_preference="full-time",
    )
    profile["target_role_text"] = "Front Desk Agent"

    result = _score(
        job_skills=skills,
        user_skill_names={"communication", "teamwork"},
        job=job,
        profile=profile,
    )
    se = result.score_explanation
    assert se.get("band_capped_by_credential") is not True
    assert se.get("band_capped_by_no_experience") is not True
    assert se.get("band_capped_by_work_type_mismatch") is not True
    # sim>=92 -> good band, no cap to demote it.
    assert result.match_band == "good"
