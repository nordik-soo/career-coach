"""Unit tests for Sprint 5 step 5 -- structured score_explanation.

Verifies the shape of the new `score_components` nested dict and the
`caps_applied` flat list across all four `skill_base.mode` branches plus
the direct-title path. Step 6 (responder narration) builds against this
shape, so any drift here breaks the prompt contract.

No DB access -- uses _score_one_job with target_role_text=None (and
monkeypatched _regulated where needed) so these run under the `nodb`
marker.
"""
from __future__ import annotations

import os

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from config import MATCH
from skillbridge.match import engine
from skillbridge.match.engine import (
    _PREF_WEIGHT,
    _REQ_WEIGHT,
    _skill_base_mode,
)

pytestmark = pytest.mark.nodb


# ---------------------------------------------------------------------------
# _skill_base_mode -- pure function
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("total_req, total_pref, expected", [
    (3, 2, "weighted"),
    (5, 0, "required_only"),
    (0, 4, "preferred_only"),
    (0, 0, "empty"),
])
def test_skill_base_mode_classifies_correctly(total_req, total_pref, expected):
    assert _skill_base_mode(total_req, total_pref) == expected


# ---------------------------------------------------------------------------
# score_components shape -- main scoring path
# ---------------------------------------------------------------------------
def _make_job(**overrides) -> dict:
    base = {
        "job_id": "job-test",
        "title": "Customer Service Representative",
        "description": "",
        "employer": "Test Employer",
        "url": "https://example.test/job",
        "location": "Sault Ste. Marie, ON",
        "region_code": "3557011",
        "posted_date": None,
        "employment_type": None,
        "noc_code": None,
    }
    base.update(overrides)
    return base


def _make_profile(**overrides) -> dict:
    base = {
        "profile_id": "profile-test",
        "preferred_location": "Sault Ste. Marie",
        "target_role_text": None,
        "work_type_preference": None,
        "shift_preference": None,
        "experience_text": "3 years of relevant experience",
    }
    base.update(overrides)
    return base


def _make_skill(name: str, *, skill_type: str | None = "required") -> dict:
    return {
        "skill_id": None,
        "skill_name": name,
        "confidence": 0.95,
        "importance_rank": 1,
        "skill_type": skill_type,
    }


def _score(*, job_skills, user_skill_names, job=None, profile=None):
    return engine._score_one_job(
        job=job or _make_job(),
        job_skills=job_skills,
        user_skill_ids=set(),
        user_skill_names=user_skill_names,
        profile=profile or _make_profile(),
    )


# ---------------------------------------------------------------------------
# Shape invariants
# ---------------------------------------------------------------------------
SCORE_COMPONENTS_KEYS = {
    "skill_base", "boosts", "title_match", "score_pre_caps", "score_post_caps",
}
SKILL_BASE_KEYS = {
    "value", "mode", "required_match_ratio", "required_weight",
    "preferred_match_ratio", "preferred_weight",
}
BOOSTS_KEYS = {
    # "location" retired in Step 1A cutover 2026-07-16 — SSM-only
    # v_current_job guarantees every candidate is SSM-verified, so
    # location is no longer a differentiating fit signal.
    "recency", "target_role", "target_noc_match",
    "work_type_fit", "shift_fit",
}
TITLE_MATCH_KEYS = {"applied", "raw_similarity"}


def test_score_components_shape_main_path():
    skills = [_make_skill("welding"), _make_skill("driving"),
              _make_skill("hand tools")]
    result = _score(
        job_skills=skills,
        user_skill_names={"welding", "driving", "hand tools"},
    )
    sc = result.score_explanation["score_components"]
    assert set(sc.keys()) == SCORE_COMPONENTS_KEYS
    assert set(sc["skill_base"].keys()) == SKILL_BASE_KEYS
    assert set(sc["boosts"].keys()) == BOOSTS_KEYS
    assert set(sc["title_match"].keys()) == TITLE_MATCH_KEYS


def test_score_components_weights_match_constants():
    # Need at least min_required_skills_for_eligibility (default 3) so the
    # main scoring path runs -- below that, _score_one_job returns either
    # the direct-title result or a 'low' band MatchResult with
    # score_explanation=None.
    skills = [_make_skill("welding"), _make_skill("driving"),
              _make_skill("hand tools")]
    result = _score(job_skills=skills,
                    user_skill_names={"welding", "driving", "hand tools"})
    sb = result.score_explanation["score_components"]["skill_base"]
    assert sb["required_weight"] == _REQ_WEIGHT
    assert sb["preferred_weight"] == _PREF_WEIGHT


# ---------------------------------------------------------------------------
# mode coverage -- all four branches of _weighted_skill_base
# ---------------------------------------------------------------------------
def test_mode_required_only_when_no_preferred_skills():
    """Legacy / all-required job -> mode 'required_only'."""
    skills = [
        _make_skill("welding"),
        _make_skill("driving"),
        _make_skill("hand tools"),
    ]   # all required
    result = _score(job_skills=skills,
                    user_skill_names={"welding", "driving", "hand tools"})
    assert result.score_explanation["score_components"]["skill_base"]["mode"] == "required_only"


def test_mode_weighted_when_mixed_required_and_preferred():
    skills = [
        _make_skill("welding"),
        _make_skill("driving", skill_type="preferred"),
        _make_skill("hand tools"),
    ]
    result = _score(job_skills=skills,
                    user_skill_names={"welding", "driving", "hand tools"})
    sb = result.score_explanation["score_components"]["skill_base"]
    assert sb["mode"] == "weighted"
    # Sanity: req_ratio*0.8 + pref_ratio*0.2 = value
    expected_value = round(
        sb["required_match_ratio"] * _REQ_WEIGHT
        + sb["preferred_match_ratio"] * _PREF_WEIGHT,
        3,
    )
    assert sb["value"] == expected_value


def test_mode_preferred_only_when_no_required_skills():
    """All-preferred is unusual but possible; engine handles it."""
    skills = [
        _make_skill("welding", skill_type="preferred"),
        _make_skill("driving", skill_type="preferred"),
        _make_skill("hand tools", skill_type="preferred"),
    ]
    result = _score(job_skills=skills,
                    user_skill_names={"welding", "driving"})
    sb = result.score_explanation["score_components"]["skill_base"]
    assert sb["mode"] == "preferred_only"


# ---------------------------------------------------------------------------
# score_pre_caps vs score_post_caps under caps
# ---------------------------------------------------------------------------
def test_score_pre_post_caps_diverge_when_cap_fires():
    skills = [
        _make_skill("welding"),
        _make_skill("driving"),
        _make_skill("class g license"),
    ]
    # Missing class g license -> credential cap fires
    result = _score(
        job_skills=skills,
        user_skill_names={"welding", "driving"},
    )
    sc = result.score_explanation["score_components"]
    assert sc["score_post_caps"] == round(MATCH.band_stretch + 0.01, 3)
    assert sc["score_pre_caps"] > sc["score_post_caps"]


def test_score_pre_post_caps_equal_when_no_cap_fires():
    skills = [_make_skill("welding"), _make_skill("driving"),
              _make_skill("hand tools")]
    result = _score(
        job_skills=skills,
        user_skill_names={"welding", "driving", "hand tools"},
    )
    sc = result.score_explanation["score_components"]
    assert sc["score_pre_caps"] == sc["score_post_caps"]


# ---------------------------------------------------------------------------
# caps_applied list
# ---------------------------------------------------------------------------
def test_caps_applied_is_empty_when_no_cap_fires():
    skills = [_make_skill("welding"), _make_skill("driving"),
              _make_skill("hand tools")]
    result = _score(
        job_skills=skills,
        user_skill_names={"welding", "driving", "hand tools"},
    )
    assert result.score_explanation["caps_applied"] == []


def test_caps_applied_records_single_cap():
    skills = [_make_skill("welding"), _make_skill("driving"),
              _make_skill("class g license")]
    result = _score(
        job_skills=skills,
        user_skill_names={"welding", "driving"},   # missing class g license
    )
    assert result.score_explanation["caps_applied"] == ["band_capped_by_credential"]


def test_caps_applied_records_all_caps_in_order():
    """When credential + no-experience + work-type all fire, all three
    appear in caps_applied in declaration order."""
    skills = [_make_skill("welding"), _make_skill("driving"),
              _make_skill("class g license")]
    result = _score(
        job_skills=skills,
        user_skill_names={"welding", "driving"},
        job=_make_job(employment_type="part-time"),
        profile=_make_profile(
            experience_text=None,
            work_type_preference="full-time",
        ),
    )
    assert result.score_explanation["caps_applied"] == [
        "band_capped_by_credential",
        "band_capped_by_no_experience",
        "band_capped_by_work_type_mismatch",
    ]


# ---------------------------------------------------------------------------
# Direct-title early-return path -- same shape, mode='direct_title'
# ---------------------------------------------------------------------------
def test_direct_title_path_has_direct_title_mode(monkeypatch):
    monkeypatch.setattr(engine, "_regulated", lambda *a, **k: None)
    skills = [_make_skill("communication"), _make_skill("teamwork")]
    job = _make_job(title="Front Desk Agent")
    profile = _make_profile()
    profile["target_role_text"] = "Front Desk Agent"

    result = _score(
        job_skills=skills,
        user_skill_names={"communication", "teamwork"},
        job=job,
        profile=profile,
    )
    sc = result.score_explanation["score_components"]
    assert set(sc.keys()) == SCORE_COMPONENTS_KEYS   # same shape as main path
    assert sc["skill_base"]["mode"] == "direct_title"
    assert sc["title_match"]["applied"] is True
    assert sc["title_match"]["raw_similarity"] is not None
    # Direct-title path doesn't apply boosts -- all zero.
    assert all(v == 0.0 for v in sc["boosts"].values())


def test_direct_title_skill_base_value_collapses_for_required_only(monkeypatch):
    """Regression for the Step 5 review fix.

    On the direct-title path, when the JD has only required skills and
    the user matches all of them, skill_base.value must equal 1.0 (the
    'required_only' branch of _weighted_skill_base), NOT the weighted
    blend 0.8 * 1.0 + 0.2 * 0.0 = 0.8 that an inline-formula bug would
    produce.
    """
    monkeypatch.setattr(engine, "_regulated", lambda *a, **k: None)
    # Two required skills -> below min_required_skills_for_eligibility (3)
    # -> direct-title early-return path runs.
    skills = [
        _make_skill("communication", skill_type="required"),
        _make_skill("teamwork", skill_type="required"),
    ]
    job = _make_job(title="Front Desk Agent")
    profile = _make_profile()
    profile["target_role_text"] = "Front Desk Agent"

    result = _score(
        job_skills=skills,
        user_skill_names={"communication", "teamwork"},   # full match
        job=job,
        profile=profile,
    )
    sb = result.score_explanation["score_components"]["skill_base"]
    assert sb["mode"] == "direct_title"
    assert sb["required_match_ratio"] == 1.0
    assert sb["preferred_match_ratio"] == 0.0
    # The fix: value must come from _weighted_skill_base (which returns
    # req_ratio when total_pref == 0), not from re-applying the weights.
    assert sb["value"] == 1.0, (
        "direct-title skill_base.value should equal _weighted_skill_base "
        "output (1.0 for all-required full match), not the weighted blend"
    )


def test_direct_title_skill_base_value_collapses_for_preferred_only(monkeypatch):
    """Sister case: all-preferred JD with full match -> value = 1.0.

    All-preferred JDs are unusual, but the engine handles them and the
    same edge-case collapse rule applies.
    """
    monkeypatch.setattr(engine, "_regulated", lambda *a, **k: None)
    skills = [
        _make_skill("communication", skill_type="preferred"),
        _make_skill("teamwork", skill_type="preferred"),
    ]
    job = _make_job(title="Front Desk Agent")
    profile = _make_profile()
    profile["target_role_text"] = "Front Desk Agent"

    result = _score(
        job_skills=skills,
        user_skill_names={"communication", "teamwork"},
        job=job,
        profile=profile,
    )
    sb = result.score_explanation["score_components"]["skill_base"]
    assert sb["required_match_ratio"] == 0.0
    assert sb["preferred_match_ratio"] == 1.0
    assert sb["value"] == 1.0


def test_direct_title_path_records_caps(monkeypatch):
    """Sanity: caps_applied is wired up correctly on the early-return path."""
    monkeypatch.setattr(engine, "_regulated", lambda *a, **k: None)
    skills = [_make_skill("communication"), _make_skill("teamwork")]
    job = _make_job(title="Front Desk Agent")
    profile = _make_profile(experience_text=None)   # triggers no-exp floor
    profile["target_role_text"] = "Front Desk Agent"

    result = _score(
        job_skills=skills,
        user_skill_names={"communication", "teamwork"},
        job=job,
        profile=profile,
    )
    assert "band_capped_by_no_experience" in result.score_explanation["caps_applied"]
    sc = result.score_explanation["score_components"]
    assert sc["score_post_caps"] == round(MATCH.band_stretch + 0.01, 3)
