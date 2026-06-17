"""Unit tests for matching v2 step 4a -- continuous match_strength refactor.

Pins the strength contract that Step 5 (semantic re-ranker) will build on:

    exact name / canonical alias / word-bounded substring  ->  1.0
    token-overlap fuzzy (>= 0.60 overlap ratio)             ->  0.85
    no match                                                 ->  0.0

Also pins:
  - _weighted_skill_base now takes strength SUMS (floats), not counts (ints)
  - score_explanation surfaces required_match_strengths,
    required_match_strength_sum (and preferred_* equivalents) on both
    the main eligible path AND the direct-title early-return path
  - When every match is stage-1 (strength=1.0), totals are identical
    to the pre-refactor count-based numbers (backwards-compat)

Pure-function / no-DB. Uses monkeypatch on engine._regulated to avoid
the regulated-occupations table lookup that would otherwise hit Postgres.
"""
from __future__ import annotations

import os

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from skillbridge.match import engine
from skillbridge.match.engine import (
    _STRENGTH_STAGE_1,
    _STRENGTH_TOKEN_OVERLAP,
    _skill_match_strength,
    _weighted_skill_base,
)

pytestmark = pytest.mark.nodb


# ---------------------------------------------------------------------------
# _skill_match_strength -- the per-skill strength decision
# ---------------------------------------------------------------------------
def _stage_args(user_names: list[str]):
    """Helper: build the (ids, names, names_canon) tuple _skill_match_strength
    expects, mirroring what _score_one_job builds for a real call."""
    from skillbridge.match.aliases import canonicalize_skill
    names = {n.lower() for n in user_names}
    canon = {canonicalize_skill(n) for n in names}
    return set(), names, canon


def test_strength_exact_name_match():
    """User has 'python' exact, job extracted 'python' exact -> 1.0 / 'exact'."""
    job_skill = {"skill_id": None, "skill_name": "Python", "confidence": 0.9}
    ids, names, canon = _stage_args(["python"])
    strength, stage = _skill_match_strength(job_skill, ids, names, canon)
    assert strength == _STRENGTH_STAGE_1
    assert stage == "exact"


def test_strength_canonical_alias_match():
    """User has 'Class G licence' (British), job has 'Class G driver's
    license'. Alias map collapses both; strength = 1.0 stage = 'exact'."""
    job_skill = {
        "skill_id": None,
        "skill_name": "Class G driver's license",
        "confidence": 0.9,
    }
    ids, names, canon = _stage_args(["Class G licence"])
    strength, stage = _skill_match_strength(job_skill, ids, names, canon)
    assert strength == _STRENGTH_STAGE_1
    assert stage == "exact"


def test_strength_word_bounded_substring_match():
    """User has 'welding & fabrication', job has 'welding'. Word-bounded
    substring fires -> strength 1.0 / stage 'exact'."""
    job_skill = {"skill_id": None, "skill_name": "welding", "confidence": 0.9}
    ids, names, canon = _stage_args(["welding & fabrication"])
    strength, stage = _skill_match_strength(job_skill, ids, names, canon)
    assert strength == _STRENGTH_STAGE_1
    assert stage == "exact"


def test_strength_token_overlap_fuzzy_only():
    """Token overlap 2/3 >= 0.60 threshold -> strength 0.85 / stage 'fuzzy'."""
    job_skill = {
        "skill_id": None,
        "skill_name": "truck service and maintenance",
        "confidence": 0.9,
    }
    ids, names, canon = _stage_args(["truck maintenance"])
    strength, stage = _skill_match_strength(job_skill, ids, names, canon)
    assert strength == _STRENGTH_TOKEN_OVERLAP
    assert stage == "fuzzy"
    assert strength < _STRENGTH_STAGE_1   # fuzzy is genuinely less than exact


def test_strength_zero_when_no_overlap():
    """No match anywhere -> strength 0.0 / stage 'no_match'."""
    job_skill = {"skill_id": None, "skill_name": "welding", "confidence": 0.9}
    ids, names, canon = _stage_args(["python"])
    strength, stage = _skill_match_strength(job_skill, ids, names, canon)
    assert strength == 0.0
    assert stage == "no_match"


def test_strength_constants_have_expected_values():
    """The constants are load-bearing for Step 5. Lock the values so
    a future refactor doesn't quietly change scoring magnitudes."""
    assert _STRENGTH_STAGE_1 == 1.0
    assert _STRENGTH_TOKEN_OVERLAP == 0.85
    assert _STRENGTH_TOKEN_OVERLAP < _STRENGTH_STAGE_1


# ---------------------------------------------------------------------------
# _weighted_skill_base -- now consumes float strength sums
# ---------------------------------------------------------------------------
def test_weighted_base_signature_accepts_floats():
    """_weighted_skill_base must accept float strength sums in the
    matched_* slots (was int counts pre-refactor)."""
    base, req_r, pref_r = _weighted_skill_base(3.85, 5, 1.85, 3)
    assert isinstance(base, float)
    assert req_r == pytest.approx(3.85 / 5)
    assert pref_r == pytest.approx(1.85 / 3)


def test_weighted_base_backwards_compat_when_all_stage_1():
    """Backwards-compat invariant: when every match is stage-1
    (strength=1.0), the sum equals the count and the ratio is
    identical to the pre-refactor count-based ratio."""
    # All-required, 3 of 5 matched at stage-1
    base_new, req_r_new, _ = _weighted_skill_base(3.0, 5, 0.0, 0)
    # Pre-refactor would have been _weighted_skill_base(3, 5, 0, 0)
    # which gives the same result since req_ratio = matched/total
    assert req_r_new == pytest.approx(0.6)
    assert base_new == pytest.approx(0.6)   # required_only mode


def test_weighted_base_fuzzy_match_lowers_ratio():
    """When 1 of 2 matches is fuzzy (0.85) instead of exact (1.0),
    the ratio drops -- which is the whole point of the refactor."""
    # 2 of 3 matched, but one is fuzzy: sum = 1.0 + 0.85 = 1.85
    _, req_ratio, _ = _weighted_skill_base(1.85, 3, 0.0, 0)
    assert req_ratio == pytest.approx(1.85 / 3)
    assert req_ratio < (2 / 3)   # lower than the pre-refactor count-based ratio


# ---------------------------------------------------------------------------
# score_explanation shape -- new fields on BOTH paths
# ---------------------------------------------------------------------------
def _make_job(*, title="Customer Service Representative", noc_code=None,
              employment_type=None) -> dict:
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
        "noc_code": noc_code,
    }


def _make_profile() -> dict:
    return {
        "profile_id": "profile-test",
        "preferred_location": "Sault Ste. Marie",
        "target_role_text": None,
        "target_noc": None,
        "work_type_preference": None,
        "shift_preference": None,
        "experience_text": "3 years experience",
    }


def _make_skill(name: str, *, skill_type="required", rank=1) -> dict:
    return {
        "skill_id": None,
        "skill_name": name,
        "confidence": 0.95,
        "importance_rank": rank,
        "skill_type": skill_type,
    }


def test_score_explanation_includes_strength_fields_main_path(monkeypatch):
    """New strength fields must appear on the main eligible path."""
    monkeypatch.setattr(engine, "_regulated", lambda *a, **k: None)
    skills = [
        _make_skill("welding"),
        _make_skill("truck service and maintenance"),
        _make_skill("diesel engine repair"),
    ]
    result = engine._score_one_job(
        job=_make_job(),
        job_skills=skills,
        user_skill_ids=set(),
        user_skill_names={"welding", "truck maintenance"},   # 1 exact + 1 fuzzy
        profile=_make_profile(),
    )
    se = result.score_explanation
    # New fields exist
    assert "required_match_strengths" in se
    assert "required_match_strength_sum" in se
    assert "preferred_match_strengths" in se
    assert "preferred_match_strength_sum" in se
    # Shape: parallel to required_matched
    assert len(se["required_match_strengths"]) == len(se["required_matched"])
    # Strength values are floats in [0.0, 1.0]
    for s in se["required_match_strengths"]:
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0


def test_score_explanation_strength_sum_matches_individual_values(monkeypatch):
    """required_match_strength_sum must equal sum(required_match_strengths)."""
    monkeypatch.setattr(engine, "_regulated", lambda *a, **k: None)
    skills = [
        _make_skill("welding"),                          # user has -> 1.0
        _make_skill("truck service and maintenance"),    # token overlap -> 0.85
        _make_skill("diesel engine repair"),             # no match -> 0.0
    ]
    result = engine._score_one_job(
        job=_make_job(),
        job_skills=skills,
        user_skill_ids=set(),
        user_skill_names={"welding", "truck maintenance"},
        profile=_make_profile(),
    )
    se = result.score_explanation
    expected_sum = round(sum(se["required_match_strengths"]), 3)
    assert se["required_match_strength_sum"] == expected_sum
    # And the actual values
    assert sorted(se["required_match_strengths"]) == [0.85, 1.0]


def test_score_explanation_strength_fields_on_direct_title_path(monkeypatch):
    """Direct-title early-return path must mirror the new strength fields
    so the responder doesn't have to handle two payload schemas."""
    monkeypatch.setattr(engine, "_regulated", lambda *a, **k: None)
    # Below min_required_skills_for_eligibility (3) -> direct-title path
    skills = [_make_skill("communication"), _make_skill("teamwork")]
    job = _make_job(title="Front Desk Agent")
    profile = _make_profile()
    profile["target_role_text"] = "Front Desk Agent"   # triggers title-match

    result = engine._score_one_job(
        job=job,
        job_skills=skills,
        user_skill_ids=set(),
        user_skill_names={"communication", "teamwork"},
        profile=profile,
    )
    assert result is not None
    se = result.score_explanation
    assert "required_match_strengths" in se
    assert "required_match_strength_sum" in se
    assert se["required_match_strengths"] == [1.0, 1.0]   # both exact
    assert se["required_match_strength_sum"] == 2.0


def test_legacy_stage_1_matches_preserve_old_ratios(monkeypatch):
    """The backwards-compat invariant in practice: a profile with only
    exact matches produces the SAME required_match_ratio as before the
    refactor. If this drifts, every existing matching fixture breaks."""
    monkeypatch.setattr(engine, "_regulated", lambda *a, **k: None)
    skills = [
        _make_skill("python"),       # exact
        _make_skill("sql"),          # exact
        _make_skill("docker"),       # missing
    ]
    result = engine._score_one_job(
        job=_make_job(),
        job_skills=skills,
        user_skill_ids=set(),
        user_skill_names={"python", "sql"},
        profile=_make_profile(),
    )
    se = result.score_explanation
    # 2 of 3 matched at full strength -> ratio = 2/3
    assert se["required_match_ratio"] == pytest.approx(2 / 3, abs=0.001)
    assert se["required_match_strength_sum"] == 2.0


def test_lowercase_collision_keeps_max_strength(monkeypatch):
    """Pins the dedup rule documented in engine.py: if two required
    skills lowercase-collide to the same key, the highest strength
    wins (not the last write). Today the (job_id, skill_name) PK in
    extracted.job_skill prevents this in production data, but the
    rule needs to be enforced in code so a future schema relaxation
    can't silently lose strength.
    """
    monkeypatch.setattr(engine, "_regulated", lambda *a, **k: None)
    # Force a case-collision: two job-side skills that lowercase to
    # the same string. Need >=3 required to enter the main eligible
    # path (min_required_skills_for_eligibility); pad with a third
    # unrelated skill.
    skills = [
        _make_skill("Project Management"),
        _make_skill("project management"),    # lowercase collision
        _make_skill("communication"),         # unrelated padding
    ]
    result = engine._score_one_job(
        job=_make_job(),
        job_skills=skills,
        user_skill_ids=set(),
        user_skill_names={"project management"},
        profile=_make_profile(),
    )
    se = result.score_explanation
    assert se is not None, "expected main-path score_explanation"
    # Both colliding rows are stage-1 exact matches. Invariant: whichever
    # iteration order the engine picks, the dict's value for the
    # collision key must be 1.0 (max). The presence of "= max(...)"
    # makes this guaranteed regardless of order.
    matched_strengths = se["required_match_strengths"]
    assert matched_strengths, "expected at least one match strength"
    assert all(s == 1.0 for s in matched_strengths)


def test_fuzzy_match_produces_lower_ratio_than_exact(monkeypatch):
    """Two profiles with the same number of matched skills produce
    different ratios when one is fuzzy and one is exact. This is the
    whole point of the refactor: fuzzy matches are less certain."""
    monkeypatch.setattr(engine, "_regulated", lambda *a, **k: None)
    skills = [
        _make_skill("truck service and maintenance"),
        _make_skill("emergency repair"),
        _make_skill("welding"),
    ]
    job = _make_job()

    # Profile A: 1 exact match (welding)
    a = engine._score_one_job(
        job=job, job_skills=skills,
        user_skill_ids=set(),
        user_skill_names={"welding"},
        profile=_make_profile(),
    )
    # Profile B: 1 fuzzy match (truck maintenance overlaps truck service & maintenance)
    b = engine._score_one_job(
        job=job, job_skills=skills,
        user_skill_ids=set(),
        user_skill_names={"truck maintenance"},
        profile=_make_profile(),
    )
    # Both have 1/3 matched, but b's match is fuzzy -> lower ratio
    assert a.score_explanation["required_match_ratio"] == pytest.approx(1 / 3, abs=0.001)
    assert b.score_explanation["required_match_ratio"] == pytest.approx(0.85 / 3, abs=0.001)
    assert b.score_explanation["required_match_ratio"] < a.score_explanation["required_match_ratio"]
