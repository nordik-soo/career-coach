"""Unit tests for Sprint 5 step 3 -- required/preferred split.

Three surfaces are exercised:

  1. `_required_or_preferred(skill)` -- bucketing logic (NULL/unknown -> required).
  2. `_weighted_skill_base(...)` -- the 0.8 / 0.2 scoring formula with edge cases.
  3. `_parse_skills(payload, accept_skill_type=True)` -- LLM-side parser defaults
     to 'required' when the LLM omits or returns an unknown skill_type value.

These are pure-function tests. They do not touch Postgres; the conftest
DB-truncate is opted out via the `nodb` marker (registered in conftest.py).
"""
from __future__ import annotations

import pytest

from skillbridge.extract.llm_based import _parse_skills
from skillbridge.match.engine import (
    _REQ_WEIGHT,
    _PREF_WEIGHT,
    _required_or_preferred,
    _weighted_skill_base,
)

pytestmark = pytest.mark.nodb


# ---------------------------------------------------------------------------
# _required_or_preferred
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("input_type, expected", [
    ("required",  "required"),
    ("preferred", "preferred"),
    # Conservative fallbacks -- 'when unsure, default to required'
    (None,        "required"),
    ("",          "required"),
    ("unknown",   "required"),
    ("optional",  "required"),  # not in our taxonomy -> required
    ("REQUIRED",  "required"),  # case-insensitive
    ("Preferred", "preferred"),
    ("  preferred  ", "preferred"),  # whitespace tolerant
])
def test_required_or_preferred_buckets_correctly(input_type, expected):
    assert _required_or_preferred({"skill_type": input_type}) == expected


def test_required_or_preferred_handles_missing_key():
    """Legacy rows from before the column existed."""
    assert _required_or_preferred({}) == "required"


# ---------------------------------------------------------------------------
# _weighted_skill_base
# ---------------------------------------------------------------------------
def test_weights_are_eighty_twenty():
    """Sanity check: the constants are what we agreed (80/20)."""
    assert _REQ_WEIGHT == 0.8
    assert _PREF_WEIGHT == 0.2
    assert _REQ_WEIGHT + _PREF_WEIGHT == 1.0


def test_weighted_base_all_required():
    """All-required jobs (including legacy NULL rows) score by req ratio alone.

    This is the v1.1 compat guarantee: when the JD has no preferreds (or
    all skills were inserted before the skill_type column existed), the
    base equals the simple match ratio that v1.1 used.
    """
    base, req_r, pref_r = _weighted_skill_base(3, 5, 0, 0)
    assert base == pytest.approx(0.6, abs=1e-6)
    assert req_r == pytest.approx(0.6)
    assert pref_r == 0.0


def test_weighted_base_all_preferred():
    """Job with only preferred skills falls back to pref_ratio alone."""
    base, req_r, pref_r = _weighted_skill_base(0, 0, 2, 4)
    assert base == pytest.approx(0.5, abs=1e-6)
    assert req_r == 0.0
    assert pref_r == pytest.approx(0.5)


def test_weighted_base_mixed_high_required_low_preferred():
    """Required dominates: 80% of 0.8 + 20% of 0.33 = ~0.71 (good band)."""
    base, req_r, pref_r = _weighted_skill_base(4, 5, 1, 3)
    expected = 0.8 * (4 / 5) + 0.2 * (1 / 3)
    assert base == pytest.approx(expected, abs=1e-6)
    assert req_r == pytest.approx(0.8)
    assert pref_r == pytest.approx(1 / 3)


def test_weighted_base_mixed_low_required_high_preferred():
    """High preferred can't rescue low required -- by design.

    A candidate matching every preferred but few required skills should
    NOT outrank one who matches the required core, because preferreds are
    a nudge, not a substitute.
    """
    base, _, _ = _weighted_skill_base(1, 5, 3, 3)   # req=20%, pref=100%
    expected = 0.8 * 0.2 + 0.2 * 1.0    # 0.16 + 0.20 = 0.36
    assert base == pytest.approx(expected, abs=1e-6)
    # And explicitly: this should land below a candidate with 4/5 required
    # and 0/3 preferred (0.8 * 0.8 + 0.2 * 0.0 = 0.64).
    competitor, _, _ = _weighted_skill_base(4, 5, 0, 3)
    assert competitor > base


def test_weighted_base_empty_job():
    """No skills extracted at all -> base 0.0, no NaN."""
    base, req_r, pref_r = _weighted_skill_base(0, 0, 0, 0)
    assert base == 0.0
    assert req_r == 0.0
    assert pref_r == 0.0


def test_weighted_base_perfect_match():
    base, req_r, pref_r = _weighted_skill_base(5, 5, 3, 3)
    assert base == pytest.approx(1.0)
    assert req_r == 1.0
    assert pref_r == 1.0


# ---------------------------------------------------------------------------
# _parse_skills(accept_skill_type=True) -- LLM-side parser
# ---------------------------------------------------------------------------
def test_parser_passes_through_required_and_preferred():
    payload = {"skills": [
        {"name": "welding", "skill_type": "required", "confidence": 0.9},
        {"name": "tig welding", "skill_type": "preferred", "confidence": 0.7},
    ]}
    skills = _parse_skills(payload, accept_skill_type=True)
    types = {s.skill_name: s.skill_type for s in skills}
    assert types == {"welding": "required", "tig welding": "preferred"}


def test_parser_defaults_to_required_when_skill_type_missing():
    """The LLM may omit skill_type on some skills -- default to required."""
    payload = {"skills": [
        {"name": "welding", "confidence": 0.9},   # no skill_type at all
        {"name": "tig welding", "skill_type": "preferred", "confidence": 0.7},
    ]}
    skills = _parse_skills(payload, accept_skill_type=True)
    types = {s.skill_name: s.skill_type for s in skills}
    assert types == {"welding": "required", "tig welding": "preferred"}


@pytest.mark.parametrize("bogus", ["", "optional", "nice_to_have", "REQUIRED   "])
def test_parser_defaults_to_required_on_garbage_skill_type(bogus):
    """Anything other than the exact string 'preferred' (case/whitespace
    insensitive) becomes 'required'. Safer for matching."""
    payload = {"skills": [{"name": "welding", "skill_type": bogus, "confidence": 0.9}]}
    skills = _parse_skills(payload, accept_skill_type=True)
    assert skills[0].skill_type == "required"


def test_parser_ignores_skill_type_when_opt_out():
    """Training and user-text extractors don't carry skill_type; they call
    _parse_skills without the flag and should NOT get a default value set."""
    payload = {"skills": [
        {"name": "welding", "skill_type": "preferred", "confidence": 0.9},
    ]}
    skills = _parse_skills(payload)   # default accept_skill_type=False
    assert skills[0].skill_type is None


def test_parser_preferred_label_is_case_insensitive():
    payload = {"skills": [
        {"name": "tig welding", "skill_type": "Preferred", "confidence": 0.9},
        {"name": "stick welding", "skill_type": "PREFERRED", "confidence": 0.9},
    ]}
    skills = _parse_skills(payload, accept_skill_type=True)
    assert all(s.skill_type == "preferred" for s in skills)
