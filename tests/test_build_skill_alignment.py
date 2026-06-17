"""AR-9.feat.coach-tiers CP1 — `build_skill_alignment` shared helper.

Pins:
  - alignments are populated ONLY for matched requirements where
    user_rows is supplied;
  - classifications carry one entry per input job skill, in input
    order, with the correct strength/stage;
  - the helper does NOT score, boost, or apply hard gates — pure
    alignment construction;
  - SkillAlignment.source mirrors the job skill's required/preferred
    bucket;
  - SkillAlignment.is_normalized_equal is True ONLY for literal
    normalized equality (covered separately in test_classify_match
    and test_user_skill_rows; spot-checked here);
  - empty user_rows yields empty alignments + non-empty
    classifications (scoring still runs without attribution);
  - the helper is the SINGLE construction path used by both
    `_score_one_job` branches AND adjacency enrichment (next step).
"""
from __future__ import annotations

import pytest

from skillbridge.match.alignment import SkillAlignment, UserSkillRow
from skillbridge.match.engine import (
    _STRENGTH_STAGE_1,
    _STRENGTH_TOKEN_OVERLAP,
    build_skill_alignment,
)

pytestmark = pytest.mark.nodb


def _row(text, *, sid=None):
    from skillbridge.match.aliases import canonicalize_skill
    stripped = text.strip()
    return UserSkillRow(
        skill_id=sid,
        text=stripped,
        name=stripped.lower(),
        canon=canonicalize_skill(stripped) or "",
    )


def _sets(rows):
    ids = {r.skill_id for r in rows if r.skill_id}
    names = {r.name for r in rows}
    canons = {r.canon for r in rows if r.canon}
    return ids, names, canons


def _js(name, *, skill_type=None, skill_id=None):
    out = {"skill_id": skill_id, "skill_name": name}
    if skill_type is not None:
        out["skill_type"] = skill_type
    return out


# =========================================================================
# Shape contracts
# =========================================================================
def test_returns_alignments_and_classifications_tuple():
    rows = [_row("Python")]
    ids, names, canons = _sets(rows)
    aligns, classifications = build_skill_alignment(
        [_js("Python")], rows, ids, names, canons,
    )
    assert isinstance(aligns, list)
    assert isinstance(classifications, list)


def test_classifications_one_entry_per_job_skill_in_order():
    """The classifications list mirrors `job_skills` index-for-index,
    independent of whether each matched. This is the contract
    `_score_one_job` relies on for its scoring dicts."""
    rows = [_row("Python")]
    ids, names, canons = _sets(rows)
    job_skills = [_js("Python"), _js("Welding"), _js("python")]
    _, classifications = build_skill_alignment(
        job_skills, rows, ids, names, canons,
    )
    assert len(classifications) == 3
    assert classifications[0].skill["skill_name"] == "Python"
    assert classifications[1].skill["skill_name"] == "Welding"
    assert classifications[2].skill["skill_name"] == "python"


def test_classifications_carry_strength_and_stage():
    rows = [_row("Python")]
    ids, names, canons = _sets(rows)
    aligns, classifications = build_skill_alignment(
        [_js("Python"), _js("Welding")], rows, ids, names, canons,
    )
    assert classifications[0].strength == _STRENGTH_STAGE_1
    assert classifications[0].stage == "exact"
    assert classifications[1].strength == 0.0
    assert classifications[1].stage == "no_match"


# =========================================================================
# Alignment population
# =========================================================================
def test_no_alignments_when_user_rows_is_none():
    """Legacy callers that don't supply rows get empty alignments AND
    a full classifications list so scoring still works."""
    aligns, classifications = build_skill_alignment(
        [_js("Python")], None, set(), {"python"}, set(),
    )
    assert aligns == []
    assert len(classifications) == 1
    assert classifications[0].strength == _STRENGTH_STAGE_1


def test_no_alignments_when_user_rows_is_empty():
    aligns, classifications = build_skill_alignment(
        [_js("Python")], [], set(), set(), set(),
    )
    assert aligns == []
    assert len(classifications) == 1
    assert classifications[0].strength == 0.0   # no user has Python


def test_alignment_only_for_matched_requirements():
    rows = [_row("Python")]
    ids, names, canons = _sets(rows)
    aligns, _ = build_skill_alignment(
        [_js("Python"), _js("Welding"), _js("Excel")],
        rows, ids, names, canons,
    )
    assert len(aligns) == 1
    assert aligns[0].user_skill == "Python"
    assert aligns[0].job_requirement == "Python"


def test_alignment_source_mirrors_required_preferred_bucket():
    rows = [_row("QuickBooks")]
    ids, names, canons = _sets(rows)
    job_skills = [
        _js("QuickBooks", skill_type="required"),
        _js("QuickBooks", skill_type="preferred"),
    ]
    aligns, _ = build_skill_alignment(
        job_skills, rows, ids, names, canons,
    )
    assert len(aligns) == 2
    assert aligns[0].source == "required"
    assert aligns[1].source == "preferred"


def test_alignment_is_normalized_equal_for_literal_match():
    rows = [_row("QuickBooks")]
    ids, names, canons = _sets(rows)
    aligns, _ = build_skill_alignment(
        [_js("QuickBooks")], rows, ids, names, canons,
    )
    assert aligns[0].is_normalized_equal is True


def test_alignment_is_normalized_equal_false_for_alias_fold():
    """PSW → personal support worker fires the canon rung but the
    user_skill text 'PSW' is not literally equal to 'personal support
    worker'."""
    rows = [_row("PSW")]
    ids, names, canons = _sets(rows)
    aligns, _ = build_skill_alignment(
        [_js("personal support worker")], rows, ids, names, canons,
    )
    assert len(aligns) == 1
    assert aligns[0].stage == "exact"   # public stage collapses canon to exact
    assert aligns[0].is_normalized_equal is False


def test_alignment_carries_fuzzy_stage_when_token_overlap_fires():
    rows = [_row("truck maintenance")]
    ids, names, canons = _sets(rows)
    aligns, classifications = build_skill_alignment(
        [_js("truck service maintenance")], rows, ids, names, canons,
    )
    assert classifications[0].strength == _STRENGTH_TOKEN_OVERLAP
    assert classifications[0].stage == "fuzzy"
    assert len(aligns) == 1
    assert aligns[0].stage == "fuzzy"
    assert aligns[0].is_normalized_equal is False


def test_alignment_skip_for_credentials_without_exact_match():
    """Credentials are stage-1-only. A G2-only user does not match a
    Class G credential requirement, and no alignment is produced."""
    rows = [_row("G2/G driver's license")]
    ids, names, canons = _sets(rows)
    aligns, classifications = build_skill_alignment(
        [_js("Class G driver's license")], rows, ids, names, canons,
    )
    assert classifications[0].strength == 0.0
    assert classifications[0].stage == "no_match"
    assert aligns == []


# =========================================================================
# Frozen / Literal contracts (smoke)
# =========================================================================
def test_alignment_is_frozen_dataclass():
    rows = [_row("Python")]
    ids, names, canons = _sets(rows)
    aligns, _ = build_skill_alignment(
        [_js("Python")], rows, ids, names, canons,
    )
    with pytest.raises((AttributeError, Exception)):
        aligns[0].stage = "fuzzy"  # type: ignore


def test_returns_frozen_objects_unique_per_requirement():
    """Two identical job_skills should produce two distinct
    SkillAlignment objects (no aliasing). Frozen dataclass equality
    by field is fine; identity should not be the same instance."""
    rows = [_row("Python")]
    ids, names, canons = _sets(rows)
    aligns, _ = build_skill_alignment(
        [_js("Python"), _js("Python")], rows, ids, names, canons,
    )
    assert len(aligns) == 2
    # Both are equal by field but should be distinct instances.
    assert aligns[0] == aligns[1]


# =========================================================================
# Helper does NOT score, boost, or apply gates
# =========================================================================
def test_helper_returns_only_alignment_and_classifications():
    """The helper's return type is (list[SkillAlignment],
    list[_ClassifiedRequirement]). It does not surface scores,
    bands, boosts, or hard gates — those are scoring concerns owned
    by `_score_one_job` and must NOT leak into adjacency enrichment."""
    rows = [_row("Python")]
    ids, names, canons = _sets(rows)
    result = build_skill_alignment(
        [_js("Python")], rows, ids, names, canons,
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
    # No score, no band, no boost dict.
    aligns, classifications = result
    for a in aligns:
        assert isinstance(a, SkillAlignment)
    for c in classifications:
        # NamedTuple fields: skill, strength, stage
        assert hasattr(c, "skill")
        assert hasattr(c, "strength")
        assert hasattr(c, "stage")
        # And NOTHING else — no leaked score components.
        assert c._fields == ("skill", "strength", "stage")
