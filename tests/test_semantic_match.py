"""Unit tests for matching v2 step 5 -- semantic re-ranker.

Pins the contract:
  * Semantic only fires when every lexical path returned strength 0.0
  * Cosine >= SEMANTIC_COSINE_THRESHOLD (default 0.70) -> strength SEMANTIC_STRENGTH_CAP (default 0.75)
  * Cosine below threshold -> 0.0 (no semantic match)
  * max-wins still rules: semantic at 0.75 NEVER overrides lexical fuzzy 0.85 or exact 1.0
  * score_explanation.*_match_stages carries "exact" / "fuzzy" / "semantic"
  * Graceful: when embeddings are absent (None), semantic stage is silent
    and engine falls back to pure lexical (no crash, no error)

No real sentence-transformer model is loaded -- the user-side matrix and
job-side embedding are constructed by hand from small fake float arrays.
"""
from __future__ import annotations

import os

os.environ.setdefault("LLM_ENABLED", "false")

import numpy as np
import pytest

from config import SEMANTIC_COSINE_THRESHOLD, SEMANTIC_STRENGTH_CAP
from skillbridge.match import engine
from skillbridge.match.engine import (
    _STRENGTH_STAGE_1,
    _STRENGTH_TOKEN_OVERLAP,
    _semantic_match_strength,
    _skill_match_strength,
)

pytestmark = pytest.mark.nodb


def _unit_vec(*coords) -> np.ndarray:
    """Helper: build a normalized float32 vector for fake embeddings."""
    v = np.asarray(coords, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


def _user_matrix(*vecs) -> np.ndarray:
    """Stack user-side vectors into the (N, dim) matrix the engine expects."""
    return np.vstack([_unit_vec(*v) for v in vecs]).astype(np.float32)


# ---------------------------------------------------------------------------
# _semantic_match_strength -- pure cosine + threshold + cap
# ---------------------------------------------------------------------------
def test_semantic_fires_at_full_cosine():
    """Perfect alignment (cosine 1.0) -> strength SEMANTIC_STRENGTH_CAP."""
    job_vec = _unit_vec(1, 0, 0)
    user_mat = _user_matrix((1, 0, 0))
    strength, idx = _semantic_match_strength(job_vec, user_mat)
    assert strength == SEMANTIC_STRENGTH_CAP
    assert idx == 0   # single row, must be index 0


def test_semantic_silent_below_threshold():
    """Cosine 0.0 -> 0.0 strength (no semantic match)."""
    job_vec = _unit_vec(1, 0, 0)
    user_mat = _user_matrix((0, 1, 0))   # orthogonal
    assert _semantic_match_strength(job_vec, user_mat) == (0.0, None)


def test_semantic_uses_best_match_across_user_skills():
    """When the user has multiple skills, the max cosine wins.
    AR-9.feat.coach-tiers CP1: the function now also returns the
    winning row index (index 1 here, where cosine = 1.0)."""
    job_vec = _unit_vec(1, 0, 0)
    user_mat = _user_matrix(
        (0, 1, 0),    # cosine 0
        (1, 0, 0),    # cosine 1.0 -- this one
        (0, 0, 1),    # cosine 0
    )
    assert _semantic_match_strength(job_vec, user_mat) == (SEMANTIC_STRENGTH_CAP, 1)


def test_semantic_threshold_just_above_fires():
    """Cosine clearly above threshold -> fires.

    We test just-above / just-below rather than the exact threshold
    because float32 precision can make exact-threshold computation
    drift by ~1e-7 either way (which would make the test brittle).
    The contract is "cosine well above threshold fires; well below
    does not" -- the half-integer-percent margin is large enough to
    survive precision drift on any modern CPU.
    """
    target_cos = SEMANTIC_COSINE_THRESHOLD + 0.05   # well above
    sin_part = float(np.sqrt(max(0.0, 1.0 - target_cos * target_cos)))
    job_vec = np.asarray([1.0, 0.0], dtype=np.float32)
    user_mat = np.asarray([[target_cos, sin_part]], dtype=np.float32)
    assert _semantic_match_strength(job_vec, user_mat) == (SEMANTIC_STRENGTH_CAP, 0)


def test_semantic_threshold_just_below_silent():
    """Cosine clearly below threshold -> no match."""
    target_cos = SEMANTIC_COSINE_THRESHOLD - 0.05   # well below
    sin_part = float(np.sqrt(max(0.0, 1.0 - target_cos * target_cos)))
    job_vec = np.asarray([1.0, 0.0], dtype=np.float32)
    user_mat = np.asarray([[target_cos, sin_part]], dtype=np.float32)
    assert _semantic_match_strength(job_vec, user_mat) == (0.0, None)


def test_semantic_none_inputs_return_zero():
    """Both None and empty matrix -> 0.0 (no embeddings available -> no match)."""
    assert _semantic_match_strength(None, _user_matrix((1, 0))) == (0.0, None)
    assert _semantic_match_strength(_unit_vec(1, 0), None) == (0.0, None)
    empty = np.zeros((0, 2), dtype=np.float32)
    assert _semantic_match_strength(_unit_vec(1, 0), empty) == (0.0, None)


def test_semantic_cap_constants_have_expected_values():
    """The threshold and cap are load-bearing. Pin them so a future
    config change doesn't silently relax the contract."""
    assert SEMANTIC_COSINE_THRESHOLD == 0.70
    assert SEMANTIC_STRENGTH_CAP == 0.75
    # Cap must stay BELOW token-overlap fuzzy so max-wins prevents
    # semantic from overriding lexical fuzzy matches.
    assert SEMANTIC_STRENGTH_CAP < _STRENGTH_TOKEN_OVERLAP
    # And well below exact.
    assert SEMANTIC_STRENGTH_CAP < _STRENGTH_STAGE_1


# ---------------------------------------------------------------------------
# _skill_match_strength -- semantic only fires when lexical = 0.0
# ---------------------------------------------------------------------------
def test_skill_strength_semantic_when_no_lexical_match():
    """User has only 'react', job needs 'frontend development'. No
    lexical overlap. Pre-computed embeddings make cosine = 1.0 ->
    semantic fires with stage='semantic'."""
    job_skill = {"skill_id": None, "skill_name": "frontend development"}
    user_skills = {"react"}
    # Make the lexical paths definitely miss (no shared tokens of len>=2).
    # Then supply embeddings that match perfectly.
    job_emb = _unit_vec(1, 0, 0)
    user_mat = _user_matrix((1, 0, 0))
    strength, stage = _skill_match_strength(
        job_skill, set(), user_skills, set(),
        job_skill_embedding=job_emb,
        user_embeddings_matrix=user_mat,
    )
    assert strength == SEMANTIC_STRENGTH_CAP
    assert stage == "semantic"


def test_skill_strength_lexical_exact_beats_semantic():
    """Even with a perfect semantic match available, an exact lexical
    match (strength 1.0) wins. The engine never even computes semantic
    when lexical exact fires."""
    job_skill = {"skill_id": None, "skill_name": "python"}
    user_skills = {"python"}
    # Provide misleading embeddings that would suggest semantic.
    # If the function reached semantic stage, it would still return 1.0
    # cap -> 0.75 strength. But exact comes first and wins with 1.0.
    job_emb = _unit_vec(1, 0, 0)
    user_mat = _user_matrix((1, 0, 0))
    strength, stage = _skill_match_strength(
        job_skill, set(), user_skills, set(),
        job_skill_embedding=job_emb,
        user_embeddings_matrix=user_mat,
    )
    assert strength == _STRENGTH_STAGE_1
    assert stage == "exact"


def test_skill_strength_lexical_fuzzy_beats_semantic():
    """Token-overlap fuzzy (0.85) > semantic cap (0.75). Lexical wins."""
    job_skill = {"skill_id": None, "skill_name": "truck service and maintenance"}
    # User skill that produces token-overlap fuzzy match (2/3 overlap).
    user_skills = {"truck maintenance"}
    # Provide perfect semantic embeddings; they should still be ignored.
    job_emb = _unit_vec(1, 0, 0)
    user_mat = _user_matrix((1, 0, 0))
    strength, stage = _skill_match_strength(
        job_skill, set(), user_skills, set(),
        job_skill_embedding=job_emb,
        user_embeddings_matrix=user_mat,
    )
    assert strength == _STRENGTH_TOKEN_OVERLAP
    assert stage == "fuzzy"


def test_skill_strength_silent_when_embeddings_absent():
    """No semantic embeddings provided -> falls through cleanly to
    no-match without crashing. (Graceful when sentence-transformers
    isn't installed.)"""
    job_skill = {"skill_id": None, "skill_name": "frontend development"}
    user_skills = {"react"}
    strength, stage = _skill_match_strength(
        job_skill, set(), user_skills, set(),
        job_skill_embedding=None,
        user_embeddings_matrix=None,
    )
    assert strength == 0.0
    assert stage == "no_match"


def test_skill_strength_silent_when_only_one_side_has_embedding():
    """Mixed: job has an embedding but user doesn't (e.g. user skills
    encoded fine but a particular job-side skill missing in DB), or
    vice versa. Either way -> no semantic match, falls through."""
    job_skill = {"skill_id": None, "skill_name": "frontend development"}
    user_skills = {"react"}
    job_emb = _unit_vec(1, 0, 0)
    strength, stage = _skill_match_strength(
        job_skill, set(), user_skills, set(),
        job_skill_embedding=job_emb,
        user_embeddings_matrix=None,
    )
    assert (strength, stage) == (0.0, "no_match")
    strength, stage = _skill_match_strength(
        job_skill, set(), user_skills, set(),
        job_skill_embedding=None,
        user_embeddings_matrix=_user_matrix((1, 0, 0)),
    )
    assert (strength, stage) == (0.0, "no_match")


# ---------------------------------------------------------------------------
# Engine integration: score_explanation surfaces match_stages
# ---------------------------------------------------------------------------
def _make_job(**overrides) -> dict:
    base = {
        "job_id": "job-test",
        "title": "Test Role",
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


def test_score_explanation_includes_match_stages(monkeypatch):
    """The new required_match_stages parallel array must appear on the
    main eligible path with the same length as required_matched."""
    monkeypatch.setattr(engine, "_regulated", lambda *a, **k: None)
    skills = [
        _make_skill("python"),       # user exact -> "exact"
        _make_skill("welding"),      # missing
        _make_skill("teamwork"),     # user exact -> "exact"
    ]
    result = engine._score_one_job(
        job=_make_job(),
        job_skills=skills,
        user_skill_ids=set(),
        user_skill_names={"python", "teamwork"},
        profile=_make_profile(),
    )
    se = result.score_explanation
    assert "required_match_stages" in se
    assert "preferred_match_stages" in se
    assert len(se["required_match_stages"]) == len(se["required_matched"])
    # Every matched skill should be tagged with one of the known stages
    for stage in se["required_match_stages"]:
        assert stage in {"exact", "fuzzy", "semantic"}


def test_score_explanation_semantic_stage_surfaces(monkeypatch):
    """When semantic fires (no lexical overlap, embeddings provided),
    the stage label must be 'semantic'."""
    monkeypatch.setattr(engine, "_regulated", lambda *a, **k: None)
    skills = [
        _make_skill("frontend development"),   # no lexical match
        _make_skill("communication"),          # filler so we have >=3
        _make_skill("teamwork"),
    ]
    # Build embeddings: "frontend development" semantically matches
    # the user's "react" via the fake unit vector.
    job_embeddings = {
        "frontend development": _unit_vec(1, 0, 0),
        "communication": _unit_vec(0, 1, 0),
        "teamwork": _unit_vec(0, 0, 1),
    }
    user_mat = _user_matrix((1, 0, 0))   # only matches "frontend development"

    result = engine._score_one_job(
        job=_make_job(),
        job_skills=skills,
        user_skill_ids=set(),
        user_skill_names={"react"},
        profile=_make_profile(),
        user_embeddings_matrix=user_mat,
        job_skill_embeddings=job_embeddings,
    )
    se = result.score_explanation
    # Locate the matched skill and confirm stage tag is 'semantic'
    matched = list(zip(se["required_matched"], se["required_match_stages"]))
    semantic_hits = [name for name, stage in matched if stage == "semantic"]
    assert "frontend development" in semantic_hits


def test_semantic_does_not_override_lexical_when_both_apply(monkeypatch):
    """If a job-side skill matches both lexically AND semantically, the
    lexical strength must win (max-wins). Stage label must be 'exact'
    or 'fuzzy', NEVER 'semantic'."""
    monkeypatch.setattr(engine, "_regulated", lambda *a, **k: None)
    skills = [
        _make_skill("python"),       # user has exact match
        _make_skill("communication"),
        _make_skill("teamwork"),
    ]
    # Even with a strong semantic embedding for "python", lexical wins.
    job_embeddings = {
        "python": _unit_vec(1, 0, 0),
    }
    user_mat = _user_matrix((1, 0, 0))

    result = engine._score_one_job(
        job=_make_job(),
        job_skills=skills,
        user_skill_ids=set(),
        user_skill_names={"python"},
        profile=_make_profile(),
        user_embeddings_matrix=user_mat,
        job_skill_embeddings=job_embeddings,
    )
    se = result.score_explanation
    # python's stage label must be 'exact', not 'semantic'.
    matched = dict(zip(se["required_matched"], se["required_match_stages"]))
    assert matched.get("python") == "exact"


# ---------------------------------------------------------------------------
# _maybe_embed_user_skill_rows graceful path
# (AR-9.feat.coach-tiers CP1: renamed from _maybe_embed_user_skills and now
# encodes UserSkillRow.name in row order, so semantic argmax indexes map
# back to rows[i].text for attribution.)
# ---------------------------------------------------------------------------
def test_maybe_embed_returns_none_when_no_skills():
    """Empty rows list -> None (no need to load the model)."""
    assert engine._maybe_embed_user_skill_rows([]) is None


def test_maybe_embed_returns_none_when_embedder_missing(monkeypatch):
    """sentence-transformers not installed -> None (graceful fallback)."""
    from skillbridge.embed import service as embed_service
    from skillbridge.match.alignment import UserSkillRow
    monkeypatch.setattr(embed_service, "get_embedder", lambda: None)
    rows = [UserSkillRow(skill_id=None, text="Python", name="python", canon="python")]
    assert engine._maybe_embed_user_skill_rows(rows) is None
