"""Step 2 anti-regression tests (2026-07-16).

Invariant:
    Title similarity is not measured by the V1 scoring path. A future
    retrieval layer may introduce it as retrieval relevance, but it
    cannot affect qualification fit.

Load-bearing assertions:
  - Removing / varying `target_role_text` (with target_noc and every
    non-title input held constant) does NOT change match_score,
    match_band, or match_eligible.
  - `_direct_title_match_score`, `_target_role_boost`, and
    `_target_role_similarity` are deleted (not just renamed).
  - The sparse-posting fallback at engine.py's ineligible branch is
    preserved and returns a deterministic ineligible MatchResult with
    match_score=0.0, match_band="low", match_eligible=False,
    ineligibility_reason starting with "too_few_required_skills".
  - Score-components schema has no `target_role` boost key and no
    `title_match` sub-dict.
  - The responder-facing surfaces (projector + serializer +
    responder prompt) do not expose title_match_similarity,
    title_match_override, or title_match — leaving them visible would
    let the responder cite title similarity as fit evidence.
  - `_target_noc_boost` STILL contributes to fit. This is documented
    temporary V1 behavior — NOT an endorsement of the V2 architecture.
    Locking it here prevents silent scope creep in Step 2.

Written to run under `pytest.mark.nodb` — every DB touchpoint is
either avoided (target_role_text=None disables `_regulated`) or
monkeypatched.
"""
from __future__ import annotations

import os

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from skillbridge.match import engine

pytestmark = pytest.mark.nodb


# ---------------------------------------------------------------------------
# Builders — same shape as test_hard_gates.py so behavior stays comparable.
# ---------------------------------------------------------------------------
def _make_job(
    *,
    title: str = "Customer Service Representative",
    noc_code: str | None = None,
    employment_type: str | None = None,
) -> dict:
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


def _make_profile(
    *,
    target_role_text: str | None = None,
    target_noc: str | None = None,
    experience_text: str | None = "5 years of relevant experience",
    work_type_preference: str | None = None,
) -> dict:
    return {
        "profile_id": "profile-test",
        "preferred_location": "Sault Ste. Marie",
        "target_role_text": target_role_text,
        "target_noc": target_noc,
        "work_type_preference": work_type_preference,
        "shift_preference": None,
        "experience_text": experience_text,
    }


def _make_skill(name: str, *, skill_type: str | None = "required") -> dict:
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
    job: dict,
    profile: dict,
):
    return engine._score_one_job(
        job=job,
        job_skills=job_skills,
        user_skill_ids=set(),
        user_skill_names=user_skill_names,
        profile=profile,
    )


# ---------------------------------------------------------------------------
# 1. Title independence — THE load-bearing test.
# ---------------------------------------------------------------------------
def test_target_role_text_does_not_change_score(monkeypatch):
    """With target_noc and every non-title input held constant, changing
    only `target_role_text` must not change match_score, match_band, or
    match_eligible.

    Stub `_regulated` so the title-derived licensing lookup (target_role
    feeds `_regulated` for credential warnings) doesn't add a confound.
    """
    monkeypatch.setattr(engine, "_regulated", lambda *a, **k: None)

    job = _make_job(title="Front Desk Agent", noc_code="64409")
    skills = [
        _make_skill("customer service"),
        _make_skill("phone communication"),
        _make_skill("payment processing"),
    ]
    user_skills = {"customer service", "phone communication"}

    variations = [
        None,                    # no target
        "Front Desk Agent",      # exact title match
        "Front Desk",            # partial fuzzy match
        "Nurse",                 # unrelated
        "Manager",               # generic role word
    ]

    results = []
    for target in variations:
        profile = _make_profile(
            target_role_text=target,
            target_noc="64409",   # HELD CONSTANT — matches job NOC
        )
        r = _score(
            job_skills=skills,
            user_skill_names=user_skills,
            job=job,
            profile=profile,
        )
        results.append((target, r.match_score, r.match_band, r.match_eligible))

    # Every variation must produce identical score/band/eligibility.
    baseline_score, baseline_band, baseline_eligible = (
        results[0][1], results[0][2], results[0][3]
    )
    for target, score, band, eligible in results[1:]:
        assert score == baseline_score, (
            f"target_role_text={target!r} changed match_score: "
            f"{baseline_score} -> {score}"
        )
        assert band == baseline_band, (
            f"target_role_text={target!r} changed match_band: "
            f"{baseline_band!r} -> {band!r}"
        )
        assert eligible == baseline_eligible, (
            f"target_role_text={target!r} changed match_eligible: "
            f"{baseline_eligible} -> {eligible}"
        )


# ---------------------------------------------------------------------------
# 2. Dead-function assertions — grep-style anti-regression.
# ---------------------------------------------------------------------------
def test_direct_title_match_score_is_deleted():
    assert not hasattr(engine, "_direct_title_match_score"), (
        "_direct_title_match_score was retired in Step 2 (2026-07-16). "
        "It must not be reintroduced without amending the Step 2 invariant."
    )


def test_target_role_boost_is_deleted():
    assert not hasattr(engine, "_target_role_boost"), (
        "_target_role_boost was retired in Step 2 (2026-07-16). "
        "Title similarity does not contribute to fit."
    )


def test_target_role_similarity_is_deleted():
    assert not hasattr(engine, "_target_role_similarity"), (
        "_target_role_similarity was retired in Step 2 (2026-07-16) as "
        "dead code once its callers were removed. If a future retrieval "
        "layer needs title similarity, it must live in a retrieval module "
        "with its own contract, not here."
    )


# ---------------------------------------------------------------------------
# 3. Sparse-posting fallback preserved.
# ---------------------------------------------------------------------------
def test_sparse_posting_with_matching_title_returns_low_ineligible(monkeypatch):
    """Sparse posting (< min_required_skills_for_eligibility) + exact
    title match must return a deterministic ineligible MatchResult, NOT
    the retired direct-title early-return."""
    monkeypatch.setattr(engine, "_regulated", lambda *a, **k: None)

    # Two skills -> below the default min_required_skills_for_eligibility (3).
    skills = [_make_skill("communication"), _make_skill("customer focus")]
    job = _make_job(title="Front Desk Agent")
    profile = _make_profile(target_role_text="Front Desk Agent")

    result = _score(
        job_skills=skills,
        user_skill_names={"communication", "customer focus"},
        job=job,
        profile=profile,
    )
    assert result is not None
    assert result.match_score == 0.0
    assert result.match_band == "low"
    assert result.match_eligible is False
    assert result.ineligibility_reason.startswith("too_few_required_skills")


# ---------------------------------------------------------------------------
# 4. Score-components schema — no title-derived keys.
# ---------------------------------------------------------------------------
def test_score_components_boosts_has_no_target_role_key(monkeypatch):
    monkeypatch.setattr(engine, "_regulated", lambda *a, **k: None)
    skills = [_make_skill("a"), _make_skill("b"), _make_skill("c")]
    result = _score(
        job_skills=skills,
        user_skill_names={"a", "b", "c"},
        job=_make_job(),
        profile=_make_profile(target_role_text="Whatever"),
    )
    assert result.score_explanation is not None
    boosts = result.score_explanation["score_components"]["boosts"]
    assert "target_role" not in boosts, (
        "Step 2 retired target_role from score_components.boosts. "
        "Leaving it exposed would let the responder cite title-derived "
        "fit even though title no longer affects score."
    )


def test_score_components_has_no_title_match_subdict(monkeypatch):
    monkeypatch.setattr(engine, "_regulated", lambda *a, **k: None)
    skills = [_make_skill("a"), _make_skill("b"), _make_skill("c")]
    result = _score(
        job_skills=skills,
        user_skill_names={"a", "b", "c"},
        job=_make_job(),
        profile=_make_profile(target_role_text="Whatever"),
    )
    sc = result.score_explanation["score_components"]
    assert "title_match" not in sc


def test_score_explanation_has_no_title_match_fields(monkeypatch):
    monkeypatch.setattr(engine, "_regulated", lambda *a, **k: None)
    skills = [_make_skill("a"), _make_skill("b"), _make_skill("c")]
    result = _score(
        job_skills=skills,
        user_skill_names={"a", "b", "c"},
        job=_make_job(),
        profile=_make_profile(target_role_text="Whatever"),
    )
    se = result.score_explanation
    assert "title_match_similarity" not in se
    assert "title_match_override" not in se


# ---------------------------------------------------------------------------
# 5. Responder-facing surface — projector + serializer + prompt.
# ---------------------------------------------------------------------------
def test_projector_dataclasses_have_no_title_match_fields():
    """The projector view must not carry title_match_similarity,
    title_match_override, or a TitleMatchView. Leaving them would let
    the responder cite title similarity as fit evidence."""
    from dataclasses import fields as dc_fields

    from skillbridge.chat import url_views

    # ScoreExplanationView
    se_field_names = {f.name for f in dc_fields(url_views.ScoreExplanationView)}
    assert "title_match_similarity" not in se_field_names
    assert "title_match_override" not in se_field_names

    # ScoreComponentsView
    sc_field_names = {f.name for f in dc_fields(url_views.ScoreComponentsView)}
    assert "title_match" not in sc_field_names

    # BoostsView
    b_field_names = {f.name for f in dc_fields(url_views.BoostsView)}
    assert "target_role" not in b_field_names
    assert "location" not in b_field_names

    # TitleMatchView entirely gone
    assert not hasattr(url_views, "TitleMatchView")
    assert not hasattr(url_views, "_project_title_match")


def test_responder_prompt_has_no_title_match_authorization():
    """The NEXT_ACTION_RESPONDER_PROMPT must not list title_match,
    title_match_similarity, or title_match_override as citable
    grounding fields."""
    from skillbridge.chat.prompts import NEXT_ACTION_RESPONDER_PROMPT

    text = NEXT_ACTION_RESPONDER_PROMPT
    # "title_match" catches both bare "title_match" and prefixed variants.
    assert "title_match" not in text, (
        "Step 2 retired title similarity from responder grounding. "
        "Any 'title_match*' authorization in the prompt would let the "
        "LLM cite title similarity as fit evidence."
    )


# ---------------------------------------------------------------------------
# 6. NOC boundary preservation — TEMPORARY V1 BEHAVIOR, not endorsement.
# ---------------------------------------------------------------------------
def test_target_noc_boost_still_contributes_when_matched(monkeypatch):
    """This test locks V1 behavior explicitly against silent scope creep.
    NOC's place in fit vs. retrieval is a decision for the future
    MatchResultV2 design, which will replace this whole scoring model
    with a three-comparator qualification matcher.

    Right now, matching NOC on both sides contributes +0.10 to score
    through the boost sum. Removing that behavior is out of Step 2's
    scope by explicit reviewer decision.
    """
    monkeypatch.setattr(engine, "_regulated", lambda *a, **k: None)
    # Partial-skill match (2 of 3) leaves skill_base below the 1.0
    # ceiling so the +0.10 NOC boost can be observed. A full-match
    # skill_base of 1.0 would clip both cases to score=1.0 and hide
    # the boost.
    skills = [_make_skill("a"), _make_skill("b"), _make_skill("c")]
    user_skills = {"a", "b"}

    # Case A: NOC matches on both sides.
    result_match = _score(
        job_skills=skills,
        user_skill_names=user_skills,
        job=_make_job(noc_code="14100"),
        profile=_make_profile(target_noc="14100"),
    )

    # Case B: NOC differs.
    result_mismatch = _score(
        job_skills=skills,
        user_skill_names=user_skills,
        job=_make_job(noc_code="14100"),
        profile=_make_profile(target_noc="99999"),
    )

    # NOC match must contribute a positive delta.
    assert result_match.match_score > result_mismatch.match_score, (
        "V1 NOC-match boost has been silently removed. Step 2 explicitly "
        "left it in place; removing it requires a separate design pass."
    )
