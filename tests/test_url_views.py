"""AR-9.bug.2a sub-step 3: SanitizedResponderView + projected types.

Covers:
  - Frozen-dataclass invariants on every projected view type
  - SanitizedURL.from_validated factory + direct construction (both
    accepted; from_validated is the production path)
  - Move-gated URL validation: only URLs the current move surfaces
    are validated; URLs at other positions are ignored
  - Single-pass validation: a URL at the same occurrence_path
    validated at most once per view construction
  - rejected_source_urls correctness (one entry per failing path,
    with structural identity)
  - Per-move allowlist scoping (prompt_urls / fallback_urls scoped to
    current move's populated slots, not unioned across moves)
  - Non-string source URL handling: None, "", int, list, dict, bool
    are silently suppressed; no synthetic Violation produced
  - Strict bool/None cap-flag projection (no truthy coercion)
  - Parity tests: _project_narration_skills vs responder._narration_skill_view;
    _project_score_explanation cap behavior vs responder._capped_score_explanation
  - MappingProxyType immutability enforcement on the Mapping field
  - Adjacent-recommendation projection structurally drops url
  - FallbackAdjacentRoleView has no url attribute (structural)
  - ScoreExplanationView full field population (engine.py schema)

No consumer migration here. No responder/handler edits. The builder
exists in isolation; tests construct minimal fake inputs.
"""
from __future__ import annotations

import dataclasses
from types import MappingProxyType

import pytest

pytestmark = pytest.mark.nodb


from skillbridge.chat.url_policy import (
    Validated,
    Violation,
    ViolationCode,
    hash_raw_token,
    validate,
)
from skillbridge.chat.url_views import (
    BoostsView,
    FallbackAdjacentRecommendationView,
    FallbackAdjacentRoleView,
    FallbackResultView,
    PromptAdjacentRecommendationsContainerView,
    PromptAdjacentRecommendationView,
    PromptAdjacentRoleView,
    PromptResultView,
    RejectedSourceURL,
    SanitizedResponderView,
    SanitizedURL,
    ScoreComponentsView,
    ScoreExplanationView,
    SkillBaseView,
    TrainingView,
    _enumerate_url_occurrence,
    _project_cap_flag,
    _project_narration_skills,
    _project_score_explanation,
    build_sanitized_responder_view_v1,
    build_sanitized_responder_view_v2,
)


# =========================================================================
# Fake input shapes — duck-typed minimal substitutes for
# ResponderInput / ResponderV2Input
# =========================================================================
class _FakeDecision:
    def __init__(self, **kwargs):
        # v1: show_matches + action. v2: final_move + other flags.
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeInputV1:
    """Substitute for ResponderInput. The builder only accesses
    decision, results, training_by_job — anything else is irrelevant
    here.
    """
    def __init__(
        self,
        decision: _FakeDecision,
        results=None,
        training_by_job=None,
    ):
        self.decision = decision
        self.results = list(results) if results else []
        self.training_by_job = dict(training_by_job) if training_by_job else {}


class _FakeInputV2:
    """Substitute for ResponderV2Input."""

    def __init__(
        self,
        decision: _FakeDecision,
        results=None,
        training_by_job=None,
        adjacent_recommendations_payload=None,
        adjacent_role_description_payload=None,
        near_miss_payload=None,
        remaining_gaps_payload=None,
    ):
        self.decision = decision
        self.results = list(results) if results else []
        self.training_by_job = dict(training_by_job) if training_by_job else {}
        self.adjacent_recommendations_payload = adjacent_recommendations_payload
        self.adjacent_role_description_payload = adjacent_role_description_payload
        self.near_miss_payload = near_miss_payload
        self.remaining_gaps_payload = remaining_gaps_payload


def _v2(move: str, **kwargs) -> _FakeInputV2:
    return _FakeInputV2(decision=_FakeDecision(final_move=move), **kwargs)


def _good_url(suffix: str = "/x") -> str:
    return f"https://example.com{suffix}"


# =========================================================================
# SanitizedURL: factory + direct construction
# =========================================================================
def test_sanitized_url_from_validated_factory():
    result = validate(_good_url("/jobs/123"))
    assert isinstance(result, Validated)
    s = SanitizedURL.from_validated(result)
    assert s.raw == result.raw_token
    assert s.canonical == result.canonical
    assert s.hash_sha256 == result.raw_token_hash


def test_sanitized_url_direct_construction_allowed_in_tests():
    """Direct construction is structurally available (used by fixtures)."""
    s = SanitizedURL(raw="raw", canonical="canon", hash_sha256="abc")
    assert s.raw == "raw"
    assert s.canonical == "canon"
    assert s.hash_sha256 == "abc"


def test_sanitized_url_factory_and_direct_produce_equal_objects():
    result = validate(_good_url())
    assert isinstance(result, Validated)
    via_factory = SanitizedURL.from_validated(result)
    via_direct = SanitizedURL(
        raw=result.raw_token,
        canonical=result.canonical,
        hash_sha256=result.raw_token_hash,
    )
    assert via_factory == via_direct


def test_sanitized_url_is_frozen():
    s = SanitizedURL(raw="r", canonical="c", hash_sha256="h")
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.canonical = "other"  # type: ignore[misc]


# =========================================================================
# Frozen-dataclass invariants
# =========================================================================
@pytest.mark.parametrize("cls,kwargs", [
    (RejectedSourceURL, dict(
        occurrence_path="results[0].url",
        violation=Violation(
            code=ViolationCode.URL_MALFORMED,
            raw_token="x", raw_token_hash=hash_raw_token("x"),
            safe_scheme=None, safe_host=None,
        ),
    )),
    (SkillBaseView, dict(
        value=0.5, mode="m", required_match_ratio=0.5, required_weight=0.8,
        preferred_match_ratio=0.5, preferred_weight=0.2,
    )),
    (BoostsView, dict(
        recency=0.0, target_noc_match=0.0,
        work_type_fit=0.0, shift_fit=0.0,
    )),
    (FallbackAdjacentRecommendationView, dict(
        title="T", employer=None, evidence_summary=None,
    )),
])
def test_view_dataclasses_are_frozen(cls, kwargs):
    obj = cls(**kwargs)
    field_name = next(iter(kwargs))
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(obj, field_name, kwargs[field_name])


# =========================================================================
# Structural absence of url on the no-URL projections
# =========================================================================
def test_fallback_adjacent_role_view_exposes_url_field():
    """Bug.2b amendment: FallbackAdjacentRoleView gained a `url` field
    so the deterministic fallback renderer can surface the
    already-validated SanitizedURL inline (no extra round-trip).
    has_validated_url stays as a defensive cross-check; both fields
    derive from the same projection step.
    """
    field_names = {f.name for f in dataclasses.fields(FallbackAdjacentRoleView)}
    assert "url" in field_names
    assert "has_validated_url" in field_names


def test_prompt_adjacent_recommendation_view_has_no_url_attribute():
    field_names = {
        f.name for f in dataclasses.fields(PromptAdjacentRecommendationView)
    }
    assert "url" not in field_names


def test_fallback_adjacent_recommendation_view_has_no_url_attribute():
    field_names = {
        f.name for f in dataclasses.fields(FallbackAdjacentRecommendationView)
    }
    assert "url" not in field_names


# =========================================================================
# MappingProxyType invariant on the Mapping field
# =========================================================================
def test_sanitized_view_rejects_plain_dict_for_mapping_field():
    """__post_init__ rejects a plain dict for the Mapping field."""
    with pytest.raises(TypeError, match="MappingProxyType"):
        SanitizedResponderView(
            prompt_results=(),
            fallback_results=(),
            prompt_present_matches_training_flat=(),
            prompt_present_matches_training_groups=(),
            fallback_present_matches_training_by_job={},  # not wrapped
            prompt_explain_gap_training_flat=(),
            fallback_explain_gap_training_flat=(),
            prompt_present_near_miss_training_flat=(),
            prompt_explain_remaining_gaps_training_flat=(),
            prompt_adjacent_recommendations=None,
            fallback_adjacent_recommendations=(),
            prompt_adjacent_role=None,
            fallback_adjacent_role=None,
            rejected_source_urls=(),
            prompt_urls=frozenset(),
            fallback_urls=frozenset(),
        )


def test_sanitized_view_accepts_mappingproxytype():
    view = SanitizedResponderView(
        prompt_results=(),
        fallback_results=(),
        prompt_present_matches_training_flat=(),
        prompt_present_matches_training_groups=(),
        fallback_present_matches_training_by_job=MappingProxyType({}),
        prompt_explain_gap_training_flat=(),
        fallback_explain_gap_training_flat=(),
        prompt_present_near_miss_training_flat=(),
        prompt_explain_remaining_gaps_training_flat=(),
        prompt_adjacent_recommendations=(),
        fallback_adjacent_recommendations=(),
        prompt_adjacent_role=None,
        fallback_adjacent_role=None,
        rejected_source_urls=(),
        prompt_urls=frozenset(),
        fallback_urls=frozenset(),
    )
    assert isinstance(
        view.fallback_present_matches_training_by_job, MappingProxyType,
    )


# =========================================================================
# Strict cap-flag projection
# =========================================================================
@pytest.mark.parametrize("raw_value,expected", [
    (None, None),
    (True, True),
    (False, False),
    (1, None),
    (0, None),
    ("yes", None),
    ("", None),
    ("True", None),
    ([], None),
    ({}, None),
    (1.0, None),
    (0.0, None),
])
def test_cap_flag_strict_projection(raw_value, expected):
    """isinstance(value, bool) gates the projection. 1/0 are int, not bool."""
    result = _project_cap_flag(raw_value)
    if expected is None:
        assert result is None
    else:
        assert result is expected
        assert isinstance(result, bool)


# =========================================================================
# Non-string source URL handling (Lock D)
# =========================================================================
@pytest.mark.parametrize("raw_url", [
    None, "", 0, 42, [], {}, True, False, ["https://example.com"],
])
def test_enumerate_skips_non_string_or_empty_url(raw_url):
    accumulator: list[tuple[str, str]] = []
    _enumerate_url_occurrence(raw_url, "results[0].url", accumulator)
    assert accumulator == []


def test_enumerate_accepts_non_empty_string():
    accumulator: list[tuple[str, str]] = []
    _enumerate_url_occurrence("https://x.com", "results[0].url", accumulator)
    assert accumulator == [("results[0].url", "https://x.com")]


def test_view_suppresses_non_string_source_url_no_violation():
    """A non-string url on a source position produces no Violation and
    no rejected_source_urls entry. The projected item's url is None.
    """
    inp = _v2(
        "present_matches",
        results=[{"title": "T", "employer": "E", "url": 42, "job_id": "j"}],
    )
    view = build_sanitized_responder_view_v2(inp)
    assert view.prompt_results[0].url is None
    assert view.fallback_results[0].url is None
    assert view.rejected_source_urls == ()


# =========================================================================
# Narration-skill parity (private helper vs responder original)
# =========================================================================
@pytest.mark.parametrize("skills", [
    None,
    [],
    ["a"],
    ["a", "b", "c"],
    ["a", "b", "c", "d", "e"],
    ["python", "sql", "git", "Class G", "WHMIS"],
    ["Class G", "Class A", "WHMIS"],
    ["x", "y", "z", "WHMIS", "Class G"],
    # Duplicates inside the top-N
    ["x", "x", "y"],
    # Many entries with credentials scattered after the cap
    ["a", "b", "c", "Class G", "d", "WHMIS", "e", "forklift cert"],
])
def test_narration_skill_parity(skills):
    """Private projection helper returns the same set of names as
    responder._narration_skill_view (returned as a tuple, not list).
    """
    from skillbridge.chat.responder import _narration_skill_view

    responder_out = _narration_skill_view(skills)
    view_out = _project_narration_skills(skills)
    assert view_out == tuple(responder_out), (
        f"Parity break: input={skills!r}, "
        f"responder={responder_out!r}, view={view_out!r}"
    )


# =========================================================================
# Score-explanation cap-rule parity
# =========================================================================
def _score_explanation_corpus():
    return [
        {},
        {"matched_skills": ["a", "b"]},
        {"matched_skills": ["a", "b", "c", "d", "e"]},
        {"required_matched": ["x", "y", "z", "w", "v"]},
        {
            "required_matched": ["a", "b", "c", "d", "Class G"],
            "required_match_strengths": [1.0, 0.85, 0.75, 0.5, 1.0],
            "required_match_stages": ["exact", "fuzzy", "semantic", "fuzzy", "exact"],
        },
        {
            "preferred_matched": ["p1", "p2", "p3", "WHMIS"],
            "preferred_match_strengths": [1.0, 1.0, 0.9, 1.0],
            "preferred_match_stages": ["exact", "exact", "fuzzy", "exact"],
        },
        {
            "matched_skills": ["a"],
            "missing_skills": ["b", "c", "d", "Class G", "e"],
            "required_missing": ["x", "y", "z", "WHMIS"],
            "preferred_missing": ["p", "q"],
        },
    ]


@pytest.mark.parametrize("se", _score_explanation_corpus())
def test_score_explanation_cap_parity(se):
    """The capping rules of _project_score_explanation match
    _capped_score_explanation for every represented field.
    """
    from skillbridge.chat.responder import _capped_score_explanation

    responder_capped = _capped_score_explanation(dict(se))
    view = _project_score_explanation(dict(se))

    if not se:
        assert view is None
        return

    assert view is not None
    if responder_capped is None:
        return

    if "matched_skills" in responder_capped:
        assert view.matched_skills == tuple(responder_capped["matched_skills"])
    if "missing_skills" in responder_capped:
        assert view.missing_skills == tuple(responder_capped["missing_skills"])
    if "required_missing" in responder_capped:
        assert view.required_missing == tuple(responder_capped["required_missing"])
    if "preferred_missing" in responder_capped:
        assert view.preferred_missing == tuple(responder_capped["preferred_missing"])
    if "required_matched" in responder_capped:
        assert view.required_matched == tuple(responder_capped["required_matched"])
    if "preferred_matched" in responder_capped:
        assert view.preferred_matched == tuple(responder_capped["preferred_matched"])
    if "required_match_strengths" in responder_capped:
        assert view.required_match_strengths == tuple(
            float(x) for x in responder_capped["required_match_strengths"]
        )
    if "required_match_stages" in responder_capped:
        assert view.required_match_stages == tuple(
            responder_capped["required_match_stages"]
        )
    if "preferred_match_strengths" in responder_capped:
        assert view.preferred_match_strengths == tuple(
            float(x) for x in responder_capped["preferred_match_strengths"]
        )
    if "preferred_match_stages" in responder_capped:
        assert view.preferred_match_stages == tuple(
            responder_capped["preferred_match_stages"]
        )


# =========================================================================
# ScoreExplanationView field population
# =========================================================================
def test_score_explanation_view_full_field_population():
    """Build a realistic engine.py-shaped score_explanation and verify
    every locked field is read into the projected view.
    """
    raw = {
        "matched_skills": ["sql", "python"],
        "missing_skills": ["docker"],
        "required_matched": ["sql"],
        "required_missing": ["docker"],
        "preferred_matched": ["python"],
        "preferred_missing": [],
        "required_match_strengths": [1.0],
        "required_match_stages": ["exact"],
        "preferred_match_strengths": [0.85],
        "preferred_match_stages": ["fuzzy"],
        "required_match_strength_sum": 1.0,
        "preferred_match_strength_sum": 0.85,
        "skill_match_ratio": 0.8,
        "required_match_ratio": 0.5,
        "required_total": 2,
        "preferred_match_ratio": 0.7,
        "preferred_total": 3,
        "recency_days": 10,
        "work_type_fit": "matched",
        "shift_fit": "no_signal",
        "credential_warning_present": False,
        "credential_gap_skills": ["Class G"],
        "work_type_user": "full-time",
        "work_type_job": "full-time",
        "band_capped_by_credential": True,
        "band_capped_by_no_experience": False,
        "caps_applied": ["band_capped_by_credential"],
        "score_components": {
            "skill_base": {
                "value": 0.6, "mode": "blend",
                "required_match_ratio": 0.5, "required_weight": 0.8,
                "preferred_match_ratio": 0.7, "preferred_weight": 0.2,
            },
            "boosts": {
                "recency": 0.0,
                "target_noc_match": 0.0, "work_type_fit": 0.02,
                "shift_fit": 0.0,
            },
            "score_pre_caps": 0.77,
            "score_post_caps": 0.65,
        },
    }
    view = _project_score_explanation(raw)
    assert view is not None
    assert view.matched_skills == ("sql", "python")
    assert view.missing_skills == ("docker",)
    assert view.required_match_strength_sum == 1.0
    assert view.preferred_match_strength_sum == 0.85
    assert view.skill_match_ratio == 0.8
    assert view.required_match_ratio == 0.5
    assert view.required_total == 2
    assert view.preferred_total == 3
    assert view.recency_days == 10
    assert view.work_type_fit == "matched"
    assert view.shift_fit == "no_signal"
    assert view.credential_warning_present is False
    assert view.credential_gap_skills == ("Class G",)
    assert view.work_type_user == "full-time"
    assert view.work_type_job == "full-time"
    assert view.band_capped_by_credential is True
    assert view.band_capped_by_no_experience is False
    # Absent flag becomes None (no coercion to False)
    assert view.band_capped_by_work_type_mismatch is None
    assert view.caps_applied == ("band_capped_by_credential",)
    assert isinstance(view.score_components, ScoreComponentsView)
    assert view.score_components.score_pre_caps == 0.77
    assert view.score_components.score_post_caps == 0.65
    assert isinstance(view.score_components.skill_base, SkillBaseView)
    assert view.score_components.skill_base.value == 0.6
    assert view.score_components.skill_base.mode == "blend"
    assert isinstance(view.score_components.boosts, BoostsView)
    assert view.score_components.boosts.recency == 0.0
    # title_match sub-dict retired in Step 2 cutover 2026-07-16.


def test_score_explanation_view_absent_caps_are_none():
    """The three band_capped_by_* flags are None when absent from
    the raw dict — preserving the absence/presence distinction.
    """
    view = _project_score_explanation({"matched_skills": ["a"]})
    assert view is not None
    assert view.band_capped_by_credential is None
    assert view.band_capped_by_no_experience is None
    assert view.band_capped_by_work_type_mismatch is None
    assert view.caps_applied == ()


# =========================================================================
# V1 builder
# =========================================================================
def test_v1_builder_returns_empty_view_when_show_matches_false():
    inp = _FakeInputV1(
        decision=_FakeDecision(show_matches=False),
        results=[{"title": "T", "url": _good_url(), "job_id": "j"}],
    )
    view = build_sanitized_responder_view_v1(inp)
    assert view.prompt_results == ()
    assert view.fallback_results == ()
    assert view.prompt_urls == frozenset()
    assert view.fallback_urls == frozenset()
    assert view.rejected_source_urls == ()


def test_v1_builder_populates_present_matches_when_show_matches_true():
    inp = _FakeInputV1(
        decision=_FakeDecision(show_matches=True),
        results=[
            {"title": "T1", "employer": "E1", "url": _good_url("/1"),
             "job_id": "job-1"},
            {"title": "T2", "employer": "E2", "url": _good_url("/2"),
             "job_id": "job-2"},
        ],
        training_by_job={
            "job-1": [{"provider": "P1", "url": _good_url("/t1")}],
            "job-2": [{"provider": "P2", "url": _good_url("/t2")}],
        },
    )
    view = build_sanitized_responder_view_v1(inp)
    assert len(view.prompt_results) == 2
    assert view.prompt_results[0].title == "T1"
    assert view.prompt_results[0].url is not None
    assert view.prompt_results[0].url.canonical == _good_url("/1")
    assert len(view.prompt_present_matches_training_flat) == 2
    assert view.prompt_urls == frozenset({
        _good_url("/1"), _good_url("/2"), _good_url("/t1"), _good_url("/t2"),
    })


# =========================================================================
# V2 move-gating: each move populates only its slots
# =========================================================================
def test_v2_present_matches_populates_results_and_training():
    inp = _v2(
        "present_matches",
        results=[{"title": "T", "url": _good_url("/r"), "job_id": "j1"}],
        training_by_job={"j1": [{"provider": "P", "url": _good_url("/t")}]},
    )
    view = build_sanitized_responder_view_v2(inp)
    assert len(view.prompt_results) == 1
    assert len(view.fallback_results) == 1
    # Bug.4: v2 populates the GROUPS slot, not the flat slot.
    assert view.prompt_present_matches_training_flat == ()
    assert len(view.prompt_present_matches_training_groups) == 1
    group = view.prompt_present_matches_training_groups[0]
    assert group.job_id == "j1"
    assert group.job_title == "T"
    assert len(group.resources) == 1
    assert "j1" in view.fallback_present_matches_training_by_job
    # Adjacency / near-miss / remaining slots empty
    assert view.prompt_explain_gap_training_flat == ()
    assert view.prompt_present_near_miss_training_flat == ()
    assert view.prompt_explain_remaining_gaps_training_flat == ()
    assert view.prompt_adjacent_recommendations is None
    assert view.prompt_adjacent_role is None


def test_v2_explain_gap_populates_only_explain_gap_training():
    inp = _v2(
        "explain_gap",
        training_by_job={"j1": [
            {"provider": "P1", "url": _good_url("/t1")},
            {"provider": "P2", "url": _good_url("/t2")},
        ]},
    )
    view = build_sanitized_responder_view_v2(inp)
    assert view.prompt_results == ()
    assert view.prompt_present_matches_training_flat == ()
    assert len(view.prompt_explain_gap_training_flat) == 2
    assert len(view.fallback_explain_gap_training_flat) == 2
    assert view.prompt_present_near_miss_training_flat == ()


def test_v2_present_near_miss_has_prompt_training_no_fallback():
    inp = _v2(
        "present_near_miss",
        training_by_job={"j1": [{"provider": "P", "url": _good_url("/t")}]},
    )
    view = build_sanitized_responder_view_v2(inp)
    assert len(view.prompt_present_near_miss_training_flat) == 1
    # No fallback training projection for this move
    assert view.fallback_present_matches_training_by_job == MappingProxyType({})
    assert view.fallback_explain_gap_training_flat == ()
    assert view.fallback_urls == frozenset()


def test_v2_explain_remaining_gaps_has_prompt_training_no_fallback():
    inp = _v2(
        "explain_remaining_gaps",
        training_by_job={"j1": [{"provider": "P", "url": _good_url("/t")}]},
    )
    view = build_sanitized_responder_view_v2(inp)
    assert len(view.prompt_explain_remaining_gaps_training_flat) == 1
    assert view.fallback_urls == frozenset()


def test_v2_recommend_adjacent_roles_populates_only_recommendations():
    inp = _v2(
        "recommend_adjacent_roles",
        adjacent_recommendations_payload={
            "recommendations": [
                {"title": "Welder", "employer": "ACME", "evidence_summary": "3 of 5",
                 "matched_skills": ["welding", "safety"], "why_adjacent": "skill_evidence",
                 "job_id": "j1", "location": "Sault Ste. Marie"},
            ],
            "total_retrieved": 12,
        },
    )
    view = build_sanitized_responder_view_v2(inp)
    container = view.prompt_adjacent_recommendations
    assert container is not None
    assert isinstance(container, PromptAdjacentRecommendationsContainerView)
    assert container.total_retrieved == 12
    assert len(container.recommendations) == 1
    rec = container.recommendations[0]
    assert rec.title == "Welder"
    assert rec.employer == "ACME"
    assert rec.location == "Sault Ste. Marie"
    assert rec.why_adjacent == "skill_evidence"
    assert rec.matched_skills == ("welding", "safety")
    # 0 URLs even if payload had one
    assert view.prompt_urls == frozenset()
    assert view.fallback_urls == frozenset()


def test_v2_describe_adjacent_role_validates_one_url():
    inp = _v2(
        "describe_adjacent_role",
        adjacent_role_description_payload={
            "job": {
                "title": "Maintenance Technician",
                "employer": "ACME",
                "location": "Sault Ste. Marie",
                "url": _good_url("/jobs/123"),
                "posted_date": "2026-06-10",
            },
            "evidence_summary": "3 of 5",
            "matched_skills": ["welding"],
            "expired": False,
        },
    )
    view = build_sanitized_responder_view_v2(inp)
    assert view.prompt_adjacent_role is not None
    assert view.prompt_adjacent_role.title == "Maintenance Technician"
    assert view.prompt_adjacent_role.location == "Sault Ste. Marie"
    assert view.prompt_adjacent_role.posted_date == "2026-06-10"
    assert view.prompt_adjacent_role.url is not None
    assert view.prompt_adjacent_role.url.canonical == _good_url("/jobs/123")
    assert view.fallback_adjacent_role is not None
    assert view.fallback_adjacent_role.has_validated_url is True
    assert view.fallback_adjacent_role.location == "Sault Ste. Marie"
    # Bug.2b: the fallback renderer now surfaces the validated URL
    # inline, so the canonical also appears in fallback_urls.
    assert view.prompt_urls == frozenset({_good_url("/jobs/123")})
    assert view.fallback_urls == frozenset({_good_url("/jobs/123")})


def test_v2_describe_adjacent_role_invalid_url_has_validated_url_false():
    inp = _v2(
        "describe_adjacent_role",
        adjacent_role_description_payload={
            "job": {
                "title": "X", "employer": None, "location": None,
                "url": "ftp://x.com",  # unsupported scheme
                "posted_date": None,
            },
            "evidence_summary": "", "matched_skills": [], "expired": False,
        },
    )
    view = build_sanitized_responder_view_v2(inp)
    assert view.fallback_adjacent_role is not None
    assert view.fallback_adjacent_role.has_validated_url is False
    assert view.prompt_adjacent_role.url is None
    assert view.prompt_urls == frozenset()
    # ftp:// rejected -> URL_UNSUPPORTED_SCHEME in rejected_source_urls
    assert len(view.rejected_source_urls) == 1
    rs = view.rejected_source_urls[0]
    assert rs.occurrence_path == "adjacent_role_description_payload.job.url"
    assert rs.violation.code is ViolationCode.URL_UNSUPPORTED_SCHEME


# =========================================================================
# Move-gating: URLs at irrelevant positions are NOT validated
# =========================================================================
def test_describe_adjacent_role_does_not_validate_results_urls():
    """A describe_adjacent_role turn doesn't surface results URLs.
    Those positions should NOT enter the validation set or appear in
    rejected_source_urls, even if the raw payload carries them.
    """
    inp = _v2(
        "describe_adjacent_role",
        results=[
            {"title": "X", "url": "ftp://junk.invalid"},  # would fail validation
        ],
        adjacent_role_description_payload={
            "job": {"title": "X", "url": _good_url(), "posted_date": None},
            "evidence_summary": "", "matched_skills": [], "expired": False,
        },
    )
    view = build_sanitized_responder_view_v2(inp)
    # No rejected_source_urls entry for results[0].url — that path wasn't surfaced.
    paths = {r.occurrence_path for r in view.rejected_source_urls}
    assert "results[0].url" not in paths


def test_explain_gap_does_not_validate_result_urls():
    inp = _v2(
        "explain_gap",
        results=[{"title": "X", "url": "ftp://junk.invalid"}],
        training_by_job={"j": [{"provider": "P", "url": _good_url()}]},
    )
    view = build_sanitized_responder_view_v2(inp)
    paths = {r.occurrence_path for r in view.rejected_source_urls}
    assert "results[0].url" not in paths


def test_recommend_adjacent_roles_validates_zero_urls():
    """The adjacency-recommendation projection has zero URL surface;
    no validation runs even when a hypothetical url is on the payload.
    """
    inp = _v2(
        "recommend_adjacent_roles",
        adjacent_recommendations_payload={
            "recommendations": [
                {"title": "T", "url": "ftp://junk"},  # ignored by projection
            ],
        },
    )
    view = build_sanitized_responder_view_v2(inp)
    assert view.rejected_source_urls == ()
    assert view.prompt_urls == frozenset()


# =========================================================================
# Single-pass validation: same source position validated once
# =========================================================================
def test_present_matches_single_pass_validation():
    """A result URL surfaces in both prompt_results and fallback_results;
    the underlying SanitizedURL is the same object embedded on each
    projected item, validated once.
    """
    inp = _v2(
        "present_matches",
        results=[
            {"title": "T", "url": _good_url("/r"), "job_id": "j1"},
        ],
        training_by_job={"j1": [{"provider": "P", "url": _good_url("/t")}]},
    )
    view = build_sanitized_responder_view_v2(inp)
    # Same canonical URL appears in both projections
    assert view.prompt_results[0].url is not None
    assert view.fallback_results[0].url is not None
    assert view.prompt_results[0].url == view.fallback_results[0].url
    assert view.prompt_results[0].url.canonical == _good_url("/r")
    # rejected_source_urls has zero entries (URL is valid)
    assert view.rejected_source_urls == ()


def test_present_matches_rejected_recorded_once_per_path():
    """If the same URL surfaces on prompt and fallback and fails
    validation, rejected_source_urls has ONE entry for that path.
    """
    inp = _v2(
        "present_matches",
        results=[{"title": "T", "url": "ftp://bad.invalid", "job_id": "j1"}],
        training_by_job={},
    )
    view = build_sanitized_responder_view_v2(inp)
    paths = [r.occurrence_path for r in view.rejected_source_urls]
    assert paths == ["results[0].url"]


# =========================================================================
# Allowlist scoping per move
# =========================================================================
def test_present_matches_allowlists_scoped_to_results_and_training():
    inp = _v2(
        "present_matches",
        results=[{"title": "T", "url": _good_url("/r"), "job_id": "j1"}],
        training_by_job={"j1": [{"provider": "P", "url": _good_url("/t")}]},
    )
    view = build_sanitized_responder_view_v2(inp)
    assert view.prompt_urls == frozenset({_good_url("/r"), _good_url("/t")})
    assert view.fallback_urls == frozenset({_good_url("/r"), _good_url("/t")})


def test_explain_gap_allowlists_have_only_training_urls():
    inp = _v2(
        "explain_gap",
        results=[{"title": "T", "url": _good_url("/r"), "job_id": "j1"}],
        training_by_job={"j1": [{"provider": "P", "url": _good_url("/t")}]},
    )
    view = build_sanitized_responder_view_v2(inp)
    # results URL never validated (move-gated); only training in allowlist
    assert _good_url("/r") not in view.prompt_urls
    assert _good_url("/r") not in view.fallback_urls
    assert _good_url("/t") in view.prompt_urls
    assert _good_url("/t") in view.fallback_urls


# =========================================================================
# Caps
# =========================================================================
def test_results_cap_is_5_per_move():
    """results[:5] applied; 6th onward not surfaced."""
    inp = _v2(
        "present_matches",
        results=[
            {"title": f"T{i}", "url": _good_url(f"/r{i}"), "job_id": f"j{i}"}
            for i in range(7)
        ],
    )
    view = build_sanitized_responder_view_v2(inp)
    assert len(view.prompt_results) == 5
    assert len(view.fallback_results) == 5
    titles = [r.title for r in view.prompt_results]
    assert titles == ["T0", "T1", "T2", "T3", "T4"]


def test_present_matches_prompt_training_cap_is_6_resources_total():
    """Bug.4: v2 cap is 6 total resources across groups (not flat-6).
    With a single job having 10 training entries, the one group's
    `resources` is capped at 6.
    """
    inp = _v2(
        "present_matches",
        results=[{"title": "T", "url": _good_url(), "job_id": "j1"}],
        training_by_job={
            "j1": [
                {"provider": f"P{i}", "url": _good_url(f"/t{i}")}
                for i in range(10)
            ],
        },
    )
    view = build_sanitized_responder_view_v2(inp)
    assert view.prompt_present_matches_training_flat == ()
    assert len(view.prompt_present_matches_training_groups) == 1
    assert len(view.prompt_present_matches_training_groups[0].resources) == 6


def test_present_matches_fallback_training_cap_is_2_per_job():
    inp = _v2(
        "present_matches",
        results=[
            {"title": "T1", "url": _good_url("/r1"), "job_id": "j1"},
            {"title": "T2", "url": _good_url("/r2"), "job_id": "j2"},
        ],
        training_by_job={
            "j1": [
                {"provider": f"P1-{i}", "url": _good_url(f"/t1-{i}")}
                for i in range(5)
            ],
            "j2": [
                {"provider": f"P2-{i}", "url": _good_url(f"/t2-{i}")}
                for i in range(5)
            ],
        },
    )
    view = build_sanitized_responder_view_v2(inp)
    fb = view.fallback_present_matches_training_by_job
    assert len(fb["j1"]) == 2
    assert len(fb["j2"]) == 2


def test_explain_gap_fallback_training_cap_is_3():
    inp = _v2(
        "explain_gap",
        training_by_job={
            "j1": [
                {"provider": f"P{i}", "url": _good_url(f"/t{i}")}
                for i in range(10)
            ],
        },
    )
    view = build_sanitized_responder_view_v2(inp)
    assert len(view.fallback_explain_gap_training_flat) == 3
    assert len(view.prompt_explain_gap_training_flat) == 6


# =========================================================================
# Unknown move yields empty view
# =========================================================================
def test_unknown_move_returns_empty_view():
    inp = _v2("acknowledge_and_continue")
    view = build_sanitized_responder_view_v2(inp)
    assert view.prompt_results == ()
    assert view.prompt_urls == frozenset()
    assert view.fallback_urls == frozenset()
    assert view.rejected_source_urls == ()


# =========================================================================
# RejectedSourceURL occurrence paths
# =========================================================================
def test_rejected_source_url_path_for_result():
    inp = _v2(
        "present_matches",
        results=[{"title": "T", "url": "ftp://bad", "job_id": "j1"}],
    )
    view = build_sanitized_responder_view_v2(inp)
    assert len(view.rejected_source_urls) == 1
    assert view.rejected_source_urls[0].occurrence_path == "results[0].url"


def test_rejected_source_url_path_for_training():
    inp = _v2(
        "explain_gap",
        training_by_job={"job-001": [{"provider": "P", "url": "ftp://bad"}]},
    )
    view = build_sanitized_responder_view_v2(inp)
    assert len(view.rejected_source_urls) == 1
    rs = view.rejected_source_urls[0]
    assert rs.occurrence_path == "training_by_job['job-001'][0].url"


def test_rejected_source_url_path_for_adjacent_role():
    inp = _v2(
        "describe_adjacent_role",
        adjacent_role_description_payload={
            "job": {"title": "X", "url": "ftp://bad", "posted_date": None},
            "evidence_summary": "", "matched_skills": [], "expired": False,
        },
    )
    view = build_sanitized_responder_view_v2(inp)
    assert len(view.rejected_source_urls) == 1
    assert view.rejected_source_urls[0].occurrence_path == \
        "adjacent_role_description_payload.job.url"


# =========================================================================
# Adjacent recommendation projection field-set check
# =========================================================================
def test_prompt_adjacent_recommendation_fields():
    field_names = {
        f.name for f in dataclasses.fields(PromptAdjacentRecommendationView)
    }
    assert field_names == {
        "job_id", "title", "employer", "location",
        "evidence_summary", "why_adjacent", "matched_skills",
    }


def test_fallback_adjacent_recommendation_fields():
    field_names = {
        f.name for f in dataclasses.fields(FallbackAdjacentRecommendationView)
    }
    assert field_names == {"title", "employer", "evidence_summary"}


def test_adjacent_recommendation_projection_silently_drops_url_field():
    """A hypothetical url field on the raw recommendation dict is
    silently dropped because the projection function reads only the
    seven allowlisted fields.
    """
    inp = _v2(
        "recommend_adjacent_roles",
        adjacent_recommendations_payload={
            "recommendations": [
                {
                    "title": "T", "employer": "E",
                    "evidence_summary": "3 of 5",
                    "matched_skills": ["a"],
                    "why_adjacent": "skill_evidence",
                    "job_id": "j", "location": "loc",
                    "url": "https://something.com",   # silently dropped
                    "future_field": "anything",       # silently dropped
                },
            ],
        },
    )
    view = build_sanitized_responder_view_v2(inp)
    container = view.prompt_adjacent_recommendations
    assert container is not None
    rec = container.recommendations[0]
    assert not hasattr(rec, "url")
    assert not hasattr(rec, "future_field")
    # No allowlist contribution
    assert view.prompt_urls == frozenset()


# =========================================================================
# V1 prompt-result projection field check
# =========================================================================
def test_prompt_result_view_includes_missing_skills():
    inp = _FakeInputV1(
        decision=_FakeDecision(show_matches=True),
        results=[{
            "title": "T", "employer": "E", "url": _good_url(),
            "missing_skills": ["docker"], "matched_skills": ["sql"],
            "job_id": "j",
        }],
    )
    view = build_sanitized_responder_view_v1(inp)
    assert view.prompt_results[0].missing_skills == ("docker",)
    assert view.prompt_results[0].matched_skills == ("sql",)


# =========================================================================
# Frozen view assertion
# =========================================================================
def test_sanitized_responder_view_is_frozen():
    view = build_sanitized_responder_view_v2(_v2("acknowledge_and_continue"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        view.prompt_results = ("x",)  # type: ignore[misc]


# =========================================================================
# describe_adjacent_role with non-string posted_date coerces to str
# =========================================================================
def test_adjacent_role_posted_date_coerces_non_string():
    import datetime
    d = datetime.date(2026, 6, 10)
    inp = _v2(
        "describe_adjacent_role",
        adjacent_role_description_payload={
            "job": {
                "title": "T", "employer": None, "location": None,
                "url": _good_url(), "posted_date": d,
            },
            "evidence_summary": "", "matched_skills": [], "expired": False,
        },
    )
    view = build_sanitized_responder_view_v2(inp)
    assert view.prompt_adjacent_role.posted_date == "2026-06-10"


# =========================================================================
# Regression tests for sub-step 3 review findings
# =========================================================================
# Finding 1: prompt container for adjacent recommendations carries
# total_retrieved alongside the recommendation tuple.
def test_recommend_adjacent_roles_container_carries_total_retrieved():
    inp = _v2(
        "recommend_adjacent_roles",
        adjacent_recommendations_payload={
            "recommendations": [
                {"title": "Welder", "employer": "ACME",
                 "evidence_summary": "3 of 5", "matched_skills": [],
                 "why_adjacent": "skill_evidence", "job_id": "j",
                 "location": "Sault Ste. Marie"},
            ],
            "total_retrieved": 5,
            # These are on the raw payload but NOT projected (not in
            # the dataclass-as-allowlist).
            "total_dropped_by_credential_gap": 1,
            "total_dropped_by_coverage_floor": 2,
        },
    )
    view = build_sanitized_responder_view_v2(inp)
    container = view.prompt_adjacent_recommendations
    assert container is not None
    assert container.total_retrieved == 5
    # total_dropped_by_* fields are silently dropped per the
    # dataclass-as-allowlist rule. The container has no such attribute.
    assert not hasattr(container, "total_dropped_by_credential_gap")
    assert not hasattr(container, "total_dropped_by_coverage_floor")


def test_recommend_adjacent_roles_container_total_retrieved_none_when_absent():
    """Absence preserved: total_retrieved is None when not on the raw
    payload, distinguishing absence from explicit 0.
    """
    inp = _v2(
        "recommend_adjacent_roles",
        adjacent_recommendations_payload={
            "recommendations": [
                {"title": "Welder", "employer": "ACME",
                 "evidence_summary": "3 of 5", "matched_skills": [],
                 "why_adjacent": "skill_evidence", "job_id": "j",
                 "location": "loc"},
            ],
            # total_retrieved omitted
        },
    )
    view = build_sanitized_responder_view_v2(inp)
    container = view.prompt_adjacent_recommendations
    assert container is not None
    assert container.total_retrieved is None


def test_recommend_adjacent_roles_container_total_retrieved_zero_preserved():
    inp = _v2(
        "recommend_adjacent_roles",
        adjacent_recommendations_payload={
            "recommendations": [],
            "total_retrieved": 0,
        },
    )
    view = build_sanitized_responder_view_v2(inp)
    container = view.prompt_adjacent_recommendations
    assert container is not None
    assert container.total_retrieved == 0


def test_recommend_adjacent_roles_container_total_retrieved_non_int_becomes_none():
    """Non-int (or bool) value for total_retrieved becomes None — no
    coercion. Sub-step 4 serializer handles the None case.
    """
    for bad in [True, False, "5", 1.5, None, [], {}]:
        inp = _v2(
            "recommend_adjacent_roles",
            adjacent_recommendations_payload={
                "recommendations": [],
                "total_retrieved": bad,
            },
        )
        view = build_sanitized_responder_view_v2(inp)
        container = view.prompt_adjacent_recommendations
        assert container is not None
        assert container.total_retrieved is None, (bad, container.total_retrieved)


def test_recommend_adjacent_roles_container_is_none_when_payload_absent():
    """No payload -> container is None (distinct from container with
    empty recommendations + total_retrieved=None).
    """
    inp = _v2("recommend_adjacent_roles", adjacent_recommendations_payload=None)
    view = build_sanitized_responder_view_v2(inp)
    assert view.prompt_adjacent_recommendations is None


def test_recommend_adjacent_roles_container_is_frozen():
    inp = _v2(
        "recommend_adjacent_roles",
        adjacent_recommendations_payload={
            "recommendations": [],
            "total_retrieved": 3,
        },
    )
    view = build_sanitized_responder_view_v2(inp)
    container = view.prompt_adjacent_recommendations
    assert container is not None
    with pytest.raises(dataclasses.FrozenInstanceError):
        container.total_retrieved = 99  # type: ignore[misc]


def test_recommend_adjacent_roles_other_moves_have_no_container():
    """Container slot is None for moves other than
    recommend_adjacent_roles.
    """
    for move in [
        "present_matches", "explain_gap", "present_near_miss",
        "explain_remaining_gaps", "describe_adjacent_role",
        "acknowledge_and_continue",
    ]:
        inp = _v2(move)
        view = build_sanitized_responder_view_v2(inp)
        assert view.prompt_adjacent_recommendations is None, move


# Finding 2: adjacent-role views are constructed whenever the payload
# exists, even when `job` is absent. Payload-level fields are preserved.
def test_describe_adjacent_role_expired_with_no_job_preserves_payload_fields():
    """expired=True with job missing: prompt + fallback views must
    still carry expired=True, evidence_summary, matched_skills.
    Job-derived fields are None and has_validated_url=False.
    """
    inp = _v2(
        "describe_adjacent_role",
        adjacent_role_description_payload={
            "job": None,
            "evidence_summary": "3 of 5 matched",
            "matched_skills": ["welding", "safety"],
            "expired": True,
        },
    )
    view = build_sanitized_responder_view_v2(inp)
    # Prompt view exists with payload fields preserved
    pr = view.prompt_adjacent_role
    assert pr is not None
    assert pr.expired is True
    assert pr.evidence_summary == "3 of 5 matched"
    assert pr.matched_skills == ("welding", "safety")
    # Job-derived fields are None
    assert pr.job_id is None
    assert pr.title is None
    assert pr.employer is None
    assert pr.location is None
    assert pr.posted_date is None
    assert pr.url is None
    # Fallback view exists with same preservation
    fb = view.fallback_adjacent_role
    assert fb is not None
    assert fb.expired is True
    assert fb.evidence_summary == "3 of 5 matched"
    assert fb.matched_skills == ("welding", "safety")
    assert fb.job_id is None
    assert fb.title is None
    assert fb.has_validated_url is False
    # No URL validated -> no rejected entries
    assert view.rejected_source_urls == ()
    assert view.prompt_urls == frozenset()


def test_describe_adjacent_role_job_not_dict_preserves_payload_fields():
    """job is some non-dict value (e.g. string): same behavior as
    job=None — views constructed, payload-level fields preserved.
    """
    inp = _v2(
        "describe_adjacent_role",
        adjacent_role_description_payload={
            "job": "garbage",   # not a dict
            "evidence_summary": "ev",
            "matched_skills": ["a"],
            "expired": False,
        },
    )
    view = build_sanitized_responder_view_v2(inp)
    pr = view.prompt_adjacent_role
    assert pr is not None
    assert pr.title is None
    assert pr.url is None
    assert pr.expired is False
    assert pr.evidence_summary == "ev"
    assert pr.matched_skills == ("a",)
    fb = view.fallback_adjacent_role
    assert fb is not None
    assert fb.title is None
    assert fb.has_validated_url is False
    assert fb.expired is False
    assert fb.evidence_summary == "ev"


def test_describe_adjacent_role_empty_payload_dict_prompt_none_fallback_present():
    """Empty dict {} is falsy in Python, so responder.py:1451's
    `... and inp.adjacent_role_description_payload` skips the prompt
    block. The view's prompt_adjacent_role MUST be None to match.

    The fallback view still exists because the deterministic fallback
    at responder.py:1786 normalizes the payload with `or {}` and
    branches on the resulting empty dict.
    """
    inp = _v2(
        "describe_adjacent_role",
        adjacent_role_description_payload={},
    )
    view = build_sanitized_responder_view_v2(inp)
    assert view.prompt_adjacent_role is None
    assert view.fallback_adjacent_role is not None
    assert view.fallback_adjacent_role.has_validated_url is False
    assert view.fallback_adjacent_role.expired is False
    assert view.fallback_adjacent_role.title is None
    assert view.prompt_urls == frozenset()


def test_describe_adjacent_role_payload_none_returns_empty_view():
    """When the payload itself is None, the role projections are None
    (no payload to project from).
    """
    inp = _v2(
        "describe_adjacent_role",
        adjacent_role_description_payload=None,
    )
    view = build_sanitized_responder_view_v2(inp)
    assert view.prompt_adjacent_role is None
    assert view.fallback_adjacent_role is None


# =========================================================================
# job_is_mapping flag (sub-step 4 amendment) — distinguishes `{}` from
# missing/None/non-dict for the serializer's "job: null" vs "job: {...}"
# decision.
# =========================================================================
def test_job_is_mapping_true_for_dict_job():
    inp = _v2(
        "describe_adjacent_role",
        adjacent_role_description_payload={
            "job": {"title": "T", "url": _good_url(), "posted_date": None},
            "evidence_summary": "", "matched_skills": [], "expired": False,
        },
    )
    view = build_sanitized_responder_view_v2(inp)
    assert view.prompt_adjacent_role.job_is_mapping is True


def test_job_is_mapping_true_for_empty_dict():
    """An empty {} job IS a mapping. The serializer will emit
    `job: {6 null fields}` for this case, matching the locked
    sub-step 4 expansion contract.
    """
    inp = _v2(
        "describe_adjacent_role",
        adjacent_role_description_payload={
            "job": {}, "evidence_summary": "", "matched_skills": [],
            "expired": False,
        },
    )
    view = build_sanitized_responder_view_v2(inp)
    assert view.prompt_adjacent_role.job_is_mapping is True


def test_job_is_mapping_false_for_none():
    inp = _v2(
        "describe_adjacent_role",
        adjacent_role_description_payload={
            "job": None, "evidence_summary": "", "matched_skills": [],
            "expired": False,
        },
    )
    view = build_sanitized_responder_view_v2(inp)
    assert view.prompt_adjacent_role.job_is_mapping is False


def test_job_is_mapping_false_for_missing_job_key():
    inp = _v2(
        "describe_adjacent_role",
        adjacent_role_description_payload={
            "evidence_summary": "", "matched_skills": [], "expired": False,
            # no "job" key
        },
    )
    view = build_sanitized_responder_view_v2(inp)
    assert view.prompt_adjacent_role.job_is_mapping is False


def test_job_is_mapping_false_for_non_dict_value():
    for bad in ["garbage", 42, [], True]:
        inp = _v2(
            "describe_adjacent_role",
            adjacent_role_description_payload={
                "job": bad, "evidence_summary": "", "matched_skills": [],
                "expired": False,
            },
        )
        view = build_sanitized_responder_view_v2(inp)
        assert view.prompt_adjacent_role.job_is_mapping is False, bad


# Truthy-gate parity: empty {} payload behavior
def test_recommend_adjacent_roles_empty_payload_dict_no_prompt_container():
    """Empty dict {} is falsy in Python, so responder.py:1439's
    `... and inp.adjacent_recommendations_payload` skips the prompt
    block. The view's prompt_adjacent_recommendations MUST be None.
    """
    inp = _v2(
        "recommend_adjacent_roles",
        adjacent_recommendations_payload={},
    )
    view = build_sanitized_responder_view_v2(inp)
    assert view.prompt_adjacent_recommendations is None
    # The fallback tuple is still empty (no recommendations to project).
    assert view.fallback_adjacent_recommendations == ()
    assert view.prompt_urls == frozenset()
    assert view.fallback_urls == frozenset()


def test_describe_adjacent_role_non_empty_payload_no_job_still_prompts():
    """A non-empty payload with `job` missing still produces a prompt
    view — responder.py:1451's truthy check passes, and the prompt
    emits `job: null` alongside the other payload fields.
    """
    inp = _v2(
        "describe_adjacent_role",
        adjacent_role_description_payload={
            "evidence_summary": "ev", "matched_skills": ["a"], "expired": False,
        },
    )
    view = build_sanitized_responder_view_v2(inp)
    assert view.prompt_adjacent_role is not None
    assert view.prompt_adjacent_role.title is None
    assert view.prompt_adjacent_role.evidence_summary == "ev"
    assert view.fallback_adjacent_role is not None
