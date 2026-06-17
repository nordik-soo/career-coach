"""AR-4 tests: dedicated adjacency scorer + rank_adjacent + drop_excluded.

Covers (per docs/adjacent-recommendations-design.md v11 / v12):
  - `_score_one_adjacent_job`:
      * formula: 0.8 * req_mean + 0.2 * pref_mean + recency_boost, clamped
      * credentials EXCLUDED from required/preferred buckets
      * NO target-title boost present
      * NO target-NOC boost present
      * recency contributes a real (small, bounded) amount
      * defensive: malformed inputs do not crash
  - `rank_adjacent`:
      * sort by score DESC, tie-break by job_id ASC
      * carries `__adjacent_score__` on the result dict
      * doesn't mutate the caller's dict
  - `drop_excluded`:
      * strips matching job_ids
      * tolerates None / wrong-type `presented_job_ids`
      * tolerates non-str entries in `presented_job_ids`

All AR-4 functions ship as dead code; no production caller dispatches
into them until AR-6.
"""
from __future__ import annotations

import datetime as _dt
import inspect

import pytest

pytestmark = pytest.mark.nodb


from skillbridge.match.adjacent import (
    _score_one_adjacent_job,
    build_user_skill_sets,
    drop_excluded,
    rank_adjacent,
)
from skillbridge.session.staging import StagedProfile, StagedSkill


def _three_concrete_skills() -> list[StagedSkill]:
    return [
        StagedSkill(skill_name="welding", source="resume", confidence=0.8),
        StagedSkill(skill_name="blueprint reading", source="resume", confidence=0.8),
        StagedSkill(skill_name="forklift operation", source="resume", confidence=0.8),
    ]


def _user_sets():
    return build_user_skill_sets(_three_concrete_skills())


def _required(name: str, **kw) -> dict:
    return {
        "skill_name": name,
        "skill_id": kw.get("skill_id"),
        "confidence": kw.get("confidence", 0.9),
        "importance_rank": kw.get("importance_rank", 1),
        "skill_type": "required",
    }


def _preferred(name: str, **kw) -> dict:
    return {
        "skill_name": name,
        "skill_id": kw.get("skill_id"),
        "confidence": kw.get("confidence", 0.7),
        "importance_rank": kw.get("importance_rank", 3),
        "skill_type": "preferred",
    }


def _job(*, job_id: str = "j", posted_date=None, skills=None) -> dict:
    return {
        "job_id": job_id,
        "title": "Welder",
        "noc_code": "72107",
        "posted_date": posted_date,
        "skills": skills or [],
    }


# =========================================================================
# _score_one_adjacent_job -- formula
# =========================================================================
def test_score_all_required_match_no_preferred() -> None:
    """3 required skills all match at 1.0 → req_mean = 1.0; no preferred;
    no recency → score = 0.8 * 1.0 = 0.8."""
    ids, names, canon = _user_sets()
    job = _job(skills=[
        _required("welding"),
        _required("blueprint reading"),
        _required("forklift operation"),
    ])
    score = _score_one_adjacent_job(job, ids, names, canon)
    assert score == pytest.approx(0.8, abs=1e-9)


def test_score_required_plus_preferred() -> None:
    """req_mean=1.0, pref_mean=1.0, no recency → 0.8 + 0.2 = 1.0."""
    ids, names, canon = _user_sets()
    job = _job(skills=[
        _required("welding"),
        _preferred("blueprint reading"),
    ])
    score = _score_one_adjacent_job(job, ids, names, canon)
    assert score == pytest.approx(1.0, abs=1e-9)


def test_score_no_match_floor() -> None:
    """Required exists but doesn't match → req_mean = 0.0; no preferred;
    no recency → score = 0.0 (floor)."""
    ids, names, canon = _user_sets()
    job = _job(skills=[
        _required("food safety"),
        _required("till operation"),
    ])
    score = _score_one_adjacent_job(job, ids, names, canon)
    assert score == pytest.approx(0.0, abs=1e-9)


def test_score_clamped_to_one() -> None:
    """A wildly-fresh posted_date + full-match required/preferred
    would push past 1.0 if unclamped. The scorer clamps."""
    ids, names, canon = _user_sets()
    fresh = _dt.date.today()
    job = _job(posted_date=fresh, skills=[
        _required("welding"),
        _preferred("blueprint reading"),
    ])
    score = _score_one_adjacent_job(job, ids, names, canon)
    assert score <= 1.0
    # Fresh recency contributes a positive boost above the no-recency
    # case, so this score sits between 0.8+0 and 0.8+0.05+0.2=1.05
    # (clamped to 1.0).
    assert score >= 0.8


def test_score_recency_contributes() -> None:
    """A fresh job scores higher than the same job with no posted_date."""
    ids, names, canon = _user_sets()
    fresh = _dt.date.today()
    job_fresh = _job(posted_date=fresh, skills=[_required("welding")])
    job_no_date = _job(posted_date=None, skills=[_required("welding")])
    assert (
        _score_one_adjacent_job(job_fresh, ids, names, canon)
        > _score_one_adjacent_job(job_no_date, ids, names, canon)
    )


def test_score_old_posting_gets_no_recency_boost() -> None:
    """A posting older than the recency window contributes 0.0."""
    ids, names, canon = _user_sets()
    old = _dt.date.today() - _dt.timedelta(days=365 * 5)
    job = _job(posted_date=old, skills=[_required("welding")])
    # Required-mean = 1.0, no preferred, no recency → 0.8.
    assert _score_one_adjacent_job(job, ids, names, canon) == pytest.approx(0.8, abs=1e-9)


# =========================================================================
# _score_one_adjacent_job -- credentials excluded
# =========================================================================
def test_score_excludes_required_credentials_from_required_bucket() -> None:
    """A required Class G credential MUST NOT contribute to req_mean.
    Otherwise a credential the user happens to have would inflate the
    score with eligibility certifications rather than transferable
    evidence."""
    user_skills = [
        StagedSkill(skill_name="welding", source="resume", confidence=0.8),
        StagedSkill(skill_name="blueprint reading", source="resume", confidence=0.8),
        StagedSkill(skill_name="forklift operation", source="resume", confidence=0.8),
        StagedSkill(skill_name="Class G License", source="resume", confidence=0.9),
    ]
    ids, names, canon = build_user_skill_sets(user_skills)
    job = _job(skills=[
        _required("welding"),                  # match → 1.0
        _required("Class G License"),          # credential → SKIPPED
    ])
    score = _score_one_adjacent_job(job, ids, names, canon)
    # If the credential were counted, req_mean = (1.0 + 1.0) / 2 = 1.0.
    # Excluded → req_mean = 1.0/1 = 1.0 too. Same value, but the
    # exclusion matters when the unmatched bucket has more items:
    job2 = _job(skills=[
        _required("welding"),                  # match → 1.0
        _required("food safety"),              # no match → 0.0
        _required("Class G License"),          # credential → SKIPPED
    ])
    score2 = _score_one_adjacent_job(job2, ids, names, canon)
    # If credential counted: (1.0 + 0.0 + 1.0) / 3 = 0.667
    # Excluded: (1.0 + 0.0) / 2 = 0.5
    # Expected: 0.8 * 0.5 = 0.4
    assert score2 == pytest.approx(0.4, abs=1e-9)


def test_score_excludes_preferred_credentials_from_preferred_bucket() -> None:
    """A preferred credential must also be excluded -- preferred-cred
    matches would silently inflate the preferred bucket."""
    user_skills = [
        StagedSkill(skill_name="welding", source="resume", confidence=0.8),
        StagedSkill(skill_name="blueprint reading", source="resume", confidence=0.8),
        StagedSkill(skill_name="forklift operation", source="resume", confidence=0.8),
        StagedSkill(skill_name="WHMIS 2015 Certificate", source="resume", confidence=0.9),
    ]
    ids, names, canon = build_user_skill_sets(user_skills)
    job = _job(skills=[
        _required("welding"),                       # → 1.0
        _preferred("WHMIS 2015 Certificate"),       # credential → SKIPPED
    ])
    score = _score_one_adjacent_job(job, ids, names, canon)
    # If counted: 0.8 * 1.0 + 0.2 * 1.0 = 1.0
    # Excluded: pref bucket empty → 0.8 * 1.0 + 0.2 * 0.0 = 0.8
    assert score == pytest.approx(0.8, abs=1e-9)


# =========================================================================
# _score_one_adjacent_job -- audit: no target boosts
# =========================================================================
def test_score_function_does_not_reference_target_title_or_noc_boost() -> None:
    """Static audit: the source of `_score_one_adjacent_job` MUST NOT
    reference target-title or target-NOC boost machinery. The whole
    point of adjacency ranking is to ignore "how close is this to the
    user's stated target" -- if the function ever started consulting
    those signals, this test trips."""
    src = inspect.getsource(_score_one_adjacent_job)
    forbidden_tokens = [
        "_target_role_boost",
        "_target_title_boost",
        "_target_noc_boost",
        "target_role_text",
        "target_noc",
    ]
    for tok in forbidden_tokens:
        assert tok not in src, (
            f"_score_one_adjacent_job references {tok!r}. Adjacency "
            f"ranking MUST NOT carry target-role bias."
        )


# =========================================================================
# _score_one_adjacent_job -- defensive boundaries
# =========================================================================
def test_score_tolerates_non_dict_job() -> None:
    """Malformed input → 0.0, no crash."""
    ids, names, canon = _user_sets()
    assert _score_one_adjacent_job(None, ids, names, canon) == 0.0  # type: ignore[arg-type]
    assert _score_one_adjacent_job("garbage", ids, names, canon) == 0.0  # type: ignore[arg-type]


def test_score_tolerates_non_list_skills_field() -> None:
    ids, names, canon = _user_sets()
    job = _job(skills=[])
    job["skills"] = "not a list"   # type: ignore[assignment]
    assert _score_one_adjacent_job(job, ids, names, canon) == 0.0


def test_score_tolerates_non_string_skill_name() -> None:
    """Malformed skill_name → row is skipped, score reflects only the
    valid rows."""
    ids, names, canon = _user_sets()
    job = _job(skills=[
        {"skill_name": 7, "skill_type": "required"},
        _required("welding"),
    ])
    score = _score_one_adjacent_job(job, ids, names, canon)
    # The valid required match → req_mean = 1.0 → score = 0.8.
    assert score == pytest.approx(0.8, abs=1e-9)


# --- AR-4 round-2: malformed skill_type ---
@pytest.mark.parametrize("bad_skill_type", [7, True, {}, []])
def test_score_tolerates_non_string_skill_type(bad_skill_type) -> None:
    """A forged skill row carrying a non-str skill_type would crash
    `_required_or_preferred` on `.strip()`. The scorer must coerce
    or skip without raising; legacy NULL semantics ("required" on
    missing/unknown) apply to malformed values."""
    ids, names, canon = _user_sets()
    job = _job(skills=[
        # Valid required match → 1.0
        _required("welding"),
        # Malformed: skill_type is the wrong type. Treated as
        # legacy-unknown ("required" bucket); skill_name doesn't
        # match the user so it's a 0.0 contribution.
        {
            "skill_name": "food safety",
            "skill_type": bad_skill_type,
        },
    ])
    score = _score_one_adjacent_job(job, ids, names, canon)
    # req_mean = (1.0 + 0.0) / 2 = 0.5; no preferred; no recency
    # → 0.8 * 0.5 = 0.4. Confirms the malformed row landed in the
    # required bucket (legacy NULL semantics) without crashing.
    assert score == pytest.approx(0.4, abs=1e-9)


# --- AR-4 round-2: malformed posted_date ---
@pytest.mark.parametrize("bad_posted", [
    "2024-01-01",                          # str
    0,                                      # int
    True,                                   # bool
    {},                                     # dict
    [],                                     # list
    __import__("datetime").datetime.now(),  # datetime (not date)
])
def test_score_tolerates_non_date_posted_date(bad_posted) -> None:
    """`_recency_boost` does `(date.today() - posted).days`. Any non-
    `date` (including `datetime`) causes TypeError. The scorer
    isolates that path and treats the value as missing-recency."""
    ids, names, canon = _user_sets()
    job = _job(posted_date=bad_posted, skills=[_required("welding")])
    # Must NOT raise. Score reflects the required match with NO
    # recency boost → 0.8 * 1.0 + 0.0 = 0.8.
    score = _score_one_adjacent_job(job, ids, names, canon)
    assert score == pytest.approx(0.8, abs=1e-9)


def test_score_empty_job_skills_returns_zero() -> None:
    ids, names, canon = _user_sets()
    job = _job(skills=[])
    assert _score_one_adjacent_job(job, ids, names, canon) == 0.0


# =========================================================================
# rank_adjacent
# =========================================================================
def test_rank_orders_by_score_desc() -> None:
    ids, names, canon = _user_sets()
    high = _job(job_id="high", skills=[
        _required("welding"),
        _preferred("blueprint reading"),
    ])
    mid = _job(job_id="mid", skills=[
        _required("welding"),
    ])
    low = _job(job_id="low", skills=[_required("food safety")])
    ranked = rank_adjacent([low, mid, high], ids, names, canon)
    assert [j["job_id"] for j in ranked] == ["high", "mid", "low"]


def test_rank_breaks_ties_by_job_id_asc() -> None:
    """Two jobs with identical structure (same score) sort by job_id."""
    ids, names, canon = _user_sets()
    a = _job(job_id="zzz", skills=[_required("welding")])
    b = _job(job_id="aaa", skills=[_required("welding")])
    ranked = rank_adjacent([a, b], ids, names, canon)
    assert [j["job_id"] for j in ranked] == ["aaa", "zzz"]


def test_rank_carries_score_on_output() -> None:
    ids, names, canon = _user_sets()
    job = _job(skills=[_required("welding")])
    ranked = rank_adjacent([job], ids, names, canon)
    assert "__adjacent_score__" in ranked[0]
    assert ranked[0]["__adjacent_score__"] == pytest.approx(0.8, abs=1e-9)


def test_rank_does_not_mutate_caller_dict() -> None:
    ids, names, canon = _user_sets()
    job = _job(skills=[_required("welding")])
    _ = rank_adjacent([job], ids, names, canon)
    assert "__adjacent_score__" not in job, (
        "rank_adjacent must return new dict copies, not mutate "
        "the caller's input."
    )


def test_rank_skips_non_dict_entries() -> None:
    ids, names, canon = _user_sets()
    ranked = rank_adjacent(
        [None, "garbage", _job(skills=[_required("welding")])],
        ids, names, canon,
    )
    assert [j["job_id"] for j in ranked] == ["j"]


def test_rank_empty_input_returns_empty() -> None:
    ids, names, canon = _user_sets()
    assert rank_adjacent([], ids, names, canon) == []


# =========================================================================
# drop_excluded
# =========================================================================
def test_drop_excluded_strips_matching_job_ids() -> None:
    ranked = [
        {"job_id": "a"}, {"job_id": "b"}, {"job_id": "c"},
    ]
    out = drop_excluded(ranked, presented_job_ids=("b",))
    assert [j["job_id"] for j in out] == ["a", "c"]


def test_drop_excluded_strips_multiple_matches() -> None:
    ranked = [
        {"job_id": "a"}, {"job_id": "b"}, {"job_id": "c"}, {"job_id": "d"},
    ]
    out = drop_excluded(ranked, presented_job_ids=("a", "c"))
    assert [j["job_id"] for j in out] == ["b", "d"]


def test_drop_excluded_no_match_returns_all() -> None:
    ranked = [{"job_id": "a"}, {"job_id": "b"}]
    out = drop_excluded(ranked, presented_job_ids=("z",))
    assert [j["job_id"] for j in out] == ["a", "b"]


def test_drop_excluded_empty_list_returns_all() -> None:
    ranked = [{"job_id": "a"}]
    out = drop_excluded(ranked, presented_job_ids=())
    assert [j["job_id"] for j in out] == ["a"]


def test_drop_excluded_none_returns_all() -> None:
    ranked = [{"job_id": "a"}]
    out = drop_excluded(ranked, presented_job_ids=None)
    assert [j["job_id"] for j in out] == ["a"]


def test_drop_excluded_wrong_type_returns_all() -> None:
    """A forged-cookie presented_job_ids of the wrong type doesn't
    crash -- it's treated as "no exclusions"."""
    ranked = [{"job_id": "a"}]
    out = drop_excluded(ranked, presented_job_ids="not a list")
    assert [j["job_id"] for j in out] == ["a"]


def test_drop_excluded_tolerates_non_str_entries() -> None:
    """Non-str entries in the exclusion list are skipped without
    crashing -- the sanitizer should already drop them, but we
    defense-in-depth here."""
    ranked = [{"job_id": "a"}, {"job_id": "b"}]
    out = drop_excluded(ranked, presented_job_ids=["a", 7, None, ""])
    assert [j["job_id"] for j in out] == ["b"]


def test_drop_excluded_tolerates_non_str_job_id() -> None:
    """A ranked entry whose job_id isn't a str (forged in-memory) is
    DROPPED entirely -- it can't be safely compared, and the cookie
    boundary should never let it through."""
    ranked = [{"job_id": 7}, {"job_id": "a"}, {"job_id": None}]
    out = drop_excluded(ranked, presented_job_ids=("z",))
    assert [j["job_id"] for j in out] == ["a"]


# --- AR-4 round-2: defensive filter runs even with empty/None exclusions ---
def test_drop_excluded_with_none_still_sanitizes_malformed_jobs() -> None:
    """The defensive contract: even when no exclusions are supplied,
    malformed ranked entries (non-dict, non-str job_id, empty job_id)
    are stripped from the output. Otherwise an un-comparable
    recommendation could surface to the user."""
    ranked = [
        None,                          # type: ignore[list-item]
        "garbage",                     # type: ignore[list-item]
        {"job_id": 7},                 # non-str id
        {"job_id": None},
        {"job_id": ""},                # empty id
        {"job_id": "a"},               # the only valid entry
    ]
    out = drop_excluded(ranked, presented_job_ids=None)
    assert [j["job_id"] for j in out] == ["a"]


def test_drop_excluded_with_empty_list_still_sanitizes_malformed_jobs() -> None:
    ranked = [
        {"job_id": 7},
        {"job_id": "a"},
        {"job_id": None},
    ]
    out = drop_excluded(ranked, presented_job_ids=[])
    assert [j["job_id"] for j in out] == ["a"]


def test_drop_excluded_with_wrong_type_still_sanitizes_malformed_jobs() -> None:
    """`presented_job_ids="not a list"` (wrong type) shouldn't bypass
    the malformed-entry filter."""
    ranked = [
        {"job_id": 7},
        {"job_id": "a"},
    ]
    out = drop_excluded(ranked, presented_job_ids="not a list")
    assert [j["job_id"] for j in out] == ["a"]


# =========================================================================
# Integration: rank → drop_excluded chained
# =========================================================================
def test_rank_then_drop_pipeline() -> None:
    """End-to-end: a small fixture covering the AR-6 dispatch pipeline
    (rank → drop_excluded → top-3 cap is the AR-6 caller's job, not
    AR-4's)."""
    ids, names, canon = _user_sets()
    accepted = [
        _job(job_id="j-best", skills=[
            _required("welding"),
            _preferred("blueprint reading"),
        ]),
        _job(job_id="j-mid", skills=[_required("welding")]),
        _job(job_id="j-low", skills=[_required("food safety")]),
    ]
    ranked = rank_adjacent(accepted, ids, names, canon)
    assert [j["job_id"] for j in ranked] == ["j-best", "j-mid", "j-low"]
    filtered = drop_excluded(ranked, presented_job_ids=("j-mid",))
    assert [j["job_id"] for j in filtered] == ["j-best", "j-low"]
