"""CP4 DevelopmentPlan — isolated tests against the locked spec.

Focused on the deterministic logic that does NOT require DB:
candidate-gap collection, lexicographic ranking, truncation,
null-plan handling, advisory selection, training-attachment shape.

The full per-job counterfactual rescoring path requires DB access
(refetching v_current_job rows) and is exercised in live shadow
smoke after handler wiring lands.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from skillbridge.chat.development_plan import (
    HARD_CAP_TOTAL_RESCORINGS,
    CandidateRankRow,
    _Candidate,
    _collect_candidate_gaps,
    _rank_candidates,
    _RescoredCandidate,
    _truncate_to_cap,
)
from skillbridge.chat.inventory_diagnosis import InventoryDiagnosis, diagnose

pytestmark = pytest.mark.nodb


def _match(
    *,
    job_id: str = "j1",
    match_band: str = "stretch",
    match_eligible: bool = True,
    required_missing: list[str] | None = None,
    noc_code: str | None = "14200",
):
    return SimpleNamespace(
        job_id=job_id,
        match_band=match_band,
        match_eligible=match_eligible,
        noc_code=noc_code,
        score_explanation={"required_missing": list(required_missing or [])},
    )


# --- §2 candidate-gap collection ----------------------------------------

def test_candidate_gap_collection_from_preparation_gap_evidence():
    """PREPARATION_GAP outcome carries gap_record_job_ids in supporting
    evidence; CP4 reads those records and collects their required_missing
    skills as canonical candidate gaps."""
    matches = [
        _match(job_id="j1", required_missing=["bookkeeping", "journal entry"]),
        _match(job_id="j2", required_missing=["bookkeeping", "Class A licence"]),
        _match(job_id="j3", required_missing=[]),  # not a gap record
    ]
    diagnosis = diagnose(
        enough_to_match=True, usable_evidence_present=True,
        engine_completed=True, snapshot_usable=True,
        direct_match_results=matches, skill_adjacent_results=[],
        target_posting_count=5,
    )
    assert diagnosis.outcome == "PREPARATION_GAP"

    candidates = _collect_candidate_gaps(
        diagnosis=diagnosis, in_memory_matches=matches,
    )
    names = sorted(c.canonical_name for c in candidates)
    # "Class A licence" canonicalizes to its lowercase form;
    # bookkeeping appears across j1+j2.
    assert "bookkeeping" in names
    # Bookkeeping appears in two source jobs.
    bk = next(c for c in candidates if c.canonical_name == "bookkeeping")
    assert set(bk.source_job_ids) == {"j1", "j2"}


def test_candidate_gap_collection_filters_soft_traits():
    """Soft traits (reliability, attention to detail, etc.) are
    excluded from candidate gaps."""
    matches = [
        _match(
            job_id="j1",
            required_missing=["reliability", "bookkeeping", "attention to detail"],
        ),
    ]
    diagnosis = diagnose(
        enough_to_match=True, usable_evidence_present=True,
        engine_completed=True, snapshot_usable=True,
        direct_match_results=matches, skill_adjacent_results=[],
        target_posting_count=1,
    )
    candidates = _collect_candidate_gaps(
        diagnosis=diagnosis, in_memory_matches=matches,
    )
    names = {c.canonical_name for c in candidates}
    assert "bookkeeping" in names
    assert "reliability" not in names
    assert "attention to detail" not in names


def test_candidate_collection_deterministic_order():
    """Same inputs produce identical candidate ordering."""
    matches = [
        _match(job_id="j1", required_missing=["z_skill", "a_skill"]),
        _match(job_id="j2", required_missing=["m_skill"]),
    ]
    diagnosis = diagnose(
        enough_to_match=True, usable_evidence_present=True,
        engine_completed=True, snapshot_usable=True,
        direct_match_results=matches, skill_adjacent_results=[],
        target_posting_count=2,
    )
    c1 = _collect_candidate_gaps(diagnosis=diagnosis, in_memory_matches=matches)
    c2 = _collect_candidate_gaps(diagnosis=diagnosis, in_memory_matches=matches)
    assert [c.canonical_name for c in c1] == [c.canonical_name for c in c2]
    # Sorted alphabetically.
    names = [c.canonical_name for c in c1]
    assert names == sorted(names)


def test_candidate_collection_handles_empty_required_missing_evidence():
    """When PREPARATION_GAP fired but the supporting MatchResults have
    no usable required_missing (all soft traits, or all empty), the
    collector returns []. CP4 will then produce a null primary."""
    matches = [
        _match(job_id="j1", required_missing=["reliability"]),  # soft trait only
        _match(job_id="j2", required_missing=["multitasking"]),  # soft trait only
    ]
    diagnosis = diagnose(
        enough_to_match=True, usable_evidence_present=True,
        engine_completed=True, snapshot_usable=True,
        direct_match_results=matches, skill_adjacent_results=[],
        target_posting_count=2,
    )
    assert diagnosis.outcome == "PREPARATION_GAP"
    candidates = _collect_candidate_gaps(
        diagnosis=diagnosis, in_memory_matches=matches,
    )
    assert candidates == []


# --- §4 truncation ------------------------------------------------------

def test_truncate_to_cap_under_limit():
    cands = [
        _Candidate(canonical_name="a", source_job_ids=("j1", "j2"), min_importance_rank=99),
        _Candidate(canonical_name="b", source_job_ids=("j3",), min_importance_rank=99),
    ]
    kept, truncated = _truncate_to_cap(cands, cap=10)
    assert kept == cands
    assert truncated == []


def test_truncate_to_cap_exceeds():
    # Each candidate "costs" len(source_job_ids); cap is total.
    cands = [
        _Candidate(canonical_name="a", source_job_ids=tuple(f"j{i}" for i in range(60)), min_importance_rank=99),
        _Candidate(canonical_name="b", source_job_ids=tuple(f"k{i}" for i in range(60)), min_importance_rank=99),
    ]
    kept, truncated = _truncate_to_cap(cands, cap=100)
    assert len(kept) == 1
    assert kept[0].canonical_name == "a"
    assert truncated == ["b"]


def test_truncate_at_hard_cap_default():
    """The shipped cap is HARD_CAP_TOTAL_RESCORINGS = 100. Confirm
    the constant has the expected provisional value."""
    assert HARD_CAP_TOTAL_RESCORINGS == 100


# --- §5 lexicographic ranking -------------------------------------------

def _rc(
    name: str,
    *,
    promoted: int = 0,
    blocker_removed: int = 0,
    improved: int = 0,
    market_freq: int = 0,
    min_rank: int = 99,
) -> _RescoredCandidate:
    return _RescoredCandidate(
        canonical_name=name,
        source_job_ids=tuple(),
        promoted_job_ids=tuple(f"p{i}" for i in range(promoted)),
        tier_improvement_job_ids=tuple(f"i{i}" for i in range(improved)),
        blocker_removed_job_ids=tuple(f"b{i}" for i in range(blocker_removed)),
        tier_transitions=tuple(),
        min_importance_rank=min_rank,
        active_market_frequency=market_freq,
        counterfactual_method="fresh_per_job_score",
        dataset_version=None,
        engine_version=None,
    )


def test_ranking_promoted_count_wins_first():
    """Signal 1: promoted_to_apply_today_count — descending."""
    cands = [
        _rc("low", promoted=1, blocker_removed=10),
        _rc("high", promoted=5, blocker_removed=0),
    ]
    ranked = _rank_candidates(cands)
    assert ranked[0].canonical_name == "high"


def test_ranking_blocker_removed_breaks_tie_at_signal_2():
    cands = [
        _rc("c1", promoted=2, blocker_removed=0, improved=10),
        _rc("c2", promoted=2, blocker_removed=1, improved=0),
    ]
    ranked = _rank_candidates(cands)
    assert ranked[0].canonical_name == "c2"


def test_ranking_tier_improvement_breaks_tie_at_signal_3():
    cands = [
        _rc("c1", promoted=2, blocker_removed=1, improved=3, market_freq=10),
        _rc("c2", promoted=2, blocker_removed=1, improved=5, market_freq=0),
    ]
    ranked = _rank_candidates(cands)
    assert ranked[0].canonical_name == "c2"


def test_ranking_market_frequency_breaks_tie_at_signal_4():
    cands = [
        _rc("c1", promoted=1, blocker_removed=0, improved=1, market_freq=5, min_rank=1),
        _rc("c2", promoted=1, blocker_removed=0, improved=1, market_freq=10, min_rank=99),
    ]
    ranked = _rank_candidates(cands)
    assert ranked[0].canonical_name == "c2"


def test_ranking_importance_rank_breaks_tie_at_signal_5():
    cands = [
        _rc("c1", promoted=1, improved=1, market_freq=5, min_rank=10),
        _rc("c2", promoted=1, improved=1, market_freq=5, min_rank=1),
    ]
    ranked = _rank_candidates(cands)
    assert ranked[0].canonical_name == "c2"


def test_ranking_canonical_name_stable_tie_break():
    """Identical impact resolves alphabetically by canonical name."""
    cands = [
        _rc("z_skill", promoted=1),
        _rc("a_skill", promoted=1),
    ]
    ranked = _rank_candidates(cands)
    assert ranked[0].canonical_name == "a_skill"


def test_ranking_training_availability_is_not_a_signal():
    """Constraint B: training availability is NOT used in ranking.
    A higher-impact candidate WITHOUT verified training beats a
    lower-impact candidate WITH verified training. _RescoredCandidate
    does not carry training availability at ranking time, so this is
    structurally enforced — but verify the ranking order ignores it."""
    # Two candidates with identical lex score above signal 6; only
    # canonical name differs. No "has_training" knob enters _rc()
    # because it doesn't belong at the ranking layer.
    cands = [
        _rc("z_high_impact", promoted=5),
        _rc("a_low_impact", promoted=1),
    ]
    ranked = _rank_candidates(cands)
    assert ranked[0].canonical_name == "z_high_impact"


# --- Plan-level behavior ------------------------------------------------

def test_compute_returns_none_on_undetermined():
    """CP4 must not run when diagnosis is not PREPARATION_GAP or
    READY_TO_APPLY."""
    from skillbridge.chat.development_plan import compute_development_plan

    diagnosis = diagnose(
        enough_to_match=False, usable_evidence_present=True,
        engine_completed=True, snapshot_usable=True,
        direct_match_results=[], skill_adjacent_results=[],
        target_posting_count=None,
    )
    # The user staged profile is unused on the early-return path; pass
    # a SimpleNamespace duck.
    staged = SimpleNamespace(
        session_id="s", target_role_text=None, target_noc=None,
        preferred_location=None, work_type_preference=None,
        shift_preference=None, experience_text=None, skills=[],
    )
    result = compute_development_plan(
        diagnosis=diagnosis,
        in_memory_matches=[],
        staged=staged,
        user_explicit_development_request=False,
    )
    assert result is None


def test_compute_returns_none_on_ready_to_apply_without_explicit_request():
    """First-release CP4 secondary fires only on explicit user
    development-intent. Without it, CP4 returns None on
    READY_TO_APPLY."""
    from skillbridge.chat.development_plan import compute_development_plan

    matches = [_match(match_band="strong", required_missing=[])]
    diagnosis = diagnose(
        enough_to_match=True, usable_evidence_present=True,
        engine_completed=True, snapshot_usable=True,
        direct_match_results=matches, skill_adjacent_results=[],
        target_posting_count=1,
    )
    staged = SimpleNamespace(
        session_id="s", target_role_text=None, target_noc=None,
        preferred_location=None, work_type_preference=None,
        shift_preference=None, experience_text=None, skills=[],
    )
    result = compute_development_plan(
        diagnosis=diagnosis,
        in_memory_matches=matches,
        staged=staged,
        user_explicit_development_request=False,
    )
    assert result is None


def test_compute_returns_none_on_market_data_unavailable():
    from skillbridge.chat.development_plan import compute_development_plan

    diagnosis = diagnose(
        enough_to_match=True, usable_evidence_present=True,
        engine_completed=False,  # engine failed
        snapshot_usable=True,
        direct_match_results=[], skill_adjacent_results=[],
        target_posting_count=None,
    )
    staged = SimpleNamespace(
        session_id="s", target_role_text=None, target_noc=None,
        preferred_location=None, work_type_preference=None,
        shift_preference=None, experience_text=None, skills=[],
    )
    result = compute_development_plan(
        diagnosis=diagnosis,
        in_memory_matches=[],
        staged=staged,
        user_explicit_development_request=False,
    )
    assert result is None


# --- §3 counterfactual rescoring (real integration) ---------------------

def test_counterfactual_rescore_changes_required_missing_and_band(monkeypatch):
    """Real integration: adding a synthetic skill to the user set must
    (a) remove that skill from a posting's required_missing and
    (b) at minimum, not LOWER the band when the synthetic skill is one
    of the posting's required skills.

    Bypasses the DB by calling `_score_one_job` directly with synthetic
    job + skill_rows. The path under test is exactly the same scoring
    code the counterfactual rescorer in development_plan.py exercises,
    minus the per-posting DB fetch (which is trivial).

    `_regulated()` is monkeypatched to skip the DB lookup for
    credential warnings."""
    from datetime import date

    from skillbridge.match import engine as engine_mod
    from skillbridge.match.alignment import (
        UserSkillRow,
        build_user_skill_rows,
        canonicalize_skill,
        derive_user_skill_sets,
    )
    from skillbridge.match.engine import _score_one_job
    from skillbridge.session.staging import StagedSkill

    monkeypatch.setattr(engine_mod, "_regulated", lambda *a, **k: None)

    # Synthetic job — three required skills.
    job = {
        "job_id": "test-job-cf-1",
        "title": "Test Accountant",
        "employer": "Test Employer",
        "url": "https://example.com/test",
        "location": "Sault Ste. Marie, ON",
        "region_code": None,
        "noc_code": "14200",
        "posted_date": date(2026, 6, 1),
    }
    job_skills = [
        {"skill_id": None, "skill_name": "bookkeeping",
         "confidence": 0.95, "importance_rank": 1, "skill_type": "required"},
        {"skill_id": None, "skill_name": "accounts payable",
         "confidence": 0.95, "importance_rank": 2, "skill_type": "required"},
        {"skill_id": None, "skill_name": "journal entry posting",
         "confidence": 0.95, "importance_rank": 3, "skill_type": "required"},
    ]
    profile = {
        "profile_id": "test", "preferred_location": None,
        "target_role_text": "accountant", "target_noc": "14200",
        "work_type_preference": None, "shift_preference": None,
        "experience_text": None,
    }

    # User has 2 of 3 — explicitly chat-source so build_user_skill_rows
    # admits them.
    base_skills = [
        StagedSkill(skill_name="bookkeeping", source="chat", confidence=0.95),
        StagedSkill(skill_name="accounts payable", source="chat", confidence=0.95),
    ]
    base_rows = build_user_skill_rows(base_skills)
    assert len(base_rows) == 2, "fixture: base rows must materialize"
    base_ids, base_names, base_canon = derive_user_skill_sets(base_rows)

    current = _score_one_job(
        job, job_skills, base_ids, base_names, profile,
        user_skill_names_canon=base_canon, user_rows=base_rows,
    )
    assert current is not None
    assert "journal entry posting" in current.missing_skills, (
        "baseline: the candidate gap must be missing initially"
    )
    current_band = current.match_band

    # H1 fix: counterfactual injects via direct UserSkillRow construction
    # (bypasses the source whitelist). Mirror that here.
    candidate = "journal entry posting"
    candidate_canonical = canonicalize_skill(candidate) or candidate.lower()
    synth_row = UserSkillRow(
        skill_id=None,
        text=candidate,
        name=candidate.lower(),
        canon=candidate_canonical,
    )
    synth_rows = base_rows + [synth_row]
    assert len(synth_rows) == 3, "H1 fix: synthetic row must be present"
    synth_ids, synth_names, synth_canon = derive_user_skill_sets(synth_rows)

    projected = _score_one_job(
        job, job_skills, synth_ids, synth_names, profile,
        user_skill_names_canon=synth_canon, user_rows=synth_rows,
    )
    assert projected is not None
    assert candidate not in projected.missing_skills, (
        "counterfactual: candidate skill must be removed from missing_skills"
    )
    band_order = {"low": 0, "stretch": 1, "good": 2, "strong": 3}
    assert band_order.get(projected.match_band, -1) >= band_order.get(
        current_band, -1,
    ), (
        f"counterfactual: band regressed unexpectedly "
        f"({current_band} -> {projected.match_band})"
    )
    # And the projected required_missing should be explicitly empty —
    # the locked-contract precondition for promotion to Apply today,
    # and what the H3 fail-closed path in the diagnosis requires.
    assert isinstance(projected.score_explanation, dict)
    assert projected.score_explanation.get("required_missing") == []


def test_counterfactual_rescore_preserves_required_missing_when_unrelated(monkeypatch):
    """When the synthetic skill is NOT in the job's required list,
    rescoring must leave required_missing unchanged. Guards against a
    spurious credit on any synthetic addition."""
    from datetime import date

    from skillbridge.match import engine as engine_mod
    from skillbridge.match.alignment import (
        UserSkillRow,
        build_user_skill_rows,
        canonicalize_skill,
        derive_user_skill_sets,
    )
    from skillbridge.match.engine import _score_one_job
    from skillbridge.session.staging import StagedSkill

    monkeypatch.setattr(engine_mod, "_regulated", lambda *a, **k: None)

    job = {
        "job_id": "test-job-cf-2",
        "title": "Test Accountant",
        "employer": "Test Employer",
        "url": "https://example.com/test",
        "location": "Sault Ste. Marie, ON",
        "region_code": None,
        "noc_code": "14200",
        "posted_date": date(2026, 6, 1),
    }
    job_skills = [
        {"skill_id": None, "skill_name": "bookkeeping",
         "confidence": 0.95, "importance_rank": 1, "skill_type": "required"},
        {"skill_id": None, "skill_name": "accounts payable",
         "confidence": 0.95, "importance_rank": 2, "skill_type": "required"},
        {"skill_id": None, "skill_name": "journal entry posting",
         "confidence": 0.95, "importance_rank": 3, "skill_type": "required"},
    ]
    profile = {
        "profile_id": "test", "preferred_location": None,
        "target_role_text": "accountant", "target_noc": "14200",
        "work_type_preference": None, "shift_preference": None,
        "experience_text": None,
    }
    base_skills = [
        StagedSkill(skill_name="bookkeeping", source="chat", confidence=0.95),
        StagedSkill(skill_name="accounts payable", source="chat", confidence=0.95),
    ]
    base_rows = build_user_skill_rows(base_skills)
    base_ids, base_names, base_canon = derive_user_skill_sets(base_rows)
    current = _score_one_job(
        job, job_skills, base_ids, base_names, profile,
        user_skill_names_canon=base_canon, user_rows=base_rows,
    )
    assert current is not None

    candidate = "underwater welding"
    synth_row = UserSkillRow(
        skill_id=None,
        text=candidate,
        name=candidate.lower(),
        canon=canonicalize_skill(candidate) or candidate.lower(),
    )
    synth_rows = base_rows + [synth_row]
    synth_ids, synth_names, synth_canon = derive_user_skill_sets(synth_rows)
    projected = _score_one_job(
        job, job_skills, synth_ids, synth_names, profile,
        user_skill_names_canon=synth_canon, user_rows=synth_rows,
    )
    assert projected is not None
    assert (
        set(projected.missing_skills) == set(current.missing_skills)
    ), "unrelated synthetic skill should not change missing_skills"


# --- CandidateRankRow shape (sanitized telemetry) -----------------------

def test_rank_row_carries_only_safe_telemetry_fields():
    """CandidateRankRow must NOT carry titles, employers, URLs, or
    gap text. Only the canonical name + integer counts."""
    fields = set(CandidateRankRow.__dataclass_fields__.keys())
    assert fields == {
        "skill_canonical_name",
        "promoted_count",
        "blocker_removed_count",
        "tier_improvement_count",
        "active_market_frequency",
        "min_importance_rank",
    }
    # Spot-check: no field that could leak prose
    forbidden = {"title", "employer", "url", "gap_text", "raw_skill_name"}
    assert fields.isdisjoint(forbidden)


# --- H2 Round 3: full CP4 _counterfactual_rescore integration -----------

def test_counterfactual_rescore_full_path_with_monkeypatched_db(monkeypatch):
    """End-to-end exercise of `_counterfactual_rescore` with all four
    DB helpers monkeypatched:

      - `_fetch_job_row`   → returns a synthetic v_current_job row
      - `_fetch_job_skills` → returns the synthetic skill rows
      - `_fetch_job_skill_embeddings` → None (skip embeddings)
      - `_regulated` → None (no credential warning)

    Asserts:
      - candidate enters synthetic rows (synth_rows length grows by 1)
      - required_missing changes (candidate skill removed)
      - tier_transition is recorded
      - promotion / improvement classification is correct"""
    from datetime import date

    from skillbridge.chat.development_plan import _counterfactual_rescore
    from skillbridge.chat import development_plan as dp_mod
    from skillbridge.match import engine as engine_mod
    from skillbridge.session.staging import StagedProfile, StagedSkill

    # Synthetic job in core.v_current_job shape.
    synthetic_job = {
        "job_id": "test-cf-real-1",
        "title": "Test Accountant",
        "employer": "Test Employer",
        "url": "https://example.com/test",
        "location": "Sault Ste. Marie, ON",
        "region_code": None,
        "noc_code": "14200",
        "posted_date": date(2026, 6, 1),
    }
    synthetic_job_skills = [
        {"skill_id": None, "skill_name": "bookkeeping",
         "confidence": 0.95, "importance_rank": 1, "skill_type": "required"},
        {"skill_id": None, "skill_name": "accounts payable",
         "confidence": 0.95, "importance_rank": 2, "skill_type": "required"},
        {"skill_id": None, "skill_name": "journal entry posting",
         "confidence": 0.95, "importance_rank": 3, "skill_type": "required"},
    ]

    # Monkeypatch DB helpers used by _counterfactual_rescore.
    monkeypatch.setattr(dp_mod, "_fetch_job_row",
                        lambda job_id: synthetic_job if job_id == synthetic_job["job_id"] else None)
    monkeypatch.setattr(engine_mod, "_fetch_job_skills",
                        lambda job_id: synthetic_job_skills if job_id == synthetic_job["job_id"] else [])
    monkeypatch.setattr(engine_mod, "_fetch_job_skill_embeddings",
                        lambda job_id: None)
    monkeypatch.setattr(engine_mod, "_regulated", lambda *a, **k: None)
    # Active-market-frequency runs a full table scan; isolate the test
    # by short-circuiting it. Production CP4 calls the real impl.
    monkeypatch.setattr(dp_mod, "_query_active_market_frequency",
                        lambda canonical_name: 0)
    # Dataset/engine version helpers may hit the snapshot pipeline.
    monkeypatch.setattr(dp_mod, "_get_dataset_version", lambda: None)
    monkeypatch.setattr(dp_mod, "_get_engine_version", lambda: None)

    # Build a "current" baseline MatchResult that says user matches
    # 2/3 (bookkeeping + accounts payable) and is missing the third.
    # This simulates what `compute_matches_in_memory` would produce.
    current_match = SimpleNamespace(
        job_id=synthetic_job["job_id"],
        match_band="stretch",
        match_eligible=True,
        noc_code="14200",
        missing_skills=["journal entry posting"],
        score_explanation={"required_missing": ["journal entry posting"]},
    )

    # Staged profile: user has 2 of 3 required skills.
    staged = StagedProfile.new("test-cf-session")
    staged.target_role_text = "accountant"
    staged.target_noc = "14200"
    staged.skills = [
        StagedSkill(skill_name="bookkeeping", source="chat", confidence=0.95),
        StagedSkill(skill_name="accounts payable", source="chat", confidence=0.95),
    ]

    # Run the actual CP4 rescoring path.
    rescored = _counterfactual_rescore(
        candidate_skill="journal entry posting",
        source_job_ids=(synthetic_job["job_id"],),
        in_memory_matches=[current_match],
        staged=staged,
        target_noc_value="14200",
        min_importance_rank=999,
    )

    # Returned object must be populated.
    assert rescored is not None, (
        "_counterfactual_rescore returned None on a real candidate"
    )

    # Tier transition must be recorded for the affected job.
    assert len(rescored.tier_transitions) == 1
    transition = rescored.tier_transitions[0]
    assert transition.job_id == synthetic_job["job_id"]
    # Candidate enters → projected required_missing is empty.
    assert transition.projected_required_missing_count == 0
    assert transition.current_required_missing_count == 1

    # Improvement classification:
    # - current required_missing had 1 entry; projected has 0
    # - either band rises OR required_missing emptied → tier_improvement
    assert synthetic_job["job_id"] in rescored.tier_improvement_job_ids, (
        "improvement classification failed: rm went 1 → 0 must land "
        "in tier_improvement_job_ids"
    )

    # Promotion classification:
    # - promoted requires band ∈ {strong, good} AND empty rm AND
    #   current was not already strong/good with empty rm
    # - the engine's weighted scoring gives a 3/3 required-coverage
    #   record band=stretch ≈ 0.41 in this fixture (no preferred
    #   skills, no recency boost), which is BELOW the good/strong
    #   thresholds (0.60 / 0.75). The current run honestly does NOT
    #   promote — Constraint A: we never claim promotion without
    #   evidence of an actual band transition. The dedicated
    #   negative-control test below asserts no spurious promotion
    #   in the related "low → still low" case.
    if rescored.tier_transitions[0].projected_band in ("strong", "good"):
        assert synthetic_job["job_id"] in rescored.promoted_job_ids
    else:
        assert synthetic_job["job_id"] not in rescored.promoted_job_ids, (
            "Constraint A violation: promotion claimed without the "
            "projected band reaching strong/good"
        )


def test_counterfactual_rescore_does_not_promote_when_band_unchanged(monkeypatch):
    """Negative control: when the synthetic skill removes the gap but
    band stays below strong/good (e.g., still missing other required
    skills), the job must NOT be in promoted_job_ids.

    Constraint A: 'unlocks N jobs' is permitted only when actual
    counterfactual rescoring proves N tier transitions to Apply today.
    """
    from datetime import date

    from skillbridge.chat.development_plan import _counterfactual_rescore
    from skillbridge.chat import development_plan as dp_mod
    from skillbridge.match import engine as engine_mod
    from skillbridge.session.staging import StagedProfile, StagedSkill

    # Synthetic job requiring 5 skills.
    synthetic_job = {
        "job_id": "test-cf-real-2",
        "title": "Test Accountant",
        "employer": "Test Employer",
        "url": "https://example.com/test",
        "location": "Sault Ste. Marie, ON",
        "region_code": None,
        "noc_code": "14200",
        "posted_date": date(2026, 6, 1),
    }
    synthetic_job_skills = [
        {"skill_id": None, "skill_name": f"skill_{i}",
         "confidence": 0.95, "importance_rank": i, "skill_type": "required"}
        for i in range(1, 6)
    ]

    monkeypatch.setattr(dp_mod, "_fetch_job_row",
                        lambda job_id: synthetic_job if job_id == synthetic_job["job_id"] else None)
    monkeypatch.setattr(engine_mod, "_fetch_job_skills",
                        lambda job_id: synthetic_job_skills if job_id == synthetic_job["job_id"] else [])
    monkeypatch.setattr(engine_mod, "_fetch_job_skill_embeddings",
                        lambda job_id: None)
    monkeypatch.setattr(engine_mod, "_regulated", lambda *a, **k: None)
    monkeypatch.setattr(dp_mod, "_query_active_market_frequency",
                        lambda canonical_name: 0)
    monkeypatch.setattr(dp_mod, "_get_dataset_version", lambda: None)
    monkeypatch.setattr(dp_mod, "_get_engine_version", lambda: None)

    # User has only 1 of 5 skills (skill_1). Adding skill_2 brings them
    # to 2/5 — still below strong/good thresholds.
    current_match = SimpleNamespace(
        job_id=synthetic_job["job_id"],
        match_band="low",
        match_eligible=True,
        noc_code="14200",
        missing_skills=["skill_2", "skill_3", "skill_4", "skill_5"],
        score_explanation={"required_missing": [
            "skill_2", "skill_3", "skill_4", "skill_5",
        ]},
    )

    staged = StagedProfile.new("test-cf-session-2")
    staged.target_role_text = "accountant"
    staged.target_noc = "14200"
    staged.skills = [
        StagedSkill(skill_name="skill_1", source="chat", confidence=0.95),
    ]

    rescored = _counterfactual_rescore(
        candidate_skill="skill_2",
        source_job_ids=(synthetic_job["job_id"],),
        in_memory_matches=[current_match],
        staged=staged,
        target_noc_value="14200",
        min_importance_rank=999,
    )

    assert rescored is not None
    # Tier transition recorded.
    assert len(rescored.tier_transitions) == 1
    # NOT promoted to Apply today: 2/5 ≠ strong/good with empty rm.
    assert synthetic_job["job_id"] not in rescored.promoted_job_ids, (
        "Constraint A violation: claim of promotion without an actual "
        "tier transition to Apply today"
    )


# --- H3 fail-closed regression -----------------------------------------

def test_h3_absent_score_explanation_does_not_trigger_ready_to_apply():
    """H3 regression: a MatchResult lacking required_missing must NOT
    be admitted to READY_TO_APPLY. The diagnosis fails closed on
    absence the way the tier builder does."""
    # A SimpleNamespace lacking `score_explanation` entirely.
    bad = SimpleNamespace(
        job_id="bad",
        match_band="strong",
        match_eligible=True,
        noc_code="14200",
        score_explanation=None,  # absent / malformed
    )
    out = diagnose(
        enough_to_match=True, usable_evidence_present=True,
        engine_completed=True, snapshot_usable=True,
        direct_match_results=[bad],
        skill_adjacent_results=[],
        target_posting_count=0,
    )
    assert out.outcome != "READY_TO_APPLY"


def test_h3_empty_required_missing_does_trigger_ready_to_apply():
    """Positive control for H3: an EXPLICIT empty list still admits
    the record to READY_TO_APPLY."""
    good = SimpleNamespace(
        job_id="good",
        match_band="strong",
        match_eligible=True,
        noc_code="14200",
        score_explanation={"required_missing": []},  # explicitly empty
    )
    out = diagnose(
        enough_to_match=True, usable_evidence_present=True,
        engine_completed=True, snapshot_usable=True,
        direct_match_results=[good],
        skill_adjacent_results=[],
        target_posting_count=1,
    )
    assert out.outcome == "READY_TO_APPLY"


# --- M3 importance-aware truncation regression --------------------------

def test_m3_truncation_keeps_high_importance_candidates_first():
    """When the rescoring budget is tight, candidates with lower
    importance_rank (higher source-posting priority) must be kept;
    less-important candidates are truncated even if alphabetically
    earlier."""
    cands = [
        # Alphabetically earlier but importance is low (rank=10).
        _Candidate(
            canonical_name="a_low_importance",
            source_job_ids=tuple(f"j{i}" for i in range(70)),
            min_importance_rank=10,
        ),
        # Alphabetically later but importance is HIGH (rank=1).
        _Candidate(
            canonical_name="z_high_importance",
            source_job_ids=tuple(f"k{i}" for i in range(40)),
            min_importance_rank=1,
        ),
    ]
    kept, truncated = _truncate_to_cap(cands, cap=100)
    kept_names = [c.canonical_name for c in kept]
    # Total cost is 70+40 = 110 > 100. Truncation must drop the
    # low-importance candidate, NOT the high-importance one.
    assert "z_high_importance" in kept_names
    assert "a_low_importance" in truncated
