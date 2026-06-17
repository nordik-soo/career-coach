"""AR-3 tests: bulk loader + user-skill sets + retrieve + accept.

Covers (per docs/adjacent-recommendations-design.md v12):
  - `_load_active_jobs_with_skills`: one SQL pass, grouped by job_id,
    stable NULL-tolerant ordering, `skill_name` row filter (NOT
    `skill_id`).
  - `build_user_skill_sets` / `build_anchor_skill_sets`: pure set
    builders. Anchor variant filters through is_non_generic_transferable.
  - `retrieve_candidates`: SSM-filtered broad retrieval; NOC
    minor-group OR skill-evidence hit.
  - `accept_candidates`: strict AND gate over evidence,
    no-required-non-credential, credential, coverage, transferable.
    Returns (accepted, drop_counts).

All AR-3 functions ship as dead code; AR-1c's activation audit
catches any production caller.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.nodb


from skillbridge.match.adjacent import (
    ADJACENT_MIN_REQUIRED_COVERAGE,
    ADJACENT_MIN_TRANSFERABLE_STRENGTH,
    _DROP_COVERAGE,
    _DROP_CREDENTIAL,
    _DROP_NO_EVIDENCE,
    _DROP_NO_REQUIRED_NON_CREDENTIAL,
    _DROP_TRANSFERABLE,
    _classify_required,
    _load_active_jobs_with_skills,
    accept_candidates,
    build_anchor_skill_sets,
    build_user_skill_sets,
    retrieve_candidates,
)
from skillbridge.session.staging import StagedProfile, StagedSkill


# =========================================================================
# Helpers
# =========================================================================
def _staged_with_skills(skills: list[StagedSkill]) -> StagedProfile:
    sp = StagedProfile.new("sess-1")
    sp.skills = skills
    return sp


def _ssm_job(
    *,
    job_id: str = "j-1",
    title: str = "Welder",
    noc_code: str = "72106",
    skills: list[dict] | None = None,
    location: str = "Sault Ste. Marie, ON",
    region_code: str = "3557011",
) -> dict:
    return {
        "job_id": job_id,
        "title": title,
        "noc_code": noc_code,
        "location": location,
        "region_code": region_code,
        "skills": skills or [],
    }


def _required_skill(name: str, **kw) -> dict:
    return {
        "skill_name": name,
        "skill_id": kw.get("skill_id"),
        "confidence": kw.get("confidence", 0.9),
        "importance_rank": kw.get("importance_rank", 1),
        "skill_type": "required",
    }


def _preferred_skill(name: str, **kw) -> dict:
    return {
        "skill_name": name,
        "skill_id": kw.get("skill_id"),
        "confidence": kw.get("confidence", 0.7),
        "importance_rank": kw.get("importance_rank", 3),
        "skill_type": "preferred",
    }


def _three_concrete_skills() -> list[StagedSkill]:
    """Evidence-floor satisfying set, all anchor-eligible."""
    return [
        StagedSkill(skill_name="welding", source="resume", confidence=0.8),
        StagedSkill(skill_name="blueprint reading", source="resume", confidence=0.8),
        StagedSkill(skill_name="forklift operation", source="resume", confidence=0.8),
    ]


# =========================================================================
# _load_active_jobs_with_skills
# =========================================================================
class _FakeCursor:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.executed_sql: str | None = None

    def execute(self, sql, params=()):
        self.executed_sql = sql

    def fetchall(self):
        return self.rows


def _patch_sync_cursor(monkeypatch, rows):
    """Patch `sync_cursor()` to yield a fake cursor that returns
    `rows` from fetchall(). Returns the cursor so the test can
    inspect the executed SQL."""
    fake = _FakeCursor(rows)

    class _Ctx:
        def __enter__(self_in):
            return fake
        def __exit__(self_in, *a):
            return False

    # `_load_active_jobs_with_skills` does a deferred
    # `from skillbridge.db import sync_cursor` -- patch the source
    # module that the deferred import resolves through.
    from skillbridge import db as db_mod
    monkeypatch.setattr(db_mod, "sync_cursor", lambda: _Ctx())
    return fake


def test_bulk_loader_query_uses_extracted_job_skill_with_correct_join(monkeypatch):
    fake = _patch_sync_cursor(monkeypatch, rows=[])
    _ = _load_active_jobs_with_skills()
    sql = fake.executed_sql or ""
    assert "extracted.job_skill" in sql, (
        "Bulk loader must query extracted.job_skill (NOT core.job_skill)."
    )
    assert "s.job_id = j.job_id" in sql, (
        "Bulk loader must join on s.job_id = j.job_id."
    )


def test_bulk_loader_query_has_stable_ordering(monkeypatch):
    """Ordering must be deterministic + NULL-tolerant + skill_name-
    tie-broken so two runs produce byte-identical results."""
    fake = _patch_sync_cursor(monkeypatch, rows=[])
    _ = _load_active_jobs_with_skills()
    sql = (fake.executed_sql or "").lower()
    assert "order by" in sql
    # The exact key sequence from v10:
    assert "posted_date desc nulls last" in sql
    assert "j.job_id" in sql
    assert "importance_rank nulls last" in sql
    assert "confidence desc nulls last" in sql
    assert "s.skill_name" in sql


def test_bulk_loader_groups_skills_by_job_id(monkeypatch):
    rows = [
        # Two skills for j1
        {"job_id": "j1", "title": "Welder", "noc_code": "72106",
         "skill_id": "s1", "skill_name": "welding",
         "confidence": 0.9, "importance_rank": 1, "skill_type": "required"},
        {"job_id": "j1", "title": "Welder", "noc_code": "72106",
         "skill_id": "s2", "skill_name": "fitting",
         "confidence": 0.8, "importance_rank": 2, "skill_type": "required"},
        # One skill for j2
        {"job_id": "j2", "title": "Fabricator", "noc_code": "72106",
         "skill_id": "s3", "skill_name": "blueprint reading",
         "confidence": 0.7, "importance_rank": 1, "skill_type": "required"},
    ]
    _patch_sync_cursor(monkeypatch, rows=rows)
    jobs = _load_active_jobs_with_skills()
    assert len(jobs) == 2
    by_id = {j["job_id"]: j for j in jobs}
    assert len(by_id["j1"]["skills"]) == 2
    assert len(by_id["j2"]["skills"]) == 1
    # Job-side fields are preserved on the outer dict
    assert by_id["j1"]["title"] == "Welder"
    assert "skill_name" not in by_id["j1"]   # skill columns scoped to .skills


def test_bulk_loader_row_filter_uses_skill_name_not_skill_id(monkeypatch):
    """Unnormalized rows (skill_id IS NULL but skill_name NOT NULL) MUST
    survive; pure LEFT-JOIN NULL-skill rows (skill_name IS NULL) are
    correctly dropped."""
    rows = [
        # LEFT-JOIN NULL row (job exists, no extracted skills): both NULL.
        {"job_id": "j0", "title": "Skill-less role", "noc_code": "",
         "skill_id": None, "skill_name": None,
         "confidence": None, "importance_rank": None, "skill_type": None},
        # Unnormalized row: skill_id NULL but skill_name set.
        {"job_id": "j1", "title": "Welder", "noc_code": "72106",
         "skill_id": None, "skill_name": "welding",
         "confidence": 0.8, "importance_rank": None, "skill_type": "required"},
    ]
    _patch_sync_cursor(monkeypatch, rows=rows)
    jobs = _load_active_jobs_with_skills()
    by_id = {j["job_id"]: j for j in jobs}
    assert by_id["j0"]["skills"] == []          # LEFT-JOIN NULL → empty
    assert len(by_id["j1"]["skills"]) == 1     # unnormalized row preserved
    assert by_id["j1"]["skills"][0]["skill_name"] == "welding"


# =========================================================================
# build_user_skill_sets / build_anchor_skill_sets
# =========================================================================
def test_user_skill_sets_collect_id_name_canonical() -> None:
    skills = [
        StagedSkill(skill_name="Welding", source="resume",
                    confidence=0.8, skill_id="abc"),
        StagedSkill(skill_name="forklift operation", source="chat",
                    confidence=0.7),
    ]
    ids, names, canon = build_user_skill_sets(skills)
    assert "abc" in ids
    assert "welding" in names
    assert "forklift operation" in names
    # `canonicalize_skill` lowercases + folds; both names appear in
    # canonical form.
    assert "welding" in canon


def test_user_skill_sets_skip_blank_names() -> None:
    skills = [StagedSkill(skill_name="", source="resume", confidence=0.9)]
    ids, names, canon = build_user_skill_sets(skills)
    assert names == set()
    assert canon == set()


def test_user_skill_sets_skip_non_staged_entries() -> None:
    """Defensive: non-StagedSkill entries (forged cookie) skip."""
    skills = [
        StagedSkill(skill_name="welding", source="resume", confidence=0.8),
        {"skill_name": "garbage"},   # type: ignore[list-item]
        None,                         # type: ignore[list-item]
    ]
    ids, names, canon = build_user_skill_sets(skills)
    assert "welding" in names


def test_anchor_skill_sets_filter_generics() -> None:
    skills = [
        StagedSkill(skill_name="welding", source="resume", confidence=0.8),
        StagedSkill(skill_name="communication", source="resume", confidence=0.9),  # generic
        StagedSkill(skill_name="forklift operation", source="resume", confidence=0.8),
    ]
    ids, names, canon = build_anchor_skill_sets(skills)
    assert "welding" in names
    assert "forklift operation" in names
    assert "communication" not in names, "generic skill must NOT enter anchor set"


def test_anchor_skill_sets_filter_credentials() -> None:
    skills = [
        StagedSkill(skill_name="Class G License", source="resume", confidence=0.9),  # cred
        StagedSkill(skill_name="welding", source="resume", confidence=0.8),
    ]
    _, names, _ = build_anchor_skill_sets(skills)
    assert "welding" in names
    assert "class g license" not in names


def test_anchor_skill_sets_filter_low_confidence() -> None:
    skills = [
        StagedSkill(skill_name="welding", source="resume", confidence=0.5),  # below floor
        StagedSkill(skill_name="forklift operation", source="resume", confidence=0.8),
    ]
    _, names, _ = build_anchor_skill_sets(skills)
    assert "forklift operation" in names
    assert "welding" not in names


# =========================================================================
# retrieve_candidates
# =========================================================================
def test_retrieve_filters_non_ssm_jobs() -> None:
    """Both jobs share the SAME minor-group prefix (72106 / 72107 →
    "7210") so retrieval admits via NOC. The Toronto job is rejected
    purely by the SSM region filter. (Note: noc_code 72107 differs
    from target_noc 72106 so the same-NOC exclusion doesn't fire.)"""
    sp = _staged_with_skills(_three_concrete_skills())
    sp.target_noc = "72106"
    ids, names, canon = build_user_skill_sets(sp.skills)
    all_jobs = [
        _ssm_job(job_id="j-ssm", noc_code="72107",
                 skills=[_required_skill("welding")]),
        _ssm_job(job_id="j-toronto", noc_code="72107",
                 location="Toronto, ON", region_code="3520005",
                 skills=[_required_skill("welding")]),
    ]
    out = retrieve_candidates(sp, snapshot=None, all_jobs=all_jobs,
                              user_ids=ids, user_names=names, user_canon=canon)
    assert [j["job_id"] for j in out] == ["j-ssm"]


def test_retrieve_via_noc_minor_group_hit() -> None:
    sp = _staged_with_skills(_three_concrete_skills())
    sp.target_noc = "72106"   # minor group prefix 7210
    ids, names, canon = build_user_skill_sets(sp.skills)
    all_jobs = [
        # noc 72107 -- same minor group 7210; no skill overlap
        _ssm_job(job_id="j-noc-only", noc_code="72107",
                 skills=[_required_skill("metalwork")]),
    ]
    out = retrieve_candidates(sp, snapshot=None, all_jobs=all_jobs,
                              user_ids=ids, user_names=names, user_canon=canon)
    assert [j["job_id"] for j in out] == ["j-noc-only"]


def test_retrieve_via_skill_evidence_hit_without_noc() -> None:
    sp = _staged_with_skills(_three_concrete_skills())
    sp.target_noc = None   # no NOC anchor
    ids, names, canon = build_user_skill_sets(sp.skills)
    all_jobs = [
        # Missing NOC entirely; only the skill-evidence path can admit
        _ssm_job(job_id="j-skill-only", noc_code="",
                 skills=[_required_skill("welding")]),
    ]
    out = retrieve_candidates(sp, snapshot=None, all_jobs=all_jobs,
                              user_ids=ids, user_names=names, user_canon=canon)
    assert [j["job_id"] for j in out] == ["j-skill-only"]


def test_retrieve_drops_unrelated_jobs() -> None:
    sp = _staged_with_skills(_three_concrete_skills())
    sp.target_noc = "72106"
    ids, names, canon = build_user_skill_sets(sp.skills)
    all_jobs = [
        # different minor group, unrelated skills
        _ssm_job(job_id="j-unrelated", noc_code="11202",
                 skills=[_required_skill("Java programming")]),
    ]
    out = retrieve_candidates(sp, snapshot=None, all_jobs=all_jobs,
                              user_ids=ids, user_names=names, user_canon=canon)
    assert out == []


# =========================================================================
# accept_candidates
# =========================================================================
def _accept(sp: StagedProfile, retrieved: list[dict]):
    ids, names, canon = build_user_skill_sets(sp.skills)
    return accept_candidates(retrieved, sp, ids, names, canon)


def test_accept_short_circuits_when_no_evidence() -> None:
    sp = _staged_with_skills([])   # no usable evidence
    accepted, drops = _accept(sp, retrieved=[_ssm_job(skills=[
        _required_skill("welding"),
        _required_skill("blueprint reading"),
    ])])
    assert accepted == []
    assert drops[_DROP_NO_EVIDENCE] == 1


def test_accept_drops_job_with_no_required_non_credential_skills() -> None:
    sp = _staged_with_skills(_three_concrete_skills())
    # Job with only required credentials -- no non-credential anchor.
    job = _ssm_job(skills=[_required_skill("Class G License")])
    accepted, drops = _accept(sp, retrieved=[job])
    assert accepted == []
    assert drops[_DROP_NO_REQUIRED_NON_CREDENTIAL] == 1


def test_accept_drops_when_required_credential_missing() -> None:
    sp = _staged_with_skills(_three_concrete_skills())
    job = _ssm_job(skills=[
        _required_skill("welding"),
        _required_skill("blueprint reading"),
        _required_skill("forklift operation"),
        # User does NOT have Class G; required credential blocks.
        _required_skill("Class G License"),
    ])
    accepted, drops = _accept(sp, retrieved=[job])
    assert accepted == []
    assert drops[_DROP_CREDENTIAL] == 1


def test_accept_passes_when_required_credential_matches() -> None:
    sp = _staged_with_skills([
        StagedSkill(skill_name="welding", source="resume", confidence=0.8),
        StagedSkill(skill_name="blueprint reading", source="resume", confidence=0.8),
        StagedSkill(skill_name="forklift operation", source="resume", confidence=0.8),
        StagedSkill(skill_name="Class G License", source="resume", confidence=0.9),
    ])
    job = _ssm_job(skills=[
        _required_skill("welding"),
        _required_skill("blueprint reading"),
        _required_skill("forklift operation"),
        _required_skill("Class G License"),
    ])
    accepted, drops = _accept(sp, retrieved=[job])
    assert len(accepted) == 1
    assert drops[_DROP_CREDENTIAL] == 0


def test_accept_drops_when_coverage_below_floor() -> None:
    """User matches 1 of 4 required non-credential skills (mean = 0.25),
    below the 0.45 coverage floor.

    Note: `_skill_match_strength` treats word-bounded substring as a
    stage-1 (=1.0) match -- so a job-side "TIG welding" would match
    user "welding" at 1.0. The job-side skills here use tokens that
    are NOT substrings of any user-side skill so the count is
    genuinely 1 of 4."""
    sp = _staged_with_skills(_three_concrete_skills())
    job = _ssm_job(skills=[
        _required_skill("welding"),                # user has "welding" → 1.0
        _required_skill("food safety"),            # unrelated → 0.0
        _required_skill("till operation"),         # unrelated → 0.0
        _required_skill("inventory counting"),     # unrelated → 0.0
    ])
    accepted, drops = _accept(sp, retrieved=[job])
    assert accepted == []
    assert drops[_DROP_COVERAGE] == 1


def test_accept_passes_when_coverage_at_or_above_floor() -> None:
    """Coverage = 2/3 = 0.667, above the 0.45 floor."""
    sp = _staged_with_skills(_three_concrete_skills())
    job = _ssm_job(skills=[
        _required_skill("welding"),               # exact → 1.0
        _required_skill("blueprint reading"),     # exact → 1.0
        _required_skill("TIG welding"),           # no match → 0.0
    ])
    accepted, drops = _accept(sp, retrieved=[job])
    assert len(accepted) == 1


def test_accept_drops_when_anchor_only_strength_below_floor() -> None:
    """The required job-skill matches the user's NON-anchor set
    (generics + low-confidence + credentials) at full strength, so
    coverage passes -- but the anchor-only set sees no match strong
    enough, and the transferable gate drops the candidate.

    Setup: the user clears `has_usable_skill_evidence` (3 concrete
    skills) AND has additional anchor-ineligible skills that the job
    requires. The job's required non-credential skills are EXACTLY
    those anchor-ineligible items, so they match the full user sets
    at 1.0 (coverage passes) but the anchor sets at 0.0 (transferable
    drops).
    """
    sp = _staged_with_skills([
        # Anchor-eligible (concrete, resume @ 0.8): clears evidence
        # floor but doesn't overlap any job-required skill below.
        StagedSkill(skill_name="electrical wiring", source="resume", confidence=0.8),
        StagedSkill(skill_name="circuit testing", source="resume", confidence=0.8),
        StagedSkill(skill_name="schematics", source="resume", confidence=0.8),
        # Anchor-INeligible (generics): rejected by
        # is_non_generic_transferable so they never enter the anchor
        # set, but still count for coverage against the full user set.
        StagedSkill(skill_name="communication", source="resume", confidence=0.9),
        StagedSkill(skill_name="teamwork", source="resume", confidence=0.9),
        StagedSkill(skill_name="time management", source="resume", confidence=0.9),
    ])
    job = _ssm_job(skills=[
        _required_skill("communication"),
        _required_skill("teamwork"),
        _required_skill("time management"),
    ])
    accepted, drops = _accept(sp, retrieved=[job])
    assert accepted == []
    # Coverage = 1.0 (all three match the full user set), but anchor
    # gate fails because the anchor sets don't include any of those
    # generics -- the transferable drop fires.
    assert drops[_DROP_TRANSFERABLE] == 1
    assert drops[_DROP_COVERAGE] == 0
    assert drops[_DROP_NO_EVIDENCE] == 0


def test_accept_drops_when_anchor_match_strength_too_low() -> None:
    """The user has a concrete skill but none match the job's
    required non-credential skills strongly enough."""
    sp = _staged_with_skills([
        StagedSkill(skill_name="electrical wiring", source="resume", confidence=0.8),
        StagedSkill(skill_name="circuit testing", source="resume", confidence=0.8),
        StagedSkill(skill_name="schematics", source="resume", confidence=0.8),
    ])
    # All required skills are unrelated to the user's set, but the
    # token "schematics" might fuzz-match "schematic reading" -- so
    # use clearly unrelated tokens to avoid accidental overlap.
    job = _ssm_job(skills=[
        _required_skill("food safety"),
        _required_skill("inventory counting"),
        _required_skill("till operation"),
    ])
    accepted, drops = _accept(sp, retrieved=[job])
    assert accepted == []
    # The user is in evidence-floor and one of {coverage, transferable}
    # must record the drop. Coverage will fire first (mean = 0.0).
    assert drops[_DROP_COVERAGE] + drops[_DROP_TRANSFERABLE] == 1


# =========================================================================
# _classify_required helper
# =========================================================================
def test_classify_required_splits_credentials_and_non_credentials() -> None:
    skills = [
        _required_skill("welding"),
        _required_skill("Class G License"),
        _preferred_skill("attention to detail"),   # preferred ignored
        _required_skill("blueprint reading"),
        _required_skill("310S"),
    ]
    req_cred, req_non_cred = _classify_required(skills)
    assert {s["skill_name"] for s in req_cred} == {
        "Class G License", "310S",
    }
    assert {s["skill_name"] for s in req_non_cred} == {
        "welding", "blueprint reading",
    }


def test_classify_required_skips_preferred_completely() -> None:
    skills = [
        _preferred_skill("Class G License"),  # preferred credential ignored
        _preferred_skill("welding"),
    ]
    req_cred, req_non_cred = _classify_required(skills)
    assert req_cred == []
    assert req_non_cred == []


# =========================================================================
# Threshold-constant sanity
# =========================================================================
# =========================================================================
# AR-3 round-2: low-quality skills cannot inflate coverage
# =========================================================================
def test_user_sets_filter_off_source_skills() -> None:
    """A fallback-source row must NOT enter the matching sets even
    when its confidence is high. Otherwise it could match a job-side
    required skill and inflate coverage past the floor."""
    skills = [
        StagedSkill(skill_name="welding", source="resume", confidence=0.8),
        StagedSkill(skill_name="food safety", source="fallback", confidence=0.9),
    ]
    _, names, _ = build_user_skill_sets(skills)
    assert "welding" in names
    assert "food safety" not in names


def test_user_sets_filter_low_confidence_skills() -> None:
    """A resume row below the confidence floor must NOT enter the
    matching sets even though source is accepted."""
    skills = [
        StagedSkill(skill_name="welding", source="resume", confidence=0.8),
        StagedSkill(skill_name="food safety", source="resume", confidence=0.4),
    ]
    _, names, _ = build_user_skill_sets(skills)
    assert "welding" in names
    assert "food safety" not in names


def test_user_sets_filter_malformed_confidence() -> None:
    """NaN / inf / boolean / out-of-range confidences are rejected."""
    skills = [
        StagedSkill(skill_name="welding", source="resume", confidence=0.8),
        StagedSkill(skill_name="bad nan", source="resume", confidence=float("nan")),
        StagedSkill(skill_name="bad inf", source="resume", confidence=float("inf")),
        StagedSkill(skill_name="bad bool", source="resume", confidence=True),  # type: ignore[arg-type]
        StagedSkill(skill_name="bad oor", source="resume", confidence=1.5),
    ]
    _, names, _ = build_user_skill_sets(skills)
    assert names == {"welding"}


def test_accept_does_not_inflate_coverage_with_low_quality_skills() -> None:
    """Reproduces the AR-3 review finding: 3 valid resume@0.8 skills
    clear has_usable_skill_evidence, then 2 additional low-confidence
    rows match the job's other required skills at 1.0 each --
    coverage would have been 1.0 (3 valid + 2 low-quality / 5 = 1.0).
    With the evidence-quality floor, only the 3 valid rows contribute
    and the 2 unrelated job-side skills score 0.0, dropping coverage
    below the 0.45 floor."""
    sp = _staged_with_skills([
        # Valid evidence-floor anchors:
        StagedSkill(skill_name="welding", source="resume", confidence=0.8),
        StagedSkill(skill_name="blueprint reading", source="resume", confidence=0.8),
        StagedSkill(skill_name="forklift operation", source="resume", confidence=0.8),
        # Low-quality synthetic rows that MUST NOT contribute:
        StagedSkill(skill_name="food safety", source="resume", confidence=0.3),
        StagedSkill(skill_name="cash handling", source="fallback", confidence=0.9),
    ])
    job = _ssm_job(skills=[
        _required_skill("welding"),                # valid → 1.0
        _required_skill("food safety"),            # low-conf → MUST be 0.0
        _required_skill("cash handling"),          # off-source → MUST be 0.0
        _required_skill("till operation"),         # unrelated → 0.0
        _required_skill("inventory counting"),     # unrelated → 0.0
    ])
    accepted, drops = _accept(sp, retrieved=[job])
    assert accepted == [], (
        "Low-quality skills should NOT inflate coverage. Without the "
        "evidence-quality floor, this job would have been accepted."
    )
    assert drops[_DROP_COVERAGE] == 1


# =========================================================================
# AR-3 round-2: same-target-NOC exclusion (different-role discovery lock)
# =========================================================================
def test_retrieve_excludes_jobs_with_same_target_noc() -> None:
    """Adjacency is different-role discovery -- a posting whose NOC
    matches the user's target exactly is the SAME occupation and must
    not appear in adjacency results."""
    sp = _staged_with_skills(_three_concrete_skills())
    sp.target_noc = "72106"
    ids, names, canon = build_user_skill_sets(sp.skills)
    all_jobs = [
        _ssm_job(job_id="j-same", noc_code="72106",
                 skills=[_required_skill("welding")]),
        _ssm_job(job_id="j-adjacent", noc_code="72107",
                 skills=[_required_skill("metalwork")]),
    ]
    out = retrieve_candidates(sp, snapshot=None, all_jobs=all_jobs,
                              user_ids=ids, user_names=names, user_canon=canon)
    assert [j["job_id"] for j in out] == ["j-adjacent"]


def test_retrieve_admits_same_noc_when_user_has_no_target_noc() -> None:
    """Without a target NOC the exclusion can't fire; the job is
    admitted via the skill-evidence path."""
    sp = _staged_with_skills(_three_concrete_skills())
    sp.target_noc = None
    ids, names, canon = build_user_skill_sets(sp.skills)
    all_jobs = [
        _ssm_job(job_id="j-anything", noc_code="72106",
                 skills=[_required_skill("welding")]),
    ]
    out = retrieve_candidates(sp, snapshot=None, all_jobs=all_jobs,
                              user_ids=ids, user_names=names, user_canon=canon)
    assert [j["job_id"] for j in out] == ["j-anything"]


# =========================================================================
# AR-3 round-2: defensive guards against malformed persisted state
# =========================================================================
def test_retrieve_tolerates_non_string_target_noc() -> None:
    """A forged cookie can persist target_noc as a non-str (e.g. int).
    The retriever must not crash; it should treat the field as absent
    and admit on the skill-evidence path only."""
    sp = _staged_with_skills(_three_concrete_skills())
    sp.__dict__["target_noc"] = 72106   # bypass __setattr__ for the test
    ids, names, canon = build_user_skill_sets(sp.skills)
    all_jobs = [
        _ssm_job(job_id="j-1", noc_code="72106",
                 skills=[_required_skill("welding")]),
    ]
    # Must NOT raise.
    out = retrieve_candidates(sp, snapshot=None, all_jobs=all_jobs,
                              user_ids=ids, user_names=names, user_canon=canon)
    # Because target_noc is treated as empty, same-NOC exclusion doesn't
    # fire AND NOC minor-group hit doesn't fire either; the skill-
    # evidence path admits the job.
    assert [j["job_id"] for j in out] == ["j-1"]


def test_retrieve_tolerates_non_dict_skill_entries() -> None:
    """Malformed skill entries (None, strings, lists) must not crash
    _skill_match_strength. The retriever skips them."""
    sp = _staged_with_skills(_three_concrete_skills())
    sp.target_noc = "72106"
    ids, names, canon = build_user_skill_sets(sp.skills)
    job = _ssm_job(job_id="j-1", noc_code="72106", skills=[])
    job["skills"] = [
        None,                                       # type: ignore[list-item]
        "not a dict",                               # type: ignore[list-item]
        [_required_skill("welding")],               # nested list → not a dict
        _required_skill("welding"),                 # valid → should match
    ]
    # Same-NOC exclusion would normally drop this; clear target_noc to
    # exercise the skill-evidence path with the malformed list.
    sp.target_noc = None
    out = retrieve_candidates(sp, snapshot=None, all_jobs=[job],
                              user_ids=ids, user_names=names, user_canon=canon)
    assert [j["job_id"] for j in out] == ["j-1"]


def test_retrieve_tolerates_non_list_skills_field() -> None:
    sp = _staged_with_skills(_three_concrete_skills())
    sp.target_noc = None
    ids, names, canon = build_user_skill_sets(sp.skills)
    job = _ssm_job(job_id="j-1", noc_code="72106", skills=[])
    job["skills"] = "not a list"   # type: ignore[assignment]
    # Skill-evidence path can't admit; the job is dropped silently.
    out = retrieve_candidates(sp, snapshot=None, all_jobs=[job],
                              user_ids=ids, user_names=names, user_canon=canon)
    assert out == []


def test_retrieve_tolerates_non_dict_job_entries() -> None:
    """An all_jobs list with garbage entries (forged blob) must not
    crash retrieval."""
    sp = _staged_with_skills(_three_concrete_skills())
    sp.target_noc = "72106"
    ids, names, canon = build_user_skill_sets(sp.skills)
    all_jobs = [
        None,                                       # type: ignore[list-item]
        "not a dict",                               # type: ignore[list-item]
        _ssm_job(job_id="j-good", noc_code="72107",
                 skills=[_required_skill("metalwork")]),
    ]
    out = retrieve_candidates(sp, snapshot=None, all_jobs=all_jobs,
                              user_ids=ids, user_names=names, user_canon=canon)
    assert [j["job_id"] for j in out] == ["j-good"]


def test_accept_tolerates_non_dict_job_in_retrieved() -> None:
    """If a forged retrieve_candidates output contained a non-dict,
    accept_candidates must drop it without crashing."""
    sp = _staged_with_skills(_three_concrete_skills())
    accepted, drops = _accept(sp, retrieved=[
        None,                                       # type: ignore[list-item]
        "garbage",                                  # type: ignore[list-item]
        _ssm_job(skills=[
            _required_skill("welding"),
            _required_skill("blueprint reading"),
        ]),
    ])
    # The non-dict entries are dropped as "no required non-credential
    # skills" (they have no skills to score). The valid job has only
    # required-non-credential skills the user matches at 1.0 each,
    # but coverage = 2/2 = 1.0 and anchor max = 1.0 -- both pass.
    assert len(accepted) == 1
    assert drops[_DROP_NO_REQUIRED_NON_CREDENTIAL] == 2


# =========================================================================
# AR-3 round-2: two-run byte-identical loader output
# =========================================================================
def test_bulk_loader_produces_byte_identical_output_across_runs(monkeypatch) -> None:
    """The combined ORDER BY + Python-side grouping must yield
    byte-identical results when called twice with the same input.
    Reproducibility guarantee: snapshot evidence summaries stay
    stable across runs of the engine."""
    import json

    rows = [
        # Two jobs each with multiple skills in a fixed order.
        {"job_id": "j1", "title": "Welder", "noc_code": "72106",
         "skill_id": "s1", "skill_name": "welding",
         "confidence": 0.9, "importance_rank": 1, "skill_type": "required"},
        {"job_id": "j1", "title": "Welder", "noc_code": "72106",
         "skill_id": None, "skill_name": "fitting",
         "confidence": 0.8, "importance_rank": 2, "skill_type": "required"},
        {"job_id": "j1", "title": "Welder", "noc_code": "72106",
         "skill_id": "s2", "skill_name": "blueprint reading",
         "confidence": 0.7, "importance_rank": 3, "skill_type": "preferred"},
        {"job_id": "j2", "title": "Fabricator", "noc_code": "72106",
         "skill_id": "s3", "skill_name": "metal fabrication",
         "confidence": 0.85, "importance_rank": 1, "skill_type": "required"},
    ]

    # Two consecutive runs with the SAME input.
    _patch_sync_cursor(monkeypatch, rows=list(rows))
    out_a = _load_active_jobs_with_skills()
    _patch_sync_cursor(monkeypatch, rows=list(rows))
    out_b = _load_active_jobs_with_skills()

    blob_a = json.dumps(out_a, sort_keys=False, default=str)
    blob_b = json.dumps(out_b, sort_keys=False, default=str)
    assert blob_a == blob_b, (
        "Two runs of _load_active_jobs_with_skills on the same input "
        "produced different output. The grouping or ordering is not "
        "deterministic."
    )

    # Sanity: the output contains both jobs with skills attached in the
    # input order (which is what the SQL ORDER BY guarantees at the DB).
    assert [j["job_id"] for j in out_a] == ["j1", "j2"]
    assert [s["skill_name"] for s in out_a[0]["skills"]] == [
        "welding", "fitting", "blueprint reading",
    ]


# =========================================================================
# AR-3 round-3: malformed skill_name string guards
# =========================================================================
def test_user_sets_tolerate_non_string_skill_name() -> None:
    """StagedSkill.skill_name is typed `str` but dataclass doesn't
    enforce. A forged cookie can smuggle int/bool/None; the builder
    must skip those entries instead of crashing on `.strip().lower()`."""
    skills = [
        StagedSkill(skill_name="welding", source="resume", confidence=0.8),
        StagedSkill(skill_name=7, source="resume", confidence=0.8),  # type: ignore[arg-type]
        StagedSkill(skill_name=True, source="resume", confidence=0.8),  # type: ignore[arg-type]
        StagedSkill(skill_name=None, source="resume", confidence=0.8),  # type: ignore[arg-type]
    ]
    ids, names, canon = build_user_skill_sets(skills)
    assert names == {"welding"}


def test_classify_required_tolerates_non_string_skill_name() -> None:
    """Job-side skill rows with non-str skill_name (forged in-memory
    list, broken extraction) must be skipped, not crash on
    `is_credential_skill_name`."""
    job_skills = [
        _required_skill("welding"),
        {"skill_name": 7, "skill_type": "required"},
        {"skill_name": True, "skill_type": "required"},
        {"skill_name": None, "skill_type": "required"},
        {"skill_name": "", "skill_type": "required"},
    ]
    req_cred, req_non_cred = _classify_required(job_skills)
    assert {s["skill_name"] for s in req_non_cred} == {"welding"}
    assert req_cred == []


def test_retrieve_tolerates_non_string_skill_name() -> None:
    """An adjacency candidate whose required-skill row carries a
    non-str skill_name must not crash retrieval."""
    sp = _staged_with_skills(_three_concrete_skills())
    sp.target_noc = None
    ids, names, canon = build_user_skill_sets(sp.skills)
    job = _ssm_job(job_id="j-1", noc_code="72106", skills=[
        {"skill_name": 7, "skill_type": "required"},
        {"skill_name": True, "skill_type": "required"},
        _required_skill("welding"),
    ])
    # Must NOT raise.
    out = retrieve_candidates(sp, snapshot=None, all_jobs=[job],
                              user_ids=ids, user_names=names, user_canon=canon)
    # The "welding" skill admits via skill-evidence; bad rows skipped.
    assert [j["job_id"] for j in out] == ["j-1"]


def test_accept_tolerates_non_string_skill_name() -> None:
    """The full accept path (which goes through _classify_required +
    inner _skill_match_strength calls) must not crash on malformed
    skill_name."""
    sp = _staged_with_skills(_three_concrete_skills())
    job = _ssm_job(skills=[
        {"skill_name": 7, "skill_type": "required"},
        {"skill_name": True, "skill_type": "required"},
        _required_skill("welding"),
        _required_skill("blueprint reading"),
    ])
    accepted, drops = _accept(sp, retrieved=[job])
    # The 2 valid required skills both match at 1.0 → coverage 1.0,
    # anchor max 1.0. Accepted.
    assert len(accepted) == 1


# =========================================================================
# AR-3 round-4: malformed user-side StagedSkill.skill_name routed through
# `accept_candidates` -- exercises the shared predicates
# (has_usable_skill_evidence + is_non_generic_transferable) end-to-end.
# =========================================================================
@pytest.mark.parametrize("bad_value", [7, True, None, {}, []])
def test_accept_tolerates_user_side_malformed_skill_name(bad_value) -> None:
    """A forged-cookie StagedSkill with non-str skill_name reaches
    `has_usable_skill_evidence` and (when anchor sets are built)
    `is_non_generic_transferable`. Both call `canonicalize_skill`,
    which does `.lower()` on the input and crashes on a non-str.
    Both predicates MUST skip malformed entries -- the whole accept
    path must complete without raising."""
    sp = _staged_with_skills([
        # Three valid evidence-floor anchors.
        StagedSkill(skill_name="welding", source="resume", confidence=0.8),
        StagedSkill(skill_name="blueprint reading", source="resume", confidence=0.8),
        StagedSkill(skill_name="forklift operation", source="resume", confidence=0.8),
        # Malformed row -- the exact value the reviewer reproduced with.
        StagedSkill(skill_name=bad_value, source="resume", confidence=0.8),  # type: ignore[arg-type]
    ])
    job = _ssm_job(skills=[
        _required_skill("welding"),
        _required_skill("blueprint reading"),
    ])
    # Must NOT raise. Acceptance for the valid job goes through.
    accepted, drops = _accept(sp, retrieved=[job])
    assert len(accepted) == 1, (
        f"accept_candidates crashed or dropped the valid job when a "
        f"malformed user-side skill_name={bad_value!r} was present."
    )


@pytest.mark.parametrize("bad_value", [7, True, None, {}, []])
def test_has_usable_skill_evidence_tolerates_malformed_skill_name(
    bad_value,
) -> None:
    """Direct unit test for the predicate. With 3 valid skills above
    the floor and an additional malformed row, the floor still
    passes."""
    from skillbridge.match.adjacent import has_usable_skill_evidence

    sp = _staged_with_skills([
        StagedSkill(skill_name="welding", source="resume", confidence=0.8),
        StagedSkill(skill_name="blueprint reading", source="resume", confidence=0.8),
        StagedSkill(skill_name="forklift operation", source="resume", confidence=0.8),
        StagedSkill(skill_name=bad_value, source="resume", confidence=0.8),  # type: ignore[arg-type]
    ])
    assert has_usable_skill_evidence(sp) is True


@pytest.mark.parametrize("bad_value", [7, True, None, {}, []])
def test_is_non_generic_transferable_tolerates_malformed_skill_name(
    bad_value,
) -> None:
    """Direct unit test for the anchor classifier. A malformed row
    must NOT match `_GENERIC_SKILL_CANONICALS` membership against a
    crashing canonicalize -- it returns False cleanly."""
    from skillbridge.match.adjacent import is_non_generic_transferable

    s = StagedSkill(skill_name=bad_value, source="resume", confidence=0.8)  # type: ignore[arg-type]
    assert is_non_generic_transferable(s) is False


# =========================================================================
# AR-3 round-3: UUID job_id coercion in the bulk loader
# =========================================================================
def test_bulk_loader_coerces_uuid_job_id_to_string(monkeypatch) -> None:
    """Postgres `core.job_posting.job_id` is a UUID column. psycopg
    deserializes it to `uuid.UUID`. Downstream consumers
    (presented_job_ids comparisons, snapshot sanitizers,
    JSON serializers) all expect strings -- the loader must coerce."""
    import uuid

    job_uuid = uuid.UUID("11111111-2222-3333-4444-555555555555")
    rows = [
        {"job_id": job_uuid, "title": "Welder", "noc_code": "72106",
         "skill_id": "s1", "skill_name": "welding",
         "confidence": 0.9, "importance_rank": 1, "skill_type": "required"},
        {"job_id": job_uuid, "title": "Welder", "noc_code": "72106",
         "skill_id": "s2", "skill_name": "fitting",
         "confidence": 0.8, "importance_rank": 2, "skill_type": "required"},
    ]
    _patch_sync_cursor(monkeypatch, rows=rows)
    jobs = _load_active_jobs_with_skills()

    assert len(jobs) == 1
    out_id = jobs[0]["job_id"]
    assert isinstance(out_id, str), (
        f"job_id should be str, got {type(out_id).__name__} ({out_id!r}). "
        f"presented_job_ids sanitization would drop a UUID at the "
        f"cookie boundary."
    )
    assert out_id == "11111111-2222-3333-4444-555555555555"


def test_bulk_loader_uuid_job_id_serializes_to_json(monkeypatch) -> None:
    """End-to-end: a uuid.UUID job_id must round-trip through
    json.dumps(default=None) without raising."""
    import json
    import uuid

    rows = [
        {"job_id": uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
         "title": "Welder", "noc_code": "72106",
         "skill_id": "s1", "skill_name": "welding",
         "confidence": 0.9, "importance_rank": 1, "skill_type": "required"},
    ]
    _patch_sync_cursor(monkeypatch, rows=rows)
    jobs = _load_active_jobs_with_skills()
    # json.dumps default has no UUID handler. If the coercion is wrong
    # this raises TypeError("Object of type UUID is not JSON serializable").
    json.dumps([j["job_id"] for j in jobs])


def test_threshold_constants_pinned() -> None:
    """v11 / v12 locked the AR-3 thresholds. Any change must be
    accompanied by a documented review pass and updated tests --
    this test pins the values explicitly so casual drift is caught."""
    assert ADJACENT_MIN_REQUIRED_COVERAGE == 0.45
    assert ADJACENT_MIN_TRANSFERABLE_STRENGTH == 0.70
