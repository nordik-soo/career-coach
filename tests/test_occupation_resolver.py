"""Unit tests for matching v2 step 2 -- title -> NOC resolver and the
engine's `_target_noc_boost`.

The resolver itself hits the DB (uses sync_cursor against
reference.occupation_title_synonym). For the boost-logic tests we
monkeypatch `resolve_title_to_noc` so they stay nodb.

What these tests pin:
  - The boost helper returns 0 when either side is None (graceful
    fallback when OaSIS data isn't loaded or title doesn't resolve)
  - The boost helper returns _TARGET_NOC_BOOST on equal NOC, 0 on mismatch
  - Engine's score_components.boosts.target_noc_match reports the value
    (so the responder can narrate it in step 6)
  - compute_matches_in_memory caches the resolved NOC on staged so it's
    not re-resolved every turn
"""
from __future__ import annotations

import os

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from skillbridge.match import engine
from skillbridge.match.engine import (
    _TARGET_NOC_BOOST,
    _target_noc_boost,
)

pytestmark = pytest.mark.nodb


# ---------------------------------------------------------------------------
# _target_noc_boost -- pure function
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("job_noc, user_noc, expected", [
    # Both set + equal -> full boost
    ("21232", "21232", _TARGET_NOC_BOOST),
    # Both set + different -> no boost
    ("21232", "72200", 0.0),
    # Either side None -> no boost (graceful fallback)
    (None, "21232", 0.0),
    ("21232", None, 0.0),
    (None, None, 0.0),
    # Empty strings treated as None
    ("", "21232", 0.0),
    ("21232", "", 0.0),
])
def test_target_noc_boost_returns_expected(job_noc, user_noc, expected):
    assert _target_noc_boost(job_noc, user_noc) == expected


def test_target_noc_boost_value_matches_constant():
    """The boost magnitude is set so skills still dominate the score.
    If this constant changes, downstream fixtures may need re-pinning."""
    assert _TARGET_NOC_BOOST == 0.10


# ---------------------------------------------------------------------------
# Engine integration -- NOC boost surfaces in score_components and
# contributes to the final score.
# ---------------------------------------------------------------------------
def _make_job(*, title="Customer Service Rep", noc_code=None,
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


def _make_profile(*, target_noc=None) -> dict:
    return {
        "profile_id": "profile-test",
        "preferred_location": "Sault Ste. Marie",
        "target_role_text": None,
        "target_noc": target_noc,
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


def test_engine_score_components_includes_noc_boost_field(monkeypatch):
    """score_components.boosts.target_noc_match must exist in BOTH the
    main eligible path and the direct-title early-return path. Step 6
    (responder narration) consumes it."""
    # _regulated() does a DB query whenever the job has noc_code OR the
    # profile has target_role_text. Stub it for nodb tests of the NOC
    # boost itself (same pattern as test_hard_gates.py).
    monkeypatch.setattr(engine, "_regulated", lambda *a, **k: None)

    skills = [_make_skill("customer service"), _make_skill("communication"),
              _make_skill("teamwork")]
    job = _make_job(noc_code="64100")          # Customer service rep NOC
    profile = _make_profile(target_noc="64100")

    result = engine._score_one_job(
        job=job, job_skills=skills,
        user_skill_ids=set(),
        user_skill_names={"customer service", "communication", "teamwork"},
        profile=profile,
    )
    assert result is not None
    boosts = result.score_explanation["score_components"]["boosts"]
    assert "target_noc_match" in boosts
    assert boosts["target_noc_match"] == round(_TARGET_NOC_BOOST, 3)


def test_engine_noc_boost_fires_only_on_match(monkeypatch):
    """Boost is 0 when NOC codes differ."""
    monkeypatch.setattr(engine, "_regulated", lambda *a, **k: None)
    skills = [_make_skill("customer service"), _make_skill("communication"),
              _make_skill("teamwork")]
    job = _make_job(noc_code="21232")          # Software developer NOC
    profile = _make_profile(target_noc="64100")  # User wants CSR

    result = engine._score_one_job(
        job=job, job_skills=skills,
        user_skill_ids=set(),
        user_skill_names={"customer service", "communication", "teamwork"},
        profile=profile,
    )
    boosts = result.score_explanation["score_components"]["boosts"]
    assert boosts["target_noc_match"] == 0.0


def test_engine_noc_boost_silent_when_oasis_data_missing(monkeypatch):
    """When the resolver returned None (no OaSIS data loaded yet), the
    boost is 0 and the engine still produces a result. Sprint 5 +
    Step 1 behavior is preserved."""
    monkeypatch.setattr(engine, "_regulated", lambda *a, **k: None)
    skills = [_make_skill("customer service"), _make_skill("communication"),
              _make_skill("teamwork")]
    job = _make_job(noc_code="64100")
    profile = _make_profile(target_noc=None)   # resolver returned None

    result = engine._score_one_job(
        job=job, job_skills=skills,
        user_skill_ids=set(),
        user_skill_names={"customer service", "communication", "teamwork"},
        profile=profile,
    )
    boosts = result.score_explanation["score_components"]["boosts"]
    assert boosts["target_noc_match"] == 0.0
    # Sanity: a full match still happens via skills alone.
    assert result.score_explanation["required_match_ratio"] == 1.0


def test_engine_noc_boost_silent_when_job_has_no_noc(monkeypatch):
    """SCCC jobs without a resolved NOC stay at boost=0. Mixed-state
    safety: a partially-backfilled DB doesn't break matching for
    unbackfilled rows."""
    monkeypatch.setattr(engine, "_regulated", lambda *a, **k: None)
    skills = [_make_skill("customer service"), _make_skill("communication"),
              _make_skill("teamwork")]
    job = _make_job(noc_code=None)              # not yet backfilled
    profile = _make_profile(target_noc="64100")

    result = engine._score_one_job(
        job=job, job_skills=skills,
        user_skill_ids=set(),
        user_skill_names={"customer service", "communication", "teamwork"},
        profile=profile,
    )
    assert result.score_explanation["score_components"]["boosts"]["target_noc_match"] == 0.0


# ---------------------------------------------------------------------------
# compute_matches_in_memory caches the resolved NOC on staged
# ---------------------------------------------------------------------------
def test_compute_matches_caches_target_noc_on_staged(monkeypatch):
    """The resolver is hit exactly once per compute_matches_in_memory
    call when staged.target_noc is None. Subsequent re-scoring of the
    same staged profile (same chat turn or next turn) reuses the cached
    NOC without an extra DB round-trip."""
    from skillbridge.session.staging import StagedProfile

    call_count = {"n": 0}

    def fake_resolver(title):
        call_count["n"] += 1
        return "21232" if title == "software developer" else None

    monkeypatch.setattr(engine, "resolve_title_to_noc", fake_resolver)
    # _fetch_eligible_jobs is called inside compute_matches_in_memory --
    # return [] so the function early-exits after resolution. We're only
    # testing the resolver-call behavior, not the matching path.
    monkeypatch.setattr(engine, "_fetch_eligible_jobs", lambda: [])

    sp = StagedProfile.new("test-noc-cache")
    sp.target_role_text = "software developer"
    assert sp.target_noc is None    # starts unresolved

    engine.compute_matches_in_memory(sp, top=1)
    assert call_count["n"] == 1
    assert sp.target_noc == "21232"  # cached on staged

    # Subsequent calls reuse the cached value -- no resolver call.
    engine.compute_matches_in_memory(sp, top=1)
    assert call_count["n"] == 1     # still 1, no re-resolution


def test_compute_matches_skips_resolution_when_target_role_text_empty(monkeypatch):
    """No target_role_text -> never call the resolver."""
    from skillbridge.session.staging import StagedProfile

    call_count = {"n": 0}

    def fake_resolver(title):
        call_count["n"] += 1
        return None

    monkeypatch.setattr(engine, "resolve_title_to_noc", fake_resolver)
    monkeypatch.setattr(engine, "_fetch_eligible_jobs", lambda: [])

    sp = StagedProfile.new("test-no-target")
    # sp.target_role_text stays None

    engine.compute_matches_in_memory(sp, top=1)
    assert call_count["n"] == 0
    assert sp.target_noc is None


# ---------------------------------------------------------------------------
# Step 2 review fix: target_noc must NOT go stale when target_role_text changes.
# Regression test for the bug surfaced in code review.
# ---------------------------------------------------------------------------
def test_target_noc_clears_when_target_role_text_changes():
    """User changing target role must invalidate the cached NOC --
    otherwise warehouse jobs could be scored with a stale software-dev NOC."""
    from skillbridge.session.staging import StagedProfile

    sp = StagedProfile.new("test-stale-noc")
    sp.target_role_text = "software developer"
    sp.target_noc = "21232"             # simulate resolver having run

    # User pivots to a different role.
    sp.target_role_text = "warehouse work"

    # target_noc must be cleared so the engine re-resolves on next match.
    assert sp.target_noc is None


def test_target_noc_preserved_when_target_role_text_set_to_same_value():
    """Setting target_role_text to its current value should NOT wipe the
    cached NOC -- nothing actually changed. Important: re-entry into
    compute_matches across multiple turns of an unchanged role must
    not pay the DB round-trip."""
    from skillbridge.session.staging import StagedProfile

    sp = StagedProfile.new("test-same-role")
    sp.target_role_text = "software developer"
    sp.target_noc = "21232"

    # Re-assignment to the same string (no actual change).
    sp.target_role_text = "software developer"

    assert sp.target_noc == "21232"


def test_target_noc_clears_via_merge_fields():
    """merge_fields path (LLM-extracted fields) goes through __setattr__
    too -- same invalidation must fire."""
    from skillbridge.session.staging import StagedProfile

    sp = StagedProfile.new("test-merge")
    sp.target_role_text = "software developer"
    sp.target_noc = "21232"

    sp.merge_fields({"target_role_text": "warehouse associate"})

    assert sp.target_role_text == "warehouse associate"
    assert sp.target_noc is None


def test_target_noc_clears_via_setattr_fallback_fill():
    """Handler's fallback_fill path uses setattr(staged, slot, value).
    That goes through our __setattr__ override -- same invalidation."""
    from skillbridge.session.staging import StagedProfile

    sp = StagedProfile.new("test-fallback-fill")
    sp.target_role_text = "software developer"
    sp.target_noc = "21232"

    # Simulate handler.py line 681 path: setattr(staged, slot, value).
    setattr(sp, "target_role_text", "data analyst")

    assert sp.target_noc is None


def test_target_noc_initial_assignment_does_not_blow_away_init_value():
    """Sanity: the dataclass __init__ also goes through __setattr__.
    Setting target_role_text=None during init (the default) must not
    leak target_noc=None back in a way that breaks anything."""
    from skillbridge.session.staging import StagedProfile

    sp = StagedProfile.new("test-init")
    assert sp.target_role_text is None
    assert sp.target_noc is None
    # Now setting both for the first time should work cleanly.
    sp.target_role_text = "software developer"
    sp.target_noc = "21232"
    assert sp.target_role_text == "software developer"
    assert sp.target_noc == "21232"


# ---------------------------------------------------------------------------
# Step 2 review fix: --resolve-noc-all now actually overwrites.
# Regression test for the inverted Python skip block.
# ---------------------------------------------------------------------------
def test_step_resolve_noc_overwrites_existing_when_only_missing_false(monkeypatch):
    """The inverted block in step_resolve_noc used to skip rows that
    already had noc_code, even when only_missing=False. Fix: SQL filter
    handles the only_missing=True case; Python no longer second-guesses."""
    from skillbridge.pipeline import orchestrator as orch

    # Hand-craft a fake DB cursor that records updates.
    fake_rows = [
        {"job_id": "j1", "title": "Software Developer", "noc_code": "OLD_21232"},
        {"job_id": "j2", "title": "Warehouse Worker", "noc_code": None},
    ]
    updates: list[tuple] = []

    class _FakeCursor:
        def __init__(self):
            self.last_select_returned = None
        def execute(self, sql, params=()):
            if "SELECT" in sql.upper():
                # Both modes ask for the full set; SQL filter would have
                # narrowed if only_missing=True, but for this test we
                # pass only_missing=False so all rows come back.
                self.last_select_returned = fake_rows
            elif "UPDATE" in sql.upper():
                updates.append(params)
        def fetchall(self):
            return self.last_select_returned or []

    fake_cursor = _FakeCursor()

    class _FakeCtxMgr:
        def __enter__(self_inner):
            return fake_cursor
        def __exit__(self_inner, *a):
            pass

    monkeypatch.setattr(orch, "sync_cursor", lambda: _FakeCtxMgr())
    # Resolver always returns a (different) NOC so we can confirm overwrite.
    monkeypatch.setattr(
        "skillbridge.match.occupation.resolve_title_to_noc_with_score",
        lambda t, *, call_source="user_target": ("NEW_99999", 0.95),
    )

    summary = orch.step_resolve_noc(only_missing=False)

    # Both rows should be resolved (overwriting j1's existing NOC).
    assert summary["resolved"] == 2
    assert summary.get("skipped_existing", 0) == 0   # the bug-key shouldn't appear
    # Confirm the existing row got an UPDATE call.
    update_params = [u for u in updates]
    assert ("NEW_99999", "j1") in update_params
    assert ("NEW_99999", "j2") in update_params


def test_step_resolve_noc_only_missing_relies_on_sql_filter(monkeypatch):
    """When only_missing=True, the SQL WHERE clause excludes rows with
    noc_code set -- Python doesn't need a second skip. Verify the WHERE
    is shaped correctly."""
    from skillbridge.pipeline import orchestrator as orch

    captured_sql: list[str] = []

    class _FakeCursor:
        def execute(self, sql, params=()):
            captured_sql.append(sql)
        def fetchall(self):
            return []   # no rows -> nothing to do

    class _FakeCtxMgr:
        def __enter__(self_inner):
            return _FakeCursor()
        def __exit__(self_inner, *a):
            pass

    monkeypatch.setattr(orch, "sync_cursor", lambda: _FakeCtxMgr())
    monkeypatch.setattr(
        "skillbridge.match.occupation.resolve_title_to_noc_with_score",
        lambda t, *, call_source="user_target": (None, 0.0),
    )

    orch.step_resolve_noc(only_missing=True)

    select_sqls = [s for s in captured_sql if "SELECT" in s.upper()]
    assert select_sqls
    assert "noc_code IS NULL" in select_sqls[0]


def test_step_resolve_noc_summary_drops_dead_skipped_existing_key():
    """The summary dict no longer carries `skipped_existing` (the inverted
    block's invariant). If it's there, somebody's regressing the fix."""
    # We can't actually call step_resolve_noc without monkeypatching the DB,
    # but we can assert by import-time inspection that no module-level
    # reference to the dead key remains. Static check: the inverted block's
    # key shouldn't appear in orchestrator source.
    import inspect
    from skillbridge.pipeline import orchestrator as orch
    src = inspect.getsource(orch.step_resolve_noc)
    assert "skipped_existing" not in src, (
        "step_resolve_noc still references skipped_existing; the inverted "
        "Python skip block may have been reintroduced"
    )
