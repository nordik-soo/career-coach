"""Tests for skillbridge.chat.gap_evidence.

Slice 1 (foundation): pin the GapEvidence shape and its invariants.
Detector logic ships in subsequent slices (Layer A / B / C).
"""
from __future__ import annotations

import pytest

from skillbridge.chat.gap_evidence import GapEvidence

# Pure-logic tests; opt out of conftest._clean_db's DB truncate.
# Mirrors tests/test_truth_summary.py:36 + tests/test_turn_state.py:23
# convention for nodb-tagged test modules.
pytestmark = pytest.mark.nodb


# ---------------------------------------------------------------------------
# Layer-by-layer construction smoke tests
# ---------------------------------------------------------------------------
def test_gap_evidence_layer_a_construction():
    """Layer A entry: target NOC standard gap from OaSIS. Always has
    skill_id, source = reference.noc_skill, importance on 0.0-5.0."""
    ev = GapEvidence(
        layer="target_noc_standard",
        source_id="14200",
        source_label="Accounting and related clerks",
        skill_id="S00123",
        skill_name="bank reconciliation",
        blocker=False,
        importance=4.5,
        source="reference.noc_skill",
    )
    assert ev.layer == "target_noc_standard"
    assert ev.source_id == "14200"
    assert ev.skill_id == "S00123"
    assert ev.skill_name == "bank reconciliation"
    assert ev.blocker is False
    assert ev.importance == 4.5
    assert ev.source == "reference.noc_skill"


def test_gap_evidence_layer_b_construction_with_null_skill_id():
    """Layer B entry: nullable skill_id when posting extractor
    didn't resolve the name to reference.skill. importance is also
    nullable -- not every posting has importance_rank populated."""
    ev = GapEvidence(
        layer="local_posting",
        source_id="job-abc-123",
        source_label="Accounting Clerk @ Diamond J Farms",
        skill_id=None,
        skill_name="QuickBooks Desktop",
        blocker=False,
        importance=None,
        source="extracted.job_skill",
    )
    assert ev.skill_id is None
    assert ev.importance is None
    assert ev.source == "extracted.job_skill"


def test_gap_evidence_layer_c_construction():
    """Layer C entry: adjacent NOC standard gap, same shape as Layer A
    but layer discriminator distinguishes it for prompt grouping."""
    ev = GapEvidence(
        layer="adjacent_noc_standard",
        source_id="13110",
        source_label="Administrative assistants",
        skill_id="S00567",
        skill_name="medical terminology",
        blocker=False,
        importance=3.0,
        source="reference.noc_skill",
    )
    assert ev.layer == "adjacent_noc_standard"
    assert ev.source == "reference.noc_skill"


def test_gap_evidence_credential_blocker_smoke():
    """The blocker field carries the binary credential-vs-not flag.
    Subsequent slices populate it via `is_credential_skill_name`;
    this slice just pins that the field accepts the boolean and the
    classifier produces the expected result on a known credential
    string (smoke check across the module boundary)."""
    from skillbridge.match.engine import is_credential_skill_name

    assert is_credential_skill_name("Class G driver's license") is True
    assert is_credential_skill_name("bank reconciliation") is False

    ev = GapEvidence(
        layer="local_posting",
        source_id="job-driving-456",
        source_label="Delivery Driver @ Acme",
        skill_id=None,
        skill_name="Class G driver's license",
        blocker=True,
        importance=None,
        source="extracted.job_skill",
    )
    assert ev.blocker is True


# ---------------------------------------------------------------------------
# Frozen / slots / equality invariants
# ---------------------------------------------------------------------------
def test_gap_evidence_is_frozen():
    """GapEvidence is a frozen dataclass; field assignment must fail
    so callers can't silently corrupt evidence after construction."""
    ev = GapEvidence(
        layer="target_noc_standard",
        source_id="14200",
        source_label="Accounting clerk",
        skill_id="S00123",
        skill_name="bank reconciliation",
        blocker=False,
        importance=4.5,
        source="reference.noc_skill",
    )
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        ev.layer = "local_posting"  # type: ignore[misc]


def test_gap_evidence_uses_slots():
    """slot-only dataclass should reject arbitrary attribute writes
    even when callers go through object.__setattr__."""
    ev = GapEvidence(
        layer="target_noc_standard",
        source_id="14200",
        source_label="Accounting clerk",
        skill_id="S00123",
        skill_name="bank reconciliation",
        blocker=False,
        importance=4.5,
        source="reference.noc_skill",
    )
    with pytest.raises(AttributeError):
        object.__setattr__(ev, "extra_field", "x")


def test_gap_evidence_equality_by_value():
    """Frozen dataclasses get value-based __eq__; two equal-content
    instances must compare equal (important for downstream dedup /
    set-based handling)."""
    a = GapEvidence(
        layer="target_noc_standard",
        source_id="14200",
        source_label="Accounting clerk",
        skill_id="S00123",
        skill_name="bank reconciliation",
        blocker=False,
        importance=4.5,
        source="reference.noc_skill",
    )
    b = GapEvidence(
        layer="target_noc_standard",
        source_id="14200",
        source_label="Accounting clerk",
        skill_id="S00123",
        skill_name="bank reconciliation",
        blocker=False,
        importance=4.5,
        source="reference.noc_skill",
    )
    assert a == b


def test_gap_evidence_inequality_when_layer_differs():
    """Two records that share every field except `layer` must NOT
    compare equal. This protects the layer discriminator from being
    a no-op when the recommender groups by layer."""
    a = GapEvidence(
        layer="target_noc_standard",
        source_id="14200",
        source_label="Accounting clerk",
        skill_id="S00123",
        skill_name="bank reconciliation",
        blocker=False,
        importance=4.5,
        source="reference.noc_skill",
    )
    b = GapEvidence(
        layer="adjacent_noc_standard",
        source_id="14200",
        source_label="Accounting clerk",
        skill_id="S00123",
        skill_name="bank reconciliation",
        blocker=False,
        importance=4.5,
        source="reference.noc_skill",
    )
    assert a != b


# ===========================================================================
# Slice 2 -- Layer A: compute_target_noc_standard_gaps
# ===========================================================================
# Tests cover:
#   - empty / None target NOC -> status="no_target_noc"
#   - target NOC valid but no OaSIS rows -> status="no_reference_skill_profile"
#   - normal case: missing skills surface as GapEvidence
#   - user already has every standard skill -> status="ok", gaps=()
#   - credential skill -> blocker=True
#   - non-credential skill -> blocker=False
#   - importance comes through; NULL importance -> None
#   - source_label is the NOC title from reference.occupation
#   - dedupe by skill_id is defensive against malformed rows
#   - the SQL helper is monkeypatched -- these tests do NOT hit the DB.
from skillbridge.chat import gap_evidence as ge_mod


def _patch_layer_a_rows(monkeypatch, rows):
    """Replace _fetch_noc_skill_rows with a stub returning the given
    row list. Keeps tests DB-free while exercising the full Layer A
    detector path."""
    monkeypatch.setattr(
        ge_mod, "_fetch_noc_skill_rows", lambda noc_code: list(rows),
    )


def test_layer_a_returns_no_target_noc_status_for_none():
    result = ge_mod.compute_target_noc_standard_gaps(
        user_skill_ids=set(), target_noc=None,
    )
    assert result.status == "no_target_noc"
    assert result.gaps == ()


def test_layer_a_returns_no_target_noc_status_for_empty_string():
    result = ge_mod.compute_target_noc_standard_gaps(
        user_skill_ids=set(), target_noc="",
    )
    assert result.status == "no_target_noc"
    assert result.gaps == ()


def test_layer_a_returns_no_target_noc_status_for_whitespace():
    result = ge_mod.compute_target_noc_standard_gaps(
        user_skill_ids=set(), target_noc="   ",
    )
    assert result.status == "no_target_noc"


def test_layer_a_returns_no_reference_skill_profile_when_oasis_empty(
    monkeypatch,
):
    """NOC has no rows in reference.noc_skill -> honest status."""
    _patch_layer_a_rows(monkeypatch, rows=[])
    result = ge_mod.compute_target_noc_standard_gaps(
        user_skill_ids={"S00123"}, target_noc="99999",
    )
    assert result.status == "no_reference_skill_profile"
    assert result.gaps == ()


def test_layer_a_surfaces_missing_skills_as_gap_evidence(monkeypatch):
    """Normal case: user missing some skills surfaces as GapEvidence
    records with the locked field semantics."""
    _patch_layer_a_rows(monkeypatch, rows=[
        {
            "skill_id": "S00123", "skill_name": "bank reconciliation",
            "importance": 4.5, "noc_title": "Accounting and related clerks",
        },
        {
            "skill_id": "S00124", "skill_name": "tax preparation",
            "importance": 3.5, "noc_title": "Accounting and related clerks",
        },
    ])
    result = ge_mod.compute_target_noc_standard_gaps(
        user_skill_ids={"S99999"}, target_noc="14200",
    )
    assert result.status == "ok"
    assert len(result.gaps) == 2
    g0, g1 = result.gaps
    assert g0.layer == "target_noc_standard"
    assert g0.source_id == "14200"
    assert g0.source_label == "Accounting and related clerks"
    assert g0.skill_id == "S00123"
    assert g0.skill_name == "bank reconciliation"
    assert g0.blocker is False
    assert g0.importance == 4.5
    assert g0.source == "reference.noc_skill"
    assert g1.skill_id == "S00124"
    assert g1.importance == 3.5


def test_layer_a_filters_out_skills_the_user_already_has(monkeypatch):
    """Skills present in user_skill_ids must NOT appear as gaps."""
    _patch_layer_a_rows(monkeypatch, rows=[
        {"skill_id": "S00123", "skill_name": "bank reconciliation",
         "importance": 4.5, "noc_title": "Accounting clerk"},
        {"skill_id": "S00124", "skill_name": "tax preparation",
         "importance": 3.5, "noc_title": "Accounting clerk"},
    ])
    # User already has S00124 (tax preparation) -> only S00123 should
    # surface.
    result = ge_mod.compute_target_noc_standard_gaps(
        user_skill_ids={"S00124"}, target_noc="14200",
    )
    assert result.status == "ok"
    assert len(result.gaps) == 1
    assert result.gaps[0].skill_id == "S00123"


def test_layer_a_user_has_all_skills_returns_ok_with_empty_gaps(
    monkeypatch,
):
    """status="ok" with empty gaps means: NOC profile exists AND user
    has every standard skill. Distinct from no_reference_skill_profile
    so the recommender can frame the response honestly."""
    _patch_layer_a_rows(monkeypatch, rows=[
        {"skill_id": "S1", "skill_name": "skill one",
         "importance": 3.0, "noc_title": "Some occupation"},
    ])
    result = ge_mod.compute_target_noc_standard_gaps(
        user_skill_ids={"S1"}, target_noc="14200",
    )
    assert result.status == "ok"
    assert result.gaps == ()


def test_layer_a_credential_skill_flagged_as_blocker(monkeypatch):
    """is_credential_skill_name keywords -> blocker=True."""
    _patch_layer_a_rows(monkeypatch, rows=[
        {"skill_id": "S_LIC", "skill_name": "Class G driver's license",
         "importance": 5.0, "noc_title": "Delivery worker"},
        {"skill_id": "S_GEN", "skill_name": "general numeracy",
         "importance": 2.0, "noc_title": "Delivery worker"},
    ])
    result = ge_mod.compute_target_noc_standard_gaps(
        user_skill_ids=set(), target_noc="74300",
    )
    assert result.status == "ok"
    by_id = {g.skill_id: g for g in result.gaps}
    assert by_id["S_LIC"].blocker is True
    assert by_id["S_GEN"].blocker is False


def test_layer_a_handles_null_importance(monkeypatch):
    """reference.noc_skill.importance is nullable (NUMERIC(3,1) with
    no NOT NULL constraint). A None value comes through unchanged."""
    _patch_layer_a_rows(monkeypatch, rows=[
        {"skill_id": "S1", "skill_name": "skill one",
         "importance": None, "noc_title": "Some occupation"},
    ])
    result = ge_mod.compute_target_noc_standard_gaps(
        user_skill_ids=set(), target_noc="14200",
    )
    assert result.status == "ok"
    assert result.gaps[0].importance is None


def test_layer_a_handles_unparseable_importance(monkeypatch):
    """Defensive: a malformed importance value (e.g. injected via a
    schema-broken row) coerces to None rather than crashing."""
    _patch_layer_a_rows(monkeypatch, rows=[
        {"skill_id": "S1", "skill_name": "skill one",
         "importance": "not a number", "noc_title": "Some occupation"},
    ])
    result = ge_mod.compute_target_noc_standard_gaps(
        user_skill_ids=set(), target_noc="14200",
    )
    assert result.status == "ok"
    assert result.gaps[0].importance is None


def test_layer_a_drops_rows_with_skill_id_or_name(monkeypatch):
    """Defensive: a row with None / empty skill_id or skill_name is
    silently dropped (not a usable gap)."""
    _patch_layer_a_rows(monkeypatch, rows=[
        {"skill_id": None, "skill_name": "valid name",
         "importance": 3.0, "noc_title": "Some occupation"},
        {"skill_id": "S_VALID", "skill_name": "",
         "importance": 3.0, "noc_title": "Some occupation"},
        {"skill_id": "S_OK", "skill_name": "actual skill",
         "importance": 3.0, "noc_title": "Some occupation"},
    ])
    result = ge_mod.compute_target_noc_standard_gaps(
        user_skill_ids=set(), target_noc="14200",
    )
    assert result.status == "ok"
    assert len(result.gaps) == 1
    assert result.gaps[0].skill_id == "S_OK"


def test_layer_a_dedupes_by_skill_id(monkeypatch):
    """Defensive: if the SQL result somehow contains duplicate
    skill_ids (e.g., via a broken JOIN or schema), only the first
    occurrence surfaces. PRIMARY KEY (noc_code, skill_id) makes this
    impossible at the DB level, but the guard protects against
    malformed monkeypatched inputs and future schema drift."""
    _patch_layer_a_rows(monkeypatch, rows=[
        {"skill_id": "S_DUP", "skill_name": "first occurrence",
         "importance": 5.0, "noc_title": "Some occupation"},
        {"skill_id": "S_DUP", "skill_name": "second occurrence",
         "importance": 2.0, "noc_title": "Some occupation"},
    ])
    result = ge_mod.compute_target_noc_standard_gaps(
        user_skill_ids=set(), target_noc="14200",
    )
    assert result.status == "ok"
    assert len(result.gaps) == 1
    assert result.gaps[0].skill_name == "first occurrence"


def test_layer_a_normalizes_target_noc_with_whitespace(monkeypatch):
    """Whitespace around target_noc is stripped before SQL lookup.
    Defensive against any planner / handler caller that doesn't
    pre-strip."""
    captured = {}

    def fake_fetch(noc_code):
        captured["noc"] = noc_code
        return []

    monkeypatch.setattr(ge_mod, "_fetch_noc_skill_rows", fake_fetch)

    ge_mod.compute_target_noc_standard_gaps(
        user_skill_ids=set(), target_noc="  14200  ",
    )
    assert captured["noc"] == "14200"


# ---------------------------------------------------------------------------
# NOC code validation (locked: exact 5-digit, no 4-digit fallback)
# Pins the design rule in code, not just comments. Slice 4 review found
# the comments claimed exact-5-digit but the code accepted any non-empty
# string after strip(). These tests pin the fix.
# ---------------------------------------------------------------------------
def test_is_valid_noc_code_accepts_exact_5_digit():
    assert ge_mod._is_valid_noc_code("14200") is True
    assert ge_mod._is_valid_noc_code("00000") is True
    assert ge_mod._is_valid_noc_code("99999") is True


def test_is_valid_noc_code_rejects_4_digit():
    """No 4-digit family-prefix fallback in first release."""
    assert ge_mod._is_valid_noc_code("1420") is False
    assert ge_mod._is_valid_noc_code("1311") is False


def test_is_valid_noc_code_rejects_6_or_more_digit():
    assert ge_mod._is_valid_noc_code("142000") is False
    assert ge_mod._is_valid_noc_code("1") is False
    assert ge_mod._is_valid_noc_code("") is False


def test_is_valid_noc_code_rejects_alphanumeric():
    """OaSIS profile codes sometimes appear as 5-digit + decimal in
    raw source CSVs ('21232.00'). The pipeline strips the decimal to
    a bare 5-digit at load time. Anything still carrying letters or
    punctuation here is malformed."""
    assert ge_mod._is_valid_noc_code("abc14") is False
    assert ge_mod._is_valid_noc_code("142A0") is False
    assert ge_mod._is_valid_noc_code("14200.0") is False
    assert ge_mod._is_valid_noc_code("14-20") is False


def test_layer_a_returns_no_target_noc_for_4_digit_input(monkeypatch):
    """Locked design enforcement: a 4-digit NOC was the suggested
    fallback option that was explicitly REJECTED in first release.
    Layer A treats it as no_target_noc rather than silently issuing
    a SQL query that's guaranteed to miss."""
    # Patch to detect any unexpected SQL hit.
    sql_calls: list[str] = []
    monkeypatch.setattr(
        ge_mod, "_fetch_noc_skill_rows",
        lambda noc: (sql_calls.append(noc), [])[1],
    )
    result = ge_mod.compute_target_noc_standard_gaps(
        user_skill_ids=set(), target_noc="1420",
    )
    assert result.status == "no_target_noc"
    assert result.gaps == ()
    assert sql_calls == []  # no DB hit attempted


def test_layer_a_returns_no_target_noc_for_alphanumeric_input(
    monkeypatch,
):
    sql_calls: list[str] = []
    monkeypatch.setattr(
        ge_mod, "_fetch_noc_skill_rows",
        lambda noc: (sql_calls.append(noc), [])[1],
    )
    result = ge_mod.compute_target_noc_standard_gaps(
        user_skill_ids=set(), target_noc="abc14",
    )
    assert result.status == "no_target_noc"
    assert sql_calls == []


def test_layer_a_returns_no_target_noc_for_6_digit_input(monkeypatch):
    sql_calls: list[str] = []
    monkeypatch.setattr(
        ge_mod, "_fetch_noc_skill_rows",
        lambda noc: (sql_calls.append(noc), [])[1],
    )
    result = ge_mod.compute_target_noc_standard_gaps(
        user_skill_ids=set(), target_noc="142000",
    )
    assert result.status == "no_target_noc"
    assert sql_calls == []


# ===========================================================================
# Slice 3 -- Layer B: compute_local_posting_gaps
# ===========================================================================
# Tests cover:
#   - empty / None match_results -> empty tuple
#   - posting with missing skills -> one GapEvidence per missing skill
#   - source_label = "title @ employer" when both present
#   - source_label falls back gracefully when employer missing
#   - skill_id is None when engine returned None for that
#     skill (extractor didn't resolve to reference.skill)
#   - skill_id populated when engine returned an ID
#   - credential skill -> blocker=True; non-credential -> blocker=False
#   - importance is always None for Layer B (engine doesn't carry it)
#   - dedupe within a single posting on (skill_id, name)
#   - SAME skill across two DIFFERENT postings is NOT deduplicated
#   - defensive: malformed MatchResult (no job_id, no name) skipped
#   - parallel lists with mismatched lengths handled gracefully
import types as _types


def _fake_mr(
    *,
    job_id: str,
    title: str = "Some Job",
    employer: str | None = "Some Employer",
    missing_skills: list[str] | None = None,
    missing_skill_ids: list[str | None] | None = None,
):
    """Build a duck-typed MatchResult-like object for Layer B tests.
    Avoids constructing a full MatchResult dataclass (which has many
    required fields irrelevant to Layer B's logic). Layer B reads its
    inputs via getattr, so any object exposing the same attribute
    names suffices."""
    return _types.SimpleNamespace(
        job_id=job_id,
        title=title,
        employer=employer,
        missing_skills=missing_skills or [],
        missing_skill_ids=missing_skill_ids or [],
    )


def test_layer_b_empty_match_results_returns_empty_tuple():
    assert ge_mod.compute_local_posting_gaps(match_results=[]) == ()


def test_layer_b_none_results_in_iterable_are_skipped():
    """A None entry mixed into the iterable doesn't crash; valid
    entries continue to be processed."""
    result = ge_mod.compute_local_posting_gaps(match_results=[
        None,
        _fake_mr(
            job_id="job-1",
            missing_skills=["bank reconciliation"],
            missing_skill_ids=["S00123"],
        ),
        None,
    ])
    assert len(result) == 1
    assert result[0].source_id == "job-1"


def test_layer_b_one_posting_one_missing_skill():
    result = ge_mod.compute_local_posting_gaps(match_results=[
        _fake_mr(
            job_id="job-abc-123",
            title="Accounting Clerk",
            employer="Diamond J Farms",
            missing_skills=["bank reconciliation"],
            missing_skill_ids=["S00123"],
        ),
    ])
    assert len(result) == 1
    ev = result[0]
    assert ev.layer == "local_posting"
    assert ev.source_id == "job-abc-123"
    assert ev.source_label == "Accounting Clerk @ Diamond J Farms"
    assert ev.skill_id == "S00123"
    assert ev.skill_name == "bank reconciliation"
    assert ev.blocker is False
    assert ev.importance is None
    assert ev.source == "extracted.job_skill"


def test_layer_b_source_label_falls_back_to_title_only():
    result = ge_mod.compute_local_posting_gaps(match_results=[
        _fake_mr(
            job_id="j1", title="Accounting Clerk", employer=None,
            missing_skills=["X"], missing_skill_ids=["SX"],
        ),
    ])
    assert result[0].source_label == "Accounting Clerk"


def test_layer_b_source_label_falls_back_to_employer_only():
    result = ge_mod.compute_local_posting_gaps(match_results=[
        _fake_mr(
            job_id="j1", title="", employer="Diamond J Farms",
            missing_skills=["X"], missing_skill_ids=["SX"],
        ),
    ])
    assert result[0].source_label == "Diamond J Farms"


def test_layer_b_source_label_untitled_when_nothing_set():
    result = ge_mod.compute_local_posting_gaps(match_results=[
        _fake_mr(
            job_id="j1", title="", employer=None,
            missing_skills=["X"], missing_skill_ids=["SX"],
        ),
    ])
    assert result[0].source_label == "(untitled posting)"


def test_layer_b_skill_id_can_be_none():
    """Engine returns `list[str | None]` because some posting-extracted
    skill names don't resolve to reference.skill. Layer B passes the
    null through unchanged."""
    result = ge_mod.compute_local_posting_gaps(match_results=[
        _fake_mr(
            job_id="j1",
            missing_skills=["QuickBooks Desktop"],
            missing_skill_ids=[None],
        ),
    ])
    assert len(result) == 1
    assert result[0].skill_id is None
    assert result[0].skill_name == "QuickBooks Desktop"


def test_layer_b_credential_skill_flagged_as_blocker():
    result = ge_mod.compute_local_posting_gaps(match_results=[
        _fake_mr(
            job_id="j1",
            missing_skills=[
                "Class G driver's license",
                "general numeracy",
            ],
            missing_skill_ids=["S_LIC", "S_GEN"],
        ),
    ])
    by_name = {g.skill_name: g for g in result}
    assert by_name["Class G driver's license"].blocker is True
    assert by_name["general numeracy"].blocker is False


def test_layer_b_importance_always_none():
    """First-release contract: engine MatchResult doesn't carry
    per-missing-skill importance through to its shape. Layer B's
    importance is unconditionally None until/unless that changes."""
    result = ge_mod.compute_local_posting_gaps(match_results=[
        _fake_mr(
            job_id="j1",
            missing_skills=["any skill"],
            missing_skill_ids=["S1"],
        ),
    ])
    assert result[0].importance is None


def test_layer_b_dedupes_within_a_single_posting():
    """A posting that lists the same missing skill twice (rare but
    possible with messy extractor output) only surfaces it once."""
    result = ge_mod.compute_local_posting_gaps(match_results=[
        _fake_mr(
            job_id="j1",
            missing_skills=["Excel", "Excel"],
            missing_skill_ids=["S_EXCEL", "S_EXCEL"],
        ),
    ])
    assert len(result) == 1


def test_layer_b_does_NOT_dedupe_same_skill_across_postings():
    """Same skill missing on two different postings = two records.
    The recommender wants to know each posting affected by the gap."""
    result = ge_mod.compute_local_posting_gaps(match_results=[
        _fake_mr(
            job_id="j1", title="Job A",
            missing_skills=["Excel"], missing_skill_ids=["S_EXCEL"],
        ),
        _fake_mr(
            job_id="j2", title="Job B",
            missing_skills=["Excel"], missing_skill_ids=["S_EXCEL"],
        ),
    ])
    assert len(result) == 2
    assert result[0].source_id == "j1"
    assert result[1].source_id == "j2"


def test_layer_b_skips_results_without_job_id():
    """Defensive: a MatchResult-like with empty/None job_id can't
    carry a meaningful source_id. Skip rather than emit garbage."""
    result = ge_mod.compute_local_posting_gaps(match_results=[
        _fake_mr(
            job_id="",
            missing_skills=["X"], missing_skill_ids=["SX"],
        ),
    ])
    assert result == ()


def test_layer_b_skips_empty_skill_names():
    """Defensive: empty / whitespace / non-str skill names don't
    produce GapEvidence."""
    result = ge_mod.compute_local_posting_gaps(match_results=[
        _fake_mr(
            job_id="j1",
            missing_skills=["", "  ", "valid"],
            missing_skill_ids=["S1", "S2", "S3"],
        ),
    ])
    assert len(result) == 1
    assert result[0].skill_name == "valid"
    assert result[0].skill_id == "S3"


def test_layer_b_handles_mismatched_parallel_list_lengths():
    """Engine constructs missing_skills and missing_skill_ids from the
    same tm_missing iteration, so they SHOULD be the same length. But
    defensively, if ids is shorter than names (e.g., from a mocked
    test or future schema drift), missing names still surface with
    skill_id=None."""
    result = ge_mod.compute_local_posting_gaps(match_results=[
        _fake_mr(
            job_id="j1",
            missing_skills=["one", "two", "three"],
            missing_skill_ids=["S1"],  # only 1 entry vs 3 names
        ),
    ])
    assert len(result) == 3
    assert result[0].skill_id == "S1"
    assert result[1].skill_id is None
    assert result[2].skill_id is None


def test_layer_b_real_match_result_shape_smoke():
    """Smoke test against the actual engine.MatchResult dataclass to
    confirm Layer B reads the attributes that engine.py:67 actually
    exposes (not just my SimpleNamespace stub). Catches the case
    where engine field renames break Layer B."""
    from skillbridge.match.engine import MatchResult
    mr = MatchResult(
        job_id="real-job-1",
        profile_id="p1",
        title="Real Job",
        employer="Real Employer",
        url=None,
        location=None,
        match_score=0.75,
        match_band="good",
        match_eligible=True,
        ineligibility_reason=None,
        matched_skills=["X"],
        missing_skills=["bank reconciliation", "Class G driver's license"],
        matched_skill_ids=["SX"],
        missing_skill_ids=["S_BANK", None],
        required_skills_count=3,
        credential_warning=None,
        posted_date=None,
        noc_code="14200",
    )
    result = ge_mod.compute_local_posting_gaps(match_results=[mr])
    assert len(result) == 2
    assert result[0].source_id == "real-job-1"
    assert result[0].source_label == "Real Job @ Real Employer"
    assert result[0].skill_id == "S_BANK"
    assert result[0].blocker is False
    assert result[1].skill_id is None  # engine returned None
    assert result[1].blocker is True   # Class G is a credential


# ===========================================================================
# Slice 4 -- Layer C: compute_adjacent_noc_standard_gaps
# ===========================================================================
# Tests cover:
#   - empty / None sideways_move -> empty tuple
#   - sideways_move with no usable noc_code -> empty tuple
#   - single sideways job with NOC + OaSIS rows -> one slice with gaps
#   - multiple sideways jobs sharing one NOC -> deduped to one slice
#   - multiple unique NOCs -> multiple slices, first-seen order
#   - NOC with no OaSIS rows -> slice with status="no_reference_skill_profile"
#   - Mixed: some NOCs have rows, others don't -> independent per-NOC status
#   - Each gap has layer="adjacent_noc_standard" (NOT target_noc_standard)
#   - Each gap's source_id is the NOC code; source_label is the NOC title
#   - Credential skill -> blocker=True
#   - User has all skills for a NOC -> status="ok" + empty gaps (vs no_profile)
#   - Whitespace in noc_code is stripped before lookup
#   - real AdjacentJob shape compatibility smoke test


def _patch_layer_c_rows_by_noc(monkeypatch, rows_by_noc: dict):
    """Stub _fetch_noc_skill_rows to return per-NOC row lists.
    Unknown NOC -> empty list (mirrors the production behavior when
    reference.noc_skill has no rows for that NOC)."""
    monkeypatch.setattr(
        ge_mod, "_fetch_noc_skill_rows",
        lambda noc_code: list(rows_by_noc.get(noc_code, [])),
    )


def _fake_adjacent_job(noc_code: str | None = "13110"):
    """Duck-typed AdjacentJob with just the field Layer C reads."""
    return _types.SimpleNamespace(noc_code=noc_code)


def test_layer_c_empty_sideways_move_returns_empty_tuple():
    assert ge_mod.compute_adjacent_noc_standard_gaps(
        user_skill_ids=set(), sideways_move=[],
    ) == ()


def test_layer_c_none_entries_in_sideways_move_are_skipped(monkeypatch):
    _patch_layer_c_rows_by_noc(monkeypatch, rows_by_noc={
        "13110": [
            {"skill_id": "S1", "skill_name": "skill one",
             "importance": 3.0, "noc_title": "Administrative assistants"},
        ],
    })
    result = ge_mod.compute_adjacent_noc_standard_gaps(
        user_skill_ids=set(),
        sideways_move=[None, _fake_adjacent_job(noc_code="13110"), None],
    )
    assert len(result) == 1
    assert result[0].noc_code == "13110"


def test_layer_c_skips_jobs_with_no_usable_noc_code(monkeypatch):
    """noc_code=None, "", or non-string entries are filtered out."""
    _patch_layer_c_rows_by_noc(monkeypatch, rows_by_noc={})
    result = ge_mod.compute_adjacent_noc_standard_gaps(
        user_skill_ids=set(),
        sideways_move=[
            _fake_adjacent_job(noc_code=None),
            _fake_adjacent_job(noc_code=""),
            _fake_adjacent_job(noc_code="   "),
        ],
    )
    assert result == ()


def test_layer_c_single_noc_with_oasis_rows_produces_gap_evidence(
    monkeypatch,
):
    _patch_layer_c_rows_by_noc(monkeypatch, rows_by_noc={
        "13110": [
            {"skill_id": "S00567", "skill_name": "medical terminology",
             "importance": 3.0, "noc_title": "Administrative assistants"},
        ],
    })
    result = ge_mod.compute_adjacent_noc_standard_gaps(
        user_skill_ids=set(),
        sideways_move=[_fake_adjacent_job(noc_code="13110")],
    )
    assert len(result) == 1
    slice_ = result[0]
    assert slice_.noc_code == "13110"
    assert slice_.noc_label == "Administrative assistants"
    assert slice_.status == "ok"
    assert len(slice_.gaps) == 1

    gap = slice_.gaps[0]
    assert gap.layer == "adjacent_noc_standard"
    assert gap.source_id == "13110"
    assert gap.source_label == "Administrative assistants"
    assert gap.skill_id == "S00567"
    assert gap.skill_name == "medical terminology"
    assert gap.importance == 3.0
    assert gap.source == "reference.noc_skill"


def test_layer_c_multiple_jobs_same_noc_dedupes_to_one_slice(
    monkeypatch,
):
    """Three sideways jobs all at NOC 13110 -> one LayerCNocSlice."""
    _patch_layer_c_rows_by_noc(monkeypatch, rows_by_noc={
        "13110": [
            {"skill_id": "S1", "skill_name": "skill one",
             "importance": 3.0, "noc_title": "Admin assistants"},
        ],
    })
    result = ge_mod.compute_adjacent_noc_standard_gaps(
        user_skill_ids=set(),
        sideways_move=[
            _fake_adjacent_job(noc_code="13110"),
            _fake_adjacent_job(noc_code="13110"),
            _fake_adjacent_job(noc_code="13110"),
        ],
    )
    assert len(result) == 1


def test_layer_c_multiple_unique_nocs_preserve_first_seen_order(
    monkeypatch,
):
    """sideways_move=[NOC_B, NOC_A, NOC_B, NOC_C] -> slices in order
    [B, A, C]. This matches CP5's stable adjacent ordering."""
    _patch_layer_c_rows_by_noc(monkeypatch, rows_by_noc={
        "13110": [
            {"skill_id": "SA", "skill_name": "alpha",
             "importance": 3.0, "noc_title": "Admin assistants"},
        ],
        "13100": [
            {"skill_id": "SB", "skill_name": "beta",
             "importance": 3.0, "noc_title": "Business services officer"},
        ],
        "12345": [
            {"skill_id": "SC", "skill_name": "gamma",
             "importance": 3.0, "noc_title": "Other"},
        ],
    })
    result = ge_mod.compute_adjacent_noc_standard_gaps(
        user_skill_ids=set(),
        sideways_move=[
            _fake_adjacent_job(noc_code="13110"),
            _fake_adjacent_job(noc_code="13100"),
            _fake_adjacent_job(noc_code="13110"),  # dup
            _fake_adjacent_job(noc_code="12345"),
        ],
    )
    assert [s.noc_code for s in result] == ["13110", "13100", "12345"]


def test_layer_c_noc_with_no_oasis_rows_returns_no_reference_skill_profile(
    monkeypatch,
):
    """The honest case: surfaced NOC has no OaSIS profile in
    reference.noc_skill. Slice carries the status; gaps is empty.
    The recommender can frame this as 'no standard skill profile
    available for that adjacent occupation'."""
    _patch_layer_c_rows_by_noc(monkeypatch, rows_by_noc={
        # 13110 returns no rows
    })
    result = ge_mod.compute_adjacent_noc_standard_gaps(
        user_skill_ids=set(),
        sideways_move=[_fake_adjacent_job(noc_code="13110")],
    )
    assert len(result) == 1
    assert result[0].status == "no_reference_skill_profile"
    assert result[0].noc_label == ""
    assert result[0].gaps == ()


def test_layer_c_mixed_noc_profile_status_per_slice(monkeypatch):
    """Some surfaced NOCs have OaSIS data, others don't. Each slice
    reports its own status independently."""
    _patch_layer_c_rows_by_noc(monkeypatch, rows_by_noc={
        "13110": [
            {"skill_id": "S1", "skill_name": "skill one",
             "importance": 3.0, "noc_title": "Admin assistants"},
        ],
        # 13100 missing -> status="no_reference_skill_profile"
    })
    result = ge_mod.compute_adjacent_noc_standard_gaps(
        user_skill_ids=set(),
        sideways_move=[
            _fake_adjacent_job(noc_code="13110"),
            _fake_adjacent_job(noc_code="13100"),
        ],
    )
    by_noc = {s.noc_code: s for s in result}
    assert by_noc["13110"].status == "ok"
    assert len(by_noc["13110"].gaps) == 1
    assert by_noc["13100"].status == "no_reference_skill_profile"
    assert by_noc["13100"].gaps == ()


def test_layer_c_user_has_all_skills_returns_ok_with_empty_gaps(
    monkeypatch,
):
    """status="ok" with empty gaps -> OaSIS profile exists AND user
    has every standard skill for this adjacent NOC. Distinct from
    no_reference_skill_profile so the recommender can frame the
    response honestly per NOC."""
    _patch_layer_c_rows_by_noc(monkeypatch, rows_by_noc={
        "13110": [
            {"skill_id": "S_HAS_IT", "skill_name": "user has this",
             "importance": 3.0, "noc_title": "Admin assistants"},
        ],
    })
    result = ge_mod.compute_adjacent_noc_standard_gaps(
        user_skill_ids={"S_HAS_IT"},
        sideways_move=[_fake_adjacent_job(noc_code="13110")],
    )
    assert len(result) == 1
    assert result[0].status == "ok"
    assert result[0].gaps == ()


def test_layer_c_credential_skill_flagged_as_blocker(monkeypatch):
    """Same is_credential_skill_name classifier as Layers A and B.
    OaSIS canonical credential names like 'Class G driver's licence'
    pass the keyword check the same way job-extracted names do."""
    _patch_layer_c_rows_by_noc(monkeypatch, rows_by_noc={
        "74300": [
            {"skill_id": "S_LIC", "skill_name": "Class G driver's license",
             "importance": 5.0, "noc_title": "Delivery worker"},
            {"skill_id": "S_GEN", "skill_name": "general numeracy",
             "importance": 2.0, "noc_title": "Delivery worker"},
        ],
    })
    result = ge_mod.compute_adjacent_noc_standard_gaps(
        user_skill_ids=set(),
        sideways_move=[_fake_adjacent_job(noc_code="74300")],
    )
    by_id = {g.skill_id: g for g in result[0].gaps}
    assert by_id["S_LIC"].blocker is True
    assert by_id["S_GEN"].blocker is False


def test_layer_c_user_skill_set_filters_per_noc(monkeypatch):
    """The user's skill set is applied across ALL NOC comparisons,
    not just one. A skill the user has should not appear as a gap
    for any adjacent NOC that lists it."""
    _patch_layer_c_rows_by_noc(monkeypatch, rows_by_noc={
        "13110": [
            {"skill_id": "S_COMMON", "skill_name": "common skill",
             "importance": 3.0, "noc_title": "Admin assistants"},
            {"skill_id": "S_UNIQUE_A", "skill_name": "unique A",
             "importance": 3.0, "noc_title": "Admin assistants"},
        ],
        "13100": [
            {"skill_id": "S_COMMON", "skill_name": "common skill",
             "importance": 3.0, "noc_title": "Business services officer"},
            {"skill_id": "S_UNIQUE_B", "skill_name": "unique B",
             "importance": 3.0, "noc_title": "Business services officer"},
        ],
    })
    result = ge_mod.compute_adjacent_noc_standard_gaps(
        user_skill_ids={"S_COMMON"},
        sideways_move=[
            _fake_adjacent_job(noc_code="13110"),
            _fake_adjacent_job(noc_code="13100"),
        ],
    )
    by_noc = {s.noc_code: s for s in result}
    # S_COMMON is filtered out of both slices because user has it.
    a_ids = {g.skill_id for g in by_noc["13110"].gaps}
    b_ids = {g.skill_id for g in by_noc["13100"].gaps}
    assert a_ids == {"S_UNIQUE_A"}
    assert b_ids == {"S_UNIQUE_B"}


def test_layer_c_skips_4_digit_noc_in_sideways_move(monkeypatch):
    """Locked design enforcement: 4-digit family-prefix codes in
    sideways_move are silently dropped by Layer C. The job itself
    can still appear in the surfaced Sideways tier (CP5's domain),
    but Layer C cannot produce a per-NOC slice for an invalid code."""
    sql_calls: list[str] = []
    monkeypatch.setattr(
        ge_mod, "_fetch_noc_skill_rows",
        lambda noc: (sql_calls.append(noc), [])[1],
    )
    result = ge_mod.compute_adjacent_noc_standard_gaps(
        user_skill_ids=set(),
        sideways_move=[_fake_adjacent_job(noc_code="1311")],
    )
    assert result == ()
    assert sql_calls == []


def test_layer_c_skips_alphanumeric_noc_in_sideways_move(monkeypatch):
    sql_calls: list[str] = []
    monkeypatch.setattr(
        ge_mod, "_fetch_noc_skill_rows",
        lambda noc: (sql_calls.append(noc), [])[1],
    )
    result = ge_mod.compute_adjacent_noc_standard_gaps(
        user_skill_ids=set(),
        sideways_move=[_fake_adjacent_job(noc_code="abc14")],
    )
    assert result == ()
    assert sql_calls == []


def test_layer_c_mixed_valid_and_invalid_noc_codes(monkeypatch):
    """Sideways_move with a mix of valid and invalid NOC codes: only
    valid ones produce slices, invalid silently dropped. No SQL hit
    for the invalid entries."""
    _patch_layer_c_rows_by_noc(monkeypatch, rows_by_noc={
        "13110": [
            {"skill_id": "S1", "skill_name": "skill one",
             "importance": 3.0, "noc_title": "Admin assistants"},
        ],
        # No mapping for "abc14" or "1311" -- they should never reach
        # the SQL helper anyway.
    })
    result = ge_mod.compute_adjacent_noc_standard_gaps(
        user_skill_ids=set(),
        sideways_move=[
            _fake_adjacent_job(noc_code="13110"),  # valid
            _fake_adjacent_job(noc_code="1311"),   # 4-digit, invalid
            _fake_adjacent_job(noc_code="abc14"),  # alphanumeric, invalid
        ],
    )
    assert len(result) == 1
    assert result[0].noc_code == "13110"


def test_layer_c_normalizes_noc_code_with_whitespace(monkeypatch):
    """Whitespace around noc_code (defensive against cookie-deserialized
    or malformed AdjacentJob entries) is stripped before SQL lookup."""
    captured: list[str] = []

    def fake_fetch(noc_code):
        captured.append(noc_code)
        return []

    monkeypatch.setattr(ge_mod, "_fetch_noc_skill_rows", fake_fetch)

    ge_mod.compute_adjacent_noc_standard_gaps(
        user_skill_ids=set(),
        sideways_move=[_fake_adjacent_job(noc_code="  13110  ")],
    )
    assert captured == ["13110"]


def test_layer_c_layer_field_is_adjacent_noc_standard_not_target(
    monkeypatch,
):
    """Critical: Layer C records carry layer='adjacent_noc_standard',
    not 'target_noc_standard'. Same shape, different discriminator --
    the recommender groups by this field."""
    _patch_layer_c_rows_by_noc(monkeypatch, rows_by_noc={
        "13110": [
            {"skill_id": "S1", "skill_name": "skill one",
             "importance": 3.0, "noc_title": "Admin assistants"},
        ],
    })
    result = ge_mod.compute_adjacent_noc_standard_gaps(
        user_skill_ids=set(),
        sideways_move=[_fake_adjacent_job(noc_code="13110")],
    )
    assert result[0].gaps[0].layer == "adjacent_noc_standard"


def test_layer_c_real_adjacent_job_shape_smoke(monkeypatch):
    """Smoke test against the real chat.tiered_evidence.AdjacentJob
    dataclass to confirm Layer C reads `noc_code` as it actually
    exists on the production type (not just my SimpleNamespace stub).

    Layer C only reads `.noc_code`, so this constructs a minimal but
    fully-typed AdjacentJob from the production dataclass. The DB
    helper is stubbed to empty so the test stays fast and DB-free.
    """
    from skillbridge.chat.tiered_evidence import AdjacentJob, JobFacts

    adj = AdjacentJob(
        job_id="job-1",
        title="Some title",
        employer="Some employer",
        location=None,
        noc_code="13110",
        url=None,
        job_facts=JobFacts(
            posted_date=None,
            posted_days_ago=None,
            location=None,
            employment_type=None,
            salary_text=None,
        ),
        skill_alignment=(),
        transferable_pairs=(),
        important_gaps=(),
        credential_warning_text=None,
        why_adjacent="same_noc_minor_group",
        strength_claim_text="transferable_lane",
    )
    monkeypatch.setattr(
        ge_mod, "_fetch_noc_skill_rows", lambda noc_code: [],
    )
    result = ge_mod.compute_adjacent_noc_standard_gaps(
        user_skill_ids=set(),
        sideways_move=[adj],
    )
    assert len(result) == 1
    assert result[0].noc_code == "13110"
    assert result[0].status == "no_reference_skill_profile"


# ===========================================================================
# Slice 5 step 1 -- RecommenderEvidence + TrainingResource shape
# ===========================================================================
# Tests cover:
#   - Per-mode RecommenderEvidence construction (local_gap_coach,
#     target_noc_standard, adjacent_noc_standard)
#   - TrainingResource construction (verified URL-bearing only)
#   - Frozen / slots invariants for both dataclasses
#   - Equality semantics
#   - Empty training tuple is the default in non-local modes
#   - No production consumer yet (the existing guard test still passes)


def test_training_resource_construction():
    """Build a TrainingResource with all required fields. Mirrors what
    the handler would build from a registry.Resource after surface_url
    validation."""
    tr = ge_mod.TrainingResource(
        skill_id="S_CLASSG",
        skill_name="Class G driver's license",
        provider="Ministry of Transportation Ontario",
        type="credential_pathway",
        url="https://www.ontario.ca/page/driving-licence",
        summary="G-license road-test scheduling and preparation",
    )
    assert tr.skill_id == "S_CLASSG"
    assert tr.skill_name == "Class G driver's license"
    assert tr.provider == "Ministry of Transportation Ontario"
    assert tr.type == "credential_pathway"
    assert tr.url.startswith("https://")
    assert tr.summary == "G-license road-test scheduling and preparation"


def test_training_resource_skill_id_can_be_none():
    """skill_id is nullable -- when the gap's skill_id wasn't resolved
    (Layer B with extractor miss), attach by skill_name only."""
    tr = ge_mod.TrainingResource(
        skill_id=None,
        skill_name="QuickBooks Desktop",
        provider="Sault College",
        type="local_training",
        url="https://saultcollege.ca/quickbooks",
        summary="Hands-on QuickBooks course, evenings",
    )
    assert tr.skill_id is None
    assert tr.skill_name == "QuickBooks Desktop"


def test_training_resource_is_frozen():
    tr = ge_mod.TrainingResource(
        skill_id="S1", skill_name="X", provider="P",
        type="local_training", url="https://x.example/", summary="s",
    )
    with pytest.raises(Exception):  # FrozenInstanceError
        tr.provider = "other"  # type: ignore[misc]


def test_training_resource_uses_slots():
    tr = ge_mod.TrainingResource(
        skill_id="S1", skill_name="X", provider="P",
        type="local_training", url="https://x.example/", summary="s",
    )
    with pytest.raises(AttributeError):
        object.__setattr__(tr, "extra", "x")


def test_recommender_evidence_local_gap_coach_construction():
    """local_gap_coach mode carries Layer B evidence plus zero or more
    TrainingResources. The handler builds this after computing
    compute_local_posting_gaps + CP4 ranking + per-skill registry
    lookup."""
    gap = ge_mod.GapEvidence(
        layer="local_posting",
        source_id="job-abc-123",
        source_label="Accounting Clerk @ Diamond J Farms",
        skill_id="S_BANK",
        skill_name="bank reconciliation",
        blocker=False,
        importance=None,
        source="extracted.job_skill",
    )
    training = ge_mod.TrainingResource(
        skill_id="S_BANK",
        skill_name="bank reconciliation",
        provider="Sault College",
        type="local_training",
        url="https://saultcollege.ca/bookkeeping",
        summary="Bookkeeping fundamentals with bank reconciliation module",
    )
    rec = ge_mod.RecommenderEvidence(
        mode="local_gap_coach",
        evidence=(gap,),
        training=(training,),
    )
    assert rec.mode == "local_gap_coach"
    assert len(rec.evidence) == 1
    assert rec.evidence[0].skill_id == "S_BANK"
    assert len(rec.training) == 1
    assert rec.training[0].provider == "Sault College"


def test_recommender_evidence_target_noc_standard_construction():
    """target_noc_standard mode carries Layer A evidence and EMPTY
    training -- the recommender does not name training providers for
    occupation-standard development areas (those modes describe
    standards, not specific paths)."""
    gap = ge_mod.GapEvidence(
        layer="target_noc_standard",
        source_id="14200",
        source_label="Accounting and related clerks",
        skill_id="S_INFO_ORDER",
        skill_name="Information Ordering",
        blocker=False,
        importance=4.5,
        source="reference.noc_skill",
    )
    rec = ge_mod.RecommenderEvidence(
        mode="target_noc_standard",
        evidence=(gap,),
        training=(),
    )
    assert rec.mode == "target_noc_standard"
    assert rec.evidence[0].layer == "target_noc_standard"
    assert rec.training == ()


def test_recommender_evidence_adjacent_noc_standard_construction():
    """adjacent_noc_standard mode carries Layer C-derived evidence
    (flat across all surfaced adjacent NOCs) and EMPTY training."""
    gap = ge_mod.GapEvidence(
        layer="adjacent_noc_standard",
        source_id="13110",
        source_label="Administrative assistants",
        skill_id="S_ACTIVE_LISTEN",
        skill_name="Active Listening",
        blocker=False,
        importance=3.8,
        source="reference.noc_skill",
    )
    rec = ge_mod.RecommenderEvidence(
        mode="adjacent_noc_standard",
        evidence=(gap,),
        training=(),
    )
    assert rec.mode == "adjacent_noc_standard"
    assert rec.evidence[0].layer == "adjacent_noc_standard"
    assert rec.training == ()


def test_recommender_evidence_is_frozen():
    rec = ge_mod.RecommenderEvidence(
        mode="local_gap_coach", evidence=(), training=(),
    )
    with pytest.raises(Exception):
        rec.mode = "target_noc_standard"  # type: ignore[misc]


def test_recommender_evidence_uses_slots():
    rec = ge_mod.RecommenderEvidence(
        mode="local_gap_coach", evidence=(), training=(),
    )
    with pytest.raises(AttributeError):
        object.__setattr__(rec, "extra", "x")


def test_recommender_evidence_equality_by_value():
    """Frozen dataclass gives us value-based __eq__ for free; the
    test pins it for downstream code that may dedup or compare
    evidence packages."""
    rec_a = ge_mod.RecommenderEvidence(
        mode="local_gap_coach", evidence=(), training=(),
    )
    rec_b = ge_mod.RecommenderEvidence(
        mode="local_gap_coach", evidence=(), training=(),
    )
    assert rec_a == rec_b


# ---------------------------------------------------------------------------
# Slice 1 contract -- no-consumer guard deleted Slice 5 step 4
# (2026-06-19). The conversational recommender now imports gap_evidence
# from recommender_assembly + responder + handler. Per the guard's own
# instructions: "delete the test in the same diff that adds the
# consumer." Tests for the new consumers live in:
#   tests/test_recommender_assembly.py
#   tests/test_recommender_fallback.py
#   tests/test_recommender_chain.py
# ---------------------------------------------------------------------------
