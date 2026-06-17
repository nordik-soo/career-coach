"""Round-34 NOC observability + freshness stabilization tests.

Coverage:
  - Resolver four-state distinction (empty / exact / fuzzy / unresolved)
  - Miss telemetry (logged on UNRESOLVED only; PII-free)
  - Inspection command structural-failure detection
  - Inspection command coverage math
  - CLI exit codes (`--strict-noc-coverage`)
  - Loader idempotency for `load_oasis_occupation_titles`

All tests use mocked `sync_cursor` -- no live DB.
"""
from __future__ import annotations

import logging
import os
from typing import Any
from unittest.mock import MagicMock

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from skillbridge.match.inspect_noc import (
    NocCoverageReport,
    UnresolvedTitleGroup,
    cli_inspect_noc_coverage,
    format_noc_coverage_report,
    inspect_noc_coverage,
)
from skillbridge.match.occupation import (
    NocResolution,
    NocResolutionState,
    resolve_title_to_noc,
    resolve_title_to_noc_with_score,
    resolve_title_to_noc_with_state,
)

pytestmark = pytest.mark.nodb


# ============================================================================
# FakeCursor harness -- script SELECT responses by call order
# ============================================================================
class _FakeCursor:
    """Mock cursor that returns scripted rows for each execute() in order.

    Each entry in `responses` is either:
      - a dict (fetchone returns it; fetchall returns [it])
      - a list (fetchone returns first or None; fetchall returns it)
      - None (fetchone returns None; fetchall returns [])
    """
    def __init__(self, responses):
        self._responses = list(responses)
        self._idx = 0
        self._last_row = None
        self._last_rows: list[dict] = []
        self.captured_sql: list[tuple[str, tuple]] = []

    def execute(self, sql, params=()):
        self.captured_sql.append((sql, params if isinstance(params, tuple) else tuple(params) if params else ()))
        if self._idx >= len(self._responses):
            self._last_row = None
            self._last_rows = []
            self._idx += 1
            return
        resp = self._responses[self._idx]
        self._idx += 1
        if resp is None:
            self._last_row = None
            self._last_rows = []
        elif isinstance(resp, dict):
            self._last_row = resp
            self._last_rows = [resp]
        elif isinstance(resp, list):
            self._last_row = resp[0] if resp else None
            self._last_rows = resp
        else:                                    # pragma: no cover
            raise TypeError(f"Unsupported response type: {type(resp)!r}")

    def fetchone(self):
        return self._last_row

    def fetchall(self):
        return self._last_rows


class _FakeCtx:
    def __init__(self, cursor):
        self._cursor = cursor
    def __enter__(self):
        return self._cursor
    def __exit__(self, *a):
        pass


def _patch_sync_cursor(monkeypatch, target_module, responses):
    """Patch `sync_cursor` in `target_module` to yield a scripted cursor."""
    cursor = _FakeCursor(responses)
    monkeypatch.setattr(target_module, "sync_cursor", lambda: _FakeCtx(cursor))
    return cursor


# ============================================================================
# Resolver four-state distinction
# ============================================================================
def test_resolver_state_empty_table(monkeypatch):
    """When the synonym table is empty AND both passes miss, return
    EMPTY_TABLE -- distinct from UNRESOLVED so telemetry / inspection
    can tell deployment-prerequisite failures apart from real
    vocabulary misses.

    Round-35 query order: exact -> fuzzy -> presence check. The
    presence check only fires when both passes return nothing.
    """
    from skillbridge.match import occupation
    _patch_sync_cursor(monkeypatch, occupation, responses=[
        None,        # exact-match query (table empty -> no rows)
        None,        # fuzzy query (table empty -> no rows)
        None,        # presence check (table empty -> None)
    ])
    res = resolve_title_to_noc_with_state("Software Developer")
    assert res.state == NocResolutionState.EMPTY_TABLE
    assert res.noc_code is None
    assert res.similarity == 0.0


def test_resolver_state_exact_uses_single_query(monkeypatch):
    """Round-35 fix-3: exact resolution must not run the presence
    check. The successful exact path is ONE query."""
    from skillbridge.match import occupation
    cursor = _patch_sync_cursor(monkeypatch, occupation, responses=[
        {"noc_code": "21232"},                            # exact match
    ])
    res = resolve_title_to_noc_with_state("Software Developer")
    assert res.state == NocResolutionState.EXACT
    assert res.noc_code == "21232"
    assert res.similarity == 1.0
    # Pin the hot-path query count: ONE query for exact resolution.
    assert len(cursor.captured_sql) == 1


def test_resolver_state_fuzzy_uses_two_queries(monkeypatch):
    """Round-35: fuzzy resolution = exact miss + fuzzy hit = 2 queries."""
    from skillbridge.match import occupation
    cursor = _patch_sync_cursor(monkeypatch, occupation, responses=[
        None,                                              # no exact hit
        {"noc_code": "21232", "sim": 0.78},                # trigram match
    ])
    res = resolve_title_to_noc_with_state("Sofwtare Developr")
    assert res.state == NocResolutionState.FUZZY
    assert res.noc_code == "21232"
    assert res.similarity == pytest.approx(0.78)
    assert res.candidate_similarity == pytest.approx(0.78)
    assert len(cursor.captured_sql) == 2


def test_resolver_state_unresolved_with_candidate_similarity_recorded(monkeypatch):
    """Genuine vocabulary miss: table is populated, no exact match,
    trigram returns a close-but-below-threshold candidate. Record the
    candidate's score so the inspection command can surface
    "almost there" titles.

    Round-35 honesty rename: the field is `candidate_similarity`
    because pg_trgm's `%` operator filters rows below its session
    threshold (default 0.3) before they reach Python -- we don't see
    the true corpus best, only the best candidate that surfaced.
    """
    from skillbridge.match import occupation
    _patch_sync_cursor(monkeypatch, occupation, responses=[
        None,                                              # exact miss
        {"noc_code": "21232", "sim": 0.45},                # below 0.6 threshold
        {"presence": 1},                                   # table populated
    ])
    res = resolve_title_to_noc_with_state("ssm java guy")
    assert res.state == NocResolutionState.UNRESOLVED
    assert res.noc_code is None
    assert res.similarity == 0.0
    assert res.candidate_similarity == pytest.approx(0.45)


def test_resolver_state_unresolved_no_trigram_hits_at_all(monkeypatch):
    from skillbridge.match import occupation
    _patch_sync_cursor(monkeypatch, occupation, responses=[
        None,                                              # no exact
        None,                                              # no trigram
        {"presence": 1},                                   # table populated
    ])
    res = resolve_title_to_noc_with_state("zzzz nothing match")
    assert res.state == NocResolutionState.UNRESOLVED
    assert res.candidate_similarity == 0.0


def test_resolver_empty_input_short_circuits_without_db(monkeypatch):
    """Empty / whitespace input must return UNRESOLVED without
    touching the DB (defensive; matches existing behavior)."""
    from skillbridge.match import occupation
    # Cursor with no responses -- if DB IS touched, fetchone returns
    # None and we'd see EMPTY_TABLE; we expect UNRESOLVED instead.
    sentinel = []
    monkeypatch.setattr(
        occupation, "sync_cursor",
        lambda: (_ for _ in ()).throw(AssertionError("DB should not be touched")),
    )
    for empty in ("", "   ", None):
        res = resolve_title_to_noc_with_state(empty)
        assert res.state == NocResolutionState.UNRESOLVED


# ============================================================================
# Backward-compat wrappers
# ============================================================================
def test_resolve_title_to_noc_returns_just_the_code(monkeypatch):
    from skillbridge.match import occupation
    _patch_sync_cursor(monkeypatch, occupation, responses=[
        {"noc_code": "72410"},                            # exact match
    ])
    assert resolve_title_to_noc("Automotive Service Technician") == "72410"


def test_resolve_title_to_noc_with_score_returns_tuple(monkeypatch):
    from skillbridge.match import occupation
    _patch_sync_cursor(monkeypatch, occupation, responses=[
        None,                                              # exact miss
        {"noc_code": "72410", "sim": 0.92},                # fuzzy hit
    ])
    noc, sim = resolve_title_to_noc_with_score("Auto Tech")
    assert noc == "72410"
    assert sim == pytest.approx(0.92)


# ============================================================================
# Miss telemetry -- log on UNRESOLVED only, PII-free
# ============================================================================
def test_miss_telemetry_logs_only_on_unresolved(monkeypatch, caplog):
    """Successful exact / fuzzy matches do NOT log. Only real
    vocabulary misses produce a `noc_resolver_miss` line.

    Round-35 query order: exact -> fuzzy -> presence check.
    """
    from skillbridge.match import occupation
    _patch_sync_cursor(monkeypatch, occupation, responses=[
        # turn 1: exact match  (1 query)
        {"noc_code": "21232"},
        # turn 2: fuzzy match  (exact miss + fuzzy hit = 2 queries)
        None, {"noc_code": "21232", "sim": 0.8},
        # turn 3: unresolved   (exact miss + fuzzy miss + presence yes = 3)
        None, None, {"presence": 1},
    ])
    with caplog.at_level(logging.INFO, logger="skillbridge.match.occupation.miss"):
        resolve_title_to_noc_with_state(
            "Software Developer", call_source="job_backfill",
        )
        resolve_title_to_noc_with_state(
            "Sofwtare Dev", call_source="job_backfill",
        )
        resolve_title_to_noc_with_state(
            "ssm random title", call_source="job_backfill",
        )
    miss_lines = [
        rec.getMessage() for rec in caplog.records
        if rec.name == "skillbridge.match.occupation.miss"
    ]
    assert len(miss_lines) == 1
    assert "noc_resolver_miss" in miss_lines[0]
    assert "ssm random title" in miss_lines[0]


def test_miss_telemetry_contains_call_source(monkeypatch, caplog):
    from skillbridge.match import occupation
    _patch_sync_cursor(monkeypatch, occupation, responses=[
        None, None, {"presence": 1},                       # unresolved, table populated
    ])
    with caplog.at_level(logging.INFO, logger="skillbridge.match.occupation.miss"):
        resolve_title_to_noc_with_state(
            "totally novel role", call_source="job_backfill",
        )
    miss_lines = [
        rec.getMessage() for rec in caplog.records
        if rec.name == "skillbridge.match.occupation.miss"
    ]
    assert len(miss_lines) == 1
    assert "call_source=job_backfill" in miss_lines[0]


def test_miss_telemetry_caps_long_titles_on_job_backfill(monkeypatch, caplog):
    """job_backfill side: the normalized title is length-capped at
    80 chars so log lines stay bounded for pathological inputs.
    (user_target side never logs the title at all -- only the hash --
    so the cap is moot there.)"""
    from skillbridge.match import occupation
    _patch_sync_cursor(monkeypatch, occupation, responses=[
        None, None, {"presence": 1},
    ])
    long_title = "extremely " * 30 + "long title"   # ~300 chars
    with caplog.at_level(logging.INFO, logger="skillbridge.match.occupation.miss"):
        resolve_title_to_noc_with_state(
            long_title, call_source="job_backfill",
        )
    miss_lines = [
        rec.getMessage() for rec in caplog.records
        if rec.name == "skillbridge.match.occupation.miss"
    ]
    assert miss_lines
    # The logged title portion must be <= 80 characters.
    import re
    m = re.search(r"title='([^']*)'", miss_lines[0])
    assert m is not None, f"could not find quoted title in: {miss_lines[0]!r}"
    assert len(m.group(1)) <= 80


def test_miss_telemetry_does_not_log_on_empty_table(monkeypatch, caplog):
    """EMPTY_TABLE is a structural failure, not a vocabulary miss --
    it should NOT add noise to the misses backlog."""
    from skillbridge.match import occupation
    _patch_sync_cursor(monkeypatch, occupation, responses=[
        None, None, None,                                  # all three queries empty
    ])
    with caplog.at_level(logging.INFO, logger="skillbridge.match.occupation.miss"):
        resolve_title_to_noc_with_state("anything")
    miss_lines = [
        rec.getMessage() for rec in caplog.records
        if rec.name == "skillbridge.match.occupation.miss"
    ]
    assert miss_lines == []


def test_miss_telemetry_does_not_log_on_job_backfill_pii_markers(monkeypatch, caplog):
    """The job_backfill log line MUST NOT carry profile / session /
    resume data."""
    from skillbridge.match import occupation
    _patch_sync_cursor(monkeypatch, occupation, responses=[
        None, None, {"presence": 1},
    ])
    with caplog.at_level(logging.INFO, logger="skillbridge.match.occupation.miss"):
        resolve_title_to_noc_with_state(
            "missed role", call_source="job_backfill",
        )
    rec = next(
        r for r in caplog.records
        if r.name == "skillbridge.match.occupation.miss"
    )
    msg = rec.getMessage()
    for pii_marker in (
        "profile_id", "session_id", "resume", "skill_id", "email", "phone",
    ):
        assert pii_marker not in msg


# ============================================================================
# Round-36: user_target logs ONLY length / token count / score
# ============================================================================
# Round-35 emitted a SHA-256 fingerprint. The reviewer correctly
# noted that SHA-256 over a constrained input space (English job
# titles + common email patterns) is dictionary-recoverable, so a
# raw hash offers no real protection. Round-36: no fingerprint at
# all. length + token_count + score are the only fields. A future
# iteration can add keyed HMAC if the operator wants the
# fingerprint signal back.
def test_user_target_miss_logs_length_and_token_count_only(monkeypatch, caplog):
    """user_target misses log NO derived fingerprint -- only length,
    token count, and score. The raw input is never logged in any
    encoded form."""
    from skillbridge.match import occupation
    _patch_sync_cursor(monkeypatch, occupation, responses=[
        None, None, {"presence": 1},
    ])
    pii_input = "bob smith bob@example.com"
    with caplog.at_level(logging.INFO, logger="skillbridge.match.occupation.miss"):
        resolve_title_to_noc_with_state(
            pii_input, call_source="user_target",
        )
    rec = next(
        r for r in caplog.records
        if r.name == "skillbridge.match.occupation.miss"
    )
    msg = rec.getMessage()
    # The raw user text MUST NOT appear
    assert "bob smith" not in msg.lower()
    assert "bob@example.com" not in msg.lower()
    assert "@" not in msg
    # No derived fingerprint of any kind
    assert "fingerprint" not in msg.lower()
    # length + token_count + score MUST appear
    assert "length=" in msg
    assert "token_count=" in msg
    assert "candidate_score=" in msg
    # Token count is len(needle.split()) -- 3 tokens here
    import re
    m_len = re.search(r"length=(\d+)", msg)
    m_tok = re.search(r"token_count=(\d+)", msg)
    assert m_len is not None and m_tok is not None
    assert int(m_len.group(1)) == len(pii_input.strip().lower())
    assert int(m_tok.group(1)) == 3


def test_user_target_miss_never_emits_a_fingerprint_field(monkeypatch, caplog):
    """Negative pin (round-36): the user_target branch must not emit
    any fingerprint / hash / digest field even when a future patch
    might add one back. This locks the contract that the only
    user_target signals are length / token_count / score."""
    from skillbridge.match import occupation
    _patch_sync_cursor(monkeypatch, occupation, responses=[
        None, None, {"presence": 1},
    ])
    with caplog.at_level(logging.INFO, logger="skillbridge.match.occupation.miss"):
        resolve_title_to_noc_with_state(
            "looking for warehouse work in steeltown",
            call_source="user_target",
        )
    rec = next(
        r for r in caplog.records
        if r.name == "skillbridge.match.occupation.miss"
    )
    msg = rec.getMessage().lower()
    for forbidden in ("fingerprint", "hash", "digest", "sha", "hmac"):
        assert forbidden not in msg, (
            f"user_target miss line must not include {forbidden!r}; "
            f"got {msg!r}"
        )


def test_user_target_miss_does_not_emit_raw_title_field(monkeypatch, caplog):
    """Negative pin: the user_target branch must NOT emit `title=...`
    in any form. The branch lives in `_log_miss`; this test guards
    a future "small change" from accidentally re-introducing raw
    user text."""
    from skillbridge.match import occupation
    _patch_sync_cursor(monkeypatch, occupation, responses=[
        None, None, {"presence": 1},
    ])
    with caplog.at_level(logging.INFO, logger="skillbridge.match.occupation.miss"):
        resolve_title_to_noc_with_state(
            "looking for warehouse work in steeltown",
            call_source="user_target",
        )
    rec = next(
        r for r in caplog.records
        if r.name == "skillbridge.match.occupation.miss"
    )
    msg = rec.getMessage()
    assert "title=" not in msg, (
        f"user_target miss line must not include raw title=...; "
        f"got {msg!r}"
    )


# ============================================================================
# Inspection command -- structural-failure detection
# ============================================================================
def test_inspect_detects_empty_synonym_table_as_structural_failure(monkeypatch):
    """Empty synonym table = deployment-prerequisite failure. Surfaces
    in `structural_failures` and exits nonzero under --strict."""
    from skillbridge.match import inspect_noc
    _patch_sync_cursor(monkeypatch, inspect_noc, responses=[
        {"n": 5},                                          # occupation_count
        {"n": 0},                                          # synonym_count_total
        # synonym_count_by_source_lang skipped because total=0
        {"n": 0},                                          # active_jobs_total
        {"n": 0},                                          # active_jobs_with_noc
        [],                                                # missing_titles fetchall
    ])
    report = inspect_noc_coverage()
    assert "empty_synonym_table" in report.structural_failures
    assert report.synonym_count_total == 0


def test_inspect_detects_empty_occupation_table_as_structural_failure(monkeypatch):
    from skillbridge.match import inspect_noc
    _patch_sync_cursor(monkeypatch, inspect_noc, responses=[
        {"n": 0},                                          # occupation_count
        {"n": 0},                                          # synonym_count_total
        {"n": 0},                                          # active_jobs_total
        {"n": 0},                                          # active_jobs_with_noc
        [],
    ])
    report = inspect_noc_coverage()
    assert "empty_occupation_table" in report.structural_failures
    assert "empty_synonym_table" in report.structural_failures


def test_inspect_populated_table_no_structural_failures(monkeypatch):
    from skillbridge.match import inspect_noc
    _patch_sync_cursor(monkeypatch, inspect_noc, responses=[
        {"n": 500},                                        # occupation_count
        {"n": 7000},                                       # synonym_count_total
        [{"source": "oasis_example", "lang": "en", "n": 4000},
         {"source": "oasis_example", "lang": "fr", "n": 2500},
         {"source": "sct_alternative", "lang": "en", "n": 500}],
        {"n": 100},                                        # active_jobs_total
        {"n": 80},                                         # active_jobs_with_noc
        [],                                                # missing_titles -> none
    ])
    report = inspect_noc_coverage()
    assert report.structural_failures == []
    assert report.synonym_count_by_source_lang == [
        {"source": "oasis_example", "lang": "en", "count": 4000},
        {"source": "oasis_example", "lang": "fr", "count": 2500},
        {"source": "sct_alternative", "lang": "en", "count": 500},
    ]


# ============================================================================
# Inspection command -- coverage math
# ============================================================================
def test_inspect_coverage_pct_math(monkeypatch):
    """coverage_pct = active_jobs_with_noc / active_jobs_total * 100."""
    from skillbridge.match import inspect_noc
    _patch_sync_cursor(monkeypatch, inspect_noc, responses=[
        {"n": 10},
        {"n": 100},
        [],                                                # source/lang skipped because we don't reach it... wait we DO if total>0
        {"n": 200},                                        # active_jobs_total
        {"n": 150},                                        # active_jobs_with_noc
        [],                                                # missing_titles
    ])
    report = inspect_noc_coverage()
    assert report.coverage_pct == 75.0


def test_inspect_coverage_zero_when_no_active_jobs(monkeypatch):
    from skillbridge.match import inspect_noc
    _patch_sync_cursor(monkeypatch, inspect_noc, responses=[
        {"n": 10},
        {"n": 100},
        [],
        {"n": 0},
        {"n": 0},
        [],
    ])
    report = inspect_noc_coverage()
    assert report.coverage_pct == 0.0
    assert report.active_jobs_total == 0


# ============================================================================
# Inspection command -- truly-unresolved grouping
# ============================================================================
def test_inspect_groups_truly_unresolved_by_normalized_title(monkeypatch):
    """Round-35 fix-4: missing-NOC titles are grouped by SQL
    `LOWER(TRIM(title))` and resolved ONCE per unique title. The
    `count` field carries the duplicate multiplier so a title
    appearing 2 times still shows count=2 in the report."""
    from skillbridge.match import inspect_noc
    from skillbridge.match import occupation
    _patch_sync_cursor(monkeypatch, inspect_noc, responses=[
        {"n": 5},                                          # occupation_count
        {"n": 100},                                        # synonym_count
        [],                                                # source/lang
        {"n": 3},                                          # active_jobs_total
        {"n": 0},                                          # active_jobs_with_noc
        # GROUP BY result: SQL aggregates the 3 jobs into 2 groups.
        [{"norm_title": "honda service tech", "n": 2},
         {"norm_title": "some other title", "n": 1}],
    ])
    # Resolver returns UNRESOLVED for both groups.
    resolver_call_counts: dict[str, int] = {}
    def resolver(title, *, call_source="user_target", emit_telemetry=True):
        resolver_call_counts[title] = resolver_call_counts.get(title, 0) + 1
        return NocResolution(
            state=NocResolutionState.UNRESOLVED,
            noc_code=None, similarity=0.0,
            candidate_similarity=0.32 if "honda" in title.lower() else 0.10,
        )
    monkeypatch.setattr(occupation, "resolve_title_to_noc_with_state", resolver)
    monkeypatch.setattr(inspect_noc, "resolve_title_to_noc_with_state", resolver)

    report = inspect_noc_coverage()
    groups = {g.normalized_title: g for g in report.truly_unresolved_titles}
    # The duplicate "honda service tech" hits the resolver ONCE.
    assert resolver_call_counts["honda service tech"] == 1
    # ...but the count in the report still reflects the 2 duplicates.
    assert groups["honda service tech"].count == 2
    assert groups["honda service tech"].candidate_similarity == pytest.approx(0.32)
    assert groups["some other title"].count == 1
    assert report.missing_noc_jobs_resolvable_now == 0


def test_inspect_counts_resolvable_now_separately(monkeypatch):
    """Missing-NOC jobs whose resolver NOW returns EXACT/FUZZY are
    counted in `missing_noc_jobs_resolvable_now`. With SQL grouping,
    `count` is multiplied into the resolvable tally so 3 duplicates
    of a now-resolvable title count as 3."""
    from skillbridge.match import inspect_noc
    from skillbridge.match import occupation
    _patch_sync_cursor(monkeypatch, inspect_noc, responses=[
        {"n": 5}, {"n": 100}, [], {"n": 5}, {"n": 0},
        # 3 duplicates of "resolvable", 2 of "not resolvable"
        [{"norm_title": "resolvable", "n": 3},
         {"norm_title": "not resolvable", "n": 2}],
    ])
    def resolver(title, *, call_source="user_target", emit_telemetry=True):
        if "resolvable" in title.lower() and "not" not in title.lower():
            return NocResolution(
                state=NocResolutionState.EXACT,
                noc_code="11111", similarity=1.0, candidate_similarity=1.0,
            )
        return NocResolution(
            state=NocResolutionState.UNRESOLVED, noc_code=None,
            similarity=0.0, candidate_similarity=0.0,
        )
    monkeypatch.setattr(occupation, "resolve_title_to_noc_with_state", resolver)
    monkeypatch.setattr(inspect_noc, "resolve_title_to_noc_with_state", resolver)
    report = inspect_noc_coverage()
    # 3 duplicates of "resolvable" -> 3 resolvable-now (not 1)
    assert report.missing_noc_jobs_resolvable_now == 3
    assert len(report.truly_unresolved_titles) == 1
    assert report.truly_unresolved_titles[0].normalized_title == "not resolvable"
    assert report.truly_unresolved_titles[0].count == 2


def test_inspect_audit_suppresses_resolver_telemetry(monkeypatch):
    """Round-35 fix-4: the audit's resolver calls must pass
    `emit_telemetry=False`. The report itself is the canonical
    record; the miss log shouldn't be flooded by the audit."""
    from skillbridge.match import inspect_noc
    from skillbridge.match import occupation
    _patch_sync_cursor(monkeypatch, inspect_noc, responses=[
        {"n": 5}, {"n": 100}, [], {"n": 2}, {"n": 0},
        [{"norm_title": "unknown a", "n": 1},
         {"norm_title": "unknown b", "n": 1}],
    ])
    seen_telemetry_flags: list[bool] = []
    def resolver(title, *, call_source="user_target", emit_telemetry=True):
        seen_telemetry_flags.append(emit_telemetry)
        return NocResolution(
            state=NocResolutionState.UNRESOLVED,
            noc_code=None, similarity=0.0, candidate_similarity=0.0,
        )
    monkeypatch.setattr(occupation, "resolve_title_to_noc_with_state", resolver)
    monkeypatch.setattr(inspect_noc, "resolve_title_to_noc_with_state", resolver)
    inspect_noc_coverage()
    # Every audit resolution must pass emit_telemetry=False.
    assert seen_telemetry_flags == [False, False]


def test_inspect_empty_table_titles_never_appear_in_truly_unresolved(monkeypatch):
    """Round-36 fix-2: when the synonym table is empty, the resolver
    returns EMPTY_TABLE for every title. Those titles were NEVER
    ASSESSED -- they're not vocabulary misses. The
    `truly_unresolved_titles` list MUST be empty; the structural
    failure already documents the empty-table condition.

    Pre-fix, the audit collapsed EMPTY_TABLE and UNRESOLVED into the
    same list and broke the four-state contract."""
    from skillbridge.match import inspect_noc
    from skillbridge.match import occupation
    _patch_sync_cursor(monkeypatch, inspect_noc, responses=[
        {"n": 0},                                          # occupation_count
        {"n": 0},                                          # synonym_count (empty!)
        # source/lang skipped because synonym_count == 0
        {"n": 3},                                          # active_jobs_total
        {"n": 0},                                          # active_jobs_with_noc
        # 2 missing-NOC titles grouped
        [{"norm_title": "honda service tech", "n": 2},
         {"norm_title": "some other title", "n": 1}],
    ])
    # Every resolution returns EMPTY_TABLE because the synonyms are
    # missing.
    monkeypatch.setattr(
        occupation, "resolve_title_to_noc_with_state",
        lambda t, *, call_source="user_target", emit_telemetry=True: NocResolution(
            state=NocResolutionState.EMPTY_TABLE,
            noc_code=None, similarity=0.0, candidate_similarity=0.0,
        ),
    )
    monkeypatch.setattr(
        inspect_noc, "resolve_title_to_noc_with_state",
        occupation.resolve_title_to_noc_with_state,
    )

    report = inspect_noc_coverage()
    # The four-state contract: EMPTY_TABLE titles never enter
    # `truly_unresolved_titles`. The empty-table condition is
    # recorded once in structural_failures instead.
    assert report.truly_unresolved_titles == []
    assert "empty_synonym_table" in report.structural_failures
    # They're also NOT counted as resolvable.
    assert report.missing_noc_jobs_resolvable_now == 0


def test_inspect_only_unresolved_state_enters_truly_unresolved(monkeypatch):
    """Defense-in-depth: mixed states across the audit's grouped
    titles -- the report MUST surface ONLY the UNRESOLVED rows.
    EXACT/FUZZY go into resolvable_now; EMPTY_TABLE goes nowhere."""
    from skillbridge.match import inspect_noc
    from skillbridge.match import occupation
    _patch_sync_cursor(monkeypatch, inspect_noc, responses=[
        {"n": 100}, {"n": 5000},
        # The SQL aliases COUNT(*) AS n; the report wraps to count.
        [{"source": "oasis_example", "lang": "en", "n": 5000}],
        {"n": 10}, {"n": 0},
        [{"norm_title": "exact title", "n": 3},
         {"norm_title": "fuzzy title", "n": 2},
         {"norm_title": "unresolved title", "n": 4},
         {"norm_title": "empty table title", "n": 1}],
    ])
    def resolver(title, *, call_source="user_target", emit_telemetry=True):
        if title == "exact title":
            return NocResolution(NocResolutionState.EXACT, "11111", 1.0, 1.0)
        if title == "fuzzy title":
            return NocResolution(NocResolutionState.FUZZY, "22222", 0.85, 0.85)
        if title == "unresolved title":
            return NocResolution(NocResolutionState.UNRESOLVED, None, 0.0, 0.40)
        # empty table title -- represents the EMPTY_TABLE state being
        # injected mid-audit (e.g. a race condition or a partially-
        # populated table). The audit must still classify it as
        # unassessed.
        return NocResolution(NocResolutionState.EMPTY_TABLE, None, 0.0, 0.0)
    monkeypatch.setattr(occupation, "resolve_title_to_noc_with_state", resolver)
    monkeypatch.setattr(inspect_noc, "resolve_title_to_noc_with_state", resolver)

    report = inspect_noc_coverage()
    # Only the UNRESOLVED title makes it into the list.
    normalized = [g.normalized_title for g in report.truly_unresolved_titles]
    assert normalized == ["unresolved title"]
    assert report.truly_unresolved_titles[0].count == 4
    # Exact + Fuzzy combined into resolvable_now (3 + 2 = 5)
    assert report.missing_noc_jobs_resolvable_now == 5


def test_resolver_emit_telemetry_false_suppresses_miss_log(monkeypatch, caplog):
    """Round-35 fix-4: the resolver respects `emit_telemetry=False`
    and does NOT log a miss line for the caller. Distinct from
    `_log_miss` (which is the unconditional emitter)."""
    from skillbridge.match import occupation
    _patch_sync_cursor(monkeypatch, occupation, responses=[
        None, None, {"presence": 1},
    ])
    with caplog.at_level(logging.INFO, logger="skillbridge.match.occupation.miss"):
        resolve_title_to_noc_with_state(
            "missed role", call_source="job_backfill",
            emit_telemetry=False,
        )
    miss_lines = [
        rec.getMessage() for rec in caplog.records
        if rec.name == "skillbridge.match.occupation.miss"
    ]
    assert miss_lines == []


# ============================================================================
# CLI exit codes
# ============================================================================
def test_cli_default_mode_returns_zero_on_structural_failure(monkeypatch):
    """Without --strict, even structural failures exit 0 -- the
    audit is informational by default."""
    from skillbridge.match import inspect_noc
    _patch_sync_cursor(monkeypatch, inspect_noc, responses=[
        {"n": 0}, {"n": 0}, {"n": 0}, {"n": 0}, [],
    ])
    assert cli_inspect_noc_coverage(strict=False) == 0


def test_cli_strict_mode_returns_nonzero_on_structural_failure(monkeypatch):
    """Production-readiness check: empty synonym/occupation tables
    fail the gate."""
    from skillbridge.match import inspect_noc
    _patch_sync_cursor(monkeypatch, inspect_noc, responses=[
        {"n": 0}, {"n": 0}, {"n": 0}, {"n": 0}, [],
    ])
    assert cli_inspect_noc_coverage(strict=True) == 1


def test_cli_strict_mode_returns_zero_on_low_coverage(monkeypatch):
    """Low coverage with populated tables MUST exit 0 even under
    --strict. Coverage is a vocabulary signal, not a structural one;
    the locked design says it stays a reported metric."""
    from skillbridge.match import inspect_noc
    from skillbridge.match import occupation
    _patch_sync_cursor(monkeypatch, inspect_noc, responses=[
        {"n": 100},                                        # occupation_count
        {"n": 5000},                                       # synonym_count
        [{"source": "oasis_example", "lang": "en", "n": 5000}],
        {"n": 100},                                        # active_jobs_total
        {"n": 1},                                          # active_jobs_with_noc
        # Round-35 audit shape: grouped (norm_title, n).
        [{"norm_title": f"missed {i}", "n": 1} for i in range(99)],
    ])
    monkeypatch.setattr(
        occupation, "resolve_title_to_noc_with_state",
        lambda t, *, call_source="user_target", emit_telemetry=True: NocResolution(
            state=NocResolutionState.UNRESOLVED,
            noc_code=None, similarity=0.0, candidate_similarity=0.0,
        ),
    )
    monkeypatch.setattr(
        inspect_noc, "resolve_title_to_noc_with_state",
        occupation.resolve_title_to_noc_with_state,
    )
    # 1 of 100 -> 1.0% coverage. No structural failures.
    assert cli_inspect_noc_coverage(strict=True) == 0


# ============================================================================
# Report rendering -- stable text format
# ============================================================================
def test_format_report_includes_all_required_metrics():
    """The pretty-printed report MUST mention every metric the locked
    scope requires."""
    report = NocCoverageReport(
        occupation_count=500,
        synonym_count_total=7000,
        synonym_count_by_source_lang=[
            {"source": "oasis_example", "lang": "en", "count": 4000},
        ],
        active_jobs_total=100, active_jobs_with_noc=80,
        active_jobs_missing_noc=20,
        missing_noc_jobs_resolvable_now=5,
        truly_unresolved_titles=[
            UnresolvedTitleGroup(
                normalized_title="honda service tech",
                count=3, candidate_similarity=0.42,
            ),
        ],
        coverage_pct=80.0,
        structural_failures=[],
    )
    txt = format_noc_coverage_report(report)
    # Required metrics from the locked scope
    assert "Occupation rows:" in txt
    assert "Synonym rows" in txt
    assert "Active jobs (total):" in txt
    assert "Active jobs with NOC:" in txt
    assert "Active jobs missing NOC:" in txt
    assert "Missing-NOC jobs resolvable:" in txt
    assert "Coverage:" in txt
    assert "80.0%" in txt
    assert "honda service tech" in txt
    assert "0.42" in txt


def test_format_report_flags_structural_failures():
    report = NocCoverageReport(
        structural_failures=["empty_synonym_table"],
    )
    txt = format_noc_coverage_report(report)
    assert "STRUCTURAL FAILURES" in txt
    assert "empty_synonym_table" in txt


# ============================================================================
# Loader idempotency
# ============================================================================
def test_load_oasis_occupation_titles_no_csvs_returns_zero(tmp_path):
    """Idempotency / robustness: missing CSV files yield 0 inserts
    without raising. The current behavior keeps the pipeline runnable
    even before the operator downloads OaSIS data."""
    from skillbridge.ingest import reference as ref
    # NOTE: the loader's default args bind at function-def time, so
    # we pass paths explicitly rather than monkeypatching module
    # constants (which would be a no-op).
    inserted = ref.load_oasis_occupation_titles(
        en_path=tmp_path / "missing_en.csv",
        fr_path=tmp_path / "missing_fr.csv",
        sct_path=tmp_path / "missing_sct.csv",
    )
    assert inserted == 0


def test_load_oasis_occupation_titles_uses_on_conflict_do_nothing(monkeypatch, tmp_path):
    """The loader's UPSERT must use ON CONFLICT DO NOTHING so re-runs
    don't duplicate synonym rows. Validate by inspecting captured
    SQL."""
    from skillbridge.ingest import reference as ref
    # Provide a minimal CSV so the loader has something to insert.
    en_csv = tmp_path / "en.csv"
    en_csv.write_text(
        "noc_code,title\n21232,Software Developer\n",
        encoding="utf-8",
    )
    captured: list[str] = []
    class _Cur:
        # OASIS-FIX: production now reads `cur.rowcount` for honest
        # insert counts. The stub returns 0 because this test only
        # asserts SQL shape, not insert counts.
        rowcount = 0
        def execute(self, sql, params=()):
            captured.append(sql)
        def fetchone(self): return None
        def fetchall(self): return []
    class _Ctx:
        def __enter__(self): return _Cur()
        def __exit__(self, *a): pass
    monkeypatch.setattr(ref, "sync_cursor", lambda: _Ctx())

    ref.load_oasis_occupation_titles(
        en_path=en_csv,
        fr_path=tmp_path / "missing_fr.csv",
        sct_path=tmp_path / "missing_sct.csv",
    )
    # The synonym INSERT must use ON CONFLICT DO NOTHING (idempotency)
    synonym_inserts = [s for s in captured if "occupation_title_synonym" in s]
    assert synonym_inserts, "expected at least one synonym INSERT"
    assert any("ON CONFLICT" in s and "DO NOTHING" in s for s in synonym_inserts)


def test_load_oasis_occupation_titles_stamps_oasis_version(monkeypatch, tmp_path):
    """Idempotency goes hand-in-hand with version stamping: the
    occupation upsert MUST update `oasis_version` so re-runs against
    a newer CSV bump every row's stamp."""
    from skillbridge.ingest import reference as ref
    en_csv = tmp_path / "en.csv"
    en_csv.write_text(
        "noc_code,title\n21232,Software Developer\n",
        encoding="utf-8",
    )
    captured: list[tuple[str, tuple]] = []
    class _Cur:
        # OASIS-FIX: production now reads `cur.rowcount` for honest
        # insert counts. The stub returns 0 because this test only
        # asserts SQL shape, not insert counts.
        rowcount = 0
        def execute(self, sql, params=()):
            captured.append((sql, tuple(params) if params else ()))
        def fetchone(self): return None
        def fetchall(self): return []
    class _Ctx:
        def __enter__(self): return _Cur()
        def __exit__(self, *a): pass
    monkeypatch.setattr(ref, "sync_cursor", lambda: _Ctx())

    ref.load_oasis_occupation_titles(
        en_path=en_csv,
        fr_path=tmp_path / "missing_fr.csv",
        sct_path=tmp_path / "missing_sct.csv",
    )
    occupation_upserts = [s for s, _ in captured if "reference.occupation" in s
                          and "ON CONFLICT" in s and "oasis_version" in s.lower()]
    assert occupation_upserts, (
        "expected occupation upsert to stamp oasis_version on conflict"
    )


# =========================================================================
# OASIS-FIX: honest insert counts + keep-first lead-statement semantics
# =========================================================================
def test_load_oasis_occupation_titles_counts_rowcount_not_attempts(
    monkeypatch, tmp_path,
):
    """Honest insert count contract: `n` returned by the loader
    must reflect actual rows inserted (`cur.rowcount`), not loop
    iterations. A re-run against an already-populated DB (all rows
    hit ON CONFLICT) must report 0, not the full attempted count."""
    from skillbridge.ingest import reference as ref
    en_csv = tmp_path / "en.csv"
    en_csv.write_text(
        "noc_code,title\n21232,Software Developer\n21232,Programmer\n",
        encoding="utf-8",
    )

    class _Cur:
        # Simulate "every INSERT hits ON CONFLICT DO NOTHING" -- a
        # re-run against fully-populated state.
        rowcount = 0
        def execute(self, sql, params=()):
            pass
        def fetchone(self): return None
        def fetchall(self): return []
    class _Ctx:
        def __enter__(self): return _Cur()
        def __exit__(self, *a): pass
    monkeypatch.setattr(ref, "sync_cursor", lambda: _Ctx())

    n = ref.load_oasis_occupation_titles(
        en_path=en_csv,
        fr_path=tmp_path / "missing_fr.csv",
        sct_path=tmp_path / "missing_sct.csv",
    )
    assert n == 0, (
        "loader must report 0 new when every INSERT hit ON CONFLICT "
        "DO NOTHING; old `n += 1` reported the attempt count instead"
    )


def test_load_oasis_lead_statements_uses_keep_first_guard(
    monkeypatch, tmp_path,
):
    """Keep-first semantics: multiple OaSIS subprofiles (21232.00,
    21232.01) collapse to the same 5-digit NOC. An unconditional
    UPDATE would let each subsequent subprofile overwrite the
    previous statement (lossy). The `WHERE ... IS NULL` guard means
    the FIRST subprofile to land for a given NOC wins; later writes
    no-op."""
    from skillbridge.ingest import reference as ref
    en_csv = tmp_path / "en.csv"
    en_csv.write_text(
        "noc_code,lead_statement\n"
        "21232.00,First sentence.\n"
        "21232.01,Second sentence.\n",
        encoding="utf-8",
    )
    fr_csv = tmp_path / "fr.csv"
    fr_csv.write_text(
        "noc_code,lead_statement\n"
        "21232.00,Première phrase.\n",
        encoding="utf-8",
    )

    captured: list[str] = []
    class _Cur:
        rowcount = 1
        def execute(self, sql, params=()):
            captured.append(sql)
        def fetchone(self): return None
        def fetchall(self): return []
    class _Ctx:
        def __enter__(self): return _Cur()
        def __exit__(self, *a): pass
    monkeypatch.setattr(ref, "sync_cursor", lambda: _Ctx())

    ref.load_oasis_lead_statements(en_path=en_csv, fr_path=fr_csv)

    # Each UPDATE must include the `IS NULL` guard so the existing
    # value is preserved.
    en_updates = [s for s in captured if "lead_statement_en" in s]
    fr_updates = [s for s in captured if "lead_statement_fr" in s]
    assert en_updates, "expected at least one EN UPDATE"
    assert fr_updates, "expected at least one FR UPDATE"
    for sql in en_updates:
        assert "lead_statement_en IS NULL" in sql, (
            f"EN UPDATE missing keep-first guard: {sql!r}"
        )
    for sql in fr_updates:
        assert "lead_statement_fr IS NULL" in sql, (
            f"FR UPDATE missing keep-first guard: {sql!r}"
        )


def test_load_oasis_lead_statements_counts_en_and_fr_independently(
    monkeypatch, tmp_path,
):
    """The pre-fix FR loop did not increment `n`, so the returned
    count was EN-only. After the fix, `n` reflects the sum of EN +
    FR rowcounts."""
    from skillbridge.ingest import reference as ref
    en_csv = tmp_path / "en.csv"
    en_csv.write_text(
        "noc_code,lead_statement\n21232,EN statement.\n",
        encoding="utf-8",
    )
    fr_csv = tmp_path / "fr.csv"
    fr_csv.write_text(
        "noc_code,lead_statement\n21232,Énoncé FR.\n",
        encoding="utf-8",
    )

    class _Cur:
        rowcount = 1   # each UPDATE matches a row
        def execute(self, sql, params=()):
            pass
        def fetchone(self): return None
        def fetchall(self): return []
    class _Ctx:
        def __enter__(self): return _Cur()
        def __exit__(self, *a): pass
    monkeypatch.setattr(ref, "sync_cursor", lambda: _Ctx())

    n = ref.load_oasis_lead_statements(en_path=en_csv, fr_path=fr_csv)
    # 1 EN UPDATE (rowcount=1) + 1 FR UPDATE (rowcount=1) = 2
    assert n == 2, (
        f"expected n == 2 (1 EN + 1 FR rowcounts), got {n!r}; "
        f"pre-fix bug was the FR loop never incrementing n"
    )
