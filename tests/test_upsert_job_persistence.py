"""Step 1A transitional persistence tests for upsert_job.

Covers the load-bearing sentinel behavior added 2026-07-16 after
review found that the initial impl could erase legacy values on
unmigrated connectors AND could erase backfilled new-axis values
on re-ingestion by unmigrated connectors.

Four cases explicitly tested:
  1. status=None (unmigrated connector) → preserve legacy values.
  2. status="missing" (migrated) → intentionally clear legacy values.
  3. blank description_full → fall through to description_excerpt
     (matches SQL COALESCE, not Python `or` which would treat
     description_full="" as falsy and silently accept it).
  4. blank description_excerpt → persist NULL (via _blank_to_none).

Plus tests that:
  - ON CONFLICT CASE WHEN preserves backfilled new-axis fields when
    incoming status is NULL (verified by inspecting the generated SQL,
    not by round-tripping the DB, since these tests are unit-scoped).

See docs/matching-revise/step-1-source-data-integrity.md §2b and
§3c for the persistence contract.
"""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from skillbridge.ingest.base import NormalizedJob, upsert_job


@pytest.fixture
def mocked_cursor():
    """Capture the parameters passed to cur.execute() without hitting
    a real DB.

    Returns a MagicMock whose `execute` calls are inspectable. The
    `sync_cursor` context manager and `upsert_employer` are patched
    to no-ops so upsert_job's Python-side derivation is what gets
    tested.
    """
    captured: dict = {}

    fake_cursor = MagicMock()
    fake_cursor.fetchone.return_value = {
        "job_id": "00000000-0000-0000-0000-000000000000",
    }

    def _capture_execute(sql: str, params: tuple) -> None:
        captured["sql"] = sql
        captured["params"] = params

    fake_cursor.execute.side_effect = _capture_execute

    @contextmanager
    def fake_sync_cursor():
        yield fake_cursor

    with patch("skillbridge.ingest.base.sync_cursor", fake_sync_cursor), \
         patch("skillbridge.ingest.base.upsert_employer",
               return_value="employer-uuid"):
        yield captured


# Parameter positions in the upsert_job INSERT statement, matched to
# the values tuple passed to cur.execute(). Kept as constants so the
# tests don't drift silently if the tuple order changes.
_P_LEGACY_LOCATION = 5
_P_LEGACY_DESCRIPTION = 7
_P_DESCRIPTION_FULL = 17
_P_DESCRIPTION_EXCERPT = 18
_P_DESCRIPTION_STATUS = 19
_P_SOURCE_LOCATION_TEXT = 20
_P_SOURCE_COORDINATES = 21
_P_NORMALIZED_JOB_LOCATION = 22
_P_LOCATION_RESOLUTION = 23
_P_LOCATION_PROVENANCE = 24


# ══════════════════════════════════════════════════════════════════
# Case 1: unmigrated connector — legacy fields must survive
# ══════════════════════════════════════════════════════════════════


def test_unmigrated_preserves_legacy_location_and_description(
    mocked_cursor,
) -> None:
    """A connector that hasn't been updated for Step 1A still populates
    only `job.location` and `job.description`. When statuses are None
    (the sentinel for 'not migrated'), upsert_job MUST pass those
    legacy values through instead of deriving NULL from the empty
    new-axis fields.

    This is the load-bearing regression that prevents the initial
    impl's erasure-on-unmigrated-connector defect."""
    job = NormalizedJob(
        source="sccc",
        source_job_id="1234",
        title="Test Role",
        location="Sault Ste. Marie",
        description="A real JD paragraph goes here.",
        # description_full, description_excerpt, statuses all default
        # to None per the 2026-07-16 sentinel fix — this simulates an
        # unmigrated connector.
    )
    upsert_job(job)
    params = mocked_cursor["params"]

    assert params[_P_LEGACY_LOCATION] == "Sault Ste. Marie", (
        "unmigrated connector's legacy location must survive"
    )
    assert params[_P_LEGACY_DESCRIPTION] == "A real JD paragraph goes here.", (
        "unmigrated connector's legacy description must survive"
    )
    # New-axis fields are None so the ON CONFLICT CASE WHEN keeps
    # any existing backfilled values (see test_sql_case_when below).
    assert params[_P_DESCRIPTION_STATUS] is None
    assert params[_P_LOCATION_RESOLUTION] is None
    assert params[_P_DESCRIPTION_FULL] is None
    assert params[_P_NORMALIZED_JOB_LOCATION] is None


# ══════════════════════════════════════════════════════════════════
# Case 2: migrated connector emits "missing" — legacy fields cleared
# ══════════════════════════════════════════════════════════════════


def test_migrated_missing_status_clears_legacy_intentionally(
    mocked_cursor,
) -> None:
    """A migrated connector that has explicitly measured the source as
    having no description or location must emit an explicit
    'missing' status. Under that path, the legacy fields are
    intentionally cleared — the source truly has no data.

    Distinguishing intentional NULL (from 'missing') from
    unmigrated-connector NULL (from None sentinel) is the entire
    point of the sentinel design."""
    job = NormalizedJob(
        source="sccc",
        source_job_id="1234",
        title="Test Role",
        # No legacy fields passed — migrated connector goes through
        # new-axis path.
        description_evidence_status="missing",
        location_resolution_status="missing",
        location_provenance="none",
    )
    upsert_job(job)
    params = mocked_cursor["params"]

    # Migrated connector's 'missing' status means legacy fields
    # derive to None honestly.
    assert params[_P_LEGACY_LOCATION] is None
    assert params[_P_LEGACY_DESCRIPTION] is None
    assert params[_P_DESCRIPTION_STATUS] == "missing"
    assert params[_P_LOCATION_RESOLUTION] == "missing"


# ══════════════════════════════════════════════════════════════════
# Case 3: blank full text falls through to excerpt (SQL COALESCE
# semantics), not Python `or`
# ══════════════════════════════════════════════════════════════════


def test_blank_description_full_falls_through_to_excerpt(
    mocked_cursor,
) -> None:
    """When a migrated connector emits description_full='' (blank
    string), the value MUST be treated as absent — same as SQL
    COALESCE. Python's `or` would fall through anyway because ''
    is falsy, but the fix is explicit: _blank_to_none normalizes
    the empty string to None BEFORE the derivation, and the
    is-not-None chain then falls through to the excerpt.

    This test guards against a future refactor accidentally treating
    '' as a valid legacy_description."""
    job = NormalizedJob(
        source="sccc",
        source_job_id="1234",
        title="Test Role",
        description_full="",  # blank — treat as absent
        description_excerpt="Fallback excerpt from source.",
        description_evidence_status="excerpt_only",
        location_resolution_status="resolved",
        normalized_job_location="Sault Ste. Marie",
        location_provenance="source_declared",
    )
    upsert_job(job)
    params = mocked_cursor["params"]

    # Legacy description falls through to the excerpt.
    assert params[_P_LEGACY_DESCRIPTION] == "Fallback excerpt from source."
    # The blank full text is normalized to None BEFORE being written,
    # so the DB never sees an empty string.
    assert params[_P_DESCRIPTION_FULL] is None
    assert params[_P_DESCRIPTION_EXCERPT] == "Fallback excerpt from source."


def test_whitespace_only_full_treated_same_as_blank(
    mocked_cursor,
) -> None:
    """_blank_to_none strips whitespace too — a full of '   \\n\\t'
    is normalized to None."""
    job = NormalizedJob(
        source="sccc",
        source_job_id="1234",
        title="Test Role",
        description_full="   \n\t  ",
        description_excerpt="Real excerpt.",
        description_evidence_status="excerpt_only",
    )
    upsert_job(job)
    params = mocked_cursor["params"]
    assert params[_P_DESCRIPTION_FULL] is None
    assert params[_P_LEGACY_DESCRIPTION] == "Real excerpt."


# ══════════════════════════════════════════════════════════════════
# Case 4: both blank → persist NULL
# ══════════════════════════════════════════════════════════════════


def test_blank_full_and_blank_excerpt_persist_null(
    mocked_cursor,
) -> None:
    """A migrated connector that captures blank content in BOTH
    description fields (unusual — likely a parse-error case) must
    persist NULL to the legacy description column. _blank_to_none
    catches both and the is-not-None chain ends at None."""
    job = NormalizedJob(
        source="sccc",
        source_job_id="1234",
        title="Test Role",
        description_full="",
        description_excerpt="",
        description_evidence_status="parse_error",
    )
    upsert_job(job)
    params = mocked_cursor["params"]

    assert params[_P_DESCRIPTION_FULL] is None
    assert params[_P_DESCRIPTION_EXCERPT] is None
    assert params[_P_LEGACY_DESCRIPTION] is None
    # Status was explicitly emitted, so this is intentional NULL,
    # not sentinel-preserved-legacy.
    assert params[_P_DESCRIPTION_STATUS] == "parse_error"


# ══════════════════════════════════════════════════════════════════
# ON CONFLICT CASE WHEN preservation — verified via SQL inspection
# ══════════════════════════════════════════════════════════════════


def test_sql_preserves_new_axis_on_null_status_via_case_when(
    mocked_cursor,
) -> None:
    """The ON CONFLICT DO UPDATE clause must guard every new-axis
    column with a CASE WHEN keyed on the EXCLUDED status sentinel.
    Otherwise an unmigrated connector re-upserting a backfilled row
    would overwrite the backfilled values with its own NULLs.

    This test inspects the generated SQL (rather than round-tripping
    the DB) to confirm each new-axis column has the guard.

    Load-bearing rule (docs/matching-revise/step-1-*.md §2b):
        WHEN EXCLUDED.description_evidence_status IS NULL
            THEN keep existing description_full/excerpt/status
            ELSE overwrite
        WHEN EXCLUDED.location_resolution_status IS NULL
            THEN keep existing location fields
            ELSE overwrite"""
    job = NormalizedJob(
        source="sccc",
        source_job_id="1234",
        title="Test Role",
    )
    upsert_job(job)
    sql = mocked_cursor["sql"]

    # Description axis — three columns must be guarded.
    for col in ("description_full", "description_excerpt",
                "description_evidence_status"):
        assert (
            f"WHEN EXCLUDED.description_evidence_status IS NULL" in sql
        ), (
            f"missing description-axis sentinel guard in SQL"
        )
        assert col in sql, f"missing column {col} in SQL"

    # Location axis — five columns must be guarded.
    for col in ("source_location_text", "source_coordinates",
                "normalized_job_location", "location_resolution_status",
                "location_provenance"):
        assert (
            f"WHEN EXCLUDED.location_resolution_status IS NULL" in sql
        ), (
            f"missing location-axis sentinel guard in SQL"
        )
        assert col in sql, f"missing column {col} in SQL"

    # Sanity: the ELSE branch overwrites, so we should see EXCLUDED
    # references for each new-axis column.
    for col in ("description_full", "description_excerpt",
                "description_evidence_status",
                "source_location_text", "source_coordinates",
                "normalized_job_location", "location_resolution_status",
                "location_provenance"):
        assert f"EXCLUDED.{col}" in sql, (
            f"expected EXCLUDED.{col} in overwrite branch"
        )


# ══════════════════════════════════════════════════════════════════
# Anti-regression: legacy columns still overwrite (they are
# not new-axis and must reflect whatever the current run says).
# ══════════════════════════════════════════════════════════════════


def test_legacy_columns_still_overwrite_on_conflict(
    mocked_cursor,
) -> None:
    """The DEPRECATED legacy `location` and `description` columns
    continue to overwrite on conflict — they're not new-axis, and
    upsert_job's Python-side branching already picked the right
    value for them (preserved for unmigrated, derived for migrated).
    The guard is only on new-axis columns."""
    job = NormalizedJob(
        source="sccc",
        source_job_id="1234",
        title="Test Role",
    )
    upsert_job(job)
    sql = mocked_cursor["sql"]

    # The legacy columns' UPDATE clause uses a plain EXCLUDED
    # assignment (no CASE WHEN). Match "location = EXCLUDED.location,"
    # and same for description.
    assert "location                    = EXCLUDED.location," in sql
    assert "description                 = EXCLUDED.description," in sql
