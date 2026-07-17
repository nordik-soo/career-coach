"""Step 1A backfill tests (2026-07-15).

Covers:
  1. Latest-raw selection determinism (DISTINCT ON, ORDER BY
     ingested_at DESC, raw_id DESC — spec §3a).
  2. Re-normalization identity: backfill emits IDENTICAL new-axis
     values as live ingest on the same raw payload — connectors are
     the single source of classification truth.
  3. Idempotency: running backfill twice produces identical row
     state.
  4. Per-source dispatch: AWIC → _normalize_awic_geojson_feature,
     SCCC → _normalize_sccc_wp_rest_item, partner CSV → _row_to_job,
     partner upload → _row_to_job (upload variant).
  5. Employer connectors are counted as unsupported_source and NOT
     silently upserted with mock data.
  6. Shape-defect payloads (non-dict feature, non-dict item)
     produce shape_defect skips.
  7. Dry-run mode computes classifications but does not call
     upsert_job.

Tests are unit-scoped — no live DB round-trip. The `_iter_latest_
raw_payloads` DB read is mocked; `upsert_job` is mocked. The
`_renormalize` dispatch is exercised directly.
"""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest


# ══════════════════════════════════════════════════════════════════
# _renormalize dispatch — per-source
# ══════════════════════════════════════════════════════════════════


class TestRenormalizeDispatch:
    """Per-source dispatch routing. Verifies that the same normalize
    function powering live ingest is what backfill invokes."""

    def test_awic_payload_dispatches_to_awic_normalizer(self):
        from skillbridge.pipeline.step1a_backfill import _renormalize
        payload = {
            "source": "awic_geojson_v1",
            "feature": {
                "geometry": {
                    "type": "Point",
                    "coordinates": [-84.32, 46.54],
                },
                "properties": {
                    "post_id": 8301829,
                    "job_title": "Community Support Worker",
                    "excerpt": "SSM support role.",
                },
            },
        }
        job, skip = _renormalize("awic_jobs", "8301829", payload)
        assert skip is None
        assert job is not None
        assert job.source == "awic_jobs"
        # Load-bearing: AWIC never emits SSM as source_location_text.
        assert job.source_location_text is None
        # Geometry preserved, location unresolved.
        assert job.source_coordinates == [-84.32, 46.54]
        assert job.location_resolution_status == "unresolved"
        assert job.location_provenance == "geometry"
        # Excerpt present → excerpt_only.
        assert job.description_evidence_status == "excerpt_only"

    def test_sccc_payload_with_item_dispatches_to_full_normalizer(self):
        """Preferred path: SCCC raw payload includes the full WP item
        (added 2026-07-16). Full renormalization via the live
        connector."""
        from skillbridge.pipeline.step1a_backfill import _renormalize
        payload = {
            "source": "sccc_wp_rest_v2",
            "wp_id": 24454,
            "slug": "test-role",
            "item": {
                "id": 24454,
                "date": "2026-07-15T12:00:00",
                "title": {"rendered": "Test Role"},
                "link": "https://sccc.example/j/test",
                "content": {"rendered": "<p>Full JD body.</p>"},
                "meta": {
                    "_job_location": "Sault Ste. Marie",
                    "_company_name": "Test Employer",
                },
            },
        }
        job, skip = _renormalize("sccc", "24454", payload)
        assert skip is None
        assert job is not None
        assert job.source == "sccc"
        assert job.normalized_job_location == "Sault Ste. Marie"
        assert job.location_resolution_status == "resolved"
        assert job.location_provenance == "source_declared"
        assert job.description_evidence_status == "full_source"

    def test_sccc_legacy_payload_signals_direct_sql_path(self):
        """Legacy SCCC raw payloads (pre-2026-07-16) don't have the
        `item` key. The dispatcher signals `sccc_legacy_shape` so the
        runner can call the direct-SQL UPDATE path
        (_backfill_legacy_sccc_row), which promotes existing DB
        description to description_full and populates the location
        axis from raw's location_raw."""
        from skillbridge.pipeline.step1a_backfill import _renormalize
        payload = {
            "source": "sccc_wp_rest_v2",
            "wp_id": 24454,
            "slug": "test-role",
            "date": "2026-07-15T12:00:00",
            "modified": "2026-07-15T12:00:00",
            "link": "https://sccc.example/j/test",
            "location_raw": "Sault Ste. Marie",
            # No 'item' key — legacy shape.
        }
        job, skip = _renormalize("sccc", "24454", payload)
        assert job is None
        assert skip == "sccc_legacy_shape"

    def test_partner_csv_payload_dispatches(self):
        from skillbridge.pipeline.step1a_backfill import _renormalize
        payload = {
            "source_job_id": "csv-001",
            "title": "Analyst",
            "location": "Sault Ste. Marie",
            "description": "Real JD content.",
        }
        job, skip = _renormalize("partner_csv", "csv-001", payload)
        assert skip is None
        assert job is not None
        assert job.description_evidence_status == "full_source"
        assert job.normalized_job_location == "Sault Ste. Marie"

    def test_partner_upload_payload_dispatches(self):
        from skillbridge.pipeline.step1a_backfill import _renormalize
        payload = {
            "source_job_id": "upl-001",
            "title": "Analyst",
            "location": "Sault Ste. Marie, ON",
            "description": "JD.",
        }
        # partner_upload sources are prefixed with partner_upload_
        job, skip = _renormalize(
            "partner_upload_acme", "upl-001", payload,
        )
        assert skip is None
        assert job is not None
        assert job.normalized_job_location == "Sault Ste. Marie"

    def test_employer_connector_source_is_unsupported(self):
        """Sault Area Hospital and City of SSM raw payloads contain
        only card snippets — not enough for reconstruction. They
        skip cleanly (not silently upserted with mock data)."""
        from skillbridge.pipeline.step1a_backfill import _renormalize
        # A card-snippet payload from sault_area_hospital.
        payload = {"snippet": "<div>Registered Nurse</div>"}
        job, skip = _renormalize(
            "sault_area_hospital", "rn-day-shift", payload,
        )
        assert job is None
        assert skip == "unsupported_source"

    def test_unknown_source_is_unsupported(self):
        from skillbridge.pipeline.step1a_backfill import _renormalize
        job, skip = _renormalize(
            "totally_new_source", "123", {"any": "thing"},
        )
        assert job is None
        assert skip == "unsupported_source"


class TestRenormalizeShapeDefects:
    """Payloads that don't conform to the source's expected shape."""

    def test_non_dict_payload_is_shape_defect(self):
        from skillbridge.pipeline.step1a_backfill import _renormalize
        job, skip = _renormalize("awic_jobs", "1", "not a dict")
        assert job is None
        assert skip == "shape_defect"

    def test_awic_payload_missing_feature_key_is_shape_defect(self):
        from skillbridge.pipeline.step1a_backfill import _renormalize
        job, skip = _renormalize(
            "awic_jobs", "1", {"source": "awic_geojson_v1"},  # no 'feature'
        )
        assert job is None
        assert skip == "shape_defect"

    def test_awic_payload_with_non_dict_feature_is_shape_defect(self):
        from skillbridge.pipeline.step1a_backfill import _renormalize
        job, skip = _renormalize(
            "awic_jobs", "1", {"feature": "not-a-dict"},
        )
        assert job is None
        assert skip == "shape_defect"

    def test_sccc_payload_missing_item_key_signals_legacy_path(self):
        """Correction 2026-07-16: pre-item raw payloads route to the
        legacy direct-SQL path via `sccc_legacy_shape`, not
        `shape_defect`. This preserves historical SSM SCCC content
        in the live market during the Step 1A transition."""
        from skillbridge.pipeline.step1a_backfill import _renormalize
        job, skip = _renormalize(
            "sccc", "1", {"source": "sccc_wp_rest_v2"},
        )
        assert job is None
        assert skip == "sccc_legacy_shape"


# ══════════════════════════════════════════════════════════════════
# Renormalize identity: same input → same classification as live
# ══════════════════════════════════════════════════════════════════


class TestRenormalizeIdentity:
    """Guarantees the classifications backfill emits match what live
    ingest would emit for the same raw payload. Single source of
    truth is the connector normalize functions."""

    def test_awic_identity_with_live_normalize(self):
        """Same feature through backfill and through the live
        connector normalize function must produce identical
        classification tuples."""
        from skillbridge.pipeline.step1a_backfill import _renormalize
        from skillbridge.ingest.partners import (
            _normalize_awic_geojson_feature,
        )
        feature = {
            "geometry": {
                "type": "Point",
                "coordinates": [-84.32, 46.54],
            },
            "properties": {
                "post_id": 42,
                "job_title": "Test",
                "excerpt": "Some excerpt.",
            },
        }
        # Live path.
        live_job, _, _ = _normalize_awic_geojson_feature(feature)
        # Backfill path.
        bfl_job, _ = _renormalize(
            "awic_jobs", "42",
            {"source": "awic_geojson_v1", "feature": feature},
        )
        assert live_job is not None and bfl_job is not None
        # Every Step 1A axis must match.
        assert live_job.description_evidence_status == bfl_job.description_evidence_status
        assert live_job.description_excerpt == bfl_job.description_excerpt
        assert live_job.location_resolution_status == bfl_job.location_resolution_status
        assert live_job.location_provenance == bfl_job.location_provenance
        assert live_job.source_coordinates == bfl_job.source_coordinates
        assert live_job.source_location_text == bfl_job.source_location_text
        assert live_job.normalized_job_location == bfl_job.normalized_job_location

    def test_awic_malformed_geometry_identity(self):
        """Correction 2026-07-16 rule: non-dict geometry produces
        invalid/geometry, not missing/none. Backfill must reproduce
        this — critical for the historical distinction between
        absent-data and corrupted-data in production diagnostics."""
        from skillbridge.pipeline.step1a_backfill import _renormalize
        payload = {
            "source": "awic_geojson_v1",
            "feature": {
                "geometry": "bad-shape",
                "properties": {
                    "post_id": 42,
                    "job_title": "Test",
                },
            },
        }
        job, _ = _renormalize("awic_jobs", "42", payload)
        assert job is not None
        assert job.location_resolution_status == "invalid"
        assert job.location_provenance == "geometry"

    def test_awic_missing_geometry_identity(self):
        """Anti-regression: geometry absent → missing/none (not
        invalid). Distinct classification from the malformed case
        above."""
        from skillbridge.pipeline.step1a_backfill import _renormalize
        payload = {
            "source": "awic_geojson_v1",
            "feature": {
                "properties": {
                    "post_id": 42,
                    "job_title": "Test",
                },
            },
        }
        job, _ = _renormalize("awic_jobs", "42", payload)
        assert job is not None
        assert job.location_resolution_status == "missing"
        assert job.location_provenance == "none"


# ══════════════════════════════════════════════════════════════════
# Latest-raw selection: determinism (DISTINCT ON semantics)
# ══════════════════════════════════════════════════════════════════


class TestLatestRawSelection:
    """SQL-level determinism: DISTINCT ON with ORDER BY
    ingested_at DESC, raw_id DESC picks the newest raw row per
    posting. Verified via cursor mock."""

    def test_iterator_uses_distinct_on_query(self):
        """The generator must issue a DISTINCT ON query. Anything
        else could produce N rows per posting and cause
        non-deterministic backfill state."""
        from skillbridge.pipeline.step1a_backfill import (
            _iter_latest_raw_payloads,
        )

        captured_sql = {}
        fake_cursor = MagicMock()
        fake_cursor.fetchall.return_value = []

        def _capture(sql, *args, **kwargs):
            captured_sql["sql"] = sql

        fake_cursor.execute.side_effect = _capture

        @contextmanager
        def fake_sync_cursor():
            yield fake_cursor

        with patch(
            "skillbridge.pipeline.step1a_backfill.sync_cursor",
            fake_sync_cursor,
        ):
            list(_iter_latest_raw_payloads())

        sql = captured_sql["sql"]
        # Must be DISTINCT ON with the deterministic ORDER BY.
        assert "DISTINCT ON (source, source_job_id)" in sql
        # Ordering: newer ingested_at first; raw_id tiebreak.
        assert "ingested_at DESC" in sql
        assert "raw_id DESC" in sql

    def test_iterator_yields_payload_from_returned_row(self):
        from skillbridge.pipeline.step1a_backfill import (
            _iter_latest_raw_payloads,
        )
        fake_cursor = MagicMock()
        fake_cursor.fetchall.return_value = [
            {"source": "awic_jobs", "source_job_id": "1",
             "payload": {"feature": {}}},
            {"source": "sccc", "source_job_id": "2",
             "payload": {"item": {}}},
        ]
        fake_cursor.execute = MagicMock()

        @contextmanager
        def fake_sync_cursor():
            yield fake_cursor

        with patch(
            "skillbridge.pipeline.step1a_backfill.sync_cursor",
            fake_sync_cursor,
        ):
            results = list(_iter_latest_raw_payloads())

        assert len(results) == 2
        assert results[0] == ("awic_jobs", "1", {"feature": {}})
        assert results[1] == ("sccc", "2", {"item": {}})

    def test_iterator_handles_json_string_payload(self):
        """Defensive: some psycopg driver versions return JSONB
        as a string rather than a dict. The iterator normalizes."""
        from skillbridge.pipeline.step1a_backfill import (
            _iter_latest_raw_payloads,
        )
        fake_cursor = MagicMock()
        fake_cursor.fetchall.return_value = [
            {"source": "awic_jobs", "source_job_id": "1",
             "payload": '{"feature": {"properties": {"post_id": 1}}}'},
        ]
        fake_cursor.execute = MagicMock()

        @contextmanager
        def fake_sync_cursor():
            yield fake_cursor

        with patch(
            "skillbridge.pipeline.step1a_backfill.sync_cursor",
            fake_sync_cursor,
        ):
            results = list(_iter_latest_raw_payloads())

        assert len(results) == 1
        _, _, payload = results[0]
        assert isinstance(payload, dict)
        assert payload["feature"]["properties"]["post_id"] == 1


# ══════════════════════════════════════════════════════════════════
# run_step1a_backfill — end-to-end orchestration
# ══════════════════════════════════════════════════════════════════


class TestRunBackfill:
    def _make_mocked_run(self, latest_rows, dry_run=False):
        """Helper: run the backfill with a mocked iterator and
        mocked narrow-UPDATE writer. Returns (stats, update_calls).

        `_update_evidence_only` is the narrow-UPDATE helper; the
        runner MUST use it instead of `upsert_job` (blocker
        finding 2026-07-16: upsert_job would rewrite
        is_active/last_seen_at/title/employer, reactivating stale
        rows and re-creating the Step 1A bug)."""
        from skillbridge.pipeline.step1a_backfill import (
            run_step1a_backfill,
        )
        update_calls: list = []

        with patch(
            "skillbridge.pipeline.step1a_backfill._iter_latest_raw_payloads",
            return_value=iter(latest_rows),
        ), patch(
            "skillbridge.pipeline.step1a_backfill._update_evidence_only",
            side_effect=lambda job: (
                update_calls.append(job), (True, "updated"),
            )[1],
        ):
            stats = run_step1a_backfill(dry_run=dry_run)
        return stats, update_calls

    def test_narrow_update_supported_sources_and_skips_others(self):
        latest_rows = [
            ("awic_jobs", "42", {
                "feature": {
                    "geometry": {"coordinates": [-84.32, 46.54]},
                    "properties": {
                        "post_id": 42, "job_title": "Test",
                    },
                },
            }),
            ("sault_area_hospital", "rn-1", {
                "snippet": "<div>Nurse</div>",
            }),
        ]
        stats, update_calls = self._make_mocked_run(latest_rows)
        assert stats.seen == 2
        assert stats.renormalized == 1
        assert stats.upserted == 1
        assert stats.skipped_unsupported_source == 1
        assert len(update_calls) == 1
        assert update_calls[0].source == "awic_jobs"

    def test_dry_run_does_not_call_writer(self):
        latest_rows = [
            ("awic_jobs", "42", {
                "feature": {
                    "geometry": {"coordinates": [-84.32, 46.54]},
                    "properties": {
                        "post_id": 42, "job_title": "Test",
                    },
                },
            }),
        ]
        stats, update_calls = self._make_mocked_run(
            latest_rows, dry_run=True,
        )
        assert stats.seen == 1
        assert stats.renormalized == 1
        assert stats.upserted == 0
        assert len(update_calls) == 0
        # But distribution counters are still computed.
        assert stats.by_location_resolution_status.get("unresolved") == 1

    def test_runner_never_imports_upsert_job(self):
        """Blocker anti-regression (2026-07-16): backfill MUST NOT
        route through upsert_job. That path reactivates historical
        rows (is_active=TRUE, last_seen_at=NOW()) and rewrites
        title/employer/dates/NOC, temporarily re-creating the
        location-integrity bug Step 1A is fixing (v_current_job
        cutover hasn't happened yet)."""
        import skillbridge.pipeline.step1a_backfill as mod
        assert not hasattr(mod, "upsert_job"), (
            "step1a_backfill must not import upsert_job — its "
            "side effects (is_active/last_seen_at rewrite) would "
            "reactivate stale historical rows into the SSM market."
        )

    def test_stats_by_source_populated(self):
        latest_rows = [
            ("awic_jobs", "1", {"feature": {
                "properties": {"post_id": 1, "job_title": "T"},
            }}),
            ("awic_jobs", "2", {"feature": {
                "properties": {"post_id": 2, "job_title": "T"},
            }}),
            ("sccc", "3", {"item": {
                "id": 3, "title": {"rendered": "T"},
                "link": "https://sccc.example/j/3",
                "content": {"rendered": "<p>Body.</p>"},
                "meta": {"_job_location": "Sault Ste. Marie"},
            }}),
        ]
        stats, _ = self._make_mocked_run(latest_rows)
        assert stats.by_source["awic_jobs"]["seen"] == 2
        assert stats.by_source["awic_jobs"]["renormalized"] == 2
        assert stats.by_source["sccc"]["seen"] == 1
        assert stats.by_source["sccc"]["renormalized"] == 1

    def test_legacy_sccc_dispatches_to_direct_sql_helper(self):
        """When _renormalize signals `sccc_legacy_shape`, the runner
        calls `_backfill_legacy_sccc_row` for direct SQL UPDATE.
        Mocked here to verify the dispatch."""
        from skillbridge.pipeline.step1a_backfill import (
            run_step1a_backfill,
        )
        latest_rows = [
            ("sccc", "24454", {
                "wp_id": 24454,
                "location_raw": "Sault Ste. Marie",
                # No 'item' key — legacy.
            }),
        ]
        update_calls: list = []
        legacy_calls: list = []

        with patch(
            "skillbridge.pipeline.step1a_backfill._iter_latest_raw_payloads",
            return_value=iter(latest_rows),
        ), patch(
            "skillbridge.pipeline.step1a_backfill._update_evidence_only",
            side_effect=lambda job: (
                update_calls.append(job), (True, "updated"),
            )[1],
        ), patch(
            "skillbridge.pipeline.step1a_backfill._backfill_legacy_sccc_row",
            side_effect=lambda sid, payload, *, dry_run: (
                legacy_calls.append((sid, payload)),
                (True, "updated", "resolved", "full_source"),
            )[1],
        ):
            stats = run_step1a_backfill(dry_run=False)

        assert stats.seen == 1
        assert stats.renormalized == 1
        assert stats.upserted == 1
        # Narrow UPDATE NOT called for legacy SCCC — direct SQL instead.
        assert len(update_calls) == 0
        assert len(legacy_calls) == 1
        assert legacy_calls[0][0] == "24454"
        assert legacy_calls[0][1]["location_raw"] == "Sault Ste. Marie"
        # Distribution reflects the projected statuses from the helper.
        assert stats.by_location_resolution_status.get("resolved") == 1
        assert stats.by_description_evidence_status.get("full_source") == 1

    def test_distribution_counters_reflect_renormalized_rows(self):
        """Load-bearing: the location_resolution_status distribution
        the backfill reports must reflect actual classification
        outcomes, not raw source counts."""
        latest_rows = [
            # AWIC valid coords → unresolved.
            ("awic_jobs", "1", {"feature": {
                "geometry": {"coordinates": [-84.32, 46.54]},
                "properties": {"post_id": 1, "job_title": "T"},
            }}),
            # AWIC missing geometry → missing.
            ("awic_jobs", "2", {"feature": {
                "properties": {"post_id": 2, "job_title": "T"},
            }}),
            # AWIC malformed geometry → invalid.
            ("awic_jobs", "3", {"feature": {
                "geometry": "bad-shape",
                "properties": {"post_id": 3, "job_title": "T"},
            }}),
        ]
        stats, _ = self._make_mocked_run(latest_rows)
        assert stats.by_location_resolution_status["unresolved"] == 1
        assert stats.by_location_resolution_status["missing"] == 1
        assert stats.by_location_resolution_status["invalid"] == 1


# ══════════════════════════════════════════════════════════════════
# Narrow-UPDATE preservation (BLOCKER 2026-07-16 anti-regression)
# ══════════════════════════════════════════════════════════════════


class TestNarrowUpdatePreservation:
    """Anti-regression for the reviewer's 2026-07-16 blocker.

    The backfill write path must NEVER re-run through `upsert_job`
    (which rewrites is_active/last_seen_at/title/employer/etc.,
    reactivating historical rows before the SSM-only view cutover).
    The narrow-UPDATE helper `_update_evidence_only` writes only:

        - the eight new-axis evidence columns (COALESCE-guarded to
          preserve values already set — Step 1B downgrade safety +
          idempotence),
        - legacy `location`, legacy `description`,
        - `updated_at`.

    It NEVER touches is_active, last_seen_at, noc_code, title,
    employer, employer_id, url, posted_date, closing_date,
    salary_*, employment_type, remote_flag, region_code.
    """

    def _capture_sql(self):
        """Return (fake_sync_cursor context, ref) where ref["sql"] and
        ref["params"] hold the last execute() call after the cursor
        context exits."""
        captured = {}
        fake_cursor = MagicMock()
        fake_cursor.rowcount = 1

        def _capture(sql, params=None, *args, **kwargs):
            captured["sql"] = sql
            captured["params"] = params

        fake_cursor.execute.side_effect = _capture

        @contextmanager
        def fake_sync_cursor():
            yield fake_cursor

        return fake_sync_cursor, captured, fake_cursor

    def _make_job(self, **overrides):
        """Build a minimal NormalizedJob with all Step 1A axes."""
        from skillbridge.ingest.base import NormalizedJob
        defaults = dict(
            source="awic_jobs",
            source_job_id="42",
            title="Test Role",
            employer="AWIC Employer",
            url="https://awic.example/j/42",
            description_full=None,
            description_excerpt="Short excerpt.",
            description_evidence_status="excerpt_only",
            source_location_text=None,
            source_coordinates=[-84.32, 46.54],
            normalized_job_location=None,
            location_resolution_status="unresolved",
            location_provenance="geometry",
        )
        defaults.update(overrides)
        return NormalizedJob(**defaults)

    def test_update_sql_does_not_touch_is_active_or_last_seen_at(self):
        """BLOCKER anti-regression: backfill must NOT reactivate
        historical rows. `is_active` and `last_seen_at` must not
        appear as write targets in the UPDATE SET clause."""
        from skillbridge.pipeline.step1a_backfill import (
            _update_evidence_only,
        )
        fake_sync_cursor, captured, _cur = self._capture_sql()
        job = self._make_job()

        with patch(
            "skillbridge.pipeline.step1a_backfill.sync_cursor",
            fake_sync_cursor,
        ):
            _update_evidence_only(job)

        sql_upper = captured["sql"].upper()
        # Split at WHERE — only the SET clause is a write target.
        set_clause = sql_upper.split("WHERE")[0]
        # These MUST NOT appear as SET targets.
        assert "IS_ACTIVE" not in set_clause
        assert "LAST_SEEN_AT" not in set_clause
        assert "NOC_CODE" not in set_clause
        assert "TITLE" not in set_clause
        assert "EMPLOYER" not in set_clause
        assert "EMPLOYER_ID" not in set_clause
        assert "URL " not in set_clause and "URL\n" not in set_clause
        assert "POSTED_DATE" not in set_clause
        assert "CLOSING_DATE" not in set_clause
        assert "SALARY_" not in set_clause
        assert "EMPLOYMENT_TYPE" not in set_clause
        assert "REMOTE_FLAG" not in set_clause
        assert "REGION_CODE" not in set_clause

    def test_update_sql_never_inserts(self):
        """`_update_evidence_only` is UPDATE-only. If the row doesn't
        exist, this is a no-op — backfill only renormalizes rows
        that already came through live ingest."""
        from skillbridge.pipeline.step1a_backfill import (
            _update_evidence_only,
        )
        fake_sync_cursor, captured, _cur = self._capture_sql()
        job = self._make_job()

        with patch(
            "skillbridge.pipeline.step1a_backfill.sync_cursor",
            fake_sync_cursor,
        ):
            _update_evidence_only(job)

        sql_upper = captured["sql"].upper()
        assert "INSERT INTO" not in sql_upper
        assert "ON CONFLICT" not in sql_upper
        assert sql_upper.strip().startswith("UPDATE")

    def test_new_axis_columns_use_coalesce_for_step_1b_safety(self):
        """The eight new-axis columns must be written with
        COALESCE(existing, incoming) so an already-set value is
        preserved (Step 1B downgrade safety + idempotence)."""
        from skillbridge.pipeline.step1a_backfill import (
            _update_evidence_only,
        )
        fake_sync_cursor, captured, _cur = self._capture_sql()
        job = self._make_job()

        with patch(
            "skillbridge.pipeline.step1a_backfill.sync_cursor",
            fake_sync_cursor,
        ):
            _update_evidence_only(job)

        sql = captured["sql"]
        # All eight new-axis columns must be guarded.
        for col in (
            "description_full",
            "description_excerpt",
            "description_evidence_status",
            "source_location_text",
            "source_coordinates",
            "normalized_job_location",
            "location_resolution_status",
            "location_provenance",
        ):
            assert f"COALESCE({col}," in sql, (
                f"{col} must be COALESCE-guarded to prevent "
                f"Step 1B downgrades"
            )

    def test_update_returns_not_found_when_no_row_matched(self):
        """rowcount=0 means the row doesn't exist — backfill only
        renormalizes existing rows. Return `(False, 'not_found')`."""
        from skillbridge.pipeline.step1a_backfill import (
            _update_evidence_only,
        )
        fake_sync_cursor, _captured, cur = self._capture_sql()
        cur.rowcount = 0
        job = self._make_job()

        with patch(
            "skillbridge.pipeline.step1a_backfill.sync_cursor",
            fake_sync_cursor,
        ):
            matched, outcome = _update_evidence_only(job)

        assert matched is False
        assert outcome == "not_found"

    def test_update_returns_updated_when_row_matched(self):
        from skillbridge.pipeline.step1a_backfill import (
            _update_evidence_only,
        )
        fake_sync_cursor, _captured, cur = self._capture_sql()
        cur.rowcount = 1
        job = self._make_job()

        with patch(
            "skillbridge.pipeline.step1a_backfill.sync_cursor",
            fake_sync_cursor,
        ):
            matched, outcome = _update_evidence_only(job)

        assert matched is True
        assert outcome == "updated"

    def test_update_params_end_with_source_and_source_job_id(self):
        """The WHERE clause binds by (source, source_job_id).
        Last two params in the parameter tuple must be those keys."""
        from skillbridge.pipeline.step1a_backfill import (
            _update_evidence_only,
        )
        fake_sync_cursor, captured, _cur = self._capture_sql()
        job = self._make_job(
            source="awic_jobs", source_job_id="load-bearing-42",
        )

        with patch(
            "skillbridge.pipeline.step1a_backfill.sync_cursor",
            fake_sync_cursor,
        ):
            _update_evidence_only(job)

        params = captured["params"]
        assert params[-2:] == ("awic_jobs", "load-bearing-42")


# ══════════════════════════════════════════════════════════════════
# Legacy-SCCC dry-run stats measure existing DB description
# ══════════════════════════════════════════════════════════════════


class TestLegacySCCCDryRunMeasured:
    """Anti-regression for reviewer's 2026-07-16 Medium finding.

    The legacy SCCC direct-SQL path must measure the existing DB
    description before stamping description_evidence_status stats.
    Previously it approximated every row as full_source, which was
    a prediction, not a measurement — could be wrong under NULL
    descriptions or a prior Step 1B enrichment run.
    """

    def _mock_cursor_returning(self, row):
        """Return a fake sync_cursor context that yields a cursor
        whose fetchone() returns `row` (dict-shaped from psycopg).
        Second execute (the UPDATE) records rowcount=1."""
        fake_cursor = MagicMock()
        fake_cursor.fetchone.return_value = row
        fake_cursor.rowcount = 1
        fake_cursor.execute = MagicMock()

        # sync_cursor is called twice: once for the SELECT peek,
        # once for the UPDATE. Both yield the same fake cursor.
        @contextmanager
        def fake_sync_cursor():
            yield fake_cursor

        return fake_sync_cursor, fake_cursor

    def test_dry_run_reports_full_source_when_db_description_present(self):
        from skillbridge.pipeline.step1a_backfill import (
            _backfill_legacy_sccc_row,
        )
        row = {
            "description": "Existing full JD content in DB.",
            "description_full": None,
            "description_evidence_status": None,
            "loc_migrated": False,
            "desc_migrated": False,
        }
        fake_sync_cursor, _cur = self._mock_cursor_returning(row)

        with patch(
            "skillbridge.pipeline.step1a_backfill.sync_cursor",
            fake_sync_cursor,
        ):
            updated, outcome, proj_loc, proj_desc = (
                _backfill_legacy_sccc_row(
                    "24454",
                    {"location_raw": "Sault Ste. Marie"},
                    dry_run=True,
                )
            )

        assert updated is False
        assert outcome == "dry_run"
        assert proj_loc == "resolved"
        assert proj_desc == "full_source"

    def test_dry_run_reports_missing_when_db_description_null(self):
        """The previous stats path claimed full_source unconditionally;
        this test proves the NULL-description case is now honestly
        reported as `missing`."""
        from skillbridge.pipeline.step1a_backfill import (
            _backfill_legacy_sccc_row,
        )
        row = {
            "description": None,
            "description_full": None,
            "description_evidence_status": None,
            "loc_migrated": False,
            "desc_migrated": False,
        }
        fake_sync_cursor, _cur = self._mock_cursor_returning(row)

        with patch(
            "skillbridge.pipeline.step1a_backfill.sync_cursor",
            fake_sync_cursor,
        ):
            _updated, _outcome, _proj_loc, proj_desc = (
                _backfill_legacy_sccc_row(
                    "24454",
                    {"location_raw": "Sault Ste. Marie"},
                    dry_run=True,
                )
            )

        assert proj_desc == "missing"

    def test_dry_run_preserves_step_1b_desc_status(self):
        """If a prior run (or Step 1B enrichment) already stamped
        `full_source` on the row, the legacy SCCC path must report
        THAT status — not overwrite it."""
        from skillbridge.pipeline.step1a_backfill import (
            _backfill_legacy_sccc_row,
        )
        row = {
            "description": "Existing content.",
            "description_full": "Existing content.",
            "description_evidence_status": "full_source",
            "loc_migrated": False,
            "desc_migrated": True,
        }
        fake_sync_cursor, _cur = self._mock_cursor_returning(row)

        with patch(
            "skillbridge.pipeline.step1a_backfill.sync_cursor",
            fake_sync_cursor,
        ):
            _updated, _outcome, _proj_loc, proj_desc = (
                _backfill_legacy_sccc_row(
                    "24454",
                    {"location_raw": "Sault Ste. Marie"},
                    dry_run=True,
                )
            )

        # Existing status wins.
        assert proj_desc == "full_source"

    def test_dry_run_preserves_step_1b_loc_status(self):
        """Same guard on the location axis — a prior loc-migrated
        classification must not be overridden by the SCCC helper."""
        from skillbridge.pipeline.step1a_backfill import (
            _backfill_legacy_sccc_row,
        )
        row = {
            "description": "x",
            "description_full": None,
            "description_evidence_status": None,
            "loc_migrated": True,
            "location_resolution_status": "resolved",
            "desc_migrated": False,
        }
        fake_sync_cursor, _cur = self._mock_cursor_returning(row)

        with patch(
            "skillbridge.pipeline.step1a_backfill.sync_cursor",
            fake_sync_cursor,
        ):
            _updated, _outcome, proj_loc, _proj_desc = (
                _backfill_legacy_sccc_row(
                    "24454",
                    # Would classify as `missing` from a NULL raw,
                    # but existing loc-migrated status wins.
                    {"location_raw": None},
                    dry_run=True,
                )
            )

        assert proj_loc == "resolved"

    def test_row_not_found_returns_none_projections(self):
        """No matching row → nothing to stamp. Projected statuses
        must be None so the runner can skip stat contributions."""
        from skillbridge.pipeline.step1a_backfill import (
            _backfill_legacy_sccc_row,
        )
        fake_sync_cursor, cur = self._mock_cursor_returning(None)
        cur.rowcount = 0

        with patch(
            "skillbridge.pipeline.step1a_backfill.sync_cursor",
            fake_sync_cursor,
        ):
            updated, outcome, proj_loc, proj_desc = (
                _backfill_legacy_sccc_row(
                    "dead-id",
                    {"location_raw": "Sault Ste. Marie"},
                    dry_run=True,
                )
            )

        assert updated is False
        assert outcome == "not_found"
        assert proj_loc is None
        assert proj_desc is None
