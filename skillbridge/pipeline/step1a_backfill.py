"""Step 1A source-data-integrity backfill (2026-07-15).

Reads the latest raw payload per (source, source_job_id) from
`raw.job_posting` and re-runs each connector's normalize function
against it. The re-normalization uses the SAME logic as live ingest
so classifications cannot drift — Step 1A's `_normalize_awic_geojson_
feature`, `_normalize_sccc_wp_rest_item`, `_row_to_job` (both CSV
variants) are the sole source of truth for description /
description_evidence_status / source_location_text /
normalized_job_location / location_resolution_status /
location_provenance.

Determinism:
  raw.job_posting is append-only ([schema.sql:427-433]). Multiple
  rows can share (source, source_job_id) over time. This module uses
  `SELECT DISTINCT ON (source, source_job_id) ... ORDER BY source,
  source_job_id, ingested_at DESC, raw_id DESC` to pick the newest
  raw row per posting. Running the backfill twice produces identical
  results.

Preservation (write path):
  Backfill DOES NOT call `upsert_job`. That function would
  additionally rewrite `is_active = TRUE`, `last_seen_at = NOW()`,
  title, employer, NOC, dates — reactivating historical rows
  before the SSM-only `v_current_job` cutover has happened. That
  temporarily re-creates the exact location-integrity bug Step 1A
  is fixing. Instead, `_update_evidence_only` runs a narrow UPDATE
  that touches only:

    - the eight new evidence columns
    - legacy `location` (rewritten from classification)
    - legacy `description` (rewritten from classification)
    - `updated_at`

  It NEVER touches: is_active, last_seen_at, noc_code, title,
  employer, employer_id, url, posted_date, closing_date,
  salary_text/low/high, employment_type, remote_flag, region_code.

  Downgrade safety (idempotence + Step 1B safety):
    The eight new-axis columns are written with COALESCE(existing,
    incoming) semantics — a value that's already set stays set.
    First run populates every NULL; subsequent runs are no-ops.
    If Step 1B enrichment later upgrades AWIC from 'excerpt_only'
    to 'full_source', re-running Step 1A backfill will NOT
    downgrade it back to 'excerpt_only'.

Scope:
  AWIC, SCCC, Partner CSV, and Partner upload rows re-normalize
  cleanly from their raw payloads. Employer-specific connectors
  (Sault Area Hospital, City of SSM) can't be re-normalized from
  the card-snippet raw payload — they update on next live ingest
  and are counted as `skipped` here.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Iterable

from skillbridge.db import sync_cursor
from skillbridge.ingest.base import NormalizedJob
from skillbridge.ingest.partners import (
    _classify_declared_location_field,
    _normalize_awic_geojson_feature,
    _normalize_sccc_wp_rest_item,
    _row_to_job as _partner_csv_row_to_job,
)
from skillbridge.ingest.partner_upload import (
    _row_to_job as _partner_upload_row_to_job,
)
from skillbridge.match.region import normalize_declared_job_location

log = logging.getLogger(__name__)


@dataclass
class BackfillStats:
    """Aggregate counters for a backfill run."""
    seen: int = 0
    renormalized: int = 0
    upserted: int = 0
    skipped_unsupported_source: int = 0
    skipped_shape_defect: int = 0
    skipped_by_connector_filter: int = 0
    by_source: dict = field(default_factory=dict)
    by_location_resolution_status: dict = field(default_factory=dict)
    by_description_evidence_status: dict = field(default_factory=dict)


# Connector sources that flow through this backfill's renormalization
# path. Employer-specific and disabled sources are intentionally
# outside this set — see module docstring §Scope.
_BACKFILLABLE_SOURCES: frozenset[str] = frozenset({
    "awic_jobs",
    "sccc",
    "partner_csv",
    "welcome_ssm",       # planned partner CSV shape
    "city_ssm",          # planned partner CSV shape (not employer connector)
})

# Sources handled by partner_upload._row_to_job specifically. Distinct
# from the partners.py path because uploads may include a
# per-partner prefix from filename convention. Read from
# core.approved_job_source at runtime rather than hardcoding — the
# upload prefix determines the source string.
_PARTNER_UPLOAD_SOURCE_PREFIX: str = "partner_upload_"


def _renormalize(
    source: str, source_job_id: str, payload,
) -> tuple[NormalizedJob | None, str | None]:
    """Re-run the appropriate connector's normalize function.

    Returns (job, skip_reason). If job is None, skip_reason explains
    why (unsupported source, shape defect, filter drop). If job is
    not None, skip_reason is None.
    """
    if not isinstance(payload, dict):
        return None, "shape_defect"

    if source == "awic_jobs":
        feature = payload.get("feature")
        if not isinstance(feature, dict):
            return None, "shape_defect"
        job, drop_reason, _ = _normalize_awic_geojson_feature(feature)
        if job is None:
            # drop_reason == "malformed" means missing post_id / title.
            # Under Step 1A this is still a genuine drop (row shouldn't
            # exist in core.job_posting either); skip.
            return None, "shape_defect"
        return job, None

    if source == "sccc":
        # Preferred path: full WP item preserved in raw payload
        # (added 2026-07-16). Full renormalization via the live
        # connector — same classification as ingest.
        item = payload.get("item")
        if isinstance(item, dict):
            job = _normalize_sccc_wp_rest_item(item)
            if job is None:
                return None, "connector_filter"
            return job, None
        # Legacy raw payload (pre-2026-07-16) — no full WP item.
        # Handled by _backfill_legacy_sccc_row via direct SQL UPDATE
        # rather than the connector-normalize path. Signal here so
        # the runner dispatches to that helper.
        return None, "sccc_legacy_shape"

    if source in _BACKFILLABLE_SOURCES:
        # partner_csv, welcome_ssm, city_ssm — all use partners._row_to_job.
        job = _partner_csv_row_to_job(payload, source=source)
        if job is None:
            return None, "shape_defect"
        return job, None

    if source.startswith(_PARTNER_UPLOAD_SOURCE_PREFIX):
        job = _partner_upload_row_to_job(payload, source=source)
        if job is None:
            return None, "shape_defect"
        return job, None

    # Unsupported source (employer connectors + anything else).
    return None, "unsupported_source"


def _derive_evidence_write_fields(job: NormalizedJob) -> dict:
    """Same normalization the live upsert path applies (base.py:280-
    294): treat "" as None on new-axis strings and derive legacy
    location / description from the migrated classification.

    Isolated here so backfill's narrow UPDATE stays in sync with the
    live path's derivation rules without pulling in `upsert_job`'s
    reactivation side effects.
    """
    def _blank_to_none(v):
        return None if isinstance(v, str) and v == "" else v

    description_full_norm = _blank_to_none(job.description_full)
    description_excerpt_norm = _blank_to_none(job.description_excerpt)
    source_location_text_norm = _blank_to_none(job.source_location_text)
    normalized_job_location_norm = _blank_to_none(job.normalized_job_location)

    connector_migrated_desc = job.description_evidence_status is not None
    connector_migrated_loc = job.location_resolution_status is not None

    if connector_migrated_desc:
        if description_full_norm is not None:
            legacy_description = description_full_norm
        elif description_excerpt_norm is not None:
            legacy_description = description_excerpt_norm
        else:
            legacy_description = None
    else:
        legacy_description = job.description

    if connector_migrated_loc:
        legacy_location = normalized_job_location_norm
    else:
        legacy_location = job.location

    source_coords_json = (
        json.dumps(job.source_coordinates)
        if job.source_coordinates is not None else None
    )
    return {
        "description_full":            description_full_norm,
        "description_excerpt":         description_excerpt_norm,
        "description_evidence_status": job.description_evidence_status,
        "source_location_text":        source_location_text_norm,
        "source_coordinates":          source_coords_json,
        "normalized_job_location":     normalized_job_location_norm,
        "location_resolution_status":  job.location_resolution_status,
        "location_provenance":         job.location_provenance,
        "legacy_location":             legacy_location,
        "legacy_description":          legacy_description,
    }


def _update_evidence_only(job: NormalizedJob) -> tuple[bool, str]:
    """Narrow UPDATE for Step 1A backfill.

    Distinct from `upsert_job`:
      - Never INSERTs. If no matching row exists, this is a no-op
        (backfill is about renormalizing rows that ALREADY exist in
        core.job_posting from prior ingest).
      - NEVER touches: is_active, last_seen_at, noc_code, title,
        employer, employer_id, url, posted_date, closing_date,
        salary_*, employment_type, remote_flag, region_code.
      - The eight new-axis columns use COALESCE(existing, incoming)
        so already-set values are preserved (Step 1B downgrade
        safety + idempotence).
      - Legacy `location` and `description` are rewritten from
        classification (removes the historical hardcoded
        "Sault Ste. Marie" label; promotes description_full into
        the legacy column consumers still read from).
      - `updated_at` is bumped so audit trails still see the row
        moved.

    Returns (updated, outcome):
      - (True, "updated") when the UPDATE matched a row.
      - (False, "not_found") when no matching row exists.
    """
    fields = _derive_evidence_write_fields(job)
    sql = """
    UPDATE core.job_posting SET
        description_full            = COALESCE(description_full,            %s),
        description_excerpt         = COALESCE(description_excerpt,         %s),
        description_evidence_status = COALESCE(description_evidence_status, %s),
        source_location_text        = COALESCE(source_location_text,        %s),
        source_coordinates          = COALESCE(source_coordinates,          %s),
        normalized_job_location     = COALESCE(normalized_job_location,     %s),
        location_resolution_status  = COALESCE(location_resolution_status,  %s),
        location_provenance         = COALESCE(location_provenance,         %s),
        location                    = %s,
        description                 = %s,
        updated_at                  = NOW()
     WHERE source = %s AND source_job_id = %s
    """
    with sync_cursor() as cur:
        cur.execute(
            sql,
            (
                fields["description_full"],
                fields["description_excerpt"],
                fields["description_evidence_status"],
                fields["source_location_text"],
                fields["source_coordinates"],
                fields["normalized_job_location"],
                fields["location_resolution_status"],
                fields["location_provenance"],
                fields["legacy_location"],
                fields["legacy_description"],
                job.source,
                job.source_job_id,
            ),
        )
        matched = cur.rowcount > 0
        return matched, "updated" if matched else "not_found"


def _backfill_legacy_sccc_row(
    source_job_id: str, payload: dict, *, dry_run: bool,
) -> tuple[bool, str]:
    """Direct-SQL backfill for pre-2026-07-16 SCCC raw payloads that
    lack the full WP item.

    Legacy SCCC raw payloads only preserved `location_raw`, not the
    full `content.rendered` field. To keep historical SSM SCCC
    content in the live market during the Step 1A transition, this
    handler:

      Location axis: classifies `location_raw` via the same helper
        the live connector uses (`_classify_declared_location_field`
        + `normalize_declared_job_location`). Result is identical to
        what live ingest would produce.

      Description axis: promotes the existing DB `description` (which
        was cleaned by `_clean_wp_html` at original ingest time) into
        `description_full` with `description_evidence_status =
        'full_source'`. This is honest: the source DID provide full
        content; we just weren't splitting it out into the new-axis
        column at ingest time.

      Preservation guards: every UPDATE target uses COALESCE or
        NULL-only CASE guards so a row already carrying Step 1B
        evidence (or a prior Step 1A run) is not downgraded.

      Legacy `location` column: not written here. The live legacy
        column for SCCC rows already carries the SSM-canonical
        string; there is no honest change to make from the direct-
        SQL path (we do not have the full item shape here).

    Returns (updated, outcome, projected_loc_status, projected_desc_status):
      - `updated` — True if the UPDATE matched a row.
      - `outcome` — "updated" | "not_found" | "dry_run".
      - `projected_loc_status` / `projected_desc_status` reflect
        what the row will actually carry AFTER this call (measured
        from existing DB state), so caller stats are honest under
        dry-run and under a re-run after Step 1B.
    """
    location_raw = payload.get("location_raw")
    source_location_text, loc_field_status = (
        _classify_declared_location_field(location_raw)
    )
    if loc_field_status == "invalid":
        normalized_loc = None
        location_resolution_status = "invalid"
        location_provenance = "source_declared"
    else:
        (normalized_loc, _remote) = normalize_declared_job_location(
            source_location_text
        )
        if normalized_loc is not None:
            location_resolution_status = "resolved"
            location_provenance = "source_declared"
        else:
            location_resolution_status = "missing"
            location_provenance = (
                "none" if source_location_text is None
                else "source_declared"
            )

    # Measure existing DB description so dry-run stats reflect what
    # the mutation would actually produce (missing vs full_source).
    # Also required by the WHERE-NULL guards below to know whether
    # this row will be written or preserved.
    with sync_cursor() as cur:
        cur.execute(
            """
            SELECT description,
                   description_full,
                   description_evidence_status,
                   location_resolution_status IS NOT NULL AS loc_migrated,
                   description_evidence_status IS NOT NULL AS desc_migrated
              FROM core.job_posting
             WHERE source = 'sccc'
               AND source_job_id = %s
            """,
            (source_job_id,),
        )
        existing = cur.fetchone()

    if existing is None:
        return False, "not_found", None, None

    if existing["desc_migrated"]:
        projected_desc_status = existing["description_evidence_status"]
    elif existing["description"] is not None:
        projected_desc_status = "full_source"
    else:
        projected_desc_status = "missing"

    projected_loc_status = (
        existing["location_resolution_status"]
        if existing["loc_migrated"]
        else location_resolution_status
    )

    if dry_run:
        return False, "dry_run", projected_loc_status, projected_desc_status

    with sync_cursor() as cur:
        cur.execute(
            """
            UPDATE core.job_posting
               SET source_location_text        =
                       COALESCE(source_location_text, %s),
                   normalized_job_location     =
                       COALESCE(normalized_job_location, %s),
                   location_resolution_status  =
                       COALESCE(location_resolution_status, %s),
                   location_provenance         =
                       COALESCE(location_provenance, %s),
                   description_full            =
                       COALESCE(description_full, description),
                   description_evidence_status =
                       CASE
                           WHEN description_evidence_status IS NULL
                                AND description IS NOT NULL
                             THEN 'full_source'
                           WHEN description_evidence_status IS NULL
                                AND description IS NULL
                             THEN 'missing'
                           ELSE description_evidence_status
                       END,
                   updated_at                  = NOW()
             WHERE source = 'sccc'
               AND source_job_id = %s
            """,
            (
                source_location_text, normalized_loc,
                location_resolution_status, location_provenance,
                source_job_id,
            ),
        )
        matched = cur.rowcount > 0
        return (
            matched,
            "updated" if matched else "not_found",
            projected_loc_status,
            projected_desc_status,
        )


def _iter_latest_raw_payloads() -> Iterable[tuple[str, str, dict]]:
    """Yield (source, source_job_id, payload) for the latest raw row
    per posting, per §3a of the spec.

    `DISTINCT ON (source, source_job_id) ... ORDER BY source,
    source_job_id, ingested_at DESC, raw_id DESC` guarantees:
      - Exactly one row per (source, source_job_id)
      - The row picked is the newest ingested_at (ties broken by the
        higher raw_id).
    """
    with sync_cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (source, source_job_id)
                source, source_job_id, payload
              FROM raw.job_posting
             ORDER BY source, source_job_id,
                      ingested_at DESC, raw_id DESC
            """
        )
        for row in cur.fetchall():
            payload = row["payload"]
            # `payload` is JSONB — psycopg returns it as dict already,
            # but be defensive against a str return in older drivers.
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:  # noqa: BLE001
                    payload = {}
            yield row["source"], row["source_job_id"], payload


def run_step1a_backfill(*, dry_run: bool = False) -> BackfillStats:
    """Iterate latest raw payloads, re-normalize, and upsert.

    dry_run=True runs the classification pass without calling
    upsert_job — useful for smoke-testing the classification logic
    against production data.
    """
    stats = BackfillStats()
    for source, source_job_id, payload in _iter_latest_raw_payloads():
        stats.seen += 1
        by_src = stats.by_source.setdefault(
            source, {"seen": 0, "renormalized": 0, "upserted": 0}
        )
        by_src["seen"] += 1

        job, skip_reason = _renormalize(source, source_job_id, payload)
        if job is None:
            if skip_reason == "unsupported_source":
                stats.skipped_unsupported_source += 1
            elif skip_reason == "shape_defect":
                stats.skipped_shape_defect += 1
            elif skip_reason == "connector_filter":
                stats.skipped_by_connector_filter += 1
            elif skip_reason == "sccc_legacy_shape":
                # Legacy SCCC raw payload — direct-SQL UPDATE path.
                # Independent of the connector-normalize dispatch
                # because the raw payload lacks the full WP item.
                updated, _outcome, proj_loc, proj_desc = (
                    _backfill_legacy_sccc_row(
                        source_job_id, payload, dry_run=dry_run,
                    )
                )
                stats.renormalized += 1
                by_src["renormalized"] += 1
                if updated:
                    stats.upserted += 1
                    by_src["upserted"] += 1
                if proj_loc is not None:
                    stats.by_location_resolution_status.setdefault(
                        proj_loc, 0
                    )
                    stats.by_location_resolution_status[proj_loc] += 1
                if proj_desc is not None:
                    stats.by_description_evidence_status.setdefault(
                        proj_desc, 0
                    )
                    stats.by_description_evidence_status[proj_desc] += 1
            continue

        stats.renormalized += 1
        by_src["renormalized"] += 1

        # Distribution counters (regardless of dry-run).
        loc_status = job.location_resolution_status or "unset"
        stats.by_location_resolution_status.setdefault(loc_status, 0)
        stats.by_location_resolution_status[loc_status] += 1

        desc_status = job.description_evidence_status or "unset"
        stats.by_description_evidence_status.setdefault(desc_status, 0)
        stats.by_description_evidence_status[desc_status] += 1

        if not dry_run:
            try:
                matched, _outcome = _update_evidence_only(job)
                if matched:
                    stats.upserted += 1
                    by_src["upserted"] += 1
            except Exception as e:  # noqa: BLE001
                log.error(
                    "backfill update failed for (%s, %s): %s",
                    source, source_job_id, e,
                )

    log.info(
        "Step 1A backfill: seen=%d renormalized=%d upserted=%d "
        "skipped_unsupported=%d skipped_shape=%d skipped_filter=%d "
        "loc_status=%s desc_status=%s",
        stats.seen, stats.renormalized, stats.upserted,
        stats.skipped_unsupported_source, stats.skipped_shape_defect,
        stats.skipped_by_connector_filter,
        dict(stats.by_location_resolution_status),
        dict(stats.by_description_evidence_status),
    )
    return stats
