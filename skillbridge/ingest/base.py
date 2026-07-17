"""Connector interface + normalized records + region filter.

Two connector families:

1. Iterable connectors  (existing pre-PR-6A pattern)
     JobConnector.fetch()       -> Iterable[NormalizedJob]
     TrainingConnector.fetch()  -> Iterable[NormalizedTrainingResource]
   The pipeline iterates and writes each item.

2. Self-contained connectors (PR 6A pattern for supplementary sources)
     SourceConnector.run()      -> ConnectorResult
   Each connector owns its fetch + normalize + upsert lifecycle and
   returns a uniform status object.

Both families share the upsert helpers in this module so the write path
is identical regardless of source.
"""
from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Iterable

from skillbridge.db import sync_conn, sync_cursor

log = logging.getLogger(__name__)


@dataclass
class NormalizedJob:
    source: str
    source_job_id: str
    title: str
    employer: str | None = None
    # DEPRECATED (Step 1A, 2026-07-15): callers should populate the
    # description_* and location_* axes below; `location` becomes a
    # copy of `normalized_job_location` (NULL when resolution didn't
    # succeed) and `description` becomes COALESCE(description_full,
    # description_excerpt). Retained for one release cycle so
    # in-flight callers don't break. See docs/matching-revise/step-1
    # -source-data-integrity.md §2b.
    location: str | None = None
    region_code: str | None = None
    description: str | None = None
    url: str | None = None
    posted_date: date | None = None
    closing_date: date | None = None
    salary_text: str | None = None
    salary_low: float | None = None
    salary_high: float | None = None
    employment_type: str | None = None
    remote_flag: bool | None = None
    noc_code: str | None = None
    raw_payload: dict = field(default_factory=dict)

    # Step 1A description axis. Provenance-derived — see spec §2c/§2g.
    #   full_source  — complete JD body captured
    #   excerpt_only — only a preview captured
    #   missing      — no description data present
    #   parse_error  — data present but couldn't be used
    #
    # Status defaults to None (NOT "missing") so `upsert_job` can
    # distinguish "connector not yet migrated to Step 1A" from
    # "connector explicitly measured this row as missing." During
    # the transitional period, unmigrated connectors leaving these
    # None cause `upsert_job` to preserve their `job.description`
    # legacy value instead of erasing it. Once every producer emits
    # a real status, the None default becomes dead code and can be
    # tightened to a required enum.
    description_full: str | None = None
    description_excerpt: str | None = None
    description_evidence_status: str | None = None

    # Step 1A location axis. Five orthogonal fields — see spec §2c.
    # Same None-default rationale as description axis.
    #
    # source_location_text: exact source-declared string (never a
    #   hardcoded fallback).
    # source_coordinates: preserved GeoJSON [lon, lat, …] when source
    #   provides one; NOT authoritative for job location.
    # normalized_job_location: trustworthy location; populated only
    #   when resolution succeeded (§2d gates the market on it).
    # location_resolution_status:
    #   resolved | unresolved | missing | conflicting | invalid
    # location_provenance:
    #   source_declared | detail_page | geometry | multiple | none
    source_location_text: str | None = None
    source_coordinates: list | None = None  # JSONB-serialized at write
    normalized_job_location: str | None = None
    location_resolution_status: str | None = None
    location_provenance: str | None = None


@dataclass
class NormalizedTrainingResource:
    provider: str
    title: str
    description: str | None = None
    url: str | None = None
    location: str | None = None
    delivery_mode: str | None = None
    cost_text: str | None = None
    duration_text: str | None = None
    duration_band: str | None = None     # short | medium | long
    resource_type: str | None = None     # workshop | course | program | counselling | online


class JobConnector(ABC):
    source_name: str = "unknown"

    @abstractmethod
    def fetch(self) -> Iterable[NormalizedJob]: ...


class TrainingConnector(ABC):
    provider_name: str = "unknown"

    @abstractmethod
    def fetch(self) -> Iterable[NormalizedTrainingResource]: ...


# ------------------------------------------------- Lenient parsing helpers
# These used to live in jobbank.py; relocated here when Job Bank was removed
# as a data source (BREAKING.md). All connectors share them.
def parse_date_loose(s: str | None) -> date | None:
    if not s:
        return None
    s = str(s).strip()
    for fmt in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%B %d, %Y",
        "%b %d, %Y",
    ):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def parse_float_loose(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return float(str(s).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return None


# Note: there is no region filter here. SkillBridge SSM only ingests from
# SSM-native sources (SCCC, local employer career pages, partner uploads).
# Every job by construction is local — see core.approved_job_source.


def derive_duration_band(duration_text: str | None) -> str | None:
    """short  ≤ 8 weeks
    medium ≤ 12 months
    long   > 12 months
    """
    if not duration_text:
        return None
    t = duration_text.lower()
    if any(k in t for k in ["hour", "day", "week", "weekend", "workshop", "1-day", "2-day"]):
        return "short"
    months = 0
    m = re.search(r"(\d+)\s*month", t)
    if m:
        months = int(m.group(1))
    y = re.search(r"(\d+)\s*year", t)
    if y:
        months = max(months, int(y.group(1)) * 12)
    if months == 0:
        return None
    if months <= 12:
        return "medium"
    return "long"


# ------------------------------------------------------------- Upsert helpers
def upsert_employer(name: str | None) -> str | None:
    """Upsert into core.employer; return employer_id."""
    if not name:
        return None
    normalized = re.sub(r"\s+", " ", name).strip().lower()
    if not normalized:
        return None
    with sync_cursor() as cur:
        cur.execute(
            """
            INSERT INTO core.employer (name, name_normalized)
            VALUES (%s, %s)
            ON CONFLICT (name_normalized) DO UPDATE SET name = EXCLUDED.name
            RETURNING employer_id
            """,
            (name, normalized),
        )
        row = cur.fetchone()
        return str(row["employer_id"]) if row else None


def write_raw_job(job: NormalizedJob) -> None:
    with sync_cursor() as cur:
        cur.execute(
            """
            INSERT INTO raw.job_posting (source, source_job_id, payload)
            VALUES (%s, %s, %s)
            """,
            (job.source, job.source_job_id, json.dumps(job.raw_payload, default=str)),
        )


def upsert_job(job: NormalizedJob) -> str | None:
    """Upsert into core.job_posting and return job_id. Updates last_seen_at.

    Step 1A (2026-07-15) transitional persistence:
      - New description/location axes are written when the connector
        provides them.
      - Legacy `location` / `description` columns are ALSO written for
        one release cycle so in-flight callers don't break.

    Determining which value wins for the legacy columns depends on
    whether the connector has been migrated to Step 1A:

      Migrated connector (emits explicit status):
        legacy `location`    = normalized_job_location
        legacy `description` = description_full,
                               else description_excerpt
                               (blank strings treated as absent —
                               matches SQL COALESCE semantics without
                               relying on Python truthiness).

      Unmigrated connector (statuses are None):
        legacy `location`    = job.location            (preserved)
        legacy `description` = job.description         (preserved)
        No erasure — see spec correction 2026-07-16.

    Status = None is the sentinel for "connector hasn't been migrated
    yet." Migrated connectors MUST emit an explicit status literal
    ('missing', 'excerpt_only', etc.) even when the source genuinely
    lacks data. When every producer is migrated, the None sentinel
    becomes dead code and the dataclass defaults can be tightened.
    """
    import json

    employer_id = upsert_employer(job.employer)

    def _blank_to_none(value: str | None) -> str | None:
        """Normalize whitespace-only strings to None so
        description_full = '' doesn't fall through to excerpt
        differently than SQL COALESCE would.
        """
        if not isinstance(value, str):
            return None
        stripped = value.strip()
        return stripped if stripped else None

    description_full_norm = _blank_to_none(job.description_full)
    description_excerpt_norm = _blank_to_none(job.description_excerpt)
    normalized_job_location_norm = _blank_to_none(job.normalized_job_location)

    # Transitional persistence: only derive legacy fields from the new
    # axes when the connector has been migrated (status is not None).
    # Unmigrated connectors keep their existing job.location /
    # job.description values.
    connector_migrated_description = (
        job.description_evidence_status is not None
    )
    connector_migrated_location = (
        job.location_resolution_status is not None
    )

    if connector_migrated_description:
        # Use is-not-None check (not Python `or`) so an explicit "" from
        # a migrated connector doesn't accidentally fall through.
        if description_full_norm is not None:
            legacy_description = description_full_norm
        elif description_excerpt_norm is not None:
            legacy_description = description_excerpt_norm
        else:
            legacy_description = None
    else:
        legacy_description = job.description

    if connector_migrated_location:
        legacy_location = normalized_job_location_norm
    else:
        legacy_location = job.location

    # Coordinates persisted as JSONB — JSON-encode if the connector
    # supplied a Python list. None stays None.
    source_coords_json = (
        json.dumps(job.source_coordinates)
        if job.source_coordinates is not None else None
    )

    sql = """
    INSERT INTO core.job_posting
        (source, source_job_id, title, employer, employer_id, location, region_code,
         description, url, posted_date, closing_date, salary_text, salary_low, salary_high,
         employment_type, remote_flag, noc_code,
         description_full, description_excerpt, description_evidence_status,
         source_location_text, source_coordinates, normalized_job_location,
         location_resolution_status, location_provenance,
         is_active, last_seen_at, updated_at)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
            %s,%s,%s,
            %s,%s,%s,
            %s,%s,
            TRUE,NOW(),NOW())
    ON CONFLICT (source, source_job_id) DO UPDATE SET
        title                       = EXCLUDED.title,
        employer                    = EXCLUDED.employer,
        employer_id                 = EXCLUDED.employer_id,
        location                    = EXCLUDED.location,
        region_code                 = EXCLUDED.region_code,
        description                 = EXCLUDED.description,
        url                         = EXCLUDED.url,
        posted_date                 = EXCLUDED.posted_date,
        closing_date                = EXCLUDED.closing_date,
        salary_text                 = EXCLUDED.salary_text,
        salary_low                  = EXCLUDED.salary_low,
        salary_high                 = EXCLUDED.salary_high,
        employment_type             = EXCLUDED.employment_type,
        remote_flag                 = EXCLUDED.remote_flag,
        noc_code                    = EXCLUDED.noc_code,
        -- Step 1A (2026-07-16): preserve backfilled new-axis values
        -- when the incoming row comes from an unmigrated connector
        -- (status sentinel is NULL). Prevents a re-ingest by an
        -- unmigrated connector from erasing a row that backfill
        -- populated. Migrated connectors emit an explicit status
        -- ('missing', 'excerpt_only', etc.) and their writes DO
        -- overwrite. See docs/matching-revise/step-1-*.md correction
        -- 2026-07-16.
        description_full = CASE
            WHEN EXCLUDED.description_evidence_status IS NULL
                THEN core.job_posting.description_full
            ELSE EXCLUDED.description_full
        END,
        description_excerpt = CASE
            WHEN EXCLUDED.description_evidence_status IS NULL
                THEN core.job_posting.description_excerpt
            ELSE EXCLUDED.description_excerpt
        END,
        description_evidence_status = CASE
            WHEN EXCLUDED.description_evidence_status IS NULL
                THEN core.job_posting.description_evidence_status
            ELSE EXCLUDED.description_evidence_status
        END,
        source_location_text = CASE
            WHEN EXCLUDED.location_resolution_status IS NULL
                THEN core.job_posting.source_location_text
            ELSE EXCLUDED.source_location_text
        END,
        source_coordinates = CASE
            WHEN EXCLUDED.location_resolution_status IS NULL
                THEN core.job_posting.source_coordinates
            ELSE EXCLUDED.source_coordinates
        END,
        normalized_job_location = CASE
            WHEN EXCLUDED.location_resolution_status IS NULL
                THEN core.job_posting.normalized_job_location
            ELSE EXCLUDED.normalized_job_location
        END,
        location_resolution_status = CASE
            WHEN EXCLUDED.location_resolution_status IS NULL
                THEN core.job_posting.location_resolution_status
            ELSE EXCLUDED.location_resolution_status
        END,
        location_provenance = CASE
            WHEN EXCLUDED.location_resolution_status IS NULL
                THEN core.job_posting.location_provenance
            ELSE EXCLUDED.location_provenance
        END,
        is_active                   = TRUE,
        last_seen_at                = NOW(),
        updated_at                  = NOW()
    RETURNING job_id
    """
    with sync_cursor() as cur:
        cur.execute(
            sql,
            (
                job.source, job.source_job_id, job.title, job.employer, employer_id,
                legacy_location, job.region_code, legacy_description, job.url,
                job.posted_date, job.closing_date, job.salary_text,
                job.salary_low, job.salary_high, job.employment_type,
                job.remote_flag, job.noc_code,
                description_full_norm, description_excerpt_norm,
                job.description_evidence_status,
                job.source_location_text, source_coords_json,
                normalized_job_location_norm,
                job.location_resolution_status, job.location_provenance,
            ),
        )
        row = cur.fetchone()
        return str(row["job_id"]) if row else None


def upsert_training(res: NormalizedTrainingResource) -> str | None:
    sql = """
    INSERT INTO core.training_resource
        (provider, title, description, url, location, delivery_mode, cost_text,
         duration_text, duration_band, resource_type, is_active, updated_at)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,TRUE,NOW())
    ON CONFLICT (provider, title) DO UPDATE SET
        description   = EXCLUDED.description,
        url           = EXCLUDED.url,
        location      = EXCLUDED.location,
        delivery_mode = EXCLUDED.delivery_mode,
        cost_text     = EXCLUDED.cost_text,
        duration_text = EXCLUDED.duration_text,
        duration_band = EXCLUDED.duration_band,
        resource_type = EXCLUDED.resource_type,
        is_active     = TRUE,
        updated_at    = NOW()
    RETURNING resource_id
    """
    with sync_cursor() as cur:
        cur.execute(
            sql,
            (
                res.provider, res.title, res.description, res.url, res.location,
                res.delivery_mode, res.cost_text, res.duration_text,
                res.duration_band, res.resource_type,
            ),
        )
        row = cur.fetchone()
        return str(row["resource_id"]) if row else None


def commit() -> None:
    with sync_conn() as conn:
        conn.commit()


def sweep_missing_jobs(source: str, seen_ids: set[str]) -> int:
    """Mark jobs as inactive when their source no longer advertises them.

    Per-source presence sweep — only touches rows WHERE source = the
    given source. Called after a successful connector run to make
    Postgres reflect what the source currently lists.

    Safety: if `seen_ids` is empty, returns 0 without touching the table.
    An empty connector response is much more likely to be an API blip
    than the source actually clearing every job at once. PR 9B adds the
    full degraded-response guard; for PR 9A we keep this simple "no IDs,
    no sweep" rule.

    Returns the number of rows deactivated.
    """
    if not seen_ids:
        return 0
    ids_list = sorted(seen_ids)
    with sync_cursor() as cur:
        cur.execute(
            """
            UPDATE core.job_posting
               SET is_active = FALSE,
                   updated_at = NOW()
             WHERE source = %s
               AND is_active = TRUE
               AND source_job_id <> ALL(%s::text[])
            """,
            (source, ids_list),
        )
        return cur.rowcount


# ============================================================================
# PR 6A — SourceConnector pattern (supplementary sources)
# ============================================================================

@dataclass
class ConnectorResult:
    """Uniform result shape every supplementary connector returns.

    Statuses:
      skipped — connector disabled or all required URLs are placeholders
      success — at least one row upserted
      warn    — ran but found nothing useful (placeholder data, empty feed)
      fail    — raised an exception during fetch / parse / upsert
    """
    source: str
    status: str
    fetched: int = 0
    normalized: int = 0
    upserted: int = 0
    message: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class SourceConnector(ABC):
    """Self-contained connector for one supplementary data source.

    Each subclass owns the full lifecycle: check enabled flag, fetch raw
    data, normalize, upsert into the appropriate table, return result.

    Implementations must NOT raise across the boundary — catch and return
    ConnectorResult(status='fail') so one bad source can't poison the run.
    """
    name: str = "abstract"

    @abstractmethod
    def run(self) -> ConnectorResult: ...


@dataclass
class NormalizedEmployer:
    """Used by RCIP, Chamber, and any other employer-list source."""
    name: str
    industry: str | None = None
    primary_location: str | None = None
    designation_type: str | None = None       # 'rcip' | 'welcome_ssm_partner' | etc
    granted_at: date | None = None
    expires_at: date | None = None
    source: str = ""


@dataclass
class NormalizedServiceProvider:
    """Used by the SSM LIP services connector."""
    name: str
    service_type: str | None = None
    description: str | None = None
    url: str | None = None
    phone: str | None = None
    email: str | None = None
    address: str | None = None
    source: str = ""


@dataclass
class NormalizedDocument:
    """Used by the AWIC reports connector (knowledge.document target)."""
    source: str
    title: str
    url: str
    document_type: str | None = None
    published_date: date | None = None
    raw_text: str | None = None
    metadata: dict | None = None


# -------------------------------------------- Upsert helpers (PR 6A targets)
def upsert_employer_designation(
    employer_name: str,
    designation_type: str,
    *,
    granted_at: date | None = None,
    expires_at: date | None = None,
    source: str = "",
    industry: str | None = None,
    primary_location: str | None = None,
) -> str | None:
    """Upsert into core.employer + core.employer_designation.

    Returns employer_id, or None if the name was empty.
    """
    if not employer_name or not employer_name.strip():
        return None
    employer_id = upsert_employer(employer_name)
    if not employer_id:
        return None
    if industry or primary_location:
        with sync_cursor() as cur:
            cur.execute(
                """
                UPDATE core.employer SET
                    industry         = COALESCE(%s, industry),
                    primary_location = COALESCE(%s, primary_location)
                WHERE employer_id = %s
                """,
                (industry, primary_location, employer_id),
            )
    with sync_cursor() as cur:
        cur.execute(
            """
            INSERT INTO core.employer_designation
                (employer_id, designation_type, granted_at, expires_at, source)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (employer_id, designation_type) DO UPDATE SET
                granted_at = COALESCE(EXCLUDED.granted_at, core.employer_designation.granted_at),
                expires_at = COALESCE(EXCLUDED.expires_at, core.employer_designation.expires_at),
                source     = COALESCE(EXCLUDED.source, core.employer_designation.source)
            """,
            (employer_id, designation_type, granted_at, expires_at, source or None),
        )
    return employer_id


def upsert_service_provider(sp: NormalizedServiceProvider) -> str | None:
    if not sp.name:
        return None
    normalized = re.sub(r"\s+", " ", sp.name).strip().lower()
    if not normalized:
        return None
    with sync_cursor() as cur:
        cur.execute(
            """
            INSERT INTO core.service_provider
                (name, name_normalized, service_type, description, url, phone, email,
                 address, source, updated_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
            ON CONFLICT (name_normalized, source) DO UPDATE SET
                service_type = EXCLUDED.service_type,
                description  = EXCLUDED.description,
                url          = EXCLUDED.url,
                phone        = EXCLUDED.phone,
                email        = EXCLUDED.email,
                address      = EXCLUDED.address,
                is_active    = TRUE,
                updated_at   = NOW()
            RETURNING provider_id
            """,
            (sp.name, normalized, sp.service_type, sp.description, sp.url, sp.phone,
             sp.email, sp.address, sp.source or None),
        )
        row = cur.fetchone()
        return str(row["provider_id"]) if row else None


def upsert_document(doc: NormalizedDocument) -> str | None:
    """Insert/refresh a knowledge.document row keyed by (source, url)."""
    if not doc.title or not doc.url:
        return None
    with sync_cursor() as cur:
        cur.execute(
            """
            INSERT INTO knowledge.document
                (source, title, url, document_type, published_date, raw_text, metadata)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (source, url) DO UPDATE SET
                title          = EXCLUDED.title,
                document_type  = EXCLUDED.document_type,
                published_date = EXCLUDED.published_date,
                raw_text       = EXCLUDED.raw_text,
                metadata       = EXCLUDED.metadata
            RETURNING document_id
            """,
            (doc.source, doc.title, doc.url, doc.document_type, doc.published_date,
             doc.raw_text, json.dumps(doc.metadata) if doc.metadata else None),
        )
        row = cur.fetchone()
        return str(row["document_id"]) if row else None
