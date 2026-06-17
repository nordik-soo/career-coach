"""Partner CSV upload bridge — audited.

Replacement for the simpler PartnerCsvConnector from PR 6A. Adds:
  - SHA-256 file dedup (same file re-dropped twice doesn't double-insert)
  - Per-partner row sourcing — filename prefix determines source value
  - Audit trail in pipeline.partner_upload

Filename convention:
    <partner_name>_YYYY-MM-DD.csv
where <partner_name> is one of the entries in core.approved_job_source.
Unknown prefixes fall through to source='partner_csv'.

Expected CSV columns (case-tolerant):
    source_job_id, title, employer, location, description, url,
    posted_date, closing_date, salary_text, employment_type, noc_code

Rows with missing title or source_job_id are skipped silently. The
pipeline.partner_upload row records how many rows were seen vs upserted
so quality issues surface in /v1/admin/data-status.
"""
from __future__ import annotations

import csv
import hashlib
import logging
import re
from pathlib import Path

from config import JOB_SOURCES
from skillbridge.db import sync_cursor
from skillbridge.ingest.base import (
    ConnectorResult,
    NormalizedJob,
    SourceConnector,
    parse_date_loose,
    parse_float_loose,
    upsert_job,
    write_raw_job,
)

log = logging.getLogger(__name__)


# Filename prefixes that map to specific approved sources.
KNOWN_PARTNER_PREFIXES = {
    "sccc",
    "welcome_ssm",
    "city_ssm",
    "sault_area_hospital",
    "city_of_ssm_hr",
    "algoma_steel",
    "sault_college_careers",
    "algoma_u_careers",
    "puc",
    "group_health_centre",
    "ymca_ssm",
    "cas_algoma",
    "adsab",
    "school_board",
}


def _config():
    for s in JOB_SOURCES:
        if s.name == "partner_csv":
            return s
    return None


def _partner_from_filename(stem: str) -> str:
    """Extract the source value from '<partner>_YYYY-MM-DD' style filenames.

    Falls back to 'partner_csv' for unrecognized prefixes — those still
    land in core.job_posting (partner_csv is approved) but lose the
    per-partner attribution.
    """
    cleaned = re.sub(r"[_-]?\d{4}-\d{2}-\d{2}.*$", "", stem).strip("_-").lower()
    if cleaned in KNOWN_PARTNER_PREFIXES:
        return cleaned
    # Try the first underscore-separated token.
    head = cleaned.split("_")[0] if cleaned else ""
    if head in KNOWN_PARTNER_PREFIXES:
        return head
    return "partner_csv"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _row_to_job(row: dict, *, source: str) -> NormalizedJob | None:
    title = (row.get("title") or row.get("Job Title") or "").strip()
    source_job_id = (
        row.get("source_job_id") or row.get("id") or row.get("job_id")
        or row.get("Job ID") or row.get("posting_id") or ""
    )
    source_job_id = str(source_job_id).strip()
    if not title or not source_job_id:
        return None
    return NormalizedJob(
        source=source,
        source_job_id=source_job_id,
        title=title,
        employer=(row.get("employer") or row.get("Employer") or "").strip() or None,
        location=(row.get("location") or row.get("Location") or "").strip() or None,
        region_code=(row.get("region_code") or "").strip() or None,
        description=(row.get("description") or row.get("Job Description") or "").strip() or None,
        url=(row.get("url") or row.get("URL") or "").strip() or None,
        posted_date=parse_date_loose(row.get("posted_date") or row.get("Date Posted")),
        closing_date=parse_date_loose(row.get("closing_date") or row.get("Closing Date")),
        salary_text=(row.get("salary_text") or row.get("Salary") or "").strip() or None,
        salary_low=parse_float_loose(row.get("salary_low") or row.get("Min Wage")),
        salary_high=parse_float_loose(row.get("salary_high") or row.get("Max Wage")),
        employment_type=(row.get("employment_type") or row.get("Employment Type") or "").strip() or None,
        remote_flag=None,
        noc_code=(row.get("noc_code") or row.get("NOC Code") or "").strip() or None,
        raw_payload=dict(row),
    )


def _already_processed(file_hash: str) -> bool:
    with sync_cursor() as cur:
        cur.execute(
            "SELECT 1 FROM pipeline.partner_upload WHERE file_sha256 = %s",
            (file_hash,),
        )
        return cur.fetchone() is not None


def _record_upload(partner: str, filename: str, file_hash: str,
                   row_count: int, upserted: int, error: str | None) -> None:
    with sync_cursor() as cur:
        cur.execute(
            """
            INSERT INTO pipeline.partner_upload
                (partner_name, filename, file_sha256, processed_at,
                 row_count, upserted_count, error)
            VALUES (%s, %s, %s, NOW(), %s, %s, %s)
            ON CONFLICT (file_sha256) DO NOTHING
            """,
            (partner, filename, file_hash, row_count, upserted, error),
        )


class PartnerUploadConnector(SourceConnector):
    """Reads any *.csv file from PARTNER_CSV_DIR, dedupes via SHA-256, logs
    every file to pipeline.partner_upload."""
    name = "partner_csv"

    def run(self) -> ConnectorResult:
        cfg = _config()
        if cfg is None or not cfg.enabled:
            return ConnectorResult(
                source="partner_csv", status="skipped",
                message="PARTNER_CSV_ENABLED is false",
            )
        directory = Path(cfg.url or "./data/partner_uploads")
        if not directory.exists():
            return ConnectorResult(
                source="partner_csv", status="skipped",
                message=f"directory {directory} does not exist",
            )

        files = sorted(directory.glob("*.csv"))
        if not files:
            return ConnectorResult(
                source="partner_csv", status="skipped",
                message=f"no CSVs in {directory}",
            )

        total_fetched = total_upserted = files_processed = files_skipped_dup = 0
        for csv_path in files:
            file_hash = _sha256(csv_path)
            if _already_processed(file_hash):
                files_skipped_dup += 1
                log.info("partner_csv: skipping %s (already processed by hash)", csv_path.name)
                continue
            partner = _partner_from_filename(csv_path.stem)
            log.info("partner_csv: reading %s -> source=%s", csv_path.name, partner)
            row_count = upserted = 0
            error: str | None = None
            try:
                with csv_path.open(encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row_count += 1
                        job = _row_to_job(row, source=partner)
                        if not job:
                            continue
                        write_raw_job(job)
                        if upsert_job(job):
                            upserted += 1
            except Exception as e:
                log.exception("partner_csv: failed reading %s", csv_path)
                error = str(e)[:300]
            _record_upload(partner, csv_path.name, file_hash, row_count, upserted, error)
            total_fetched += row_count
            total_upserted += upserted
            files_processed += 1

        msg = (
            f"processed {files_processed} new file(s), "
            f"skipped {files_skipped_dup} duplicate(s); "
            f"{total_upserted}/{total_fetched} rows upserted"
        )
        status = "success" if total_upserted > 0 else (
            "warn" if files_processed > 0 else "skipped"
        )
        return ConnectorResult(
            source="partner_csv", status=status,
            fetched=total_fetched, normalized=total_fetched,
            upserted=total_upserted, message=msg,
        )
