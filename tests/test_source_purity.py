"""Source-purity invariant — the SSM-only product rule lives in the database.

PR 7A added core.approved_job_source as a lookup table with a FOREIGN KEY
constraint on core.job_posting.source. Federal sources (jobbank, statcan,
census) are not listed and cannot be inserted.

This is the fourth durable invariant alongside consent-boundary, delete-
cascade, and route-guard. If anyone re-introduces a federal source in the
future, these tests fail loudly.
"""
from __future__ import annotations

import pytest
from psycopg.errors import ForeignKeyViolation

from skillbridge.db import sync_cursor


# Sources that must NEVER appear in core.approved_job_source.
PROHIBITED_SOURCES = {"jobbank", "job_bank", "statcan", "stat_canada", "census",
                      "opportunext", "lmi", "esdc_lmi"}

# Sources required to be present (the SSM-allowed list from PR 7A,
# extended in 2026-07 by AWIC v1 which added `awic_jobs` as a peer
# local JD source alongside `sccc` -- see BREAKING.md).
REQUIRED_APPROVED_SOURCES = {
    "sccc", "awic_jobs", "welcome_ssm", "city_ssm", "partner_csv",
    "sault_area_hospital", "city_of_ssm_hr", "algoma_steel",
    "sault_college_careers", "algoma_u_careers", "puc",
    "group_health_centre", "ymca_ssm", "cas_algoma", "adsab", "school_board",
}


def _approved_sources() -> set[str]:
    with sync_cursor() as cur:
        cur.execute("SELECT source FROM core.approved_job_source")
        return {r["source"] for r in cur.fetchall()}


def _fk_exists_on_job_posting_source() -> bool:
    with sync_cursor() as cur:
        cur.execute(
            """
            SELECT 1
              FROM pg_constraint con
              JOIN pg_class c ON c.oid = con.conrelid
              JOIN pg_namespace n ON n.oid = c.relnamespace
             WHERE con.contype = 'f'
               AND n.nspname = 'core'
               AND c.relname = 'job_posting'
               AND con.confrelid = 'core.approved_job_source'::regclass
            """
        )
        return cur.fetchone() is not None


# ============================================================== TESTS
def test_fk_constraint_exists():
    """core.job_posting.source must reference core.approved_job_source."""
    assert _fk_exists_on_job_posting_source(), (
        "Foreign key from core.job_posting.source to "
        "core.approved_job_source(source) is missing. The source-purity "
        "guarantee is not enforced. Restore the FK in sql/schema.sql."
    )


def test_no_federal_sources_in_approved_list():
    """No national/federal source name may be in the approved list."""
    approved = _approved_sources()
    leaked = PROHIBITED_SOURCES & approved
    assert not leaked, (
        f"Prohibited federal source(s) appeared in core.approved_job_source: "
        f"{sorted(leaked)}. SkillBridge SSM is local-only — see BREAKING.md."
    )


def test_required_ssm_sources_present():
    """Every SSM source that's supposed to be approved must be in the table."""
    approved = _approved_sources()
    missing = REQUIRED_APPROVED_SOURCES - approved
    assert not missing, (
        f"Required SSM sources are missing from core.approved_job_source: "
        f"{sorted(missing)}. Re-run sql/schema.sql or check the seed INSERT."
    )


def test_federal_source_insertion_is_blocked():
    """Direct attempt to insert a federal source must fail the FK constraint."""
    with sync_cursor() as cur:
        try:
            cur.execute(
                """
                INSERT INTO core.job_posting
                    (source, source_job_id, title, posted_date, is_active)
                VALUES ('jobbank', 'test_should_fail', 'Test', CURRENT_DATE, TRUE)
                """
            )
        except ForeignKeyViolation:
            return  # expected
        except Exception as e:
            # Some psycopg versions wrap; accept any insertion failure.
            assert "foreign key" in str(e).lower() or "violates" in str(e).lower(), (
                f"Expected FK violation, got: {e!r}"
            )
            return
    pytest.fail("INSERT with source='jobbank' succeeded — the FK is not enforcing source purity")


def test_every_orchestrator_connector_uses_approved_source():
    """Every connector registered in the orchestrator must produce rows
    whose source is in core.approved_job_source. We check this by name —
    the EmployerConnector.source_name field is the value that goes into
    core.job_posting.source.
    """
    from skillbridge.ingest.employers import ALL_EMPLOYER_CONNECTORS
    from skillbridge.ingest.partner_upload import KNOWN_PARTNER_PREFIXES
    from skillbridge.ingest.partners import ALL_CONNECTORS as ALL_PARTNER_CONNECTORS

    approved = _approved_sources()

    # Employer connectors: each has a source_name class attribute.
    employer_sources = {c.source_name for c in ALL_EMPLOYER_CONNECTORS}
    missing = employer_sources - approved
    assert not missing, (
        f"Employer connector source_name(s) not in approved_job_source: "
        f"{sorted(missing)}"
    )

    # Partner-feed connectors: each has a source_name class attribute.
    partner_sources = {c.source_name for c in ALL_PARTNER_CONNECTORS}
    missing = partner_sources - approved
    assert not missing, (
        f"Partner connector source_name(s) not in approved_job_source: "
        f"{sorted(missing)}"
    )

    # Partner-upload connector: every known prefix must be approved
    # (unknown prefixes fall through to 'partner_csv' which IS approved).
    missing = KNOWN_PARTNER_PREFIXES - approved
    assert not missing, (
        f"Partner upload prefix(es) not in approved_job_source: "
        f"{sorted(missing)}"
    )
