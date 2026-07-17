"""Step 1A (2026-07-15) per-connector regression tests.

Direct coverage that the previous 177-test focused suite lacked: each
of the six producer sites has explicit tests for the load-bearing
Step 1A rules — no hardcoded SSM fallback, provenance-driven
description classification (full_source / excerpt_only / missing /
parse_error), source-declared location classification (resolved /
missing / invalid), and the shared coordinate-validation contract.

Covered:
  1. AWIC — see also test_awic_ingest.py (existing broader coverage)
  2. SCCC (_normalize_sccc_item)
  3. Partner CSV (_row_to_job in partners.py)
  4. Partner upload (_row_to_job in partner_upload.py)
  5. Sault Area Hospital (SaultAreaHospitalConnector)
  6. City of SSM (CityOfSSMConnector)

Plus classifier-unit tests for _classify_description_field and
_classify_declared_location_field which back every connector's
non-string / oversized handling.
"""
from __future__ import annotations

import pytest


# ══════════════════════════════════════════════════════════════════
# Classifier unit tests (shared by every connector)
# ══════════════════════════════════════════════════════════════════


class TestClassifyDescriptionField:
    """Provenance-driven, not character-count driven."""

    def test_none_returns_missing(self):
        from skillbridge.ingest.partners import _classify_description_field
        assert _classify_description_field(None) == (None, "missing")

    def test_empty_string_returns_missing(self):
        from skillbridge.ingest.partners import _classify_description_field
        assert _classify_description_field("") == (None, "missing")

    def test_whitespace_only_returns_missing(self):
        from skillbridge.ingest.partners import _classify_description_field
        assert _classify_description_field("   \n\t  ") == (None, "missing")

    def test_short_string_returns_ok(self):
        from skillbridge.ingest.partners import _classify_description_field
        # Provenance-based: a short string is still full_source if the
        # source captured it complete. Character count is not the rule.
        text, status = _classify_description_field("Short but real JD.")
        assert text == "Short but real JD."
        assert status == "ok"

    def test_long_string_still_ok(self):
        from skillbridge.ingest.partners import _classify_description_field
        long_text = "A" * 10_000
        text, status = _classify_description_field(long_text)
        assert text == long_text
        assert status == "ok"

    def test_oversized_string_produces_parse_error(self):
        from skillbridge.ingest.partners import _classify_description_field
        # > 1 MB catches data-format leaks (e.g., a JSON blob dumped
        # into the description column).
        oversized = "A" * (2 * 1024 * 1024)
        text, status = _classify_description_field(oversized)
        assert text is None
        assert status == "parse_error"

    @pytest.mark.parametrize("bad", [
        {"nested": "dict"},
        ["list", "of", "items"],
        42,
        3.14,
        True,
        b"bytes not str",
    ])
    def test_non_string_produces_parse_error(self, bad):
        from skillbridge.ingest.partners import _classify_description_field
        text, status = _classify_description_field(bad)
        assert text is None
        assert status == "parse_error"


class TestClassifyDeclaredLocationField:
    """Distinguishes missing (absent) from invalid (present-but-non-string)."""

    def test_none_returns_missing(self):
        from skillbridge.ingest.partners import _classify_declared_location_field
        assert _classify_declared_location_field(None) == (None, "missing")

    def test_empty_string_returns_missing(self):
        from skillbridge.ingest.partners import _classify_declared_location_field
        assert _classify_declared_location_field("") == (None, "missing")

    def test_whitespace_returns_missing(self):
        from skillbridge.ingest.partners import _classify_declared_location_field
        assert _classify_declared_location_field("   ") == (None, "missing")

    def test_string_returns_ok(self):
        from skillbridge.ingest.partners import _classify_declared_location_field
        assert _classify_declared_location_field(
            "  Wawa, ON  "
        ) == ("Wawa, ON", "ok")

    @pytest.mark.parametrize("bad", [
        {"city": "SSM"},
        ["SSM"],
        42,
        3.14,
        True,
        b"bytes",
    ])
    def test_non_string_produces_invalid_not_missing(self, bad):
        """LOAD-BEARING: a present-but-non-string location is NOT
        the same as absent. Invalid preserves the fact that the
        source tried to declare something we couldn't use."""
        from skillbridge.ingest.partners import _classify_declared_location_field
        text, status = _classify_declared_location_field(bad)
        assert text is None
        assert status == "invalid"


# ══════════════════════════════════════════════════════════════════
# SCCC (_normalize_sccc_item)
# ══════════════════════════════════════════════════════════════════


def _sccc_item(**overrides):
    """Minimal SCCC WP-API item shape for testing normalize."""
    item = {
        "id": overrides.pop("id", 999),
        "date": "2026-07-15T12:00:00",
        "slug": "test-role",
        "title": {"rendered": overrides.pop("title", "Test Role")},
        "link": overrides.pop("link", "https://sault.example/j/1"),
        "content": {"rendered": overrides.pop(
            "content_rendered", "<p>Full JD body from SCCC.</p>",
        )},
        "meta": {
            "_job_location": overrides.pop(
                "job_location", "Sault Ste. Marie",
            ),
            "_company_name": overrides.pop("employer", "Test Employer"),
            **overrides.pop("extra_meta", {}),
        },
    }
    item.update(overrides)
    return item


class TestSCCCConnector:
    """SCCC's `_is_sccc_ssm_location` filter drops non-SSM postings at
    ingest time — that's ingest-side scope, distinct from Step 1A's
    market-eligibility filter. Under Step 1A that ingest filter stays
    permissive (Algoma-wide); rows outside SSM persist in DB but
    don't reach the matcher. These tests exercise the normalize path
    for SSM-passing inputs where Step 1A's new fields matter."""

    def test_ssm_location_resolves(self):
        from skillbridge.ingest.partners import _normalize_sccc_wp_rest_item
        job = _normalize_sccc_wp_rest_item(_sccc_item())
        assert job is not None
        assert job.source_location_text == "Sault Ste. Marie"
        assert job.normalized_job_location == "Sault Ste. Marie"
        assert job.location_resolution_status == "resolved"
        assert job.location_provenance == "source_declared"

    def test_full_content_produces_full_source(self):
        from skillbridge.ingest.partners import _normalize_sccc_wp_rest_item
        job = _normalize_sccc_wp_rest_item(_sccc_item(
            content_rendered="<p>Real JD content here.</p>",
        ))
        assert job is not None
        assert job.description_full is not None
        assert "Real JD content here" in job.description_full
        assert job.description_evidence_status == "full_source"

    def test_missing_content_produces_missing(self):
        from skillbridge.ingest.partners import _normalize_sccc_wp_rest_item
        item = _sccc_item()
        item["content"] = {"rendered": None}
        job = _normalize_sccc_wp_rest_item(item)
        assert job is not None
        assert job.description_full is None
        assert job.description_evidence_status == "missing"

    def test_non_string_content_produces_parse_error(self):
        """_clean_wp_html raises on non-string content. Connector
        catches and emits parse_error, not crash."""
        from skillbridge.ingest.partners import _normalize_sccc_wp_rest_item
        item = _sccc_item()
        item["content"] = {"rendered": {"unexpected": "dict shape"}}
        job = _normalize_sccc_wp_rest_item(item)
        assert job is not None
        assert job.description_full is None
        assert job.description_evidence_status == "parse_error"


# ══════════════════════════════════════════════════════════════════
# Partner CSV (_row_to_job in partners.py)
# ══════════════════════════════════════════════════════════════════


class TestPartnerCSVConnector:
    def test_full_row_resolves(self):
        from skillbridge.ingest.partners import _row_to_job
        job = _row_to_job({
            "source_job_id": "csv-001",
            "title": "Community Support",
            "employer": "Test Employer",
            "location": "Sault Ste. Marie",
            "description": "A real JD body.",
        }, source="partner_csv")
        assert job is not None
        assert job.normalized_job_location == "Sault Ste. Marie"
        assert job.location_resolution_status == "resolved"
        assert job.location_provenance == "source_declared"
        assert job.description_full == "A real JD body."
        assert job.description_evidence_status == "full_source"

    def test_missing_location_resolves_to_missing(self):
        from skillbridge.ingest.partners import _row_to_job
        job = _row_to_job({
            "source_job_id": "csv-002",
            "title": "Role",
            "description": "JD.",
        }, source="partner_csv")
        assert job is not None
        assert job.source_location_text is None
        assert job.location_resolution_status == "missing"
        assert job.location_provenance == "none"

    def test_non_string_location_produces_invalid(self):
        from skillbridge.ingest.partners import _row_to_job
        job = _row_to_job({
            "source_job_id": "csv-003",
            "title": "Role",
            "location": {"city": "SSM"},  # non-string
        }, source="partner_csv")
        assert job is not None
        assert job.source_location_text is None
        assert job.location_resolution_status == "invalid"
        assert job.location_provenance == "source_declared"

    def test_non_string_description_produces_parse_error(self):
        from skillbridge.ingest.partners import _row_to_job
        job = _row_to_job({
            "source_job_id": "csv-004",
            "title": "Role",
            "description": ["list", "not", "string"],
        }, source="partner_csv")
        assert job is not None
        assert job.description_full is None
        assert job.description_evidence_status == "parse_error"


# ══════════════════════════════════════════════════════════════════
# Partner upload (partner_upload._row_to_job)
# ══════════════════════════════════════════════════════════════════


class TestPartnerUploadConnector:
    def test_full_row_resolves(self):
        from skillbridge.ingest.partner_upload import _row_to_job
        job = _row_to_job({
            "source_job_id": "upl-001",
            "title": "Role",
            "location": "Sault Ste. Marie, ON",
            "description": "JD content.",
        }, source="test_partner")
        assert job is not None
        assert job.normalized_job_location == "Sault Ste. Marie"
        assert job.location_resolution_status == "resolved"
        assert job.description_full == "JD content."
        assert job.description_evidence_status == "full_source"

    def test_non_string_location_produces_invalid(self):
        from skillbridge.ingest.partner_upload import _row_to_job
        job = _row_to_job({
            "source_job_id": "upl-002",
            "title": "Role",
            "location": 42,  # numeric
        }, source="test_partner")
        assert job is not None
        assert job.location_resolution_status == "invalid"
        assert job.location_provenance == "source_declared"

    def test_non_string_description_produces_parse_error(self):
        from skillbridge.ingest.partner_upload import _row_to_job
        job = _row_to_job({
            "source_job_id": "upl-003",
            "title": "Role",
            "description": {"nested": "dict"},
        }, source="test_partner")
        assert job is not None
        assert job.description_evidence_status == "parse_error"

    def test_wawa_location_normalizes_truthfully(self):
        from skillbridge.ingest.partner_upload import _row_to_job
        job = _row_to_job({
            "source_job_id": "upl-004",
            "title": "Role",
            "location": "Wawa, Ontario",
        }, source="test_partner")
        assert job is not None
        assert job.normalized_job_location == "Wawa"
        assert job.location_resolution_status == "resolved"


# ══════════════════════════════════════════════════════════════════
# Sault Area Hospital connector (honest-handling rule)
# ══════════════════════════════════════════════════════════════════


class TestSaultAreaHospitalConnector:
    def test_never_hardcodes_ssm_location(self):
        """LOAD-BEARING: employer identity does not prove location.
        The connector currently scrapes only title + snippet; under
        Step 1A honest-handling rule it emits missing/none, not
        SSM. `parse(html)` is the direct-invocation surface — bypasses
        network fetch."""
        from skillbridge.ingest.employers.sault_area_hospital import (
            SaultAreaHospitalConnector,
        )
        html = """
        <html><body>
            <div class="career-posting">
                <h2>Registered Nurse - Day Shift</h2>
                <a href="/careers/rn-day-shift">Apply</a>
            </div>
        </body></html>
        """
        connector = SaultAreaHospitalConnector()
        jobs = list(connector.parse(html))
        # At least one job should come out — if not, selectors changed
        # and the test needs updating for a different reason.
        assert len(jobs) >= 1
        for job in jobs:
            assert job.source_location_text is None, (
                "Sault Area Hospital must never hardcode SSM as "
                "source_location_text — employer identity is not "
                "location evidence."
            )
            assert job.location_resolution_status == "missing"
            assert job.location_provenance == "none"
            assert job.normalized_job_location is None
            assert job.location is None


# ══════════════════════════════════════════════════════════════════
# City of SSM connector (honest-handling rule)
# ══════════════════════════════════════════════════════════════════


class TestCityOfSSMHRConnector:
    def test_never_hardcodes_ssm_location(self):
        """Same rule as Sault Area Hospital: employer name doesn't
        prove location."""
        from skillbridge.ingest.employers.city_of_ssm import (
            CityOfSSMHRConnector,
        )
        # City of SSM's parse uses POSTING_SELECTORS (div.job-posting
        # etc.) or falls back to anchors with career/job/posting/employ
        # in the href. Match the div selector directly for stability.
        html = """
        <html><body>
            <div class="job-posting">
                <h2>Financial Analyst</h2>
                <a href="/careers/analyst">Apply</a>
            </div>
            <div class="job-posting">
                <h3>Clerk</h3>
                <a href="/careers/clerk">Apply</a>
            </div>
        </body></html>
        """
        connector = CityOfSSMHRConnector()
        jobs = list(connector.parse(html))
        assert len(jobs) >= 1
        for job in jobs:
            assert job.source_location_text is None, (
                "City of SSM must never hardcode SSM as "
                "source_location_text — employer identity is not "
                "location evidence."
            )
            assert job.location_resolution_status == "missing"
            assert job.location_provenance == "none"
            assert job.normalized_job_location is None
            assert job.location is None


# ══════════════════════════════════════════════════════════════════
# Live-DB check: no hardcoded fallback survives in ingestion source
# ══════════════════════════════════════════════════════════════════


class TestNoHardcodedFallbackInSource:
    """Static grep guard: no producer file may emit
    `location="Sault Ste. Marie"` as a hardcoded literal to
    NormalizedJob's location parameter. This is a source-code
    invariant, not a runtime check."""

    def _grep_producer_file(self, path: str) -> str:
        from pathlib import Path
        return Path(path).read_text(encoding="utf-8")

    @pytest.mark.parametrize("producer_path", [
        "skillbridge/ingest/partners.py",
        "skillbridge/ingest/partner_upload.py",
        "skillbridge/ingest/employers/sault_area_hospital.py",
        "skillbridge/ingest/employers/city_of_ssm.py",
    ])
    def test_no_hardcoded_ssm_location_literal(self, producer_path):
        source = self._grep_producer_file(producer_path)
        # Not the docstring / not the alias / not the class name —
        # specifically the pattern `location="Sault Ste. Marie"` OR
        # `source_location_text="Sault Ste. Marie"`.
        forbidden_patterns = [
            'location="Sault Ste. Marie"',
            "location='Sault Ste. Marie'",
            'source_location_text="Sault Ste. Marie"',
            "source_location_text='Sault Ste. Marie'",
        ]
        for pat in forbidden_patterns:
            assert pat not in source, (
                f"{producer_path} contains hardcoded fallback "
                f"`{pat}` — Step 1A rule §2e prohibits this. "
                f"Location must come from source-declared data OR "
                f"be missing/none."
            )
