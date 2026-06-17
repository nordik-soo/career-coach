"""Sault Chamber of Commerce business directory connector — STUB.

When wired (PR 6B+):
  - SAULT_CHAMBER_DIRECTORY_URL points at a CSV/JSON feed
  - Each entry upserts core.employer with industry + primary_location
  - Optionally adds a 'sault_chamber_member' designation for enrichment

Currently a stub that returns a uniform ConnectorResult so the registry
stays complete. Activate by setting SAULT_CHAMBER_ENABLED=true and pointing
SAULT_CHAMBER_DIRECTORY_URL at a real feed.
"""
from __future__ import annotations

from config import SAULT_CHAMBER_DIRECTORY_URL, SAULT_CHAMBER_ENABLED
from skillbridge.ingest.base import ConnectorResult, SourceConnector


class SaultChamberConnector(SourceConnector):
    name = "sault_chamber"

    def run(self) -> ConnectorResult:
        if not SAULT_CHAMBER_ENABLED:
            return ConnectorResult(
                source=self.name, status="skipped",
                message="SAULT_CHAMBER_ENABLED is false",
            )
        if not SAULT_CHAMBER_DIRECTORY_URL or SAULT_CHAMBER_DIRECTORY_URL.startswith("PLACEHOLDER"):
            return ConnectorResult(
                source=self.name, status="skipped",
                message="SAULT_CHAMBER_DIRECTORY_URL is placeholder",
            )
        return ConnectorResult(
            source=self.name, status="warn",
            message=(
                "Sault Chamber parser not yet implemented "
                "(PR 6A is registry-only; activate parser in a future PR)"
            ),
        )
