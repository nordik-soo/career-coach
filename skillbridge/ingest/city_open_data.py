"""City of Sault Ste. Marie open data / GIS connector — STUB.

Lower priority than Tier-1 partner sources. Useful for:
  - geography enrichment (postal-FSA mapping inside the city)
  - public-sector employer enumeration

Currently a stub that activates when CITY_SSM_OPEN_DATA_ENABLED=true and
CITY_SSM_OPEN_DATA_URL points at a real feed.
"""
from __future__ import annotations

from config import CITY_SSM_OPEN_DATA_ENABLED, CITY_SSM_OPEN_DATA_URL
from skillbridge.ingest.base import ConnectorResult, SourceConnector


class CityOpenDataConnector(SourceConnector):
    name = "city_open_data"

    def run(self) -> ConnectorResult:
        if not CITY_SSM_OPEN_DATA_ENABLED:
            return ConnectorResult(
                source=self.name, status="skipped",
                message="CITY_SSM_OPEN_DATA_ENABLED is false",
            )
        if not CITY_SSM_OPEN_DATA_URL or CITY_SSM_OPEN_DATA_URL.startswith("PLACEHOLDER"):
            return ConnectorResult(
                source=self.name, status="skipped",
                message="CITY_SSM_OPEN_DATA_URL is placeholder",
            )
        return ConnectorResult(
            source=self.name, status="warn",
            message=(
                "City of SSM open data parser not yet implemented "
                "(PR 6A is registry-only; activate parser in a future PR)"
            ),
        )
