"""SSM Local Immigration Partnership (LIP) service directory connector.

Loads the LIP's roster of settlement / employment / training / counselling
services for newcomers into core.service_provider. Activates the moment
SSM_LIP_SERVICES_URL points at a real CSV/JSON feed.

PR 6A: data only. The chat layer doesn't yet surface service providers as
fallbacks — that wiring lands in a future PR. The data being available
makes that wiring trivial.
"""
from __future__ import annotations

import csv
import io
import logging
from typing import Iterable

import httpx

from config import SSM_LIP_ENABLED, SSM_LIP_SERVICES_URL
from skillbridge.ingest.base import (
    ConnectorResult,
    NormalizedServiceProvider,
    SourceConnector,
    upsert_service_provider,
)

log = logging.getLogger(__name__)


class SSMLIPServicesConnector(SourceConnector):
    name = "ssm_lip"

    def run(self) -> ConnectorResult:
        if not SSM_LIP_ENABLED:
            return ConnectorResult(
                source=self.name, status="skipped",
                message="SSM_LIP_ENABLED is false",
            )
        if not SSM_LIP_SERVICES_URL or SSM_LIP_SERVICES_URL.startswith("PLACEHOLDER"):
            return ConnectorResult(
                source=self.name, status="skipped",
                message="SSM_LIP_SERVICES_URL is placeholder",
            )

        fetched = upserted = 0
        try:
            for row in self._fetch(SSM_LIP_SERVICES_URL):
                fetched += 1
                sp = NormalizedServiceProvider(
                    name=(row.get("name") or "").strip(),
                    service_type=(row.get("service_type") or row.get("type") or "").strip() or None,
                    description=(row.get("description") or "").strip() or None,
                    url=(row.get("url") or "").strip() or None,
                    phone=(row.get("phone") or "").strip() or None,
                    email=(row.get("email") or "").strip() or None,
                    address=(row.get("address") or "").strip() or None,
                    source="ssm_lip",
                )
                if not sp.name:
                    continue
                if upsert_service_provider(sp):
                    upserted += 1
        except Exception as e:
            log.exception("SSM LIP fetch failed")
            return ConnectorResult(
                source=self.name, status="fail",
                fetched=fetched, upserted=upserted, message=str(e)[:200],
            )

        status = "success" if upserted > 0 else "warn"
        return ConnectorResult(
            source=self.name, status=status,
            fetched=fetched, normalized=fetched, upserted=upserted,
        )

    def _fetch(self, url: str) -> Iterable[dict]:
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "").lower()
            if "json" in ct:
                data = resp.json()
                payload = data if isinstance(data, list) else (
                    data.get("services") or data.get("providers") or data.get("data") or []
                )
                yield from payload
            else:
                reader = csv.DictReader(io.StringIO(resp.text))
                yield from reader
