"""AWIC Labour-Market Reports connector.

Loads AWIC report metadata into knowledge.document. PR 6A scope: store URL +
title + ingested_at only. No indicator extraction, no embedding, no RAG.
That is a future PR once we understand the actual report formats AWIC
publishes.

Two input modes (both optional, both used if available):
  AWIC_REPORTS_URL   — a manifest (CSV/JSON) listing report URLs + titles
  AWIC_DATA_FEED_URL — a structured indicator feed (future use; currently
                       just acknowledged but not parsed)

The manifest is the only producing path in PR 6A. Once AWIC partnership
conversations converge on a real export shape, this connector grows up.
"""
from __future__ import annotations

import csv
import io
import logging
from typing import Iterable

import httpx

from config import AWIC_DATA_FEED_URL, AWIC_ENABLED, AWIC_REPORTS_URL
from skillbridge.ingest.base import (
    ConnectorResult,
    NormalizedDocument,
    SourceConnector,
    upsert_document,
)

log = logging.getLogger(__name__)


class AWICReportsConnector(SourceConnector):
    name = "awic"

    def run(self) -> ConnectorResult:
        if not AWIC_ENABLED:
            return ConnectorResult(
                source=self.name, status="skipped",
                message="AWIC_ENABLED is false",
            )
        reports_set = AWIC_REPORTS_URL and not AWIC_REPORTS_URL.startswith("PLACEHOLDER")
        feed_set = AWIC_DATA_FEED_URL and not AWIC_DATA_FEED_URL.startswith("PLACEHOLDER")
        if not reports_set and not feed_set:
            return ConnectorResult(
                source=self.name, status="skipped",
                message="AWIC_REPORTS_URL and AWIC_DATA_FEED_URL are placeholders",
            )

        fetched = upserted = 0
        try:
            if reports_set:
                for row in self._fetch_manifest(AWIC_REPORTS_URL):
                    fetched += 1
                    title = (row.get("title") or row.get("name") or "").strip()
                    url = (row.get("url") or "").strip()
                    if not title or not url:
                        continue
                    doc = NormalizedDocument(
                        source="awic",
                        title=title,
                        url=url,
                        document_type=(row.get("type") or "report"),
                    )
                    if upsert_document(doc):
                        upserted += 1
            if feed_set:
                # PR 6A: acknowledge but do not parse — indicator extraction is
                # a future PR with its own contract.
                log.info(
                    "AWIC_DATA_FEED_URL is set but indicator extraction is "
                    "not yet implemented (PR 6A is registry-only)"
                )
        except Exception as e:
            log.exception("AWIC fetch failed")
            return ConnectorResult(
                source=self.name, status="fail",
                fetched=fetched, upserted=upserted, message=str(e)[:200],
            )

        status = "success" if upserted > 0 else "warn"
        return ConnectorResult(
            source=self.name, status=status,
            fetched=fetched, normalized=fetched, upserted=upserted,
            message=f"loaded {upserted} document(s) from AWIC manifest",
        )

    def _fetch_manifest(self, url: str) -> Iterable[dict]:
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "").lower()
            if "json" in ct:
                data = resp.json()
                payload = (
                    data if isinstance(data, list)
                    else data.get("reports") or data.get("documents") or []
                )
                yield from payload
            else:
                reader = csv.DictReader(io.StringIO(resp.text))
                yield from reader
