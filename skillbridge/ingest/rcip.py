"""RCIP Designated Employer connector.

Loads the Rural Community Immigration Pilot designated-employer list into
core.employer + core.employer_designation (designation_type='rcip').

Two input modes (URL wins if both set and non-placeholder):
  RCIP_EMPLOYER_LIST_URL — HTTP feed (CSV or JSON)
  RCIP_EMPLOYER_LIST_CSV — local file path

PR 6A scope: data only. No matching boost, no chat badge — that is PR 6B.
"""
from __future__ import annotations

import csv
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

import httpx

from config import (
    RCIP_EMPLOYER_LIST_CSV,
    RCIP_EMPLOYER_LIST_URL,
    RCIP_ENABLED,
)
from skillbridge.ingest.base import (
    ConnectorResult,
    SourceConnector,
    upsert_employer_designation,
)

log = logging.getLogger(__name__)


def _parse_date(s: str | None):
    if not s:
        return None
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


class RCIPEmployerConnector(SourceConnector):
    name = "rcip"

    def run(self) -> ConnectorResult:
        if not RCIP_ENABLED:
            return ConnectorResult(
                source=self.name, status="skipped",
                message="RCIP_ENABLED is false",
            )

        try:
            rows, source_used = self._load_rows()
        except Exception as e:
            log.exception("RCIP fetch failed")
            return ConnectorResult(
                source=self.name, status="fail", message=str(e)[:200],
            )

        if rows is None:
            return ConnectorResult(
                source=self.name, status="skipped",
                message="RCIP_EMPLOYER_LIST_URL is placeholder and no local CSV found",
            )

        rows_list = list(rows)
        fetched = len(rows_list)
        upserted = 0
        for row in rows_list:
            name = (row.get("employer_name") or row.get("name") or "").strip()
            if not name:
                continue
            employer_id = upsert_employer_designation(
                employer_name=name,
                designation_type="rcip",
                granted_at=_parse_date(row.get("granted_at")),
                expires_at=_parse_date(row.get("expires_at")),
                industry=(row.get("industry") or None),
                primary_location=(row.get("location") or row.get("primary_location") or None),
                source="rcip",
            )
            if employer_id:
                upserted += 1

        status = "success" if upserted > 0 else "warn"
        msg = f"loaded {upserted} RCIP employer(s) from {source_used}"
        if upserted == 0:
            msg = f"no usable rows in {source_used}"
        return ConnectorResult(
            source=self.name, status=status,
            fetched=fetched, normalized=fetched, upserted=upserted, message=msg,
        )

    # ---------------------------------------------- internal
    def _load_rows(self) -> tuple[Iterable[dict] | None, str]:
        if RCIP_EMPLOYER_LIST_URL and not RCIP_EMPLOYER_LIST_URL.startswith("PLACEHOLDER"):
            return self._fetch_remote(RCIP_EMPLOYER_LIST_URL), RCIP_EMPLOYER_LIST_URL
        csv_path = Path(RCIP_EMPLOYER_LIST_CSV) if RCIP_EMPLOYER_LIST_CSV else None
        if csv_path and csv_path.exists():
            return self._read_csv(csv_path), str(csv_path)
        return None, ""

    def _fetch_remote(self, url: str) -> Iterable[dict]:
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "").lower()
            if "json" in ct:
                data = resp.json()
                payload = data if isinstance(data, list) else (
                    data.get("employers") or data.get("data") or []
                )
                yield from payload
            else:
                reader = csv.DictReader(io.StringIO(resp.text))
                yield from reader

    def _read_csv(self, path: Path) -> Iterable[dict]:
        with path.open(encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            yield from reader
