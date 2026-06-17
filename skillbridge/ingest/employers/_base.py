"""EmployerConnector base — public-page parser pattern.

Each subclass overrides:
  source_name   — must match a row in core.approved_job_source
  config_key    — name in config.EMPLOYER_SOURCES
  parse(html)   — yields NormalizedJob items

The base handles:
  - enabled / placeholder gating
  - polite HTTP fetch (timeout, User-Agent identifying the project)
  - uniform ConnectorResult emission
  - raw payload archival + core.job_posting upsert
  - never raises across the boundary

Subclasses MUST set source_name to a value present in core.approved_job_source
or the upsert will fail the FK constraint. The source-purity test catches
this at the registry level.
"""
from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Iterable

import httpx

from config import EMPLOYER_SOURCES
from skillbridge.ingest.base import (
    ConnectorResult,
    NormalizedJob,
    SourceConnector,
    upsert_job,
    write_raw_job,
)

log = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "SkillBridge-SSM/0.1 (research; nordik.org)"


def _cfg(name: str):
    for s in EMPLOYER_SOURCES:
        if s.name == name:
            return s
    return None


class EmployerConnector(SourceConnector):
    """Self-contained connector for one SSM employer's career page."""

    source_name: str = "abstract"   # MUST match a row in core.approved_job_source
    config_key: str = "abstract"
    name: str = "abstract"          # SourceConnector.name (mirrors source_name)
    timeout_seconds: int = 30

    def run(self) -> ConnectorResult:
        cfg = _cfg(self.config_key)
        if cfg is None or not cfg.enabled:
            return ConnectorResult(
                source=self.source_name, status="skipped",
                message=f"{self.config_key.upper()}_ENABLED is false",
            )
        if not cfg.url or cfg.url.startswith("PLACEHOLDER"):
            return ConnectorResult(
                source=self.source_name, status="skipped",
                message=f"{self.config_key.upper()}_URL is placeholder",
            )

        try:
            html = self._fetch(cfg.url)
        except Exception as e:
            log.warning("%s fetch failed: %s", self.source_name, e)
            return ConnectorResult(
                source=self.source_name, status="fail",
                message=f"fetch failed: {str(e)[:150]}",
            )

        try:
            jobs = list(self.parse(html))
        except Exception as e:
            log.exception("%s parser raised", self.source_name)
            return ConnectorResult(
                source=self.source_name, status="fail",
                message=f"parser error: {str(e)[:150]}",
            )

        fetched = len(jobs)
        upserted = 0
        for job in jobs:
            # Force source to this connector's approved name. Defence in depth:
            # even if parse() forgets to set it, the FK keeps us honest.
            job.source = self.source_name
            try:
                write_raw_job(job)
                if upsert_job(job):
                    upserted += 1
            except Exception as e:
                log.warning("%s upsert failed for %s: %s",
                            self.source_name, job.source_job_id, e)
                continue

        if upserted == 0 and fetched == 0:
            return ConnectorResult(
                source=self.source_name, status="warn",
                fetched=0, upserted=0,
                message="page fetched but parser yielded zero jobs — verify selectors",
            )
        return ConnectorResult(
            source=self.source_name,
            status="success" if upserted > 0 else "warn",
            fetched=fetched, normalized=fetched, upserted=upserted,
        )

    # ------------------------------------------------- HTTP
    def _fetch(self, url: str) -> str:
        with httpx.Client(
            timeout=self.timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": DEFAULT_USER_AGENT},
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.text

    # ------------------------------------------------- override
    @abstractmethod
    def parse(self, html: str) -> Iterable[NormalizedJob]: ...
