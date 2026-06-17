"""Training / skill-improvement resource connectors.

All STUBS. Wire in the parser once the source page structure is confirmed.
Northland and SCCC services need partner conversations; Sault College and
Algoma U can be parsed from their public program pages (with care for ToS).
"""
from __future__ import annotations

import logging
from typing import Iterable

import httpx
from bs4 import BeautifulSoup

from config import TRAINING_SOURCES
from skillbridge.ingest.base import (
    NormalizedTrainingResource,
    TrainingConnector,
    derive_duration_band,
)

log = logging.getLogger(__name__)


def _cfg(name: str):
    for s in TRAINING_SOURCES:
        if s.name == name:
            return s
    return None


def _fetch_html(url: str) -> str | None:
    try:
        with httpx.Client(timeout=60, follow_redirects=True,
                          headers={"User-Agent": "SkillBridge-SSM/0.1 (research)"}) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.text
    except Exception as e:
        log.warning("Fetch failed %s: %s", url, e)
        return None


# ----------------------------------------------------------- Sault College
class SaultCollegeConnector(TrainingConnector):
    """Sault College full-time programs + continuing education.

    Real implementation should:
      - GET SAULT_COLLEGE_PROGRAMS_URL
      - parse program cards (title, description, duration, URL)
      - emit NormalizedTrainingResource per program
    """
    provider_name = "Sault College"

    def fetch(self) -> Iterable[NormalizedTrainingResource]:
        cfg = _cfg("sault_college")
        if cfg is None or not cfg.enabled:
            log.info("Sault College disabled — skipping")
            return
        html = _fetch_html(cfg.url)
        if not html:
            return
        # TODO: parse program list. Their HTML structure changes; do this once
        # you can verify selectors against the live page.
        soup = BeautifulSoup(html, "lxml")
        for card in soup.select("a[href*='/program/']"):
            title = (card.get_text() or "").strip()
            href = card.get("href")
            if not title or not href:
                continue
            url = href if href.startswith("http") else f"https://www.saultcollege.ca{href}"
            yield NormalizedTrainingResource(
                provider=self.provider_name,
                title=title,
                description=None,
                url=url,
                location="Sault Ste. Marie",
                delivery_mode="in_person",
                resource_type="program",
            )


# ----------------------------------------------------------- Algoma University
class AlgomaUConnector(TrainingConnector):
    """Algoma University programs of study.

    Stub: similar parser to Sault College once selectors are confirmed.
    """
    provider_name = "Algoma University"

    def fetch(self) -> Iterable[NormalizedTrainingResource]:
        cfg = _cfg("algoma_u")
        if cfg is None or not cfg.enabled:
            log.info("Algoma U disabled — skipping")
            return
        log.info("Algoma U connector is a stub — add program parser")
        return
        yield  # pragma: no cover


# --------------------------------------------- Northland Adult Learning Centre
class NorthlandConnector(TrainingConnector):
    """Northland Adult Learning Centre (ESL, computer, essential skills).

    The most useful Tier-1 partner for newcomer training. Until they provide
    a feed, drop a CSV into ./data/partner_uploads/northland_<date>.csv with
    columns: title, description, url, delivery_mode, cost_text, duration_text.
    """
    provider_name = "Northland Adult Learning Centre"

    def fetch(self) -> Iterable[NormalizedTrainingResource]:
        cfg = _cfg("northland")
        if cfg is None or not cfg.enabled:
            log.info("Northland disabled — skipping")
            return
        if not cfg.url or cfg.url.startswith("PLACEHOLDER"):
            log.warning("NORTHLAND_RESOURCES_URL is placeholder — set it in .env")
            return
        log.info("Northland connector is a stub — wire parser once URL is real")
        return
        yield  # pragma: no cover


# ---------------------------------------------------- SCCC services / workshops
class SCCCServicesConnector(TrainingConnector):
    """SCCC workshops + counselling. Treated as training-like resources."""
    provider_name = "Sault Community Career Centre"

    def fetch(self) -> Iterable[NormalizedTrainingResource]:
        cfg = _cfg("sccc_services")
        if cfg is None or not cfg.enabled:
            log.info("SCCC services disabled — skipping")
            return
        if not cfg.url or cfg.url.startswith("PLACEHOLDER"):
            log.warning("SCCC_SERVICES_URL is placeholder — set it in .env")
            return
        log.info("SCCC services connector is a stub")
        return
        yield  # pragma: no cover


ALL_TRAINING_CONNECTORS: list[type[TrainingConnector]] = [
    SaultCollegeConnector,
    AlgomaUConnector,
    NorthlandConnector,
    SCCCServicesConnector,
]
