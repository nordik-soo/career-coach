"""Sault Area Hospital careers — reference parser.

URL is set via SAULT_AREA_HOSPITAL_URL. The selectors here are a starting
point against a typical hospital career page; verify against the live page
and tune before enabling in production.

This is a REFERENCE PARSER — its job is to demonstrate the pattern, not
to be the final implementation. The "warn: parser yielded zero jobs"
result from the base class is the signal to update selectors.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Iterable

from bs4 import BeautifulSoup

from skillbridge.ingest.base import NormalizedJob
from skillbridge.ingest.employers._base import EmployerConnector

log = logging.getLogger(__name__)


class SaultAreaHospitalConnector(EmployerConnector):
    source_name = "sault_area_hospital"
    config_key = "sault_area_hospital"
    name = "sault_area_hospital"

    # Selector starting points — tune to match the live page.
    POSTING_SELECTORS = [
        "div.career-posting",
        "div.job-listing",
        "li.job",
        "article.posting",
    ]
    TITLE_SELECTORS = ["h2", "h3", ".job-title", "a.title"]
    LINK_SELECTORS = ["a"]

    def parse(self, html: str) -> Iterable[NormalizedJob]:
        soup = BeautifulSoup(html, "lxml")
        cards = []
        for sel in self.POSTING_SELECTORS:
            cards = soup.select(sel)
            if cards:
                log.info("SaultAreaHospital: matched %d cards via '%s'", len(cards), sel)
                break
        if not cards:
            # Fallback: every anchor that looks like a job link.
            cards = [a for a in soup.select("a")
                     if a.get("href") and re.search(r"(career|job|posting)", a.get("href", ""), re.I)]
            log.info("SaultAreaHospital: fell back to anchor scan (%d candidates)", len(cards))

        seen: set[str] = set()
        for card in cards:
            title = self._first_text(card, self.TITLE_SELECTORS) or card.get_text(strip=True)
            title = title.strip() if title else ""
            link = card.select_one("a")
            href = link.get("href") if link else None
            if not title or len(title) < 4:
                continue
            sid = re.sub(r"\W+", "-", (href or title)).strip("-").lower()[:120]
            if not sid or sid in seen:
                continue
            seen.add(sid)
            # Step 1A (2026-07-15) honest-handling rule for employer-
            # specific connectors: employer identity does not prove a
            # posting's location. The scraper captures only a title +
            # a card-level HTML snippet — no per-posting location and
            # not enough text for description provenance. Emit missing
            # on both axes; the posting stays outside the live SSM
            # market until per-posting extraction ships.
            yield NormalizedJob(
                source=self.source_name,
                source_job_id=sid,
                title=title,
                employer="Sault Area Hospital",
                # Legacy `location` and `description` derived by
                # upsert_job. NEVER hardcode "Sault Ste. Marie" here.
                url=self._abs_url(href),
                posted_date=None,
                # Step 1A description axis.
                description_full=None,
                description_excerpt=None,
                description_evidence_status="missing",
                # Step 1A location axis. No per-posting location signal
                # from the card scraper. Honest posture: missing.
                source_location_text=None,
                source_coordinates=None,
                normalized_job_location=None,
                location_resolution_status="missing",
                location_provenance="none",
                raw_payload={"snippet": str(card)[:1000]},
            )

    @staticmethod
    def _first_text(card, selectors: list[str]) -> str | None:
        for sel in selectors:
            el = card.select_one(sel)
            if el and el.get_text(strip=True):
                return el.get_text(strip=True)
        return None

    def _abs_url(self, href: str | None) -> str | None:
        if not href:
            return None
        if href.startswith("http"):
            return href
        # Prefix-relative or root-relative — best effort. Operators should
        # set SAULT_AREA_HOSPITAL_URL to the page that contains absolute links.
        return href
