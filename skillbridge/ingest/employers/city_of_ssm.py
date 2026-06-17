"""City of Sault Ste. Marie HR careers — reference parser.

Same shape as sault_area_hospital — selectors are starting points; tune
against the live page (https://saultstemarie.ca careers section).

If the City switches to a hosted ATS (Workday, BambooHR, etc.) the right
move is replace this parser with one that hits the ATS's stable JSON
endpoint instead of the rendered HTML.
"""
from __future__ import annotations

import logging
import re
from typing import Iterable

from bs4 import BeautifulSoup

from skillbridge.ingest.base import NormalizedJob
from skillbridge.ingest.employers._base import EmployerConnector

log = logging.getLogger(__name__)


class CityOfSSMHRConnector(EmployerConnector):
    source_name = "city_of_ssm_hr"
    config_key = "city_of_ssm_hr"
    name = "city_of_ssm_hr"

    # Selector starting points — tune to match the live page.
    POSTING_SELECTORS = [
        "div.job-posting",
        "li.career-listing",
        "tr.posting-row",
        "div.opening",
        "div.posting",
    ]

    def parse(self, html: str) -> Iterable[NormalizedJob]:
        soup = BeautifulSoup(html, "lxml")
        cards = []
        for sel in self.POSTING_SELECTORS:
            cards = soup.select(sel)
            if cards:
                log.info("CityOfSSM: matched %d cards via '%s'", len(cards), sel)
                break
        if not cards:
            cards = [a for a in soup.select("a")
                     if a.get("href") and re.search(r"(career|job|posting|employ)", a.get("href", ""), re.I)]
            log.info("CityOfSSM: fell back to anchor scan (%d candidates)", len(cards))

        seen: set[str] = set()
        for card in cards:
            link = card if card.name == "a" else card.select_one("a")
            href = link.get("href") if link else None
            title = (
                (card.select_one(".job-title") or card.select_one("h2")
                 or card.select_one("h3") or card)
                .get_text(strip=True)
            ) if card else ""
            title = (title or "").strip()
            if not title or len(title) < 4:
                continue
            sid = re.sub(r"\W+", "-", (href or title)).strip("-").lower()[:120]
            if not sid or sid in seen:
                continue
            seen.add(sid)
            yield NormalizedJob(
                source=self.source_name,
                source_job_id=sid,
                title=title,
                employer="City of Sault Ste. Marie",
                location="Sault Ste. Marie",
                url=href if (href and href.startswith("http")) else href,
                raw_payload={"snippet": str(card)[:1000]},
            )
