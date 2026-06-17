"""Algoma Steel career page — STUB parser.

TODO before enabling:
  1. Visit the live ALGOMA_STEEL_URL.
  2. Identify the HTML structure for postings (or the ATS JSON endpoint if any).
  3. Replace the empty parse() with real selectors yielding NormalizedJob.
  4. Set ALGOMA_STEEL_ENABLED=true.
"""
from __future__ import annotations

import logging
from typing import Iterable

from skillbridge.ingest.base import NormalizedJob
from skillbridge.ingest.employers._base import EmployerConnector

log = logging.getLogger(__name__)


class AlgomaSteelConnector(EmployerConnector):
    source_name = "algoma_steel"
    config_key = "algoma_steel"
    name = "algoma_steel"

    def parse(self, html: str) -> Iterable[NormalizedJob]:
        log.info("AlgomaSteel parser is a stub — fill in selectors before enabling")
        return
        yield  # pragma: no cover
