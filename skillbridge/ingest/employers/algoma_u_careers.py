"""Algoma University HR / careers — STUB parser.

NOTE: distinct from program ingestion. This is Algoma U's HR career page
(jobs hosted BY the university). Programs of study live in
skillbridge/ingest/training.py.

TODO before enabling:
  1. Visit the live ALGOMA_U_CAREERS_URL.
  2. Identify HTML structure for postings.
  3. Replace empty parse() with real selectors.
  4. Set ALGOMA_U_CAREERS_ENABLED=true.
"""
from __future__ import annotations

import logging
from typing import Iterable

from skillbridge.ingest.base import NormalizedJob
from skillbridge.ingest.employers._base import EmployerConnector

log = logging.getLogger(__name__)


class AlgomaUCareersConnector(EmployerConnector):
    source_name = "algoma_u_careers"
    config_key = "algoma_u_careers"
    name = "algoma_u_careers"

    def parse(self, html: str) -> Iterable[NormalizedJob]:
        log.info("AlgomaUCareers parser is a stub — fill in selectors before enabling")
        return
        yield  # pragma: no cover
