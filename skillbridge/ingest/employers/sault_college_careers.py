"""Sault College HR / careers — STUB parser.

NOTE: distinct from training-program ingestion. This is Sault College's
HR career page (jobs hosted BY the college). Training programs (jobs the
college trains FOR) live in skillbridge/ingest/training.py.

TODO before enabling:
  1. Visit the live SAULT_COLLEGE_CAREERS_URL.
  2. Identify the HTML structure for postings.
  3. Replace the empty parse() with real selectors yielding NormalizedJob.
  4. Set SAULT_COLLEGE_CAREERS_ENABLED=true.
"""
from __future__ import annotations

import logging
from typing import Iterable

from skillbridge.ingest.base import NormalizedJob
from skillbridge.ingest.employers._base import EmployerConnector

log = logging.getLogger(__name__)


class SaultCollegeCareersConnector(EmployerConnector):
    source_name = "sault_college_careers"
    config_key = "sault_college_careers"
    name = "sault_college_careers"

    def parse(self, html: str) -> Iterable[NormalizedJob]:
        log.info("SaultCollegeCareers parser is a stub — fill in selectors before enabling")
        return
        yield  # pragma: no cover
