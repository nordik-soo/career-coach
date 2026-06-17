"""Algoma District Services Administration Board (ADSAB) career page — STUB.

TODO before enabling:
  1. Visit the live ADSAB_URL.
  2. Identify HTML structure for postings.
  3. Replace empty parse() with real selectors.
  4. Set ADSAB_ENABLED=true.
"""
from __future__ import annotations

import logging
from typing import Iterable

from skillbridge.ingest.base import NormalizedJob
from skillbridge.ingest.employers._base import EmployerConnector

log = logging.getLogger(__name__)


class ADSABConnector(EmployerConnector):
    source_name = "adsab"
    config_key = "adsab"
    name = "adsab"

    def parse(self, html: str) -> Iterable[NormalizedJob]:
        log.info("ADSAB parser is a stub — fill in selectors before enabling")
        return
        yield  # pragma: no cover
