"""PUC Services (Sault Ste. Marie utility) career page — STUB parser.

TODO before enabling:
  1. Visit the live PUC_URL.
  2. Identify HTML structure for postings.
  3. Replace empty parse() with real selectors.
  4. Set PUC_ENABLED=true.
"""
from __future__ import annotations

import logging
from typing import Iterable

from skillbridge.ingest.base import NormalizedJob
from skillbridge.ingest.employers._base import EmployerConnector

log = logging.getLogger(__name__)


class PUCConnector(EmployerConnector):
    source_name = "puc"
    config_key = "puc"
    name = "puc"

    def parse(self, html: str) -> Iterable[NormalizedJob]:
        log.info("PUC parser is a stub — fill in selectors before enabling")
        return
        yield  # pragma: no cover
