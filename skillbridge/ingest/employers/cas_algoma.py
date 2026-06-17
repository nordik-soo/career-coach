"""Children's Aid Society of Algoma career page — STUB parser.

TODO before enabling:
  1. Visit the live CAS_ALGOMA_URL.
  2. Identify HTML structure for postings.
  3. Replace empty parse() with real selectors.
  4. Set CAS_ALGOMA_ENABLED=true.
"""
from __future__ import annotations

import logging
from typing import Iterable

from skillbridge.ingest.base import NormalizedJob
from skillbridge.ingest.employers._base import EmployerConnector

log = logging.getLogger(__name__)


class CASAlgomaConnector(EmployerConnector):
    source_name = "cas_algoma"
    config_key = "cas_algoma"
    name = "cas_algoma"

    def parse(self, html: str) -> Iterable[NormalizedJob]:
        log.info("CASAlgoma parser is a stub — fill in selectors before enabling")
        return
        yield  # pragma: no cover
