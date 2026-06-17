"""YMCA of Sault Ste. Marie career page — STUB parser.

TODO before enabling:
  1. Visit the live YMCA_SSM_URL.
  2. Identify HTML structure for postings.
  3. Replace empty parse() with real selectors.
  4. Set YMCA_SSM_ENABLED=true.
"""
from __future__ import annotations

import logging
from typing import Iterable

from skillbridge.ingest.base import NormalizedJob
from skillbridge.ingest.employers._base import EmployerConnector

log = logging.getLogger(__name__)


class YMCASSMConnector(EmployerConnector):
    source_name = "ymca_ssm"
    config_key = "ymca_ssm"
    name = "ymca_ssm"

    def parse(self, html: str) -> Iterable[NormalizedJob]:
        log.info("YMCASSM parser is a stub — fill in selectors before enabling")
        return
        yield  # pragma: no cover
