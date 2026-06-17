"""Group Health Centre career page — STUB parser.

TODO before enabling:
  1. Visit the live GROUP_HEALTH_CENTRE_URL.
  2. Identify HTML structure for postings.
  3. Replace empty parse() with real selectors.
  4. Set GROUP_HEALTH_CENTRE_ENABLED=true.
"""
from __future__ import annotations

import logging
from typing import Iterable

from skillbridge.ingest.base import NormalizedJob
from skillbridge.ingest.employers._base import EmployerConnector

log = logging.getLogger(__name__)


class GroupHealthCentreConnector(EmployerConnector):
    source_name = "group_health_centre"
    config_key = "group_health_centre"
    name = "group_health_centre"

    def parse(self, html: str) -> Iterable[NormalizedJob]:
        log.info("GroupHealthCentre parser is a stub — fill in selectors before enabling")
        return
        yield  # pragma: no cover
