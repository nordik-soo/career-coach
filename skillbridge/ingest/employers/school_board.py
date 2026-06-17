"""Local school board(s) career page — STUB parser.

This is intentionally a single aggregated source. SSM has multiple boards
(Algoma DSB, HSCDSB, the Conseil scolaire catholique/public boards). For
MVP they all map to source='school_board'. If a specific board needs
separate tracking later, add a new approved_job_source row + new connector.

TODO before enabling:
  1. Decide which board's URL goes here (or aggregate via multiple URLs).
  2. Visit the live page(s).
  3. Replace empty parse() with real selectors.
  4. Set SCHOOL_BOARD_ENABLED=true.
"""
from __future__ import annotations

import logging
from typing import Iterable

from skillbridge.ingest.base import NormalizedJob
from skillbridge.ingest.employers._base import EmployerConnector

log = logging.getLogger(__name__)


class SchoolBoardConnector(EmployerConnector):
    source_name = "school_board"
    config_key = "school_board"
    name = "school_board"

    def parse(self, html: str) -> Iterable[NormalizedJob]:
        log.info("SchoolBoard parser is a stub — fill in selectors before enabling")
        return
        yield  # pragma: no cover
