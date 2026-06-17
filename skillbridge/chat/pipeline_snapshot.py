"""AR-9.feat.coach-tiers CP1 step 12 — pipeline snapshot.

A small frozen snapshot the deterministic fallback uses ONLY for the
empty-tier message ("Nothing on the board matches yet — current view
has X active postings, last refreshed Y."). The snapshot must NOT
affect matching or tier selection.

Two layers, deliberately separate:
  - `_format_publish_at_text` — pure formatter. Takes a datetime or
    None and returns the user-facing string (or None when the input
    is None). Testable with no DB.
  - `fetch_pipeline_snapshot` — runs the two DB queries
    (count of `core.v_current_job`, latest `pipeline.dataset_state`
    row for `pointer_key='current_jobs'`) and returns the snapshot.

None handling: when `pipeline.dataset_state` has no row for
`pointer_key='current_jobs'`, `last_publish_at_text` is None — never
substituted with the current time.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from skillbridge.db import sync_cursor


_AMERICA_TORONTO = ZoneInfo("America/Toronto")
_PUBLISH_AT_FORMAT = "%Y-%m-%d %H:%M ET"


@dataclass(frozen=True)
class PipelineSnapshot:
    """Side-channel snapshot of the ingestion pipeline's published
    state. Carried separately from the tier evidence and the view —
    consumed ONLY by the deterministic fallback's empty-tier branch.

    Fields:
      total_active_jobs    — count of `core.v_current_job` rows at
                              query time.
      last_publish_at_text — formatted timestamp string
                              ("YYYY-MM-DD HH:MM ET") for the latest
                              `pipeline.dataset_state` row with
                              `pointer_key='current_jobs'`. None when
                              no such row exists. NEVER substituted
                              with the current time.
    """
    total_active_jobs: int
    last_publish_at_text: str | None


# =========================================================================
# Pure formatter — DB-free, isolated for unit tests
# =========================================================================
def _format_publish_at_text(dt: datetime | None) -> str | None:
    """Format an `America/Toronto` timestamp as
    "YYYY-MM-DD HH:MM ET". Naive datetimes are treated as UTC (the
    `pipeline.dataset_state.published_at` column is TIMESTAMPTZ; psycopg
    typically returns either a tz-aware UTC datetime or, when the
    column is read as naive, the underlying UTC value).

    Returns None when the input is None — no substitution to the
    current time. This preserves the contract that the absence of a
    publish record is a distinct signal from a known refresh point.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_et = dt.astimezone(_AMERICA_TORONTO)
    return dt_et.strftime(_PUBLISH_AT_FORMAT)


# =========================================================================
# DB fetcher — kept separate so tests can exercise the formatter
# without touching Postgres.
# =========================================================================
def fetch_pipeline_snapshot() -> PipelineSnapshot:
    """Run the two queries and return a `PipelineSnapshot`.

    Query 1: count of `core.v_current_job`.
    Query 2: `published_at` from `pipeline.dataset_state` where
             `pointer_key = 'current_jobs'`.

    Both queries are read-only and bounded; safe for chat-turn use.
    """
    with sync_cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM core.v_current_job")
        count_row = cur.fetchone()
        total = int(count_row["n"]) if count_row else 0

        cur.execute(
            "SELECT published_at "
            "FROM pipeline.dataset_state "
            "WHERE pointer_key = 'current_jobs'"
        )
        publish_row = cur.fetchone()
    raw_publish_at = publish_row["published_at"] if publish_row else None
    return PipelineSnapshot(
        total_active_jobs=total,
        last_publish_at_text=_format_publish_at_text(raw_publish_at),
    )
