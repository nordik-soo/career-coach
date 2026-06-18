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

# Step 9 (2026-06-17): user-facing sector names for NOC 2021 broad
# categories. The first digit of a noc_code identifies the broad
# occupational category per Statistics Canada's NOC 2021 structure.
# Names are chosen for coach voice — short, plain English, no jargon.
# Unknown / NULL noc_code rows contribute to the "other" bucket which
# the renderer skips (only NAMED categories surface).
_NOC_BROAD_CATEGORY_NAMES: dict[str, str] = {
    "0": "management",
    "1": "business and finance",
    "2": "sciences and engineering",
    "3": "healthcare",
    "4": "education and social services",
    "5": "arts and culture",
    "6": "sales and service",
    "7": "trades and transport",
    "8": "agriculture and natural resources",
    "9": "manufacturing and utilities",
}


@dataclass(frozen=True)
class PipelineSnapshot:
    """Side-channel snapshot of the ingestion pipeline's published
    state. Carried separately from the tier evidence and the view —
    consumed by the deterministic fallback's empty-tier branch AND
    (Step 9, 2026-06-17) by the SHAPE 2 enhanced no-match fallback
    so the user gets a Sault Ste. Marie market panorama instead of
    a dead-end "sorry, nothing fits" response.

    Fields:
      total_active_jobs    — count of `core.v_current_job` rows at
                              query time.
      last_publish_at_text — formatted timestamp string
                              ("YYYY-MM-DD HH:MM ET") for the latest
                              `pipeline.dataset_state` row with
                              `pointer_key='current_jobs'`. None when
                              no such row exists. NEVER substituted
                              with the current time.
      top_sectors          — Step 9 (2026-06-17): top NOC-major-group
                              sector names by posting count (e.g.,
                              ("healthcare", "trades", "admin")).
                              Empty tuple when noc_code is NULL on
                              every row OR the table is empty. Used by
                              the SHAPE 2 enhanced no-match fallback
                              to give the user a market panorama
                              ("most postings are in X, Y, Z").
      top_employers        — Step 9 (2026-06-17): top hiring employer
                              names by posting count (e.g.,
                              ("Health Sciences North", "Algoma Family
                              Services", "CMHA")). Empty tuple when
                              the table is empty. Used by the SHAPE 2
                              enhanced fallback to surface concrete
                              employer names the user can recognize.
    """
    total_active_jobs: int
    last_publish_at_text: str | None
    top_sectors: tuple[str, ...] = ()
    top_employers: tuple[str, ...] = ()


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
    """Run the queries and return a `PipelineSnapshot`.

    Query 1: count of `core.v_current_job`.
    Query 2: `published_at` from `pipeline.dataset_state` where
             `pointer_key = 'current_jobs'`.
    Query 3 (Step 9, 2026-06-17): top 3 NOC broad categories by
             posting count — for the SHAPE 2 enhanced market summary.
             Rows with NULL noc_code are excluded (they can't be
             grouped). Unknown first-digit values (rare; shouldn't
             happen in normal data) are also skipped at rendering.
    Query 4 (Step 9, 2026-06-17): top 3 hiring employers by posting
             count — concrete names the user can recognize. Rows
             with NULL employer are excluded.

    All queries are read-only and bounded; safe for chat-turn use.
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

        # Step 9: top sectors (NOC broad-category buckets).
        cur.execute(
            "SELECT LEFT(noc_code, 1) AS broad_cat, COUNT(*) AS n "
            "FROM core.v_current_job "
            "WHERE noc_code IS NOT NULL AND LENGTH(noc_code) >= 1 "
            "GROUP BY broad_cat "
            "ORDER BY n DESC "
            "LIMIT 5"
        )
        sector_rows = cur.fetchall()

        # Step 9: top hiring employers.
        cur.execute(
            "SELECT employer, COUNT(*) AS n "
            "FROM core.v_current_job "
            "WHERE employer IS NOT NULL AND TRIM(employer) <> '' "
            "GROUP BY employer "
            "ORDER BY n DESC "
            "LIMIT 3"
        )
        employer_rows = cur.fetchall()

    # Map broad-category codes to user-facing names. Skip unknown
    # codes (defensive — shouldn't happen with NOC 2021 data) and
    # cap at 3 named sectors total.
    top_sectors = tuple(
        _NOC_BROAD_CATEGORY_NAMES[row["broad_cat"]]
        for row in sector_rows
        if row["broad_cat"] in _NOC_BROAD_CATEGORY_NAMES
    )[:3]

    top_employers = tuple(
        row["employer"] for row in employer_rows
        if isinstance(row["employer"], str) and row["employer"].strip()
    )

    return PipelineSnapshot(
        total_active_jobs=total,
        last_publish_at_text=_format_publish_at_text(raw_publish_at),
        top_sectors=top_sectors,
        top_employers=top_employers,
    )
