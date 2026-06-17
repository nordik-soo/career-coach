"""Read-only NOC coverage audit.

Round-34 NOC observability slice. Produces a structured report the
operator can pipe into stdout or feed into CI:

  - occupation_count
  - synonym_count_by_source_lang
  - active_jobs_total
  - active_jobs_with_noc
  - active_jobs_missing_noc
  - missing_noc_jobs_resolvable_now
  - truly_unresolved_titles  (grouped by normalized title)
  - coverage_pct
  - structural_failures      (empty_synonym_table | empty_occupation_table)

Discipline:
  - READ-ONLY. The audit never writes. Re-runnable from any
    environment with READ access to reference + core tables.
  - Resolver state distinction (empty_table | exact | fuzzy |
    unresolved) is honored so deployment-prerequisite failures
    aren't silently lumped in with vocabulary misses.
  - Production-readiness exit code (`--strict`) returns nonzero
    ONLY for structural failures, NEVER for low coverage. Coverage
    is a reported metric, not a hard gate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from skillbridge.db import sync_cursor
from skillbridge.match.occupation import (
    NocResolutionState,
    resolve_title_to_noc_with_state,
)

log = logging.getLogger(__name__)


# Cap on titles included in the truly-unresolved grouping. The report
# is meant to give a sense of the backlog, not catalogue every single
# title; a deep tail just bloats the output. The grouped counts above
# the cap aren't lost (they show up in `active_jobs_missing_noc` and
# `missing_noc_jobs_resolvable_now`); only the per-title detail is
# bounded.
_MAX_UNRESOLVED_GROUPS = 50
_TITLE_NORMALIZE_CAP = 80


@dataclass(frozen=True)
class UnresolvedTitleGroup:
    """One row in the truly-unresolved table."""
    normalized_title: str           # length-capped lower-trimmed
    count: int                       # how many active jobs share this title
    # Best similarity returned by pg_trgm's `%` operator (round-35
    # rename: previously "best_fuzzy_similarity"). The operator
    # applies a session threshold (default 0.3); rows below that
    # threshold never reach us. So this is "candidate" similarity --
    # what the SQL layer surfaced for this title -- not corpus-wide
    # best. 0.0 means no candidate above the session threshold.
    candidate_similarity: float


@dataclass
class NocCoverageReport:
    occupation_count: int = 0
    synonym_count_total: int = 0
    synonym_count_by_source_lang: list[dict[str, Any]] = field(default_factory=list)
    active_jobs_total: int = 0
    active_jobs_with_noc: int = 0
    active_jobs_missing_noc: int = 0
    missing_noc_jobs_resolvable_now: int = 0
    truly_unresolved_titles: list[UnresolvedTitleGroup] = field(default_factory=list)
    coverage_pct: float = 0.0
    structural_failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "occupation_count": self.occupation_count,
            "synonym_count_total": self.synonym_count_total,
            "synonym_count_by_source_lang": list(self.synonym_count_by_source_lang),
            "active_jobs_total": self.active_jobs_total,
            "active_jobs_with_noc": self.active_jobs_with_noc,
            "active_jobs_missing_noc": self.active_jobs_missing_noc,
            "missing_noc_jobs_resolvable_now": self.missing_noc_jobs_resolvable_now,
            "truly_unresolved_titles": [
                {"title": g.normalized_title,
                 "count": g.count,
                 "candidate_similarity": g.candidate_similarity}
                for g in self.truly_unresolved_titles
            ],
            "coverage_pct": self.coverage_pct,
            "structural_failures": list(self.structural_failures),
        }


def inspect_noc_coverage() -> NocCoverageReport:
    """Run the read-only audit and return a structured report.

    Never writes. Per-row resolver calls dispatch through
    `resolve_title_to_noc_with_state` so the four-state distinction
    is preserved end to end.
    """
    report = NocCoverageReport()

    with sync_cursor() as cur:
        # ---- Reference-side counts ----
        cur.execute("SELECT COUNT(*) AS n FROM reference.occupation")
        row = cur.fetchone()
        report.occupation_count = int((row or {}).get("n", 0))

        cur.execute("SELECT COUNT(*) AS n FROM reference.occupation_title_synonym")
        row = cur.fetchone()
        report.synonym_count_total = int((row or {}).get("n", 0))

        if report.synonym_count_total > 0:
            cur.execute(
                """
                SELECT source, lang, COUNT(*) AS n
                  FROM reference.occupation_title_synonym
                 GROUP BY source, lang
                 ORDER BY source, lang
                """
            )
            report.synonym_count_by_source_lang = [
                {"source": r["source"], "lang": r["lang"], "count": int(r["n"])}
                for r in cur.fetchall()
            ]

        # Structural failures: deployment prerequisites not met.
        if report.occupation_count == 0:
            report.structural_failures.append("empty_occupation_table")
        if report.synonym_count_total == 0:
            report.structural_failures.append("empty_synonym_table")

        # ---- Job-side counts ----
        cur.execute(
            """
            SELECT COUNT(*) AS n
              FROM core.job_posting
             WHERE is_active = TRUE
            """
        )
        row = cur.fetchone()
        report.active_jobs_total = int((row or {}).get("n", 0))

        cur.execute(
            """
            SELECT COUNT(*) AS n
              FROM core.job_posting
             WHERE is_active = TRUE
               AND noc_code IS NOT NULL
               AND noc_code <> ''
            """
        )
        row = cur.fetchone()
        report.active_jobs_with_noc = int((row or {}).get("n", 0))
        report.active_jobs_missing_noc = (
            report.active_jobs_total - report.active_jobs_with_noc
        )

        # ---- Resolvable-now sweep: GROUP BY in SQL ----
        # Round-35: aggregate missing-NOC jobs by normalized title in
        # the database, then resolve each UNIQUE title once. The prior
        # implementation iterated per row, which:
        #   - hit the resolver N times for N duplicates of the same title
        #   - flooded the miss log with repeated lines per duplicate
        # SQL grouping reduces both to one resolve per distinct title.
        cur.execute(
            """
            SELECT LOWER(TRIM(title)) AS norm_title, COUNT(*) AS n
              FROM core.job_posting
             WHERE is_active = TRUE
               AND (noc_code IS NULL OR noc_code = '')
               AND title IS NOT NULL
               AND TRIM(title) <> ''
             GROUP BY LOWER(TRIM(title))
            """
        )
        missing_groups: list[tuple[str, int]] = [
            (r["norm_title"], int(r["n"])) for r in cur.fetchall()
        ]

    # The resolver opens its own cursor; close the outer one first
    # so per-title resolution doesn't dogpile a single connection.
    # Telemetry is suppressed because the report IS the audit output;
    # logging duplicates the signal and risks user_target/job_backfill
    # confusion at the log sink.
    unresolved_groups: list[UnresolvedTitleGroup] = []
    resolvable_now = 0
    for norm_title, count in missing_groups:
        res = resolve_title_to_noc_with_state(
            norm_title,
            call_source="job_backfill",
            emit_telemetry=False,
        )
        if res.state in (NocResolutionState.EXACT, NocResolutionState.FUZZY):
            resolvable_now += count
            continue
        if res.state == NocResolutionState.UNRESOLVED:
            unresolved_groups.append(UnresolvedTitleGroup(
                normalized_title=norm_title[:_TITLE_NORMALIZE_CAP],
                count=count,
                candidate_similarity=res.candidate_similarity,
            ))
            continue
        # Round-36 four-state contract: EMPTY_TABLE means these
        # titles were NEVER ASSESSED (no synonyms to compare against).
        # Reporting them as "truly unresolved" would blur the
        # deployment-prerequisite failure into the vocabulary-miss
        # bucket. The structural_failures list already records the
        # empty-table condition; titles drop here.
        # (NocResolutionState.EMPTY_TABLE -- explicit no-op.)

    # Sort by frequency desc, cap to bound output.
    unresolved_groups.sort(key=lambda g: -g.count)
    report.missing_noc_jobs_resolvable_now = resolvable_now
    report.truly_unresolved_titles = unresolved_groups[:_MAX_UNRESOLVED_GROUPS]

    if report.active_jobs_total > 0:
        report.coverage_pct = round(
            100.0 * report.active_jobs_with_noc / report.active_jobs_total, 2,
        )

    return report


def format_noc_coverage_report(report: NocCoverageReport) -> str:
    """Pretty-print the report for stdout. Stable enough to grep."""
    lines: list[str] = []
    lines.append("=== NOC coverage audit ===")
    lines.append(f"Occupation rows:               {report.occupation_count}")
    lines.append(f"Synonym rows (total):          {report.synonym_count_total}")
    if report.synonym_count_by_source_lang:
        lines.append("Synonym rows by source x lang:")
        for entry in report.synonym_count_by_source_lang:
            lines.append(
                f"  - source={entry['source']:<20s} "
                f"lang={entry['lang']:<2s}  count={entry['count']}"
            )
    lines.append("")
    lines.append(f"Active jobs (total):           {report.active_jobs_total}")
    lines.append(f"Active jobs with NOC:          {report.active_jobs_with_noc}")
    lines.append(f"Active jobs missing NOC:       {report.active_jobs_missing_noc}")
    lines.append(
        f"Missing-NOC jobs resolvable:   {report.missing_noc_jobs_resolvable_now}"
    )
    lines.append(f"Coverage:                      {report.coverage_pct}%")
    lines.append("")
    if report.structural_failures:
        lines.append("STRUCTURAL FAILURES:")
        for f in report.structural_failures:
            lines.append(f"  ! {f}")
        lines.append("")
    if report.truly_unresolved_titles:
        lines.append(
            f"Truly unresolved titles (top {len(report.truly_unresolved_titles)}, "
            f"capped at {_MAX_UNRESOLVED_GROUPS}):"
        )
        for g in report.truly_unresolved_titles:
            lines.append(
                f"  - {g.normalized_title!r:<60s} "
                f"count={g.count:<3d} candidate_sim={g.candidate_similarity:.2f}"
            )
    else:
        lines.append("(No unresolved titles after resolver sweep.)")
    return "\n".join(lines)


def cli_inspect_noc_coverage(*, strict: bool = False) -> int:
    """CLI entry: print the report to stdout, return exit code.

    Exit code semantics (production-readiness rule):
      - `--strict` mode: nonzero iff `structural_failures` is non-empty
        (the deployment prerequisite isn't met). NEVER fails on low
        coverage -- that's a vocabulary signal, not a structural one.
      - Default mode (no `--strict`): always returns 0. Used for ad-hoc
        operator audits where exit code shouldn't gate a script.
    """
    report = inspect_noc_coverage()
    print(format_noc_coverage_report(report))
    if strict and report.structural_failures:
        log.warning(
            "noc_coverage_structural_failures: %s",
            report.structural_failures,
        )
        return 1
    return 0
