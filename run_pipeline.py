"""SkillBridge SSM pipeline CLI.

Usage:
    python run_pipeline.py --all                       # full daily refresh
    python run_pipeline.py --schema                    # apply sql/schema.sql once
    python run_pipeline.py --reference                 # load NOC/OaSIS/regulated
    python run_pipeline.py --jobs                      # ingest all partner job feeds
    python run_pipeline.py --sync-source sccc          # ingest one source + presence sweep
    python run_pipeline.py --extract                   # re-extract skills
    python run_pipeline.py --training                  # training resources
    python run_pipeline.py --status                    # current data status
    python run_pipeline.py --skip reference,training --all
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from config import LOG_LEVEL, PROJECT_ROOT
from skillbridge.db import executescript, sync_cursor
from skillbridge.pipeline import orchestrator

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("pipeline")


def apply_schema() -> None:
    schema = PROJECT_ROOT / "sql" / "schema.sql"
    log.info("Applying %s", schema)
    executescript(schema)
    log.info("Schema applied")


def show_status() -> None:
    with sync_cursor() as cur:
        cur.execute("SELECT * FROM pipeline.v_data_status")
        row = cur.fetchone()
    if not row:
        log.info("No data status row")
        return
    for k, v in row.items():
        print(f"  {k:30s} {v}")


def sync_single_source(source_name: str) -> int:
    """Run one partner connector + its per-source presence sweep.

    Mirrors what step_ingest_jobs does, but scoped to a single source.
    Useful for testing a connector end-to-end (e.g. SCCC) without firing
    every other connector in the registry.
    """
    from skillbridge.ingest.base import (
        sweep_missing_jobs,
        upsert_job,
        write_raw_job,
    )
    from skillbridge.ingest.partners import ALL_CONNECTORS

    matched = next(
        (c for c in ALL_CONNECTORS if c.source_name == source_name), None
    )
    if matched is None:
        log.error(
            "No partner connector named %r. Known: %s",
            source_name,
            ", ".join(c.source_name for c in ALL_CONNECTORS),
        )
        return 2

    conn = matched()
    upserted = 0
    seen_ids: set[str] = set()
    failed = False
    try:
        for job in conn.fetch():
            write_raw_job(job)
            if upsert_job(job):
                upserted += 1
                seen_ids.add(job.source_job_id)
    except Exception as e:
        log.error("Connector %s failed: %s", source_name, e)
        failed = True

    if failed or not seen_ids:
        log.info(
            "%s: %d jobs upserted, sweep skipped (%s)",
            source_name, upserted,
            "fetch failed" if failed else "no jobs returned",
        )
        return 1 if failed else 0

    deactivated = sweep_missing_jobs(source_name, seen_ids)
    log.info(
        "%s: %d jobs upserted, %d deactivated",
        source_name, upserted, deactivated,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="run_pipeline.py")
    parser.add_argument("--schema", action="store_true", help="apply sql/schema.sql")
    parser.add_argument("--all", action="store_true", help="full pipeline run")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--jobs", action="store_true",
                        help="run partner-feed job connectors (SCCC, Welcome SSM)")
    parser.add_argument("--sync-source", metavar="NAME", default=None,
                        help="run ONE partner connector + its per-source sweep "
                             "(e.g. --sync-source sccc)")
    parser.add_argument("--employers", action="store_true",
                        help="run SSM employer career-page connectors")
    parser.add_argument("--uploads", action="store_true",
                        help="process any partner CSV uploads in PARTNER_CSV_DIR")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--training-extract", action="store_true")
    parser.add_argument("--supplementary", action="store_true",
                        help="run supplementary source connectors (RCIP, AWIC, LIP, etc.)")
    parser.add_argument("--stale-sweep", action="store_true")
    parser.add_argument("--resolve-noc", action="store_true",
                        help="resolve active jobs' title -> NOC via OaSIS lexicon "
                             "(matching v2 step 2; idempotent, skips already-set rows)")
    parser.add_argument("--resolve-noc-all", action="store_true",
                        help="like --resolve-noc but re-resolves every row, "
                             "overwriting existing noc_code")
    parser.add_argument("--embed", action="store_true",
                        help="embed extracted job skills via sentence-transformers "
                             "(matching v2 step 5; idempotent, skips already-embedded "
                             "rows of the current model_version)")
    parser.add_argument("--embed-all", action="store_true",
                        help="like --embed but re-embeds every row "
                             "(e.g. after a model upgrade)")
    parser.add_argument("--status", action="store_true")
    # Round-34 NOC observability: read-only audit + optional production-
    # readiness exit code. The audit walks reference + active jobs and
    # reports coverage / unresolved titles. `--strict-noc-coverage`
    # makes the script return nonzero only on structural failures
    # (empty synonym table / empty occupation table). Low coverage is
    # ALWAYS exit 0 -- coverage is a reported metric, not a hard gate.
    parser.add_argument(
        "--inspect-noc-coverage", action="store_true",
        help="run a read-only NOC coverage audit and print the report",
    )
    parser.add_argument(
        "--strict-noc-coverage", action="store_true",
        help="with --inspect-noc-coverage, exit non-zero on STRUCTURAL "
             "failures (empty synonym or occupation table). Does NOT "
             "fail on low coverage.",
    )
    parser.add_argument("--skip", default="", help="comma-separated step names to skip")
    parser.add_argument("--limit", type=int, default=None,
                        help="extraction batch limit")
    args = parser.parse_args()

    if args.schema:
        apply_schema()

    if args.status:
        show_status()
        return 0

    if args.inspect_noc_coverage:
        from skillbridge.match.inspect_noc import cli_inspect_noc_coverage
        return cli_inspect_noc_coverage(strict=args.strict_noc_coverage)

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    if args.all:
        result = orchestrator.run_all(skip=skip)
        log.info("Pipeline result: %s", result["status"])
        return 0 if result["status"] in {"success", "partial"} else 1

    if args.reference:
        orchestrator.step_reference()
    if args.jobs:
        orchestrator.step_ingest_jobs()
    if args.sync_source:
        rc = sync_single_source(args.sync_source)
        if rc != 0:
            return rc
    if args.employers:
        orchestrator.step_ingest_employer_pages()
    if args.uploads:
        orchestrator.step_ingest_partner_uploads()
    if args.stale_sweep:
        orchestrator.step_stale_sweep()
    if args.resolve_noc or args.resolve_noc_all:
        orchestrator.step_resolve_noc(only_missing=not args.resolve_noc_all)
    if args.embed or args.embed_all:
        orchestrator.step_embed_job_skills(only_missing=not args.embed_all)
    if args.extract:
        orchestrator.step_extract_job_skills(limit=args.limit)
    if args.training:
        orchestrator.step_ingest_training()
    if args.training_extract:
        orchestrator.step_extract_training_skills(limit=args.limit)
    if args.supplementary:
        results = orchestrator.step_ingest_supplementary()
        for r in results:
            log.info("  %(source)s: %(status)s — %(message)s", r)

    if not any([args.schema, args.all, args.reference, args.jobs,
                args.sync_source, args.employers, args.uploads, args.extract,
                args.training, args.training_extract, args.supplementary,
                args.stale_sweep, args.resolve_noc, args.resolve_noc_all,
                args.embed, args.embed_all, args.status,
                args.inspect_noc_coverage]):
        parser.print_help()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
