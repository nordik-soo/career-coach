"""Pipeline orchestrator — daily refresh of jobs + training + skill extraction.

Flow (from design §15, updated by matching v2 steps 2 + 5):
  reference_check
  → ingest_sources
  → normalize_jobs (done inside connectors)
  → deduplicate_jobs (handled by UNIQUE constraints + upsert)
  → mark_last_seen (done inside upsert)
  → stale_job_sweep
  → extract_job_skills
  → resolve_noc           (matching v2 step 2: title -> NOC 2021 backfill)
  → embed_job_skills      (matching v2 step 5: sentence-transformer vectors;
                           MUST come after extract_job_skills)
  → pull_training_resources
  → extract_training_skills
  → ingest_supplementary
  → quality_check         (sees ALL v2 data; dataset_version reflects it)
  → publish_current_dataset
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from config import STALE_SWEEP_NOT_SEEN_DAYS
from skillbridge.db import sync_cursor
from skillbridge.extract import default_extractor
from skillbridge.extract.base import invalidate_reference_cache
from skillbridge.ingest.awic import AWICReportsConnector
from skillbridge.ingest.base import (
    ConnectorResult,
    SourceConnector,
    sweep_missing_jobs,
    upsert_job,
    upsert_training,
    write_raw_job,
)
from skillbridge.ingest.chamber import SaultChamberConnector
from skillbridge.ingest.city_open_data import CityOpenDataConnector
from skillbridge.ingest.employers import ALL_EMPLOYER_CONNECTORS
from skillbridge.ingest.partner_upload import PartnerUploadConnector
from skillbridge.ingest.partners import ALL_CONNECTORS as ALL_PARTNER_CONNECTORS
from skillbridge.ingest.rcip import RCIPEmployerConnector
from skillbridge.ingest.reference import load_all as load_reference
from skillbridge.ingest.services import SSMLIPServicesConnector
from skillbridge.ingest.training import ALL_TRAINING_CONNECTORS
from skillbridge.versions import dataset_version_today


# PR 6A: supplementary connectors (employer enrichment, services, knowledge docs).
# Run AFTER training so all primary tables are populated, but BEFORE quality
# check so the data-status view reflects them in the same dataset_version.
SUPPLEMENTARY_CONNECTORS: list[type[SourceConnector]] = [
    RCIPEmployerConnector,
    SSMLIPServicesConnector,
    AWICReportsConnector,
    SaultChamberConnector,
    CityOpenDataConnector,
]

log = logging.getLogger(__name__)


# ----------------------------------------------------- Steps
def step_reference() -> dict[str, int]:
    log.info("[step] reference data load")
    out = load_reference()
    invalidate_reference_cache()
    return out


def step_embed_job_skills(only_missing: bool = True) -> dict[str, int]:
    """Embed extracted job skills with the configured sentence-transformer.

    Matching v2 step 5c. Reads from extracted.job_skill, writes vectors
    to extracted.job_skill_embedding keyed by (job_id, skill_name,
    model_version). Idempotent: when `only_missing=True` (default),
    skips skill rows that already have an embedding for the current
    model version. Pass False to re-embed everything (e.g. after a
    model upgrade where vector quality matters more than disk).

    Graceful degradation:
      - sentence-transformers not installed -> log warning, return {} summary
      - pgvector extension missing -> psycopg raises; caller handles

    Returns: dict with `embedded` and `skipped_existing` counts.
    """
    from config import EMBEDDING_MODEL_VERSION
    from skillbridge.embed.service import get_embedder

    log.info("[step] embed job skills (model=%s, only_missing=%s)",
             EMBEDDING_MODEL_VERSION, only_missing)
    embedder = get_embedder()
    if embedder is None:
        log.warning(
            "  sentence-transformers not installed; skipping --embed step. "
            "Engine will fall back to lexical-only matching."
        )
        return {"embedded": 0, "skipped_existing": 0,
                "unavailable_sentence_transformers": 1}

    summary = {"embedded": 0, "skipped_existing": 0}
    # Pull every active job's skill rows. When only_missing=True we
    # filter via LEFT JOIN against existing embeddings of the current
    # model version. Single round-trip; works fine at 35-job scale.
    #
    # If pgvector isn't installed, extracted.job_skill_embedding
    # doesn't exist (schema soft-fails on CREATE). Catch UndefinedTable
    # and exit gracefully -- same graceful-degradation pattern as
    # sentence-transformers being missing.
    import psycopg
    try:
        with sync_cursor() as cur:
            if only_missing:
                cur.execute(
                    """
                    SELECT s.job_id, s.skill_name
                      FROM extracted.job_skill s
                      JOIN core.job_posting j USING (job_id)
                      LEFT JOIN extracted.job_skill_embedding e
                             ON e.job_id = s.job_id
                            AND e.skill_name = s.skill_name
                            AND e.model_version = %s
                     WHERE j.is_active = TRUE
                       AND e.job_id IS NULL
                    """,
                    (EMBEDDING_MODEL_VERSION,),
                )
            else:
                cur.execute(
                    """
                    SELECT s.job_id, s.skill_name
                      FROM extracted.job_skill s
                      JOIN core.job_posting j USING (job_id)
                     WHERE j.is_active = TRUE
                    """
                )
            rows = list(cur.fetchall())
    except psycopg.errors.UndefinedTable:
        log.warning(
            "  extracted.job_skill_embedding table missing -- pgvector "
            "extension not installed. Skipping --embed. Install pgvector "
            "and re-run --schema to enable."
        )
        return {"embedded": 0, "skipped_existing": 0,
                "unavailable_pgvector": 1}

    if not rows:
        log.info("  no skills to embed (everything up-to-date)")
        return summary

    # Batch-encode all skill names at once. all-MiniLM-L6-v2 handles
    # ~32 items per batch comfortably; the wrapper sets that internally.
    skill_names = [r["skill_name"] for r in rows]
    log.info("  encoding %d skill phrases...", len(skill_names))
    vectors = embedder.encode_many(skill_names)

    # pgvector accepts a Python list-of-floats via psycopg; convert from
    # numpy to plain list at insert time.
    with sync_cursor() as cur:
        for r, vec in zip(rows, vectors):
            cur.execute(
                """
                INSERT INTO extracted.job_skill_embedding
                    (job_id, skill_name, model_version, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (job_id, skill_name, model_version) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    created_at = NOW()
                """,
                (r["job_id"], r["skill_name"], EMBEDDING_MODEL_VERSION,
                 vec.tolist()),
            )
            summary["embedded"] += 1
    log.info("[step] embed done: %s", summary)
    return summary


def step_resolve_noc(only_missing: bool = True) -> dict[str, int]:
    """Backfill core.job_posting.noc_code by resolving each active job's
    title against reference.occupation_title_synonym.

    Matching v2 step 2. Idempotent. When `only_missing=True` (default),
    rows that already have noc_code set are skipped. Pass False to
    re-resolve every active row.

    Returns a small summary dict: resolved / skipped_no_match.
    """
    from skillbridge.match.occupation import resolve_title_to_noc_with_score

    log.info("[step] resolve NOC codes for active jobs (only_missing=%s)", only_missing)
    summary = {"resolved": 0, "skipped_no_match": 0}
    with sync_cursor() as cur:
        # SQL-level filter already does the right thing for both modes:
        # only_missing=True  -> SELECT rows where noc_code is NULL/empty
        # only_missing=False -> SELECT all active rows (overwrites are OK)
        where = "WHERE is_active = TRUE"
        if only_missing:
            where += " AND (noc_code IS NULL OR noc_code = '')"
        cur.execute(f"SELECT job_id, title, noc_code FROM core.job_posting {where}")
        rows = list(cur.fetchall())
        for r in rows:
            noc, sim = resolve_title_to_noc_with_score(
                r.get("title"), call_source="job_backfill",
            )
            if not noc:
                summary["skipped_no_match"] += 1
                log.info(
                    "  no NOC match for job_id=%s title=%r",
                    str(r["job_id"])[:8], r.get("title"),
                )
                continue
            cur.execute(
                "UPDATE core.job_posting SET noc_code = %s WHERE job_id = %s",
                (noc, r["job_id"]),
            )
            summary["resolved"] += 1
            log.info(
                "  resolved job_id=%s title=%r -> NOC %s (sim=%.2f)",
                str(r["job_id"])[:8], r.get("title"), noc, sim,
            )
    log.info("[step] resolve NOC done: %s", summary)
    return summary


def step_ingest_jobs() -> dict[str, dict[str, int]]:
    """Run each registered partner connector and do a per-source presence sweep.

    For every connector we collect the set of source_job_ids actually
    returned this run. After a successful fetch we sweep: any prior job
    from this source whose ID isn't in the seen set is marked is_active=false.
    The sweep is per-source — SCCC's behaviour can't affect City of SSM's
    data, and vice versa.

    On connector failure we DO NOT sweep — last successful state is
    preserved. PR 9A keeps this minimal; PR 9B adds the empty/degraded
    response guard, advisory locks, and a pipeline.source_sync audit row.
    """
    log.info("[step] ingest partner job feeds")
    counts: dict[str, dict[str, int]] = {}
    for cls in ALL_PARTNER_CONNECTORS:
        conn = cls()
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
            log.error("Connector %s failed: %s", conn.source_name, e)
            failed = True

        if failed or not seen_ids:
            deactivated = 0
            log.info(
                "  %s: %d jobs upserted, sweep skipped (%s)",
                conn.source_name, upserted,
                "fetch failed" if failed else "no jobs returned",
            )
        else:
            deactivated = sweep_missing_jobs(conn.source_name, seen_ids)
            log.info(
                "  %s: %d jobs upserted, %d deactivated",
                conn.source_name, upserted, deactivated,
            )
        counts[conn.source_name] = {
            "upserted": upserted,
            "deactivated": deactivated,
        }
    return counts


def step_ingest_employer_pages() -> list[dict]:
    """Run every employer career-page connector (SourceConnector pattern).

    Two reference parsers, nine stubs. All gated by their *_ENABLED flag
    and *_URL placeholder check. Returns one ConnectorResult dict per
    connector.
    """
    log.info("[step] ingest employer career pages")
    out: list[dict] = []
    for cls in ALL_EMPLOYER_CONNECTORS:
        conn = cls()
        try:
            result = conn.run()
        except Exception as e:
            log.exception("Employer connector %s crashed", conn.source_name)
            result = ConnectorResult(
                source=conn.source_name, status="fail", message=str(e)[:200],
            )
        log.info(
            "  %s: %s (upserted=%d) %s",
            result.source, result.status, result.upserted, result.message or "",
        )
        out.append(result.as_dict())
    return out


def step_ingest_partner_uploads() -> dict:
    """Process any *.csv in PARTNER_CSV_DIR — the bridge mechanism."""
    log.info("[step] ingest partner CSV uploads")
    conn = PartnerUploadConnector()
    result = conn.run()
    log.info(
        "  partner_csv: %s (upserted=%d) %s",
        result.status, result.upserted, result.message or "",
    )
    return result.as_dict()


def step_stale_sweep() -> int:
    log.info("[step] stale sweep (>%d days not seen)", STALE_SWEEP_NOT_SEEN_DAYS)
    with sync_cursor() as cur:
        cur.execute(
            """
            UPDATE core.job_posting
               SET is_active = FALSE
             WHERE is_active = TRUE
               AND last_seen_at < NOW() - (%s || ' days')::interval
            """,
            (str(STALE_SWEEP_NOT_SEEN_DAYS),),
        )
        return cur.rowcount


def step_extract_job_skills(limit: int | None = None) -> int:
    log.info("[step] extract job skills")
    extractor = default_extractor()
    sql_select = """
    SELECT j.job_id, j.title, j.description, j.updated_at
      FROM core.job_posting j
      LEFT JOIN extracted.job_skill s ON s.job_id = j.job_id
     WHERE j.is_active = TRUE
       AND (s.job_id IS NULL OR s.extractor_version <> %s)
     GROUP BY j.job_id
     ORDER BY j.updated_at DESC NULLS LAST
    """
    if limit:
        sql_select += f" LIMIT {int(limit)}"
    n = 0
    with sync_cursor() as cur:
        cur.execute(sql_select, (extractor.version,))
        jobs = list(cur.fetchall())
    for j in jobs:
        try:
            skills = extractor.extract_from_job(title=j["title"], description=j.get("description") or "")
        except Exception as e:
            log.warning("Extraction failed for job %s: %s", j["job_id"], e)
            continue
        with sync_cursor() as cur:
            cur.execute("DELETE FROM extracted.job_skill WHERE job_id = %s", (j["job_id"],))
            for s in skills:
                cur.execute(
                    """
                    INSERT INTO extracted.job_skill
                        (job_id, skill_id, skill_name, raw_phrase, confidence,
                         importance_rank, skill_type, extractor_version)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (job_id, skill_name) DO UPDATE SET
                        skill_id = EXCLUDED.skill_id,
                        confidence = EXCLUDED.confidence,
                        importance_rank = EXCLUDED.importance_rank,
                        skill_type = EXCLUDED.skill_type,
                        extractor_version = EXCLUDED.extractor_version
                    """,
                    (j["job_id"], s.skill_id, s.skill_name, s.raw_phrase,
                     s.confidence, s.importance_rank, s.skill_type,
                     extractor.version),
                )
        n += 1
    log.info("  extracted skills for %d jobs (extractor=%s)", n, extractor.version)
    return n


def step_ingest_training() -> dict[str, int]:
    log.info("[step] ingest training resources")
    counts: dict[str, int] = {}
    for cls in ALL_TRAINING_CONNECTORS:
        conn = cls()
        n = 0
        try:
            for res in conn.fetch():
                if upsert_training(res):
                    n += 1
        except Exception as e:
            log.error("Training connector %s failed: %s", conn.provider_name, e)
        counts[conn.provider_name] = n
        log.info("  %s: %d resources", conn.provider_name, n)
    return counts


def step_extract_training_skills(limit: int | None = None) -> int:
    log.info("[step] extract training skills")
    extractor = default_extractor()
    sql = """
    SELECT r.resource_id, r.title, r.description, r.updated_at
      FROM core.training_resource r
      LEFT JOIN extracted.training_skill s ON s.resource_id = r.resource_id
     WHERE r.is_active = TRUE
       AND (s.resource_id IS NULL OR s.extractor_version <> %s)
     GROUP BY r.resource_id
     ORDER BY r.updated_at DESC NULLS LAST
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    n = 0
    with sync_cursor() as cur:
        cur.execute(sql, (extractor.version,))
        rows = list(cur.fetchall())
    for r in rows:
        try:
            skills = extractor.extract_from_training(title=r["title"], description=r.get("description") or "")
        except Exception as e:
            log.warning("Training extraction failed for %s: %s", r["resource_id"], e)
            continue
        with sync_cursor() as cur:
            cur.execute("DELETE FROM extracted.training_skill WHERE resource_id = %s", (r["resource_id"],))
            for s in skills:
                cur.execute(
                    """
                    INSERT INTO extracted.training_skill
                        (resource_id, skill_id, skill_name, raw_phrase, confidence, extractor_version)
                    VALUES (%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (resource_id, skill_name) DO UPDATE SET
                        skill_id = EXCLUDED.skill_id,
                        confidence = EXCLUDED.confidence,
                        extractor_version = EXCLUDED.extractor_version
                    """,
                    (r["resource_id"], s.skill_id, s.skill_name, s.raw_phrase,
                     s.confidence, extractor.version),
                )
        n += 1
    log.info("  extracted skills for %d training resources", n)
    return n


def step_ingest_supplementary() -> list[dict]:
    """Run every supplementary SourceConnector.

    Each connector returns a ConnectorResult. We collect them as plain dicts
    for JSONB-friendly storage in pipeline.run.metrics. One failing
    connector never poisons the run — its result just carries status='fail'.
    """
    log.info("[step] ingest supplementary sources")
    out: list[dict] = []
    for cls in SUPPLEMENTARY_CONNECTORS:
        conn = cls()
        try:
            result = conn.run()
        except Exception as e:
            log.exception("Connector %s crashed", conn.name)
            result = ConnectorResult(
                source=conn.name, status="fail", message=str(e)[:200],
            )
        log.info(
            "  %s: %s (fetched=%d, upserted=%d) %s",
            result.source, result.status, result.fetched, result.upserted,
            result.message or "",
        )
        out.append(result.as_dict())
    return out


def step_quality_check() -> dict[str, str]:
    """Lightweight quality summary; full quality gate is future work."""
    with sync_cursor() as cur:
        cur.execute("SELECT * FROM pipeline.v_data_status")
        status = cur.fetchone() or {}
    checks: dict[str, str] = {}
    checks["jobs_present"] = "pass" if (status.get("current_jobs") or 0) > 0 else "warn"
    checks["skills_extracted"] = "pass" if (status.get("extracted_job_skills") or 0) > 0 else "warn"
    checks["training_present"] = "pass" if (status.get("active_training_resources") or 0) > 0 else "warn"
    log.info("Quality summary: %s", checks)
    return checks


def step_publish(dataset_version: str, run_id: str) -> None:
    log.info("[step] publish dataset_version=%s", dataset_version)
    with sync_cursor() as cur:
        cur.execute(
            """
            INSERT INTO pipeline.dataset_state (pointer_key, dataset_version, run_id, published_at)
            VALUES ('current_jobs', %s, %s, NOW())
            ON CONFLICT (pointer_key) DO UPDATE SET
                dataset_version = EXCLUDED.dataset_version,
                run_id          = EXCLUDED.run_id,
                published_at    = EXCLUDED.published_at
            """,
            (dataset_version, run_id),
        )


# ----------------------------------------------------- Run orchestration
def _start_run() -> str:
    with sync_cursor() as cur:
        cur.execute(
            "INSERT INTO pipeline.run (status) VALUES ('running') RETURNING run_id"
        )
        return str(cur.fetchone()["run_id"])


def _finish_run(run_id: str, status: str, metrics: dict | None = None,
                dataset_version: str | None = None) -> None:
    import json
    with sync_cursor() as cur:
        cur.execute(
            """
            UPDATE pipeline.run
               SET status = %s, finished_at = NOW(), metrics = %s, dataset_version = %s
             WHERE run_id = %s
            """,
            (status, json.dumps(metrics or {}, default=str), dataset_version, run_id),
        )


def _record_step(run_id: str, name: str, status: str, metrics: dict | None = None,
                 error: str | None = None) -> None:
    import json
    with sync_cursor() as cur:
        cur.execute(
            """
            INSERT INTO pipeline.step (run_id, name, status, finished_at, metrics, error)
            VALUES (%s, %s, %s, NOW(), %s, %s)
            """,
            (run_id, name, status, json.dumps(metrics or {}, default=str), error),
        )


def run_all(skip: set[str] | None = None) -> dict:
    """End-to-end pipeline run. `skip` accepts step names to bypass."""
    skip = skip or set()
    run_id = _start_run()
    dataset_version = dataset_version_today()
    metrics: dict = {"started_at": datetime.now(timezone.utc).isoformat()}
    overall = "success"
    try:
        # Matching v2 step 5 review fix: resolve_noc and embed_job_skills
        # must be part of run_all() or normal pipeline runs leave them
        # stale. Order matters:
        #   extract_job_skills  -> produces the skill rows
        #   resolve_noc         -> independent of skills; tags jobs with NOC
        #   embed_job_skills    -> reads the extracted skill names produced
        #                          by extract_job_skills, MUST come after
        # quality_check moves to the end so the dataset_version it writes
        # reflects ALL the matching-v2 data (NOC + embeddings included).
        for step_name, func in [
            ("reference",                step_reference),
            ("ingest_jobs",              step_ingest_jobs),               # partner feeds
            ("ingest_employer_pages",    step_ingest_employer_pages),     # SSM employer career pages
            ("ingest_partner_uploads",   step_ingest_partner_uploads),    # CSV bridge
            ("stale_sweep",              step_stale_sweep),
            ("extract_job_skills",       step_extract_job_skills),
            ("resolve_noc",              step_resolve_noc),               # matching v2 step 2
            ("embed_job_skills",         step_embed_job_skills),          # matching v2 step 5 (after extract)
            ("ingest_training",          step_ingest_training),
            ("extract_training_skills",  step_extract_training_skills),
            ("ingest_supplementary",     step_ingest_supplementary),
            ("quality_check",            step_quality_check),
        ]:
            if step_name in skip:
                _record_step(run_id, step_name, "skipped")
                continue
            try:
                result = func()
                metrics[step_name] = result
                _record_step(run_id, step_name, "success", metrics={"result": result})
            except Exception as e:
                log.exception("Step %s failed: %s", step_name, e)
                _record_step(run_id, step_name, "failed", error=str(e))
                overall = "partial"
        step_publish(dataset_version, run_id)
        _record_step(run_id, "publish", "success",
                     metrics={"dataset_version": dataset_version})
    except Exception as e:
        log.exception("Pipeline failed: %s", e)
        overall = "failed"
        metrics["error"] = str(e)
    metrics["finished_at"] = datetime.now(timezone.utc).isoformat()
    metrics["dataset_version"] = dataset_version
    _finish_run(run_id, overall, metrics, dataset_version)
    log.info("Run %s finished: %s", run_id, overall)
    return {"run_id": run_id, "status": overall, "metrics": metrics}
