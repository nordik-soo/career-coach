# Breaking changes

This file records intentional breaking changes — what was removed,
when, and why. Restore from git history if you need to revisit a
removed component.

---

## PR 7A — SSM-only product rule (2026-05)

### Removed

**`skillbridge/ingest/jobbank.py`** — Job Bank Open Data connector.

**Config entries removed:**
- `JOBBANK_ENABLED`
- `JOBBANK_PACKAGE_API`
- `JOBBANK_MONTHS_BACK`
- `REGION_CSD_CODES`
- `REGION_CD_CODES`
- `REGION_ER_CODES`

**Code removed:**
- `is_local_job()` from `skillbridge/ingest/base.py`. Region filtering is
  obsolete: every approved source is SSM-native by construction.
- `_parse_date` / `_parse_float` from `jobbank.py` (relocated to
  `ingest/base.py` as `parse_date_loose` / `parse_float_loose`).

### Why

SkillBridge SSM is a Sault Ste. Marie product. The grant funds local
relevance, not national coverage. Federal data sources (Job Bank, StatCan,
Census) diluted the product's value proposition by mixing distant jobs
with local ones. The decision is to **never source from national feeds**.

Reference taxonomies (NOC 2021, OaSIS skills, Ontario regulated
occupations) remain — they are **classification dictionaries**, not
job sources. They help the system understand local jobs without bringing
non-SSM jobs into the product.

### Enforcement

The product rule is encoded in the schema, not just in policy:

```sql
core.job_posting.source REFERENCES core.approved_job_source(source)
```

The approved list contains only SSM partner names and SSM employer names.
Any attempt to insert a row with `source='jobbank'` (or any other federal
source name) fails the FK constraint at the database level.

The durable invariant test is `tests/test_source_purity.py`. Five tests
verify the FK exists, the approved list contains no prohibited sources,
the required SSM sources are present, federal inserts are blocked, and
every orchestrator-registered connector produces an approved source name.

### Migration notes for existing deployments

If your database has Job Bank data already loaded:

```sql
-- 1. Delete Job Bank rows before adding the FK
DELETE FROM core.job_posting WHERE source = 'jobbank';

-- 2. Apply the new schema (schema.sql is idempotent)
\i sql/schema.sql

-- 3. Verify
SELECT source, COUNT(*) FROM core.job_posting GROUP BY source;
```

For fresh deployments: `psql -f sql/schema.sql` works as-is.

### What this means for the pilot

Until partner agreements land, the system has fewer working data paths.
The viable paths post-PR-7A:

1. **Partner CSV uploads** (`./data/partner_uploads/*.csv`) — bridge
   mechanism for SCCC / Welcome to SSM / NORDIK while feeds are negotiated.
2. **Sault Area Hospital + City of SSM HR** — reference parsers in
   `skillbridge/ingest/employers/`. Selectors are starting points and
   need verification against live pages.
3. **9 employer stubs** — Algoma Steel, Sault College careers, Algoma U
   careers, PUC, Group Health Centre, YMCA SSM, CAS Algoma, ADSAB, school
   board(s). Each activates by setting its `*_ENABLED=true` + `*_URL`.

If none of the above produce data, the dashboard is empty — which is the
honest state of "we don't have local source agreements yet" rather than a
misleading view filled with federal postings.

---

## Previous PRs

- PR 1 — Consent-before-persistence (Redis/cookie sessions, opaque tokens)
- PR 2 — Delete cascade with PII shell + readiness checks
- PR 3 — Auth enforcement, `/me` paths, admin gate, `POST /v1/profiles` removed
- PR 5 — `JOB_FRESHNESS_DAYS` removed; 30-day rule documented as product invariant
- PR 6A — Supplementary source connectors (RCIP, AWIC, LIP, Chamber, City Open Data)
