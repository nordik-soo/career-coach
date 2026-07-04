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

## AWIC jobs — local aggregator provenance rule (2026-07)

### Added

**`awic_jobs`** — new approved local JD source (`core.approved_job_source`).
Peer to `sccc`. Feeds `core.job_posting` via the standard ingest path;
after ingest, downstream code (matching engine, recommender, responder)
does not care whether a posting came from SCCC or AWIC. No AWIC-specific
matching logic, no separate recommender path, no RAG. Every JD source
follows the same pattern.

**Ingest path:**
`AWICJobsConnector` in `skillbridge/ingest/partners.py` reads the public
GeoJSON endpoint `https://awic.ca/wp-json/wedatatools/v1/get-jobs-geojson`
that powers AWIC's own portal Jobs Map. Single request; ~160 features
per snapshot; no auth; no pagination.

### The load-bearing policy — durable, do not weaken

AWIC is a **local aggregator**. Some postings' `properties.url` values
point to third-party sources including Job Bank. SkillBridge ingests
**AWIC's curated metadata layer only**, NOT the third-party sources
behind the apply URLs.

Every AWIC-derived row in `core.job_posting` is stamped with
`source = 'awic_jobs'`. The third-party URL is stored in the `url`
column as the apply-URL only — it is provenance / navigation context,
NOT source identity.

**This is how AWIC coexists with the durable no-federal-source rule
in PR 7A:**

- `tests/test_source_purity.py::test_no_federal_sources_in_approved_list`
  checks that names like `'jobbank'`, `'statcan'`, `'census'` are NOT
  in `core.approved_job_source`. AWIC is `'awic_jobs'`, so this test
  continues to pass.
- The FK constraint (`test_fk_constraint_exists`) ensures every row in
  `core.job_posting` has an approved `source`. AWIC rows carry
  `source = 'awic_jobs'`, so the FK is satisfied.
- The rule is preserved *by source identity*, not by URL inspection.
  A future reviewer must not "helpfully" reject AWIC postings whose
  `properties.url` points at a federal domain — that would confuse
  provenance (who curated) with content (where to apply).

### SSM-only scope preserved

AWIC covers the full Algoma District (Wawa, Chapleau, Blind River,
etc.). The AWIC connector filters to Sault Ste. Marie using a
**coordinate bounding box** (`SSM_BBOX_*` in `config.py`). Postings
without coordinates are dropped rather than assumed local. This
preserves the SSM-only product rule established in PR 7A.

Approximately 118 of 161 features in the live feed at capture time
(2026-07-04) fall within the default bounding box. The bounds are
env-refinable (`SSM_BBOX_LAT_MIN`, etc.) without a code change.

### NOC 2021 handling

AWIC often carries the NOC 2021 code in `properties.nocs_2021`. When
it does (exactly 5 digits, numeric), the connector passes it through
verbatim and the downstream `resolve_title_to_noc_with_score`
backfill is a no-op for that row. When the code is missing or
invalid, `noc_code` is left `None` and the existing backfill runs
normally. Same downstream behavior as SCCC (which never carries the
code), just with a faster path when AWIC provides it.

### Description-only text (v1 limitation)

AWIC's GeoJSON returns `properties.excerpt` (short summary), not full
JD text. This is used as-is for `NormalizedJob.description`. AWIC's
standard WP REST endpoint (`/wp-json/wp/v2/job-posts/{id}`) returned
404 when probed on 2026-07-04, so full-body fetch is deferred to v2
research. Skill extraction quality on excerpts is expected to be
lower than SCCC's full-JD case; this is acknowledged, not accidental.

### Renaming clarification (existing `AWIC_ENABLED` was for reports)

Do NOT reuse the pre-existing `AWIC_ENABLED` flag (or `AWIC_REPORTS_URL`
/ `AWIC_DATA_FEED_URL`) for jobs. Those belong to `AWICReportsConnector`
in `skillbridge/ingest/awic.py`, which writes to `knowledge.document`
(reports metadata) — a completely different family from the jobs
ingest path.

Jobs use `AWIC_JOBS_ENABLED` / `AWIC_JOBS_FEED_URL` /
`AWIC_JOBS_FEED_FORMAT`. Sharing a flag between reports and jobs would
tie two unrelated features together and confuse future operators.

---

## Previous PRs

- PR 1 — Consent-before-persistence (Redis/cookie sessions, opaque tokens)
- PR 2 — Delete cascade with PII shell + readiness checks
- PR 3 — Auth enforcement, `/me` paths, admin gate, `POST /v1/profiles` removed
- PR 5 — `JOB_FRESHNESS_DAYS` removed; 30-day rule documented as product invariant
- PR 6A — Supplementary source connectors (RCIP, AWIC, LIP, Chamber, City Open Data)
