# SkillBridge SSM API

Newcomer-centred skill-matching API for Sault Ste. Marie. Backbone for the
SkillBridge SSM MVP: daily job dashboard, natural chat intake, job
recommendations, personal skill gap analysis, local training suggestions.

**SkillBridge SSM is a Sault Ste. Marie–specific platform.** It does not
source job postings from national feeds (Job Bank, StatCan, Census). All
postings come from SSM partner organizations or SSM employer career pages.
NOC 2021 and OaSIS remain as reference dictionaries (not job sources).
See [BREAKING.md](BREAKING.md) for the source-purity decision.

## Quick start

```bash
# 1. Postgres 14+ with pg_trgm + pgcrypto
createdb skillbridge
psql skillbridge -f sql/schema.sql

# 2. Python env
python -m venv .venv
. .venv/Scripts/Activate.ps1   # or source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# edit .env: set PGPASSWORD and ANTHROPIC_API_KEY
# replace PLACEHOLDER_* values for partner sources you have credentials for

# 4. Pipeline (one-shot)
python run_pipeline.py --all

# 5. API
uvicorn api:app --reload --port 8000
# Swagger: http://localhost:8000/docs
```

## What's in the box

| Path | What |
|---|---|
| `config.py` | env-driven configuration |
| `sql/schema.sql` | MVP schema (reference / core / extracted / profile / analytics / interaction / pipeline / raw) |
| `skillbridge/db.py` | async psycopg pool + helpers |
| `skillbridge/llm.py` | Haiku-based Anthropic client (cheap default) |
| `skillbridge/extract/` | `SkillExtractor` interface + rule-based + LLM-based |
| `skillbridge/ingest/` | one real connector (Job Bank) + clear stubs for partner sources |
| `skillbridge/match/` | JobMatchEngine v1.0.0 + training recommender |
| `skillbridge/chat/` | chat handler (LLM → JSON → DB → engine → explanation) |
| `skillbridge/routes/` | all `/v1/*` endpoints |
| `skillbridge/pipeline/` | orchestrator for daily refresh |
| `api.py` | FastAPI app |
| `run_pipeline.py` | CLI for ingestion + extraction + publish |

## LLM cost

Default model is **Claude Haiku 4.5** (`claude-haiku-4-5-20251001`) — the
cheap+fast Anthropic tier. The system is also designed to work without an
LLM: `LLM_ENABLED=false` switches all extractors to the rule-based fallback.

## Data sources

`.env.example` lists every source with a clear `PLACEHOLDER_*` value where
you'll need to plug in a real URL or API key after the partner agreement is
signed. Until then, only **Job Bank Open Data** and **partner CSV uploads**
(into `./data/partner_uploads/`) will actually produce postings.

See `skillbridge/ingest/*.py` — each connector has a docstring with what's
implemented and what's stubbed.

## Endpoints

See `skillbridge/routes/` or hit `/docs` once running.

| Auth | Method | Path |
|---|---|---|
| public | GET | `/v1/jobs`, `/v1/jobs/{id}` |
| public | GET | `/v1/training-resources`, `/v1/training-resources/{id}` |
| public | POST | `/v1/chat/messages` (anonymous or bearer-authenticated) |
| public | POST | `/v1/consent` (exchanges staged session for bearer token) |
| bearer | GET | `/v1/profiles/me`, `/v1/profiles/me/skills` |
| bearer | PATCH / DELETE | `/v1/profiles/me` |
| bearer | POST | `/v1/matches/jobs` |
| bearer | GET | `/v1/profiles/me/job-matches`, `/me/job-matches/{match_id}`, `/me/skill-gaps/{job_id}` |
| bearer | GET | `/v1/profiles/me/training-recommendations` |
| bearer | POST | `/v1/recommendations/feedback` |
| admin  | GET | `/v1/admin/data-status` |
| admin  | POST | `/v1/admin/pipeline/refresh` |

Auth headers:
- `bearer`: `Authorization: Bearer <session_token>` returned from `POST /v1/consent`.
- `admin`:  `Authorization: Bearer admin:<ADMIN_API_KEY>`.

Profile routes never accept a `profile_id` from the client — it's always
derived from the bearer token. This eliminates the "anyone with a UUID
can read any profile" class of bug.

All responses use the SkillBridge envelope:

```json
{
  "data": { ... },
  "meta": {
    "request_id": "req_...",
    "dataset_version": "ssm-jobs-2026-05-20",
    "engine_version": "job-match-v1.0.0",
    "data_as_of": "2026-05-20T06:00:00-04:00"
  }
}
```

## Running tests

Tests use a separate database (`skillbridge_test` by default) and don't
require Redis or Anthropic — `tests/conftest.py` forces
`SESSION_STORE=cookie` and `LLM_ENABLED=false` so a local Postgres is the
only external dependency.

```powershell
# PowerShell
createdb skillbridge_test
$env:PGDATABASE = "skillbridge_test"
psql -f sql/schema.sql

pytest tests/ -v
```

```bash
# Bash / zsh
createdb skillbridge_test
PGDATABASE=skillbridge_test psql -f sql/schema.sql

PGDATABASE=skillbridge_test pytest tests/ -v
```

The `_clean_db` autouse fixture truncates every consent-guarded table
before each test, so tests run in any order and leave no residue.

### Durable invariants

Two tests carry forward the most important contracts:

- [`tests/test_consent_boundary.py`](tests/test_consent_boundary.py) —
  anonymous chat must write zero rows to any consent-guarded table; consent
  grant must atomically persist + issue a token.
- [`tests/test_delete_cascade.py`](tests/test_delete_cascade.py) — profile
  deletion must clear every PII-bearing row + scrub the profile shell; a
  schema scan asserts every FK to `profile.user_profile` is in the delete
  chain so a future table addition can't silently regress this.
- [`tests/test_route_guard.py`](tests/test_route_guard.py) — every route
  is either explicitly public, profile-authenticated, or admin-authenticated;
  profile and admin tokens cannot cross-access; `POST /v1/profiles` stays
  removed.
- [`tests/test_source_purity.py`](tests/test_source_purity.py) — the
  `core.job_posting.source` FK constraint exists; no federal source can
  enter the table; every orchestrator-registered connector uses an
  approved SSM source name.

## Product invariants

These are not configurable. They are part of what the product is. To
change one, write a schema migration and bump the relevant version
published in the API envelope.

- **Daily refresh.** Pipeline runs once per day. The dashboard label is
  "Updated daily — last refresh: ...". Not "real-time."
- **30-day freshness window.** A job is "current" if it is active, was
  posted within the last 30 days, and (where a closing date exists) has
  not closed. Enforced by `core.v_current_job` in
  [sql/schema.sql](sql/schema.sql).
- **14-day stale sweep.** A job not re-seen in any source feed for 14 days
  is marked `is_active = FALSE`. Tunable via `STALE_SWEEP_NOT_SEEN_DAYS`
  for ops but should be left at 14 in production.

## Pilot principles

- Dashboard is a **public utility** — no personalization, no scores.
- Chat is the **personalized surface** — gated by consent.
- LLM **extracts and explains**; never invents numbers, jobs, or URLs.
- Every match traces to a versioned dataset + engine.
- System runs without an LLM (degraded but functional).
