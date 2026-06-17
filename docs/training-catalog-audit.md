# Training Catalog Audit

Status: research deliverable · 2026-06-04 · for the gap-recommendation slice

## Why this doc exists

Before building the skill-gap recommendation system, we need an honest baseline:
does a useful training catalog exist in the database today, or are we starting
from zero? The audit produces one binary decision at the end:

> **Extend the existing DB rows, or seed a new curated registry?**

Maintaining both as equal sources of truth would create stale/conflicting
recommendations. This decision must be made before any registry code is written.

## 1. Schema

Two tables in [sql/schema.sql](../sql/schema.sql):

### `core.training_resource`

```sql
CREATE TABLE IF NOT EXISTS core.training_resource (
    resource_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider           TEXT NOT NULL,
    title              TEXT NOT NULL,
    description        TEXT,
    url                TEXT,
    location           TEXT,
    delivery_mode      VARCHAR(30),       -- in_person | online | hybrid
    cost_text          TEXT,
    duration_text      TEXT,
    duration_band      VARCHAR(10),       -- short | medium | long
    resource_type      VARCHAR(30),       -- workshop | course | program | counselling | online
    is_active          BOOLEAN NOT NULL DEFAULT TRUE,
    ingested_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (provider, title)
);
```

### `extracted.training_skill`

```sql
CREATE TABLE IF NOT EXISTS extracted.training_skill (
    resource_id        UUID REFERENCES core.training_resource(resource_id) ON DELETE CASCADE,
    skill_id           VARCHAR(40) REFERENCES reference.skill(skill_id),
    skill_name         TEXT NOT NULL,
    raw_phrase         TEXT,
    confidence         NUMERIC(3,2),
    extractor_version  VARCHAR(40) NOT NULL,
    extracted_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (resource_id, skill_name)
);
```

**Observation**: the schema is reasonable. `core.training_resource` has the
expected fields for surfacing to a chat user, and `extracted.training_skill`
gives us the gap → resource lookup the recommender needs. The `(provider, title)`
unique constraint is the right shape to prevent duplicates.

What the schema does NOT have:

- No `verified_at` / `verified_by` audit trail
- No explicit `credential_pathway` vs `apprenticeship` resource type
- No way to mark `referral_only` resources (URL-less entries that should
  point to SCCC instead of pretending to be a course)

These gaps don't disqualify the schema — they're additions a future migration
could fold in. But the schema does NOT, today, encode the resource-type
taxonomy we agreed on for the recommender.

## 2. Row counts (production DB, 2026-06-04)

| Table | Row count |
|---|---|
| `core.training_resource` | **0** |
| `extracted.training_skill` | **0** |

Both tables are empty. There is no training catalog to extend.

## 3. Ingestion connectors

[skillbridge/ingest/training.py](../skillbridge/ingest/training.py) defines
four connectors:

| Connector | Provider | Status |
|---|---|---|
| `SaultCollegeConnector` | Sault College | **Stub**. BeautifulSoup parser exists but gated by `SAULT_COLLEGE_ENABLED=false`. Targets `a[href*='/program/']` selectors which their site no longer guarantees |
| `AlgomaUConnector` | Algoma University | **Stub**. Body is `log.info("...stub..."); return; yield`. No fetch logic |
| `NorthlandConnector` | Northland Adult Learning Centre | **Stub**. Same shape as Algoma. Comment says: "Until they provide a feed, drop a CSV into ./data/partner_uploads/..." (no CSV exists today) |
| `SCCCServicesConnector` | Sault Community Career Centre | **Stub** |

All four URLs in `TRAINING_SOURCES` ([config.py:209](../config.py))
are `PLACEHOLDER`-gated env vars. None are configured in `.env`.

The `data/` directory contains only `.gitkeep` — no partner CSV uploads, no
seed data.

## 4. Recommender code path

[skillbridge/match/recommend.py](../skillbridge/match/recommend.py) implements
`suggest_for_skill(skill_name)` against the DB via a `pg_trgm`
similarity-and-ILIKE join over `core.training_resource × extracted.training_skill`.
This code works correctly — when the DB is populated, it would return ranked
suggestions per skill.

Because the DB is empty, every call returns the `_NO_LOCAL_MATCH` sentinel:

```python
_NO_LOCAL_MATCH = TrainingSuggestion(
    resource_id=None,
    provider="Sault Community Career Centre",
    title="Speak with a career counsellor",
    url=None,
    ...
)
```

The handler's `_attach_training` ([handler.py:516](../skillbridge/chat/handler.py))
explicitly filters this sentinel out so the chat's `TRAINING:` block stays
empty rather than rendering a misleading "training card":

```python
if s.resource_id is None:
    continue   # drop _NO_LOCAL_MATCH from prompt payload
```

So the responder LLM today has **no TRAINING data to narrate**. It improvises
generic referrals ("contact Sault College") because there's nothing else to
use, and the prompt's grounding rule ("URLs must come from RESULTS or
TRAINING") prevents it from inventing URLs.

## 5. Gap coverage (the priority list)

Against the 13 priority gaps named in the recommender plan, here's what the
DB covers:

| Gap | DB coverage |
|---|---|
| 310T technician certification | none |
| Class G driver's license | none |
| Class A/D/Z driver's license | none |
| WHMIS | none |
| First Aid / CPR | none |
| Forklift | none |
| Food Handler / Food Safe | none |
| PSW (Personal Support Worker) | none |
| Microsoft Excel | none |
| Microsoft Office (general) | none |
| QuickBooks / basic accounting | none |
| Customer service | none |
| CompTIA / IT support | none |

**13 of 13 priority gaps have zero coverage.** This is expected given the
zero-row finding above, but stated explicitly so the gap is visible.

## 6. URL stability

There are no URLs in the table to assess for freshness. No `verified_at`
column exists on the schema.

## 7. Existing chat plumbing

Good news for downstream integration: the **prompt + responder plumbing
already handles training as a first-class concept.**

- The system prompts ([prompts.py](../skillbridge/chat/prompts.py)) already
  include `TRAINING:` block instructions: *"If TRAINING has entries, mention
  the most relevant ones (up to 3) with their URLs"*
- The `OUTCOME_RESPONDER_PROMPT` for v2 documents the same
- The SCOPE BOUNDARIES section already says: *"TRAINING DISCUSSIONS ARE IN
  SCOPE. ... Use the TRAINING block when present; if it's empty, recommend
  Sault Community Career Centre and Sault College's continuing-education
  catalogue as starting points"*
- The user-block builder (`_build_user_block_v2`) already serializes
  `inp.training_by_job` into a `TRAINING:` block when match results are
  present
- The policy regex already rejects responder output that mentions URLs
  outside RESULTS or TRAINING

In other words: **the entire LLM-narration side is already wired** for a
populated training catalog. We just need to feed it data.

## 8. Decision: extend DB or seed YAML?

**Seed YAML.**

The DB-extension path would require:
1. Authoring + maintaining web scrapers for Sault College / Algoma U
   (current connectors are stubs and Sault College's HTML targets are
   no longer guaranteed)
2. Partner agreements with Northland and SCCC (per the existing TODOs)
3. URL-discovery work for credential pathways like 310T that don't live
   on those four providers' sites at all (Skilled Trades Ontario,
   DriveTest, etc.)
4. An ongoing freshness-monitoring layer the schema doesn't support today

The YAML-seed path:
1. Hand-curate ~13 priority gaps with verified URLs in a versioned file
2. Load into a registry singleton at server startup
3. Plug into the existing `_attach_training` → `TRAINING:` block plumbing
4. The existing prompt + policy infrastructure carries unchanged

**Reasons YAML wins for v1**:

- Zero rows in the DB means there's nothing to extend
- The DB schema doesn't encode the resource-type taxonomy we need
  (credential_pathway, apprenticeship, referral_only)
- Connectors are stubs that need partner conversations before becoming
  real; that's a months-long deliverable, not a sprint
- YAML is reviewable in PRs — every URL change goes through code review
- No admin UI is required for v1
- We can migrate YAML → DB later once schema + connectors mature

### What about the existing DB schema?

Keep it as-is. It's not in our way. Future work can either:

- (a) Stand a one-way `seed-yaml-into-db` job up later when admin UI ships, OR
- (b) Mark the DB tables `is_active=FALSE` on seed and treat YAML as canonical

For v1, leave the DB alone. The `_attach_training` handler call already
returns nothing useful from it; we'll point that function at the YAML
registry instead.

## 9. Next steps (informs but does not commit slice scope)

1. **Schema for the YAML registry** — must add: per-resource `type`
   taxonomy (credential_pathway | local_training | apprenticeship |
   online_course | referral_only), `verified_at`, `verified_by`, a
   canonical `gap` field for lookup, and an optional list of skill
   aliases the engine's missing_skills strings might use.
2. **Provider allowlist doc** (`docs/training-providers-allowlist.md`) —
   the agreed list of trusted sources with concrete examples.
3. **YAML seed** for 13 priority gaps, reviewed by the lead engineer
   before merge.
4. **Registry loader** — pure-Python singleton, no DB calls.
5. **Recommender swap** — point `_attach_training` at the registry by
   gap name instead of/in addition to `suggest_for_skill`.
6. **Prompt carve-out** — allow bullets in `explain_gap` responses when
   narrating training resources.
7. **Unknown-gap telemetry** — log structured INFO line when the
   recommender is asked for a gap not in the registry.

## 10. Summary

| Question | Answer |
|---|---|
| How many training resources exist? | 0 |
| Which providers? | none |
| Do they have real URLs? | n/a |
| Local, online, or generic? | n/a |
| Do any map to 310T / Class G / WHMIS / forklift / PSW / Excel? | none of them |
| Are rows fresh / usable? | n/a |
| **Decision** | **Seed YAML registry** |
