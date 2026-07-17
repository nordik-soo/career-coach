# Step 1B — AWIC SSM Recovery (REJECTED — DO NOT IMPLEMENT)

> **STATUS: REJECTED (2026-07-16).**
> Employer identity, title text, and excerpt text CANNOT establish a
> posting's job location. This draft's proposed derivation from those
> fields — however carefully labelled as `employer_declared`,
> `title_declared`, `excerpt_declared` — reintroduces the exact
> location-integrity defect Step 1A was locked to fix. **Renaming an
> inference does not make it evidence.**
>
> Do not implement any part of §4 (the classifier), §4.2 (the three
> new provenance values), §4.3 (the known-SSM-employer allowlist),
> §4.4 (the ingestion wiring), or §4.6 (the one-time enrichment CLI).
>
> **What is preserved from this draft (for design history):**
>
>   - §2 — the URL-domain probe that invalidated the original
>     "fetch each AWIC detail page" plan. Every one of the 189 AWIC
>     apply-URLs points at a third-party host; 21 of them at
>     `jobbank.gc.ca` which is explicitly forbidden by BREAKING.md.
>     This is a durable factual finding.
>
>   - The five reviewer objections that killed the metadata-inference
>     approach (recorded verbatim in §10 REJECTION RATIONALE below).
>
> **Direction for the eventual valid Step 1B (not designed here):**
> evidence-first recovery — ask AWIC for an explicit location field;
> deduplicate AWIC URLs against SCCC authoritative postings; add
> first-party connectors for employers whose OWN detail page carries
> an explicit "Location: …" field; skip Indeed / Job Bank / Eluta /
> per-vendor parser fleets; preserve unrecoverable rows as
> `unresolved`; use explicit evidence-upgrade transitions (not
> COALESCE) that permit `unresolved → resolved` only on stronger
> authoritative evidence and block downgrades.

**Original draft below is preserved unedited for historical context.
It does not reflect the current Step 1B direction.**

---

**Status (original):** DRAFT for reviewer. Not locked. Do NOT implement until locked.
**Date:** 2026-07-16.
**Parent:** Step 1A source-data integrity (locked + shipped as commit `1dc5766`).
**Governing invariant:** The five-line invariant from [qualification-v2-architecture.md](qualification-v2-architecture.md).

## 1 — Why Step 1B exists

Step 1A locked `v_current_job` to `location_resolution_status='resolved' AND normalized_job_location='Sault Ste. Marie'`. That correctly excluded all 189 AWIC postings because AWIC v1 metadata provides no source-declared location text — only geometry, which is not authoritative (a "Community Support Worker" in Wawa carries coordinates pointing at downtown SSM).

The live SSM market is currently 27 SCCC-only postings. AWIC's 189 rows sit in `core.job_posting` waiting to be honestly classified. Step 1B recovers the SSM-verified subset of those 189 rows into the market **without violating the source-integrity rule**.

## 2 — What CANNOT be done (constraint discovery)

The original plan was "fetch each AWIC detail page." A live probe against `core.job_posting` for `source='awic_jobs'` reveals this plan is not viable:

**None of the 189 `properties.url` values point at awic.ca.** Every AWIC posting's apply-URL is a THIRD-PARTY host:

| Domain | Count | Notes |
|---|---:|---|
| sah.on.ca | 46 | Sault Area Hospital career page |
| ca.indeed.com | 40 | Indeed |
| www.jobbank.gc.ca | 21 | Federal — **forbidden per BREAKING.md** |
| www.eluta.ca | 19 | Eluta |
| employment-solutions.ca | 14 | Recruitment vendor |
| saultcareercentre.ca | 13 | SCCC (dup with our own SCCC source) |
| www.sootoday.com | 12 | Local news outlet |
| www.careerbeacon.com | 8 | CareerBeacon |
| www.wawarehc.com | 6 | Wawa community site (not SSM) |
| Others | 10 | jobillico, scotiabank, kijiji, staples, welcometossm |

Two of these categories rule out the fetching approach:

- **jobbank.gc.ca (21 rows) is forbidden.** BREAKING.md's durable source-purity rule prohibits any fetch against federal sources. Step 1B cannot fetch these under any policy.
- **Everything else is heterogeneous.** Twelve different domains each with its own ToS, robots policy, HTML structure, rate-limit expectation, and per-vendor parser burden. Building a fetcher fleet for what would recover fewer than 130 postings is disproportionate engineering for a project the size of SkillBridge.

**Step 1B therefore does not fetch anything.** It derives verified SSM status from AWIC's own metadata (title, employer, excerpt) which we already have.

## 3 — What CAN be done (recovery via existing metadata)

AWIC's own metadata carries strong location signals in fields we already store. From the same live probe:

- 12% of excerpts (23 rows) explicitly mention SSM aliases
- 26% of AWIC rows are employer=Sault Area Hospital Foundation (SAH), a known SSM entity
- Multiple titles include location clauses ("Customer Experience Associate - Sault Ste. Marie")
- 44% of rows have an employer name whose text alone flags SSM/Sault/Algoma (crude heuristic; the true honest signal is higher when title + excerpt are also parsed)

**Realistic coverage target: 100–130 of 189 AWIC rows recovered as SSM-verified.** Combined with 27 SCCC, the live market grows to roughly 130–160 postings. Not dramatic, but honest.

The remaining 60–90 AWIC rows have no discoverable SSM signal from AWIC's own metadata (e.g., "Scotiabank" with a generic title and no location in the excerpt). They stay unresolved — Step 1A's rule holds: unknown evidence never becomes a resolved-elsewhere match.

## 4 — Design (contract)

### 4.1 New pure function

Add to [skillbridge/ingest/partners.py](../../skillbridge/ingest/partners.py):

```python
def _derive_awic_location_from_metadata(
    title: str,
    employer: str | None,
    excerpt: str | None,
    known_ssm_employers: frozenset[str],
) -> tuple[str | None, str, str]:
    """Return (normalized_job_location, location_resolution_status,
    location_provenance) from AWIC's own metadata fields.

    Ordered precedence (first match wins):
      1. Employer name (after case-fold + punctuation strip) matches
         an entry in `known_ssm_employers` — a curated set derived
         from `core.approved_job_source` employer connectors.
         → (SSM, resolved, employer_declared)
      2. Employer text runs through `normalize_declared_job_location`
         and canonicalizes to SSM.
         → (SSM, resolved, employer_declared)
      3. Title text runs through `normalize_declared_job_location`
         and canonicalizes to SSM.
         → (SSM, resolved, title_declared)
      4. Excerpt text runs through `normalize_declared_job_location`
         and canonicalizes to SSM.
         → (SSM, resolved, excerpt_declared)
      5. Steps 2/3/4 canonicalize to a non-SSM Algoma community
         (Wawa, Blind River, Chapleau, etc.).
         → (that community's canonical form, resolved,
            <same provenance>_declared)
      6. None of the above resolves.
         → (None, unresolved, geometry)  # unchanged from Step 1A

    Pure function — no DB, no HTTP. Testable end-to-end with
    fixtures. Every code path reuses `normalize_declared_job_location`
    from Step 1A so the SSM alias set is a single source of truth.
    """
```

### 4.2 New enum values

`location_provenance` currently accepts `source_declared`, `detail_page`, `geometry`, `multiple`, `none`. Step 1B adds:

- `employer_declared` — location resolved from the employer field
- `title_declared` — location resolved from the title field
- `excerpt_declared` — location resolved from the excerpt field

All three signal to consumers that the classification is INFERRED from AWIC's own metadata, not from a source-supplied location text. Distinct provenance keeps analytics/telemetry honest.

**Alternative rejected:** reusing `source_declared` for these three. Rejected because the classification is DERIVED, not source-supplied. Conflating them would hide the inference chain from downstream analytics.

**Schema change required.** The `location_provenance` CHECK constraint at [sql/schema.sql](../../sql/schema.sql) (added in Step 1A) extends to include the three new values. This is a schema migration — the Step 1B commit installs the extended constraint.

### 4.3 Known-SSM-employer allowlist

Seed the `known_ssm_employers` set from the employer connectors already vetted in `core.approved_job_source` (WHERE scope='employer'):

```
sault_area_hospital       → "Sault Area Hospital", "Sault Area Hospital Foundation"
city_of_ssm_hr            → "City of Sault Ste. Marie", "City of SSM"
algoma_steel              → "Algoma Steel", "Algoma Steel Inc"
sault_college_careers     → "Sault College"
algoma_u_careers          → "Algoma University"
puc                       → "PUC Services", "PUC"
group_health_centre       → "Group Health Centre"
ymca_ssm                  → "YMCA of Sault Ste. Marie", "YMCA SSM"
cas_algoma                → "Children's Aid Society of Algoma", "CAS Algoma"
adsab                     → "Algoma District Services Administration Board", "ADSAB"
school_board              → the local school-board names as they appear
```

Stored as a module-level `frozenset[str]` in `partners.py` (case-folded, punctuation-normalized). Not a config knob — this is a curated allowlist that ships with code review. Adding an employer to the allowlist is a code change, not an env toggle.

### 4.4 Wiring the derivation into ingestion

In `_normalize_awic_geojson_feature`, AFTER the current Step 1A classification block that emits `(unresolved, geometry)`, INSERT a derivation call:

```python
# Step 1A default: (unresolved, geometry) from coordinates.
location_resolution_status = "unresolved"
location_provenance = "geometry"
normalized_job_location = None

# Step 1B enrichment: try to derive from metadata.
derived_norm, derived_status, derived_prov = (
    _derive_awic_location_from_metadata(
        title_str, employer_str, excerpt_text,
        _KNOWN_SSM_EMPLOYERS,
    )
)
if derived_status == "resolved":
    normalized_job_location    = derived_norm
    location_resolution_status = "resolved"
    location_provenance        = derived_prov
```

No other AWIC fields change. If derivation says unresolved, Step 1A's coordinate-preserving behavior stands.

### 4.5 Idempotence and Step 1A anti-regression

Downgrade guards from Step 1A's `_update_evidence_only` and the SCCC-legacy direct-SQL path remain in force. Any consumer that re-invokes the classifier on an already-`resolved` row via COALESCE-guarded UPDATE will NOT downgrade.

**New anti-regression rule:** a Step 1B enrichment run must never re-`unresolve` a row that Step 1A already resolved. Concretely: if `_derive_awic_location_from_metadata` returns `unresolved` for a row that was previously resolved by (say) manual DB edit, the write path COALESCE-guard preserves the resolved value. This is the same guard the Step 1A backfill uses; no new machinery.

### 4.6 One-time enrichment path for the existing 189 rows

The classifier is pure and deterministic. To pull the existing 189 AWIC rows through the new classifier immediately (without waiting for the next AWIC daily fetch), Step 1B ships a one-time enrichment CLI analogous to `--step1a-backfill`:

```
python run_pipeline.py --step1b-awic-derive [--dry-run]
```

Reads latest `raw.job_posting` payloads for `source='awic_jobs'`, re-runs `_normalize_awic_geojson_feature` (which now includes the derivation), then updates `core.job_posting` via a narrow UPDATE that only touches the eight new-axis columns + legacy `location` + `updated_at`. Same COALESCE guards. Same downgrade safety.

Retired analogously to Step 1A backfill: keep the CLI until every environment (local + staging + production) has been enriched, then a follow-up commit deletes it.

### 4.7 What does NOT change

- No new HTTP client.
- No third-party fetch of any kind.
- No AWIC connector-level rate-limit tuning (no new outbound traffic).
- No changes to `v_current_job` (it already filters on the correct columns).
- No changes to `_location_boost` — that was deleted in Step 1A.
- No changes to the matching engine.
- No changes to the SCCC or partner CSV or employer connectors.
- No changes to Step 1A's `_update_evidence_only` narrow UPDATE — Step 1B uses the same helper for the one-time enrichment.

## 5 — Test plan

Load-bearing tests before Step 1B ships:

1. **`_derive_awic_location_from_metadata` — unit table.** 20+ fixtures covering:
   - Known-SSM employer (SAH) → resolved / employer_declared
   - Employer text canonicalizes to SSM → resolved / employer_declared
   - Title contains SSM alias, employer generic → resolved / title_declared
   - Excerpt contains SSM alias, title/employer generic → resolved / excerpt_declared
   - Excerpt contains "Wawa" → resolved / excerpt_declared / normalized_job_location != SSM
   - No signal anywhere → unresolved / geometry / None
   - Employer text contains SSM alias AND Wawa (title says Wawa, employer says SAH) → SAH wins per precedence (employer over title)
   - Empty strings, None fields → unresolved
   - Employer name with typo — allowlist match only after case-fold + punctuation strip
2. **AWIC ingestion integration test.** A synthetic AWIC feature with employer="Sault Area Hospital Foundation" and no coords, no excerpt → produces `resolved/employer_declared/Sault Ste. Marie`. Same feature with employer="Scotiabank" and generic title → produces `unresolved/geometry`.
3. **Schema-constraint anti-regression.** Attempting to INSERT a row with `location_provenance='bogus'` fails; INSERT with `location_provenance='employer_declared'` succeeds.
4. **Backfill CLI dry-run.** Against a fixture DB with the current 189 AWIC rows, reports the expected distribution.
5. **`v_current_job` growth check after enrichment.** Post-enrichment, `SELECT COUNT(*) FROM core.v_current_job` returns something in the 100–160 range (SCCC's 27 + AWIC-derived).
6. **Downgrade anti-regression.** A previously-resolved AWIC row is not downgraded to `unresolved` by a re-run of the enrichment.
7. **jobbank-URL anti-regression.** An AWIC row whose `properties.url` is `www.jobbank.gc.ca/...` is treated identically to any other AWIC row (URL is never inspected as source-purity policy).

## 6 — Rollout order (proposed)

1. Design lock (this document + reviewer approval).
2. Schema migration commit: extend `location_provenance` CHECK constraint to include the three new values. No behavior change until the classifier ships.
3. Classifier + AWIC-ingest wiring + tests as one commit.
4. `--step1b-awic-derive` CLI + tests as one commit.
5. Local DB dry-run → verify projected market growth.
6. Local DB live enrichment → verify with reviewer's SSM-only integrity query (must still be 0 non-SSM in v_current_job).
7. Reviewer sign-off.
8. Push + staging + production migration.
9. After every environment is enriched, follow-up commit deletes the CLI.

## 7 — Open design questions for reviewer

These are the calls that gate the design lock:

**Q1.** Three new provenance values (`employer_declared`, `title_declared`, `excerpt_declared`), or one consolidated `derived_from_metadata`? Three preserves the inference chain in analytics; one is simpler.

**Q2.** Known-SSM-employer allowlist source of truth: hardcoded module-level frozenset, or read from `core.approved_job_source` at runtime? Hardcoded is auditable in review; runtime read allows adding an employer via schema migration without a code change.

**Q3.** When Step 1B classifies a row as `resolved/non-SSM-Algoma-community` (e.g., Wawa), does it stay OUT of `v_current_job` (correct behavior — market is SSM-only) OR should the classifier only emit non-`None` when the resolution is SSM? Both are defensible; the first preserves honest data for future analytics, the second keeps the classifier's output space narrower.

**Q4.** Excerpt scanning: whole-excerpt text vs. only the first N words? Whole-excerpt catches "we're based in Sault Ste. Marie but also have Wawa openings" (which then resolves ambiguously — needs a tiebreak rule). First-N is more conservative.

**Q5.** Do I need to keep a `derivation_confidence` field, or is the four-tier precedence enough? Step 1B's classifier is deterministic, so confidence is arguably always 1.0 for its output — but a downstream evaluator (Step 6 comparator's coverage math) might want a per-row derivation confidence for the same reasons Step 1A tracks extraction_confidence.

## 8 — What Step 1B does NOT do

- Does not fetch third-party pages.
- Does not touch federal sources (jobbank.gc.ca URLs are never inspected).
- Does not modify SCCC, partner CSV, upload, or employer connectors.
- Does not touch the matching engine, recommender, or renderer.
- Does not add or change any prompt.
- Does not introduce learned classification (rule-based only).

## 9 — What Step 1B unblocks

- The live SSM market grows from 27 to ~100–160 postings.
- AWIC rows carry honest, auditable location provenance.
- Step 2 (title-to-fit removal) can proceed with a market size that gives meaningful test signal.
- Steps 3–8 (V2 qualification matcher) can be shadow-run against a market that isn't distorted by the AWIC gap.

---

**Reviewer action requested:** answer Q1–Q5, or amend the shape of the design. Nothing implements until this document is marked LOCKED.

---

## 10 — REJECTION RATIONALE (2026-07-16)

The reviewer rejected this draft. Recorded here verbatim so future readers see WHY the metadata-inference direction was killed:

### R1 — Employer identity is not location evidence

> This proposed rule is unsafe: Employer matches Sault Area Hospital → job location = Sault Ste. Marie. A local employer can advertise: Remote work; Regional work; Satellite-site work; A position located elsewhere. Step 1A explicitly locked: Employer identity does not prove a posting's location. The draft reverses that decision.
>
> Its own test demonstrates the problem: Employer = SAH, Title says Wawa → employer wins → classify as SSM. That test would deliberately lock a false location into production.

### R2 — Title and excerpt are not declared-location fields

> This text is not necessarily the job location: "Our organization is based in Sault Ste. Marie, with this opening serving Wawa." Scanning it for Sault Ste. Marie would produce the wrong answer.
>
> Calling the provenance `title_declared`, `excerpt_declared`, `employer_declared` is also inaccurate. AWIC did not declare those fields to be job-location fields. The proposed system inferred location from unrelated text.

### R3 — Conflict with exact-match normalization

> `normalize_declared_job_location()` intentionally rejects wrapped prose: "Customer Associate - Sault Ste. Marie", "Based near SSM", "Wawa / SSM". It accepts an entire declared-location value such as: "Sault Ste. Marie, ON". Therefore, passing complete titles and excerpts into that function will not produce the projected recovery. Achieving the estimated 100–130 recovered rows would require substring or heuristic scanning, which Step 1A explicitly prohibited.

### R4 — Unsupported market-growth estimate

> These claims are not verified: 55–70% recovered; market grows to 100–160. Employer-name and substring counts measure possible hints, not verified job locations.
>
> A test like `assert 100 <= current_market_count <= 160` is inappropriate — it rewards recovery volume and could encourage false classifications. Correctness tests should verify exact evidence decisions, not a desired market size.

### R5 — Backfill cannot upgrade with COALESCE

> Existing AWIC rows already contain `location_resolution_status='unresolved'` and `location_provenance='geometry'`. The proposed write uses COALESCE(existing, incoming). Because those existing values are non-null, they remain `unresolved`/`geometry` even when Step 1B supplies a new classification. The proposed backfill therefore cannot perform the upgrade it claims.
>
> A future enrichment update needs explicit transition rules: `unresolved/missing/invalid → resolved` only when stronger valid evidence arrives, while preventing `detail_page/source_declared` resolved evidence → weaker derived evidence.

### Answers to Q1–Q5 (from reviewer)

- **Q1 (provenance values).** None of the proposed `*_declared` values should ship. If metadata inference is retained for analytics, use `location_resolution_status='inferred'` and `location_provenance='metadata'` and keep it OUT of `v_current_job`. Do not promote inferred evidence to `resolved`.
- **Q2 (employer allowlist).** Neither hardcoded nor database-driven employer allowlists establish location. `core.approved_job_source` means "approved ingestion source" — NOT "every job from this employer is located in SSM."
- **Q3 (resolved non-SSM).** When an explicit, authoritative location says Wawa: `normalized_job_location='Wawa'`, `location_resolution_status='resolved'`. Preserve that fact; `v_current_job` correctly excludes it. Do not make the classifier SSM-only.
- **Q4 (excerpt scanning).** Neither whole-excerpt nor first-N scanning is safe. Only accept an explicit structured field or a tightly identified label such as `Location: Sault Ste. Marie`. If multiple locations are present, classify as `conflicting`/`unresolved` rather than choosing the first.
- **Q5 (confidence).** Do not add numeric confidence. A deterministic rule can still make an uncertain inference — rule executed deterministically ≠ evidence is certainly correct. Use evidence categories: `resolved`, `unresolved`, `missing`, `invalid`, `conflicting`, `inferred`. Only `resolved` from authoritative location evidence enters the market.

### Recommended Step 1B direction (for the eventual valid design)

Evidence-first recovery strategy:

1. Ask AWIC / feed owner for an explicit location field.
2. Recover duplicates through existing trusted connectors (SCCC deduplication against AWIC's `properties.url`).
3. Add selected first-party connectors where location is explicitly stated on the employer's OWN detail page.
4. Avoid Indeed, Job Bank, Eluta, and arbitrary parser fleets.
5. Preserve all unrecoverable AWIC rows as `unresolved`.
6. Implement explicit evidence-upgrade transitions instead of COALESCE.

**Illustrative cases:**

- AWIC URL duplicates an SCCC posting → use the authoritative SCCC posting; do not promote the AWIC copy heuristically.
- SAH first-party detail page explicitly says `Location: Sault Ste. Marie` → `resolved`/SSM, eligible for `v_current_job`.
- Employer says SAH but no job location appears → `unresolved`, excluded.
- Title says "Wawa" → useful retrieval hint, not authoritative location unless parsed from an explicit location field.

### The durable lesson

Renaming an inference does not make it evidence. Before proposing any derivation, cross-check every input, transformation, status, and output against the locked invariants of prior steps. See feedback memory: `feedback_cross_check_derivations.md`.
