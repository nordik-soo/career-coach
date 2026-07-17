# Step 1A — Source-Data Integrity (SSM-only market)

**Status:** SPEC v4 — corrections after v3 audit landed. Ready to code.
**Blocks:** every downstream step in [project_matching_revise]. Nothing else in the plan can be trusted until source data is preserved intact and the SSM market boundary is honest.
**Parent:** [matching-revise (workforce memory)](../../../../workforce/.claude/memory/project_matching_revise.md) — path is workforce-relative; the two repos live side-by-side under `C:/Users/NazmulHossen/workforce/`.
**Follow-on:** Step 1B (AWIC detail-page fetch) — required for AWIC postings to re-enter the live SSM market.

**Change log from v3:**
- **One shared SSM location-alias registry.** New function `normalize_declared_job_location` extends `skillbridge.match.region` and shares `_SSM_PROPER_LOCATION_ALIASES` (`{sault ste. marie, sault ste marie, ssm}`) with `is_ssm_region_job`. Verified region codes `_SSM_PROPER_REGION_CODES` (`{3557011, ssm}`) remain exclusively part of `is_ssm_region_job`'s higher-precedence `region_code` path — `normalize_declared_job_location` does not consult them (region codes aren't location strings; they belong on `job.region_code`, not `job.location`). The v3-invented `"sault"` and `"sault sainte marie"` aliases are dropped.
- **Exact alias matching after cleanup.** The new function does NOT reuse `is_ssm_region_job`'s word-boundary `.search()` — that would admit "Wawa / SSM" and "North of SSM" as SSM matches, contradicting §7 principle 3. Cleanup happens first (whitespace, "(Remote)", ", ON"/", Ontario"); then the cleaned string is compared for **exact case-insensitive equality** against the alias set.
- **`is_ssm_region_job`'s location fallback is retrofitted** to consult the new function's result — region-code precedence unchanged. This tightens `is_ssm_region_job`'s existing substring-of-string behavior (intentional; documented as a fix, not a regression).
- **View change is a system-wide market redefinition.** `v_current_job` is consumed by public jobs API, pipeline snapshot, JD matching, adjacency, detail refetch, development plan, legacy chat. Rollout gets a `dataset_version` / `engine_version` bump per the schema.sql:466 rule, plus per-consumer regression tests.
- **Backfill deterministically selects the latest raw payload per `(source, source_job_id)`** via `SELECT DISTINCT ON (source, source_job_id) ... ORDER BY source, source_job_id, ingested_at DESC, raw_id DESC` (see §3a). One row per posting; no global row limit.
- **Coordinate validation locked:** array length ≥ 2, longitude ∈ [-180, 180], latitude ∈ [-90, 90], additional values preserved and ignored (3-D GeoJSON points are valid).
- **AWIC definition of done matches per-row rules** (unresolved/missing/invalid per coordinate state; no universal "unresolved" assertion).
- **Config cleanup:** `_location_boost` deletion carries `MATCH_LOCATION_BOOST_LOCAL_CSD` with it in Step 1A. `LOCAL_CITIES` is renamed to `SCCC_INGEST_LOCALITIES` (ingestion-owned) so it can no longer be misread as a matching-market allowlist.
- **`description_evidence_status="parse_error"`** assignment path made explicit.

---

## 1. Problem statement (code + live-DB verified 2026-07-15)

### 1a. AWIC — location hardcoded, description truncated to excerpt

At [partners.py:556-580](../../skillbridge/ingest/partners.py#L556-L580):

```python
job = NormalizedJob(
    source="awic_jobs", ...
    location="Sault Ste. Marie",                       # hardcoded
    description=_str_or_none(props.get("excerpt")),    # excerpt only
    raw_payload={"source": "awic_geojson_v1", "feature": feature},
)
```

Live-verified: the AWIC feature's `properties` object contains `{url, hash, nocs, type, email, excerpt, post_id, duration, employer, job_title, nocs_2021, experience, employer_id, employer_slug}` — no full description, no location, no address. The stored raw feature is preserved intact but is not a full source posting.

Geometry is unreliable for job location: Community Support Worker's coordinates `[-84.331146, 46.533713]` point at downtown SSM despite the actual role being in Wawa. Reverse-geocoding these produces a lie that looks authoritative. Structurally distrust geometry for job-location purposes.

### 1b. SCCC — full description preserved; ingestion scope is Algoma-wide by config

At [partners.py:261](../../skillbridge/ingest/partners.py#L261) SCCC captures full `content.rendered` from WP. Location at [partners.py:255](../../skillbridge/ingest/partners.py#L255) uses source-declared `meta._job_location`.

SCCC ingestion filter `_is_sccc_ssm_location` at [partners.py:584](../../skillbridge/ingest/partners.py#L584) gates on `LOCAL_CITIES` which includes Wawa, Elliot Lake, Blind River, Chapleau. Under SSM-only product scope, SCCC ingestion may keep this permissive filter (archival purpose), but the matcher's view must not surface non-SSM postings.

### 1c. Downstream contamination is a market eligibility defect

Users see non-SSM postings labeled as SSM jobs because:
- AWIC hardcodes location before ingest can distinguish.
- The matcher's view (`v_current_job`) admits every current row regardless of city.

This is not a scoring problem to soften; it's a market boundary that must be moved to the truth. Under SSM-only product scope, the matching engine's input becomes SSM-verified postings only.

### 1d. `is_ssm_region_job` is the existing SSM authority

[region.py:72](../../skillbridge/match/region.py#L72) defines `is_ssm_region_job(job) → bool` with:

- Alias set `{sault ste. marie, sault ste marie, ssm}`, matched at word boundaries so "Rossmore" containing "ssm" as a substring never slips through.
- Verified region codes `{3557011, ssm}` (Statistics Canada CSD 3557011 for SSM; legacy fixtures use literal "SSM").
- Decision precedence: region_code (if present) is authoritative; otherwise fall back to location text; otherwise conservatively reject.
- Explicit "does NOT consult `config.LOCAL_CITIES`" — the module was written specifically to avoid the multi-community allowlist trap.

Consumed by [handler.py](../../skillbridge/chat/handler.py), [development_plan.py](../../skillbridge/chat/development_plan.py), [adjacent.py](../../skillbridge/match/adjacent.py).

**Step 1A extends this module** — it does not create a competing authority. The canonicalization function that populates `normalized_job_location` and the runtime predicate share one alias/code registry.

---

## 2. Locked design (v4)

### 2a. Schema changes to `core.job_posting`

**Description axis — three columns:**

| Column | Type | Purpose |
|---|---|---|
| `description_full` | TEXT | Complete JD body when source provided it. NULL when only an excerpt was captured. |
| `description_excerpt` | TEXT | Short preview when source separately provides one. NULL when full text serves both. |
| `description_evidence_status` | VARCHAR(20) | `full_source` \| `excerpt_only` \| `missing` \| `parse_error`. Provenance-derived. |

**Location axis — five columns:**

| Column | Type | Purpose |
|---|---|---|
| `source_location_text` | TEXT | Exact source-declared location string. NULL when source did not provide one. Never a hardcoded fallback. |
| `source_coordinates` | JSONB | Preserved GeoJSON coordinate array when source provides one. Represents the source's indexing point, NOT authoritative for job location. |
| `normalized_job_location` | TEXT | Trustworthy job location. Populated only when resolution succeeded. Otherwise NULL. |
| `location_resolution_status` | VARCHAR(20) | `resolved` \| `unresolved` \| `missing` \| `conflicting` \| `invalid`. |
| `location_provenance` | VARCHAR(20) | `source_declared` \| `detail_page` \| `geometry` \| `multiple` \| `none`. |

**Legacy columns** (`location`, `description`) retained one release cycle; `location` becomes a copy of `normalized_job_location` (NULL when not resolved) and `description` becomes `COALESCE(description_full, description_excerpt)`.

### 2b. `NormalizedJob` shape changes

At [base.py:34](../../skillbridge/ingest/base.py#L34), extend the dataclass with the eight new fields plus the two deprecated legacy fields. Directly maps to the columns in §2a.

### 2c. Location normalization — one shared authority, exact match after cleanup

**Extend `skillbridge.match.region`** (do not create a new module). Add a normalization function next to `is_ssm_region_job`:

```
def normalize_declared_job_location(
    source_location_text: str | None,
) -> tuple[str | None, bool]:
    """Deterministic normalization for `normalized_job_location`.

    Also returns non-SSM cities truthfully ("Wawa", "Elliot Lake",
    etc.) — hence the name. SSM classification is one specific
    outcome; the function's role is broader.

    Reuses `_SSM_PROPER_LOCATION_ALIASES` (the location-alias registry
    shared with `is_ssm_region_job`) — see region.py — but matches
    EXACTLY against the cleaned string, not via word-boundary substring
    search. Does NOT consult `_SSM_PROPER_REGION_CODES`; those are
    scoped to `is_ssm_region_job`'s higher-precedence `region_code`
    path, and belong on `job.region_code`, not `job.location`.

    Returns (normalized_location, remote_flag).
    normalized_location is:
      - "Sault Ste. Marie" iff the cleaned string EXACTLY equals a
        registered SSM alias (case-insensitive)
      - the cleaned, Title-Cased input otherwise (Wawa → "Wawa")
      - None if the input is blank or non-string
    remote_flag is True iff a "(Remote)" suffix was stripped.
    """
```

**Rationale for exact match:** the existing `is_ssm_region_job` at [region.py:97](../../skillbridge/match/region.py#L97) uses `p.search(loc)` — word-boundary substring search. That admits "Wawa / SSM" and "North of SSM" as SSM matches, which contradicts §7 principle 3 ("no substring inference"). The new function does NOT reuse the search patterns; it compares the cleaned string against alias literals with case-insensitive equality.

**Pipeline (applied in order):**

1. Reject non-string / empty → return `(None, False)`.
2. Trim; collapse internal whitespace to single spaces.
3. Strip trailing `"(Remote)"` (case-insensitive, optional whitespace); set `remote_flag = True` if stripped.
4. Strip trailing `", ON"` or `", Ontario"` (case-insensitive).
5. Compare the lowercased result **exactly** against the alias set. Exact match → return `("Sault Ste. Marie", remote_flag)`.
6. Otherwise return `(title_case(cleaned_input), remote_flag)`.

**Alias registry (unchanged from `is_ssm_region_job`):**

```
_SSM_PROPER_LOCATION_ALIASES = {
    "sault ste. marie",
    "sault ste marie",
    "ssm",
}
```

No `"sault"` bare. No `"sault sainte marie"`. No `", ON"` suffix variants baked into the alias set — steps 3-4 of the cleanup pipeline handle those.

**Forbidden inference paths** (any of these firing is a bug):
- Fuzzy string match (e.g., "Sault Ste. Maria" → SSM)
- Substring match ("Ste. Marie" alone → SSM)
- Word-boundary substring ("Wawa / SSM" → SSM). ← Explicitly ruled out by exact-match rule.
- Coordinate → CSD reverse-geocoding
- URL-based inference (`saultcareercentre.ca` → SSM)
- Employer-name inference (`Sault Area Hospital` → SSM)

If a row's `source_location_text` doesn't exactly match an alias, `normalized_job_location` gets the cleaned Title-Cased value verbatim. Wawa, Elliot Lake etc. get their real names — no fake SSM tag.

**Alias additions require evidence.** Adding a new alias means seeing it in actual source-location data AND relocking the SSM-proper scope alongside `chat/prompts.py` and `chat/responder.py` per the region.py header comment. Not a spec-time invention.

### 2c.1 `is_ssm_region_job` retrofit

Update the location fallback path in [region.py:97](../../skillbridge/match/region.py#L97) to consult `normalize_declared_job_location`:

```
def is_ssm_region_job(job: dict[str, Any]) -> bool:
    # Precedence 1: region_code (unchanged).
    code_raw = job.get("region_code")
    code = code_raw.strip().lower() if isinstance(code_raw, str) else ""
    if code:
        return code in _SSM_PROPER_REGION_CODES

    # Precedence 2: exact normalized location match (was: word-boundary
    # substring search). Tightened 2026-07-15 (Step 1A) to remove
    # "Wawa / SSM"-style substring admits.
    loc = job.get("location")
    normalized, _remote = normalize_declared_job_location(loc)
    return normalized == "Sault Ste. Marie"
```

**Behavior change (intentional):** inputs that previously matched via substring but not exact equality are now rejected. Examples:

- `"Wawa / SSM"` (previously matched via word-bounded "ssm"; now False)
- `"North of SSM"` (previously matched; now False)
- `"SSM area"` (previously matched; now False)

These were bugs: word-bounded substring search cannot enforce SSM-proper scope. The retrofit is documented as a fix in the schema migration notes so downstream consumers (chat/handler, chat/development_plan, match/adjacent) can update fixtures if any depended on the substring behavior. The `_SSM_PROPER_LOCATION_PATTERNS` list at region.py:54 is retired in Step 1A.

### 2d. Coordinate validation

`source_coordinates` acceptance rules:

- Must be a JSON array.
- Array length ≥ 2. Entries beyond index 1 are preserved verbatim but ignored for validation (3-D GeoJSON points remain valid).
- Element 0 (longitude): finite numeric in `[-180, 180]`.
- Element 1 (latitude): finite numeric in `[-90, 90]`.
- Any other shape → `source_coordinates = NULL`, `location_resolution_status = "invalid"`, `location_provenance = "geometry"`.

### 2e. Market eligibility (replaces the location boost)

**System-wide impact acknowledged.** `v_current_job` is consumed by the public jobs API, `pipeline.v_data_status`, JD matching, adjacency, follow-up detail, development planning, and legacy chat paths. Tightening the view is a market-definition change, not a matcher-local one.

Per [schema.sql:466-469](../../sql/schema.sql#L466-L469) the view is user-facing product truth and not env-configurable. Changing it requires:

- Migration adding the SSM eligibility clauses.
- `dataset_version` / `engine_version` bumps in the API envelope.
- Regression tests for every consumer (jobs API, matching, adjacency, detail refetch, development plan, pipeline status, legacy chat).

**New `v_current_job` definition:**

```
SELECT *
  FROM core.job_posting
 WHERE is_active = TRUE
   AND last_seen_at >= NOW() - INTERVAL '2 days'
   AND (closing_date IS NULL OR closing_date >= CURRENT_DATE)
   AND location_resolution_status = 'resolved'
   AND normalized_job_location = 'Sault Ste. Marie'
```

**Consumer-visible effects:**

| Consumer | Before Step 1A | After Step 1A |
|---|---|---|
| Public jobs API | All Algoma-district current rows | SSM-verified rows only |
| `pipeline.v_data_status` | Algoma-district count | SSM-verified count |
| JD matching | Boost + include Algoma | Include SSM-only, no boost |
| Adjacency | Already SSM-only via `is_ssm_region_job` | Unchanged behavior; matches view |
| Detail refetch | May load any current row | SSM-only |
| Development plan | Uses `is_ssm_region_job` | Unchanged behavior |
| Legacy chat | May quote Algoma rows | SSM-only |

**Engine consumption:**
- `_location_boost` at [engine.py:889](../../skillbridge/match/engine.py#L889) is **deleted**.
- `MATCH_LOCATION_BOOST_LOCAL_CSD` in [config.py](../../config.py) is **removed in Step 1A** (not deferred). Leaving the constant dangling risks reintroduction and contradicts §7 principle 2.
- `_fetch_eligible_jobs` at [engine.py:1899](../../skillbridge/match/engine.py#L1899) continues to read `v_current_job`; the stricter view enforces SSM-only automatically.

### 2f. Per-connector required behavior

**AWIC connector** (`_normalize_awic_feature`, [partners.py:556](../../skillbridge/ingest/partners.py#L556)):

Description axis:
```
description_full            = None       # AWIC v1 GeoJSON doesn't carry full body
description_excerpt         = props.get("excerpt") or None
description_evidence_status = "excerpt_only" if excerpt present
                              "missing"      if excerpt absent
```

Location axis:
```
source_location_text       = None        # AWIC v1 has no source location property
source_coordinates         = feature.geometry.coordinates if passes §2d validation
                             None        otherwise
normalized_job_location    = None        # never inferred from geometry
location_resolution_status = "unresolved" if source_coordinates nonblank (valid)
                             "missing"    if coordinates absent entirely
                             "invalid"    if coordinate data present but malformed
location_provenance        = "geometry"  if any coordinate attempt (valid, invalid, present-then-rejected)
                             "none"      if coordinates absent entirely
```

**Never** set `source_location_text = "Sault Ste. Marie"` under any input.
**Never** treat geometry as authoritative for job location.

**All new AWIC rows are excluded from the live SSM market until Step 1B.**

**SCCC connector** (`_normalize_sccc_item`, [partners.py:245](../../skillbridge/ingest/partners.py#L245)):

Description axis:
```
description_full            = _clean_wp_html(_get(item, "content", "rendered")) if nonblank
                              None otherwise
description_excerpt         = None
description_evidence_status = "full_source" if description_full nonblank
                              "missing"     otherwise
```

Location axis:
```
source_location_text       = meta.get("_job_location")   # exact source string
source_coordinates         = None                        # SCCC doesn't provide coordinates
(normalized_job_location, remote_flag) = normalize_declared_job_location(source_location_text)
location_resolution_status = "resolved" if normalized_job_location nonblank
                             "missing"  otherwise
location_provenance        = "source_declared" if source_location_text nonblank
                             "none"            otherwise
```

SCCC ingestion filter `_is_sccc_ssm_location` remains permissive (Algoma-wide) for archival. The eligibility gate happens at `v_current_job`, not at ingestion.

**Partner CSV connector** ([partners.py:653](../../skillbridge/ingest/partners.py#L653)):

Same shape as SCCC: `description_full` from the CSV description column; `source_location_text` from the CSV location column; `normalize_declared_job_location` for normalization; provenance `"source_declared"`.

**Partner upload connector** ([partner_upload.py:104](../../skillbridge/ingest/partner_upload.py#L104)):

Same shape as SCCC. `description_full` from `row.get("description") or row.get("Job Description")`; `source_location_text` from `row.get("location") or row.get("Location")`; `normalize_declared_job_location` for normalization; provenance `"source_declared"`. Blank strings are treated as absent (`missing` status).

**Employer-specific connectors — Sault Area Hospital ([employers/sault_area_hospital.py:67](../../skillbridge/ingest/employers/sault_area_hospital.py#L67)) and City of SSM ([employers/city_of_ssm.py:67](../../skillbridge/ingest/employers/city_of_ssm.py#L67)):**

Both currently hardcode `location="Sault Ste. Marie"`. Under Step 1A this hardcoded fallback is banned — same rule as AWIC. The connectors must either:

- **Extract the real location from the source page.** Sault Area Hospital's careers listing sometimes carries a per-role location field (site, department, city). Same for the City of SSM careers page. If the source page declares a location per posting, extract it as `source_location_text` and canonicalize via `normalize_declared_job_location`.
- **OR emit `missing` / `unresolved` and stay outside the live market.** When the source page doesn't declare a location per posting, honest handling is:
  ```
  source_location_text       = None
  source_coordinates         = None
  normalized_job_location    = None
  location_resolution_status = "missing"
  location_provenance        = "none"
  description_evidence_status = "full_source" if description captured, else "missing"
  ```

**Neither connector may fall back to `source_location_text = "Sault Ste. Marie"` on the assumption that Sault Area Hospital roles are always in Sault or that City of SSM roles are always in SSM.** Even for employers with "SSM" in the name, individual postings can be at satellite sites (Sault Area Hospital operates campuses in Thessalon and Wawa; City of SSM occasionally lists roles at outlying facilities). The eligibility gate at `v_current_job` will exclude these postings until per-posting location extraction is implemented — that's honest; the pre-Step-1A behavior was a lie.

### 2g. `description_evidence_status="parse_error"` assignment paths

Explicit list of causes:

- **SCCC:** `_clean_wp_html` raises or returns non-string on decode failure. Row still persists with `description_full = NULL`, `description_evidence_status = "parse_error"`.
- **Partner CSV:** description column value present but not a string (e.g., a number, list, or serialization error). Same treatment.
- **AWIC:** the `excerpt` property present but not a string (defensive; unlikely under GeoJSON schema).
- **Any connector:** description candidate present but exceeds a sanity size cap (e.g., > 1 MB — evidence of a data-format leak). Same treatment.

Distinguishes "we had data but couldn't use it" (`parse_error`) from "we didn't have data" (`missing`). Extractor and downstream consumers treat both as unusable, but ops dashboards can separate them.

### 2h. What is explicitly OUT of scope for Step 1A

- **Step 1B — AWIC detail-page fetch.** Required before AWIC re-enters live market. Owns its own spec.
- LLM-based description completeness classification (Step 4).
- Structured requirement extraction (Step 3).
- Legacy `location` / `description` column removal (one release cycle after Step 1A).
- Broader-than-SSM scope re-expansion (explicitly rejected under SSM-only product scope).
- Renaming `LOCAL_CITIES` to `SCCC_INGEST_LOCALITIES` is in-scope for Step 1A (see §5.4).

---

## 3. Migration strategy

### 3a. Backfill deterministically selects the latest raw payload

`raw.job_posting` is append-only ([schema.sql:427-433](../../sql/schema.sql#L427-L433)) — one `(source, source_job_id)` pair can have multiple rows over time. Backfill must select the latest:

```sql
WITH latest_raw AS (
    SELECT DISTINCT ON (source, source_job_id)
        source, source_job_id, payload
      FROM raw.job_posting
     ORDER BY source, source_job_id, ingested_at DESC, raw_id DESC
)
UPDATE core.job_posting cp
   SET ...
  FROM latest_raw lr
 WHERE cp.source = lr.source
   AND cp.source_job_id = lr.source_job_id;
```

This makes backfill idempotent — subsequent runs read the same "latest" raw row and produce identical results.

### 3b. Per-row status determination

**For AWIC rows** (from latest `raw.job_posting.payload`):

```
excerpt = payload -> 'feature' -> 'properties' ->> 'excerpt'
coords  = payload -> 'feature' -> 'geometry' -> 'coordinates'

description_full            = NULL
description_excerpt         = excerpt (may be NULL)
description_evidence_status = "excerpt_only" if excerpt nonblank
                              "missing"      otherwise

source_location_text        = NULL
source_coordinates          = coords if passes §2d validation (valid [lon,lat,…])
                              NULL   if coords absent, or absent from feature entirely
normalized_job_location     = NULL
location_resolution_status  = "unresolved" if source_coordinates nonblank
                              "missing"    if coords absent from payload
                              "invalid"    if coords present but failed §2d validation
location_provenance         = "geometry"   if any coordinate data attempted
                              "none"       if coords absent from payload
```

**For SCCC rows:**

```
content      = payload -> 'item' -> 'content' ->> 'rendered'
location_raw = payload -> 'item' -> 'meta' ->> '_job_location'

description_full            = _clean_wp_html(content) if nonblank else NULL
                              (parse_error if _clean_wp_html raises)
description_excerpt         = NULL
description_evidence_status = "full_source" if description_full nonblank
                              "parse_error" if _clean_wp_html failed
                              "missing"     if content absent/blank

source_location_text        = location_raw (may be NULL)
source_coordinates          = NULL
(normalized_job_location, remote_flag) = normalize_declared_job_location(source_location_text)
location_resolution_status  = "resolved" if normalized_job_location nonblank
                              "missing"  otherwise
location_provenance         = "source_declared" if source_location_text nonblank
                              "none"            otherwise
```

**For partner CSV rows:** analogous.

### 3c. Legacy field backfill

```
location    = normalized_job_location   # NULL when resolution didn't succeed
description = COALESCE(description_full, description_excerpt)
```

Historical AWIC `location = "Sault Ste. Marie"` values are wiped — the intended correction. Historical SCCC rows retain location (canonicalized) because source truth was preserved.

### 3d. Rollout order (locked 2026-07-16)

**Preparation phase** (additive, no behavioral change):
1. Schema migration (adds 8 new columns; nullable). Zero-downtime.
2. Add `normalize_declared_job_location` to `skillbridge.match.region` (reuses existing alias/code registry).
3. Update `NormalizedJob` dataclass with the 8 new fields (status defaults = `None` sentinel per correction 2026-07-16). Update `upsert_job` with transitional persistence (preserve legacy on `None` status; guard new-axis with `ON CONFLICT CASE WHEN`).

**Producer migration phase**:
4. Connector updates — all six `NormalizedJob(...)` producer sites:
    - `partners.py::_normalize_awic_feature` (AWIC)
    - `partners.py::_normalize_sccc_item` (SCCC)
    - `partners.py::_normalize_partner_csv_row` (Partner CSV)
    - `partner_upload.py` (Partner upload endpoint)
    - `employers/sault_area_hospital.py` (Sault Area Hospital — honest handling per §2f)
    - `employers/city_of_ssm.py` (City of SSM — honest handling per §2f)
5. Backfill script (deterministic per §3a; per-row per §3b).

**Symbol removal phase** (must precede consumer regression tests — those tests assert `_location_boost` and `MATCH_LOCATION_BOOST_LOCAL_CSD` are absent):
6. Prepare SSM-only view migration DDL (not applied yet).
7. Delete `_location_boost` from `engine.py`; remove `MATCH_LOCATION_BOOST_LOCAL_CSD` from `config.py`.
8. Rename `LOCAL_CITIES` → `SCCC_INGEST_LOCALITIES` (config + `.env` + `_is_sccc_ssm_location` reference).

**Behavioral cutover phase** (single atomic step):
9. Apply `v_current_job` view change + `dataset_version` / `engine_version` bump as one behavioral cutover. Before this step users still see the old market; after, they see SSM-only.

**Verification phase**:
10. Run all consumer regression tests (jobs API, pipeline snapshot, JD matching, adjacency, detail refetch, development plan, legacy chat) against the new symbol set and view.
11. Live-DB smoke: Wawa Community Support Worker absent from live match; `SELECT COUNT(*) FROM core.v_current_job WHERE normalized_job_location != 'Sault Ste. Marie' OR location_resolution_status != 'resolved'` returns 0.
12. Sign off Step 1A DoD.

**Follow-on** (separate PR / initiative):
13. Legacy column removal — one release cycle after Step 1A sign-off.
14. Step 1B kicks off.

**Correction 2026-07-16 (order rationale):**
- Consumer regression tests can't pass while `_location_boost` still exists (they assert the symbol is absent). Cleanup MUST precede tests.
- Producer migration + backfill can happen before the view change because the new columns are additive; the view change is what actually switches the market.
- View change + version bump are batched as a single atomic behavioral cutover so consumers can't see a half-migrated state (SSM-only rules with a stale engine_version, or an updated engine_version claiming SSM-only truth against a broader view).

**Ordering safety net:** the `ON CONFLICT CASE WHEN` guards in §2b mean an unmigrated producer re-ingesting a backfilled row cannot erase the backfilled values — the guard makes ordering step 4-before-5 belt-and-braces rather than load-bearing.

### 3e. Rollback

Each step reversible. Rollback plan:

- View migration reversible by restoring the pre-Step-1A `v_current_job` definition; new columns remain populated.
- `_location_boost` deletion reversible by restoring the function; behavioral gate becomes the view.
- `normalize_declared_job_location` addition is additive-only.
- `LOCAL_CITIES` rename requires updating callers in the same PR.

Version bump is one-way; rollback resets to `vN` explicitly.

---

## 4. Test plan

### 4a. Location normalization (unit)

**Exact SSM alias matches after cleanup:**
- `"Sault Ste. Marie"` → `("Sault Ste. Marie", False)`
- `"Sault Ste Marie"` → `("Sault Ste. Marie", False)`
- `"Sault Ste. Marie, ON"` → `("Sault Ste. Marie", False)` (suffix stripped, then exact match)
- `"Sault Ste. Marie, Ontario"` → `("Sault Ste. Marie", False)`
- `"SAULT STE. MARIE"` → `("Sault Ste. Marie", False)` (case-insensitive equality)
- `"SSM"` → `("Sault Ste. Marie", False)`
- `"Sault Ste. Marie (Remote)"` → `("Sault Ste. Marie", True)`
- `"Sault Ste. Marie, ON (Remote)"` → `("Sault Ste. Marie", True)` (Remote strip → suffix strip → exact match)
- `"  sault ste. marie  "` (whitespace) → `("Sault Ste. Marie", False)`

**Non-SSM cities (truthful normalization, NOT SSM):**
- `"Wawa"` → `("Wawa", False)`
- `"Wawa, Ontario"` → `("Wawa", False)`
- `"Elliot Lake"` → `("Elliot Lake", False)`

**Substring / word-boundary false-positive prevention** (exact-match rule is what saves these):
- `"Wawa / SSM"` → `("Wawa / Ssm", False)` — cleaned string is not an alias literal.
- `"North of SSM"` → `("North Of Ssm", False)` — cleaned string is not an alias literal.
- `"Rossmore, ON"` (contains "ssm" as substring of "rossmore") → `("Rossmore", False)`.
- `"SSM area"` → `("Ssm Area", False)`.

**Fuzzy / dropped-alias false-positive prevention:**
- `"Sault Ste. Maria"` (fuzzy) → `("Sault Ste. Maria", False)`
- `"Ste. Marie"` alone → `("Ste. Marie", False)`
- `"Sault"` alone → `("Sault", False)` (dropped alias)
- `"Sault Sainte Marie"` → `("Sault Sainte Marie", False)` (dropped alias — evidence-driven adds only)

**Bidirectional equivalence with `is_ssm_region_job` (location-only inputs, no region_code):**

```
normalize_declared_job_location(raw)[0] == "Sault Ste. Marie"
    ==
is_ssm_region_job({"location": raw})
```

Parametrized over the full fixture set above (SSM aliases, non-SSM cities, substring false-positives, fuzzy false-positives). Both sides return the same boolean for every input. This asserts they behave as one authority — no drift between canonicalization and runtime checks.

**Region-code precedence** (`is_ssm_region_job` only): when `job.region_code = "3557011"` (verified SSM CSD), `is_ssm_region_job` returns True regardless of what the location string looks like. That path is preserved and does NOT go through the new function.

### 4b. Ingest-side (per-connector unit)

**AWIC:**
- Valid `[lon, lat]` coordinates, excerpt present →
  `source_coordinates` = coord array, `source_location_text = NULL`, `normalized_job_location = NULL`, `location_resolution_status = "unresolved"`, `location_provenance = "geometry"`, `description_evidence_status = "excerpt_only"`.
- Valid `[lon, lat, alt]` coordinates (3-D GeoJSON) → same as `[lon, lat]`; array preserved with altitude.
- `[91.0, 0.0]` (latitude > 90) → `source_coordinates = NULL`, `location_resolution_status = "invalid"`, `location_provenance = "geometry"`.
- `["a", "b"]` (non-numeric) → `location_resolution_status = "invalid"`.
- Coordinates key absent → `location_resolution_status = "missing"`, `location_provenance = "none"`.
- Excerpt absent → `description_evidence_status = "missing"`.
- Regardless of input, `source_location_text` is never `"Sault Ste. Marie"`.

**SCCC:**
- Full WP content + `_job_location = "Sault Ste. Marie"` → `description_full` populated, `description_evidence_status = "full_source"`, `normalized_job_location = "Sault Ste. Marie"`, `location_resolution_status = "resolved"`, `location_provenance = "source_declared"`.
- Full WP content + `_job_location = "Sault Ste. Marie (Remote)"` → `remote_flag = True`, `source_location_text` preserves "(Remote)" verbatim, `normalized_job_location = "Sault Ste. Marie"`.
- WP content raises on parse → `description_evidence_status = "parse_error"`, `description_full = NULL`.
- Missing `_job_location` → `normalized_job_location = NULL`, `location_resolution_status = "missing"`.

**Partner CSV:**
- Description column populated → `description_full` set, `description_evidence_status = "full_source"`.
- Location column populated → `normalize_declared_job_location` applied; SSM aliases collapse; non-SSM cities preserved verbatim.

**Partner upload:**
- Same test shape as Partner CSV (uses `row.description` / `row.Job Description` / `row.location` / `row.Location`).
- Row without a description column → `description_evidence_status = "missing"`.
- Row without a location column → `location_resolution_status = "missing"`.

**Sault Area Hospital:**
- **No test case may hardcode "Sault Ste. Marie"** as `source_location_text` when the source page didn't declare it. Under Step 1A the connector emits `location_resolution_status = "missing"` for any posting whose page lacks a location field.
- Test asserts `source_location_text` is `None` OR matches a source-page-extracted string exactly — never the string "Sault Ste. Marie" as a fallback.
- Postings emitted with `missing` status must be excluded from `v_current_job` (verified in §4d consumer tests).

**City of SSM:**
- Same rules as Sault Area Hospital. The employer name does not imply the posting is in SSM; individual roles may be at outlying facilities.
- Test asserts no hardcoded location fallback under any input.

### 4c. Market eligibility (include/exclude)

- SCCC row with `normalized_job_location = "Sault Ste. Marie"` AND `location_resolution_status = "resolved"` → returned by `_fetch_eligible_jobs`.
- SCCC row with `normalized_job_location = "Wawa"` → NOT returned.
- AWIC row with `location_resolution_status = "unresolved"` → NOT returned.
- Row with `location_resolution_status = "conflicting"` → NOT returned.
- Row with legacy `location = "Sault Ste. Marie"` but `normalized_job_location = NULL` → NOT returned (view checks the new column, not legacy).
- `MatchResult.score_explanation` has no `location_boosted` entry.
- `MATCH_LOCATION_BOOST_LOCAL_CSD` symbol does not exist in `config.py`.

### 4d. Consumer regression (system-wide market change)

Each consumer of `v_current_job` gets a regression test asserting the SSM-only invariant:

- Public jobs API: `GET /v1/jobs` returns no rows where `normalized_job_location != "Sault Ste. Marie"`.
- Pipeline data status: `pipeline.v_data_status.jobs_current` count matches SSM-verified count.
- JD matching: no results with non-SSM location; boost signals absent from `score_explanation`.
- Adjacency: `is_ssm_region_job` still True for every candidate (was already true; now enforced twice).
- Detail refetch: hitting the detail endpoint for a non-SSM row returns 404 (not silently rendered).
- Development plan: recommendations quote SSM postings only.
- Legacy chat: (if any live paths remain) never quotes non-SSM rows.

### 4e. Backfill (idempotency and determinism)

- Backfill twice → identical row states.
- Latest raw payload wins: insert two synthetic `raw.job_posting` rows with different `ingested_at` for the same `(source, source_job_id)`; assert the newer wins.
- Historical AWIC rows: `description_evidence_status = "excerpt_only"` OR `"missing"` per row; `location_resolution_status` matches per-row rules; `normalized_job_location = NULL` for all.
- Historical SCCC rows: rows canonicalizing to SSM keep visibility; others exit the market.

### 4f. Live-DB smoke (post-backfill)

- `SELECT COUNT(*) FROM core.v_current_job WHERE normalized_job_location != 'Sault Ste. Marie' OR location_resolution_status != 'resolved'` returns **0**.
- `SELECT COUNT(*) FROM core.v_current_job WHERE source = 'awic_jobs'` returns **0** (until Step 1B).
- Community Support Worker (`job_id=2d1675bc-0c07-48d4-bf60-f16b4fcb8653`) absent from live match results.
- Live match session with target "community support" returns only rows where `normalized_job_location = "Sault Ste. Marie"`.

### 4g. Engine-side

- `_location_boost` function removed from `engine.py`; import errors surface at test time if anything still references it.
- `MATCH_LOCATION_BOOST_LOCAL_CSD` removed from `config.py`.

---

## 5. Residual decisions (locked)

### 5.1 SCCC ingestion stays permissive
Yes — Algoma-wide for archival. Rationale: `core.job_posting` may contain Algoma-wide rows; the eligibility gate at `v_current_job` provides SSM-only visibility without requiring re-scraping if scope ever re-expands.

### 5.2 Step 1B follows Step 1A
Yes — Step 1B (AWIC detail-page fetch) begins immediately after Step 1A ships. Step 1A honestly excludes AWIC until Step 1B lands. Step 1B spec written after Step 1A merges so it can be informed by live consumer breakage patterns and how many AWIC detail URLs remain live.

### 5.3 Location boost constant cleanup
**Removed in Step 1A**, not deferred. `_location_boost` deletion and `MATCH_LOCATION_BOOST_LOCAL_CSD` removal ship together. Rationale: leaving the constant after removing its only valid consumer invites accidental reintroduction and contradicts §7 principle 2.

### 5.4 `LOCAL_CITIES` rename
**In-scope for Step 1A.** Rename to `SCCC_INGEST_LOCALITIES` (config, `.env`, `ingest/partners.py:584` reference). The rename removes the possibility of the variable being misread as a matching-market allowlist. Bare deletion is not safe while SCCC ingestion still filters on it.

---

## 6. Definition of done

- [ ] Schema migration merged; 8 new columns exist on `core.job_posting`.
- [ ] `skillbridge.match.region` extended with `normalize_declared_job_location`; alias/code registry is the single shared authority.
- [ ] AWIC connector never emits `source_location_text = "Sault Ste. Marie"`; verified by unit test.
- [ ] Sault Area Hospital connector never emits `source_location_text = "Sault Ste. Marie"` as a fallback; verified by unit test.
- [ ] City of SSM connector never emits `source_location_text = "Sault Ste. Marie"` as a fallback; verified by unit test.
- [ ] Partner upload connector applies `normalize_declared_job_location` to CSV row `location` / `Location` columns; verified by unit test.
- [ ] `upsert_job` transitional persistence: 7 unit tests covering unmigrated-preserves-legacy, migrated-missing-clears, blank-full-falls-through, blank-both-persist-null, SQL CASE WHEN guards on new-axis columns, and legacy-columns-still-overwrite semantics.
- [ ] AWIC per-row rules confirmed by test: valid coords → `"unresolved"`; missing → `"missing"`; malformed → `"invalid"`.
- [ ] SCCC connector populates `description_full` from `content.rendered`, `source_location_text` from `_job_location`, applies `normalize_declared_job_location`, and produces `parse_error` when HTML clean fails.
- [ ] Coordinate validation matches §2d (length ≥ 2, lon ∈ [-180, 180], lat ∈ [-90, 90], extras preserved).
- [ ] Backfill script uses `DISTINCT ON (source, source_job_id) ORDER BY ingested_at DESC, raw_id DESC`; runs idempotently.
- [ ] `v_current_job` migration adds SSM eligibility clauses; `dataset_version` / `engine_version` bumped in the API envelope.
- [ ] Per-consumer regression tests (jobs API, pipeline status, JD matching, adjacency, detail refetch, development plan, legacy chat) pass with SSM-only invariant.
- [ ] `_location_boost` function deleted from `engine.py`; `MATCH_LOCATION_BOOST_LOCAL_CSD` removed from `config.py`.
- [ ] `LOCAL_CITIES` renamed to `SCCC_INGEST_LOCALITIES` throughout (config, `.env`, ingest partners reference).
- [ ] Existing `is_ssm_region_job` tests updated to reference the shared canonicalization module (prevents drift).
- [ ] `SELECT COUNT(*) FROM core.v_current_job WHERE normalized_job_location != 'Sault Ste. Marie' OR location_resolution_status != 'resolved'` returns **0**.
- [ ] Community Support Worker (`job_id=2d1675bc-0c07-48d4-bf60-f16b4fcb8653`) absent from live match results (not merely lower-scored).
- [ ] `description_evidence_status` and `location_resolution_status` distributions logged for first week.

Steps 2-12 assume the above. Step 1B is required for AWIC re-entry but does not block Steps 2-12.

---

## 7. Load-bearing principles (final)

1. **Only postings with a verified, normalized Sault Ste. Marie location enter the current matching engine.**
2. **Location does not increase qualification fit.** Eligibility, not scoring.
3. **Preserve what the source actually said.** No fabrication of location, no truncation of description.
4. **One shared SSM authority with exact-match semantics.** `is_ssm_region_job` and `normalize_declared_job_location` share the alias/code registry AND the exact-match rule; the pre-Step-1A word-boundary substring path is retired; no drift between runtime checks and normalization.
5. **Distinguish declared location from coordinates.** Coordinates are a signal, not an answer.
6. **Backfill is deterministic and idempotent.** Latest raw payload wins per `(source, source_job_id)`.
7. **Unknown data receives no eligibility and no confident fit claim.** `location_resolution_status != "resolved"` → out of market.

Every design decision above derives from these seven rules.
