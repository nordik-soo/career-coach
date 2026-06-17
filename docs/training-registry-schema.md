# Training Registry — Schema Spec

Status: schema spec · 2026-06-04 · prerequisite for the YAML seed

This document defines the structure of the training registry that the
recommender will consume. The registry is a versioned YAML file at
`data/training_registry.yaml`. Loaded once at server startup into an
in-memory singleton; no DB calls in the v1 path.

## Top-level structure

The registry is a YAML list of **gap entries**. Each gap entry maps a
canonical credential or skill name to one or more training resources.

```yaml
# data/training_registry.yaml
version: 1                # bump on breaking schema changes only
registry_verified_at: "2026-06-04"   # when this file was last reviewed end-to-end
gaps:
  - canonical_name: "310T technician certification"
    aliases:
      - "310T"
      - "310T certificate"
      - "310T certificate of qualification"
      - "truck and coach technician certificate"
    category: "credential"
    description: >-
      Ontario compulsory trade certification for truck and coach technicians.
      Required by most heavy-vehicle service employers in the province.
    resources:
      - provider: "Skilled Trades Ontario"
        type: "credential_pathway"
        url: "https://www.skilledtradesontario.ca/..."
        summary: >-
          The provincial regulator's pathway for completing the 310T trade
          certification, including apprenticeship registration and exam
          requirements.
        verified_at: "2026-06-04"
        verified_by: "lead-engineer"
      - provider: "Sault College"
        type: "apprenticeship"
        url: "https://www.saultcollege.ca/..."
        summary: >-
          Local in-class instruction component of the 310T apprenticeship.
        verified_at: "2026-06-04"
        verified_by: "lead-engineer"
      - provider: "Sault Community Career Centre"
        type: "referral_only"
        url: null
        summary: >-
          Local employment-and-training counsellors who can map your current
          experience against 310T apprenticeship requirements.
        verified_at: "2026-06-04"
        verified_by: "lead-engineer"
```

## Field reference

### Gap-level fields

| Field | Required | Type | Purpose |
|---|---|---|---|
| `canonical_name` | yes | str | The human-readable form of the gap. The recommender displays this to the user; the registry keys against it for lookup. Must be unique across the registry |
| `aliases` | yes | list[str] | Alternate phrasings the match engine may produce in `credential_gap_skills` / `missing_skills`. The recommender normalizes both sides and matches by intersection. Always include the canonical name's common short form ("310T") and the engine's verbatim form ("310T certificate of qualification" — copy from real match output) |
| `category` | yes | enum | One of: `credential`, `skill`, `license`, `safety_training`. Drives narration shape — credentials are "pathways," skills are "training," etc. |
| `description` | yes | str | 1-2 sentences. What this is, in plain language. The responder LLM uses this when explaining the gap |
| `resources` | yes | list | 1-4 resource entries (see below). Ordered by recommendation priority — first entry surfaces first in chat |

### Resource-level fields

| Field | Required | Type | Purpose |
|---|---|---|---|
| `provider` | yes | str | Must match an entry on the [training-providers-allowlist](training-providers-allowlist.md). Free-text otherwise but reviewers reject |
| `type` | yes | enum | One of: `credential_pathway` (authoritative regulator/issuer), `local_training` (program/course at a local provider), `apprenticeship` (workplace + classroom), `online_course` (MOOC or vendor cert page), `referral_only` (no URL — counsellor referral) |
| `url` | conditional | str \| null | Required for all `type` values EXCEPT `referral_only`, which must have `url: null`. URLs must point to a structured, durable path (not a marketing landing page) |
| `summary` | yes | str | 1-2 sentences explaining what this specific resource is. Used by the responder LLM to narrate; NOT a sales pitch |
| `verified_at` | yes | date (YYYY-MM-DD) \| null | When this URL was last checked. **`null` means "pending verification"** and is a legal value at load time; the loader treats it the same as an expired entry — URL suppressed at runtime, provider name + generic guidance surfaced instead. Stale dated entries (> 6 months old) get the same treatment |
| `verified_by` | yes | str | GitHub username / PR URL / human name. So future contributors know who last vouched |

### `type` taxonomy

The five resource types each map to a distinct narration shape. The
responder prompt (or fallback) is allowed to phrase each differently.

| Type | Meaning | Sample narration |
|---|---|---|
| `credential_pathway` | The authoritative issuer of the credential | "Skilled Trades Ontario runs the official pathway for 310T." |
| `local_training` | A course/program offered locally | "Sault College offers a continuing-ed track that covers this." |
| `apprenticeship` | Workplace + classroom hybrid | "Sault College runs the apprenticeship's classroom component locally." |
| `online_course` | MOOC or vendor official course/cert page | "Microsoft Learn has a free official Excel certification path." |
| `referral_only` | No URL; point at a counsellor instead | "SCCC counsellors can help you map this." |

`referral_only` is the safety valve when no authoritative URL exists for a
gap. SCCC is the typical provider for `referral_only` entries because they're
the local single-point-of-contact for employment guidance.

## What the recommender consumes

### Lookup algorithm (high level)

Given a gap string from `credential_gap_skills` or `missing_skills`:

1. Normalize: lowercase, strip punctuation, collapse whitespace
2. Match against each registry entry's `canonical_name` ∪ `aliases`
   (normalized the same way)
3. First match wins — registry order is the tie-breaker
4. Filter each matched gap's `resources` by freshness:
   - `verified_at` within 6 months: surface the URL
   - Older: surface the provider name, suppress the URL, append guidance
   - `referral_only` entries are always surfaced (no URL anyway)
5. Cap at 3 resources per gap in the responder TRAINING block

### What goes into the TRAINING block

The responder gets a TRAINING block formatted to match the existing shape
(handler's `_attach_training` output). Each entry in the block is one
training resource:

```json
{
  "provider": "Skilled Trades Ontario",
  "title": "310T trade certification pathway",
  "url": "https://www.skilledtradesontario.ca/...",
  "type": "credential_pathway",
  "for_gap": "310T technician certification",
  "summary": "The provincial regulator's pathway for ..."
}
```

This is a slight change from today's shape (which uses `for_skill`); the
registry adds `for_gap`, `summary`, and `type`. The handler glue code
translates registry resource records to this shape.

### Unknown gaps (telemetry)

When a gap from the engine has NO match in the registry:

- Recommender returns an empty resource list for that gap
- Logs INFO line: `training_registry session=... unknown_gap=<gap_name>`
- The responder still narrates the gap (using `ConversationContext` for
  context) but cannot cite a URL — falls back to generic SCCC guidance

The telemetry log is how the registry grows from real usage rather than
guesses. Reviewers periodically grep the log for `unknown_gap=` patterns
and PR new entries.

## Validation rules (enforced at load time)

The registry loader rejects the file at server startup if any of these fail:

| Rule | Failure mode |
|---|---|
| Top-level `version` field is `1` | Hard error — schema migration needed |
| Every gap has a `canonical_name` and at least one alias | Hard error |
| `canonical_name` is unique across the registry | Hard error |
| Every resource has all required fields | Hard error |
| `type` is one of the 5 enum values | Hard error |
| If `type != "referral_only"`, `url` is non-null and starts with `https://` | Hard error |
| If `type == "referral_only"`, `url` is null | Hard error |
| `verified_at` is either `null` (pending) OR a YYYY-MM-DD date string | Hard error |
| `verified_by` is either `null` (pending) OR a non-empty string | Hard error |

**Note on provider names**: the loader does NOT cross-check `provider`
values against the allowlist doc. Provider-name normalization across
multiple forms ("CCOHS" vs "Canadian Centre for Occupational Health and
Safety") would be brittle as a load-time check, and provider sign-off is
a human-review concern anyway. A separate test
(`tests/test_training_registry_allowlist.py`) does that check at CI time
with explicit normalization rules — that's the right place for a "soft"
policy assertion. The loader's job is structural validation only.

Loader validation runs once at startup. A bad registry aborts startup with
a clear error rather than silently corrupting recommendations.

## What the schema deliberately does NOT include

These were considered and rejected for v1:

| Field | Why excluded |
|---|---|
| `cost` | URLs change pricing more often than we'd want to re-verify. Surface generic phrasing in the prompt ("contact provider for current cost") instead |
| `duration_band` | Existing DB schema has it; the responder doesn't need it for the narration shape we're building. Add back if the prompt benefits from it |
| `prerequisites` | Encoding prereqs reliably is hard; surface in the `summary` text instead |
| `language` | All SSM resources we vouch for offer English at minimum; non-English availability is a `summary` note when known |
| `enrollment_window` | Too volatile. Direct users to the provider's enrollment page via the URL |
| `success_rate` / ROI metrics | We don't have authoritative data; would be inventing |
| `instructor` | Same — and irrelevant for organizational providers |

Future work can add any of these without a schema migration if a real
product need surfaces.

## Future migration (out of scope)

When the registry grows past what's manageable in a single YAML file:

1. Migrate to per-gap files: `data/training/310t.yaml`, `data/training/class_g.yaml`, etc.
2. Add a build step that consolidates them at load time
3. Move to DB only after an admin UI exists to maintain entries safely

None of this is needed for v1. A single YAML file with 13-20 gap entries is
~300 lines of readable, reviewable text.
