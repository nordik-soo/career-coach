# Matching Eval Corpus — Schema Spec

Status: schema spec · 2026-07-01 · M1 prerequisite for the calibration harness (M2)

This document defines the frozen gold-label corpus that the matching evaluation suite
consumes. The corpus is a versioned YAML file at `data/matching_eval_corpus.yaml`, loaded
by `tests/test_matching_eval.py` (successor to `test_matching_fixtures.py`). It replaces
live-DB-pinned fixtures with a fully self-contained truth table.

Design goals, in priority order:

1. **Zero skips, ever.** Every case runs on every CI invocation with `LLM_ENABLED=false` and
   no database. Data drift cannot erode coverage because there is no external data.
   `pytest.skip` is structurally forbidden in the eval module (enforced by CI, see
   §Validation).
2. **Same enforcement split as the training registry.** Loader does structural validation;
   CI does policy validation; humans do semantic review of expectations. See
   `docs/training-registry-schema.md` for the pattern this copies.
3. **Dual consumer.** The eval suite (pass/fail per case) and the M2 calibration harness
   (per-stage attribution, threshold sweeps) read the same file. One corpus, two lenses.

## Top-level structure

```yaml
# data/matching_eval_corpus.yaml
corpus_version: 1                             # bump on breaking schema changes only
corpus_verified_at: "2026-07-01"
engine_version_pinned: "job-match-v1.2.0"     # expectations were authored against
posting_bank:
  - <posting entry>                           # shared frozen postings, referenced by id
cases:
  - <case entry>                              # one scenario = one profile vs. the bank
```

`engine_version_pinned` is informational, not a gate. When an intentional engine change
alters expected outputs, update the affected expectations and this field **in the same
commit**, with a BREAKING.md-style note in the commit message explaining which cases
changed and why. An eval-suite diff with no expectation-change rationale is a regression,
not a recalibration.

## Posting bank

Frozen synthetic postings. Each is the minimal projection the engine scores against —
mirror the columns `_fetch_eligible_jobs` / `_fetch_job_skills` produce, not the raw ingest
shape.

```yaml
posting_bank:
  - posting_id: "pb_truck_coach_grfn"         # stable slug, never reused
    title: "Truck and Coach Technician"
    employer: "Garden River First Nation"
    noc_code: "72410"
    location: "Garden River, ON"
    region_code: "SSM"
    employment_type: "full-time"
    posted_days_ago: 5                        # relative — resolved against a frozen "today"
    description_snippet: >-
      Optional. Only needed when the case exercises description-dependent
      logic (shift detection, work-type inference).
    skills:
      - name: "truck service and maintenance"
        requirement: "required"               # required | preferred
        is_credential: false
      - name: "310T certificate of qualification"
        requirement: "required"
        is_credential: true
      - name: "class g licence"
        requirement: "required"
        is_credential: true
      - name: "welding"
        requirement: "preferred"
        is_credential: false
    embedding_profile: "default"              # see §Semantic determinism
```

Field notes:

| Field | Required | Purpose |
|---|---|---|
| `posting_id` | yes | Referenced by cases. Once published in a tagged corpus version, never delete or repurpose — supersede with a new id and mark the old one `retired: true` so historical calibration reports stay comparable |
| `transcribed_from_sccc` | yes | Boolean. `true` means the posting was copied verbatim from a real live SCCC row (title / employer / NOC / skills / requirement labels / credential flags) at the timestamp in the corpus header, then frozen. `false` means the posting was NOT copied from live SCCC — see the semantic-overload note below |
| `posted_days_ago` | yes | The suite resolves dates against a frozen anchor date (loader constant), so recency-boost behaviour is deterministic and never ages out |
| `skills[].requirement` | yes | The corpus asserts required/preferred labels directly. This is the JD-extractor's *output* contract, deliberately: the eval measures the engine, not the extractor. Extractor quality gets its own corpus later — do not conflate them (this is the lesson from the removed F6 fixture) |
| `skills[].is_credential` | yes | Explicit, not inferred via `is_credential_skill_name` at load. A CI check asserts the flag agrees with `is_credential_skill_name` and fails loudly on disagreement — that disagreement is a real bug in one of the two places |
| `embedding_profile` | no | Default `"default"`. See §Semantic determinism |

**Semantic overload on `transcribed_from_sccc: false`.** The current single boolean
conflates two distinct posting states:

- **Reconstruction pending:** the posting represents a real SCCC row that has NOT yet
  been copied verbatim into the corpus. The header (`data/matching_eval_corpus.yaml`
  top-of-file `POSTING TRANSCRIPTION` note) explicitly names which postings are in
  this state. These must be transcribed before the corpus gates CI.
- **Synthetic by design:** the posting is a fully-constructed test artifact for a
  specific classifier scenario (e.g. a part-time admin posting to exercise the
  work-type cap). Inline comments in the YAML mark these with "Synthetic — no
  transcription needed." These must NEVER be transcribed; doing so would remove the
  scenario from the corpus.

The `transcribed_from_sccc: false` flag currently applies to BOTH states, and which
state a specific posting is in is communicated only via YAML comments. A future
schema bump may introduce an `is_synthetic: bool` field to disambiguate. Deferred
to keep this correction pass scoped to schema language only. Until then, treat the
YAML header's `POSTING TRANSCRIPTION` list as the authoritative source for which
postings still owe SCCC transcription work.

## Case entries

One case = one staged profile scored against the full posting bank, with expectations at
three levels: **turn-level diagnosis**, **per-job outcomes**, and **per-skill match provenance**.

```yaml
cases:
  - case_id: "c_truck_coach_no_class_g"
    description: >-
      Truck & coach apprentice without an explicit Class G claim.
      Credential cap MUST fire regardless of skill overlap; the cap
      reason is the actionable signal, not the band.
    categories: ["credential_gap", "cap_semantics"]

    profile:
      target_role: "truck and coach technician apprentice"
      skill_phrases:
        - "welding"
        - "truck maintenance"
        - "vehicle inspection"
        - "parts fabrication"
        - "diesel repair"
      experience_text: "Apprentice Truck & Coach Technician at Northern Fleet Services"
      education_text: "Truck & Coach Technician Apprenticeship — Sault College"
      work_type_preference: "full-time"

    expect:
      diagnosis: "PREPARATION_GAP"

      jobs:
        - posting_id: "pb_truck_coach_grfn"
          band: "stretch"                     # strong | good | stretch | explore | none
          cap_reasons: ["band_capped_by_credential"]
          cap_reasons_forbidden: []           # explicit absence assertions, see below
          matched_required:
            - requirement: "truck service and maintenance"
              via_stage: "fuzzy"              # exact | fuzzy | semantic
              user_skill: "truck maintenance"
          missing_required_contains:
            - "class g licence"
          blocking_credential: "class g licence"

      jobs_absent: []                         # posting_ids that must NOT surface at all
```

### Case-level fields

| Field | Required | Purpose |
|---|---|---|
| `case_id` | yes | Unique. Stable across corpus versions — calibration reports diff by it |
| `categories` | yes | 1–3 tags from the closed set in §Coverage. Drives the CI coverage floor |
| `profile` | yes | Exactly the `StagedProfile` construction surface the current fixtures use (`target_role`, `skill_phrases`, `experience_text`, `education_text`, `work_type_preference`). No new profile fields without a schema version bump |
| `expect.diagnosis` | yes | One of the six `inventory_diagnosis` outcomes. Every case asserts this even when the interesting behaviour is per-job — diagnosis regressions are the cheapest to catch and the most user-visible |
| `expect.jobs` | no | Per-job assertions. Omit for pure-diagnosis cases (e.g. `UNDETERMINED` with thin evidence) |
| `expect.jobs_absent` | no | Postings that must not appear in results at all (out-of-family leakage, work-type exclusion). This is the assertion shape that would have caught the 14404/13110 off-target leak on the recommender side |

### Per-job expectation fields

| Field | Required | Purpose |
|---|---|---|
| `band` | yes | Closed enum (tier `MatchLabel`, see §Vocabularies). `none` means the job surfaces in no tier |
| `band_at_least` | no | Boolean modifier on `band`. When `true`, `band` is interpreted as a MINIMUM — the case passes when the actual `MatchLabel` is at that band OR any strictly stronger band (ordered `strong > good > stretch > explore_later > none`). Default `false` (exact-match). Used for happy-path cases where either `good` or `strong` is legitimately correct because both map to the same `apply_today` tier slot under v6 |
| `cap_reasons` | yes (may be `[]`) | Exact set semantics: every listed reason must be present in `score_explanation.caps_applied`. Closed vocabulary, §Vocabularies |
| `cap_reasons_forbidden` | no | Reasons that must NOT fire. This encodes the F2 lesson: "holding Class G removes the cap, band may legitimately stay stretch." Asserting absence is a first-class expectation, not a comment |
| `matched_required` | no | List of `{requirement, via_stage, user_skill}` triples asserted against `build_skill_alignment` output. `via_stage` uses the closed stage vocabulary. This is what makes stage provenance a *pinned contract* before M3 surfaces it to users — if a skill silently slips from `exact` to `semantic`, the corpus catches it even though the band didn't move |
| `missing_required_contains` | no | Substring-tolerant (engine phrasing drifts with alias curation); everything else in this schema is exact-match. **Assertion shape does NOT count-partition** — it only checks presence, not credential-vs-learnable-gap counts (see §MatchLabel derivation below) |
| `blocking_credential` | no | Asserts `_has_critical_credential_gap` identified this specific credential |

### MatchLabel derivation (`tiered_evidence.py:567-604`, scoring-v6)

The tier `MatchLabel` a job receives is derived from **three signals**, not one:

- **Score band** — result of `_band(score)`: `strong / good / stretch / low`
- **Blocker count** — number of credential gaps in `required_missing`
  (`is_credential_skill_name == True`)
- **Learnable count** — number of non-credential gaps in `required_missing`

Decision flow (top-down, first matching rule wins):

```
0. Filtered → None:
   - match_eligible is False
   - match_score < 0.30 (visibility floor)
   - score_explanation.required_missing absent
   - credential-only gap profile AND no actionable training

1. band == "low" (0.30 ≤ score < 0.40)          → explore_later
2. learnable_count >= 5                          → explore_later
3. blocker_count >= 2                            → explore_later
4. band == "stretch"                             → stretch
5. band in {strong, good} AND learnable in 3..4  → stretch
6. band in {strong, good} AND blocker_count == 1 → stretch
7. band == "strong" (no blocker, ≤2 learnable)   → strong
8. band == "good"   (no blocker, ≤2 learnable)   → good
```

**Corpus implication:** a case pinned to `band: "good"` implicitly asserts that the
profile produces `blocker_count == 0` AND `learnable_count ≤ 2` AND raw score band
is `good`. If the classifier moves the label because the profile actually has 3+
learnable gaps, the corpus catches a `band` disagreement but currently has no way to
independently pin the **why** (which of the three signals differed).

**Known assertion gap:** the current schema does NOT expose `expected_blocker_count` /
`expected_learnable_count` per-job fields. Adding them would let cases pin classifier
rules directly (e.g. "this profile has 4 learnable gaps → rule 5 → stretch"). Deferred
to a future schema bump; for now, cases assert only the final `band` and rely on the
`description` block to name the classifier rule under test.

## Vocabularies (closed, copied from engine — do not invent)

```yaml
band:                [strong, good, stretch, explore_later, none]
via_stage:           [exact, fuzzy, semantic]                    # no_match is not assertable as a stage
cap_reasons:         [band_capped_by_credential,
                      band_capped_by_no_experience,
                      band_capped_by_work_type_mismatch]
diagnosis:           [UNDETERMINED, MARKET_DATA_UNAVAILABLE, READY_TO_APPLY,
                      PREPARATION_GAP, SKILL_ADJACENT_AVAILABLE, NO_OPPORTUNITY_FOUND]
categories:          [credential_gap, cap_semantics, no_match, negative_control,
                      semantic_bridge, fuzzy_boundary, adjacent_only, thin_evidence,
                      work_type, direct_title, family_gate, ready_to_apply]
```

**Rule:** when the engine grows a new cap reason or diagnosis outcome, the vocabulary here is
updated in the same PR, with at least one new case exercising it. An engine enum value
with zero corpus coverage fails CI (§Validation, policy layer).

**Note on `band`:** the corpus's `band` field asserts against the tier `MatchLabel`
(`tiered_evidence.py:564`, produced by `_classify_match_label`), NOT the raw score band
from `_band()` (`engine.py:730-737`). Two distinct layers exist in code:

- **Score band** (`_band(score)` returns `strong / good / stretch / low`): the raw score
  cutoff, driven by `MATCH.band_strong / band_good / band_stretch` thresholds
  (`config.py:438-440`, currently 0.75 / 0.60 / 0.40)
- **Tier MatchLabel** (`strong / good / stretch / explore_later`): the user-facing tier
  outcome, derived from three signals (see the learnable-gap note under
  §Per-job expectation fields). The corpus asserts the tier label because that is what
  users actually see. A raw-band `low` score maps to the tier label `explore_later` via
  rule 1 of `_classify_match_label`. `none` in the corpus vocabulary means the case's
  `_classify_match_label` returned `None` — the job was filtered before tier assignment
  (fail-closed on missing `required_missing`, sub-visibility-floor score, or
  "actionable nothing" credential-only-gap-with-no-training).

### Tier semantics (v6, `tiered_evidence.py:186-213`)

Under v6 (scoring-v6, locked 2026-06-17), the user-facing `TieredEvidence` bundle has
three direct-target tier slots plus one adjacent-role slot, and multiple `MatchLabel`
values map to a single direct-target slot:

```
apply_today     ← jobs with MatchLabel in {strong, good}    (shared slot)
worth_a_try     ← jobs with MatchLabel == stretch
explore_later   ← jobs with MatchLabel == explore_later
sideways_move   ← adjacent-NOC jobs (unchanged; not classifier-controlled)
```

The shared `apply_today` slot is why a case can legitimately pin `band: "good"` with
`band_at_least: true` (see next section) — either a `good` or a `strong` MatchLabel
satisfies "surfaces under apply_today," and the case shouldn't fail if the engine picked
one over the other for a borderline profile.

**Note on `via_stage`:** the internal cascade has more rungs (skill_id → name exact → canonical
→ substring → fuzzy → semantic), but `build_skill_alignment` exposes the three-value
stage (`alignment.py:94`). The corpus pins the exposed contract. If M3 promotes finer rungs
into the alignment output, that is a corpus schema bump with a migration note.

## Semantic determinism

Semantic-stage cases (`semantic_bridge`, `fuzzy_boundary` categories) must be reproducible
without a live embedding service. The corpus supports frozen pairwise similarities:

```yaml
embedding_fixtures:
  - a: "client intake coordination"
    b: "customer onboarding"
    cosine: 0.74
```

The eval harness monkeypatches `_semantic_match_strength`'s similarity lookup with this
table; any pair not listed resolves to 0.0. This makes threshold-boundary cases exact: a case
can pin "cosine 0.74 → matches at threshold 0.70, capped at strength 0.75" and a sibling case
pins "cosine 0.68 → no_match." These same fixtures are the seed inputs for M2's threshold-
sensitivity sweep.

Real-embedding behaviour is measured in M2 against a *snapshot* of live data — it does not
belong in the pass/fail corpus, because a model or index change would reintroduce exactly
the drift-skip failure mode this corpus exists to kill.

## Coverage floor (CI-enforced)

Minimum case counts per category, asserted by a structural test:

| Category | Min | Rationale |
|---|---|---|
| `credential_gap` | 6 | The highest-stakes failure class: false "you qualify" on a regulated trade. Include cap-fires and cap-must-not-fire pairs (F1/F2 pattern) |
| `negative_control` | 4 | Software developer vs. SSM trades bank, airline pilot, etc. Every one asserts `jobs_absent` for the whole bank plus a non-READY diagnosis |
| `no_match` / `thin_evidence` | 6 | At least one case per relevant diagnosis outcome; `UNDETERMINED` and `MARKET_DATA_UNAVAILABLE` need constructed inputs (empty evidence; engine-failure injection) |
| `semantic_bridge` + `fuzzy_boundary` | 8 | Both sides of both thresholds. These are the cases M6's alias-promotion loop will gradually convert from `via_stage: semantic` to `via_stage: exact` — the corpus diff is the progress metric |
| `cap_semantics` / `work_type` / `family_gate` | 6 | Gates and demotions with `cap_reasons_forbidden` counterparts |
| `ready_to_apply` / `direct_title` / `adjacent_only` | 6 | The happy paths, so a regression can't hide behind "all the failure cases still fail correctly" |

**Total floor: 36 cases. Target for v1: 45–55.**

**Known gap: `explore_later` tier has no dedicated coverage.** Under v6
(`tiered_evidence.py:567-604`) `explore_later` is a first-class user-facing tier with
its own classifier rules (`band == "low"`; `learnable_count >= 5`;
`blocker_count >= 2`). None of the current categories exercises those rules
specifically — cases exercising them fall under `cap_semantics` or `thin_evidence`
today. Adding an `explore_later` category with its own minimum count is deferred to
the coverage-fill slice (Step 5), not this schema-correction pass, because it changes
the floor from 36 to 39+ and introduces a new category policy before real cases
exist. Contributors filling coverage in Step 5 should decide whether to add the
category or subsume `explore_later` cases under the existing ones.

## Validation — the three-layer split

**Loader (structural, at parse):** unique ids; every `expect.jobs[].posting_id` resolves into
the bank; all enum fields within the closed vocabularies; `cap_reasons` ∩
`cap_reasons_forbidden` empty; every case has ≥1 category; `retired: true` postings
referenced by no active case. Loader failures raise — a malformed corpus is a build break,
not a warning.

**CI (policy):** no `pytest.skip` / `skipif` anywhere in the eval module (grep-based structural
test, same spirit as the arbiter's Pass-1/Pass-2 exhaustive check); coverage floor met;
`is_credential` flags agree with `is_credential_skill_name`; every engine cap-reason and
diagnosis enum value has ≥1 exercising case; corpus loads and full suite passes with
`LLM_ENABLED=false` and `SESSION_STORE=cookie`, no DB, no network.

**Human review (semantic):** are the expectations *right*? A wrong expected band that the
engine reproduces is a green suite asserting a bug. Review triggers: any PR touching
`expect:` blocks; any `engine_version_pinned` bump. Reviewer confirms the rationale in the
case `description` still justifies the expectation — every case must carry a description that
states *why* the expectation holds, F1/F2 style, not just what it is.

## What this corpus deliberately does not cover

- **Extractor quality** (resume → skills, JD → labeled skills). Profiles and postings enter
  post-extraction. Separate corpus, later.
- **Responder narration.** That's M5's policy-gate territory; this file pins engine +
  diagnosis outputs only.
- **Live-data realism.** M2's calibration snapshot handles that. The corpus trades realism
  for permanence — that's the point.

## Migration from test_matching_fixtures.py

The five current fixtures map directly: F1 → `c_truck_coach_no_class_g`, F2 → its
`cap_reasons_forbidden` sibling, and the `depends_on_data` fixtures get their pinned postings
transcribed into the bank from the current SCCC rows (one-time copy — after that, the
bank is authoritative and immutable). Keep `test_matching_fixtures.py` running until the
new suite hits the coverage floor, then delete it in the same PR that flips CI to the corpus. No
period where both are optional.
