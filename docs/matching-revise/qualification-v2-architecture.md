# Qualification V2 Architecture — Locked Contracts

**Status:** LOCKED (2026-07-16). Amendments approved through reviewer sign-off across four review rounds. This document is the architectural spine for Steps 3–8 of the matching-revise plan. Every implementation in those steps must conform to the eight contracts below.

**Not covered here:** Steps 1A (source integrity), 2 (title-to-fit removal), 9 (two-stage retrieval), 10 (taxonomy), 11 (labelled eval set — schema locked here, corpus itself belongs to Step 11), 12 (calibration), 13+ (learned rerankers). Those are their own documents.

---

## The Locked Rule (Invariant)

Every downstream decision derives from this five-line invariant:

1. **Title and NOC retrieve relevant jobs.** They never contribute to qualification fit.
2. **Eligibility checks mandatory gates.** Missing licences, work authorization, mandatory transportation — pass / fail / unknown, not averaged.
3. **Skills, education, and experience measure qualification.** Three dimensions. Nothing else.
4. **Coverage reports how much evidence supported that measurement.** Rendered alongside fit; fit never surfaced without it.
5. **Preferences influence ranking only.** Shift, work type, salary, distance — never lower qualification fit.

Corollary: **unknown evidence never becomes a failed requirement.** Absence must be genuinely established (not merely unspecified) before `not_met` may be reported.

---

## Contract 1 — Requirement Effect Classification

One `job_requirement` table with a classification field:

```
requirement_effect ∈ { qualification, eligibility, preference }
```

Examples:

| Requirement | Effect |
|---|---|
| Python | qualification.skill |
| Bachelor's degree | qualification.education |
| 3 years of casework | qualification.experience |
| Registered Nurse licence | eligibility.credential |
| Valid driver's licence | eligibility.transportation |
| Security clearance | eligibility.clearance |
| Evening shift | preference.schedule |
| Hybrid work | preference.work_type |
| Salary expectation | preference.compensation |

**Behavior:**
- Qualification → contributes to fit (via one of three comparators)
- Eligibility → pass / fail / unknown gate; never averaged into fit
- Preference → ranking signal only; never lowers fit

A shift mismatch must never lower qualification fit. A mandatory licence must never be diluted inside an average score.

**Schema enforcement:** CHECK constraint on the enum + comparator dispatch keyed strictly on `requirement_effect`. Comparators refuse requirements whose effect does not match their dimension.

---

## Contract 2 — Fit Combination Formula

Each of the three qualification dimensions returns:

```json
{
  "score": 0.78,
  "status": "evaluated",
  "coverage": 0.90,
  "weight": 0.60
}
```

**Status enum (exactly these five values):**
- `evaluated` — real comparison performed
- `unknown_job_evidence` — JD didn't provide the required requirement or the extractor's confidence was below threshold
- `unknown_candidate_evidence` — candidate side had no reliable data
- `not_applicable` — this JD did not include any requirement of this dimension
- `comparator_error` — comparator raised or returned malformed output; dimension is fail-soft excluded

**Combine only evaluated dimensions.** Unknown never becomes zero.

Calculation is a two-step:

**Step A — effective evidence weight per dimension:**
```
effective_weight_i = dimension_weight_i × dimension_coverage_i
```
(A dimension with `status ≠ evaluated` contributes `dimension_coverage = 0`.)

**Step B — the two top-level outputs:**
```
evidence_coverage = Σ effective_weight_i  /  Σ dimension_weight_i

qualification_fit = Σ (score_i × effective_weight_i)  /  Σ effective_weight_i
                    (over evaluated dimensions only)
```

**Worked example:**

| Dimension | score | status | coverage | weight | effective_weight |
|---|---|---|---|---|---|
| Skills | 0.80 | evaluated | 0.90 | 0.60 | 0.54 |
| Education | — | unknown_job_evidence | 0.00 | 0.15 | 0.00 |
| Experience | 0.60 | evaluated | 1.00 | 0.25 | 0.25 |

```
evidence_coverage = (0.54 + 0.00 + 0.25) / 1.00 = 0.79
qualification_fit = (0.80 × 0.54 + 0.60 × 0.25) / 0.79 = 0.737
```

Output:
```json
{ "qualification_fit": 0.737, "evidence_coverage": 0.79 }
```

**No `dimension_coverage` top-level field.** Dimension availability is already represented by each dimension's `status`. Two coverage numbers under the same name would silently corrupt calibration math.

**Weights themselves are calibrated in Step 12 from the eval set — not chosen now.**

---

## Contract 3 — Coverage Behavior

Never return a band without coverage.

**Normal case:**
```json
{
  "qualification_fit": 0.74,
  "evidence_coverage": 0.85,
  "band": "good"
}
```

**Insufficient evidence:**
```json
{
  "qualification_fit": 0.90,
  "evidence_coverage": 0.35,
  "band": null,
  "assessment_status": "insufficient_evidence"
}
```

User-facing response for the insufficient case:

> The available requirements align well, but the posting does not provide enough information for a confident overall match rating.

Coverage threshold for `band=null` is calibrated in Step 12. Until calibration, band is only rendered when `evidence_coverage ≥ 0.60` (starting default; subject to eval-set feedback).

**Renderer invariant:** `qualification_fit` is never rendered without `evidence_coverage`. Enforced at template layer per the renderer architecture (Graph decides / Evidence constrains / LLM explains / Template protects).

---

## Contract 4 — Alternative Requirements (Groups)

Two group rules only: `any` and `all`. No dependency DAG.

```json
{
  "group_id": "education_or_experience",
  "group_rule": "any",
  "requirements": [
    { "type": "education",  "value": "college diploma" },
    { "type": "experience", "value": "3 years equivalent experience" }
  ]
}
```

**ANY group truth table:**

| Condition | Result |
|---|---|
| At least one member `met` | met |
| No `met`, at least one `partial` | partial |
| All known members `not_met` (no unknowns) | not_met |
| Otherwise | unknown |

**ALL group truth table (LOCKED):**

| Condition | Result |
|---|---|
| Every member `met` | met |
| At least one member `not_met` | not_met |
| At least one `met`, no `not_met`, at least one `unknown` | partial |
| All members `unknown` | unknown |

Example: `First Aid = met, CPR = unknown, rule = ALL` → `partial`. Cannot be `met` (CPR unknown); cannot honestly be `not_met` (First Aid confirmed).

---

## Contract 5 — Extraction Quality on Both Sides

Both job-side and candidate-side evidence carry the same quality envelope:

```json
{
  "value": "Bachelor's degree",
  "source": "job_description",
  "extraction_status": "extracted",
  "extraction_confidence": 0.93
}
```

**Confidence threshold ownership: comparator-level, configurable.**

Each comparator holds its own threshold as configuration:
```
skill_comparator.threshold      = configurable
education_comparator.threshold  = configurable
experience_comparator.threshold = configurable
```

**Store raw extraction confidence permanently.** Never store only a computed `is_reliable` boolean — that would require re-extraction whenever a threshold changes.

At comparison time:
```
confidence ≥ comparator_threshold  →  compare evidence
confidence <  comparator_threshold  →  status = unknown_(job|candidate)_evidence
```

Every comparison result records which policy evaluated it:
```json
{
  "comparison_policy_version": "qualification-v1",
  "confidence_threshold": 0.70
}
```

**Absence semantics:**
- High-confidence evidence present → compare normally
- Low-confidence evidence → unknown, not not_met
- Missing source content (no section, no field) → unknown
- Explicit evidence absent (section present, requirement absent) → `not_met` ONLY when absence can genuinely be established

Most resume omissions remain unknown, not automatically `not_met`.

---

## Contract 6 — Responsibilities Treatment

Responsibilities are stored separately from requirements.

```
"Coordinate individual support plans"
"Work with community partners"
"Maintain client documentation"
```

Their two legitimate uses:

1. **Infer implied skill requirements** — "Coordinate individual support plans" → derive `support planning` skill  
2. **Infer experience domain** — feed the responsibility-similarity dimension of the experience comparator

**Responsibilities never directly score.** No `responsibility_met = true/false`. Only the derived skill/experience requirements participate in qualification comparison.

---

## Contract 7 — Comparator Failure Semantics

Every comparator fails independently and softly.

**On comparator crash / malformed output:**
```
dimension.status = comparator_error
dimension excluded from qualification_fit combination
dimension's dimension_weight × 0 in coverage rollup (via effective_weight)
other dimensions still render normally
error logged internally, not surfaced to the user
```

**Never do this:**
```
comparator crash → dimension.score = 0
```

**Never do this:**
```
one comparator crash → whole job comparison fails
```

The whole match fails ONLY if the core engine or market query is unavailable — not on any single dimension failure.

---

## Contract 8 — Evaluation Set and Shadow-Run

**Eval-set-first ordering.** Build the labelled evaluation set BEFORE shadow-running V2. Otherwise the shadow-run measures only V2-vs-V1 disagreement without ground truth.

**Label schema (per pair):**
```json
{
  "skill_fit":  "met|partial|not_met|unknown",
  "education":  "met|partial|not_met|unknown",
  "experience": "met|partial|not_met|unknown",
  "eligibility": "pass|fail|unknown",
  "overall_band": "strong|good|stretch|not_a_match",
  "important_missing_requirements": ["2 years crisis-response experience"]
}
```

`important_missing_requirements` lets us measure "did V2 catch the requirement the human labeler flagged as important?" — a stronger signal than aggregate band correlation.

**Shadow-run measurement — asymmetric by design.**

V1 emits a single `match_score` + `match_band`. It cannot be scored on eligibility, dimension-level fit, coverage, or requirement omission — those axes do not exist in V1. Do not try to reconstruct them post-hoc.

| Engine | Measurements available |
|---|---|
| V1 | band agreement, ranking quality |
| V2 | band agreement, ranking quality, eligibility errors, false Strong/Good, unknown-as-missing, important-requirement recall, coverage accuracy |
| Human labels | Truth |

Shadow-run report reads: `V1 → 2 measurements, V2 → complete measurements, human labels → truth`. Never `V1 vs V2 on eligibility`.

**Initial size:** 50–100 labelled job/candidate pairs. Split into calibration set and held-out evaluation set.

**Maintenance loop (lightweight, durable):**

Every eval-set artifact is versioned:
```
eval_set_version
label_schema_version
weight_policy_version
engine_version
```

Minimum policy:
1. Start with 50–100 labelled job/candidate pairs.
2. Separate calibration examples from held-out evaluation examples.
3. Add confirmed false matches and missed matches as new candidates.
4. Version every approved evaluation-set update.
5. Version every weight and threshold change.
6. Reject releases that exceed an agreed regression tolerance on the held-out set.

No quarterly cadence yet — re-evaluate at release milestones and when enough new labelled cases accumulate.

---

## Recommended Order (Locked)

1. Finish Step 1A source integrity (in progress — impls 9–15 pending)
2. Complete Step 2: remove all title-to-fit behavior (`_direct_title_match_score`, `_target_role_boost`, title-forced Stretch override)
3. Amend the requirement-schema design with the contracts above (this document)
4. Build the first labelled evaluation set (per Contract 8)
5. Add structured `job_requirement` table + extraction-quality schema
6. Add candidate evidence quality envelope (mirrors JD side)
7. Build the three requirement comparators (skill / education / experience) per Contracts 2, 5, 7
8. Add fit + coverage + eligibility outputs (MatchResultV2)
9. Run V2 in shadow mode against V1 and the eval set (per Contract 8 asymmetry rule)
10. Calibrate weights and thresholds from the eval set; cut consumers over
11. Replace fetch-all execution with two-stage retrieval

Steps 3–8 cannot start until Step 1A and Step 2 are signed off. Steps 5–7 cannot proceed until Contracts 1–7 are amended into the requirement-schema design (Step 3 of the order).

---

## Design Rule (One Sentence)

> Do not keep enlarging `_score_one_job`. Preserve its skill-alignment logic as the skill comparator, and move the three-dimensional comparison into a separate V2 qualification matcher governed by the eight contracts above.
