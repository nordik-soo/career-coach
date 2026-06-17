# Near-Miss Gap Analysis — Design

Status: design **locked** · 2026-06-05 · ready for build

Open questions section was reviewed and all 7 decisions are locked
below (Q1-Q6 + an added Q7 covering precondition ownership). Slice N-1
was tightened — see "Gap classification" — to reuse the existing
registry `category` field instead of introducing a new one.

## Why this document exists

The 2026-06-05 live test surfaced a product correctness gap, not a
polish item.

A candidate uploaded a Truck and Coach Technician apprentice resume,
asked for that same role, and the chat replied *"I don't see one in
today's Sault Ste. Marie postings."* The dataset has exactly one
matching posting (Garden River First Nation, posted 11 days prior). The
matching engine ranked it #1 with a perfect title match (similarity =
1.0). It was hidden because the candidate is missing 10 of 12 required
skills plus two credentials (310T + Class G), so the band was capped to
`low`. The presentation layer treats low-band as not-shown.

The post-live-test telemetry fix introduced `band="low_only"` to
distinguish "engine found candidates but they were filtered" from
"engine found nothing." That's the diagnostic; this design turns it
into the actual product behavior.

| State | Today | Should be |
|---|---|---|
| Engine returns strong/good | `present_matches` | unchanged |
| Engine returns stretch | `present_matches` (stretch_only) | unchanged |
| Engine returns low-band with title match | `present_no_match` | **`present_near_miss`** (NEW) |
| Engine returns low-band, no title match | `present_no_match` | `present_no_match` |
| Engine returns nothing eligible | `present_no_match` | `present_no_match` |

For SkillBridge SSM specifically, the difference is grant-aligned. The
NORDIK/Algoma research mandate is skill-gap analysis for newcomers, not
job-board hit/miss. A newcomer asking *"can I work as a truck
technician here?"* deserves *"yes, the role exists; here are the two
real barriers"* — not *"no jobs"*.

## Decision summary (for reviewers)

- **Add a new `OutcomeMove` value `present_near_miss`** distinct from
  `present_matches` and `present_no_match`.
- **Trigger is conjunctive (4 ANDs).** Misses any one → fall through to
  existing behavior. Conservative on purpose.
- **Gap classification reuses the existing registry `category` field**
  (`credential` / `license` / `safety_training` / `skill`) and MAPS it
  to the near-miss vocabulary (`credential` / `core_skill`). No new
  YAML field. For gaps surfaced from `job_skills.skill_name` that
  aren't in the registry, a keyword heuristic classifies them and logs
  telemetry for backlog triage. Operational job-as-acquired requirements
  (on-call, contract supervision) are filtered out — they're not
  closeable gaps.
- **Handler computes preconditions; arbiter stays simple.** The handler
  builds `near_miss_candidates` only when baseline-evidence + specific-
  target gates pass; if the list is empty, arbiter falls through to
  `present_no_match`. The arbiter does NOT recompute preconditions.
- **Decision lives in arbiter pass 2.** `resolve_match_outcome` gains
  ONE new input (`near_miss_candidates: list`) and one new output
  branch. Match-count→outcome decisions stay in one place.
- **Responder reuses `_registry_grounded_explain_gap_fallback`** for
  the "here's the path" narration. One source of truth for "how do I
  close gap X."
- **Text-only first.** No UI card design in this slice. Card and gap
  prioritization ("310T unlocks 4 jobs") are explicitly deferred.
- **NOT flag-gated.** This is a correctness fix, not an experimental
  rollout. Flag-on-by-default behavior; rollback by git revert if it
  ever needs to come out.

## Architecture

### Current flow (post-Slice-D, MESSAGE_UNDERSTANDING_ENABLED=true)

```
router/planner decides move
    │
    ▼ proceed_to_match
arbiter pass 1 verifies (RunEngine)
    │
    ▼
compute_matches_in_memory()
    │
    ▼
_build_results_block() -> (results, band_signal)
    │   results may be [] with band_signal="low_only"  (post-telemetry-fix)
    ▼
arbiter pass 2: resolve_match_outcome(match_count, ...)
    │
    ├─ match_count > 0 → present_matches
    └─ match_count == 0 → present_no_match  ← swallows the low_only case
```

### Proposed flow

```
compute_matches_in_memory()
    │
    ▼
_build_results_block() -> (results, band_signal)
    │
    ▼ (NEW step, in handler before arbiter pass 2)
        if (band_signal == "low_only"
            AND truth.target_role_specificity == "specific"
            AND has_baseline_evidence(staged, truth)):
            near_miss_candidates = filter_near_miss_candidates(
                low_matches, target_role_text, target_noc,
            )
        else:
            near_miss_candidates = []
    │
    ▼
arbiter pass 2: resolve_match_outcome(
    match_count,
    near_miss_candidates,         # NEW (list; empty when preconds fail)
    caps_applied,
    planner_reason_code,
    planner_tone,
)
    │
    ├─ match_count > 0                              → present_matches
    ├─ match_count == 0 AND near_miss_candidates    → present_near_miss   (NEW)
    └─ match_count == 0 AND no near_miss_candidates → present_no_match
```

`filter_near_miss_candidates` lives in `skillbridge/match/near_miss.py`
(new module — locked Q1). The handler owns the precondition logic so
the arbiter stays a thin outcome-selector. If the handler does NOT pass
a non-empty list, the arbiter treats it as the existing
`present_no_match` case verbatim — no behavior change for any path
that doesn't qualify.

## Trigger conditions (locked: conjunctive, 4 ANDs)

`present_near_miss` fires when ALL of the following hold:

1. **`band_signal == "low_only"`** — engine found eligible candidates,
   all below stretch.
2. **`truth.target_role_specificity == "specific"`** — user has named a
   concrete role. Without a target, "near-miss to what?" has no answer.
3. **Candidate has baseline evidence** — resume parsed (`resume_parse_quality
   in {"full", "partial"}`) OR `chat_skill_count >= 3`. Without baseline,
   every required skill reads as a "gap" and the analysis is noise.
4. **At least one low-band candidate matches title or NOC** — see filter
   rules below. Without this, low_only is unrelated roles and we shouldn't
   pretend the role "exists in town."

Miss any → fall through to existing `present_no_match`.

The "baseline evidence" gate is deliberately strict. For a near-empty
profile, the right response is the existing `ask_one_clarifying_question`
path, not gap analysis against nothing.

### What counts as "matches title or NOC"

A low-band candidate is a near-miss if ANY of:

- `score_explanation.title_match_override == True` (engine already
  decided it's a title match — strongest signal)
- `score_explanation.title_match_similarity >= 0.85` (high lexical
  similarity even if override didn't fire)
- `job.noc_code` equals `staged.target_noc` (NOC resolved from target
  role and matches the job's NOC)

NOT a near-miss:
- Score-based proximity alone (irrelevant if it's a different role)
- Skills overlap alone (lots of unrelated roles share generic skills)

This is intentionally strict. We can loosen later if live data shows
real near-misses being missed. Starting strict avoids "near-miss" becoming
"any low job."

## Gap classification

### The two-source rule

A gap surfacing from `score_explanation.required_missing` is classified
by ONE of two sources, in this order:

1. **Registry hit** — gap name matches a canonical/alias in
   `data/training_registry.yaml`. Use the existing `Gap.category`
   field, mapped to near-miss vocabulary (see table below).
2. **Heuristic fallback** — gap name doesn't match the registry. Apply
   keyword rules + log telemetry.

No new YAML field is added. The reviewer caught that
`gap_category` would have shadowed/duplicated `category`. Mapping
from the existing field is honest and avoids drift.

### Mapping: registry `category` → near-miss category

The registry has these 4 categories today (verified against the live
YAML, 13 entries):

| Registry `category` | Maps to | Surface in narration? |
|---|---|---|
| `credential` | `credential` | **Yes** — leads narration |
| `license` | `credential` | **Yes** — leads narration |
| `safety_training` | `credential` | **Yes** — leads narration |
| `skill` | `core_skill` | **Yes** — follows credentials |

Anything `credential`-like (regulated, certified, or safety-mandated)
collapses to the single near-miss bucket `credential`. The newcomer
doesn't care about the legal distinction between a "credential" and a
"license"; what matters is "this needs to be earned somewhere
authoritative." `skill` becomes `core_skill`.

The full 13-entry distribution in today's YAML lands as:

| Today's count | Maps to |
|---|---|
| 3 `credential` + 2 `license` + 4 `safety_training` = **9** | `credential` |
| 4 `skill` = **4** | `core_skill` |

### Heuristic for non-YAML gaps (locked Q3)

When the engine surfaces a gap not present in the registry — common,
because `job_skills.skill_name` covers a wider vocabulary than the 13
curated YAML entries — classify with keyword rules and log:

```python
def classify_unregistered_gap(name: str) -> Literal["credential", "core_skill", "operational"]:
    s = name.lower()
    if any(k in s for k in (
        "certif", "license", "licence", "ticket", "qualification", "credential",
    )):
        cat = "credential"
    elif any(k in s for k in (
        "availability", "supervision", "tracking", "on-call", "on call",
        "shift willing", "hour tracking",
    )):
        cat = "operational"
    else:
        cat = "core_skill"
    log.info("near_miss heuristic_classified gap=%r category=%s", name, cat)
    return cat
```

The INFO log uses the existing `unknown_gap=` pattern as a model. The
telemetry lets us review popular non-YAML gaps over time and promote
them into the curated registry — same backlog flow as today.

### Operational is FILTERED, not narrated

Only `credential` and `core_skill` reach the responder. `operational`
gaps are dropped during candidate-list construction so the responder
never sees them. The responder doesn't need a CANNOT rule for it; the
data simply isn't there.

This is the only category that changes the gap LIST presented. The
responder still cares about CAN/CANNOT phrasing rules (see contract
below), but the gap menu it works from is already pre-filtered.

### Why mapping (not new field)

The reviewer's call. Three reasons:

1. **The data already exists** for all 13 YAML entries. Adding a new
   field invites drift if the two are ever out of sync.
2. **One source of truth.** A future contributor editing
   `category: license` doesn't have to remember to also edit
   `gap_category: credential` in lockstep.
3. **`near_miss_category` is reserved as a future field** if we ever
   need a near-miss-specific value that doesn't fit the existing
   taxonomy. We start without it.

## Responder contract

### CAN say

- *"I found [role] postings, but they're not a realistic match yet."*
- *"The main blockers are: [credential], [credential], [core_skill]."*
- *"The most important next step is [strongest credential gap], then
  [next gap]."*
- Provider names from the training registry for the named credentials
  (DriveTest, Skilled Trades Ontario, Sault College, etc.)
- "plus N more similar postings" when more than one near-miss exists
- Sault Community Career Centre as the human-help referral

### CANNOT say

- *"You qualify for…"* — they don't
- *"This is a good fit / good match"* — it's a near-miss
- *"Stretch match"* — distinct category; reserved for stretch band
- Job-specific operational requirements as "gaps" (on-call, MTO
  supervision, etc.)
- Dollar amounts, salary ranges, statistics
- Adjacent role suggestions ("try heavy equipment instead") — that's
  `offer_refinement` territory, separate move
- Any URL outside the training registry (same constraint as today)
- More than 3 credentials + 3 core skills in one response (cognitive
  load; the user is being told they're not ready for this role —
  drowning them in 12 gaps reads as "give up")

### Tone

`warm_supportive`. Honest about the gap, optimistic about the path.
NOT `honest_redirect` — that reads as "I can't help with this" which is
the wrong frame when we're actively helping with the gap.

## Worked example: Michael Carter, truck-tech apprentice

**Input.** Resume parsed (`resume_parse_quality=full`). Target role:
`"truck and coach technician"`. Skills extracted from resume: diesel
engine repair, brake and suspension systems, air brakes, welding,
diagnostic tools (5 skills).

**Engine output.** Truck and Coach Technician (Garden River First
Nation):
- `match_band = "low"` (in actual session — synthetic profile got
  `stretch`; depends on exact skill extraction)
- `score = ~0.30`
- `title_match_override = True`
- `score_explanation.required_missing` contains 10 items
- `score_explanation.credential_gap_skills = ["310T certificate of
  qualification", "Class G driver's license"]`

**Build results block.** `band_signal = "low_only"`, `results = []`
(post-telemetry-fix).

**Near-miss filter.** One candidate passes (`title_match_override =
True`).

**Arbiter pass 2.** Triggers `present_near_miss`.

**Gap classification.** Each entry from `required_missing` is run
through `classify_gap(name, registry)`. Registry hits use the
existing `category`; misses fall to the keyword heuristic.

| Gap | Source | Registry `category` | Near-miss bucket | Surface? |
|---|---|---|---|---|
| 310T certificate of qualification | registry | `credential` | `credential` | yes |
| Class G driver's license | registry | `license` | `credential` | yes |
| truck service and maintenance | heuristic | — | `core_skill` | yes |
| emergency repair | heuristic | — | `core_skill` | maybe (cap at 3 skill items) |
| emissions testing preparation | heuristic | — | `core_skill` | maybe |
| wheel end inspection | heuristic | — | `core_skill` | drop (cap) |
| parts fabrication | heuristic | — | `core_skill` | drop (cap) |
| motor vehicle inspection | heuristic | — | `core_skill` | drop (cap) |
| MTO contract supervision | heuristic (matches `supervision`) | — | `operational` | filtered |
| driver hour tracking | heuristic (matches `hour tracking`) | — | `operational` | filtered |
| on-call availability | heuristic (matches `on-call`) | — | `operational` | filtered |

After cap (3+3) and credential-first ordering, the narration sees:

```
credentials: [310T certificate of qualification, Class G driver's license]
core_skills: [emergency repair, emissions testing preparation, truck service and maintenance]
```

(`core_skills` alphabetically stable per Q4 lock.)

**Responder output (target shape).**

> "I found a Truck and Coach Technician posting at Garden River First
> Nation, but it's not a realistic match yet. The main blockers are
> your 310T certification, your Class G driver's license, and a few
> core skills like truck service and maintenance and emergency repair.
>
> The most important next step is your 310T pathway — Skilled Trades
> Ontario handles that, with continuing-ed support at Sault College.
> Class G testing goes through DriveTest.
>
> Want to walk through the 310T path first?"

(Exact wording is responder's job; this shows the shape and grounding.)

## Tests to add

### New tests

| Module | Test count (est.) | Coverage |
|---|---|---|
| `tests/test_match_near_miss.py` (new file) | ~18 | `classify_gap` — each of 13 YAML entries maps to expected bucket; heuristic table for credential / operational / core_skill defaults; INFO telemetry log fires once. `filter_near_miss_candidates` — title-override wins, similarity≥0.85 wins, NOC match wins, score-only doesn't pass. |
| `tests/test_chat_arbiter.py` (extend) | ~5 | Pass 2 emits `present_near_miss` when `match_count==0` AND `near_miss_candidates` non-empty; empty-list = legacy `present_no_match` preserved; new reason code surfaces |
| `tests/test_chat_handler_v2.py` (extend) | ~4 | Handler computes preconditions: low_only + specific target + baseline → calls filter; missing any precondition → empty list. Truth log gains `near_miss=N` field. |
| `tests/test_chat_transcripts.py` (extend) | 1 scenario | Michael truck-tech case, end-to-end through router → engine → near-miss → responder |
| `tests/test_chat_responder_v2.py` (extend) | ~4 | Forbidden phrases ("you qualify", "good fit", "stretch match"), required phrases ("not a realistic match yet"), credential-then-skill ordering, max 3+3 cap |

### Ported tests (existing → adapted)

| Existing | Becomes |
|---|---|
| Any `test_*` asserting `present_no_match` when band_signal is low_only with a title-match | Becomes a `present_near_miss` assertion |

### Live regression tests (manual, post-build)

1. Cold session: upload Michael truck-tech resume; confirm; say `same role` → `present_near_miss` with 310T + Class G + 1-2 skill gaps narrated
2. Cold session: upload empty / minimal profile; ask for truck tech → falls through to `ask_one_clarifying_question` (baseline-evidence gate)
3. Mid-session: ask for `electrician` (no jobs of that title in dataset) → `present_no_match`, NOT near-miss (no title match)
4. Mid-session: ask for `tour guide` (one low-band tour guide job exists per current data) → behavior depends on candidate evidence; verify the right outcome fires

## Build slices

Each independently shippable. Build → test → review → next.

### Slice N-1 — Category-mapping helper (~30 min)

- NEW: `skillbridge/match/near_miss.py` with `classify_gap(name: str,
  registry: TrainingRegistry) -> Literal["credential", "core_skill",
  "operational"]`
- Reads registry by alias/canonical match; falls back to keyword
  heuristic with INFO telemetry log
- NO YAML changes (existing `category` field reused)
- NO `Gap` dataclass changes
- Tests: each of the 13 current YAML entries maps to the expected
  near-miss bucket; heuristic table for `credential`/`operational`/
  `core_skill` defaults; telemetry log fires once per unregistered gap

### Slice N-2 — near-miss candidate filter (~1 hour)

- `filter_near_miss_candidates(low_matches, target_role_text,
  target_noc) -> list[MatchResult]`
- Filter rules from "What counts as 'matches title or NOC'" above:
  `title_match_override`, `title_match_similarity >= 0.85`, or
  `job.noc_code == target_noc`
- ~10 unit tests covering each filter branch + the disqualification
  cases (score-only proximity, skill-overlap-only)

### Slice N-3 — arbiter pass 2 extension (~1 hour)

- Add `present_near_miss` to `OutcomeMove` Literal
- Extend `resolve_match_outcome` signature with ONE new parameter:
  `near_miss_candidates: list[MatchResult]` (default `[]`)
- New arbiter reason code: `ARBITER_REASON_NEAR_MISS = "title_match_with_major_gaps"`
- Branch logic (locked Q7): if `match_count == 0 AND
  near_miss_candidates`, emit `present_near_miss`. Otherwise existing
  behavior.
- Tests: each branch fires correctly; empty list = legacy behavior
  preserved (rollback safety)

### Slice N-4 — responder text (~2 hours)

- New `_present_near_miss_fallback_v2` deterministic narrator
- New phrasing in responder prompt for the LLM path (with the
  CAN/CANNOT contract)
- Policy regex updates: pre-existing patterns continue to apply; one
  new pattern blocks "good fit"/"good match" in near-miss turns
- Tests for fallback shape + LLM-policy round-trip

### Slice N-5 — handler wiring (~30 min)

- Compute `near_miss_candidates` after `_build_results_block`
- Pass into `resolve_match_outcome`
- Update truth log to include near-miss count: `near_miss=N`

### Slice N-6 — live regression + sign-off (~1 hour)

- Run the 4 live scenarios
- Update transcript test with Michael case
- Sign-off → merge

Total estimate: ~6 hours across 6 slices, with review between each.

## Locked decisions (reviewed 2026-06-05)

The 6 open questions were reviewed and locked. Future contributors:
don't reopen these without a new design step.

### Q1. Where does `filter_near_miss_candidates` live?

**Locked: new module `skillbridge/match/near_miss.py`.**

Reasoning: separate concern (post-engine filtering), new testable
surface, doesn't bloat `engine.py`. The recommender (`recommend.py`)
is about which training closes a known gap; near-miss filtering is
about which JOBS qualify as a near-miss candidate — different shape.

### Q2. Category source on the `Gap` dataclass

**Locked: reuse the existing `Gap.category` field.** Map registry
categories → near-miss vocabulary in `near_miss.py` (see the
classification section above).

Do NOT add `gap_category` to YAML. Do NOT shadow `category`. If a
future scenario needs a near-miss-specific value the existing taxonomy
can't express, introduce `near_miss_category` THEN — not now.

### Q3. Heuristic for non-YAML gaps

**Locked: keyword heuristic + telemetry log.**

```
contains {certif, license, licence, ticket, qualification, credential}
    → credential
contains {availability, supervision, tracking, on-call, shift willing, hour tracking}
    → operational
otherwise
    → core_skill
```

Each classified-via-heuristic gap logs `INFO near_miss
heuristic_classified gap=... category=...` so the registry backlog gets
real data on which YAML entries to add next.

### Q4. How many gaps to narrate

**Locked: max 3 credentials + max 3 core_skills.** Credentials first.
Stable ordering (alphabetical) for v1. Impact-based prioritization
("310T unlocks 4 jobs") is a follow-up slice — keep v1 focused on
shipping the outcome, not the recommendation story.

### Q5. Resume says credential exists but engine says missing

**Locked: trust the engine.** If `required_missing` includes a
credential the resume confirmed, that's an extraction/matching bug —
not a near-miss-layer responsibility. Double-handling it in `near_miss`
would mask the underlying bug. The near-miss code consumes
`required_missing` verbatim.

### Q6. Multi-job near-miss aggregation

**Locked: highest-scoring only + "plus N similar postings."**

The responder narrates gaps from the top-ranked near-miss candidate.
A trailing count line ("…plus 2 more Truck and Coach Technician
postings with similar requirements") signals the role isn't a one-off
in the dataset. No union, no intersection. Same pattern as v2.1's
multi-entity decision.

### Q7 (added in review). Where does precondition logic live?

**Locked: handler computes; arbiter decides.**

The handler is responsible for:
- Checking `band_signal == "low_only"`
- Checking `truth.target_role_specificity == "specific"`
- Checking baseline-evidence (resume parsed OR ≥3 chat skills)
- If all three hold, calling `filter_near_miss_candidates(...)` and
  passing the result to `resolve_match_outcome`
- If any precondition fails, passing `near_miss_candidates=[]`

The arbiter is responsible for:
- A single check: `match_count == 0 AND near_miss_candidates` → emit
  `present_near_miss`. Otherwise existing behavior.

This split keeps `resolve_match_outcome` a thin outcome-selector and
puts the truth/staging-aware precondition logic next to the rest of
the handler's truth-aware code. The arbiter remains a pure function
of its arguments; nothing about it changes shape beyond one extra
input parameter.

## What I do NOT promise

- The heuristic in Q3 will mis-classify some gaps. We'll fix them by
  promoting into YAML.
- The "not a realistic match yet" wording is not pre-tested with
  newcomers. We may need to soften further or harden depending on
  feedback.
- The 4-condition trigger may have false negatives (real near-misses
  that don't fire). I'd rather start strict and loosen than start
  loose and tighten.

## Rollback

This is NOT flag-gated.

Reasoning: `MESSAGE_UNDERSTANDING_ENABLED` and `TRAINING_REGISTRY_ENABLED`
were rolled out with flags because they were behavior-altering
experiments. `present_near_miss` is a correctness fix — the current
"no jobs" reply for an existing same-title-but-far-from-ready role is
a product bug for SkillBridge's grant scope.

If it ships broken: `git revert` the slices. Same pattern as any other
non-flagged correctness fix.

## Verdict

This is the right next product slice. It directly serves the
grant-funded skill-gap-analysis purpose, has a clear architectural
shape, reuses existing infrastructure (registry, arbiter, responder
fallback patterns), and is bounded to ~6 hours of focused work.

Cost: engineering time + your call on the 6 open questions + your
input on classifying the 13 existing YAML entries.

Win: SkillBridge stops saying "no jobs" when the answer is "the role
exists, here are the real barriers." That distinction is the product.

Awaiting review.
