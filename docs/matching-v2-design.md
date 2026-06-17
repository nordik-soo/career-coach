# Matching v2 — Design Direction (Hybrid + OaSIS)

Status: draft · Target: pickup after Sprint 5 + slice 4 close-out

The current matcher is deterministic, auditable, and honest about credentials.
Its weak spot is semantic equivalence: "brake and suspension systems" vs
"brake system inspection and repair" are the same job to a human and to a
candidate, but the matcher correctly reports zero overlap because the tokens
disagree. The next leverage isn't more rules; it's normalization plus a
narrow semantic layer that re-ranks, not decides.

This document is a direction, not an architecture. It commits to a shape,
flags what we explicitly defer, and orders the work so each piece is
shippable independently.

---

## 1. Where Sprint 5 + slice 4 left us

Pipeline shape today:

```
JD-side: LLM extraction (required|preferred labels)  →  extracted.job_skill
                                                                 │
User-side: resume LLM extraction → StagedSkill list              │
                                                                 │
                                  ↓
Match per JD skill:
  1. exact skill_id (reference.skill table)
  2. exact name (lowercased)
  3. canonical name (47-entry alias map)
  4. word-bounded substring + token overlap (≥0.60)
                                                                 │
                                                                 ↓
Score = 0.8 × required_match_ratio + 0.2 × preferred_match_ratio
      + small boosts (location, recency, target_role)
Hard gates: credential cap, no-experience floor, work-type mismatch
Band: strong | good | stretch | low
score_explanation: full structured breakdown + caps_applied list
```

**What works**
- Single-letter / short skill names no longer wildcard-match unrelated jobs
  (the "r" bug, slice 4 follow-up).
- Credentials always survive the top-N filter (slice 4b carve-out).
- Caps fire honestly and the responder names them.
- Legacy (skill_type=NULL) rows score exactly as v1.1.

**What's missing**
- **Semantic equivalence**: token overlap doesn't bridge "diesel engine
  repair" ↔ "diesel engine diagnosis and repair", or "brake and suspension
  systems" ↔ "brake system inspection and repair". Today these read as
  missing skills even though they're the same competency.
- **Hierarchy**: a React developer doesn't get credit toward "JS framework";
  a PostgreSQL DBA doesn't surface for a SQL role.
- **Bilingual lexicon**: French resume entries don't match English JD entries
  and vice versa, despite being the same skill.
- **Different-title same-role**: two employers post the same job under
  different titles ("Software Developer" vs "Application Programmer";
  "Truck and Coach Technician" vs "Diesel Mechanic"). The matcher today
  treats them as unrelated postings — a user typing "software developer"
  finds the first but not the second, even when the underlying work is
  identical. This is at the occupation layer, not the skill layer.

These are the four classes of false-negative the next pass should close.

---

## 2. OaSIS as normalization dictionary

[OaSIS](https://noc.esdc.gc.ca/Oasis/OasisWelcome) (Occupational and Skills
Information System) and its companion
[SCT](https://noc.esdc.gc.ca/SkillsTaxonomy/SkillsTaxonomyWelcome) (Skills
and Competencies Taxonomy) are ESDC's public, free datasets. Both are
distributed as CSV on Canada's
[Open Government Portal](https://open.canada.ca/) under the Open Government
Licence — Canada. OaSIS 2025 covers 900 occupations using NOC 2021 codes.

For SkillBridge specifically, after verifying the actual file structure:

**What we'd use:**

| Asset | Source file | Why it matters here |
|---|---|---|
| NOC 2021 codes + example titles per occupation | OaSIS `example-titles_oasis_2025.csv` | Maps free-text job titles ("Software Developer", "Application Programmer", "Coding Specialist") to the same NOC. **The strongest fit.** Powers the occupation-normalization layer (§3). |
| Alternative job titles (extra synonym layer) | SCT `Alternative Titles 2023.csv` | A second pass of title synonyms beyond OaSIS example titles. Bilingual EN/FR. |
| Broad O*NET-style skill descriptors | OaSIS `skills_oasis_2025.csv` + SCT `Descriptors` | ~240 standardized descriptors (Reading comprehension, Active listening, Critical thinking, Operation and Control, etc.) rated 1-5 per occupation. Useful for canonicalizing **soft** skills in resumes ("verbal communication" → OaSIS "Active listening"). |
| English / French parallel entries throughout | Both datasets, paired `_oasis_`/`_sipec_` files | Newcomers to SSM include francophones; OaSIS canonical IDs are language-agnostic so the same ID matches across both. |

**What we explicitly wouldn't use:**

- OaSIS wage / outlook data — out of scope; we don't surface wage or
  forecast claims in the chat (see SCOPE BOUNDARIES rule).
- OaSIS regulated-occupation cross-references — we have
  `core.regulated_occupation` already; OaSIS can supplement but not
  replace, since our table carries Ontario-specific licensing notes.

**What OaSIS does NOT give us (important):**

This is the part I had to verify after writing the first draft of this doc.
OaSIS is occupation-centric and competency-framework-shaped (like O*NET),
not a granular skill taxonomy. Specifically:

- **No synonym lexicon for tech / trades / credential vocabulary.** There
  is no OaSIS canonical for "Python", "AWS Sagemaker", "Next.js", "Class G
  licence", "310T certificate", "WHMIS". These are too specific for the
  240-descriptor framework. Our existing 47-entry alias map stays as
  evidence-driven canonicalization for this layer; the semantic re-ranker
  (§4 Stage 3) handles what the alias map doesn't.
- **No skill-hierarchy edges between specific skills.** OaSIS groups its
  240 descriptors into 7 categories (skills, abilities, personal
  attributes, knowledge, interests, work activities, work context), but
  there are no parent/child edges like "React is-a JS framework". The
  child-beats-parent asymmetry in §4 will need a different substrate —
  either a small hand-curated edges file for the cases we observe, or
  embeddings-as-proxy via the semantic re-ranker.

So OaSIS's value concentrates on the **occupation layer** (§3) more than
the skill layer. The skill-side wins from OaSIS are limited to broad
soft-skill canonicalization.

**Integration cost (estimate, not a commitment):**

- One-time import: ~1 day for the occupation/title layer alone (see §6
  step 1). The skill descriptor import is a separate, later decision.
- Recurring re-sync: OaSIS publishes versioned releases roughly yearly.
  Build a version-aware re-import (idempotent, replaces stale rows).
- Disk: occupation + title-synonym CSVs are small (~5 MB combined).
- Failure modes: download blip → keep prior data, log warning, retry on
  next pipeline run. Never crash matching because OaSIS is stale.

**Doesn't solve:**

- Resume phrases that aren't in OaSIS at all (which is most tech vocab —
  see "What OaSIS does NOT give us" above). The semantic layer (§4)
  handles these, not the dictionary.

---

## 3. Occupation normalization (different titles, same role)

Skill-level normalization (§2) bridges *"Python" ↔ "programming languages"*.
Occupation-level normalization bridges *"Software Developer" ↔ "Application
Programmer"* — same role, different vocabulary, different employers. Both
postings should surface together when a user types either phrase.

OaSIS uses NOC 2021 codes for the occupation layer. The work is to populate
`core.job_posting.noc_code` reliably and use it as a candidate-pool signal
in the matcher.

**Pipeline addition:**

```
At ingest:
  JD title  →  OaSIS NOC lexicon lookup  →  noc_code (e.g. 21232)
                          ↓ (miss / ambiguous)
                  small LLM resolver  →  noc_code, or NULL if uncertain

At chat:
  staged.target_role_text  →  same resolver  →  staged.target_noc
                                                      ↓
  Matcher BOOSTS jobs whose noc_code matches user's target_noc,
  with a smaller boost for SIBLING NOCs (via OaSIS occupation hierarchy).
  Does NOT filter -- the user can still see adjacent roles.
```

**Worked example:**

| | Company A | Company B |
|---|---|---|
| Raw title | "Software Developer" | "Application Programmer" |
| Resolved NOC 2021 | 21232 | 21232 |
| User's target_role_text → NOC | "software developer" → 21232 | (same) |
| Engine | NOC match boost on both jobs; skill scoring runs as usual | (same) |
| Responder | "I found 2 software-developer roles in SSM — one at Company A, one at Company B…" | |

**Design call: boost, not filter.**

Sault Ste. Marie's job market is small. A filter would over-narrow ("no
software developer roles in SSM" when there's an adjacent IT support role
the user could plausibly take). A boost re-orders so the on-target roles
surface first, while adjacent roles stay visible. The responder's
DATASET-FIRST rule keeps the framing honest: "I don't see a direct match,
but here are adjacent roles you might consider."

**What this doesn't do:**

- Cross-NOC clustering by inferred role family. Two NOCs that aren't
  marked as siblings in OaSIS stay distinct, even if a human would group
  them. Reasonable — better to be conservative than to invent affinity
  the taxonomy doesn't support.
- Replace the existing `_direct_title_match_score` fast-path. NOC match
  becomes the preferred signal; title-token similarity is the fallback
  when NOC resolution fails.

---

## 4. Hybrid matching: deterministic core + semantic re-ranker

The deterministic pipeline stays; the semantic layer plugs in as a
re-ranker on cases the deterministic pipeline reports as miss.

**Three-stage match decision per JD skill:**

```
For each JD skill j and each user skill u:

  Stage 1 — Deterministic ID/name match
    • Exact OaSIS canonical_id match            → matched (strength 1.0)
    • Synonym match via OaSIS lexicon            → matched (strength 1.0)
    • Word-bounded substring (current behavior)  → matched (strength 1.0)
    • Token overlap ≥ 0.60 (current behavior)    → matched (strength 0.85)

  Stage 2 — Hierarchy traversal (via OaSIS edges)
    • user has CHILD of j (specialization)       → matched (strength 0.85)
    • user has PARENT of j (general knowledge)   → matched (strength 0.70)
    • user has SIBLING of j (related domain)     → matched (strength 0.50)
    • paths longer than 2 hops: not a match (noise)

  Stage 3 — Semantic re-ranker
    • Cosine similarity of (u_embedding, j_embedding) ≥ 0.75
      → matched (strength = cosine_sim)
    • Below threshold: not a match
```

**The match_strength feeds into the weighted base score**, replacing the
current binary "matched / missing" booleans. The 0.8 / 0.2 required-vs-
preferred weights stay; what changes is that a stage-2 or stage-3 match
contributes less than a stage-1 match to the ratio. Missing remains
"strength 0".

**Why semantic stays as a re-ranker, not a primary scorer:**

1. **Auditability.** Every "because" clause the responder narrates needs
   a deterministic citation. Saying "you match because cosine similarity
   was 0.81" is not a story a newcomer can act on. We narrate from the
   deterministic stages (1 + 2) when present, and from a curated phrasing
   ("your X overlaps with their Y") for stage-3 hits.
2. **Cost / latency.** Embedding every JD skill × every user skill at
   chat time is fine for small datasets but doesn't scale. Pre-embed at
   ingest time (once per skill phrase, cached); only re-embed on prompt
   changes.
3. **Failure mode containment.** A miscalibrated embedding model that
   thinks "Python" matches "Marketing principles" should not be able to
   produce a strong match. By keeping it as a re-ranker (only kicks in
   when stages 1+2 found nothing), the deterministic pipeline still
   defines the band; embeddings only add transferable-skill credit.

**Embedding model choice (recommendation, not commitment):**

- A small sentence-transformer model (e.g. `all-MiniLM-L6-v2`, ~80MB,
  runs on CPU in ~5ms per phrase).
- Embed each skill phrase once at ingest; store the 384-dim vector
  alongside the extracted skill row.
- pgvector if we already have it (we do — `CREATE EXTENSION pgvector`
  shows up in schema.sql). Use `vector` column type, IVFFlat index for
  ANN lookup if/when corpus grows.

**Why this shape and not the bigger algorithm from earlier:**

The full algorithm (mandatory tier with hard gating, calibration model,
min_level / min_years extraction) needs data substrates we don't have:
labeled outcomes, level annotations, time-since-use data. Building those
is a 6-12 month project. The hybrid above is achievable in 2-3 weeks
and unlocks the largest class of false negatives. We can graduate to
mandatory tiers and calibration when the data justifies it.

---

## 5. What we explicitly don't build yet

- **Level / years / recency extraction.** Requires either a labeled
  training set (which we don't have) or LLM-based extraction of
  quantitative phrases like "5+ years of Python" — feasible but
  expensive in tokens, and the JD-side data is often vague ("strong
  Python skills"). Defer until we have a real failure case that the
  current binary matched/missing can't address.
- **Mandatory tier with hard rejection.** Today's credential cap covers
  the most important case (a missing licence demotes band to stretch).
  A general mandatory tier would mostly add complexity without changing
  what we surface. Defer.
- **Calibration model on top of FinalScore.** Requires hire / interview
  outcome data per match. SkillBridge has zero outcome data right now
  and the academic grant scope doesn't include collection. Defer
  indefinitely; revisit if NORDIK or SCCC starts logging placement
  outcomes.
- **Bonus for adjacent skills not in the JD.** Risk of resume-stuffing
  rewards outweighs the benefit at MVP scale. Defer.
- **Replacing `reference.skill` outright with OaSIS.** Migration risk
  is high; better to augment — add OaSIS canonical IDs as a new column,
  let the resolver prefer them, deprecate the old IDs over a release
  cycle.

---

## 6. Recommended pickup order

Each piece is independently shippable and reviewable.

1. **Occupation taxonomy import (OaSIS + SCT occupation/title only).**
   ~1-1.5 days. Imports OaSIS `example-titles_oasis_2025.csv` and SCT
   `Alternative Titles 2023.csv` from open.canada.ca. Adds
   `reference.occupation` (NOC 2021 codes + canonical titles in EN/FR
   + lead statement) and `reference.occupation_title_synonym` (synonym
   string, language, NOC, source). Adds `core.job_posting.noc_2021_code`
   column (the existing `noc_code` stays untouched for compat). No
   matcher changes yet; no skill-side import yet. Records the imported
   OaSIS / SCT release in `pipeline.dataset_state`. Validation: SQL
   query showing J occupations and S title synonyms, broken down by
   language.

2. **Occupation normalization (§3).** ~2-3 days. Title → NOC resolver
   on both the JD ingest path (populates
   `core.job_posting.noc_2021_code` from the title via lookup against
   `reference.occupation_title_synonym`, falling back to a small LLM
   call when ambiguous) and the chat path (sets `staged.target_noc`
   from `staged.target_role_text`). Engine adds an occupation-match
   boost (NOC equal → boost; no match → zero). Does NOT filter the
   candidate pool. Validation: new fixtures pinning the "Software
   Developer" ↔ "Application Programmer" cluster on a shared NOC.
   Confirm that adjacent roles (e.g., IT support) still surface for a
   user typing "software developer", just with lower ranking.

3. **Soft-skill descriptor import (decision after Step 2 ships).**
   ~1-2 days, contingent. Once occupation normalization is live and
   we've seen 2-3 chats with it, decide whether the OaSIS skill
   descriptor framework (the ~240 broad O*NET-style competencies) is
   worth importing. Likely yes for canonicalizing soft skills
   ("communication", "problem solving"), likely no for our trades /
   tech vocab. If yes, this step adds `reference.skill_descriptor`
   and a separate resolver. If no, skip. **The alias map + semantic
   re-ranker continue handling tech/trades vocab regardless.**

4. **Hierarchy-aware match strength.** ~2-3 days. Stage 2 of §4.
   `_score_one_job` produces `match_strength` per skill instead of
   binary matched/missing. `_weighted_skill_base` consumes ratios of
   summed strengths, not counts. Backwards-compat path: legacy
   strength=1.0 when stage 1 fires (preserves v1.2 numbers exactly).
   Validation: at least three new fixtures showing child-beats-parent
   asymmetry against real OaSIS data.

5. **Semantic re-ranker.** ~3-5 days. Stage 3 of §4. Embed at ingest,
   query at match time, fold into match_strength when stages 1-2 miss.
   Validation: a fixture pinning the "diesel engine repair" ↔ "diesel
   engine diagnosis and repair" case at strength ≥ 0.75.

6. **Responder updates.** ~½ day. Update the prompt so the responder
   can narrate stage-2 hits ("your React lines up with their JS
   framework requirement"), stage-3 hits ("your X reads as a close
   relative of their Y, though not identical"), and occupation-cluster
   summaries ("I found 3 software-developer roles across 3 employers").
   This is the only step that touches prompts.py.

Stop-and-review checkpoints after **each** numbered item. Don't bundle.

---

## 7. Open questions

- **Keep `reference.skill` table or replace?** Recommend keep + augment.
  Migration risk dominates the elegance gain.
- **Bilingual responder support?** OaSIS gives us the lexicon for free.
  Worth a separate scope decision; out of band here.
- **Why invest in OaSIS now vs. wait for more user data?** Because the
  deterministic pipeline is already saturated. Adding more hand-curated
  aliases is diminishing returns; the alias map only catches what we've
  already observed fail. OaSIS is the next form of leverage that doesn't
  require us to collect data we don't have.
- **Cost of embeddings at SkillBridge scale?** Tiny — 35 JDs × ~12 skills
  each = ~420 embedding rows, recomputed only on re-extract. Even with a
  larger corpus, this is a negligible compute layer.
- **Occupation match: boost or filter?** Recommend boost. SSM is a small
  market and filtering on NOC would over-narrow ("no software developer
  roles" when there's an adjacent IT support role the user could plausibly
  take). Boost preserves the candidate pool; ranking reflects role fit.
- **How to handle JDs whose title doesn't resolve to a NOC at all?** Fall
  through to today's title-token similarity. NOC resolution is best-effort;
  a NULL `noc_code` keeps the row scored on skills + title-token only, same
  as today.

---

## Decision required

Sign off on the **direction**: deterministic core + OaSIS for occupation
normalization (primary win) + alias map and semantic re-ranker for
skill-side normalization (because OaSIS doesn't ship a granular skill
taxonomy). Not the architecture. Each numbered step above is a separate
sprint slice with its own scope, sign-off, and tests.

The biggest calls in this doc:

1. **Invest in OaSIS now vs. continuing to hand-curate aliases.** Answer:
   yes, now, but scoped tightly to the **occupation layer** (where OaSIS
   has authoritative data). The skill-side OaSIS work is a separate,
   contingent decision after Step 2 ships.
2. **Use OaSIS for occupation normalization (NOC 2021).** Answer: yes
   — different-title-same-role is a real failure mode in the SCCC
   dataset today (two "Truck and Coach Technician" postings; potentially
   more variation as the dataset grows).
3. **Occupation match is a boost, not a filter.** SSM is too small to
   filter aggressively.
4. **Skill-side OaSIS is partial-fit, not full-fit.** The skill
   descriptor framework is O*NET-style broad competencies, not granular
   tech / trades vocabulary. The alias map stays as the canonical layer
   for tech/trades; the semantic re-ranker (§4 Stage 3) handles
   non-canonical cases. Step 3 in the pickup order is contingent and
   re-evaluated after the occupation win lands.
