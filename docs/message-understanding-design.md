# Message Understanding + Deterministic Router — Design

Status: design draft · 2026-06-04 · pre-build, requires review

## Why this document exists

Three live-test failures in three slices, all the same architectural shape:
the planner (Haiku) ignored its grounding rules and emitted a move the
truth summary did not support.

| Incident | Truth said | Haiku emitted | Patch |
|---|---|---|---|
| TAC hallucination | training response expected | invented providers | wired TRAINING block, arbiter override |
| Class G redirect | scope=[], intent=asking_about_gap | `redirect_scope` (invented) | arbiter override Rule 3 |
| Excel course question | intent=asking_about_gap, registry=[Excel] | `ask_one_clarifying_question` | (proposed: another arbiter override) |

Each individual patch was correct. The pattern is not. We are using the
LLM planner to make deterministic decisions and patching the arbiter
each time it overrides a rule.

**The product behavior we want is testable, repeatable, and explainable.**
The LLM stays in the system for what it is good at: writing natural
prose, handling genuine conversational ambiguity, picking tone.

This doc designs the layer that takes the high-risk routing decisions
away from Haiku and gives them to deterministic Python.

## Decision summary (for reviewers)

- **Build a new `message_understanding` module** that consolidates the
  scattered classifiers (intent regex, scope regex, registry scan)
  into one structured output.
- **Build a new `routing` module** (alongside arbiter) that consumes
  the understanding and decides high-confidence moves deterministically,
  bypassing the planner.
- **Planner is NOT killed.** It still runs for ambiguous turns — that's
  the architectural contract: LLM for natural conversation, deterministic
  router for safety-critical/structured routing.
- **Flag-gated rollout** behind `MESSAGE_UNDERSTANDING_ENABLED` (default
  false initially) using the same pattern as `TRAINING_REGISTRY_ENABLED`.
  Flip default once live-tested.
- **Two existing arbiter overrides become dead code** (scope-overreach
  Rule 3, and the proposed Excel-case override). Removed in a later slice.

## Architecture

### Current flow (chat-orchestration-v2 + post-Slice-9)

```
user message
    │
    ▼
build_truth_summary()  ── computes intent, scope, registry_gaps, ...
    │
    ▼
gates  ── short-circuit for greeting/empty/upload
    │
    ▼
plan_next_move()  ── HAIKU CALL ── decides the move
    │
    ▼
validate_planner_intent()  ── arbiter pass 1, overrides if needed
    │
    ▼  RunEngine? → engine → resolve_match_outcome (pass 2)
    │
    ▼
compose_response_v2()  ── HAIKU CALL ── narrates
```

The planner is asked to make EVERY routing decision. The arbiter catches
its mistakes after the fact via override rules.

### Proposed flow

```
user message
    │
    ▼
build_truth_summary()  ── unchanged: profile/state computation
    │
    ▼
understand_message()  ── NEW: consolidates classifiers
    │     returns MessageUnderstanding with primary_intent + confidence
    │
    ▼
gates  ── unchanged
    │
    ▼
route_from_understanding()  ── NEW deterministic router
    │
    ├─ HIGH confidence ──→ DECISION emitted directly, planner SKIPPED
    │
    └─ MEDIUM/LOW confidence ──→ plan_next_move() ── HAIKU CALL (as today)
    │
    ▼
validate_planner_intent()  ── arbiter pass 1, unchanged for medium/low;
    │                          high-confidence cases bypass entirely
    ▼
RunEngine → engine → pass 2 → compose_response_v2  ── unchanged
```

The arbiter's high-priority safety rules (scope override on real
violation, proceed-to-match independent recheck) STAY. The planner-
overreach overrides become unreachable and are deleted in a follow-up.

## `MessageUnderstanding` shape

```python
from dataclasses import dataclass
from typing import Literal


PrimaryIntent = Literal[
    "scope_violation",       # immigration, national wages, non-local city
    "training_request",      # asking about credential/skill training
    "gap_explanation",       # asking why X is a gap (post-match drilldown)
    "job_search",            # impatient_proceed, "show me jobs"
    "confirmation",          # "yes", "alright", "looks right"
    "decline",               # "no", "skip", "not now"
    "correction",            # "actually", "I meant"
    "ambiguous",             # router can't decide; planner handles
]


Confidence = Literal["high", "medium", "low"]


EntityType = Literal[
    "registry_gap",          # canonical credential/skill from training_registry
    "scope_keyword",         # PR, work permit, Express Entry, etc.
]


@dataclass(frozen=True)
class DetectedEntity:
    type: str                # EntityType value
    canonical_name: str      # "Microsoft Excel", "immigration", etc.
    matched_text: str        # the actual substring that matched in user_message
    source: str              # "registry_gap_alias" | "scope_pattern" | etc.


@dataclass(frozen=True)
class MessageUnderstanding:
    primary_intent: str      # PrimaryIntent value
    confidence: str          # Confidence value
    entities: tuple[DetectedEntity, ...]
    reason: str              # human-readable "why this classification fired"
                              # for logs + transcript test debugging
```

### Confidence definitions

| Level | Meaning | Routing |
|---|---|---|
| `high` | Deterministic match with strong signal (e.g. scope keyword present; registry entity + training-action word together; clear impatient_proceed phrase) | Router emits decision; planner SKIPPED |
| `medium` | One strong signal but mixed context (e.g. registry entity alone without training word; "any course" without entity) | Planner runs with the understanding as additional input |
| `low` | No clear signal | Planner handles fully (today's behavior) |

The 3-level scale gives the router clear "skip planner" cases (high)
while preserving Haiku for nuance (low). Medium is the "have a hint
but consult the LLM" middle ground.

## Router priority rules

The router applies these rules in order. **Earlier rules win when
multiple could fire.** Each rule has a confidence level; if it fires
at `high`, the planner is skipped; otherwise the planner is consulted
with the understanding passed in as additional context.

```
Rule 1 (HIGH): scope_violation
  entities contains any scope_keyword
  → primary_intent=scope_violation, confidence=high
  → router emits: final_move=redirect_scope
                  reason_code=<derived from scope tag>
                  tone=honest_redirect

Rule 2 (HIGH): training_request + registry_gap (BOTH present)
  intent regex matches asking_about_gap patterns
  AND entities contains at least one registry_gap
  → primary_intent=training_request, confidence=high
  → router emits: final_move=explain_gap
                  reason_code=credential_gap_present
                  tone=warm_supportive

Rule 3 (HIGH): training_request without registry_gap
  intent regex matches asking_about_gap patterns
  AND no registry_gap entity
  → primary_intent=training_request, confidence=high
  → router emits: final_move=ask_one_clarifying_question
                  reason_code=insufficient_profile_evidence
                  ask_slot=skills_text
                  tone=warm_supportive
  (User wants training but no specific gap named — ask them which.)

Rule 4 (HIGH): job_search via impatient_proceed
  intent regex matches impatient_proceed patterns
  AND truth.enough_to_match=true (already verified by truth_summary)
  AND truth.usable_evidence_present=true
  → primary_intent=job_search, confidence=high
  → router emits: final_move=proceed_to_match
                  (then existing arbiter pass 2 engine flow)

Rule 5 (MEDIUM): registry_gap entity present, no training intent
  entities contains registry_gap
  AND intent is not asking_about_gap
  → primary_intent=ambiguous, confidence=medium
  → planner runs; understanding is passed as context.
  (Could be skill claim "I have Excel" or could be soft training mention.)

Rule 6 (MEDIUM): impatient_proceed but truth doesn't support match
  intent regex matches impatient_proceed
  AND (enough_to_match=false OR usable_evidence_present=false)
  → primary_intent=job_search, confidence=medium
  → planner runs (it can pick the right ask_slot via existing logic).

Rule 7 (LOW): default
  No strong signal matched.
  → primary_intent=ambiguous, confidence=low
  → planner runs as today.
```

### What this means in practice

Several previously-buggy cases now route deterministically:

| Live message | Rule | Result |
|---|---|---|
| "Can I apply for PR while looking?" | 1 (scope) | redirect_scope, planner skipped |
| "how can I get my Class G?" | 2 (training+entity) | explain_gap, planner skipped, registry surfaces |
| "online Excel course" | 2 (training+entity) | explain_gap, planner skipped, registry surfaces |
| "where can I do course for learning excel" | 2 (training+entity) | explain_gap, planner skipped |
| "any course do you recommend" | 3 (training, no entity) | ask "which skill", planner skipped |
| "show me jobs" with profile | 4 (job_search) | proceed_to_match, planner skipped |
| "I have Excel, find me jobs" | 5 → planner | planner decides; understanding gives entity context |
| "what about that one?" | 7 → planner | planner handles ambiguous reference |
| "yes" / "alright" | 7 → planner | planner uses last_assistant_move context |

The three failure modes that triggered this refactor are all caught by
Rules 1 and 2 — high-confidence, planner not called, no opportunity to
overreach.

## What `build_truth_summary` keeps vs hands off

Truth summary stays focused on profile/state computation. Understanding
takes over message-level classification.

| Concern | Today (truth_summary) | After (split) |
|---|---|---|
| `target_role_specificity` | Yes | truth_summary |
| `resume_parse_quality` + counts | Yes | truth_summary |
| `enough_to_match` + `usable_evidence_present` | Yes | truth_summary |
| `filled_slots`, `declined_slots`, `target_role_text` | Yes | truth_summary |
| `user_intent_signal` (regex-based) | Yes | **understand_message** |
| `scope_violations_detected` (regex on message) | Yes | **understand_message** |
| `registry_gaps_in_message` | Yes (Layer 2 from training-intent slice) | **understand_message** |
| `match_count` etc. | Yes | truth_summary |

`build_truth_summary` keeps its current signature and behavior for
backward compatibility; the duplicated classification fields stay
populated for the planner's existing prompt rules. They are sourced
from understand_message internally rather than running regexes inline.

### Implementation note

`build_truth_summary` will internally call `understand_message` and
copy the relevant fields into the TruthSummary. Same external interface;
new internal source of truth. This means no callers change shape.

## What the planner work survives

The planner stays. It is called for:

- All medium / low confidence routes (Rule 5, 6, 7)
- Multi-turn conversational context that the router can't reason about
  ("alright", "what about that one", "actually I meant X")
- Tone selection on ambiguous turns

The planner's grounding-rules prompt is **simplified** because the
deterministic router pre-handles scope and training-request routing.
Rules like "scope_violations non-empty → redirect_scope" become unreachable
in production but are retained in the prompt as defense in depth.

## Which arbiter rules become dead code

Arbiter Pass 1 currently has:

| Rule | Purpose | Survives? |
|---|---|---|
| 1 — planner=None → fallback_to_legacy | LLM failure recovery | **Stays** |
| 2 — scope_violations non-empty → override to redirect_scope | Catches planner ignoring scope rule | **Dead** (router handles) |
| 3 — planner=redirect_scope + scope=[] → override (post-Class-G) | Catches planner inventing scope | **Dead** (router decides scope) |
| 4 — proceed_to_match + truth not ready → ask | Engine safety net | **Stays** |
| 5 — ask + slot strongly filled → reroute | Duplicate-ask guard | **Stays** |
| 6 — passthrough | Default | **Stays** |

Rules 2 and 3 are deleted in a later cleanup slice. Their tests are also
deleted. Rule 4 (engine safety) remains as defense in depth even though
the router won't route to proceed_to_match without verified signals.

## Test plan

### New tests

| Module | Test count (est.) | Coverage |
|---|---|---|
| `tests/test_message_understanding.py` | ~30 | Each PrimaryIntent value, each Confidence level, edge cases between rules |
| `tests/test_chat_routing.py` | ~15 | Each router rule fires correctly; planner skipped on high-confidence; planner called on medium/low; understanding passed through |

### Ported tests (existing → adapted)

| Existing | Becomes |
|---|---|
| `tests/test_truth_summary.py::test_intent_classification` (parametrized) | Adapted: classification still works via build_truth_summary, which delegates to understand_message |
| `tests/test_truth_summary.py::test_scope_violation_detection` | Same — still surfaces via truth_summary externally |
| `tests/test_training_registry.py::test_find_gaps_in_message_*` | Stays — same registry method, more callers |

### Deleted tests (architecture made them moot)

| Test | Why dead |
|---|---|
| `test_pass1_overrides_redirect_scope_to_explain_gap_when_intent_is_gap` | Router catches this, planner never emits invented redirect_scope |
| `test_pass1_overrides_redirect_scope_to_ask_when_intent_is_not_gap` | Same |
| `test_pass1_invented_redirect_scope_override_preserves_planner_tone_on_gap` | Same |
| `test_pass1_redirect_scope_with_real_scope_violation_still_redirects` | Router emits redirect on scope; arbiter Rule 1 (planner=None) still in place but never sees this case |
| The proposed-but-not-built Excel-case override | Never written |

### Live regression tests (manual, post-build)

Same scenarios that broke during the post-Slice-9 live tests:

1. Cold session: `how can I get my Class G driver's licence` → DriveTest/Ontario.ca/SCCC
2. Cold session: `online Excel course` → Microsoft Learn/Coursera/Sault College
3. Cold session: `where I can do course for learning excel` → same as #2
4. Cold session: `Can I apply for PR while looking for work?` → SCCC referral, no immigration advice
5. After matches: `how do I get my 310T?` → Skilled Trades Ontario + Sault College
6. Skill claim: `I have Excel and forklift experience, find me jobs` → matching path, not training

## Edge cases (acknowledged risks)

Honest list of things I am NOT certain the design handles. Surfacing them
in advance so they get attention before code:

1. **"any course" without entity** — Rule 3 routes to `ask_one_clarifying_question` with `skills_text` slot. But the question phrasing matters: should the bot ask "which skill?" or "what work are you looking for?" Different from the existing intake question. The responder probably needs a new fallback shape for this case.

2. **Multiple entities of different types** — "Excel AND forklift courses". Rule 2 fires; router emits explain_gap. But the explain_gap responder narrates ONE gap typically. Need to decide: explain all, narrow to first, or ask which one. Defer to a later slice or handle in the explain_gap fallback's iteration over training_by_job.

3. **Mixed scope + training**: `"can I get a forklift course for PR?"` Scope rule wins (Rule 1, priority). But the chat might benefit from acknowledging the forklift question too. Acceptable risk for v1 — the redirect is correct; a follow-up turn handles the forklift question.

4. **Multi-turn context**: `"yes"` after the bot asked `"do you want training or jobs?"` — confirmation primary_intent, planner consults `last_assistant_move`. Router doesn't have multi-turn intelligence; planner is correct here. Confirmed working as-is.

5. **Skill-claim confusion with training**: `"I want to do Excel for jobs"` — has training_action words (`do`) + entity (Excel) + job intent (`for jobs`). Where does it route? Need to think; possibly: training_request wins, suggest Excel training. Worth confirming in tests.

6. **Spell variations / typos**: `"exel course"` — Excel alias miss. Rule 2 doesn't fire. Falls to Rule 7 (planner). Planner might guess correctly. Acceptable — fuzzy matching on aliases is a separate, larger problem.

7. **Existing chat extractor**: not affected by this refactor. It still runs for slot/skill extraction after gates and before truth_summary. Its job stays as-is.

## Rollback

Same pattern as `TRAINING_REGISTRY_ENABLED`:

```python
# config.py addition
MESSAGE_UNDERSTANDING_ENABLED = _bool("MESSAGE_UNDERSTANDING_ENABLED", False)
```

When `False` (default during rollout):
- `understand_message` is NOT called
- Router does NOT run
- Existing planner-first flow runs unchanged
- All existing tests pass

When `True`:
- Understanding is computed
- Router fires per priority rules
- Planner-skip happens for high-confidence cases

The flag becomes the single rollback switch. If the live test surfaces
something the design missed, flip the flag and the chat is back to the
known-good (planner-first) path while we fix forward.

Default flips to `True` after live regression tests pass (same flow as
`CHAT_ORCHESTRATOR=v2` and `TRAINING_REGISTRY_ENABLED`).

## Build slices

Each independently shippable. Build slice → test → review → next slice.

### Slice A — `message_understanding` module + tests (~2-3 hours)

- `skillbridge/chat/message_understanding.py`: `MessageUnderstanding`, `DetectedEntity`, `understand_message()`
- `tests/test_message_understanding.py`: ~30 unit tests per primary_intent, per confidence level, edge cases
- No integration yet; module is dead code until Slice B wires it up
- Independently mergeable

### Slice B — `routing` module + arbiter integration (~2-3 hours)

- `skillbridge/chat/routing.py`: `route_from_understanding()` returning `RouterDecision | None` (None = planner needed)
- `tests/test_chat_routing.py`: ~15 tests per rule
- Handler integration behind `MESSAGE_UNDERSTANDING_ENABLED` flag (default false)
- When flag is off, handler runs current planner-first path
- When flag is on, handler calls understand_message → router → if router returns a decision, skip planner; else continue to planner

### Slice C — live test, flip flag default (~1-2 hours)

- Live test the 6 regression scenarios above
- Verify logs show planner-skip on high-confidence cases
- Verify chat quality unchanged on ambiguous cases (planner still narrating)
- Flip flag default to true
- One-line PR

### Slice D — cleanup dead arbiter rules (~30 min)

- Delete arbiter Rule 2 (scope override after planner) and Rule 3
  (planner-overreach overrides)
- Delete corresponding tests
- Tighten planner prompt: remove rules that the router pre-handles
  (or keep them as defense-in-depth — design call)

Total: ~6-8 hours across 4 slices, with review between each.

## Locked design decisions (reviewed 2026-06-04)

These were the 6 open questions; all have been signed off. Locked here
so future contributors don't reopen.

1. **Module location**: flat `skillbridge/chat/message_understanding.py`.
   Single module, single concern. Do not create a package yet.

2. **`MessageUnderstanding` as separate dataclass**, distinct from
   `TruthSummary`. TruthSummary owns profile / match-readiness state;
   MessageUnderstanding owns current-message interpretation. Do not
   mix them.

3. **`Confidence` 3-level**: `high` / `medium` / `low`. Enterprise systems
   need the middle. Binary collapses the "have a hint, still want
   planner" case.

4. **Rule 3 (training intent, no entity) gets a NEW responder phrasing**.
   The standard skill-intake question is wrong for training questions.
   Required wording shape:

   > "Sure — what skill or certificate do you want training for? For
   > example Excel, WHMIS, forklift, Class G, or 310T."

   This is a new fallback variant. Concrete examples from the registry
   help the user choose without overwhelming them. Add to the responder
   v2 fallback set, NOT a one-off prompt string.

5. **`MESSAGE_UNDERSTANDING_ENABLED` is a NEW flag**, not `CHAT_ORCHESTRATOR=v3`.
   This is v2.1 — a routing-layer change, not a whole new orchestrator.
   The two flags are independent: `CHAT_ORCHESTRATOR` controls the
   pipeline shape; `MESSAGE_UNDERSTANDING_ENABLED` controls whether
   deterministic routing pre-empts the planner.

6. **Multi-entity behavior**: emit `explain_gap` for the FIRST entity.
   Responder narrates that one and offers to cover the next:
   *"...want me to walk through WHMIS next?"* Don't try to answer five
   training questions at once. Multi-resource narration is a later slice.

### Additional locked rule

**Planner MUST NOT run for high-confidence scope-violation or
training-with-entity cases.** This is the architectural core. If the
router emits a decision at HIGH confidence for Rules 1 or 2, the planner
is bypassed entirely — no Haiku call, no chance to overreach. Verified
by integration tests that mock `plan_next_move` to fail loudly if called.

### Locked priority order (from highest to lowest)

```
1. scope_violation                       (HIGH)  → router decides, planner skipped
2. training_request + registry_entity    (HIGH)  → router decides, planner skipped
3. training_request without entity        (HIGH)  → router decides, planner skipped
4. job_search (impatient + truth ready)  (HIGH)  → router decides, planner skipped
5. registry_entity without training intent (MEDIUM) → planner consulted with hints
6. job_search, truth not ready            (MEDIUM) → planner consulted (existing ask logic)
7. ambiguous                              (LOW)   → planner handles fully (today's behavior)
```

This priority order is the single source of truth for routing. Tests
assert it exhaustively.

## What I do NOT promise

Per the discipline established this session:

- I do NOT promise this refactor catches every future planner-overreach
  case. New ones may surface; if they do, they go in the router's priority
  rules deterministically rather than becoming new arbiter overrides.
- I do NOT promise the build estimates (6-8 hours total) are accurate.
  They are best-guess from this design; if Slice A takes 4 hours, that's
  data, not failure.
- I do NOT promise the planner prompt stays unchanged. We may discover
  the planner's medium-confidence handling needs prompt clarification.

What I do commit to:

- No code until this design is reviewed and signed off
- Each slice tested in isolation, suite green between slices
- Live test after Slice C, fix-forward if needed via the flag
- Explicit "we don't know X" calls in the next design doc if I write one

## Verdict

This is the architectural fix the live-test data has been pointing to
for three slices. It is not a guess — it is the consolidation of three
overreach incidents into one structural pattern.

The cost is engineering time (one focused day). The win is the loop
ends. Planner-overreach cases stop being "one more arbiter override"
and start being "router rule already covers it." Future training
questions, scope questions, and high-confidence routing become testable
and deterministic.

Awaiting review.
