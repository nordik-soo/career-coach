# Chat Orchestration v2 — Design Direction

Status: draft · Target: next sprint (before Resume Extraction v2)

Matching v2 gave the system a stronger brain. The chat now grounds every
"because" in `score_explanation`, caps narration honestly, and refuses
to invent. But it still **sounds** form-shaped: "Before I show you
what's out there...", "I hear you, but before we find the right
match...", "Can you walk me through your previous jobs?"

Those aren't prose problems — they're state-machine transitions
bleeding into the response. Today's pipeline is two-layer:

```
state machine decides slot  ──→  responder writes
```

The state machine's `missing experience_text → ask experience_text`
rule is what produces "Can you walk me through your previous jobs?"
even when the user just uploaded a resume and said "same role." No
prompt edit can fully hide that.

This document scopes the next sprint: insert a **conversation planner**
between the state machine (now a truth layer) and the responder, so
the conversational *move* is an LLM decision while the *truth* stays
deterministic. The goal is ChatGPT-style timing — not unconstrained
chat, but a flow that remembers context, doesn't repeat itself,
proceeds when there's enough to proceed, and asks only when asking
materially changes the outcome.

---

## 1. Where the chat is today

```
user message ─→ extract slot updates ─→ intake_state.decide()
                                              │
                                              ▼
                                    Decision(action, ask_slot)
                                              │
                                              ▼
                                    responder LLM writes prose
```

The Decision is deterministic and slot-driven. Strengths: predictable,
testable, never invents. Weaknesses: every transition is visible in
the prose. The bot can't proceed to match unless slots are filled, and
it can't read user impatience ("same role", "see my CV", "just show
jobs") because that signal isn't in any slot.

Live test 3 (electrical journeyman resume) demonstrated it cleanly:
the user said "same role" after uploading a CV with skills + work
history; the bot replied "Before we find the right match, I need to
get a clearer picture of your work background." That's the state
machine talking, not a conversation.

---

## 2. The three-layer architecture

```
                ┌──────────────────────────────────────────┐
                │              TRUTH LAYER                 │
                │  (deterministic, current intake_state    │
                │   + new compact summary object)          │
                │                                          │
                │  resume_parse_quality, target_role,      │
                │  enough_to_match, missing_critical,      │
                │  match_count, last_user_intent, caps,    │
                │  scope_violations                        │
                └────────────────┬─────────────────────────┘
                                 │  truth_summary (JSON)
                                 ▼
                ┌──────────────────────────────────────────┐
                │      3 DETERMINISTIC GATES               │
                │  (skip planner only for non-decisions)   │
                │                                          │
                │  • First-turn greeting                   │
                │  • Resume just uploaded -> review flow   │
                │  • Empty / whitespace input -> re-prompt │
                └────────────────┬─────────────────────────┘
                                 │  (90%+ of turns flow past)
                                 ▼
                ┌──────────────────────────────────────────┐
                │            PLANNER LAYER                 │
                │  (Haiku, JSON-only, tight prompt)        │
                │                                          │
                │  Returns: move + reason_code + ask_slot  │
                │           + tone (all closed enums)      │
                │  Output: ~60-100 tokens                  │
                │  Target latency: < 400ms                 │
                └────────────────┬─────────────────────────┘
                                 │  planner_decision
                                 ▼
                ┌──────────────────────────────────────────┐
                │   DETERMINISTIC ARBITER -- PASS 1        │
                │  (validates planner's intent move)       │
                │                                          │
                │  • Unknown move? -> legacy fallback      │
                │  • ask_slot already filled?              │
                │  • Planner emitted present_matches?      │
                │    -> override to proceed_to_match       │
                │  • All SCOPE_BOUNDARIES still apply      │
                └────────────────┬─────────────────────────┘
                                 │  validated intent
                                 ▼
                ┌──────────────────────────────────────────┐
                │   DETERMINISTIC ARBITER -- PASS 2        │
                │  (only if intent == proceed_to_match)    │
                │                                          │
                │  Run match engine, then resolve:         │
                │  • match_count > 0  -> present_matches   │
                │  • match_count == 0 -> present_no_match  │
                │  • caps_applied -> set tone, force cap   │
                │    narration in responder                │
                └────────────────┬─────────────────────────┘
                                 │  final outcome move
                                 ▼
                ┌──────────────────────────────────────────┐
                │         NATURAL RESPONDER                │
                │  (Haiku, prose, existing grounding rules)│
                │                                          │
                │  Inputs: outcome move + truth + tone hints │
                │  Output: 1-4 sentences, natural flow,    │
                │          unchanged scope-boundary rules  │
                │                                          │
                │  NEVER sees proceed_to_match -- the      │
                │  arbiter always resolves it to an        │
                │  outcome (present_matches or             │
                │  present_no_match) before reaching here. │
                └──────────────────────────────────────────┘
```

The state machine doesn't go away — it becomes the **truth layer**. It
still computes slot fills, intake state, declined slots. It just
doesn't decide the next conversational move.

---

## 3. The truth summary object

A compact dict the planner consumes per turn. Built deterministically
from the existing staged profile + decision context. No prose.

```json
{
  "user_message": "same role",
  "last_assistant_move": "ask_role_question",
  "last_asked_slot": "target_role_text",

  "resume_uploaded": true,
  "resume_parse_quality": "skills_only",
  "resume_facts_summary": {
    "skill_count": 24,
    "work_history_count": 0,
    "certifications_count": 1,
    "education_count": 0
  },

  "target_role_text": "electrical journeyman",
  "target_role_specificity": "specific",
  "work_type_preference": null,

  "enough_to_match": true,
  "enough_to_match_reason": "resume_skills_plus_target_role",
  "missing_critical": [],

  "match_count": 0,
  "best_match_band": null,
  "caps_applied": [],

  "user_intent_signal": "impatient_proceed",
  "scope_violations_detected": []
}
```

**Critical fields:**

- `enough_to_match`: **deterministic boolean**, computed in the truth
  layer, not inferred by the planner. Logic:

  ```
  enough_to_match = (
      target_role_text is not None
      AND usable_evidence_present
      AND (
          resume_skill_count >= 5
          OR chat_skill_count >= 3
          OR work_history_count >= 1
          OR user_explicitly_asked_to_match
      )
  )
  ```

  Where `usable_evidence_present` is:

  ```
  usable_evidence_present = (
      resume_parse_quality not in {"failed", "no_resume"}
      OR chat_skill_count >= 3
  )
  ```

  This guards against a real failure mode: a failed-scan resume + an
  impatient user ("see my CV") + a target role would otherwise mark
  enough_to_match true with no actual evidence behind it. The planner
  would then say `proceed_to_match`, the arbiter would run an empty
  profile against the engine, and the chat would surface either zero
  matches or (worse) noisy matches against thin profiles. Tuning
  happens in Python, not in the LLM's head.
- `user_intent_signal`: closed enum (`asking_question`,
  `impatient_proceed`, `declining`, `confirming`, `correcting`,
  `redirecting`). Computed from a small intent classifier (regex +
  short keyword match against the user message — same alias-map
  discipline; no LLM call).
- `resume_parse_quality`: closed enum (`full`, `skills_only`,
  `work_only`, `partial`, `failed`, `no_resume`). Surfaces the
  extraction warning shape from Resume Extraction v2 once it ships;
  pre-v2, defaults to `partial` when any fact group is empty.

The truth layer is where existing Sprint 5 invariants stay enforced
(SSM-only, no national feeds, evidence-bound). Planner consumes; it
never overrides.

---

## 4. The three deterministic gates

These exist because they aren't routing decisions — they're either
state transitions or "no input to route."

**Evaluation order matters.** Gates are checked in this strict order;
the first match wins and short-circuits the rest. The order is chosen
so that compound turns (e.g. first turn AND resume upload AND empty
text) route to the correct gate without ambiguity:

| # | Gate | Trigger | Action |
|---|---|---|---|
| 1 | **Empty / whitespace input** | `user_message.strip() == ""` AND no file upload this turn | Re-prompt with a soft "Tell me a bit about what you're looking for." No decision to make |
| 2 | **Resume just uploaded** | This turn included a file upload | Fixed UX moment: parse resume → show facts → ask "did I get this right?". Always. Wins over greeting because a first-turn resume upload should NOT be greeted first then asked to re-confirm — show what we read |
| 3 | **First-turn greeting** | `message_count == 0` AND `_normalize_for_greeting_match(user_message)` ∈ `_GREETING_PHRASES` AND not caught above (no upload, non-empty text) | Fixed welcome message, ask for goal. No context for the planner to plan around |
|   | (else) | | Pass through to planner |

**First-turn job intent is not a gate; it passes to the planner.** A
user opening the chat with "I'm looking for warehouse work" or "truck
and coach technician" must NOT fire the greeting gate — that would
short-circuit real routable content into the canned "What kind of
work are you looking for?" reply, which is the exact regression v2
exists to remove. The content guard on Gate 3 (whitelist match
against `_GREETING_PHRASES` after normalization) is load-bearing:
substantive first-turn messages fall through to the planner, where
the truth summary + LLM call decide whether to ask, proceed, or
redirect. The whitelist contains zero job-domain words by construction,
so the bug cannot reappear without a code change visible in review.

**That's it. Three. No more.**

Discipline: if you find yourself adding a fourth gate, ask hard whether
it's actually a routing decision in disguise. "User said 'alright'
after the assistant asked a confirmation" is NOT a gate — it's a
routing decision the planner can make naturally (and might choose
`proceed_to_match` instead of yet-another-acknowledge).

The whole point of orchestration v2 is that the planner is good enough
to handle the "small" cases naturally. If we keep adding shortcuts
because we don't trust the planner, we've reinvented the state
machine.

---

## 5. The planner layer

Haiku call with a tight prompt. Inputs: the truth summary above. Output:
JSON, one object, ~60-100 tokens.

```json
{
  "move": "proceed_to_match",
  "reason_code": "resume_confirmed_target_same_role",
  "ask_slot": null,
  "tone": "brief_confident"
}
```

**Move taxonomy (closed enum, 9 values) — split into INTENT moves
and OUTCOME moves:**

INTENT moves are what the **planner emits**. They describe what the
system should DO next. Some intent moves don't reach the responder
directly — the arbiter resolves them into an outcome first (see §6).

OUTCOME moves are what the **responder narrates**. They describe what
the user will see.

| Move | Kind | When |
|---|---|---|
| `acknowledge_and_continue` | outcome | User confirmed; reflect briefly and move forward |
| `confirm_resume_summary` | outcome | First turn after upload (also handled by gate; here for completeness) |
| `proceed_to_match` | **intent** | Enough to match; planner says go. Arbiter runs the engine then resolves to `present_matches` or `present_no_match` based on actual match_count. **Never reaches the responder directly.** |
| `ask_one_clarifying_question` | outcome | Need exactly one thing to make matches actionable |
| `present_matches` | outcome | Matches computed AND match_count > 0; narrate them |
| `present_no_match` | outcome | Matches computed AND match_count == 0; honest dataset-first response |
| `explain_gap` | outcome | User asked about a specific cap / missing skill |
| `offer_refinement` | outcome | After matches, user wants to narrow/broaden |
| `redirect_scope` | outcome | User went off-scope; gentle redirect |

**Reason code (closed enum, ~20 values):** documents *why* the planner
picked the move. Examples: `resume_uploaded_target_known`,
`user_explicitly_asked_for_jobs`, `missing_work_type_preference`,
`zero_matches_in_dataset`, `credential_gap_present`,
`user_impatience_signal`, `target_role_unclear`. Closed enum so the
arbiter and tests can switch on it.

**Ask slot (closed enum or null):** which slot to ask about, if any.
Must be a known intake slot name. Null when `move != ask_one_clarifying_question`.

**Tone hint (closed enum, 4 values):** `brief_confident`, `warm_supportive`,
`honest_redirect`, `excited_share`. Plain enums; responder maps these
to phrasing guidance.

**Why closed enums everywhere:**

- LLM is well-suited to picking from a small fixed set
- Arbiter can switch deterministically on the value
- Transcript tests have a stable assertion surface
- No "informational LLM output" creeps in as load-bearing

**Cost / latency budget:**

- Prompt: < 500 tokens (truth summary + move/reason taxonomies)
- Output: < 100 tokens (JSON only)
- No chain-of-thought
- Cached system prompt (Anthropic prompt caching)
- Target latency: < 400ms
- Marginal cost per turn: ~$0.0001 (Haiku tier)

Net per-session: pennies. The optimization that matters is keeping the
prompt and output tight, not skipping planner calls.

---

## 6. The deterministic arbiter

The planner output is **advisory**. The arbiter has final say. Same
defense-in-depth pattern as matching v2's max-wins rule. The arbiter
runs in **two passes**: first to validate the planner's intent move,
then (only if the intent was `proceed_to_match`) to resolve it into
an outcome move based on the actual engine result.

### 6.1 Pass 1 — validate the planner's move

| Rule | Violation → |
|---|---|
| Move must be a known enum value | Fallback to legacy intake_state.decide() |
| If `move == ask_one_clarifying_question`, `ask_slot` must be set AND not already in `staged.filled_slots()` | Drop to `proceed_to_match` if `enough_to_match`, else fallback to existing intake-priority slot |
| If `move == redirect_scope`, truth must have `scope_violations_detected` non-empty | Override to whatever truth actually supports |
| If `move == present_matches` or `present_no_match` directly from the planner | Override to `proceed_to_match` and route to pass 2 — planner doesn't get to skip the engine run |
| Planner output failed JSON parse / timed out | Fallback: today's intake_state.decide() decision |

After pass 1, the arbiter holds a validated **intent** (or has already
fallen back). If the intent is anything other than `proceed_to_match`,
skip pass 2; the move is already an outcome.

### 6.2 Pass 2 — resolve `proceed_to_match` to an outcome

When pass 1 settles on `proceed_to_match`:

```
1. Arbiter runs the match engine (compute_matches_in_memory)
2. Inspect match_count + caps_applied + band of top result
3. Resolve to one of:
     - present_matches    (match_count > 0)
     - present_no_match   (match_count == 0)
4. Pass-2-specific guards:
     - If caps_applied non-empty AND outcome == present_matches,
       set tone hint to honest_redirect; force responder to name
       the cap (matching v2 step 6 CAPS APPLIED rule still applies)
     - Engine output's SSM-only / scope invariants are already
       enforced upstream; arbiter doesn't re-check
```

The responder NEVER sees `proceed_to_match`. By the time the move
reaches the responder, it's always an **outcome**.

This separation matters because the planner can't predict
match_count. The planner's job is to decide intent ("we have enough,
go match"); the engine determines the outcome ("here are the
matches" vs "no current match"). Splitting these makes the planner
small and stable — it doesn't need to peek at match results to pick
the right narration mode.

### 6.3 Fallback path

If anything in pass 1 or pass 2 fails (planner JSON malformed, engine
errors, unexpected state), the arbiter falls back to the existing
`intake_state.decide()` path. System degrades to today's behavior,
not to crash. Same graceful-degradation pattern as sentence-transformers
/ pgvector missing.

The arbiter also enforces all Sprint 5 SCOPE_BOUNDARIES invariants —
those don't move into the planner; they stay deterministic guards.

---

## 7. The natural responder

The responder layer's prompt updates lightly:

1. Takes `final_move` from the arbiter (not a raw `Decision` object)
2. Takes `tone` from the arbiter
3. Keeps all existing SCOPE_BOUNDARIES, MATCH STAGES, CAPS APPLIED,
   DATASET-FIRST rules — those are still load-bearing
4. Knows the move's narration shape (e.g., `proceed_to_match` →
   "Okay, looking at..." vs `acknowledge_and_continue` → "Got it. ...")
5. Doesn't see raw state-machine flags. It writes from the move +
   truth, not from "ASK_QUESTIONS with slot=experience_text."

The grounding rules from Step 6 stay verbatim. The responder is still
not allowed to invent. The only change is that it's writing from a
**move semantic** instead of a slot-fill state.

---

## 8. Transcript regression tests

This is the gate. Same shape as `test_matching_fixtures.py`.

Each test is a **scripted turn sequence**. We assert against
**final-move sequences** (the outcome that reaches the responder),
not planner moves directly. This matters because some turns route
through a deterministic gate and never see the planner, and because
the arbiter resolves `proceed_to_match` into `present_matches` or
`present_no_match` before the responder sees it.

```python
@pytest.mark.parametrize("scenario", [
    {
        "name": "same_role_after_resume_upload_with_enough_evidence",
        "turns": [
            {"role": "user", "message": "[upload electrical CV]"},
            {"role": "user", "message": "alright"},
            {"role": "user", "message": "same role"},
        ],
        "expected_final_move_sequence": [
            "confirm_resume_summary",      # gate (resume upload), planner skipped
            "acknowledge_and_continue",    # planner
            "present_no_match",            # planner said proceed_to_match;
                                            # arbiter pass 2 resolved to
                                            # present_no_match (electrical
                                            # not in current SSM dataset)
        ],
        "expected_responder_NOT_to_contain": [
            "Can you walk me through your previous jobs",
            "Before we find the right match",
            "I hear you, but",
        ],
    },
    ...
])
```

Test scenarios to lock in v1:

- `same_role_after_resume_upload_with_enough_evidence` (electrical CV)
- `see_my_cv_impatience_signal` (user explicitly says "see my CV")
- `alright_after_resume_review` (one-word continuation)
- `software_developer_no_match` (Nazmul's CV, full-time)
- `truck_credential_gap` (Michael's CV, 310T missing)
- `user_changes_direction` ("actually I want X instead")
- `user_asks_about_specific_job` (drill-down after present_matches)
- `redirect_off_scope` (user asks about wages, immigration, etc.)

The asserted-NOT-to-contain list is critical: it locks in that the
robotic phrases from today's chat **cannot return** without a test
failure.

---

## 9. What we don't build yet

| Item | Reason |
|---|---|
| **Multi-turn conversational memory beyond `conversation_context`** | The truth summary already carries last move + last asked slot. Adding richer memory invites scope creep |
| **Agent loops / tool use from the planner** | Planner returns one move per turn. No tools, no chains. Adds determinism, removes latency |
| **Streaming responder output** | Nice-to-have UX; orthogonal to the orchestration design |
| **Voice / TTS mode** | Out of scope |
| **Multi-language planner** | Planner output is enum-only; works regardless of UI language. Multilingual responder is a separate sprint (same as matching v2) |
| **A/B testing planner vs. legacy state machine** | Env flag `CHAT_ORCHESTRATOR=v1\|v2` exists for rollback, not for A/B. Production sticks to one version per release |

---

## 10. Recommended pickup order

Same shape as matching v2 + resume v2: independently shippable slices
with sign-off between each.

1. **Truth summary object + schema** (`skillbridge/chat/truth_summary.py`).
   ~1 day. Deterministic builder that reads staged + decision context
   and produces the JSON. Unit tests against scripted staged profiles.
   No LLM, no chat behavior change yet — this is just the data shape.

2. **Three deterministic gates** (`skillbridge/chat/gates.py`).
   ~½ day. Pure functions: `is_first_turn`, `is_resume_upload_turn`,
   `is_empty_input`. Each returns a fixed move or None (pass through
   to planner). Unit tests for boundary cases.

3. **Planner LLM call + JSON schema validation**
   (`skillbridge/chat/planner.py`). ~1-2 days. Tight prompt with move
   + reason taxonomies inlined. Pydantic schema for output validation.
   Fallback path: JSON parse fail → return None, arbiter falls back to
   legacy state-machine decision.

4. **Deterministic arbiter** (`skillbridge/chat/arbiter.py`).
   ~1 day. All 5 rules from §6 enforced. Pure function: takes planner
   output + truth summary, returns final_move. Lots of unit tests for
   override paths.

5. **Responder prompt update** (`skillbridge/chat/prompts.py`).
   ~½ day. Updates `NEXT_ACTION_RESPONDER_PROMPT` to consume
   move + tone instead of raw state-machine action. SCOPE_BOUNDARIES
   stay verbatim. Prompt-content regression tests guard the rules.

6. **Handler integration** (`skillbridge/chat/handler.py`).
   ~1 day. Wires truth_summary → gates → planner → arbiter → responder.
   Env flag `CHAT_ORCHESTRATOR=v2` defaults true. Legacy path stays
   available for rollback for one release cycle.

7. **Transcript regression suite** (`tests/test_chat_transcripts.py`).
   ~2 days. The 8 scripted scenarios from §8. Mocks the planner LLM
   with deterministic responses for unit tests; uses real Haiku for
   integration tests gated behind a manual flag.

Total: ~7-8 days. Stop-and-review checkpoints after each numbered
item. Don't bundle.

---

## 11. Open questions

- **Should the planner see RESUME_FACTS verbatim or just the summary?**
  Recommend summary only (`skill_count`, `work_history_count`). The
  verbatim facts are for the responder; the planner just needs to
  know "we have skills" or "we don't."

- **User-intent classification: regex vs. tiny LLM?** Recommend regex
  + keyword list for v1 (same alias-map discipline). Upgrade to a
  small classifier only if real chats show drift. The intent enum
  is short (~6 values).

- **What's the response when arbiter falls back?** Recommend: legacy
  state-machine path. Same UX as today. Better than a generic error.

- **Multi-turn planner context — how much history?** Recommend: just
  `last_assistant_move` + `last_asked_slot`. Adding the full
  transcript invites scope creep and unbounded token cost. If the
  planner needs more context, that's a signal the truth summary is
  under-specified.

- **When does the planner skip a turn entirely?** Never inside a
  user-message processing flow. Three gates handle skip cases; the
  planner is the default for everything else.

---

## Decision required

Sign off on the **direction**: planner-by-default with 3 minimal
deterministic gates + deterministic arbiter + tight planner config.
Not the architecture. Each numbered step is its own slice with
sign-off and tests.

The biggest calls in this doc:

1. **Planner is the default, not the exception.** Defending against
   "deterministic creep" is a load-bearing design choice — if the
   gate list grows past 3, the system reverts to today's rigidity.
   Answer: yes — discipline matters more than incremental cost
   savings.

2. **Planner output is advisory, arbiter has final say.** Same
   defense-in-depth pattern as matching v2's max-wins. Answer: yes —
   LLM-as-router fails quietly; deterministic guardrails are how we
   catch it.

3. **Closed enums everywhere in planner output.** Move, reason_code,
   ask_slot, tone all bounded. Answer: yes — testability and
   stability over expressiveness.

4. **Build orchestration v2 BEFORE resume extraction v2.** Same
   reasoning as the conversation: user pain is visible in the
   conversation flow first. Answer: yes — even with imperfect
   extraction, orchestration v2 makes the chat feel materially
   better.
