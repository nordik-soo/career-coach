# Remaining-Gaps Iteration — Design

Status: design v9 **locked** · 2026-06-08 · ready for build

Thirty-nine documentation issues from eight rounds of code review
have been integrated:

Round 1 (five architectural concerns):
1. Router contract — handler synthesizes ArbiterDecision directly,
   NOT via PlannerMove
2. Cookie-size caps on the snapshot + a serialization-size test
3. `redirect_scope` does NOT clear the snapshot (temporary diversion,
   not a topic change)
4. `_resolve_credential_anaphor` is a NEW function — pattern reused
   from `_resolve_target_role_anaphor`, not the function itself
5. "think I got X" carries epistemic uncertainty even with entity
   match — confirmation required, not subtraction

Round 2 (seven follow-up consistency issues):
6. Bootstrap path is reachable: detection accepts `snapshot=None`
   and returns a `kind="bootstrap"` typed result
7. Clarification responder contract: typed `clarification_payload`
   field + deterministic `_render_clarification` (NOT the generic
   "Tell me a bit more" line)
8. Provider grounding wired into R-4: handler explicitly populates
   `training_by_job` for the LEAD remaining credential so the
   responder can name providers verbatim
9. Cookie-size test measures the FULL signed StagedProfile (not
   just `json.dumps(snapshot)`); size-budget math corrected for
   realistic worst-case (~2.3 KB snapshot, tight against 3.3 KB
   JSON budget)
10. `last_discussed_credential_canonical` is a NEW persisted
    StagedProfile field with explicit set / clear lifecycle; the
    resolver cannot return wrong-credential answers across turns
11. All stale references swept: no remaining mentions of
    "PrimaryIntent: remaining_gaps_request", "arbiter passes through
    Pass 1", "router rule", or scope-clears-snapshot
12. Ungrounded responder claims removed (system MUST NOT prescribe
    how non-credential gaps are typically closed without verified
    TRAINING data)

Round 3 (four blocking issues + four stale references):
13. **Accumulated assumptions across turns**: NEW field
    `staged.last_assumed_completed_credentials` holds the
    union of claimed-completed canonicals over the snapshot's lifetime.
    Without this, "what else?" on a turn AFTER "assume I have 310S"
    would re-narrate 310S — defeating the feature's purpose. Field is
    conversation state (NOT profile evidence), cleared with the
    snapshot.
14. **LLM bypassed by early return in `compose_response_v2`**, not
    routed around in `_fallback_reply_v2`. The reviewer correctly
    pointed out that fallback only runs when the LLM call fails;
    clarification text MUST be deterministic on synthesis paths.
15. Cookie test uses the actual production API:
    `CookieSessionStore.save(staged)` returns the signed value
    measured against the 4 KB browser limit.
16. R-3 handler branches on `intent.kind`, NOT on snapshot existence.
    Otherwise `kind="bootstrap"` (which by definition has no snapshot)
    is unreachable, making Q4 unimplementable.

Plus stale-reference sweep: decision summary updated to remove
scope-clears-snapshot; `other_jobs` references renamed to
`other_jobs_meta` throughout; ambiguity test table rewritten to use
`kind="confirm"` instead of `ambiguous=True`; the "usually come
through the job" line removed from the worked example; multi-turn
accumulation worked example added (Turns 5b → 6 → 7 → 8 showing
persistence across turns and clearing on new match).

Round 4 (five blocking + four stale build steps + one policy doc):
17. **Hypothetical provenance via `mode` field**: accumulated
    credentials are typed records `{canonical, mode}` so a turn
    after "if I had X" can NEVER produce narration that treats X
    as actually completed. Responder gets an `any_hypothetical`
    flag and MUST use conditional tense whenever any subtraction
    is hypothetical. Without this, the system can silently launder
    a hypothetical into an apparent claim.
18. **`pending_credential_confirmation` field**: typed
    `{canonical, action}` state set when the handler emits a
    confirmation; consumed and cleared on the user's next answer.
    Affirmative "yes" / negative "no" / unrelated message are all
    handled deterministically. Without this, "yes" after a
    confirmation question can't be interpreted reliably AND
    retraction confirmations are impossible.
19. **Cookie documentation accuracy**: `resume_facts_json` is
    COMPACTED, not removed; it still costs cookie space. Test
    ceiling tightened to 3800 bytes (was 4096) to leave margin for
    Set-Cookie header attributes (Path / HttpOnly / SameSite /
    Secure) that browsers count toward the 4 KB limit.
20. **Ordered append-and-dedupe** for accumulation (not
    `list(set(...))`): preserves snapshot order on first occurrence
    so narration, cookie serialization, and tests stay
    deterministic. Cap-truncation drops latest entries first.
21. Build steps swept: R-1 clearing rules NOW exclude scope; R-2
    no longer references an "ambiguity flag" (discriminated union
    IS the contract); R-4 explicitly uses accumulated state
    (not current-turn-only); multi-turn accumulation case added
    to the 11 (was 8) live regression tests; accumulation tests
    moved from detection module to handler module (it's handler
    state, not detection state); clarification dispatch
    explicitly bypasses `_policy_ok_v2` because templates are
    trusted by construction.

Round 5 (five blocking contract issues + minor stale text):
22. **Retraction path now executes.** A NEW `kind="retract"` shape was
    added to the discriminated union, and the handler's `match
    intent.kind:` block has an explicit `case "retract"` branch that
    REMOVES the named canonical from
    `last_assumed_completed_credentials`. Without this, the
    `pending_action="remove"` → "yes" → walk-back path had nowhere to
    land; v5 documented the confirmation question but no execution.
    A second ArbiterReason (`ARBITER_REASON_REMAINING_GAPS_RETRACTED`)
    distinguishes "user added a hypothetical/claim" from "user walked
    one back" in transcripts and telemetry.
23. **Detector signature carries all four state inputs.**
    `detect_remaining_gaps_intent` now takes `accumulated_credentials`,
    `pending_confirmation`, AND `last_discussed_canonical` as required
    keyword args — not just snapshot + registry. Without these, the
    detector cannot read pending state to decide whether "yes" means
    add or remove, cannot match retraction language against the
    accumulated set, and cannot resolve "it" / "that licence" to the
    last-discussed credential. v5's signature only passed snapshot +
    registry and treated state reads as a handler concern; that
    split made the detection ordering rules unimplementable.
24. **Mixed shapes swept.** All references to the old
    `assumed_completed_canonicals` (bare canonical list, lost
    provenance) and turn-local `is_hypothetical` boolean have been
    replaced by the round-4 typed forms throughout: detection emits
    `current_turn_claims: list[{canonical, mode}]`; payload to
    responder carries per-entry `mode` plus an `any_hypothetical`
    flag derived from the union. §9 (REMAINING_GAPS block), §12
    (negation tests), §13 (ambiguity tests), the worked example
    Turn 5/5b detection blocks, Q7 telemetry, and the Tests table
    are all consistent. The doc no longer carries two competing
    payload field names.
25. **Legacy action mapping wired in R-3.** R-3 explicitly adds
    `_FINAL_MOVE_TO_LEGACY_ACTION["explain_remaining_gaps"] =
    intake_state.ACTION_PRESENT_MATCHES` so the v1 `next_action`
    label exposed to analytics / legacy clients is `PRESENT_MATCHES`
    (a match continuation), not the default `ASK_QUESTIONS`
    fallback. A handler test pins the mapping so a future refactor
    that drops it is caught in CI.
26. **Defensive cookie deserialization specified in R-1.**
    `StagedProfile.from_json` MUST validate the four new fields
    per-key (snapshot dict shape, accumulated list of typed records
    with mode in {hypothetical, claimed}, last_discussed string,
    pending dict with action in {add, remove}); malformed entries
    are dropped or defaulted, NEVER coerced silently. Same discipline
    as the existing wrong-type-drop loop, extended to nested dicts
    and enums. Tests cover each malformed-shape variant.

Plus minor sweeps: R-6 scenario count corrected from 8 to 11
(matches the live-regression test list); total estimate revised
from ~8 to ~10 hours to cover the retraction path, defensive
deserialization, and legacy-mapping work; the all-credentials-closed
narration no longer references SCCC (or any provider) because
`training_by_job` is empty on that branch and ungrounded-provider
policy forbids the mention.

Round 6 (six contract issues):
27. **Pending-clear ownership pinned to the handler.** The detector
    is a pure function and cannot mutate StagedProfile. v6
    documented "negative / unrelated reply clears pending" without
    saying WHO does the clearing — leaving the contract
    unimplementable. v7 specifies the handler shape: save
    `staged.pending_credential_confirmation` to a local, set the
    field to `None`, then call the detector with the saved value.
    Pending is re-set ONLY when the new turn synthesizes another
    `kind="confirm"` with a non-empty canonical. There is no
    scenario where stale pending state survives a turn.
28. **Pending-set guarded against `canonical=None`.** The
    "got it" / "I have that" disambiguation branch returns
    `kind="confirm"` with `confirmation_target_canonical=None`
    (the system asks which credential). v6's architecture diagram
    stored a pending entry anyway — which would be (a) useless
    (the detector's pending-consume branch cannot act on it) and
    (b) a defensive-deserialization violation (R-1 drops dicts
    whose canonical is not a non-empty string). v7 guards the
    pending-set with `if isinstance(canonical, str) and
    canonical:` so the field stays `None` on disambiguation turns;
    the user's next reply names the credential and runs fresh
    detection.
29. **R-3 detector call now uses the 6-arg signature.** v6 R-3
    still showed `detect_remaining_gaps_intent(message, snapshot,
    registry)` — the v5 shape — which was inconsistent with R-2's
    signature (round-5 fix). v7 R-3 shows the full save-and-clear
    pending pattern, the full keyword-arg call including
    `accumulated_credentials`, `pending_confirmation`, and
    `last_discussed_canonical`, and the five-arm `match intent.kind`
    block with an explicit `case "retract"` branch that filters the
    accumulated list and synthesizes the retraction-reason outcome.
30. **R-4 retraction execution + tests specified.** v6 R-4
    documented append-and-dedupe / promotion only; the retract path
    was implicit. v7 R-4 spells out: the removal is executed in
    R-3's `case "retract"` branch BEFORE R-4 runs; R-4 then builds
    the payload from the POST-removal accumulated list (no special
    case in the math); the ArbiterReason is
    `ARBITER_REASON_REMAINING_GAPS_RETRACTED`;
    `last_discussed_credential_canonical` updates to the LEAD
    remaining credential (which, after retraction, is typically
    the re-emerged credential); `training_by_job` is regrounded
    for that credential; six new direct retract handler tests
    cover removal, payload, reason code, last_discussed update,
    `any_hypothetical` re-derivation, training regrounding, and
    no-op-on-missing-canonical.
31. **Final stale-text sweep.** v6 still carried three pre-v6
    field references that v7 removes: the pending lifecycle row
    saying the remove path "produces `kind=\"subtract\"`" (now
    correctly `kind="retract"`); the §3 accumulation-trigger row
    listing only `kind="subtract"` (now lists both subtract and
    retract); the worked-example narrative saying `is_hypothetical=
    True` (now uses `any_hypothetical` + per-entry mode); and Q2
    referencing `assumed_completed_canonicals` (now
    `current_turn_claims`).
32. **Responder prompt no longer requests unsupported reasoning.**
    v6 §9 step 3 said "Briefly explain why each remaining gap
    matters for this role." The payload supplies only gap NAMES —
    no rationale, no transferability scoring, no impact ranking.
    Any "why it matters" sentence is invented content. v7 step 3
    is explicit: "Use ONLY the names supplied … Do NOT explain why
    each gap matters … the payload supplies names only." Provider
    mentions are also re-pinned to the TRAINING blocks in step 4.

Round 7 (four implementation blockers + diagram sweep):
33. **None-guard around the match block.** v7 R-3 had `match
    intent.kind:` directly; since the detector returns
    `RemainingGapsIntent | None`, a `None` return would have raised
    `AttributeError` at runtime and crashed the chat. v8 hoists the
    None case into an explicit `if intent is None:` branch BEFORE
    the match, with `continue_normal_dispatch()` inside. The match
    block also gains a `case _:` defensive arm that logs and
    continues for unknown future kinds rather than silently
    falling through.
34. **Negation against accumulated state always retracts.** v7
    required the "actually" hedge ("actually I don't have X") to
    trigger retraction confirmation; plain "I don't have X" against
    an already-assumed credential routed to explain_gap, leaving
    the stale assumption in place. v8 broadens step 2 of the
    detection ordering: ANY explicit negation pattern targeting a
    canonical present in `accumulated_credentials` returns
    `kind="confirm"` with `pending_action="remove"`. Step 3
    (standard negation) only fires for entities NOT in accumulated.
    Regression scenario #6 corrected; new #6b added for the
    pre-accumulated case; §13 ambiguity row + §2 Decision Summary
    rewritten; Q1 / step ordering invariant test added to R-2.
35. **Pending state preserved on detector failure.** v7 had the
    handler clear `pending_credential_confirmation` BEFORE the
    detector call — correct ownership, but if the detector raised
    (registry-load race, regex layer bug, downstream import
    error), the user's pending question silently disappeared. v8
    wraps the detector call in try/except: on exception, restore
    `saved_pending` to the StagedProfile field, log
    `remaining_gaps_detection_failed`, set `intent = None`, and
    fall through to the None-guard branch. The user gets a
    non-empty reply AND their pending state is intact for the next
    turn.
36. **Registry-failure graceful degradation specified.** Snapshot
    capture, detection, and handler-side training grounding all
    depend on the training registry; the registry is gated by
    `TRAINING_REGISTRY_ENABLED` (default off in code) and can fail
    to load. v8 adds §4a with three operating modes:
    - Mode A (registry loaded): canonical alias resolution through
      `registry.lookup`; the happy path documented throughout
    - Mode B (registry unavailable at capture): `_capture_match_snapshot`
      stores `canonical = _normalize(display)` (lowercase + strip
      + collapse whitespace); snapshot shape unchanged; the log
      `remaining_gaps_registry_unavailable_at_capture` fires once
    - Mode C (registry unavailable at detection): detector signature
      accepts `registry=None`; user-side entity resolution falls
      back to normalized-substring matching; ambiguous input
      returns `kind="confirm"` with `confirmation_target_canonical=
      None` rather than guessing
    R-1 / R-2 / R-3 each gain explicit tests for their slice of
    the degradation chain. Detector signature is updated to
    `registry: TrainingRegistry | None` throughout.

Diagram sweep: the top architecture diagram now shows the full
6-arg detector call (matching R-2 / R-3 prose), the try/except
restore-pending pattern, the seven-step detection ordering (pending
→ accumulated-retraction → standard-negation → uncertainty →
completion → hypothetical → generic), and the `if intent is None`
hoist before the match block. The diagram no longer lists
`kind=None` as a discriminated union shape (it's not — None is the
absence of intent, hoisted to a top-level branch).

Round 8 (three identity contracts + minor cleanup):
37. **Snapshot-anchored identity contract (§4.0 + §4.2).** Earlier
    drafts described detection as "resolve user input through
    registry.lookup, compare against snapshot canonicals" — which
    breaks when the snapshot was captured in Mode B (canonical =
    normalized display, e.g. "310s automotive technician license")
    and a later turn runs in Mode A (registry resolves user input
    to "310S automotive technician certification"). The two strings
    are different keys; subtraction would silently miss. v9 makes
    the snapshot the identity authority: every canonical used in
    `current_turn_claims`, `retract_canonical`,
    `confirmation_target_canonical`, accumulation, payload, and
    narration MUST be a value pulled verbatim from a
    `snapshot.lead_job.credential_gaps[*].canonical` slot. The new
    `_resolve_user_ref_to_snapshot_canonical` helper does a
    registry-assisted lookup AND a cross-mode bridge (second pass
    that runs `registry.lookup` on each snapshot `display` to
    confirm the user reference and the snapshot entry alias to the
    same registry canonical), THEN falls through to the
    deterministic fallback. The function returns the SNAPSHOT'S
    stored canonical, never a freshly-resolved value. A
    handler-level test enforces "every detected canonical exists
    in the snapshot" as a single-source-of-truth invariant.
38. **Deterministic fallback matching algorithm (§4.3).** The v8
    fallback ("prefix or token overlap") had no threshold and
    would match generic words like "license" against every
    credential. v9 defines the exact algorithm:
    `_GENERIC_CREDENTIAL_TOKENS` is a frozenset of stop-words
    ("license", "licence", "certification", "certificate",
    "permit", "the", "a", "an", "of", "and", "my", "got", "have",
    …); both user input and snapshot display strings are
    normalized (lowercase + non-alphanumeric → space + collapse)
    and tokenized; the generic set is subtracted; a snapshot
    entry matches IFF the user's non-generic tokens are a
    non-empty subset of the entry's non-generic tokens. Match
    succeeds only when exactly ONE snapshot entry qualifies —
    zero matches return None (route to existing planner); two-plus
    matches return None and the higher-level detector emits
    `kind="confirm"` with `confirmation_target_canonical=None`
    to force a clarification. No probabilistic ranking, no partial
    credit, no silent guess.
39. **Registry identity decoupled from `TRAINING_REGISTRY_ENABLED`
    (§4a table).** The flag historically gated everything
    registry-related; v9 splits it: alias resolution (snapshot
    capture + detection identity) ALWAYS attempts the registry
    load regardless of the flag; resource surfacing (provider
    names + URLs via `_registry_training_for_gap`) keeps the
    existing flag gate. A user running with the flag off and a
    loaded registry gets correct canonical resolution AND
    no-providers-named narration — both correct. R-1 / R-2 / R-4
    each gain explicit `TRAINING_REGISTRY_ENABLED=False +
    registry=loaded` combination tests.

Plus minor cleanup: the Turn 5b worked-example narration was
showing "With the 310S in hand" — a past-tense framing that
would have silently converted the hypothetical into an apparent
claim, the exact regression the `any_hypothetical` flag was
introduced to prevent. v9 narration now opens "If you've got the
310S in hand" with an explicit comment pinning the rule. The
detection-test summary row in §Tests was still citing v6 numbers
(~35 tests, "actually" retraction only) — updated to ~50 covering
all v8/v9 invariants (identity, fallback determinism, ordering,
all-explicit-negations retraction, Mode C, flag-decoupled
identity).

All seven open questions are also locked below.

## Why this document exists

The 2026-06-08 live test of `meeting_02_310s_automotive_weak.pdf`
surfaced a structural conversational-memory gap. Daniel's match card
showed two credential gaps for the Great Lakes Honda role:

- 310S Automotive Technician License
- G2/G driver's license

He asked about 310S, got the training answer, then asked variants of
"what else?" four turns in a row. The system kept narrating 310S
because:

1. `staged.last_presented_credential_gaps` is captured but not
   serialized into the LLM prompt — the model is structurally blind
   to the broader gap set
2. The router has no "remaining gaps" rule; "what else after X?" looks
   like another training_request and routes back through Rule 2
3. There is no mechanism for hypothetical / assumed completion: "if I
   have X" should subtract X from the displayed set without mutating
   the candidate's profile

The reviewer correctly framed this as a "follow-up reasoning layer
missing above the match result" — not an LLM prompt-tuning problem,
not a near-miss problem, not a registry data problem. A separate
architectural feature is required.

The reviewer also rejected a tempting smaller patch ("just serialize
PRESENTED_CONTEXT into the prompt and add a sentence") as insufficient:
the LLM would still infer hypothetical-vs-real, anaphoric reference
("got it" = what?), and ranking — leaving variability we've spent the
v2.1/v2.2 arc removing.

This document specifies the deterministic shape: detection in the
handler, subtraction in pure code, narration from a structured block.

## Decision summary (for reviewers)

- **Per-job snapshot** of the most-recent match, NOT a flattened
  credential list. Preserves job_id, title, employer, and the gap
  split (credentials / core_skills) so subtraction stays job-scoped.
- **Detection is deterministic**, not LLM-judged. Three conservative
  pattern layers: explicit-completion, hypothetical-completion,
  remaining-gap-request. Ambiguous statements ROUTE TO A FOCUSED
  CLARIFYING QUESTION, not to silent subtraction.
- **Canonical alias resolution** through `registry.lookup` so
  "310S licence" subtracts the engine-stored "310S Automotive
  Technician License" reliably.
- **New outcome `explain_remaining_gaps`** distinct from `explain_gap`.
  Handler-synthesized: planner, router, and engine all skip.
- **Structured `REMAINING_GAPS` block** sent to the responder. The LLM
  narrates from it; never infers, never invents. Deterministic
  fallback when LLM fails policy.
- **Profile is NEVER mutated** by hypothetical completion. Subtraction
  is request-scoped and recomputed each turn from the saved snapshot.
- **Context expires** when (a) a new match is presented, (b)
  target_role_text changes, (c) session resets, OR (d) the engine
  returns no presentable matches. Scope redirects (PR / immigration
  questions) do NOT clear the snapshot — they are temporary
  diversions, not topic changes. The user can return to the
  career-path conversation immediately afterward.
- **Negation must NOT subtract.** "I don't have 310S" leaves the
  remaining_credentials list unchanged. The detection rule has TWO
  branches: (a) negation against an entity in
  `accumulated_credentials` initiates retraction confirmation
  (`kind="confirm"` with `pending_action="remove"`) — recovers from
  an earlier hypothetical / claim; (b) negation against an entity
  NOT in accumulated returns `kind=None` and falls through to
  existing routing. Pending consumption + accumulated-retraction are
  ordered BEFORE completion and uncertainty patterns so an explicit
  negation always wins over a parser-flag elsewhere in the message.

## Architecture

### Current flow (post-v2.2)

```
present_matches turn
    │
    ▼
_capture_presented_context()
    │
    ├─ staged.last_presented_job_titles     (flat tuple)
    ├─ staged.last_presented_caps_applied   (flat tuple)
    └─ staged.last_presented_credential_gaps (flat tuple)
    │
    ▼
NEXT TURN: user asks "what else after 310S?"
    │
    ▼
build_truth_summary -> message_understanding -> router
    │
    ▼ Rule 2 fires (training_request + entity "310S")
    │
    ▼ explain_gap with TRAINING block for 310S
    │
    ▼ LLM re-narrates 310S apprenticeship details
                                            ^^^^^^^^^^^^^^^^^^^^^
                                     no signal that G2/G is also missing
```

### Proposed flow

```
present_matches turn
    │
    ▼
_capture_match_snapshot()    (NEW; replaces _capture_presented_context)
    │
    └─ staged.last_match_snapshot: dict[str, Any] | None
        {
          "captured_at_turn": <message_count int>,
          "lead_job": {
            "job_id": "...",
            "title":  "310S Licensed Automotive Technician",
            "employer": "Great Lakes Honda",
            "credential_gaps": [
              {"display": "310S Automotive Technician License",
               "canonical": "310S automotive technician certification"},
              {"display": "G2/G driver's license",
               "canonical": "Class G driver's license"},
            ],
            "core_skill_gaps": [
              "Honda vehicle experience",
              "dealership experience",
              "preventative maintenance",
              "automotive diagnostics",
            ],
            # operational gaps NOT included (already filtered by classify_gap)
          },
          "other_jobs_meta": [...]   # minimal metadata; NOT used in v1 subtraction
        }
    │
    ▼
NEXT TURN: user asks "what else after 310S?"
    │
    ▼ saved_pending = staged.pending_credential_confirmation
    ▼ staged.pending_credential_confirmation = None     (handler-owned
                                                         clear, §2)
    ▼ try:
    ▼     intent = detect_remaining_gaps_intent(
    ▼         message,
    ▼         staged.last_match_snapshot,           # may be None
    ▼         registry,                              # may be None
    ▼         accumulated_credentials=
    ▼             staged.last_assumed_completed_credentials,
    ▼         pending_confirmation=saved_pending,
    ▼         last_discussed_canonical=
    ▼             staged.last_discussed_credential_canonical,
    ▼     )
    ▼ except Exception:                            (registry / regex
    ▼     restore saved_pending; intent = None      failure is non-fatal;
    ▼                                               §1 Mode C)
        │   detection ordering (first match wins):
        │     1. pending consumption (yes/no/unrelated against
        │        saved_pending)
        │     2. retraction against accumulated (ANY explicit
        │        negation of an entity in accumulated_credentials
        │        -> kind="confirm" pending_action="remove";
        │        "actually" hedge NOT required)
        │     3. standard negation for non-accumulated entities
        │        -> kind=None (fall through to explain_gap)
        │     4. uncertainty markers ("think I got X")
        │        -> kind="confirm" pending_action="add"
        │     5. explicit completion ("I have X / passed X")
        │        -> kind="subtract" mode=claimed
        │     6. explicit hypothetical ("if/after/once/assume I have X")
        │        -> kind="subtract" mode=hypothetical
        │     7. generic remaining ("what else?", "anything else?")
        │        -> kind="subtract" (with current_turn_claims=[])
        │        if snapshot exists, else kind="bootstrap"
        │   all entity resolution runs through registry.lookup when
        │   the registry is loaded; falls back to normalized display
        │   text in Mode B / Mode C (see §4a)
        │
        └─ returns RemainingGapsIntent | None where None means
           "no remaining-gaps pattern matched" (handled by the
           explicit `if intent is None` guard below, NOT a kind
           on the union). When the intent is truthy, one of these
           four discriminated shapes:

             kind="subtract"        # explicit / hypothetical completion
                current_turn_claims: list[{canonical, mode}]
                # mode is "hypothetical" or "claimed";
                # handler appends to accumulated, preserves order
                # and dedupes; promotes hypothetical->claimed when
                # canonical already present with weaker mode

             kind="retract"         # completion previously claimed is
                                     # now being walked back
                retract_canonical: str
                # handler removes this canonical from
                # last_assumed_completed_credentials, then
                # synthesizes explain_remaining_gaps so the next
                # response correctly re-shows the credential as
                # remaining

             kind="confirm"         # uncertainty marker fired
                confirmation_target_canonical: str | None
                confirmation_target_display: str
                pending_action: "add" | "remove"
                # "add"    -> the system will ask
                #             "have you completed X?"
                #             ; a yes turns into kind="subtract"
                #             with mode=claimed on the next turn
                # "remove" -> the system will ask
                #             "to confirm, you don't have X?"
                #             ; a yes turns into kind="retract"
                #             on the next turn

             kind="bootstrap"       # remaining-gap intent but snapshot=None
                # no entity payload; the synthesis is "I haven't shown
                # matches yet -- want to search first?"

           (When the function returns None — no remaining-gaps pattern
            matched — the handler hoists that into a top-level branch
            BEFORE the match block. The four kinds above are the only
            values intent.kind can take when intent is truthy.)
    │
    ▼ handler-level synthesis (NOT router/planner contract):
        if intent is None:
            -> continue_normal_dispatch()    # standard planner/router
        else:
          match intent.kind:
            case "subtract":
                -> append intent.current_turn_claims to
                   staged.last_assumed_completed_credentials
                   (ordered append + dedupe + hypothetical->claimed
                    promotion)
                -> synthesize ArbiterDecision(
                    final_move="explain_remaining_gaps",
                    reason_code=ARBITER_REASON_REMAINING_GAPS,
                    tone="warm_supportive",
                    arbiter_action="handler_synthesized_remaining_gaps")
                -> populate inp.remaining_gaps_payload (see §9)
            case "retract":
                -> REMOVE intent.retract_canonical from
                   staged.last_assumed_completed_credentials
                   (filter: keep entries where canonical != target)
                -> synthesize ArbiterDecision(
                    final_move="explain_remaining_gaps",
                    reason_code=ARBITER_REASON_REMAINING_GAPS_RETRACTED,
                    tone="warm_supportive",
                    arbiter_action="handler_synthesized_remaining_gaps")
                -> populate inp.remaining_gaps_payload — the retracted
                   credential is back in remaining_credentials, and the
                   responder template explicitly acknowledges the
                   recalculation ("Okay, recalculating with the 310S
                   still needed -- the remaining credentials are ...")
            case "confirm":
                -> synthesize ArbiterDecision(
                    final_move="ask_one_clarifying_question",
                    reason_code="confirm_credential_completion",
                    tone="warm_supportive",
                    arbiter_action="handler_synthesized_clarification",
                    ask_slot=None)
                -> populate inp.clarification_payload (see §11 below)
                -> if isinstance(intent.confirmation_target_canonical, str)
                       and intent.confirmation_target_canonical:
                       SET staged.pending_credential_confirmation = {
                           "canonical": intent.confirmation_target_canonical,
                           "action":    intent.pending_action,
                       }
                   else:
                       # Disambiguation question ("got it" with no entity)
                       # -- user will reply with a credential NAME, not yes/no.
                       # Recording a pending entry with canonical=None would be
                       # both useless (the detector's pending-consume branch
                       # cannot act on it) AND a defensive-deserialization
                       # violation (R-1 drops dicts whose canonical is not a
                       # non-empty string). Leave pending UNSET; the next turn
                       # runs fresh detection on whatever credential the user
                       # named.
                       staged.pending_credential_confirmation = None
            case "bootstrap":
                -> synthesize ArbiterDecision(
                    final_move="ask_one_clarifying_question",
                    reason_code="bootstrap_match_request",
                    tone="warm_supportive",
                    arbiter_action="handler_synthesized_clarification",
                    ask_slot=None)
                -> populate inp.clarification_payload (kind="bootstrap")

        Planner is SKIPPED. Engine is SKIPPED. validate_planner_intent
        is SKIPPED. Same control-flow pattern as gates.py: handler
        emits the ArbiterDecision directly when the deterministic
        signal is unambiguous. The synthesized ArbiterDecision flows
        STRAIGHT to compose_response_v2 without traversing Pass 1 or
        Pass 2.

        Rationale: route_from_understanding() returns PlannerDecision,
        whose `move` field is the PlannerMove Literal. Adding
        explain_remaining_gaps to PlannerMove would expose a Pass-2-style
        outcome to Pass 1 callers and weaken the existing dispatch
        invariants. Handler-level synthesis avoids that contamination
        entirely.
    │
    ▼ handler computes remaining_gaps_payload (subtract kind only):
        # Ordered append-and-dedupe so accumulation stays deterministic
        # across processes and serializations.
        all_assumed = list(staged.last_assumed_completed_credentials)  # PRIOR turns
        seen = {a["canonical"] for a in all_assumed}
        for claim in intent.current_turn_claims:                       # THIS turn
            if claim["canonical"] not in seen:
                all_assumed.append(claim)                              # {canonical, mode}
                seen.add(claim["canonical"])
            else:
                # Promote hypothetical -> claimed when a stronger claim arrives
                for existing in all_assumed:
                    if (existing["canonical"] == claim["canonical"]
                        and existing["mode"] == "hypothetical"
                        and claim["mode"] == "claimed"):
                        existing["mode"] = "claimed"

        assumed_canonicals = {a["canonical"] for a in all_assumed}
        remaining_credentials = [
            g for g in snapshot["lead_job"]["credential_gaps"]
            if g["canonical"] not in assumed_canonicals
        ]
        remaining_core_skills = snapshot["lead_job"]["core_skill_gaps"]  # no v1 subtraction

        # Persist accumulation BEFORE rendering, so the next turn
        # sees the union; cap drops latest entries (ordered, not random).
        staged.last_assumed_completed_credentials = all_assumed[:MAX_CRED_GAPS]
        any_hypothetical = any(a["mode"] == "hypothetical" for a in all_assumed)
    │
    ▼ handler populates training_by_job for the LEAD remaining credential:
        if remaining_credentials and TRAINING_REGISTRY_ENABLED:
            lead_canonical = remaining_credentials[0]["canonical"]
            training_by_job = _registry_training_for_gap(
                staged, discovered_gaps=[lead_canonical])
        This step is what grounds provider names in the responder reply
        (DriveTest / Ontario.ca / SCCC for G2/G, etc.). Without it the
        policy regex rejects every provider name as ungrounded.
    │
    ▼ pass payloads to responder:
        ResponderV2Input.remaining_gaps_payload
        ResponderV2Input.training_by_job (lead credential's resources)
        ResponderV2Input.clarification_payload (None on subtract kind)
    │
    ▼ responder narrates from the structured blocks; deterministic
       fallback when LLM fails policy
```

The new outcome reaches the responder WITHOUT going through
`validate_planner_intent` (Pass 1) or `resolve_match_outcome` (Pass 2).
A new arbiter invariant test (§Tests) enforces this.

## Locked decisions

### 1. Snapshot shape: per-job, NOT flattened

```python
last_match_snapshot = {
    "captured_at_turn": int,                # staged.message_count at capture
    "lead_job": {
        "job_id":   str,
        "title":    str,                    # truncated to MAX_TITLE_CHARS=80
        "employer": str | None,             # truncated to MAX_EMPLOYER_CHARS=60
        "credential_gaps": list[{           # capped at MAX_CRED_GAPS=5
            "display":   str,               # engine string; truncated to 80
            "canonical": str,               # registry-canonical; truncated to 80
        }],
        "core_skill_gaps": list[str],       # capped at MAX_SKILL_GAPS=5
                                            #   display strings only; canonical
                                            #   not needed for v1 (no skill
                                            #   subtraction)
    },
    "other_jobs_meta": list[{               # capped at MAX_OTHER_JOBS=3
        "job_id": str,                      # MINIMAL metadata only --
        "title":  str,                      # no gap lists for other jobs;
    }],                                     #   they're stored for future job-pivot
                                            #   but never used in v1 subtraction
}
```

**Cookie-size constraint.** Sessions are stored in a signed cookie
(SESSION_STORE=cookie). Browser limit is 4 KB total per cookie
including Set-Cookie header overhead (name, attributes,
HttpOnly, Path, SameSite, Secure). The session cookie payload
(the signed value alone, what `CookieSessionStore.save` returns)
must stay well under 4 KB to leave header margin.

The signed value carries:
- the JSON-encoded StagedProfile from
  `staged.to_json(redact_for_cookie=True)`. Redaction:
  - `resume_text` → `None` (fully dropped; raw text can be 10-50 KB)
  - `resume_facts_json` → `compact_facts()` form (COMPACTED, NOT
    fully removed; the compact form strips evidence excerpts but
    keeps structured skills / work_history / certifications /
    languages summary signals — typically 200-800 bytes
    depending on resume richness)
  - all other StagedProfile fields are sent unchanged
- the `TimestampSigner` suffix (~88 bytes: dot separator + base64
  timestamp + base64 HMAC-SHA1)

**Available budget for the JSON payload** (corrected from v4):
- 4096 bytes browser-cookie-budget limit
- minus ~200 bytes for Set-Cookie header attributes
- minus ~88 bytes for the signer suffix
- minus ~800 bytes for compacted resume_facts_json worst-case
- = **~3 KB usable** for the rest of the StagedProfile (skills,
  work_history, target_role_text, intake_state, suppressed_fact_ids,
  AND the three new remaining-gaps fields)

Realistic worst-case sizing for the snapshot alone (corrected from
the earlier estimate, which under-counted):

- Lead job overhead (job_id UUID 36 + title up to 80 + employer up
  to 60 + structural JSON braces/keys): ~250 bytes
- 5 credential_gaps × (display 80 + canonical 80 + JSON shape): ~1100 bytes
- 5 core_skill_gaps × (string up to 80 + JSON shape): ~500 bytes
- Subtotal lead_job: **~1850 bytes**
- other_jobs_meta: 3 × (UUID 36 + title 80 + JSON shape ~25): ~430 bytes
- Snapshot total worst-case: **~2.3 KB**

That's tight. Combined with the existing StagedProfile fields,
worst-case sessions could approach the 3.3 KB JSON budget. Tighter
caps may be required after measuring real session sizes; the test
below is the authoritative gate.

**Test requirement: measure the full signed cookie, not just the
snapshot.** Snapshot-only JSON size is not the binding constraint;
the full serialized + signed StagedProfile is.

```python
def test_full_signed_session_under_browser_budget_with_max_state():
    """The actual constraint is the browser's 4 KB per-cookie limit
    INCLUDING Set-Cookie header attributes. The signed value alone
    must stay well under 4 KB to leave header margin (Path, HttpOnly,
    SameSite, Secure, Max-Age, Domain in production).

    Build a StagedProfile populated to realistic worst-case:
    - max snapshot (5 credentials + 5 skills + 3 other_jobs_meta)
    - max accumulated assumptions (5 entries with mode field)
    - pending_credential_confirmation set
    - last_discussed_credential_canonical set
    - compacted resume_facts_json worst-case (~800 bytes)
    - typical staged.skills (e.g. 20 entries)
    - typical work_history (e.g. 3 entries via compact_facts)
    Then confirm CookieSessionStore.save() returns a value comfortably
    under the budget."""
    from skillbridge.session.cookie_store import CookieSessionStore
    staged = _build_worst_case_staged_with_max_remaining_gaps_state()
    store = CookieSessionStore(secret="x" * 48)
    signed_value = store.save(staged)
    assert len(signed_value.encode("utf-8")) < 3800, (
        f"signed session value is {len(signed_value)} bytes; ceiling "
        f"is 3800 to leave ~300 bytes margin for Set-Cookie attributes "
        f"(Path / HttpOnly / SameSite / Secure / Max-Age). If this "
        f"fails, tighten MAX_CRED_GAPS / MAX_SKILL_GAPS / MAX_OTHER_JOBS "
        f"or reduce the cap on last_assumed_completed_credentials."
    )


def test_remaining_gaps_fields_contribution_observable():
    """Diagnostic-only: measure ONLY the remaining-gaps fields'
    contribution to cookie size. This is NOT the binding test (see
    above) — it's a regression alarm so we notice when one component
    grows."""
    fields = {
        "last_match_snapshot": _build_max_size_snapshot(),
        "last_assumed_completed_credentials": [
            {"canonical": "x" * 80, "mode": "hypothetical"}
            for _ in range(MAX_CRED_GAPS)
        ],
        "last_discussed_credential_canonical": "x" * 80,
        "pending_credential_confirmation": {
            "canonical": "x" * 80, "action": "add",
        },
    }
    size = len(json.dumps(fields).encode("utf-8"))
    assert size < 3000, (
        f"remaining-gaps fields JSON size {size} exceeds 3 KB regression "
        f"budget. Other StagedProfile fields need at least 800 bytes."
    )
```

Implementation MUST run the first test against the actual
`CookieSessionStore.save()` path, not a `json.dumps` shortcut.
The real path includes:
- `staged.to_json(redact_for_cookie=True)`:
  - `resume_text` → `None` (fully dropped)
  - `resume_facts_json` → `compact_facts()` form (COMPACTED, not
    removed — still costs cookie space)
- `itsdangerous.TimestampSigner.sign(...)` (~88 bytes suffix)

Only the production path is reliable for the size assertion. The
**3800-byte ceiling** is the binding number, not 4096 — the latter
leaves no room for the Set-Cookie response header attributes the
browser counts toward the 4 KB-per-cookie limit.

Rationale: a single flat credential list cannot answer "what's left for
THIS job?" when the user pivots between jobs in the same session.
Per-job structure preserves the answer; the lead job is the one used
for subtraction. `other_jobs_meta` stores enough to PIVOT to another
job in a future feature (job_id + title), but NOTHING heavyweight that
would replicate the gap-analysis state. That replication would blow
cookies in any session with 3+ matches.

The `display` / `canonical` split is the canonical-alias resolution
fix. Engine emits "310S Automotive Technician License"; registry
canonical is "310S automotive technician certification"; user types
"310S licence". All three resolve to the same `canonical` value and
the subtraction works.

### 2. Completion language: explicit only, ambiguity gets clarification

**High-confidence completion patterns (deterministic subtraction):**

| Category | Pattern | Example |
|---|---|---|
| Definite past | `\bI (have|got|earned|finished|passed|completed)\s+(my|the)?\s*<entity>` | "I have my 310S" |
| Definite present | `\bI (now have|already have|currently have|am certified in)\s+<entity>` | "I already have G2" |
| Hypothetical | `\b(if|once|after|assuming?|suppose)\s+I\s+(have|got|get|earn|finish|pass)\s+(my|the)?\s*<entity>` | "after I get 310S" |
| Anchored hypothetical | `\b(assume|let'?s say|imagine|say)\s+I\s+(have|got)\s+<entity>` | "assume I have it" |

**Uncertainty markers force confirmation, even with entity match.**
"Think I got X" / "I'm pretty sure I have X" / "I probably finished
X" all carry epistemic uncertainty. Even if the entity resolves to a
snapshot credential, the user is signaling they're NOT certain. The
system must NOT subtract on uncertain claims; it must ask:

| Pattern | Example | Response |
|---|---|---|
| `\b(think|believe|guess)\s+I\s+(have|got|finished|completed)\s+<entity>` | "I think I got 310S" | Ask: "Just to make sure — have you completed your 310S, or are you still working through it?" |
| `\b(probably|maybe|might have)\s+(have|got|finished)\s+<entity>` | "I might have my G2 already" | Same confirmation |
| `\bpretty sure\b` + completion verb | "pretty sure I have 310S" | Same confirmation |

Rationale: the design's safety contract is that hypothetical
subtraction is REQUEST-SCOPED but DEFINITE. Subtracting on uncertain
claims silently shifts the system's understanding while the user is
themselves uncertain. The cheap recovery is a confirmation question;
the expensive failure mode is acting on a claim the user would
themselves have walked back if asked.

`<entity>` must resolve via `registry.lookup` to one of the entries in
`snapshot.lead_job.credential_gaps[*].canonical`.

**Bare anaphor ("it", "that", "the licence") needs a NEW dedicated
resolver — not a reuse of `_resolve_target_role_anaphor`.** That
function reads `staged.resume_facts_json["work_history"]` to resolve
"same role" → the resume's current job title. It cannot resolve "it"
or "that licence" in a credential-completion sentence because those
references aren't anchored to resume work history; they're anchored
to the most-recently-discussed credential gap from the snapshot.

**`last_discussed_canonical` is a persisted session field, NOT a
turn-local computation.** Without persistence the resolver can't
honor recency across turns: if the user discusses credential B
in turn N, then says "got it" in turn N+1, the resolver MUST
return B — not the snapshot's `[0]` entry (which would be A).

**Anaphor alone is NOT enough for confirmation interpretation
(round-4 review).** Knowing WHICH credential was discussed is one
half; the other half is knowing WHY the system asked — to add
(completion confirmation) or remove (retraction confirmation). A
bare "yes" with no pending-action context is ambiguous.

Two NEW StagedProfile fields:

```python
# session/staging.py — added to StagedProfile dataclass

# Recency for credential anaphor resolution ("it", "that licence")
last_discussed_credential_canonical: str | None = None

# Pending confirmation state — set when the handler synthesizes a
# `kind="confirm"` clarification; consumed and cleared on the user's
# next turn so "yes" / "no" / retraction is interpretable
# deterministically.
pending_credential_confirmation: dict | None = None
# Shape: {"canonical": str, "action": "add" | "remove"}
#   action="add"    -> the system asked "have you completed X?"
#                      ; "yes" -> add to last_assumed_completed_credentials
#                      ; "no"  -> do nothing; clear pending
#   action="remove" -> the system asked "to confirm — you don't have X?"
#                      ; "yes" -> remove X from last_assumed_completed_credentials
#                      ; "no"  -> keep X in the set; clear pending
```

**Cookie cost**: canonical 80 chars + action 8 chars + JSON shape ≈
130 bytes for `pending_credential_confirmation`. `last_discussed`
adds 90 bytes. Combined ≈ 220 bytes, accounted for in the §1
budget recalculation.

**Lifecycle of `last_discussed_credential_canonical`:**

| Trigger | Action |
|---|---|
| Handler emits `explain_gap` with a specific credential entity | Set to that credential's canonical |
| Handler emits `explain_remaining_gaps` with subtraction | Set to the LEAD remaining credential's canonical |
| Handler emits the `confirm` clarification | Set to `intent.confirmation_target_canonical` so the next turn can resolve "yes I got it" |
| Snapshot is cleared (new match / no_match / target_role change / session reset) | Cleared together with snapshot |
| `explain_remaining_gaps` produces an empty remaining_credentials list | Cleared (no credential to anchor against) |

**Lifecycle of `pending_credential_confirmation`:**

| Trigger | Action |
|---|---|
| Handler synthesizes `kind="confirm"` for an *addition* (system asks "have you completed X?") | Set to `{canonical: <X>, action: "add"}` |
| Handler synthesizes `kind="confirm"` for a *retraction* (user said "actually I don't have X" against the accumulated set) | Set to `{canonical: <X>, action: "remove"}` |
| User next turn → handler **clears** `pending_credential_confirmation` (saved local copy is passed to detection); detection sees affirmative ("yes", "yes I have", "yeah") | Returns `kind="subtract"` with the pending canonical (mode=`claimed`) when saved `action="add"`; returns `kind="retract"` with the pending canonical when saved `action="remove"` |
| User next turn → handler clears pending; detection sees negative ("no", "not yet", "still working on it") | Returns `kind=None`; no subtraction or retraction; falls through to normal dispatch |
| User next turn → handler clears pending; detection sees an unrelated message | Returns whatever the standard detection ordering produces against the new message (may be `kind=None`, may be another `kind="confirm"` for a different credential) |
| Snapshot is cleared | Pending field cleared together with snapshot |

**Clearing ownership (critical):** the detector is a pure function —
it reads `pending_confirmation` but CANNOT mutate StagedProfile (it
doesn't receive a reference; it receives a value copy). The HANDLER
owns clearing. Required handler shape, per turn, BEFORE the detection
call:

```python
# Save and clear pending state up-front, regardless of what the user
# typed. Pending semantics are "valid for one turn only" -- by the
# time detection sees the user's reply, the pending question is
# already in the past.
saved_pending = staged.pending_credential_confirmation
staged.pending_credential_confirmation = None

intent = detect_remaining_gaps_intent(
    message, snapshot, registry,
    accumulated_credentials=staged.last_assumed_completed_credentials,
    pending_confirmation=saved_pending,   # value copy, not the field
    last_discussed_canonical=staged.last_discussed_credential_canonical,
)

# Re-set pending ONLY if the new turn produced kind="confirm" with
# a real canonical (see architecture diagram case "confirm" guard).
```

This single discipline answers all three pending-reply cases (yes /
no / unrelated) with no extra branches: the handler always clears,
and only re-sets when the new turn synthesizes another `kind="confirm"`
with a non-empty canonical. There is no scenario where stale pending
state survives a turn.

**Cookie cost**: a canonical credential name fits in 80 bytes (covered
by the same MAX_CANONICAL_CHARS=80 cap used for snapshot fields). Added
to the StagedProfile total: ~90 bytes including JSON key + quotes +
field separator. Stays well within the cookie budget recalculated in §1.

**Resolver shape:**

```python
def _resolve_credential_anaphor(
    message: str,
    snapshot: dict,
    *,
    last_discussed_canonical: str | None,
) -> str | None:
    """Resolve 'it' / 'that' / 'the licence' to a snapshot credential
    canonical name. Returns None when no anaphor pattern fires or no
    candidate gap is available to anchor against.

    Pattern reuse: same `_ANAPHORIC_TARGET_PATTERNS` shape as
    `_resolve_target_role_anaphor` (deterministic regex against
    pronominal / definite-article phrases). The RESOLUTION TARGET
    differs: this resolver looks at the snapshot's credential gap
    list (with last_discussed_canonical for recency tie-breaking),
    not at resume work history.
    """
    # patterns: "it", "that", "this one", "the licence", "the certificate",
    #           "the credential", "the cert"
    if not _matches_credential_anaphor_patterns(message):
        return None
    gaps = (snapshot.get("lead_job") or {}).get("credential_gaps") or []
    if not gaps:
        return None
    # Recency: if the persisted last-discussed canonical is in the
    # snapshot's credential list, prefer it. This is what makes
    # "it" / "that" resolve to the credential ACTUALLY being
    # discussed, not the snapshot's [0] entry.
    if last_discussed_canonical:
        for g in gaps:
            if g["canonical"] == last_discussed_canonical:
                return last_discussed_canonical
    # Fall back to first snapshot credential ONLY when there is no
    # persisted prior context (e.g. immediate post-match turn where
    # nothing specific has been discussed yet).
    return gaps[0]["canonical"]
```

Pattern strategy is reused from the target-role resolver; the function
is new. This keeps `_resolve_target_role_anaphor` focused on its
single job (resume-anchored role names) and avoids overloading it
with a credential-resolution responsibility it wasn't designed for.

**Ambiguous patterns (route to clarification, NOT silent subtraction):**

| Pattern | Why ambiguous | Response |
|---|---|---|
| "got it" with no entity | Could be acknowledgement of explanation OR completion claim | Ask: "Got the 310S licence, or got the path I just explained?" |
| "I have that" with stale antecedent | "that" could refer to multiple things | Ask: "Which one do you mean — the 310S or the G2/G?" |
| "done with X" | "done with" = "finished discussing" OR "finished earning"? | Ask: "Done meaning you've got it, or done discussing it?" |

**Negation patterns (must NEVER subtract):**

| Pattern |
|---|
| `\b(don'?t|do not|haven'?t|have not|won'?t|will not|never|no)\b.*\b(have|got|finish|complete|pass|earn|hold)\b` |
| `\b(missing|without|lacking|no)\s+<entity>` |
| `\bneed (to get|to earn|to finish|to pass)\b` (signals NOT acquired) |

Negation is checked FIRST. If a negation matches anywhere referring to
a gap entity, that gap is NOT subtracted regardless of any positive
pattern in the same message.

### 3. Assumption accumulation across turns (NOT profile mutation)

**Critical design correction (round-3 review):** subtraction must be
based on the FULL accumulated assumption set across the snapshot's
lifetime, not just the current turn's. Otherwise:

> Turn 1: *"Assume I have 310S"* → 310S subtracted, response names G2/G
> Turn 2: *"What else?"*       → no entity → no current-turn subtraction
>                                → response re-names 310S
>
> The user just told the system to treat 310S as done and the system
> immediately forgot.

The fix is **conversation-scoped accumulation**, distinct from profile
evidence:

NEW StagedProfile field — list of typed records preserving the
**mode of each claim** (hypothetical vs explicit):

```python
# session/staging.py
last_assumed_completed_credentials: list[dict] = field(default_factory=list)
# Each entry shape (typed via TypedDict in implementation):
#   {"canonical": str, "mode": "hypothetical" | "claimed"}
```

**Why `mode` is critical (round-4 review): an accumulated bare list of
canonicals loses provenance.** After:

> Turn 5b: *"if I had 310S"* → adds 310S (hypothetical)
> Turn 6: *"what else?"* → current-turn `is_hypothetical=False`

The responder's payload `is_hypothetical` field would say False on
Turn 6 because THIS turn carries no claim — but 310S is in the
accumulated set as a hypothetical. The responder could then narrate
*"With your 310S done, the next step..."* — silently converting a
hypothetical into an apparent claim. The mode field prevents this:
narration MUST stay conditional ("if you've got the 310S in hand…")
while ANY accumulated assumption is hypothetical.

**Semantics:**
- Conversation state, NOT profile evidence
- Tied to the snapshot's lifetime; cleared with the snapshot
- Ordered list (NOT set) — preserves snapshot order of first occurrence
  so narration, cookie serialization, and tests stay deterministic
- Deduped by canonical name; capped at `MAX_CRED_GAPS=5`
- Cookie cost: 5 × (80-byte canonical + ~25-byte mode + JSON shape)
  ≈ 600 bytes max
- Promotion rule: if a canonical is currently mode=`hypothetical` and
  the user later makes an explicit claim ("I actually have it now"),
  the existing entry is promoted to mode=`claimed` (NOT duplicated)

**Subtraction uses the accumulated list, preserving order:**

```python
def compute_remaining(snapshot, staged, current_turn_claims):
    """current_turn_claims is a list[dict] in the same shape as the
    persisted field — each entry has {canonical, mode}."""
    # Ordered append-and-dedupe: keep snapshot-order of first occurrence
    all_assumed: list[dict] = list(staged.last_assumed_completed_credentials)
    seen = {a["canonical"] for a in all_assumed}
    for claim in current_turn_claims:
        canonical = claim["canonical"]
        if canonical in seen:
            # Promote hypothetical -> claimed when a stronger claim arrives
            for existing in all_assumed:
                if existing["canonical"] == canonical and existing["mode"] == "hypothetical" and claim["mode"] == "claimed":
                    existing["mode"] = "claimed"
            continue
        all_assumed.append(claim)
        seen.add(canonical)

    # Subtract from snapshot
    assumed_canonicals = {a["canonical"] for a in all_assumed}
    remaining = [
        g for g in snapshot["lead_job"]["credential_gaps"]
        if g["canonical"] not in assumed_canonicals
    ]
    return remaining, all_assumed
```

After subtraction succeeds, the handler PERSISTS the ordered list back
(truncated to the cap — note: append-and-dedupe preserves snapshot
order, so the truncation drops the LATEST entries first, not random
ones):

```python
staged.last_assumed_completed_credentials = all_assumed[:MAX_CRED_GAPS]
```

**Payload to responder carries the mode information:**

```python
remaining_gaps_payload = {
    ...,
    "assumed_completed_credentials": [
        {"display": "...", "canonical": "...", "mode": "hypothetical"},
        {"display": "...", "canonical": "...", "mode": "claimed"},
    ],
    "any_hypothetical": True,   # derived: any entry has mode=hypothetical
    # When True, the responder MUST use conditional narration
    # ("if you've got…", "assuming you have…")
    # When False (all claimed), it may use past-tense narration
    # ("with your X done…")
}
```

**Lifecycle of `last_assumed_completed_credentials`:**

| Trigger | Action |
|---|---|
| Handler emits `explain_remaining_gaps` via `kind="subtract"` | Union the turn's claims into the field (ordered append + dedupe + hypothetical→claimed promotion); cap and persist |
| Handler emits `explain_remaining_gaps` via `kind="retract"` | Filter the named canonical OUT of the field (keep entries where `canonical != intent.retract_canonical`); persist the shorter list |
| Snapshot is cleared (new match / no_match / target_role_text change / session reset) | Cleared together with snapshot |
| User explicit retraction ("actually I don't have 310S") | Detection layer detects retraction language against the accumulated set → returns `kind="confirm"` with `pending_action="remove"`. Handler synthesizes the clarification "Just to confirm — you don't have your 310S? I'll recalculate against that." and sets `pending_credential_confirmation = {canonical, action: "remove"}`. On the NEXT turn an affirmative answer makes detection return `kind="retract"`, and the handler's `kind="retract"` branch removes the canonical from the accumulated list. v1 does NOT silently auto-retract. |

**Privacy / correctness lock still holds:**
- `staged.skills` is NEVER appended with hypothetical credentials
- `staged.resume_facts_json` is NEVER edited
- `staged.last_match_snapshot` is NEVER edited mid-conversation
  (only on a new `present_matches` turn)
- The accumulated list is conversation state — it does NOT feed
  the matching engine, does NOT reach `_effective_facts_view`,
  does NOT influence which jobs are presented. It only feeds
  the remaining-gaps subtraction inside this feature.

A user saying "if I had 310S" still does NOT result in their stored
profile claiming 310S. The accumulated field is a per-conversation
scratchpad scoped to the current snapshot; a new match clears it.

### 4. Canonical alias resolution

#### 4.0 Identity contract — the snapshot is authoritative

Every credential identity used anywhere in this feature MUST be a
canonical value stored in the snapshot's `credential_gaps[*].canonical`
at capture time. Subtraction, retraction, accumulation, payload, and
narration ALL use snapshot-stored canonicals. The detector NEVER
invents a fresh canonical from a registry resolution and compares
against the snapshot — that risks the Mode-B-then-Mode-A divergence
flagged in round-7 review.

Concretely: at capture time `_capture_match_snapshot` resolves each
engine display string ONCE through whatever resolver is available
(registry in Mode A, normalization in Mode B), stores both `display`
and `canonical`, and that stored `canonical` is the identity of that
gap for the lifetime of the snapshot. Detection's job is to take a
user-typed reference and answer "which snapshot entry is this?" — the
answer is always one of the snapshot's stored canonicals (or None).

The detector NEVER:
- runs `registry.lookup` on user input and compares the result
  directly against snapshot canonicals
- emits a `canonical` value (in `current_turn_claims`,
  `retract_canonical`, or `confirmation_target_canonical`) that was
  not pulled verbatim from a snapshot entry

This keeps a snapshot captured in Mode B usable in a later Mode-A
turn: even if the registry later resolves "310S licence" to
"310S automotive technician certification", the snapshot stored
"310s automotive technician license", and detection returns the
LATTER (the snapshot's stored value) because that's what every
downstream comparison expects.

#### 4.1 Snapshot capture (Mode A path)

For each entry in `snapshot.lead_job.credential_gaps`:

```python
display = engine_missing_skill_name   # "310S Automotive Technician License"
hit = registry.lookup(display)
canonical = hit.canonical_name if hit else _normalize_canonical(display)
# canonical here is the snapshot's IDENTITY for this gap;
# every later comparison anchors back to it.
```

For gaps NOT in the registry (e.g., "Honda vehicle experience"),
`_normalize_canonical(display)` is the canonical. These are
core_skill gaps, not credentials — and v1 doesn't subtract skill
gaps (Open Question 5).

#### 4.2 Detection-time resolution — anchor to the snapshot

When the user message contains a gap reference, detection must
identify the SNAPSHOT ENTRY the reference points to, then return
THAT entry's stored `canonical`. The two-step resolver:

```python
def _resolve_user_ref_to_snapshot_canonical(
    user_substring: str,
    snapshot: dict,
    registry: TrainingRegistry | None,
) -> str | None:
    """Map a user-typed credential reference to the snapshot entry it
    points to. Returns the SNAPSHOT'S stored canonical, NOT a
    freshly-computed registry canonical. Returns None when no unique
    snapshot entry matches.
    """
    gaps = (snapshot.get("lead_job") or {}).get("credential_gaps") or []
    if not gaps:
        return None

    # (a) Registry-assisted exact match. Mode A only.
    if registry is not None:
        hit = registry.lookup(user_substring)
        if hit is not None:
            target = hit.canonical_name
            for g in gaps:
                if g["canonical"] == target:
                    return g["canonical"]              # snapshot stored it
                # Cross-mode bridge: snapshot was captured in Mode B
                # (its canonical is the normalized display) while we're
                # NOW in Mode A. The registry's resolution may not
                # equal the snapshot's value but should still alias to
                # the same display. Confirm via a second lookup.
                snap_hit = registry.lookup(g["display"])
                if snap_hit is not None and snap_hit.canonical_name == target:
                    return g["canonical"]              # return SNAPSHOT's
            # Registry resolved the user input but no snapshot entry
            # aliases to it: this credential is in the registry but
            # NOT a gap for the current match. Fall through to (b);
            # otherwise the user said "I have my Smart Serve" while
            # discussing an auto-tech role and we ignore it correctly.

    # (b) Deterministic normalized-token matching. Always runs.
    return _match_user_ref_by_tokens(user_substring, gaps)
```

#### 4.3 Deterministic fallback matching

`_match_user_ref_by_tokens` is the Mode-B / Mode-C resolver and the
fall-through when registry resolution found nothing comparable in
the snapshot. It MUST be exact, deterministic, and require a unique
candidate — no "prefix or token overlap" handwaving.

```python
# Generic credential vocabulary that carries no identity information.
# Matching on these alone would let "license" match every snapshot
# credential, which is exactly the failure mode flagged in review.
_GENERIC_CREDENTIAL_TOKENS = frozenset({
    "license", "licence", "certification", "certificate", "permit",
    "the", "a", "an", "of", "and", "or", "my", "your", "i", "got",
    "have", "had", "for", "to", "from",
})

def _tokens(s: str) -> frozenset[str]:
    # lowercase, strip surrounding whitespace, replace non-alphanumerics
    # with a single space, collapse whitespace, split on whitespace
    normalized = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    return frozenset(normalized.split()) - _GENERIC_CREDENTIAL_TOKENS

def _match_user_ref_by_tokens(
    user_substring: str,
    gaps: list[dict],
) -> str | None:
    user_tokens = _tokens(user_substring)
    if not user_tokens:
        # User input contained ONLY generic words ("the license") --
        # cannot disambiguate; clarification path will fire.
        return None
    candidates: list[str] = []
    for g in gaps:
        snap_tokens = _tokens(g["display"])              # tokens from the
                                                          # original display
                                                          # string, not the
                                                          # canonical
        if not snap_tokens:
            continue                                      # snapshot entry was
                                                          # all-generic too;
                                                          # cannot be matched
                                                          # by this resolver
        # Match rule: user_tokens must be a NON-EMPTY SUBSET of snap_tokens
        # AND have at least one identifier-carrying token in common
        # (i.e., at least one shared non-generic token).
        if user_tokens.issubset(snap_tokens):
            candidates.append(g["canonical"])             # snapshot stored it
    if len(candidates) == 1:
        return candidates[0]
    # Zero candidates -> no resolution; user named something not in the
    # snapshot, route through standard explain_gap or unknown-gap flow.
    # Two+ candidates -> ambiguous; detector returns kind="confirm" with
    # confirmation_target_canonical=None to force a clarification.
    return None
```

**Algorithm properties:**
- Deterministic: same input → same output, no probabilistic ranking
- No partial credit: matches a snapshot entry only when EVERY
  non-generic user token appears in that entry's display tokens
- No generic-token matches: "license" alone matches NOTHING
- Unique-candidate-only: if multiple snapshot entries qualify,
  return None and force a clarification question (the user said
  something genuinely ambiguous — asking is correct)
- Snapshot-anchored: returns the snapshot's stored `canonical`
  verbatim, NOT a freshly-resolved value

Worked example — Mode B snapshot, Mode A detection:
- Snapshot (captured Mode B) stores
  `[{display: "310S Automotive Technician License",
     canonical: "310s automotive technician license"},
    {display: "G2/G driver's license",
     canonical: "g2 g driver s license"}]`
- User types "310S licence" in a later (Mode A) turn
- Step (a): registry resolves "310S licence" → "310S automotive
  technician certification". No snapshot canonical matches that
  string. Second pass tries `registry.lookup` on each snapshot
  `display`; `lookup("310S Automotive Technician License")` returns
  the same canonical_name "310S automotive technician certification",
  so step (a) matches the snapshot entry and returns its STORED
  canonical `"310s automotive technician license"` (NOT the
  registry's canonical).
- Step (b) is not reached.

#### 4.4 Single-source-of-truth invariant test

A handler-level test enforces the identity contract:

```python
def test_detection_always_returns_snapshot_canonicals():
    snapshot = _build_snapshot_with_canonicals(["X", "Y"])
    for message in COMPLETION_PATTERNS_AGAINST_X:
        intent = detect_remaining_gaps_intent(message, snapshot, ...)
        assert intent.kind == "subtract"
        for claim in intent.current_turn_claims:
            assert claim["canonical"] in {"X", "Y"}
    for message in RETRACTION_PATTERNS_AGAINST_X:
        intent = detect_remaining_gaps_intent(
            message, snapshot, ...,
            accumulated_credentials=[{"canonical": "X", "mode": "claimed"}],
        )
        assert intent.kind == "confirm"
        assert intent.confirmation_target_canonical in {"X", "Y"}
```

#### 4a. Registry-failure graceful degradation

`registry.lookup` is the canonical-alias mechanism. The feature MUST
degrade gracefully — remaining-gaps reasoning is the safety-critical
path that closes a live-observed failure pattern, and crashing the
chat because the registry is unavailable is strictly worse than
running on normalized display strings.

**Registry identity is decoupled from `TRAINING_REGISTRY_ENABLED`
(round-7 cleanup).** The flag controls one and only one thing:
whether TRAINING resource entries (provider names, URLs, address /
hours metadata) reach the responder for grounded provider mentions.
It does NOT control canonical alias resolution. Two separate concerns
ride one registry file today; the remaining-gaps feature uses only
the alias side:

| Concern | Mechanism | Gated by `TRAINING_REGISTRY_ENABLED`? |
|---|---|---|
| Canonical alias resolution (`registry.lookup(display) → canonical_name`) | Always attempt to load the local YAML for SnapshotCapture + detection identity | **No** — load the registry regardless of the flag |
| TRAINING resource surfacing (`_registry_training_for_gap` → provider+URL payload) | Only populate `training_by_job` when the flag is set | **Yes** — same gate as today, unchanged |

R-1 / R-2 / R-4 changes:
- `_capture_match_snapshot` and `detect_remaining_gaps_intent` MUST
  attempt registry load even when `TRAINING_REGISTRY_ENABLED` is
  false. If the load succeeds, Mode A applies for both. If the load
  fails (file missing, YAML parse error), Mode B / Mode C applies
  for both. The flag is not part of the conditional.
- `_registry_training_for_gap` in R-4 retains its existing flag check
  — that's the resource-surfacing path the flag was designed for.
- A user running with `TRAINING_REGISTRY_ENABLED=false` therefore
  still gets correct canonical resolution (no Mode-B fallback for
  identity), but training providers are NOT named in responses. The
  responder narration reads:
  > *"the next required credential is your G2/G driver's licence"*
  rather than
  > *"the next required credential is your G2/G driver's licence —
  > DriveTest handles the road test"*
- Tests in R-1 / R-2 explicitly cover the flag=false + registry=loaded
  combination: snapshot canonicals are registry-resolved; detection
  still uses Mode A path (a); `training_by_job` is empty in R-4.

Three operating modes (now keyed by registry LOAD state, not the
flag):

Three modes:

**Mode A — registry available (the happy path).** `registry` is a
loaded `TrainingRegistry` instance. Snapshot capture stores both
`display` (engine string) and `canonical` (registry-canonical) per
credential gap. Detection resolves user references through
`registry.lookup`; subtraction compares canonical-to-canonical. This
is the mode the worked example, tests, and prose throughout this doc
describe.

**Mode B — registry UNAVAILABLE at snapshot-capture time.** When the
registry is `None` or fails to load:

- `_capture_match_snapshot` MUST still capture the snapshot. The
  `credential_gaps` entries store `canonical = _normalize(display)`
  where `_normalize` lowercases, strips, and collapses internal
  whitespace. NOT `canonical = display` raw — without normalization,
  trivial casing differences ("310S Automotive Technician License"
  vs "310s automotive technician license") would fail the
  subtraction equality check.
- The snapshot is otherwise identical in shape; the
  `display`/`canonical` split is preserved (canonical is the
  normalized display rather than a registry-resolved value).
- `training_by_job` cannot be populated (it requires the registry).
  The responder runs without provider names; the ungrounded-provider
  policy regex is satisfied because the responder simply doesn't
  mention providers when `training_by_job` is empty.

**Mode C — registry UNAVAILABLE at detection time.** Snapshot was
captured in Mode A or Mode B; at detection time the registry is
`None` (e.g. lazy-load failed). The detector accepts `registry=None`
and falls back to the SAME `_normalize` function for user-message
entity resolution. A user typing "310s licence" still matches the
snapshot entry that stored `canonical = "310s automotive technician
license"` IF the substring after normalization is a prefix or token
overlap of the canonical. Match precision degrades — "310s" alone
won't match "class g driver's license" but also can't disambiguate
"licence" between G2/G and 310S. When ambiguous, detection returns
`kind="confirm"` with `confirmation_target_canonical=None` rather
than guessing — the same path taken for bare "got it".

**Snapshot-Mode-B / detection-Mode-A asymmetry is allowed.** A
snapshot captured without the registry stores normalized canonicals;
if the registry loads later in the session, detection can still run
through `registry.lookup` for user-side resolution — the comparison
key is whatever the snapshot wrote at capture time. The handler does
NOT recapture the snapshot when the registry becomes available; that
would clear accumulated state and is the wrong tradeoff.

**Detector signature contract update.** `registry` is typed
`TrainingRegistry | None` rather than `TrainingRegistry`:

```python
def detect_remaining_gaps_intent(
    message: str,
    snapshot: dict | None,
    registry: TrainingRegistry | None,    # MAY be None
    *,
    accumulated_credentials: list[dict],
    pending_confirmation: dict | None,
    last_discussed_canonical: str | None,
) -> RemainingGapsIntent | None:
    ...
```

**Logging.** Each mode logs once per transition:
- `remaining_gaps_registry_unavailable_at_capture` (Mode B entry)
- `remaining_gaps_registry_unavailable_at_detection` (Mode C entry)
- No log on Mode A — that's the happy path.

These logs feed the observability backlog (we want to know how often
the registry is unavailable in production; the answer should be
"approximately never," and a sustained Mode-B / Mode-C rate is itself
an incident signal).

### 5. Credential vs skill subtraction rules

**v1 SUBTRACTS credentials only.** Reasoning:
- Credentials are discrete, named, registry-tracked
- Skills are diffuse ("Honda experience" is graduated, not yes/no)
- Skill subtraction needs different semantics ("I have some Honda
  experience" — partial; "I worked at Honda for 3 years" — strong)
- Trying to subtract skills without a clear semantics is the variability
  loop we're trying to avoid

v2 (later slice) may extend to skill subtraction with explicit
clarification: "When you say you have Honda experience, is it dealership-
based or independent?" Out of scope here.

### 6. All blockers assumed closed

When `remaining_credentials == []` after subtraction AND
`core_skill_gaps` is non-empty:

Responder narration template:
> *"With the 310S and G2/G in hand, the credentials line up. The
> remaining items are experience and skill gaps — Honda dealership
> work, automotive diagnostics. Want me to look at what shops are
> hiring locally so you can apply with your current experience plus
> the licences you've outlined?"*

**Prompt / fallback rule**: the system MAY label remaining items as
"experience" or "skill" gaps (those categories are derived from
`classify_gap` upstream), but it MUST NOT make claims about how those
gaps are typically closed ("usually come on the job", "best learned
through a course", etc.) unless the TRAINING block carries verified
data for the specific item. Without verified data, the system's role
is to surface the gap and ask, not to prescribe how to close it.

When BOTH credential and skill lists are empty:
> *"If you've got all of those, the job posting itself is the next
> step — check the listing and consider applying."*

**Provider-name grounding still required.** The all-closed branch is
the obvious place to mention SCCC as an application-review resource,
but mentioning ANY provider — including SCCC — requires the same
TRAINING-block grounding rule as the rest of the responder: the name
appears in the reply ONLY when a verified `TRAINING` block entry
carries that provider for a presented gap on this turn.

The all-closed branch has, by definition, no credential gap to attach
TRAINING to. R-4 therefore does NOT populate `training_by_job` for
this branch (there is no `lead_canonical` to feed
`_registry_training_for_gap`). The responder MUST NOT name SCCC,
DriveTest, Sault College, or any other provider in this branch —
the static templated text above is the entire safe surface. Adding a
second sentence that references a provider name is a policy violation
detected by the existing ungrounded-provider regex.

If we later want a "SCCC can review your application" follow-up here,
R-4 must populate a NEW grounded payload field (e.g. an
"application_review_referral" with the SCCC TRAINING entry) so the
provider name is grounded by data, not by handler-side string
literals.

**CANNOT say**: "you qualify", "good fit", "you're qualified", "you're
a match" — same forbidden vocabulary as near-miss. The user has only
CLAIMED completion; we shouldn't certify the match.

### 7. Multi-job behavior

Snapshot stores `lead_job` (top-ranked) and `other_jobs_meta`
(minimal metadata only). Subtraction is computed against `lead_job`
only.

Rationale: "what else?" without role qualification refers to the most
recently discussed role (the lead). Asking the user to disambiguate
between multiple jobs in the same turn adds friction without value;
the top job IS the implicit antecedent.

If the user explicitly asks about a different job ("what about the
truck tech role?"), that's a different feature: job pivot. Out of
scope here; tracked as future work.

### 8. Dedicated move: `explain_remaining_gaps`

New value in `OutcomeMove` Literal. Synthesized by the HANDLER, not
emitted by the planner or router. Engine NOT re-run.

```python
OutcomeMove = Literal[
    ...,                               # existing
    "explain_remaining_gaps",           # NEW
]

# Reason codes on the ArbiterDecision
ARBITER_REASON_REMAINING_GAPS = "remaining_credential_gaps_after_assumption"
ARBITER_REASON_REMAINING_GAPS_RETRACTED = "remaining_credential_gaps_after_retraction"
# Two distinct reasons so transcript tests + telemetry distinguish
# "user added a hypothetical/claim" from "user walked one back".
# Both end at the same final_move (explain_remaining_gaps).

# New ArbiterAction values (logged for transcript tests)
ArbiterAction = Literal[
    ...,                                              # existing
    "handler_synthesized_remaining_gaps",              # NEW
    "handler_synthesized_clarification",               # NEW (for ambiguity path)
]
```

**Why handler synthesis, not router/planner:** `route_from_understanding`
returns `PlannerDecision | None`. The `PlannerMove` Literal cannot
contain `explain_remaining_gaps` without exposing a Pass-2-style outcome
to the planner's contract (the planner is supposed to choose between
ask / explain_gap / proceed_to_match / etc. — not engine-bypass terminal
outcomes). Adding it to `PlannerMove` would weaken the existing dispatch
invariants AND make every planner unit test deal with a value the
planner never emits.

The clean pattern is the same one the gates use: when a deterministic
signal is unambiguous, the handler synthesizes the `ArbiterDecision`
directly and the planner/router/engine all skip. The
`arbiter_action="handler_synthesized_*"` value is the operational
telemetry that surfaces this in logs / transcript tests.

Pass-1 / Pass-2 invariants are preserved: this outcome NEVER comes
through `validate_planner_intent` (Pass 1) or `resolve_match_outcome`
(Pass 2). It's emitted by the handler BEFORE either is called. A new
test in `test_chat_arbiter.py` enumerates inputs and confirms neither
arbiter pass can produce `explain_remaining_gaps` — only handler
synthesis can.

Tone: `warm_supportive`. Same as `explain_gap` and `present_near_miss`.

### 9. Structured `REMAINING_GAPS` prompt block

Serialized into `_build_user_block_v2` when `final_move ==
"explain_remaining_gaps"` AND `inp.remaining_gaps_payload` is set:

```
REMAINING_GAPS:
{
  "role": "310S Licensed Automotive Technician",
  "employer": "Great Lakes Honda",
  "assumed_completed_credentials": [
    {"display":   "310S Automotive Technician License",
     "canonical": "310S automotive technician certification",
     "mode":      "hypothetical"}
  ],
  "remaining_credentials": [
    {"display":   "G2/G driver's license",
     "canonical": "Class G driver's license"}
  ],
  "remaining_core_skills": [
    "Honda vehicle experience",
    "dealership experience",
    "preventative maintenance",
    "automotive diagnostics"
  ],
  "any_hypothetical": true
}
```

`any_hypothetical` is derived: `any(a["mode"] == "hypothetical" for a in
assumed_completed_credentials)`. The flag and the per-entry `mode` are
the SINGLE source of provenance — there is no separate
`is_hypothetical` flag anywhere in the payload, the detection union, or
the truth log. Older drafts of this doc used `is_hypothetical` as a
turn-local boolean; that has been fully replaced by `any_hypothetical`
+ per-entry mode so the accumulated set carries provenance across
turns.

The LLM uses this VERBATIM. Prompt rule (to add to
`OUTCOME_RESPONDER_PROMPT`):

> *"explain_remaining_gaps — The user has indicated (hypothetically or
> in claim) they've completed one or more credentials from the
> recent match. REMAINING_GAPS.assumed_completed_credentials tells you
> what they've claimed; REMAINING_GAPS.remaining_credentials and
> REMAINING_GAPS.remaining_core_skills tell you what's left for the
> stated role.*
>
> *Required shape:*
> 1. *Acknowledge the assumption. When REMAINING_GAPS.any_hypothetical
>    is true, you MUST use conditional tense ("If you've got
>    [credential]…", "Assuming you have…"). Only when any_hypothetical
>    is false may you use past-tense framing ("With your [credential]
>    done…"). The per-entry `mode` field documents which assumptions
>    are which.*
> 2. *Name the next credential gap explicitly if any; otherwise pivot
>    to the skill gaps.*
> 3. *Use ONLY the names supplied in
>    REMAINING_GAPS.remaining_credentials and
>    REMAINING_GAPS.remaining_core_skills. Do NOT explain why each
>    gap matters for this role; do NOT speculate about typical
>    timelines, course duration, transferability, or career impact.
>    The payload supplies names only — any "why it matters" sentence
>    is invented content. Acceptable: "the next required item is
>    {name}". Not acceptable: "{name} matters because most employers
>    expect…".*
> 4. *Close with a next-step offer (training path or job-application
>    direction). The provider names you may use are listed in
>    REMAINING_GAPS-attached TRAINING blocks for the lead remaining
>    credential; do NOT name providers absent from TRAINING.*
>
> *NEVER say "you qualify", "good fit", "good match", "you're
> qualified". The user has only CLAIMED completion. Do not validate
> beyond the claim.*
>
> *NEVER invent gaps outside REMAINING_GAPS. NEVER subtract gaps the
> user didn't claim (i.e., never go beyond
> assumed_completed_credentials).*"

**Deterministic fallback** mirrors the `_present_near_miss_fallback_v2`
shape:

```python
def _present_remaining_gaps_fallback_v2(inp: ResponderV2Input) -> str:
    payload = inp.remaining_gaps_payload or {}
    # ... build narration from the structured block, NO inference
```

### 10. Context expiry / clearing

`staged.last_match_snapshot` is cleared / replaced when:

| Trigger | Action |
|---|---|
| New `present_matches` turn | Replaced with new snapshot |
| New `present_no_match` turn | Cleared (no role to attach to) |
| `target_role_text` changes | Cleared (new role = new gaps) |
| Session reset / new session_id | Naturally gone (per-session field) |

**Explicitly NOT cleared on `redirect_scope`.** A scope redirect is a
temporary diversion (user asked about PR mid-conversation), not a
topic change. The user can return to the career-path conversation
immediately afterward — e.g.

> *Turn N:   "what about my 310S apprenticeship hours toward PR?"*
> *System:   redirect_scope -> SCCC referral*
> *Turn N+1: "okay, back to the job — what else do I need?"*

Clearing the snapshot on the redirect would force a re-bootstrap of
the career conversation that the user clearly hasn't ended. Slice 8's
conversation-context preservation rule already protects this for the
existing presented-context fields; the new snapshot follows the same
discipline.

Other moves that explicitly DO NOT clear:
- `explain_gap` (the original explain-this-credential flow — snapshot
  is still about the same role)
- `explain_remaining_gaps` (the new feature itself — running it must
  not erase its own state)
- `ask_one_clarifying_question` (continued conversation)
- `acknowledge_and_continue` (no topic change)

Snapshots do NOT expire by turn count. A user can ask "what else for
that role?" many turns later as long as the role is still the active
target. This matches enterprise expectations (career-advice
conversations span weeks, not minutes).

### 11. Clarification responder contract (typed payload, not generic ask)

When the handler synthesizes `ask_one_clarifying_question` for either
the `confirm` or `bootstrap` cases above, the existing responder path
for that move is:

```python
# responder.py existing path
if move == "ask_one_clarifying_question":
    if d.ask_slot:
        return _single_ask(d.ask_slot, target)
    return "Tell me a bit more about what you're looking for."  # generic fallback
```

`ask_slot=None` on the synthesized decision means the responder hits
the generic line — which is exactly the wrong text for credential
confirmation or job-search bootstrap. We need a typed payload that the
responder uses to produce the right question.

New field on `ResponderV2Input`:

```python
clarification_payload: dict | None = None
```

Expected shapes (discriminated union by `kind`):

```python
# Credential completion confirmation
{
    "kind": "credential_completion_confirmation",
    "credential_canonical": "310S automotive technician certification",
    "credential_display":   "310S licence",   # what the user typed / saw
}

# Job-search bootstrap (no snapshot when remaining-gap intent fired)
{
    "kind": "bootstrap_match_request",
    # no entity payload; the rendered text is static
}
```

**Critical (round-3 review): the LLM must be skipped entirely for
clarifications, not merely routed around in the fallback.** Updating
only `_fallback_reply_v2` does NOT prevent the LLM from running —
`compose_response_v2` calls `llm.call(...)` FIRST and only invokes the
fallback when the LLM is disabled or its output fails policy. A
clarification rendered by the LLM defeats the purpose: the LLM may
ask the wrong question, paraphrase the credential name, or insert
filler.

The fix is an **early return at the top of `compose_response_v2`**:

```python
def compose_response_v2(inp: ResponderV2Input) -> str:
    # Slice R-5 addition: synthesized clarifications are deterministic
    # and MUST NOT touch the LLM. Renders from the typed payload
    # before any LLM call.
    if inp.clarification_payload:
        return _render_clarification(inp.clarification_payload)

    if not is_enabled():
        return _fallback_reply_v2(inp)
    user_block = _build_user_block_v2(inp)
    reply = call(OUTCOME_RESPONDER_PROMPT, user_block, max_tokens=500)
    if not reply or not _policy_ok_v2(reply, inp):
        return _fallback_reply_v2(inp)
    return reply
```

`_fallback_reply_v2` keeps a parallel `clarification_payload` branch
as defense-in-depth: if a future code path somehow reaches the
fallback with the payload set (e.g., LLM disabled when clarification
fires), the same renderer runs.

`_render_clarification` is deterministic and templated:

| kind | Output template |
|---|---|
| `credential_completion_confirmation` | *"Just to make sure — have you completed your {credential_display}, or are you still working toward it? Want to point you at the right next step."* |
| `bootstrap_match_request` | *"I haven't shown you any local matches yet — want me to look for roles in your target field first, then we can walk through the gaps together?"* |

These templates are STATIC text, not LLM-narrated. The early return
in `compose_response_v2` (above) bypasses BOTH the LLM call AND the
`_policy_ok_v2` regex sweep — the templated strings are trusted by
construction:

- The templates contain no provider names, no URLs, no salary
  language, no scope-violation content, no "you qualify" claims —
  by construction, not by enforcement
- Static review (PR diff) of the templates is the audit mechanism;
  policy regex would be redundant on input we wrote ourselves

This is documented as an explicit policy carve-out for trusted
templated text. Future templates added to `_render_clarification`
MUST be reviewed against the same content rules; that review
happens at code-review time, not runtime.

Tests: `_render_clarification` is unit-tested per kind; responder
integration test confirms the typed payload reaches the renderer and
the generic fallback line is NEVER returned when `clarification_payload`
is set.

### 12. Tests for negation

**Negation patterns MUST NOT subtract.** Each pattern below tests that
the gap remains in the `remaining_credentials` list.

| Test input | Expected behavior |
|---|---|
| "I don't have 310S" | 310S stays in remaining |
| "I haven't got my Class G yet" | Class G stays |
| "I'm missing the 310S" | 310S stays |
| "Without the 310S, what can I do?" | 310S stays |
| "I need to get my 310S first" | 310S stays |
| "I never finished my apprenticeship" | No subtraction (no entity) |

Each is a parametrized test against `detect_remaining_gaps_intent`
asserting the result is `None` (no remaining-gaps pattern matched —
falls through to existing planner/router/engine flow). Equivalently:
the handler step that would have appended `current_turn_claims` to
`staged.last_assumed_completed_credentials` never runs.

### 13. Tests for ambiguity

**Ambiguous phrases ROUTE TO CLARIFICATION, not silent subtraction.**

| Test input (no entity context) | Expected behavior |
|---|---|
| "got it" (no entity context) | `kind="confirm"` with `confirmation_target_canonical=None`, `pending_action="add"` → "Could you say which credential you mean — 310S, G2/G, or something else?" |
| "I have that" (anaphor resolves via `last_discussed_credential_canonical`) | `kind="confirm"` with the resolved canonical, `pending_action="add"` → "Just to make sure — have you completed your {credential}, or are you still working toward it?" |
| "I have that" (no `last_discussed_credential_canonical` AND multiple snapshot credentials) | `kind="confirm"` with `confirmation_target_canonical=None`, `pending_action="add"` → ask which |
| "done with it" | `kind="confirm"` with `pending_action="add"` → ask: "Done meaning you've got it, or done discussing it?" |
| "yes" — prior turn synthesized `kind="confirm"` with `pending_action="add"` | Detection layer reads `staged.pending_credential_confirmation` (NOT `last_discussed_credential_canonical`) → returns `kind="subtract"` with `current_turn_claims=[{canonical: pending.canonical, mode: "claimed"}]`; pending field cleared |
| "yes" — prior turn synthesized `kind="confirm"` with `pending_action="remove"` (retraction confirm) | Detection layer reads `staged.pending_credential_confirmation` → returns `kind="retract"` with `retract_canonical=pending.canonical`; pending field cleared |
| "no" — prior turn synthesized `kind="confirm"` (either action) | Detection returns `kind=None`; pending field cleared; no subtraction or retraction |
| ANY explicit negation ("I don't have 310S", "I haven't got my 310S", "I'm missing the 310S", "actually I don't have 310S") — 310S is in `accumulated_credentials` | Detection returns `kind="confirm"` with `confirmation_target_canonical="310S ..."`, `pending_action="remove"` → "Just to confirm — you don't have your 310S? I'll recalculate against that." The "actually" hedge is NOT required to trigger retraction. |
| Same negation patterns but 310S is NOT in `accumulated_credentials` (post-match, no prior claim) | Detection returns `kind=None`; falls through to existing planner/router (standard explain_gap path for 310S, unchanged from today) |
| "okay" alone | No remaining-gaps pattern; detection returns `kind=None`; falls through to default planner |

The clarification text is templated, not LLM-judged:
> *"Got the [most-recently-discussed-credential] itself, or got the
> path I just explained? Just want to make sure I point you to the
> right next step."*

(If most-recently-discussed-credential isn't tracked, ask:
*"Could you say which credential you mean — was it 310S, G2/G, or
something else?"*)

### Conservative-deterministic-with-clarification refinement

The reviewer added this lock explicitly. Restated:

- Deterministic patterns are CONSERVATIVE — they fire only on high-
  confidence completion language with explicit entity reference
- Patterns whose interpretation is ambiguous (e.g., "got it" alone)
  do NOT silently auto-subtract; instead they route to a focused
  clarification question
- This preserves correctness over coverage: a missed completion claim
  is fixed by the user asking again with clearer language; a false
  completion claim is harder to recover from (the user thinks the
  system understood, but it didn't)

## Worked example: Daniel / weak-resume / Great Lakes Honda

**Setup state (after the match turn):**

```python
staged.last_match_snapshot = {
    "captured_at_turn": 3,
    "lead_job": {
        "job_id": "<honda-uuid>",
        "title": "310s Licensed Automotive Technician",
        "employer": "Great Lakes Honda",
        "credential_gaps": [
            {"display": "310S Automotive Technician License",
             "canonical": "310S automotive technician certification"},
            {"display": "G2/G driver's license",
             "canonical": "Class G driver's license"},
        ],
        "core_skill_gaps": [
            "Honda vehicle experience",
            "dealership experience",
            "preventative maintenance",
            "automotive diagnostics",
            "electrical systems troubleshooting",
        ],
    },
    "other_jobs_meta": [],
}
```

**Turn 5 user message:** *"think I got 310S licence. then what else need to get the job"*

**Detection result (discriminated union, kind="confirm"):**
```python
RemainingGapsIntent(
    kind="confirm",
    confirmation_target_canonical="310S automotive technician certification",
    confirmation_target_display="310S licence",   # what the user typed
    pending_action="add",                         # completion confirm,
                                                  # not retraction
)
```

(Per locked Q1: "think I got X" carries epistemic uncertainty even when
X matches a snapshot credential. The system asks rather than acting on
the claim. See §2 "Uncertainty markers force confirmation".)

**Handler synthesis:**
```python
ArbiterDecision(
    final_move="ask_one_clarifying_question",
    reason_code="confirm_credential_completion",
    tone="warm_supportive",
    arbiter_action="handler_synthesized_clarification",
    ask_slot=None,
)
inp.clarification_payload = {
    "kind": "credential_completion_confirmation",
    "credential_canonical": "310S automotive technician certification",
    "credential_display": "310S licence",
}
# staged.last_discussed_credential_canonical is set to the same
# canonical so that an anaphoric reply on the next turn ("yes I
# got it") resolves correctly.
staged.last_discussed_credential_canonical = (
    "310S automotive technician certification"
)
```

**Target responder output (deterministic; rendered from
`clarification_payload`, no LLM call):**
> *"Just to make sure — have you completed your 310S licence, or are
> you still working toward it? Want to point you at the right next
> step."*

If the user confirms on the next turn ("yes I have it"), the handler
clears `pending_credential_confirmation` and detection consumes the
saved copy (`pending_action="add"`), firing `kind="subtract"` with
`current_turn_claims=[{canonical: "310S ...", mode: "claimed"}]`. If
the user clarifies as hypothetical instead ("if I had it, what
else?"), `_resolve_credential_anaphor` resolves "it" to the persisted
`last_discussed_credential_canonical` and detection fires
`kind="subtract"` with `mode="hypothetical"` in the claim. The
responder's `any_hypothetical` flag becomes `True` whenever the
accumulated set contains a hypothetical entry. Either way, the system
commits to a definite reading before subtracting.

**Turn 5b: hypothetical version** — user message: *"if I had 310S,
what else for this job?"*

**Detection result (discriminated union, kind="subtract"):**
```python
RemainingGapsIntent(
    kind="subtract",
    current_turn_claims=[
        {"canonical": "310S automotive technician certification",
         "mode":      "hypothetical"},
    ],
)
```

**Handler synthesis (Turn 5b: explicit hypothetical):**
- `staged.last_match_snapshot` exists; detection returned `kind="subtract"`
- Handler synthesizes ArbiterDecision directly (planner, router,
  validate_planner_intent, resolve_match_outcome, engine all SKIPPED):
```python
ArbiterDecision(
    final_move="explain_remaining_gaps",
    reason_code=ARBITER_REASON_REMAINING_GAPS,
    tone="warm_supportive",
    arbiter_action="handler_synthesized_remaining_gaps",
    ask_slot=None,
)
# Per §10: staged.last_discussed_credential_canonical updated to the
# LEAD remaining credential after subtraction (Class G in this case),
# so an anaphoric follow-up like "what about that one?" resolves to G2/G.
staged.last_discussed_credential_canonical = "Class G driver's license"
```

**Handler subtraction:**
- credential_gaps minus 310S canonical → just G2/G left
- core_skill_gaps unchanged

**Payload to responder:**
```python
remaining_gaps_payload = {
    "role": "310s Licensed Automotive Technician",
    "employer": "Great Lakes Honda",
    "assumed_completed_credentials": [
        {"display":   "310S Automotive Technician License",
         "canonical": "310S automotive technician certification",
         "mode":      "hypothetical"},
    ],
    "remaining_credentials": [
        {"display":   "G2/G driver's license",
         "canonical": "Class G driver's license"},
    ],
    "remaining_core_skills": [
        "Honda vehicle experience",
        "dealership experience",
        "preventative maintenance",
        "automotive diagnostics",
        "electrical systems troubleshooting",
    ],
    "any_hypothetical": True,   # derived from per-entry modes;
                                # Turn 5b: 310S is hypothetical
                                # -> responder MUST stay conditional
}

# Provider grounding for the LEAD remaining credential (G2/G):
training_by_job = _registry_training_for_gap(
    staged, discovered_gaps=["Class G driver's license"]
)
# -> {gap:Class G driver's license: [
#     {provider: "DriveTest", ...},
#     {provider: "Ontario.ca", ...},
#     {provider: "Sault Community Career Centre", ...},
# ]}
```

**Target responder output (Turn 5b is the hypothetical case, so
`any_hypothetical=True` and narration MUST stay conditional):**
> *"If you've got the 310S in hand, the next required credential for
> the Great Lakes Honda role is your G2/G driver's licence — DriveTest
> handles the road test and Ontario.ca has the full graduated-licensing
> overview.*
>
> *Beyond that, there are experience and skill gaps: Honda-specific
> work, dealership familiarity, preventative maintenance, automotive
> diagnostics. Your tire-and-lube background at Algoma is a starting
> point you can build on once the licences are in place.*
>
> *Want to dig into the G2/G path next, or look at what local shops
> are hiring while you work toward it?"*

The opening "If you've got" — not "With" — is the
`any_hypothetical=True` lock from §3. The earlier draft of this
example said "With the 310S in hand", which would have silently
converted a hypothetical into an apparent claim. Past-tense framing
("With your 310S done…") is reserved for the all-`mode="claimed"`
case (`any_hypothetical=False`).

**Turn 6: accumulated assumption — pure "what else?" with no new entity**

After Turn 5b, `staged.last_assumed_completed_credentials` was
persisted as:
```python
[{"canonical": "310S automotive technician certification",
  "mode": "hypothetical"}]
```

User message: *"what else?"*

**Detection result:**
```python
RemainingGapsIntent(
    kind="subtract",                  # generic remaining-gap request
    current_turn_claims=[],            # nothing claimed in THIS turn
)
```

**Handler subtraction (accumulation reused; mode preserved):**
```python
all_assumed = list(staged.last_assumed_completed_credentials)
# = [{"canonical": "310S ...", "mode": "hypothetical"}]
# No new claims this turn, nothing to append

assumed_canonicals = {a["canonical"] for a in all_assumed}
remaining_credentials = [
    g for g in snapshot["lead_job"]["credential_gaps"]
    if g["canonical"] not in assumed_canonicals
]
# remaining_credentials still excludes 310S. G2/G remains.

any_hypothetical = True       # 310S is still mode=hypothetical
# Persisted state unchanged this turn (no new claims).
```

**Target responder output (still conditional because
`any_hypothetical=True`):**
> *"If you've got the 310S in hand, the remaining required credential
> for the Great Lakes Honda role is your G2/G driver's licence —
> same path we looked at last turn. Want me to walk through what
> shops are hiring while you work toward it, or focus on the G2/G
> timeline?"*

The "If you've got" framing is the **provenance lock** in action: a
hypothetical from a prior turn does NOT silently become an apparent
claim two turns later.

**Turn 7: user adds a second hypothetical**

User message: *"and if I had my G2/G too?"*

**Detection result:**
```python
RemainingGapsIntent(
    kind="subtract",
    current_turn_claims=[
        {"canonical": "Class G driver's license", "mode": "hypothetical"}
    ],
)
```

**Handler subtraction (ordered append-and-dedupe):**
```python
all_assumed = list(staged.last_assumed_completed_credentials)
# = [{"canonical": "310S ...", "mode": "hypothetical"}]

# Class G is new -> append (snapshot order preserved on first-occurrence)
all_assumed.append(
    {"canonical": "Class G driver's license", "mode": "hypothetical"}
)
# all_assumed is now [310S, Class G] in that order

remaining_credentials = []      # both credentials assumed-done
remaining_core_skills = [...]   # unchanged

staged.last_assumed_completed_credentials = all_assumed[:MAX_CRED_GAPS]
any_hypothetical = True         # both entries mode=hypothetical
```

**Target responder output (all-credentials-closed branch, still
conditional, NO provider name):**
> *"If you've got the 310S and G2/G in hand, the credentials would
> line up. The remaining items are experience and skill gaps —
> Honda dealership work, automotive diagnostics, preventative
> maintenance. Want me to look at what shops are hiring locally so
> you'd be ready to apply once the licences are sorted?"*

Note: no SCCC / Sault College / DriveTest reference — `training_by_job`
is empty on this branch (no remaining credential to ground against),
and the responder's ungrounded-provider policy forbids naming
providers without TRAINING-block backing.

Note: "would line up" / "you'd be ready" — conditional throughout
because both credentials are still hypothetical. If the user later
confirms with explicit claims ("I actually have 310S now"), the
entry promotes to `mode="claimed"`; once ALL entries are claimed,
`any_hypothetical=False` and the responder may use past-tense
narration.

**Turn 8: new match presented — accumulation clears**

User message: *"actually find me something different — maybe heavy
equipment"*. The handler updates `staged.target_role_text`, the
engine reruns, a new match is presented. As part of
`_capture_match_snapshot`:

```python
# New present_matches → snapshot replaced, all four fields cleared
staged.last_match_snapshot = new_snapshot
staged.last_assumed_completed_credentials = []
staged.last_discussed_credential_canonical = None
staged.pending_credential_confirmation = None
```

The new snapshot's credentials are evaluated fresh. The user's
prior hypotheticals about the automotive role do NOT carry into
the heavy-equipment match. This is the **conversation state
scoping**: assumptions live only as long as the snapshot they
attached to.

## Tests

### New tests

| Module | Count (est.) | Coverage |
|---|---|---|
| `tests/test_remaining_gaps_detection.py` (new) | ~50 | Each explicit completion pattern returns `kind="subtract"` with `current_turn_claims` containing `{canonical, mode}` entries (mode=`claimed` for explicit, `hypothetical` for "if/once/after"); each PLAIN negation pattern against a snapshot-only entity (NOT in `accumulated_credentials`) returns `kind=None`; each uncertainty pattern returns `kind="confirm"` with `pending_action="add"`; bootstrap case (`snapshot=None` + remaining-gap message) returns `kind="bootstrap"`; **snapshot-anchored identity** (§4.0): every returned canonical is a value present in `snapshot.lead_job.credential_gaps[*].canonical`, NEVER a freshly-resolved registry value; **deterministic token fallback** (§4.3): user input with only generic tokens ("the license") returns None; user input matching multiple snapshot entries returns None; user input matching exactly one entry returns that entry's stored canonical; **negation-against-accumulated retraction (v8)**: ALL explicit negations targeting an accumulated entity ("I don't have X", "I haven't got my X", "I'm missing the X", "without X", "actually I don't have X") return `kind="confirm"` with `pending_action="remove"` — the "actually" hedge is NOT required; same patterns targeting a snapshot-only entity (not accumulated) return `kind=None`; **ordering invariant** (§R-2 step list): single message with both negation and completion patterns fires retraction confirmation FIRST; bare anaphor resolves against `last_discussed_credential_canonical` when it's set; falls back to the snapshot's first credential entry when last_discussed is None (immediate post-match turn before anything specific has been discussed) — matches the §2 resolver lifecycle; multi-entity claims; **pending_credential_confirmation consumption (saved-copy semantics)**: detector NEVER mutates the passed dict; affirmative with `action="add"` returns `kind="subtract"` with `current_turn_claims=[{canonical, mode:"claimed"}]`; affirmative with `action="remove"` returns `kind="retract"` with `retract_canonical=...`; negative returns `kind=None`; unrelated falls through to fresh detection; **Mode C tests** (§4a): detector accepts `registry=None` and uses token-fallback resolver; ambiguous input returns `kind="confirm"` with `confirmation_target_canonical=None` rather than guessing; **flag-decoupled identity**: with `TRAINING_REGISTRY_ENABLED=False` and registry loaded, canonical resolution still works (the flag gates resource surfacing in R-4, NOT identity) |
| `tests/test_remaining_gaps_handler.py` (new) | ~15 | Snapshot capture on present_matches; snapshot + accumulation + last_discussed + pending all cleared together on present_no_match / target_role_text change / session reset; NOT cleared on redirect_scope, explain_gap, explain_remaining_gaps, ask_one_clarifying_question, acknowledge_and_continue; subtraction is request-scoped (staged.skills never mutated); payload built correctly; **accumulation persists across turns**: simulate Turn N "if I had 310S" then Turn N+1 "what else?" and confirm 310S stays subtracted; **mode preservation**: hypothetical entry stays hypothetical across "what else?" turns; promotion: hypothetical → claimed when user explicitly confirms; cap drops latest entries first (order preserved); **order determinism**: serialize → sign → unsign → deserialize round-trip preserves accumulated order; full signed-cookie value under 3800 bytes with max snapshot + max accumulated + pending + last_discussed |
| `tests/test_chat_arbiter.py` (extend) | ~6 | `explain_remaining_gaps` is in `OutcomeMove`; new reason code constant exported; INVARIANT — neither `validate_planner_intent` (Pass 1) nor `resolve_match_outcome` (Pass 2) can produce `explain_remaining_gaps` (the new outcome reaches the responder ONLY via handler synthesis); new `ArbiterAction` values reachable |
| `tests/test_chat_responder_v2.py` (extend) | ~8 | REMAINING_GAPS block serialized in user_block on the new outcome only; forbidden phrases ("you qualify", "good fit") rejected; deterministic fallback shape; provider grounding via training_by_job for the next credential |
| `tests/test_chat_transcripts.py` (extend) | 1 scenario | Daniel/weak-resume scenario end-to-end |

### Live regression tests (manual, post-build)

1. Daniel weak-resume → "think I got 310S, what else?" →
   `ask_one_clarifying_question` with the focused confirmation:
   *"Just to make sure — have you completed your 310S licence, or
   are you still working toward it?"* (locked Q1: uncertainty
   forces confirmation, NOT silent subtraction).
   System sets `pending_credential_confirmation = {canonical: "310S
   ...", action: "add"}`.
2. Continue from #1 → reply "yes I have it" → detection consumes
   `pending_credential_confirmation` → `explain_remaining_gaps`
   with 310S subtracted (mode=`claimed` since the user explicitly
   confirmed); reply names G2/G + skill gaps; NO "you qualify"
3. Same flow → "if I had 310S, what else?" → explicit hypothetical →
   `explain_remaining_gaps` naming G2/G + skill gaps; reply opens
   with conditional "If you've got the 310S in hand…" (because
   `any_hypothetical=True`)
4. **Multi-turn accumulation (the core feature)**: continue from #3
   → "what else?" with no entity → `explain_remaining_gaps` still
   shows G2/G + skill gaps; reply STILL conditional ("If you've
   got the 310S in hand…") because 310S is in the accumulated set
   with mode=`hypothetical`
5. Continue from #4 → "and if I had my G2/G too?" →
   `explain_remaining_gaps` with all credentials closed; reply
   pivots to skill gaps; STILL conditional ("If you've got those
   two…") because both are hypothetical. Reply contains NO provider
   names (no SCCC, no DriveTest) — `training_by_job` is empty on the
   all-closed branch and the ungrounded-provider regex would reject
   any provider mention. Test asserts the ungrounded-provider regex
   passes on this reply
6. Same flow as #5 (310S+G2/G hypothetically closed) → "I don't have
   310S" — the plain wording, no "actually" hedge. Detection layer
   sees 310S is in `accumulated_credentials` and the message is an
   explicit negation against it → returns `kind="confirm"` with
   `pending_action="remove"`. Handler synthesizes "Just to confirm —
   you don't have your 310S? I'll recalculate against that." and
   sets `pending_credential_confirmation = {canonical: "310S ...",
   action: "remove"}`. (NEW in v8: previously this scenario was
   documented as routing through explain_gap, which would have left
   the stale hypothetical assumption in place — a recoverability
   failure mode.)
6b. Bare-resume contrast: starting from the post-match state with
    NO accumulated assumptions, → "I don't have 310S" — 310S is in
    `snapshot.credential_gaps` but NOT in `accumulated_credentials`.
    Detection returns `kind=None`; falls through to normal routing;
    standard explain_gap path runs for 310S. This is the original
    intent of "negation must not subtract" — preserved for the
    not-yet-assumed case.
7. **Retraction**: continue from #5 (310S+G2/G hypothetically closed)
   → "actually I don't have 310S" → handler emits
   `kind="confirm"` clarification "Just to confirm — you don't have
   your 310S? I'll recalculate against that." Sets
   `pending_credential_confirmation = {canonical: "310S ...",
   action: "remove"}`.
8. Continue from #7 → "yes that's right" → 310S removed from
   accumulated; next turn re-shows 310S as remaining
9. Same flow → "got it" alone (no entity) → focused clarification
   asking which credential, NOT silent subtraction
10. After a target_role change → all four fields cleared;
    "what else?" routes to bootstrap clarification ("I haven't
    shown you any local matches yet — want me to look for roles
    first?")
11. After a `redirect_scope` turn (user asked about PR) → snapshot
    AND accumulated assumptions both PRESERVED; following turn
    "back to the job, what else do I need?" still works against the
    original match with the same hypothetical credentials still
    accumulated

## Build slices

Each independently shippable. Build → test → review → next.

### Slice R-1 — Snapshot data structure + capture + state fields (~1.5 hours)

Add the four new StagedProfile fields together (they share a
lifecycle and are atomically cleared with the snapshot):

```python
last_match_snapshot: dict | None = None
last_assumed_completed_credentials: list[dict] = field(default_factory=list)
last_discussed_credential_canonical: str | None = None
pending_credential_confirmation: dict | None = None
```

- Replace `_capture_presented_context` with `_capture_match_snapshot`
  (writes the new structured field; LEGACY fields can stay populated
  in parallel until R-5 to avoid breaking existing fallbacks)
- Canonical normalization for each credential gap via `registry.lookup`
- **Clearing rules wired (NOT including scope_redirect):**
  - new `present_matches` → snapshot replaced; the other three
    fields cleared to empty
  - `present_no_match` / `target_role_text` change / session reset
    → snapshot cleared; the other three cleared too
  - `redirect_scope` → NOT cleared (snapshot survives temporary
    scope diversions per locked §10)
  - `explain_gap`, `explain_remaining_gaps`, `ask_one_clarifying_question`,
    `acknowledge_and_continue` → NOT cleared
- Cookie-size test against `CookieSessionStore.save()` (binding
  test: signed value < 3800 bytes with all four fields at max)
- **Snapshot capture must work when `registry is None`** (Mode B in
  §4a). `_capture_match_snapshot` MUST NOT raise / return None
  simply because the registry failed to load. Required behavior:
  ```python
  def _capture_match_snapshot(staged, results, registry):
      def _resolve(display: str) -> str:
          if registry is None:
              return _normalize_canonical(display)  # lowercase + strip
          hit = registry.lookup(display)
          return hit.canonical_name if hit else _normalize_canonical(display)
      ...
  ```
  Snapshot fields are populated identically in either mode (the
  `display`/`canonical` split is preserved); the only difference is
  what `canonical` resolves to. Emit
  `remaining_gaps_registry_unavailable_at_capture` log ONCE per
  capture when entering Mode B.
- **Dedupe `credential_gaps` by resolved canonical, preserving first
  occurrence** (round-9 R-1 invariant). The engine may emit two
  display strings that alias to the same registry credential (e.g.
  "G2 driver's licence" + "Class G driver's license" both resolve
  to canonical "Class G driver's license"). Storing both in the
  snapshot would duplicate remaining gaps in the responder block AND
  create ambiguous identity for subtraction (which `display` does
  the user's "I have my G" refer to?). Required behavior:
  ```python
  seen_canonicals: set[str] = set()
  credential_gaps: list[dict] = []
  for raw in engine_credential_gaps[:MAX_CRED_GAPS]:
      canonical = _resolve(raw.display)
      if canonical in seen_canonicals:
          continue                                # dedupe; keep first
      seen_canonicals.add(canonical)
      credential_gaps.append({"display": raw.display,
                              "canonical": canonical})
  ```
  Dedupe runs AFTER the cap so the cap counts unique credentials,
  not raw engine entries. `core_skill_gaps` are NOT deduped (they're
  display-only strings; duplicates would already be filtered upstream
  by `classify_gap`).
- Tests for Mode B snapshot capture: registry=None → snapshot still
  built; credential_gaps entries have `canonical` populated with
  normalized display text; subtraction comparison still works
  case-insensitively against user input
- Test for dedupe invariant: engine emits two display strings whose
  registry canonicals collide ("G2 driver's licence" + "Class G
  driver's license" both → "Class G driver's license"); snapshot
  stores exactly ONE credential_gaps entry, the FIRST one
  encountered (preserves engine ranking order)
- **Defensive deserialization in `StagedProfile.from_json`.** The
  signed cookie is HMAC-verified, so its contents are tamper-resistant
  for the lifetime of the signing key. But (a) the key can rotate,
  (b) a forged cookie that somehow passes signature check should not
  crash the handler, (c) a malformed older session blob from an
  intermediate deploy should degrade gracefully — same discipline as
  the existing `from_json` (lines 273-291) which drops wrong-typed
  fields rather than crashing. R-1 MUST add per-field validation for
  the four new fields, applied AFTER `json.loads` and BEFORE
  `cls(**data, skills=skills)`:

  - `last_match_snapshot`:
    - if not a dict → set to `None`
    - if `lead_job` missing / not a dict → set to `None`
    - if `lead_job.title` / `lead_job.employer` / `lead_job.job_id`
      not strings → truncate-or-default to `""` / `None` / `""`
    - cap `lead_job.title` to `MAX_TITLE_CHARS=80`,
      `lead_job.employer` to `MAX_EMPLOYER_CHARS=60`
    - cap `credential_gaps` list to `MAX_CRED_GAPS=5`; drop entries
      that aren't `{display: str, canonical: str}` shape; truncate
      each string to 80
    - cap `core_skill_gaps` list to `MAX_SKILL_GAPS=5`; drop
      non-string entries; truncate each to 80
    - cap `other_jobs_meta` list to `MAX_OTHER_JOBS=3`; drop
      malformed entries

  - `last_assumed_completed_credentials`:
    - if not a list → set to `[]`
    - drop entries that aren't dicts
    - drop entries where `canonical` is not a string
    - drop entries where `mode` is not in `{"hypothetical", "claimed"}`
      (NOT silently coerce — an unknown mode could come from a future
      version's enum extension, but the v1 handler can't trust it)
    - truncate `canonical` to 80
    - cap list to `MAX_CRED_GAPS=5` after filtering (drop tail, NOT
      random — preserves order)

  - `last_discussed_credential_canonical`:
    - if not a string and not `None` → set to `None`
    - truncate to 80

  - `pending_credential_confirmation`:
    - if not a dict and not `None` → set to `None`
    - if dict but `canonical` not a string OR `action` not in
      `{"add", "remove"}` → set to `None` (malformed pending state
      is dropped entirely, NOT coerced — the only safe action on a
      malformed pending field is to forget the question was ever
      asked)
    - truncate `canonical` to 80

  Implementation hint: mirror the existing list-wrong-type-defaults
  loop pattern in `from_json` (line 283-286), extended for the new
  dict / string validations.
- Tests in `test_chat_handler_v2.py` + `test_session_staging.py`,
  including: each malformed-shape variant deserializes to the
  documented default; a complete valid roundtrip preserves all four
  fields exactly; an unknown `mode` value drops the entry rather than
  crashing; an unknown `action` value clears `pending_credential_confirmation`

### Slice R-2 — Remaining-gaps detection module (~1.5 hour)

- New `skillbridge/chat/remaining_gaps.py`
- Complete signature (all four state inputs required for correct
  detection of retraction, pending consumption, and anaphor
  resolution):

```python
def detect_remaining_gaps_intent(
    message: str,
    snapshot: dict | None,
    registry: TrainingRegistry | None,        # may be None (Mode C, §4a)
    *,
    accumulated_credentials: list[dict],     # current staged.last_assumed_completed_credentials
    pending_confirmation: dict | None,        # current staged.pending_credential_confirmation
    last_discussed_canonical: str | None,     # current staged.last_discussed_credential_canonical
) -> RemainingGapsIntent | None:
    ...
```

  Returns the discriminated union shape (`kind="subtract" | "retract"
  | "confirm" | "bootstrap"` or `None`). **There is NO `ambiguous`
  flag** — ambiguity is expressed by returning `kind="confirm"` with
  appropriate `confirmation_target` + `pending_action` fields.
- Detection ordering (first-match wins, top to bottom). Note the
  ordering: pending is FIRST (a yes/no in flight must be consumed
  before anything else); retraction-against-accumulated is SECOND
  (any explicit negation targeting an already-assumed credential
  must walk back the assumption, NOT route to explain_gap); standard
  negation is THIRD (only fires when the negated entity is NOT in
  accumulated state):
  1. **Consume `pending_confirmation` if set.** An affirmative answer
     ("yes" / "yeah" / "I do") executes the pending action: if
     `action="add"`, returns `kind="subtract"` with
     `current_turn_claims=[{canonical: pending.canonical,
     mode: "claimed"}]`; if `action="remove"`, returns
     `kind="retract"` with `retract_canonical=pending.canonical`. A
     negative ("no" / "not yet") returns `kind=None`. An unrelated
     message falls through to fresh detection on this turn (the
     pending field was already cleared by the handler — §2). Pending
     consumption is independent of accumulated state.
  2. **Retraction against `accumulated_credentials` — ANY explicit
     negation language targeting an assumed credential.** If the
     message matches any of the negation patterns documented in §2
     ("I don't have X", "I haven't got my X", "I'm missing the X",
     "actually I don't have X", "I never finished my X") AND X
     resolves to a canonical PRESENT in `accumulated_credentials`,
     return `kind="confirm"` with `pending_action="remove"` and
     `confirmation_target_canonical = matched canonical`. This is
     the single-confirm safety layer before removing. **The
     "actually" hedge is NOT required**: any explicit negation of an
     already-assumed credential MUST initiate retraction
     confirmation, not silently leave the stale assumption in place.
     Without this rule, the user can't recover from an earlier
     hypothetical except by ending the conversation and starting
     fresh. Step 2 only fires when the negated entity is in
     `accumulated_credentials`; otherwise step 3 takes over.
  3. **Standard negation for non-accumulated entities.** If the
     message matches a negation pattern AND the entity is NOT in
     `accumulated_credentials` (or no entity is resolvable), return
     `kind=None`. Examples: "I don't have 310S" when 310S has never
     been claimed → no remaining-gaps action, the standard
     planner/router routes to explain_gap for 310S as today.
     "Without 310S, what can I do?" → same. "I never finished my
     apprenticeship" → no entity → `kind=None`.
  4. Uncertainty markers ("think I got X", "pretty sure I have X")
     against snapshot or accumulated → `kind="confirm"` with
     `pending_action="add"` and confirmation_target = the matched
     credential
  5. Explicit completion ("I have X", "I got my X", "I passed X")
     → `kind="subtract"` with the claim in `current_turn_claims`
     (mode=`claimed`)
  6. Explicit hypothetical ("if I had X", "assume I have X",
     "after I get X") → `kind="subtract"` with mode=`hypothetical`
  7. Generic remaining-gap request ("what else?", "anything else?")
     → if snapshot exists, return `kind="subtract"` with
     `current_turn_claims=[]` (uses accumulated state only); if
     snapshot is None, return `kind="bootstrap"`
- Bare-anaphor resolution via `_resolve_credential_anaphor` (uses
  `last_discussed_canonical` for recency)
- Telemetry log for unrecognized "completion-ish" patterns (backlog
  for future additions — same pattern as `unknown_gap=` logging)
- ~45 unit tests covering:
  - Each explicit completion pattern returns `kind="subtract"` with
    `mode="claimed"`; each explicit hypothetical returns
    `mode="hypothetical"`
  - Each plain negation pattern targeting a snapshot-only entity
    (NOT in `accumulated_credentials`) returns `kind=None`
  - **Negation-against-accumulated retraction (NEW v8)**: each of
    "I don't have X", "I haven't got my X", "I'm missing the X",
    "without X", "actually I don't have X" against a credential
    in `accumulated_credentials` returns `kind="confirm"` with
    `pending_action="remove"` — the "actually" hedge is NOT
    required; the rule is "any explicit negation targeting an
    already-assumed credential triggers retraction confirmation"
  - **Ordering invariant**: a single message that contains BOTH a
    negation pattern AND a completion pattern (e.g. "actually I
    don't have 310S but I do have G2/G") fires retraction
    confirmation first because step 2 wins over step 5 in the
    detection ordering. Test the exact precedence: pending → retract
    → standard negation → uncertainty → completion → hypothetical →
    generic
  - Each uncertainty pattern returns `kind="confirm"` with
    `pending_action="add"`
  - Bootstrap case fires correctly when snapshot=None and the
    message is a remaining-gap-shaped request
  - **pending_confirmation consumption (saved-copy semantics)**: the
    detector NEVER mutates the passed `pending_confirmation` dict;
    affirmative with `action="add"` returns `kind="subtract"`
    mode=`claimed`; affirmative with `action="remove"` returns
    `kind="retract"`; negative returns `kind=None`; unrelated falls
    through to fresh detection
  - **Mode C (`registry=None`) tests**: detector accepts
    `registry=None` and does not raise; entity resolution falls
    back to normalized-substring matching; ambiguous user input
    returns `kind="confirm"` with `confirmation_target_canonical=None`
    rather than guessing
  - **No silent mutation**: even on `kind="retract"` /
    `kind="confirm"`, the detector returns a fresh dataclass — the
    handler is the sole owner of StagedProfile writes

### Slice R-3 — Handler synthesis (~1 hour)

NOT router/planner wiring. Per the locked architectural decision in
§8, the handler synthesizes the ArbiterDecision directly when the
deterministic detection in R-2 fires. Planner and engine both skip.

- Add `"explain_remaining_gaps"` to `OutcomeMove` Literal
- Add `"handler_synthesized_remaining_gaps"` and
  `"handler_synthesized_clarification"` to `ArbiterAction` Literal
- Add `ARBITER_REASON_REMAINING_GAPS` and
  `ARBITER_REASON_REMAINING_GAPS_RETRACTED` constants
- **Legacy action mapping for the new outcome.** The handler exposes a
  v1 `next_action` label on every response so analytics / legacy
  frontend consumers see a familiar value. `_FINAL_MOVE_TO_LEGACY_ACTION`
  in `skillbridge/chat/handler.py` (around line 1383) is the lookup
  table; an outcome NOT in this table falls through to
  `ACTION_ASK_QUESTIONS`, which would be WRONG for
  `explain_remaining_gaps` (it's a match continuation, not a
  clarifying question). R-3 MUST add:
  ```python
  _FINAL_MOVE_TO_LEGACY_ACTION = {
      ...,                                                       # existing
      "explain_remaining_gaps": intake_state.ACTION_PRESENT_MATCHES,
  }
  ```
  Same legacy value as `explain_gap` / `present_matches` (a continuation
  of the match conversation, not a topic-change ask). A handler test
  asserts `_final_move_to_legacy_action("explain_remaining_gaps") ==
  intake_state.ACTION_PRESENT_MATCHES` so a future refactor that drops
  the mapping is caught by CI.
- New handler step (between `build_truth_summary` and the existing
  planner/router dispatch). The handler MUST:
  1. Save `staged.pending_credential_confirmation` to a local
     `saved_pending` variable AND set the StagedProfile field to
     `None` BEFORE the detection call (pending-clear ownership, §2).
  2. Call `detect_remaining_gaps_intent` inside a try/except. Detection
     is a regex layer and should be infallible, but it depends on
     `registry.lookup` for alias resolution. A registry-loading
     failure or any unexpected exception from the regex layer MUST
     NOT lose the pending confirmation — the user already answered
     a question we asked; dropping that state and routing through
     the default planner would re-ask the same question and look
     broken. On exception: restore `saved_pending` to the
     StagedProfile field, log the failure, and fall through to the
     existing planner/router/engine flow.
  3. Call `detect_remaining_gaps_intent` with ALL FOUR state
     inputs (the detector is pure and cannot read StagedProfile
     directly):

```python
saved_pending = staged.pending_credential_confirmation
staged.pending_credential_confirmation = None

try:
    intent = detect_remaining_gaps_intent(
        message,
        staged.last_match_snapshot,                # may be None
        registry,                                   # may be None on
                                                    #   registry-load
                                                    #   failure (§ below)
        accumulated_credentials=
            staged.last_assumed_completed_credentials,
        pending_confirmation=saved_pending,
        last_discussed_canonical=
            staged.last_discussed_credential_canonical,
    )
except Exception as exc:                            # pragma: deliberate
    # Detection failure is non-fatal. Restore the pending state so
    # the user's "yes" / "no" is still consumable on the NEXT turn
    # (the system effectively re-asks the question by leaving the
    # pending dict intact). Then continue through normal routing so
    # the user gets *some* reply, not a 500.
    log.exception("remaining_gaps_detection_failed: %s", exc)
    staged.pending_credential_confirmation = saved_pending
    intent = None    # forces the None-guard branch below
```

- **Guard against `intent is None` BEFORE the match block.**
  `detect_remaining_gaps_intent` returns `RemainingGapsIntent | None`;
  the None return is the "no remaining-gaps pattern matched" exit. A
  bare `match intent.kind` on `None` raises `AttributeError` at
  runtime, crashing the handler. The branch shape MUST be:

```python
if intent is None:
    # No remaining-gaps pattern; fall through to existing
    # planner/router/engine flow unchanged. Pending was already
    # cleared by the save-and-clear step above, so a stale pending
    # question never crosses into the next turn.
    continue_normal_dispatch()
else:
    # All four union shapes are guaranteed truthy intent values
    # with a string `.kind` attribute.
    match intent.kind:
        case "subtract":
            # snapshot is required for subtract; detection guarantees it
            synthesize_remaining_gaps_decision(
                staged, intent, snapshot,
                reason=ARBITER_REASON_REMAINING_GAPS,
            )
            skip_planner_router_engine()
        case "retract":
            # snapshot is required for retract; detection guarantees it.
            # Handler REMOVES intent.retract_canonical from accumulated,
            # then synthesizes explain_remaining_gaps with the
            # retraction-specific reason code so transcript tests +
            # telemetry distinguish "added" from "walked back".
            staged.last_assumed_completed_credentials = [
                a for a in staged.last_assumed_completed_credentials
                if a["canonical"] != intent.retract_canonical
            ]
            synthesize_remaining_gaps_decision(
                staged, intent, snapshot,
                reason=ARBITER_REASON_REMAINING_GAPS_RETRACTED,
            )
            skip_planner_router_engine()
        case "confirm":
            # snapshot may or may not exist; clarification doesn't
            # depend on it. Pending state is set ONLY when
            # intent.confirmation_target_canonical is a non-empty string
            # (architecture diagram case "confirm" guard).
            synthesize_clarification_decision(staged, intent)
            skip_planner_router_engine()
        case "bootstrap":
            # snapshot is None by definition of this kind
            synthesize_clarification_decision(staged, intent)
            skip_planner_router_engine()
        case _:
            # Defensive: an unrecognised intent.kind value (e.g. from
            # a future detector version) MUST fall through, NOT raise.
            # A new kind is a feature-gate problem, not a 500-error
            # problem. Log a warning so the gap is visible.
            log.warning("remaining_gaps_unknown_kind=%r", intent.kind)
            continue_normal_dispatch()
```

The four-arm match-block is exhaustive over the documented union; the
`case _:` arm is a defense-in-depth catch for future extensions that
would otherwise silently fall through Python's match semantics. The
None case is hoisted OUT of the match so static readers see "intent
may be None" as a first-class branch, not as a hidden discriminated
shape.

- Pattern mirrors gates.py exactly: deterministic signal → handler
  synthesizes ArbiterDecision → responder runs → return
- ~8 handler tests:
  - bootstrap test: snapshot=None + remaining-gap message →
    `kind="bootstrap"` → clarification synthesized
  - arbiter invariant: the new outcome value reaches the responder
    without going through `validate_planner_intent` or
    `resolve_match_outcome`
  - **pending save-and-clear test**: a turn that arrives with
    `staged.pending_credential_confirmation` set ALWAYS exits the
    handler with the field cleared, regardless of how detection
    classified the message (yes / no / unrelated / kind=None — all
    four exit states verified)
  - **case "retract" reachability**: simulate the two-turn retraction
    flow (turn N "actually I don't have 310S" while 310S is in
    accumulated → confirm synthesized with pending_action="remove";
    turn N+1 "yes that's right" → `kind="retract"` → 310S removed
    from accumulated; outcome reason is
    `ARBITER_REASON_REMAINING_GAPS_RETRACTED`)
  - **canonical=None guard**: a turn that returns `kind="confirm"`
    with `confirmation_target_canonical=None` MUST leave
    `staged.pending_credential_confirmation` as `None` (not store a
    pending entry with `canonical=None`)
  - legacy mapping test: see the
    `_FINAL_MOVE_TO_LEGACY_ACTION["explain_remaining_gaps"]` pin
    above
  - **None-guard**: detector returns `None` → handler routes through
    `continue_normal_dispatch()`; NO `AttributeError` raised; the
    request completes with an existing-planner reply
  - **Detector-failure pending restore**: monkeypatch
    `detect_remaining_gaps_intent` to raise `RuntimeError`; the
    handler catches the exception, restores
    `staged.pending_credential_confirmation` to its pre-call value,
    logs `remaining_gaps_detection_failed`, sets `intent = None`,
    and continues through normal routing — the response is non-empty
    and the pending state is preserved for the next turn
  - **Unknown intent.kind catch-all**: synthesize a fake intent with
    `kind="future_extension"` → handler hits `case _:`, logs
    `remaining_gaps_unknown_kind`, and continues through normal
    routing

### Slice R-4 — Handler subtraction + payload build (~2 hours)

R-4 covers BOTH paths into `explain_remaining_gaps`: the additive
`kind="subtract"` path AND the destructive `kind="retract"` path. The
payload SHAPE is the same for both; only the accumulation mutation
and the ArbiterReason differ. Both end at the same `final_move`.

**Subtract path (`kind="subtract"`):**

- Handler computes `remaining_credentials` and `remaining_core_skills`
  using the **ordered append-and-dedupe** of
  `staged.last_assumed_completed_credentials` + current-turn claims
  (see §3 algorithm). Snapshot order preserved on first occurrence;
  hypothetical → claimed promotion handled.
- Handler persists the updated `last_assumed_completed_credentials`
  back to staged BEFORE rendering — so the next turn sees the union.
  Cap to `MAX_CRED_GAPS` by dropping latest entries first (NOT
  random — `[:MAX_CRED_GAPS]` on the ordered list).
- ArbiterReason: `ARBITER_REASON_REMAINING_GAPS`.

**Retract path (`kind="retract"`):**

- Removal already executed in R-3's `case "retract"` branch (the
  accumulated list is filtered IN PLACE before R-4 runs). R-4 reads
  the post-removal `staged.last_assumed_completed_credentials` and
  builds the payload from there — same call into the payload-builder
  as the subtract path, no special case in the math:
  ```python
  accumulated = staged.last_assumed_completed_credentials
  assumed_canonicals = {a["canonical"] for a in accumulated}
  remaining_credentials = [
      g for g in snapshot["lead_job"]["credential_gaps"]
      if g["canonical"] not in assumed_canonicals
  ]
  # The retracted canonical re-appears in remaining_credentials by
  # construction: it's no longer in `accumulated`, so the filter
  # keeps it.
  ```
- ArbiterReason: `ARBITER_REASON_REMAINING_GAPS_RETRACTED` (distinct
  from the add path so transcripts + telemetry can tell them apart).
- `last_discussed_credential_canonical` is updated to the LEAD
  remaining credential AFTER recomputation. When the retracted
  credential is the first remaining entry (the common case: user
  walks back the credential they were just discussing), the field
  points at the retracted canonical — so an anaphoric follow-up
  like "what about that one?" correctly resolves to the credential
  the user just put back on the list.
- The responder MAY use a recalculation acknowledgement template
  on the retraction reason ("Okay, recalculating with the {display}
  back on the list…") instead of the standard subtract framing.

**Shared payload build (both paths):**

- Handler computes `any_hypothetical = any(a["mode"] == "hypothetical"
  for a in accumulated)` and includes it in the payload. The
  responder uses this flag to choose conditional vs definite tense
  in narration. On the retract path, `any_hypothetical` is computed
  over the POST-removal accumulated set — removing the last
  hypothetical entry flips the flag to `False` automatically.
- Handler populates `training_by_job` for the LEAD remaining
  credential via `_registry_training_for_gap(staged,
  discovered_gaps=[lead_canonical])` — same grounding pattern as
  explain_gap turns, and (importantly) gated by
  `TRAINING_REGISTRY_ENABLED` the same way explain_gap is. When the
  flag is off, `training_by_job` stays empty and the responder
  cannot name providers; canonical identity from §4.0 is unaffected
  because canonical resolution is decoupled from the flag (§4a).
  Same logic on retract: the newly re-emerged credential is the
  lead, and its training resources are surfaced when the flag is on.
- Builds `remaining_gaps_payload` dict with `assumed_completed_credentials`
  (list with mode) + `remaining_credentials` + `remaining_core_skills`
  + `any_hypothetical` flag + `role` + `employer`.
- Passes to `ResponderV2Input.remaining_gaps_payload` (new field).
- Updates `staged.last_discussed_credential_canonical` to the LEAD
  remaining credential after recomputation (so anaphors on the next
  turn point at the right thing — applies to BOTH paths).
- Truth log gains:
  - `remaining_gaps_intent=<subtract|retract|confirm|bootstrap|none>`
  - `current_turn_claims_count=<int>` (claims THIS turn)
  - `accumulated_credentials_count=<int>` (total after append+dedupe)
  - `any_hypothetical=<bool>` (provenance flag passed to responder)
  - `remaining_credentials_count=<int>`
  - `remaining_skills_count=<int>`
  - `pending_action=<add|remove|none>` (what was confirmed, if any)
- ~12 handler tests including:
  - **Multi-turn accumulation: Turn N adds 310S (hypothetical),
    Turn N+1 "what else?" still subtracts 310S** (the regression
    that round-3 review flagged)
  - Promotion: hypothetical 310S → claimed when user explicitly
    confirms; entry mode updates, no duplicate
  - Cap behavior: 6th unique canonical drops the latest (not random)
  - Order preservation across cookie roundtrip (serialize → sign →
    unsign → deserialize → same order)
  - **Retract path — accumulated removal**: starting state
    accumulated=`[{310S, hypothetical}, {Class G, hypothetical}]`;
    `intent.kind="retract"`, `retract_canonical="310S ..."`;
    post-condition `staged.last_assumed_completed_credentials ==
    [{Class G, hypothetical}]`
  - **Retract path — payload recomputation**: payload's
    `remaining_credentials` contains the retracted credential as
    its first entry; `assumed_completed_credentials` no longer
    contains it
  - **Retract path — ArbiterReason**: synthesized decision carries
    `ARBITER_REASON_REMAINING_GAPS_RETRACTED`, not
    `ARBITER_REASON_REMAINING_GAPS` (transcript-level distinguishability)
  - **Retract path — `last_discussed` follow-anaphora**: after a
    retract that re-emerges credential X, `staged.last_discussed_
    credential_canonical == X` so a next-turn "what about that one?"
    resolves to X
  - **Retract path — `any_hypothetical` re-derivation**: starting
    state `[{310S, hypothetical}, {Class G, claimed}]`; retract 310S;
    payload `any_hypothetical == False` (the last hypothetical entry
    was the one removed)
  - **Retract path — `training_by_job` regrounded**: after retract,
    `training_by_job` is populated for the re-emerged credential
    (so the responder can name providers without policy violation)
  - **Retract idempotency / missing canonical**: `kind="retract"`
    against a canonical NOT in the accumulated set leaves the list
    unchanged (the filter is a keep-if-not-equal pass; nothing to
    remove is a no-op, not an error)

### Slice R-5 — Responder narration + fallback + prompt + clarification renderer (~2 hours)

- Add `remaining_gaps_payload: dict | None` field to `ResponderV2Input`
- Add `clarification_payload: dict | None` field to `ResponderV2Input`
- `_build_user_block_v2` serializes `REMAINING_GAPS:` block when
  `final_move == "explain_remaining_gaps"` AND `remaining_gaps_payload`
  is set
- Update `OUTCOME_RESPONDER_PROMPT` with `explain_remaining_gaps`
  narration shape + CANNOT list (no "you qualify", no claims about
  how skill gaps are typically closed without verified data)
- New `_present_remaining_gaps_fallback_v2` deterministic narrator
- New `_render_clarification(clarification_payload) -> str` deterministic
  templated renderer for `credential_completion_confirmation` and
  `bootstrap_match_request` payloads
- Update `ask_one_clarifying_question` dispatch in `_fallback_reply_v2`
  to call `_render_clarification` when `clarification_payload` is set
  (BEFORE the existing `ask_slot` / generic-line paths)
- Policy regex: forbid "you qualify", "good fit", "good match",
  "qualified" on `explain_remaining_gaps` turns (reuses the
  `_NEAR_MISS_FORBIDDEN_PATTERNS` table — same words, new gate)
- Policy regex: forbid claims about how non-credential gaps are typically
  closed ("usually come on the job", "best learned through a course",
  "typically a course", etc.) on `explain_remaining_gaps` turns when
  no verified training data is present in TRAINING for the named gap
- Wire the new fallback into `_fallback_reply_v2`
- ~12 responder tests

### Slice R-6 — Live regression + transcript scenario + ship (~1.5 hours)

- Add `remaining_gaps_310s_automotive_weak` transcript scenario
- Live-test all 11 manual scenarios from "Live regression tests" above
  (including the retraction add/confirm pair #7-#8 and the
  redirect_scope preservation case #11)
- Confirm no regression in existing `present_matches` /
  `explain_gap` / `present_near_miss` flows
- One-line PR for the snapshot-field migration (deprecate the three
  legacy `last_presented_*` fields after fallbacks are updated)

Total estimate: ~10 hours across 6 slices, with review between each
(revised up from ~8 to reflect the new retraction path, defensive
cookie deserialization spec, and legacy-mapping wiring added in v6).

## Locked decisions (reviewed 2026-06-08)

All seven open questions were reviewed and locked. Future contributors:
don't reopen these without a new design step.

### Q1. "Think I got X" interpretation

**Locked: ambiguous → clarify.** Even when X resolves to a snapshot
credential, the verb "think" expresses epistemic uncertainty. Only
explicit claims ("I have X", "I got X", "I passed X") OR explicit
hypotheticals ("if I have X", "assume I have X", "after I get X")
subtract. Uncertain claims route to a confirmation question; the
answer determines the NEXT turn's subtraction.

Pattern table in §2 documents the full uncertainty-marker set
(`think|believe|guess|probably|maybe|pretty sure`).

### Q2. Multi-credential claim in one message

**Locked: subtract all explicitly matched credentials.** A single
message can claim multiple completions ("I have both 310S and Class
G"). Detection runs per-entity; all entities matching the
high-confidence patterns are emitted as separate `{canonical, mode}`
entries inside `current_turn_claims`. The handler appends them to
`last_assumed_completed_credentials` (ordered append + dedupe).
Forcing the user to chunk claims across multiple turns reads as the
system not listening.

The uncertainty rule (Q1) applies per-entity: "I have my 310S and
think I have my G2" subtracts 310S but not G2 (the G2 claim is
uncertain).

### Q3. Resume already shows the credential

**Locked: subtract for this turn AND log
`WARNING potential_extraction_defect`.** The user's answer is useful
regardless of what extraction surfaced; subtracting here gives them
the right next-step guidance. The log entry feeds the extraction-defect
backlog (the meeting_01 scenario where Class G was dropped is the
canonical example).

### Q4. No snapshot exists when remaining-gap-intent fires

**Locked: clarify / bootstrap job search.** "After 310S, what else?"
without a prior match is structurally meaningless. The handler
synthesizes:
```python
ArbiterDecision(
    final_move="ask_one_clarifying_question",
    reason_code="bootstrap_match_request",
    arbiter_action="handler_synthesized_clarification",
)
```
With responder narration along the lines of:
> *"I haven't shown you any local matches yet — want me to look for
> roles in your target field first, then we can walk through the
> gaps together?"*

This converts a no-context "what else?" into a productive search
bootstrap rather than a confused response.

### Q5. Skill-gap subtraction in v1

**Locked: acknowledge but do not subtract.** When the user mentions
skill experience in passing ("I have some Honda experience"), the
responder narration warmly acknowledges it but the structured
`remaining_core_skills` list stays as the authoritative view.
Responder prompt rule:

> *"If the user mentions experience or skills outside the assumed
> credentials, acknowledge naturally but keep remaining_core_skills
> as the authoritative gap list. Do not subtract skills in v1."*

This avoids the variability loop of trying to deterministically
reason about graded skill claims ("some experience" = partial?
"three years" = strong?).

### Q6. Multi-job pivot detection

**Locked: out of scope for v1.** `other_jobs_meta` exists in the
snapshot so a future feature can pivot, but v1 always uses
`lead_job`. Explicit "what about the truck tech role?" questions are
deferred.

### Q7. Telemetry

**Locked: shape expanded for retraction + provenance.** Extend the
truth log line with:
```
remaining_gaps_intent=<subtract|retract|confirm|bootstrap|none>
current_turn_claims_count=<int>
accumulated_credentials_count=<int>
any_hypothetical=<bool>
remaining_credentials_count=<int>
remaining_skills_count=<int>
pending_action=<add|remove|none>
```

The `remaining_gaps_intent` field covers all five detection outcomes:
- `subtract`: subtraction ran (explicit completion or hypothetical)
- `retract`: a previously-claimed credential was walked back
- `confirm`: uncertainty / retraction-language detected, clarification
  synthesized
- `bootstrap`: no snapshot, bootstrap clarification synthesized
- `none`: no remaining-gaps pattern matched; normal routing

`pending_action` records whether a confirm was synthesized for an
add or remove flow on this turn, OR what action a `kind="subtract"` /
`kind="retract"` consumed from `pending_credential_confirmation` (so
the log makes the cause of a removal traceable).

## Rollback

Snapshot is purely additive (`last_match_snapshot` and
`last_discussed_credential_canonical` are None for legacy sessions
and any session that hasn't yet hit a `present_matches` turn). The
new outcome is reached only through the handler-synthesis branch
added in Slice R-3. If that synthesis branch is gated off (single
`if False:` or removal of the call to `detect_remaining_gaps_intent`),
no turn ever produces `explain_remaining_gaps` and the chat behaves
identically to today. The existing planner/router/engine flow is
untouched by the changes in R-1..R-6 (other than adding the new
`OutcomeMove` enum value, which existing code paths never produce).

NO flag-gating is planned. Same reasoning as near-miss: this is a
correctness fix for live-observed product behavior, not an
experimental rollout.

## What I do NOT promise

- The completion pattern table catches every phrasing a user might
  use. We'll iterate; the telemetry log surfaces gaps.
- The ambiguous-vs-explicit distinction is 100% correct. Edge cases
  ("think I might have" — uncertain) will need explicit testing
  decisions.
- The responder's deterministic fallback prose reads as naturally as
  the LLM's happy path. The fallback is correctness-first.

## Forward note: adjacency precondition

When the adjacent-recommendations feature is built later in the
roadmap, the missing-credential exclusion (hard-filter out adjacent
jobs the candidate is credentially blocked from) MUST use structured
required-credential evidence — i.e. job-skill rows tagged as
credentials via the registry's `category` field or an explicit
credential flag — not just regex on credential-keyword strings in
skill names.

Reasoning: keyword-based credential detection produces false
positives ("software license" appearing in a software-engineer job
posting is not a regulatory licence requirement) and false negatives
(a skill named "WHMIS 2015" without the word "license" or
"certificate" is still a credential). The category field on
`extracted.job_skill` (when populated by the extractor) is the
authoritative signal.

This precondition is tracked separately for the adjacency feature
and does NOT block remaining-gaps build. Remaining-gaps uses the
engine's existing `credential_gap_skills` field on the score
explanation, which is already structurally populated by the cap-
applying engine pass — unchanged from today.

## Verdict

This is an earned architectural feature, not a prompt patch. It maps
directly onto the live-test failure pattern, follows the same
deterministic-routing template as v2.1 + near-miss + anaphor, and
reuses three existing helpers (`classify_gap`, `registry.lookup`,
anaphor pattern from `_resolve_target_role_anaphor`). The cost is
~7 hours of focused slice work across 6 slices, each independently
testable and shippable.

The win is enterprise-grade follow-up reasoning: post-match
conversations can iterate through gaps the user works toward,
preserving correctness (no profile mutation) and groundedness (no
LLM inference about completion semantics).

Locked and ready to build.
