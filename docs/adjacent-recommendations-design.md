# Adjacent-Recommendations — Design

Status: design v12 **amendment** · 2026-06-09 · signed off + AR-1a
implementation correction
(v1..v11 signed off → AR-1a implementation surfaced a cookie-budget
collision and a feature-gating contract gap → v12 amendment locks
the corrections.)

## v12 amendment — what changed (and why)

AR-1a implementation measured the existing R-1 worst-case signed
cookie at **3781 / 3800 bytes** (19 bytes headroom). v11's AR-1
cap estimates (`MAX_PRESENTED_JOB_IDS=24` × `MAX_JOB_ID_CHARS=64`
plus `last_adjacent_snapshot` at 3 items × full caps) added **2135
signed bytes** — the design's "~360 bytes" estimate was ~6× off.
Two amendments lock the corrections:

### Amendment 1 — Redis-mode activation gate

The entire adjacency feature is **gated to Redis-mode sessions**.
In cookie mode:

- `pending_adjacent_offer` is never set (no soft offer surfaces).
- `last_adjacent_snapshot` is never written.
- `last_match_snapshot["presented_job_ids"]` is never populated.
- `detect_adjacent_intent` / `retrieve_candidates` /
  `accept_candidates` / `rank_adjacent` /
  `resolve_adjacent_followup` / `describe_adjacent_role` short-
  circuit before any work.

The gate predicate (lands in AR-6 alongside the rest of activation):

```python
def _adjacency_enabled(staged: StagedProfile) -> bool:
    """Adjacency activation gate. Returns True only when the
    session store is Redis-backed; cookie-mode users see the
    pre-AR-1 experience (no soft offer, no recommend, no describe).
    The cookie ceiling (3800 bytes signed) does not have headroom
    for the AR-1 state alongside the R-1 worst case; rather than
    silently degrade in cookie mode, the feature is suppressed
    cleanly."""
    from skillbridge.session import get_store
    from skillbridge.session.redis_store import RedisSessionStore
    return isinstance(get_store(), RedisSessionStore)
```

All AR-6 wiring (handler turn-entry hook, soft-offer append, intent
detection dispatch, ordinal-followup dispatch) consults this gate
before doing any user-visible work.

### Amendment 2 — Cap tightening + lossless JSON minification

The shipped caps are tightened versus v11's speculative values to
align with the existing R-1 cookie discipline:

| Field | v11 | v12 (shipped) | Rationale |
|---|---|---|---|
| `MAX_PRESENTED_JOB_IDS` | 24 | **16** | Cookie budget + dedup horizon |
| `MAX_JOB_ID_CHARS` | 64 | **40** = `MAX_CANONICAL_CHARS` | Width parity with existing snapshot job_ids |
| `MAX_EVIDENCE_CHARS` | 120 | **100** | Sized for "3 of 5 required, 2 transferable"-class summaries |
| `MAX_MATCHED_SKILLS` | 5 | **4** | Render fit |
| `MAX_SKILL_CHARS` | 40 | **32** | Average canonical skill length is ~12-25 chars |
| `MAX_ADJACENT_ITEMS` | 3 | **3** (unchanged) | Surface cap = data cap |

`to_json(redact_for_cookie=True)` additionally performs **lossless
JSON minification** on the AR-1 keys: when they hold their
dataclass defaults (which under Amendment 1's gate is always true
in cookie mode), they are dropped from the serialized payload
entirely. `from_json` reconstructs the defaults when the keys are
missing. This is NOT a value-bearing redaction — if a non-default
value somehow appears in a cookie payload (bug), it survives
serialization and the cookie-size test trips, surfacing the bug
instead of hiding it.

### v1..v11 contracts preserved

Everything else from v11 stands: scope precedence
(`truth.scope_violations_detected` raw tags), TTL forward-shift
placement (before any dispatch, survives `fallback_to_legacy`),
bulk-loader stable ordering, save-and-clear at handler entry,
reoffer suppression, lead-result via `results[0]`, dedicated SSM
predicate, non-credential-only scorer, anchor-only transferable
gate, `describe_adjacent_role` render via live fetch, executable
TTL, defensive deserialization across the cookie boundary.

---

## v11 design body follows
(Status line preserved for traceability; corrections above
supersede any conflicting v11 text on caps or cookie persistence.)

Status: design v11 **draft** · 2026-06-09 · pending review
(v1..v10 → two precise corrections (raw scope tags + TTL-shift
placement) → v11. Every contract here has been grep-grounded
against the live codebase before being written.)

## Why this document exists

After a credential-capped match — a candidate with a Truck and Coach
Technician apprentice resume gets shown a 310T role demoted to stretch
because they're missing the licence — the system currently has nothing
useful to say beyond "here's the training path." The newcomer's real
question is often *"OK, what other local roles is some of my existing
experience transferable to?"*

That's not a near-miss question (the role exists, the user has gaps
in THAT role). It's an adjacency question (which OTHER local roles is
the user closer to — without adding a new credential first?). This
design adds a deterministic adjacency engine that runs on demand,
surfaces Sault-Ste.-Marie-proper jobs the user is not blocked from
by any required credential, ranks them through a dedicated
non-credential-only scorer (no target-title bias), and narrates them
through the existing responder contract.

## Locked corrections from v10 review

### Scope tag names — three raw tags, no `off_topic`

`_detect_scope_violations` at `truth_summary.py:599-630` populates
`TruthSummary.scope_violations_detected` with raw tag strings
ONLY, not reason codes. The three real tags (truth_summary.py:626-628):

- `"immigration"`
- `"national_wages"`
- `"non_ssm_city"`

The arbiter converts these to reason codes (`scope_violation_immigration`,
`scope_violation_wages`, `scope_violation_non_ssm`) via
`arbiter._scope_reason_code` per the comment at truth_summary.py:602-606.
**There is no `off_topic` tag** — `_detect_scope_violations` never
emits one. v10's hypothetical *"what other roles in baking?"* test
cannot pass through this signal path and is dropped from the v11
test plan.

v11 lock: the adjacency guard reads raw tags via
`bool(truth.scope_violations_detected)`. The transcript and unit
tests reference the three real tags. Off-topic coverage is
explicitly deferred until `_detect_scope_violations` gains an
`off_topic` tag (separate slice, out of scope here).

### TTL forward-shift — placed before standard dispatch and any early return

v10's pseudocode placed the snapshot TTL forward-shift inside the
`else` branch AFTER `standard_v2_path(...)` returned. That's
wrong because `_try_v2_path` can early-return on
`fallback_to_legacy` (handler.py:846-848: when Pass 1 returns an
`ArbiterDecision` with `arbiter_action == "fallback_to_legacy"`,
`_try_v2_path` returns `None` and the caller falls through to v1).
A scope-violation turn that also triggers a fallback path would
skip the shift entirely, and the snapshot would die after one
on-topic recovery turn — exactly the failure mode v10 was meant
to prevent.

v11 lock: the shift runs **immediately after `scope_violated` is
computed**, before any hook, planner call, or early return:

```python
truth = build_truth_summary(staged, message, ...)
scope_violated = bool(truth.scope_violations_detected)
if scope_violated:
    shift_adjacent_snapshot_ttl(staged)    # v11: BEFORE any dispatch
```

`shift_adjacent_snapshot_ttl` is a pure helper defined in AR-1 as
inert; AR-6 wires its call site:

```python
def shift_adjacent_snapshot_ttl(staged: StagedProfile) -> None:
    snap = staged.last_adjacent_snapshot
    if not isinstance(snap, dict):
        return
    created = snap.get("created_message_count")
    if not isinstance(created, int):
        return
    snap["created_message_count"] = created + 1
    staged.last_adjacent_snapshot = snap
```

Idempotent on a single turn (it's only called once per turn), and
the shift survives every downstream branch: fallback_to_legacy,
standard arbiter, planner exception, etc.

### v1→v10 corrections preserved

NOC unit-group = 5 digits, minor group = 4; jaccard removed;
retrieval-vs-acceptance separated; dedicated `_score_one_adjacent_job`
(no target-title / target-NOC boost); credentials excluded from
rank; recency contribution real and clamped
(`0.8*req + 0.2*pref + recency`); forbidden-vocabulary list;
trigger precedence (R-3 first); evidence floor; missing-NOC jobs
allowed via skill-evidence path; full `presented_job_ids` exclusion;
provider-free empty-result narration; required-only credential
filter; transferable gate uses anchor-only sets; soft offer also on
`present_no_match` with evidence; `canonicalize_skill(s.skill_name)`
everywhere; unweighted v1 coverage; `skill_type` via
`_required_or_preferred`; `posted_date` for recency;
`_skill_match_strength(job_skill, ids, names, canon)` signature;
`last_match_snapshot` as dict; render-by-fetch ordinal detail;
`created_message_count` captured BEFORE `staged.touch()`;
`OutcomeMove` parity + `PlannerMove` exclusion + planner "YOU MUST
NOT EMIT" + responder outcome prompt parity; `customer service`
out of generic set; `no_required_non_credential_skills` drop; SSM
proper only with dedicated alias set + verified region codes (NOT
`LOCAL_CITIES`); `pending_adjacent_offer` slot with
`_sanitize_pending`-shaped deserializer; bulk loader against
`extracted.job_skill` with `s.job_id = j.job_id` + `if
r.get("skill_name")` row filter + stable NULL-tolerant ordering;
`_band` reused from engine.py:483; caps read from
`lead_result["score_explanation"]["caps_applied"]`
(engine.py:588); save-and-clear in `handle_anonymous` after
session load; reoffer suppression via `if not pending_offer`;
"next meaningful turn" (non-blank text OR file upload) consumes
the flag; lead-result via `results[0] if results else None`;
activation deferred to AR-6; soft-offer append + flag-set BETWEEN
`compose_response_v2(...)` and `staged.touch()` at handler.py:996;
scope precedence via `if not scope_violated:` on both adjacency
hooks.

## Decision summary

- **Two new outcomes**: `recommend_adjacent_roles` and
  `describe_adjacent_role`. Both handler-synthesized.
- **Scope wins**. Adjacency hooks fire ONLY when
  `truth.scope_violations_detected` is empty (a list of the
  three raw tags above).
- **TTL shift placed before any dispatch** so it survives
  fallback_to_legacy and every other early return.
- **Planner and standard direct-match engine never called** on
  adjacency-intent turns.
- **Trigger user-initiated.** Save-and-clear at handler entry
  consumes `pending_adjacent_offer` regardless of downstream
  outcome.
- **Soft offer**: handler-side append inside `_try_v2_path`
  BEFORE the existing `staged.touch()` + `store.save(staged)`;
  reoffer-suppressed via `if not pending_offer`.
- **Activation deferred to AR-6.**
- **SSM proper only**: dedicated alias set + verified region codes.
- **Two-stage engine**: broad retrieval, strict AND acceptance.
- **Anchor-only transferable gate.**
- **Non-credential adjacency scorer**, no target bias.
- **Cap at 3 recommendations** per turn.
- **One-turn snapshot**, ordinal-resolvable, executable TTL,
  scope-violation forward-shift, render-by-live-fetch.
- **Bulk loader**: one SQL pass with stable NULL-tolerant
  ordering.

## Architecture

### Handler-turn lifecycle (final placement)

```python
# handler.py @ handle_anonymous (existing flow, with v9 hook):
def handle_anonymous(message, session_id, *, file_bytes=None,
                     filename=None, ...):
    uploaded_file = file_bytes is not None
    if not uploaded_file and (not message or not message.strip()):
        return _empty_input_response()       # pure blank short-circuit
    staged = load_staged_profile(session_id)

    # v7 hook (inert in AR-1; AR-6 introduces the setter).
    pending_offer = bool(staged.pending_adjacent_offer)
    if pending_offer:
        staged.pending_adjacent_offer = False

    # ... existing: resume-upload review, scope-canned-gates, etc.
    return _try_v2_path(
        staged=staged, message=message, pending_offer=pending_offer,
        uploaded_file=uploaded_file, ...,
    )
```

```python
# handler.py @ _try_v2_path (existing flow, with v10+v11 hooks):
def _try_v2_path(*, staged, message, pending_offer, ...):
    truth = build_truth_summary(staged, message, ...)
    scope_violated = bool(truth.scope_violations_detected)

    # v11 LOCK: TTL forward-shift IMMEDIATELY after scope_violated
    # is computed, BEFORE any hook / planner / early return.
    # Survives fallback_to_legacy at handler.py:848 and every other
    # downstream branch.
    if scope_violated:
        shift_adjacent_snapshot_ttl(staged)

    # Remaining-gaps hook (R-3) -- existing.

    # v10 SCOPE PRECEDENCE: adjacency hooks fire ONLY when no scope
    # violation. On violation, the standard arbiter path runs and
    # redirect_scope is emitted via planner.py:326-327.
    if not scope_violated:
        followup_match = resolve_adjacent_followup(
            message, staged.last_adjacent_snapshot, staged.message_count,
        )
        if followup_match is not None:
            final = synthesize_describe_adjacent_role(followup_match)
            results = []
        else:
            intent = detect_adjacent_intent(
                message=message, staged=staged,
                has_usable_skill_evidence=has_usable_skill_evidence(staged),
                pending_offer=pending_offer,
            )
            if isinstance(intent, AdjacentIntent):
                created_msg = staged.message_count            # BEFORE touch
                user_ids, user_names, user_canon = build_user_skill_sets(staged.skills)
                all_jobs  = _load_active_jobs_with_skills()
                retrieved = retrieve_candidates(staged, snapshot, all_jobs,
                                                user_ids, user_names, user_canon)
                accepted  = accept_candidates(retrieved, staged,
                                              user_ids, user_names, user_canon)
                ranked    = rank_adjacent(accepted, user_ids, user_names, user_canon)
                top3      = drop_excluded(ranked, snapshot.get("presented_job_ids", ()))[:3]
                final     = synthesize_recommend_adjacent_decision(top3)
                results   = []
                persist_last_adjacent_snapshot(staged, top3, created_msg)
            elif isinstance(intent, NeedsEvidenceIntent):
                final = synthesize_clarification(...)
                results = []
            else:
                final, results = standard_v2_path(...)        # may return None on fallback_to_legacy
                if final is None:
                    return None
    else:
        final, results = standard_v2_path(...)
        if final is None:
            return None

    reply = compose_response_v2(ResponderV2Input(decision=final, ...))

    # v9 soft-offer append + flag set, BETWEEN compose_response_v2 and
    # the existing touch + save at handler.py:996.
    if not pending_offer:
        if final.final_move == "present_matches":
            lead_result = results[0] if (results and isinstance(results[0], dict)) else None
            if lead_result and should_emit_soft_offer_on_matches(lead_result, staged):
                reply = reply.rstrip() + "\n\n" + _SOFT_OFFER_LINE
                staged.pending_adjacent_offer = True
        elif final.final_move == "present_no_match":
            if should_emit_soft_offer_on_no_match(staged):
                reply = reply.rstrip() + "\n\n" + _SOFT_OFFER_LINE
                staged.pending_adjacent_offer = True

    # ... existing ask_slot bookkeeping ...

    # Existing handler.py:996-1010.
    staged.touch()
    new_session_id = store.save(staged)
    return _build_v2_response(
        staged=staged, new_session_id=new_session_id, reply=reply,
        ...
    )
```

Two early-return paths survive the shift correctly:

- **`standard_v2_path` returns None on fallback_to_legacy**
  (handler.py:846-848). The TTL shift already ran before the
  dispatch, so the snapshot stays alive when v1 takes over.
- **An exception in `standard_v2_path`** doesn't reach the
  bookkeeping below, but the shift already mutated the snapshot
  and the cookie write happens via v1's own touch/save. The
  shift is idempotent on a single turn — the v1 path doesn't
  shift again.

### Snapshot TTL helper (AR-1 inert, AR-6 wired)

```python
def shift_adjacent_snapshot_ttl(staged: StagedProfile) -> None:
    """Shift last_adjacent_snapshot.created_message_count forward
    by 1. Called on scope-violation turns so the ordinal-followup
    window survives a single redirect_scope digression.

    Pure helper; no-op when no snapshot is live or the field is
    malformed. Lands as a private helper in AR-1; AR-6 introduces
    the call site in `_try_v2_path`.
    """
    snap = staged.last_adjacent_snapshot
    if not isinstance(snap, dict):
        return
    created = snap.get("created_message_count")
    if not isinstance(created, int):
        return
    snap["created_message_count"] = created + 1
    staged.last_adjacent_snapshot = snap
```

### Bulk job + skills loader (v10 stable ordering, preserved)

```python
def _load_active_jobs_with_skills() -> list[dict]:
    rows = _exec("""
        SELECT j.*,
               s.skill_id, s.skill_name, s.confidence,
               s.importance_rank, s.skill_type
          FROM core.v_current_job j
          LEFT JOIN extracted.job_skill s ON s.job_id = j.job_id
         ORDER BY j.posted_date DESC NULLS LAST,
                  j.job_id,
                  s.importance_rank NULLS LAST,
                  s.confidence DESC NULLS LAST,
                  s.skill_name
    """)
    # ... grouping + skill_name row filter (unchanged from v10) ...
```

### Soft-offer eligibility, soft-offer line, slot, SSM predicate, detector, anchor classifier, retrieval, acceptance, ranking, snapshot, ordinal resolution

All bodies unchanged from v9/v10.

### `OutcomeMove` and prompt parity

Unchanged. AR-1 inert declarations; AR-6 dispatch activation.

## Empty-result narration, forbidden vocabulary, telemetry

Unchanged.

## Tests

| Layer | Coverage |
|---|---|
| **Scope precedence (raw tags, v11)** | "what other roles help with PR?" → `_detect_scope_violations` returns `["immigration"]` → adjacency hooks SKIPPED, planner emits `redirect_scope` with reason `scope_violation_immigration`; "what other roles pay more nationally?" → `["national_wages"]` → adjacency SKIPPED, `scope_violation_wages`; "what other roles in Toronto?" → `["non_ssm_city"]` → adjacency SKIPPED, `scope_violation_non_ssm`. In each case: pending_offer was True at entry → CLEARED; last_adjacent_snapshot PRESERVED via TTL forward-shift; presented_job_ids unchanged. **off_topic coverage explicitly deferred** until `_detect_scope_violations` emits an off_topic tag (out of scope here) |
| **TTL forward-shift placement (v11)** | shift runs BEFORE `standard_v2_path(...)`; on `fallback_to_legacy` → standard_v2_path returns None at handler.py:848 → `_try_v2_path` returns None to v1 → snapshot has already been shifted, survives v1's turn; on any exception thrown by standard_v2_path → shift already happened pre-call, mutation persists |
| **TTL shift on adjacency-intent path** | scope_violated=False → shift_adjacent_snapshot_ttl NOT called (only fires under scope_violated branch); the followup-hook still resolves via standard TTL |
| **TTL shift idempotence** | two scope-violation turns in a row shift `created_message_count` by 1 each; the third on-topic turn still resolves the snapshot |
| **Bulk loader stable ordering** | query contains `ORDER BY j.posted_date DESC NULLS LAST, j.job_id, s.importance_rank NULLS LAST, s.confidence DESC NULLS LAST, s.skill_name`; two runs produce byte-identical `evidence_summary` AND `matched_skills`; fixture with two NULL-skill_id rows verifies `skill_name` tie-breaker controls order |
| Bulk loader still matches `_fetch_job_skills` semantics | `(importance_rank NULLS LAST, confidence DESC)` prefix matches engine.py:202 |
| AR-1 activation safety | `pending_adjacent_offer` never set anywhere in main after AR-1; grep audit: only AR-6 commits the setter; handler save-and-clear hook is a no-op; `shift_adjacent_snapshot_ttl` defined but never called in production (until AR-6) |
| AR-1..AR-5 dead-code audit | no production call site dispatches to `detect_adjacent_intent` / `_load_active_jobs_with_skills` / `resolve_adjacent_followup` / `shift_adjacent_snapshot_ttl` before AR-6 |
| Persistence placement | soft-offer-append + flag-set sits BETWEEN `compose_response_v2(...)` and `staged.touch()` at handler.py:996; `handle_anonymous` does NOT call save() after `_try_v2_path` returns |
| Lead-result extraction | `results[0] if (results and isinstance(results[0], dict)) else None`; `ArbiterDecision.results` is never referenced |
| Reoffer suppression — credential cap | pending_offer=True, user "no thanks", standard path returns present_matches with credential cap → NO new soft-offer; flag ends False |
| Reoffer suppression — no match | pending_offer=True, user "no thanks", standard path returns present_no_match → NO new offer; flag ends False |
| Reoffer suppression — scope violation | pending_offer=True, message triggers raw tag `immigration` → standard path returns redirect_scope → NO new soft-offer (final_move guard); flag ends False; snapshot shifted forward |
| Ambiguous-reply path | pending_offer=True, "I guess?", detector → None, standard path returns present_no_match → NO new offer; flag ends False |
| Affirmative path | pending_offer=True, "yes", detector → AdjacentIntent → decision = recommend_adjacent_roles → NO soft-offer wiring; flag ends False; snapshot persisted with current `created_message_count` |
| Save-and-clear placement | runs after session load, BEFORE: resume-upload review, scope-canned gates, `_try_v2_path` |
| Blank input — no upload | returns BEFORE session load; flag NOT consumed |
| Blank input + upload | reaches session load; save-and-clear runs; flag IS consumed |
| Multi-turn flag persistence | turn N sets flag; turn N+1 pure blank → flag persists; turn N+2 non-blank "yes" → AdjacentIntent |
| Handler soft-offer wiring | on present_matches + credential-only cap + evidence AND pending_offer=False → reply ends with `_SOFT_OFFER_LINE`, flag TRUE; same on present_no_match + evidence; suppressed when any other cap; suppressed when no evidence; suppressed when pending_offer=True at entry; NEVER fires on redirect_scope |
| `is_credential_only_band_cap` | reads from `score_explanation["caps_applied"]` AND `["score_components"]["score_pre_caps"]`; pre-cap good/strong + sole credential cap → True; pre-cap stretch → False; caps contains another flag → False; missing score_pre_caps → False; missing score_explanation → False |
| `_band` reuse | adjacency calls engine helper |
| `pending_adjacent_offer` lifecycle | defensive-deserialize rejects non-bool; cookie round-trip; SETTER only in AR-6 |
| `is_ssm_region_job` | region_code "3557011" → True; "SSM" → True; "algoma" → False; "toronto" → False; missing code + location "Sault Ste. Marie" → True; missing code + location "Wawa, ON" → False; missing both → False; LOCAL_CITIES not consulted |
| `detect_adjacent_intent` | pure; explicit phrasings → AdjacentIntent; same-role-gap → None; pending_offer + "yes" → AdjacentIntent; pending_offer + "I guess" → None; pending_offer=False + "yes" alone → None; NeedsEvidenceIntent path |
| `has_usable_skill_evidence` | 3 resume @ 0.7 → True; resume-less + 3 chat @ 0.7 → True; 3 credentials only → False |
| `is_non_generic_transferable` | "communication" → False; "customer service" → True; "welding" resume @ 0.7 → True; resume @ 0.4 → False |
| `retrieve_candidates` | SSM-proper filter; NOC minor-group; skill-evidence; missing job-NOC allowed |
| `accept_candidates` | required-credential gate; preferred credential does NOT drop; `no_required_non_credential_skills`; unweighted coverage; ANCHOR-only transferable |
| `_score_one_adjacent_job` | credentials excluded; no target-title boost; no target-NOC boost; recency real; clamped to [0, 1] |
| `drop_excluded` | excludes every job_id in `presented_job_ids` |
| Snapshot — `presented_job_ids` | cap 24; deduped; wrong-type sanitized; cookie round-trip under 3800 bytes |
| Snapshot — `last_adjacent_snapshot` | TTL +1 only; cleared by next match decision; cleared by target_role_text change; `created_message_count` BEFORE touch(); scope-violation forward-shift survives one digression |
| `resolve_adjacent_followup` | ordinal "the second one"; "#3"; title-suffix; ambiguous → None; out-of-range → None; stale → None |
| `describe_adjacent_role` render | live job fetched by id; expired → deterministic fallback; evidence_summary + matched_skills from snapshot |
| `OutcomeMove` / planner parity | both new moves in arbiter Literal; `PlannerMove` excludes them; planner "YOU MUST NOT EMIT" lists both; responder outcome prompt lists both |
| Transcript — accept (AR-6) | credential-capped 310T → soft-offer + flag set → blank turn (flag persists) → "yes" → adjacency surfaces; next turn "the second one" → describe_adjacent_role; turn after → snapshot dead |
| Transcript — decline (AR-6) | credential-capped 310T → soft-offer + flag set → "no thanks" → flag cleared, same outcome → NO new offer; later turn → offer re-surfaces |
| Transcript — scope digression (AR-6, v11 raw tags) | credential-capped 310T → soft-offer + flag set → "what other roles help with PR?" → raw tag `immigration` → shift_adjacent_snapshot_ttl called BEFORE dispatch → redirect_scope, flag cleared → next on-topic "the second one" → describe_adjacent_role (snapshot still live thanks to shift) |
| Transcript — fallback_to_legacy under scope (AR-6, v11) | scope violation + pass-1 returns fallback_to_legacy → `_try_v2_path` returns None at handler.py:848 → v1 takes over → snapshot's `created_message_count` was already shifted before the dispatch → next on-topic turn still resolves the snapshot |

## Locked open-question answers

- **QA**: Fixture-tune thresholds inside AR-3.
- **QB**: Keep same-employer roles.
- **QC**: Soft offer also on genuine `present_no_match` with
  evidence.
- **QD**: Ordinal resolution in scope (AR-5 component, activated
  AR-6).
- **QE**: Aggregate counters only.
- **QF**: Locked affirmative set; ambiguous → clarification.

## Build slices (v11)

- **AR-1** (state, schema, helpers — all inert):
  - `is_ssm_region_job` predicate;
  - `OutcomeMove` extension + `PlannerMove` exclusion + planner
    "YOU MUST NOT EMIT" + responder outcome prompt declarations;
  - `pending_adjacent_offer` slot + `_sanitize_pending_adjacent_offer`;
  - handler save-and-clear hook in `handle_anonymous` (no-op
    until AR-6 sets the flag);
  - `has_usable_skill_evidence` + `is_credential_only_band_cap` +
    `should_emit_soft_offer_on_matches` /
    `should_emit_soft_offer_on_no_match`;
  - `last_adjacent_snapshot` dict contract + TTL helpers including
    `shift_adjacent_snapshot_ttl` (pure helper — no production
    caller until AR-6);
  - `presented_job_ids` dict-key contract on `last_match_snapshot`;
  - cookie round-trip test at full caps under 3800 bytes;
  - blank-input lock tests;
  - activation-safety grep audit.
- **AR-2** (dead, tested detection):
  - `detect_adjacent_intent` (pure; threaded `pending_offer`);
  - `is_non_generic_transferable`;
  - `_AFFIRMATIVE_REPLIES`;
  - NeedsEvidenceIntent path;
  - ~35 unit tests.
- **AR-3** (dead, tested retrieval + acceptance):
  - `_load_active_jobs_with_skills` with stable NULL-tolerant
    ordering + reproducibility test;
  - `build_user_skill_sets` + `build_anchor_skill_sets`;
  - `retrieve_candidates`;
  - `accept_candidates` (anchor-only gate,
    `no_required_non_credential_skills` drop);
  - fixture-tuned threshold pass.
- **AR-4** (dead, tested ranking):
  - `_score_one_adjacent_job`;
  - `drop_excluded`;
  - integration tests.
- **AR-5** (dead, tested ordinal follow-up):
  - `resolve_adjacent_followup`;
  - `describe_adjacent_role` synthesizer + live-fetch render +
    expired-job fallback;
  - TTL boundary tests.
- **AR-6** (activation slice):
  - `_try_v2_path` scope-violated computation + IMMEDIATE
    `shift_adjacent_snapshot_ttl(staged)` call BEFORE any hook
    or dispatch;
  - adjacent-followup hook + adjacent-intent hook gated on
    `if not scope_violated:`;
  - soft-offer append + flag set BETWEEN `compose_response_v2`
    and `staged.touch()`; reoffer-suppressed;
  - responder narration for both new moves;
  - empty-result branch wording;
  - forbidden-vocabulary gate additions;
  - payload caps enforced;
  - the FOUR transcript paths (accept / decline / ambiguous /
    no-offer) PLUS the scope-violation digression transcript
    PLUS the fallback_to_legacy-under-scope transcript turn
    GREEN here.
- **AR-7** (acceptance + telemetry hardening):
  - End-to-end multi-turn transcripts;
  - aggregate-telemetry assertions;
  - cookie-size headroom assertion under realistic worst case.

Total estimate after reviewer iterations: ~35 hours (up from v10's
34h — AR-6 grew by the fallback_to_legacy transcript; AR-1 grew by
the shift_adjacent_snapshot_ttl helper and its dead-code audit
line).

## What I do NOT promise

- The adjacency definition is a v1 heuristic.
- The credential filter is only as good as
  `is_credential_skill_name`.
- A `chat_confirmed` evidence taxonomy is NOT delivered here.
- A weighted coverage formula is NOT promised for v1.
- An Algoma service region expansion is OUT of scope.
- **Off-topic scope handling is OUT of scope here.**
  `_detect_scope_violations` currently emits no `off_topic` tag;
  adding one is a separate slice. Until then, off-topic
  questions fall through to the planner without an explicit
  scope redirect.
- Adjacency does not certify employability.
- Acceptance thresholds fixture-tuned inside AR-3.
- This design runs ALONGSIDE the existing matcher.
- AR-1..AR-5 ship NO user-visible behavior change.

## Verdict

v11 corrects the two v10 precision issues:
- **Scope tags are raw**:
  `TruthSummary.scope_violations_detected` carries
  `"immigration"` / `"national_wages"` / `"non_ssm_city"`. The
  arbiter maps them to `scope_violation_*` reason codes. Tests
  use the raw tags. Off-topic is explicitly deferred because
  `_detect_scope_violations` doesn't emit an `off_topic` tag.
- **TTL forward-shift placement is BEFORE dispatch**:
  `shift_adjacent_snapshot_ttl(staged)` runs immediately after
  `scope_violated` is computed, before any hook, planner call,
  or early return. Survives `fallback_to_legacy` at
  handler.py:848 and every other downstream branch.

Open for the v11 reviewer pass before AR-1.
