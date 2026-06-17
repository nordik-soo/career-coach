"""Adjacency primitives — evidence floor + soft-offer eligibility.

Two related groups of functions:

  - `has_usable_skill_evidence(staged)` is the evidence-quality floor
    the adjacency engine (AR-3) and detector (AR-2) consult before
    proposing recommendations. Three skills under the canonical
    classifier with `source in {"resume", "chat"}` and confidence ≥
    0.6.

  - `is_credential_only_band_cap(lead_result)` and the soft-offer
    predicates wrap the rule for when the chat handler is allowed to
    append the "want me to look at related roles?" prompt to a
    standard-path response. Inputs come from the per-result score
    explanation that the engine already produces -- NOT from the
    arbiter's union of caps across multiple results (see v6+ design
    review).

This module is DEAD CODE until AR-6 wires the predicates into
`handle_anonymous` / `_try_v2_path`. AR-1c's activation-safety grep
audit confirms no production caller dispatches into these helpers
yet. The Redis-mode activation gate (`_adjacency_enabled` in AR-6)
short-circuits ALL of these calls in cookie-mode sessions.
"""
from __future__ import annotations

from typing import Any

from skillbridge.match.aliases import canonicalize_skill
from skillbridge.match.alignment import (
    _ACCEPTED_EVIDENCE_SOURCES,
    _MIN_EVIDENCE_CONFIDENCE,
    _is_valid_normalized_score,
    build_user_skill_rows,
    derive_user_skill_sets,
)
from skillbridge.match.engine import _band, is_credential_skill_name
from skillbridge.session.staging import StagedProfile, StagedSkill


# Minimum count of non-credential evidence-eligible skills required
# before adjacency is willing to recommend. Below this, the user has
# nothing meaningful to rank against.
ADJACENT_MIN_USER_SKILLS: int = 3


# AR-2 anchor classifier. Evidence-backed generic-skill set;
# canonical forms produced by `canonicalize_skill`. Common surface
# variants are enumerated so a user's resume entry like "communication
# skills" or "strong work ethic" doesn't slip through as a transferable
# anchor.
#
# v4 lock: "customer service" is DELIBERATELY NOT generic -- it's
# concrete evidence for SCCC retail / hospitality / frontline roles.
#
# The acceptance gate (AR-3) provides defense in depth: even a skill
# that escapes this filter still has to clear a strength threshold
# against a required job skill, so a rare unhandled variant doesn't
# trip the gate on its own.
_GENERIC_SKILL_CANONICALS: frozenset[str] = frozenset({
    "communication", "communication skills",
    "teamwork", "team work",
    "leadership", "leadership skills",
    "time management",
    "problem solving", "problem solving skills",
    "attention to detail",
    "organization", "organizational", "organizational skills",
    "interpersonal", "interpersonal skills",
    "work ethic", "strong work ethic",
    "adaptability",
    "collaboration",
    "multitasking", "multi tasking",
})


_MIN_TRANSFERABLE_CONFIDENCE: float = 0.6


def is_non_generic_transferable(s: StagedSkill) -> bool:
    """Anchor classifier for the AR-3 transferable-anchor gate.

    True iff the staged skill is:
      - a real StagedSkill instance with `skill_name` of type `str`;
      - non-credential (per `is_credential_skill_name`);
      - non-generic (canonical NOT in `_GENERIC_SKILL_CANONICALS`);
      - from an accepted evidence source (resume or chat);
      - confidence is a finite, non-boolean numeric in [0, 1] AT OR
        ABOVE the floor (mirrors `has_usable_skill_evidence`).

    The acceptance gate scores REQUIRED non-credential job skills
    against the ANCHOR-only user sets, so an unrelated concrete skill
    (e.g. a matched credential or an off-domain skill that didn't pass
    this classifier) cannot inflate the gate.

    Defensive: a forged-cookie StagedSkill could carry skill_name as
    int/bool/dict; `canonicalize_skill` calls `.lower()` and would
    crash. Skip those entries here so they never reach the engine
    helpers.
    """
    if not isinstance(s, StagedSkill):
        return False
    if s.source not in _ACCEPTED_EVIDENCE_SOURCES:
        return False
    if not _is_valid_normalized_score(s.confidence):
        return False
    if s.confidence < _MIN_TRANSFERABLE_CONFIDENCE:
        return False
    if not isinstance(s.skill_name, str):
        return False
    canonical = canonicalize_skill(s.skill_name)
    if not canonical:
        return False
    if is_credential_skill_name(canonical):
        return False
    if canonical in _GENERIC_SKILL_CANONICALS:
        return False
    return True


def has_usable_skill_evidence(staged: StagedProfile) -> bool:
    """Evidence-quality floor for adjacency.

    True iff `staged.skills` contains at least `ADJACENT_MIN_USER_SKILLS`
    DISTINCT canonical skills that:
      - are NOT credential names (per the shared classifier
        `is_credential_skill_name` -- so a user with three certificates
        and no work skills doesn't trip the floor);
      - carry `source in _ACCEPTED_EVIDENCE_SOURCES` (resume-derived
        or chat-confirmed, not a fallback/synthetic source);
      - carry a finite, non-boolean confidence in [0.0, 1.0] that is at
        or above `_MIN_EVIDENCE_CONFIDENCE`.

    Counts unique canonical names so a user with three duplicate
    "forklift" entries (resume + two chat re-mentions) cannot trip the
    three-skill floor with what is really one skill.

    Resume presence is NOT required -- a resume-less user with three
    high-confidence chat-confirmed skills clears the floor.
    """
    seen_canonicals: set[str] = set()
    for s in staged.skills:
        if not isinstance(s, StagedSkill):
            # Defensive: from_json sanitization should already prevent
            # this, but the predicate is called on live staged objects
            # whose `skills` list is mutable.
            continue
        if s.source not in _ACCEPTED_EVIDENCE_SOURCES:
            continue
        if not _is_valid_normalized_score(s.confidence):
            # Rejects NaN / inf / booleans / out-of-range. A NaN
            # confidence would have passed `< 0.6` silently (every
            # comparison with NaN is False), so without this guard a
            # forged record could clear the floor.
            continue
        if s.confidence < _MIN_EVIDENCE_CONFIDENCE:
            continue
        # Defensive: a forged StagedSkill with skill_name=int/bool/dict
        # would crash inside `canonicalize_skill` (`.lower()`). Skip
        # malformed entries before they reach the engine helpers.
        if not isinstance(s.skill_name, str):
            continue
        canonical = canonicalize_skill(s.skill_name)
        if not canonical:
            continue
        if is_credential_skill_name(canonical):
            continue
        seen_canonicals.add(canonical)
        if len(seen_canonicals) >= ADJACENT_MIN_USER_SKILLS:
            return True
    return False


# ---------------------------------------------------------------- soft offer
# The exact cap flag the engine emits when the lead match's band was
# capped down to "stretch" purely because the user is missing a
# required credential. See engine.py:603 where this flag is appended
# to score_explanation.caps_applied. Reuse the string verbatim so a
# future engine-side rename surfaces here at import time, not at
# runtime.
_CRED_CAP_FLAG: str = "band_capped_by_credential"


def _result_caps(lead_result: dict[str, Any]) -> tuple[str, ...]:
    """Return the per-result caps_applied list, defensively.

    Engine.py:588 stores caps under `score_explanation`, NOT at the
    result top level (a v6 review correction). v12 reads them from
    the correct path. Wrong type / missing key ->empty tuple.
    """
    se = lead_result.get("score_explanation") or {}
    if not isinstance(se, dict):
        return ()
    caps = se.get("caps_applied") or ()
    if not isinstance(caps, (list, tuple)):
        return ()
    return tuple(c for c in caps if isinstance(c, str))


def is_credential_only_band_cap(lead_result: dict[str, Any]) -> bool:
    """True iff the lead match's pre-cap band was good or strong AND
    the ONLY cap applied to it was the credential cap.

    Inputs come from the per-result `score_explanation` that the
    engine already produces (engine.py:1042+, engine.py:1335+):
      - `["score_components"]["score_pre_caps"]` -- pre-cap score
      - `["caps_applied"]` -- per-result cap-flag list

    `score_pre_caps` is converted to a band via `engine._band` so the
    thresholds stay in sync with `MATCH.band_strong / band_good /
    band_stretch`.

    Returns False on any malformed input (missing keys, wrong types,
    or unexpected pre-cap band).
    """
    if not isinstance(lead_result, dict):
        return False
    se = lead_result.get("score_explanation") or {}
    if not isinstance(se, dict):
        return False
    components = se.get("score_components") or {}
    if not isinstance(components, dict):
        return False
    score_pre_caps = components.get("score_pre_caps")
    if not _is_valid_normalized_score(score_pre_caps):
        # Rejects NaN / inf / booleans / out-of-range. True would
        # otherwise pass an isinstance int check and float() to 1.0,
        # triggering a spurious credential-only offer.
        return False

    pre_cap_band = _band(float(score_pre_caps))
    if pre_cap_band not in {"good", "strong"}:
        return False

    return _result_caps(lead_result) == (_CRED_CAP_FLAG,)


def should_emit_soft_offer_on_matches(
    lead_result: dict[str, Any],
    staged: StagedProfile,
) -> bool:
    """The composite eligibility for appending the soft offer to a
    `present_matches` response: credential-only cap on the lead result
    AND the user has usable skill evidence."""
    return (
        is_credential_only_band_cap(lead_result)
        and has_usable_skill_evidence(staged)
    )


def should_emit_soft_offer_on_no_match(staged: StagedProfile) -> bool:
    """The eligibility for appending the soft offer to a genuine
    `present_no_match` response: the user has usable skill evidence
    (so we have something to anchor adjacency against)."""
    return has_usable_skill_evidence(staged)


# The exact wording the handler appends to the responder reply when a
# soft-offer eligibility check passes. Single source of truth -- the
# design v11 §"Soft offer on credential-capped matches" locks the
# user-facing string here so transcript tests and a future
# localization pass have one place to read it.
_SOFT_OFFER_LINE = (
    "If you'd like, I can also look for related roles where some of "
    "your existing skills transfer -- just say *what other roles?*"
)


# ---------------------------------------------------------------- activation
def _adjacency_enabled() -> bool:
    """Adjacency activation gate (v12 amendment).

    Returns True only when the active session store is Redis-backed.
    Cookie-mode users see the pre-AR-1 experience -- no soft offer, no
    intent dispatch, no engine call, no persistence. The 3800-byte
    signed-cookie ceiling does NOT have headroom for the full AR-1
    state alongside the R-1 worst case (measured: R-1 baseline = 3781
    signed bytes; AR-1 fields at cap add ~2135 bytes). Rather than
    silently degrade in cookie mode, the entire feature is suppressed.

    AR-6 wires this gate into every adjacency entry point:
      - handler save-and-clear hook
      - soft-offer append step
      - detect_adjacent_intent dispatch
      - resolve_adjacent_followup dispatch
      - recommend_adjacent_roles synthesis
      - describe_adjacent_role synthesis

    AR-1c ships the predicate as DEAD CODE -- no production caller
    dispatches into it yet. The activation-safety grep audit in
    tests/test_ar1c_parity_and_activation.py confirms that.

    Implementation: imports are deferred so a circular import via
    session_store factory at package-load time isn't triggered.

    Two gates AND'd together:
      - `_is_redis_mode()`: the v12 amendment lock -- adjacency state
        cannot fit in the cookie-mode 3800-byte ceiling.
      - `ADJACENCY_ACTIVATION_ENABLED` feature flag: default OFF in
        production. Lifts when AR-6c lands so the responder payload
        threading is in place before Redis users see any adjacency
        outcome. Acceptance tests flip the flag to True via env var
        or monkeypatch.
    """
    from skillbridge.session import get_store
    from skillbridge.session.redis_store import RedisSessionStore

    if not isinstance(get_store(), RedisSessionStore):
        return False
    from config import ADJACENCY_ACTIVATION_ENABLED
    return ADJACENCY_ACTIVATION_ENABLED


# ---------------------------------------------------------------- synthesis
# Pure ArbiterDecision factories for the two handler-synthesized
# outcomes. Building these here -- rather than in handler.py alongside
# `_synthesize_remaining_gaps_decision` -- keeps the adjacency module
# self-contained: AR-6 will import these, wire the dispatch, and that's
# the ONLY place these helpers get called from in production.
#
# AR-1c ships them as DEAD CODE so:
#   (a) the OutcomeMove reachability invariant
#       (test_every_outcome_move_is_reachable_through_some_path) can
#       exercise the helpers and confirm each new enum value has at
#       least one producer; and
#   (b) AR-6's slice can wire them into _try_v2_path without
#       introducing new ArbiterDecision-building code at activation
#       time.
#
# No staged mutation, no I/O. The responder payloads
# (adjacent_recommendations_payload, adjacent_role_description_payload)
# are attached to ResponderV2Input separately in AR-6.
# =========================================================================
# AR-4 ranking + exclusion
# =========================================================================
def _score_one_adjacent_job(
    job: dict,
    user_ids: set[str],
    user_names: set[str],
    user_canon: set[str],
) -> float:
    """Dedicated adjacency scorer (v11 / v12). Distinct from
    `engine._score_one_job` because it MUST NOT carry the target-title
    or target-NOC boost -- those exist to bias the matcher toward the
    user's stated target role, which is precisely the OPPOSITE of what
    adjacency wants. Adjacency ranks roles based on transferable
    evidence + freshness, NOT how similar they are to the user's
    current target.

    Score components:
        required_mean   -- mean of `_skill_match_strength` over
                           required NON-credential skills
        preferred_mean  -- same, over preferred non-credential skills
        recency_boost   -- `engine._recency_boost(posted_date)`,
                           returns [0.0, 0.05]

    Formula:
        score = 0.8 * required_mean + 0.2 * preferred_mean + recency_boost
        score = max(0.0, min(1.0, score))   # explicit clamp

    Credentials are EXCLUDED from the scoring loops -- the credential
    gate in `accept_candidates` already ensures a candidate's required
    credentials are satisfied; counting them again as "matched skills"
    here would inflate the score with eligibility certifications
    rather than transferable evidence.
    """
    from skillbridge.match.engine import (
        _recency_boost,
        _required_or_preferred,
        _skill_match_strength,
    )

    if not isinstance(job, dict):
        return 0.0

    raw_skills = job.get("skills") or []
    if not isinstance(raw_skills, list):
        raw_skills = []

    required_strengths: list[float] = []
    preferred_strengths: list[float] = []
    for s in raw_skills:
        if not isinstance(s, dict):
            continue
        name = s.get("skill_name")
        if not isinstance(name, str) or not name.strip():
            continue
        # Skip credentials entirely -- see docstring.
        if is_credential_skill_name(name):
            continue
        # `_required_or_preferred` does `(skill_type or "").strip().lower()`.
        # A non-str skill_type (int/bool/dict from a forged blob) would
        # crash `.strip()`. Coerce to a string-shaped dict first so the
        # helper sees a benign value -- legacy NULL semantics ("required"
        # on missing/unknown) are preserved.
        if not isinstance(s.get("skill_type"), (str, type(None))):
            safe_skill = dict(s)
            safe_skill["skill_type"] = ""
            bucket = _required_or_preferred(safe_skill)
        else:
            bucket = _required_or_preferred(s)
        strength, _stage = _skill_match_strength(
            s, user_ids, user_names, user_canon,
        )
        if bucket == "required":
            required_strengths.append(strength)
        else:
            preferred_strengths.append(strength)

    req_mean = (
        sum(required_strengths) / len(required_strengths)
        if required_strengths else 0.0
    )
    pref_mean = (
        sum(preferred_strengths) / len(preferred_strengths)
        if preferred_strengths else 0.0
    )

    # `_recency_boost` expects a `date` (NOT a datetime; `date -
    # datetime` raises TypeError because datetime subclasses date but
    # the arithmetic isn't symmetric). Persisted state can leak
    # datetime / str / int; only pass real `date` instances.
    from datetime import date as _date, datetime as _datetime

    posted = job.get("posted_date")
    if isinstance(posted, _date) and not isinstance(posted, _datetime):
        rec_boost = _recency_boost(posted)
    else:
        rec_boost = 0.0

    raw = 0.8 * req_mean + 0.2 * pref_mean + rec_boost
    return max(0.0, min(1.0, raw))


def rank_adjacent(
    accepted: list[dict],
    user_ids: set[str],
    user_names: set[str],
    user_canon: set[str],
) -> list[dict]:
    """Sort accepted candidates by `_score_one_adjacent_job` DESC.

    The score is computed once per job and stashed on the returned
    dict copy as `__adjacent_score__` so the responder/payload code
    (AR-6) can quote it without recomputing. Ties break by `job_id`
    (string sort) -- deterministic.
    """
    scored: list[tuple[float, str, dict]] = []
    for job in accepted:
        if not isinstance(job, dict):
            continue
        score = _score_one_adjacent_job(job, user_ids, user_names, user_canon)
        # Carry the score forward without mutating the caller's dict.
        job_with_score = dict(job)
        job_with_score["__adjacent_score__"] = score
        jid = job.get("job_id")
        jid_str = jid if isinstance(jid, str) else ""
        scored.append((score, jid_str, job_with_score))

    # Sort: score DESC, then job_id ASC (tie-breaker for reproducibility).
    scored.sort(key=lambda triple: (-triple[0], triple[1]))
    return [t[2] for t in scored]


def drop_excluded(
    ranked: list[dict],
    presented_job_ids,
) -> list[dict]:
    """Strip jobs whose `job_id` appears in `presented_job_ids`.

    The exclusion list is populated by the matcher after every
    `present_matches` / `present_near_miss` turn (AR-1a contract). A
    different-role recommendation should never re-surface a specific
    posting the user has already been shown.

    Defensive contract (AR-4 review round 2):
      - `presented_job_ids` of wrong type / None ->treated as empty
        exclusion set, but the dict-and-str sanitization on the
        ranked list still runs.
      - Non-str / empty `job_id` on a ranked entry ->entry dropped
        from output. The cookie boundary should never let one
        through, but presenting an un-comparable entry as a
        recommendation is USER-FACING wrong.
      - Non-dict ranked entries ->dropped.
    """
    if isinstance(presented_job_ids, (list, tuple, set, frozenset)):
        excluded: set[str] = {
            x for x in presented_job_ids if isinstance(x, str) and x
        }
    else:
        excluded = set()

    out: list[dict] = []
    for j in ranked:
        if not isinstance(j, dict):
            continue
        jid = j.get("job_id")
        if not isinstance(jid, str) or not jid:
            continue
        if jid in excluded:
            continue
        out.append(j)
    return out


def _synthesize_recommend_adjacent_roles_decision():
    """Pure ArbiterDecision factory for the recommend_adjacent_roles
    outcome. No arguments -- the responder payload is attached
    separately in AR-6's wiring."""
    from skillbridge.chat.arbiter import (
        ARBITER_REASON_ADJACENT_RECOMMENDATIONS,
        ArbiterDecision,
    )
    return ArbiterDecision(
        final_move="recommend_adjacent_roles",
        reason_code=ARBITER_REASON_ADJACENT_RECOMMENDATIONS,
        tone="brief_confident",
        arbiter_action="handler_synthesized_adjacent_recommendations",
        ask_slot=None,
        caps_applied=(),
        notes=None,
    )


# =========================================================================
# AR-3 retrieval + acceptance pipeline
# =========================================================================
# All five functions below ship as DEAD CODE: no production caller
# dispatches into any of them until AR-6 wires the Redis-gated
# adjacency hook in `_try_v2_path`. The activation-safety audit
# (test_ar1c_parity_and_activation.py) catches any leak.

ADJACENT_MIN_REQUIRED_COVERAGE: float = 0.45
ADJACENT_MIN_TRANSFERABLE_STRENGTH: float = 0.70


def _load_active_jobs_with_skills() -> list[dict]:
    """One SQL pass: every `core.v_current_job` row joined with its
    `extracted.job_skill` rows, grouped by job_id, ordered by stable
    NULL-tolerant keys so the per-job skill list is reproducible across
    runs.

    Replaces the N+1 pattern at engine.py:1428 / :1531 for the
    adjacency path. The existing `_fetch_job_skills(job_id)` is
    untouched for the legacy matcher path.

    Stable ordering rationale (v10 fix):
        ORDER BY j.posted_date DESC NULLS LAST,
                 j.job_id,
                 s.importance_rank NULLS LAST,
                 s.confidence DESC NULLS LAST,
                 s.skill_name
    The trailing `s.skill_name` (NOT NULL per schema.sql:166) is the
    deterministic tie-breaker for unnormalized rows whose `skill_id`
    is NULL, so two runs produce byte-identical evidence summaries.

    Row filter: `if r.get("skill_name")` -- `extracted.job_skill.skill_id`
    is nullable (only the step-5 normalization sweep populates it),
    so filtering on `skill_id` would silently drop valid unnormalized
    rows. `skill_name` is `TEXT NOT NULL`. Pure LEFT-JOIN NULL-skill
    rows (jobs with no extracted skills) have `skill_name IS NULL`
    and are correctly skipped, while the job itself stays in the
    candidate pool (rejected later by `no_required_non_credential_skills`).
    """
    from skillbridge.db import sync_cursor

    sql = """
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
    """
    with sync_cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    by_id: dict[str, dict] = {}
    SKILL_COLS = {"skill_id", "skill_name", "confidence",
                  "importance_rank", "skill_type"}
    for r in rows:
        jid = str(r["job_id"])
        job = by_id.get(jid)
        if job is None:
            job = {k: r[k] for k in r if k not in SKILL_COLS}
            # Postgres returns `core.job_posting.job_id` as a
            # `uuid.UUID`. If we kept the raw object on the job dict,
            # downstream code that expects strings (presented_job_ids
            # comparisons, snapshot sanitizers that demand
            # `isinstance(x, str)`, JSON serialization) would silently
            # break. Overwrite with the str-coerced form -- same
            # value the dict key uses.
            job["job_id"] = jid
            job["skills"] = []
            by_id[jid] = job
        if r.get("skill_name"):
            job["skills"].append({k: r[k] for k in SKILL_COLS})
    return list(by_id.values())


def build_user_skill_sets(
    skills: list[StagedSkill],
) -> tuple[set[str], set[str], set[str]]:
    """Build the three sets `_skill_match_strength` consumes — derived
    from the authoritative `UserSkillRow` list.

    Eligibility gate (same as before; now owned by `build_user_skill_rows`):
      - isinstance(s, StagedSkill);
      - source in {"resume", "chat"};
      - confidence is a finite, non-boolean numeric in [0, 1];
      - confidence >= 0.6;
      - skill_name is a non-empty str after .strip().

    The sets are derived from `build_user_skill_rows(skills)` in a
    single pass via `derive_user_skill_sets` — there is no parallel
    construction path. Scoring (which reads the sets) and attribution
    (which reads the rows) therefore cannot disagree about which user
    skills were eligible.

    Why the gate exists at all (AR-3 review round 2):
      Without it, three resume @ 0.8 anchor skills can clear the
      `has_usable_skill_evidence` floor, then OFF-SOURCE or
      LOW-CONFIDENCE rows added to the staged profile (e.g. a
      synthetic fallback extraction with confidence=0.3) would
      inflate `coverage` against an unrelated job's required
      skills.
    """
    return derive_user_skill_sets(build_user_skill_rows(skills))


def build_anchor_skill_sets(
    skills: list[StagedSkill],
) -> tuple[set[str], set[str], set[str]]:
    """Anchor-only sets for the AR-3 transferable-anchor gate.

    Filters `skills` through `is_non_generic_transferable` BEFORE
    building the (ids, names, canon) triple, so the gate scores
    required non-credential job skills ONLY against the user's
    anchor-eligible skills. An unrelated concrete skill (a matched
    credential, an off-source row, a generic) cannot inflate the
    gate because it never enters these sets.
    """
    anchors = [s for s in skills if is_non_generic_transferable(s)]
    return build_user_skill_sets(anchors)


def _safe_str(value) -> str:
    """Defensive str-cast for persisted fields whose forged-cookie
    types we don't trust. Non-str (or None) ->empty string."""
    return value if isinstance(value, str) else ""


def retrieve_candidates(
    staged: StagedProfile,
    snapshot: dict | None,
    all_jobs: list[dict],
    user_ids: set[str],
    user_names: set[str],
    user_canon: set[str],
) -> list[dict]:
    """Broad SSM-filtered retrieval (v11 stage 1).

    A candidate enters the pool when it's SSM-region-proper, NOT in
    the user's exact target NOC, AND matches one of:
      - NOC minor-group hit (first 4 chars of `noc_code` match
        `staged.target_noc[:4]`), OR
      - Skill-evidence hit (`_skill_match_strength` returns non-zero
        for at least one job skill against the user sets).

    Same-target-NOC exclusion (AR-3 review round 2):
        Adjacency is "DIFFERENT-role discovery" by design. A job
        whose `noc_code` equals the user's target NOC exactly is
        the user's current target occupation; it must not appear in
        adjacency results. (Specific postings already presented in
        prior turns are excluded separately via
        `presented_job_ids` -- but that's per-posting; this is
        per-occupation.)

    Defensive guards (AR-3 review round 2):
        Cookie-deserialized state can carry malformed types
        (e.g. target_noc=72106 as int). String fields are read via
        `_safe_str`; `job["skills"]` is coerced to a list and each
        entry to a dict before reaching `_skill_match_strength`.

    NO acceptance gating here -- the strict AND gate runs in
    `accept_candidates`.
    """
    from skillbridge.match.engine import _skill_match_strength
    from skillbridge.match.region import is_ssm_region_job

    target_noc = _safe_str(staged.target_noc).strip()
    target_minor = target_noc[:4]
    out: list[dict] = []
    for job in all_jobs:
        if not isinstance(job, dict):
            continue
        if not is_ssm_region_job(job):
            continue
        job_noc = _safe_str(job.get("noc_code")).strip()
        # Same-target-NOC exclusion.
        if target_noc and job_noc and job_noc == target_noc:
            continue
        noc_hit = bool(target_minor) and job_noc[:4] == target_minor
        skill_hit = False
        raw_skills = job.get("skills") or []
        if not isinstance(raw_skills, list):
            raw_skills = []
        for js in raw_skills:
            if not isinstance(js, dict):
                continue
            # `_skill_match_strength` does `(skill_name or "").lower()`;
            # a non-str (e.g. `skill_name=7`) survives the `or ""`
            # short-circuit and crashes on `.lower()`. Guard at the
            # boundary so engine code is never called with a malformed
            # skill row.
            name = js.get("skill_name")
            if not isinstance(name, str) or not name.strip():
                continue
            strength, _stage = _skill_match_strength(
                js, user_ids, user_names, user_canon,
            )
            if strength > 0.0:
                skill_hit = True
                break
        if noc_hit or skill_hit:
            out.append(job)
    return out


# Telemetry drop reason constants. Per-candidate drop counts roll up
# into the turn's aggregate telemetry log line (AR-7).
_DROP_NO_EVIDENCE = "no_evidence"
_DROP_NO_REQUIRED_NON_CREDENTIAL = "no_required_non_credential_skills"
_DROP_CREDENTIAL = "credential"
_DROP_COVERAGE = "coverage"
_DROP_TRANSFERABLE = "transferable"


def _classify_required(job_skills: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split a job's skills into (required_credentials, required_non_credential).
    Preferred skills are ignored at acceptance time -- they only matter for
    ranking (AR-4). Defensive: non-dict entries are skipped (forged-cookie
    safety)."""
    from skillbridge.match.engine import _required_or_preferred

    req_cred: list[dict] = []
    req_non_cred: list[dict] = []
    for s in job_skills:
        if not isinstance(s, dict):
            continue
        if _required_or_preferred(s) != "required":
            continue
        # Defensive: extracted.job_skill.skill_name is TEXT NOT NULL in
        # the schema, but a forged in-memory job list (test or
        # cookie-deserialized) could carry a non-str. Drop those
        # entries so credentials/coverage scoring can't crash on
        # `.lower()`.
        name = s.get("skill_name")
        if not isinstance(name, str) or not name.strip():
            continue
        if is_credential_skill_name(name):
            req_cred.append(s)
        else:
            req_non_cred.append(s)
    return req_cred, req_non_cred


def accept_candidates(
    retrieved: list[dict],
    staged: StagedProfile,
    user_ids: set[str],
    user_names: set[str],
    user_canon: set[str],
) -> tuple[list[dict], dict[str, int]]:
    """Strict AND gate over the retrieval pool.

    Returns (accepted_jobs, drop_counts). The drop_counts dict carries
    aggregate counters keyed by `_DROP_*` constants for telemetry.

    Drop sequence (first failing gate wins; per-candidate exclusive):
      1. `_DROP_NO_EVIDENCE` -- evidence floor failed (whole pool
         drops on the first iteration; we short-circuit).
      2. `_DROP_NO_REQUIRED_NON_CREDENTIAL` -- job has no required
         non-credential skills to score against (e.g. credential-only
         postings).
      3. `_DROP_CREDENTIAL` -- the user fails at least one required
         credential check.
      4. `_DROP_COVERAGE` -- mean strength across required
         non-credential skills below `ADJACENT_MIN_REQUIRED_COVERAGE`.
      5. `_DROP_TRANSFERABLE` -- no required job-skill is matched at
         or above `ADJACENT_MIN_TRANSFERABLE_STRENGTH` by the user's
         anchor-only sets.

    AR-3 is dead until AR-6; see `match/adjacent.py` module docstring.
    """
    from skillbridge.match.engine import _skill_match_strength

    drops: dict[str, int] = {
        _DROP_NO_EVIDENCE: 0,
        _DROP_NO_REQUIRED_NON_CREDENTIAL: 0,
        _DROP_CREDENTIAL: 0,
        _DROP_COVERAGE: 0,
        _DROP_TRANSFERABLE: 0,
    }

    if not has_usable_skill_evidence(staged):
        drops[_DROP_NO_EVIDENCE] = len(retrieved)
        return [], drops

    anchor_ids, anchor_names, anchor_canon = build_anchor_skill_sets(staged.skills)
    accepted: list[dict] = []

    for job in retrieved:
        if not isinstance(job, dict):
            drops[_DROP_NO_REQUIRED_NON_CREDENTIAL] += 1
            continue
        raw_skills = job.get("skills") or []
        if not isinstance(raw_skills, list):
            raw_skills = []
        req_cred, req_non_cred = _classify_required(raw_skills)

        # Gate 1: a job with NO required non-credential skills can't
        # be scored.
        if not req_non_cred:
            drops[_DROP_NO_REQUIRED_NON_CREDENTIAL] += 1
            continue

        # Gate 2: every required credential must match. A missing one
        # blocks adjacency entirely (the whole point of adjacency is
        # to surface roles the user is NOT credentially blocked from).
        credential_blocked = False
        for c in req_cred:
            strength, _ = _skill_match_strength(
                c, user_ids, user_names, user_canon,
            )
            if strength <= 0.0:
                credential_blocked = True
                break
        if credential_blocked:
            drops[_DROP_CREDENTIAL] += 1
            continue

        # Gate 3: coverage = mean(strength) over required-non-credential.
        strengths = [
            _skill_match_strength(s, user_ids, user_names, user_canon)[0]
            for s in req_non_cred
        ]
        coverage = sum(strengths) / max(1, len(strengths))
        if coverage < ADJACENT_MIN_REQUIRED_COVERAGE:
            drops[_DROP_COVERAGE] += 1
            continue

        # Gate 4: anchor-only transferable gate.
        anchor_max = 0.0
        for s in req_non_cred:
            strength, _ = _skill_match_strength(
                s, anchor_ids, anchor_names, anchor_canon,
            )
            if strength > anchor_max:
                anchor_max = strength
            if anchor_max >= ADJACENT_MIN_TRANSFERABLE_STRENGTH:
                break
        if anchor_max < ADJACENT_MIN_TRANSFERABLE_STRENGTH:
            drops[_DROP_TRANSFERABLE] += 1
            continue

        accepted.append(job)

    return accepted, drops


def _synthesize_describe_adjacent_role_decision():
    """Pure ArbiterDecision factory for the describe_adjacent_role
    outcome. Built when the user resolves an ordinal reference
    ("tell me about the second one") against the live
    last_adjacent_snapshot in AR-6."""
    from skillbridge.chat.arbiter import (
        ARBITER_REASON_ADJACENT_DESCRIPTION,
        ArbiterDecision,
    )
    return ArbiterDecision(
        final_move="describe_adjacent_role",
        reason_code=ARBITER_REASON_ADJACENT_DESCRIPTION,
        tone="brief_confident",
        arbiter_action="handler_synthesized_adjacent_description",
        ask_slot=None,
        caps_applied=(),
        notes=None,
    )
