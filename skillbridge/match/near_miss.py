"""Near-miss gap classification + candidate filter for chat
orchestration v2.2.

See docs/near-miss-gap-analysis-design.md for the full spec.

This module exposes two public functions:

  classify_gap(name, registry) -> NearMissCategory
      Classifies one missing-skill name into a near-miss bucket
      (credential / core_skill / operational).

  filter_near_miss_candidates(low_matches, target_role_text, target_noc)
      -> list[MatchResult]
      Filters a list of low-band MatchResult instances down to the
      subset that genuinely qualifies as a "near-miss" -- i.e. the
      candidate has named a specific target role and at least one
      low-band job matches that target by title or NOC. Output
      preserves engine ranking (input order).

classify_gap classifies one missing-skill name into one of three
near-miss buckets:

    credential    -> regulated / certified / licensed. Surfaces FIRST
                     in the responder's near-miss narration (highest
                     impact). Examples: 310T, Class G, WHMIS.

    core_skill    -> trainable workplace skill, not a credential.
                     Surfaces AFTER credentials. Examples: brake
                     system repair, transmission diagnostics, Excel
                     formulas.

    operational   -> job-as-acquired requirement (on-call availability,
                     contract supervision, hour tracking). FILTERED
                     out of the candidate list -- the responder never
                     sees these. They're not closeable training gaps;
                     calling them gaps would misrepresent the path.

Two-source rule (locked design Q2 + Q3):

  1. Registry hit (preferred): the name matches a canonical_name or
     alias in `data/training_registry.yaml`. Use the existing
     `Gap.category` field, mapped to near-miss vocabulary via
     `_REGISTRY_TO_NEAR_MISS`. NO new YAML field; existing data is
     the source of truth.

  2. Heuristic fallback: the name doesn't match the registry. Apply
     keyword rules + log INFO telemetry so unregistered gaps can be
     reviewed and promoted into the YAML over time (same backlog
     pattern as today's `unknown_gap=...` log).

This module is DEAD CODE in production until Slice N-3 extends the
arbiter to emit `present_near_miss` and Slice N-5 wires the handler
to compute preconditions and call `filter_near_miss_candidates`. Both
public functions are fully tested in isolation so the integration
slices are safe.
"""
from __future__ import annotations

import logging
import re
from typing import Literal

from skillbridge.match.engine import MatchResult
from skillbridge.training.registry import TrainingRegistry

log = logging.getLogger(__name__)


# ============================================================================
# Filter thresholds (locked design 2026-06-05)
# ============================================================================
# A low-band candidate qualifies as a near-miss if it matches the
# user's target by title or NOC. Score-based proximity alone is NOT
# enough -- a low-band "marketing coordinator" job at score 0.18 is
# not a near-miss for a truck technician just because of generic
# skill overlap on "communication". The whole point of near-miss is
# "the role exists" -- which requires matching the role, not just
# any skills.
TITLE_MATCH_SIMILARITY_THRESHOLD: float = 0.85


# ============================================================================
# Closed enum (the near-miss vocabulary)
# ============================================================================
NearMissCategory = Literal["credential", "core_skill", "operational"]


# ============================================================================
# Registry category -> near-miss bucket mapping
# ============================================================================
# The training registry has 4 categories today:
#   credential / license / safety_training / skill
# Per locked Q2 (2026-06-05 design review), we map them into the
# 2-of-3 near-miss buckets that the responder narrates. (Operational
# is NEVER a registry-sourced classification -- the YAML only models
# closeable gaps.)
#
# Why this collapse: at the newcomer-career-advice level, the legal
# distinction between "credential" and "license" doesn't change the
# path. What matters is "this needs to be earned somewhere
# authoritative." A future-proofing safety net: any registry category
# not listed here raises rather than silently falling through.
_REGISTRY_TO_NEAR_MISS: dict[str, NearMissCategory] = {
    "credential":      "credential",
    "license":         "credential",
    "safety_training": "credential",
    "skill":           "core_skill",
}


# ============================================================================
# Heuristic keyword tables (used when the registry doesn't know the gap)
# ============================================================================
# These keywords are tested via simple substring search on a lowercased
# name. Order matters: CREDENTIAL is checked first, then OPERATIONAL,
# then fallback to core_skill. Patterns are deliberately conservative
# -- false positives in either direction misclassify a gap, so add
# only well-evidenced keywords here.
_CREDENTIAL_KEYWORDS: tuple[str, ...] = (
    "certif",         # certificate, certification, certified
    "license",
    "licence",
    "ticket",         # "forklift ticket", "fall protection ticket"
    "qualification",  # "certificate of qualification"
    "credential",
)

_OPERATIONAL_KEYWORDS: tuple[str, ...] = (
    "availability",       # "on-call availability", "weekend availability"
    "supervision",        # "MTO contract supervision"
    "tracking",           # "driver hour tracking"
    "on-call",
    "on call",
    "shift willing",      # "shift willingness"
    "hour tracking",
)


# ============================================================================
# Public entry point
# ============================================================================
def classify_gap(name: str, registry: TrainingRegistry) -> NearMissCategory:
    """Classify a missing-skill name into one of the near-miss buckets.

    Pure function. Logs INFO telemetry exactly once per heuristic
    classification -- registry hits do NOT log because they're known
    curated entries. The telemetry tells us which non-YAML gaps are
    surfacing in the wild so we can decide which to promote into the
    registry.

    Inputs:
      - name: the missing-skill name from `MatchResult.score_explanation
        .required_missing` or `credential_gap_skills`. Raw -- may be
        cased like "Class G driver's license" or
        "MTO contract supervision".
      - registry: loaded TrainingRegistry. Pass `get_registry()` from
        the call site so caching is preserved.

    Returns:
      One of "credential" / "core_skill" / "operational".

    Empty / None / whitespace-only input returns "core_skill" as the
    safest fallback (avoids raising in the middle of a candidate-list
    build). This case is logged at WARNING because it shouldn't happen
    in production -- a gap name from the engine should always be
    non-empty.
    """
    if not name or not name.strip():
        log.warning(
            "near_miss classify_gap received empty name; defaulting to core_skill",
        )
        return "core_skill"

    # ---- Source 1: registry lookup (alias or canonical match) ----
    # `lookup` is the right surface -- it does exact alias/canonical
    # match WITHOUT the message-scan blocklist (_ALIASES_NOT_FOR_MESSAGE_SCAN).
    # The blocklist is for cold-message scanning where "Excel" could
    # accidentally match in plain English; here we have a known gap
    # name from the engine, so the bare alias is the right key.
    hit = registry.lookup(name)
    if hit is not None:
        mapped = _REGISTRY_TO_NEAR_MISS.get(hit.category)
        if mapped is None:
            # Defense-in-depth: a registry entry with a category value
            # we don't recognize is a config bug, not runtime data.
            # Raise so the test suite catches it before production.
            raise ValueError(
                f"registry Gap {hit.canonical_name!r} has unknown "
                f"category={hit.category!r}; expected one of "
                f"{sorted(_REGISTRY_TO_NEAR_MISS)}"
            )
        return mapped

    # ---- Source 2: heuristic fallback (with telemetry) ----
    cat = _classify_by_heuristic(name)
    log.info(
        "near_miss heuristic_classified gap=%r category=%s",
        name, cat,
    )
    return cat


# ============================================================================
# Heuristic body (no logging; the caller logs once per call)
# ============================================================================
def _classify_by_heuristic(name: str) -> NearMissCategory:
    """Pure keyword-based classifier. No I/O, no logging. The public
    entry point logs ONCE per heuristic decision; keeping this pure
    lets tests assert the keyword tables without log-mocking gymnastics.

    Priority is credential > operational > core_skill. A gap matching
    both a credential keyword and an operational keyword (rare, but
    possible -- e.g. a hypothetical "supervisor licence") classifies
    as credential because the credential category drives narration
    while operational drops the entry entirely; mis-flagging as
    operational would silently lose a real credential gap.
    """
    lowered = name.lower()
    if _contains_any(lowered, _CREDENTIAL_KEYWORDS):
        return "credential"
    if _contains_any(lowered, _OPERATIONAL_KEYWORDS):
        return "operational"
    return "core_skill"


def _contains_any(haystack: str, keywords: tuple[str, ...]) -> bool:
    """True when `haystack` contains any keyword as a substring.

    Substring (not word-boundary) is intentional: "certif" must match
    "certification", "certificate", "certified" with one entry. This
    is a minor false-positive risk for unrelated strings like
    "certifiable" -- acceptable given the conservative keyword set
    and the telemetry log that surfaces edge cases.
    """
    return any(k in haystack for k in keywords)


# ============================================================================
# Near-miss candidate filter (Slice N-2)
# ============================================================================
def filter_near_miss_candidates(
    low_matches: list[MatchResult],
    target_role_text: str | None,
    target_noc: str | None,
) -> list[MatchResult]:
    """Return the subset of low-band matches that qualify as a
    near-miss for the user's target.

    Inputs:
      - low_matches: candidates the handler has already filtered to
        the band == "low" + match_eligible == True subset. This
        function does NOT re-check the band/eligible invariants
        in the happy path; the contract is "you pre-filtered."
        But the defensive guard below drops any rogue non-low /
        non-eligible entry that slipped through, since misclassifying
        a stretch match as near-miss would silently demote the user's
        existing experience.
      - target_role_text: the user's target role string (e.g. "truck
        and coach technician"). When None / empty / whitespace, no
        candidate can be a title-match near-miss -- the function
        falls back to NOC matching alone, and if NOC is also None,
        returns []. The handler should not call this without one or
        the other, but the defensive return preserves the property
        "no target -> no near-miss" cleanly.
      - target_noc: NOC code resolved from target_role_text (e.g.
        "7321"). When None, NOC-based qualification is skipped.

    Returns the qualifying subset in INPUT ORDER. The engine sorts
    by (match_eligible, match_score) before we get here, so input
    order == engine ranking. Tests pin this.

    A candidate qualifies as near-miss if ANY of:
      (a) score_explanation.title_match_override is True
          -- the engine already decided this is a title match
      (b) score_explanation.title_match_similarity >= 0.85
          -- high lexical similarity to the target role
      (c) candidate.noc_code == target_noc
          -- engine-resolved NOC equals the user's target NOC

    A candidate is REJECTED (not a near-miss) when:
      - all three conditions above are false, OR
      - the candidate is not match_eligible OR not band == "low"
        (defensive guard against caller error)

    Score-based proximity alone is intentionally NOT a qualifier.
    Generic skill overlap is intentionally NOT a qualifier. A near-miss
    is "the role exists but you're under-qualified" -- which requires
    the role to actually be the user's role, not just a vaguely
    skill-overlapping role.
    """
    # Fast-fail: no target signal at all -> nothing can be a near-miss.
    has_role = bool(target_role_text and target_role_text.strip())
    has_noc = bool(target_noc and target_noc.strip())
    if not has_role and not has_noc:
        return []

    out: list[MatchResult] = []
    for m in low_matches:
        # Defensive: input contract says pre-filtered to eligible low
        # band, but a caller error here would silently change product
        # behavior (stretch matches sneaking into near-miss). Drop
        # anything that doesn't meet the band/eligible invariants
        # and log it once so the upstream bug surfaces.
        if not m.match_eligible:
            log.warning(
                "near_miss filter received ineligible candidate "
                "(job_id=%s) -- caller should pre-filter; dropping",
                m.job_id,
            )
            continue
        if m.match_band != "low":
            log.warning(
                "near_miss filter received non-low band candidate "
                "(job_id=%s band=%s) -- caller should pre-filter; "
                "dropping", m.job_id, m.match_band,
            )
            continue
        if _qualifies_as_near_miss(m, target_noc=target_noc):
            out.append(m)
    return out


# ============================================================================
# Payload builder for ResponderV2Input.near_miss_payload (Slice N-5)
# ============================================================================
# Defaults locked at design-time. Q4: cap 3 credentials + 3 core skills,
# stable (alphabetical) ordering for v1. Impact-based ordering ("which
# gap unlocks the most jobs") is a later slice.
DEFAULT_CREDENTIAL_CAP: int = 3
DEFAULT_CORE_SKILL_CAP: int = 3


def build_near_miss_payload(
    candidates: list[MatchResult],
    registry: TrainingRegistry,
    *,
    credential_cap: int = DEFAULT_CREDENTIAL_CAP,
    core_skill_cap: int = DEFAULT_CORE_SKILL_CAP,
) -> dict[str, Any]:
    """Build the dict the handler hands to ResponderV2Input.near_miss_payload.

    Inputs:
      - candidates: the output of `filter_near_miss_candidates` -- a
        non-empty list of near-miss MatchResult instances, in engine
        ranking order. Caller MUST pre-filter; this function does not
        re-check qualification.
      - registry: training registry used for:
          (a) classifying each gap via `classify_gap`
          (b) CANONICAL-NAME ALIGNMENT -- if the engine-surfaced gap
              name matches a registry alias, the canonical name is
              substituted so `_find_grounded_provider` in the responder
              can match it against `training_by_job["for_gap"]`. This
              is the Slice N-4 reviewer note: engine names like "310T
              certificate of qualification" align to canonical
              "310T technician certification".

    Output shape (matches ResponderV2Input.near_miss_payload contract):
      {
        "role":            str,          # job title from the top candidate
        "employer":        str | None,
        "job_count":       int,          # total candidates passed in
        "credential_gaps": list[str],    # capped, canonical-aligned, alpha-sorted
        "core_skill_gaps": list[str],    # capped, alpha-sorted
      }

    Algorithm:
      1. Pick the top candidate (engine ranking [0]); use its title /
         employer for the payload header.
      2. Pull its `score_explanation.required_missing`. If empty, fall
         back to `credential_gap_skills` -- some engine code paths
         populate one but not the other.
      3. For each gap name, run `classify_gap(name, registry)`:
            - credential / core_skill -> keep, after canonical alignment
            - operational             -> drop (filtered upstream)
      4. Cap each bucket at the configured cap value.
      5. Sort alphabetically (locked Q4 stable order for v1).

    Multi-candidate (locked Q6): we narrate the highest-scoring
    candidate only; the total count surfaces via `job_count` so the
    responder can pluralize ("I found 3 postings...").

    Empty/None gap lists are possible -- a candidate with zero
    surfaced missing skills (rare but seen with very thin job_skill
    coverage) still returns a valid payload; the responder's
    defensive fallback handles the no-gap case.
    """
    if not candidates:
        raise ValueError(
            "build_near_miss_payload called with empty candidate list; "
            "caller must short-circuit before this function"
        )

    top = candidates[0]
    raw_gaps = _extract_raw_gap_names(top)

    credentials: list[str] = []
    core_skills: list[str] = []
    seen: set[str] = set()  # dedup within each bucket

    for raw in raw_gaps:
        canonical_name = _canonical_or_raw(raw, registry)
        if canonical_name in seen:
            continue
        category = classify_gap(canonical_name, registry)
        if category == "operational":
            # Filtered out per design; never narrated.
            continue
        if category == "credential":
            credentials.append(canonical_name)
        elif category == "core_skill":
            core_skills.append(canonical_name)
        seen.add(canonical_name)

    credentials = sorted(credentials)[:credential_cap]
    core_skills = sorted(core_skills)[:core_skill_cap]

    return {
        "role":            top.title,
        "employer":        top.employer,
        "job_count":       len(candidates),
        "credential_gaps": credentials,
        "core_skill_gaps": core_skills,
    }


def _extract_raw_gap_names(candidate: MatchResult) -> list[str]:
    """Pull missing-skill names from the engine's score_explanation.

    Prefers `required_missing` (the canonical engine field). Falls back
    to `credential_gap_skills` if required_missing is missing or empty
    -- some code paths populate one but not the other (e.g. when a job
    has only credential-tagged requirements).

    Returns a list preserving original engine order. Dedup happens in
    the caller after canonical alignment, since two engine names can
    collapse to one canonical (e.g. "310T" alias + "310T technician
    certification" canonical both map to the same canonical).
    """
    expl = candidate.score_explanation or {}
    required_missing = expl.get("required_missing") or []
    if required_missing:
        return [str(g) for g in required_missing if g]
    credential_gaps = expl.get("credential_gap_skills") or []
    return [str(g) for g in credential_gaps if g]


def _canonical_or_raw(name: str, registry: TrainingRegistry) -> str:
    """Look up a gap name in the registry. If hit, return the canonical
    name (so downstream `for_gap` lookup matches). If miss, return the
    raw name verbatim (the engine surfaced it; we don't invent).

    This is the per-Slice-N-4-review canonical-name alignment helper.
    Without this, `_find_grounded_provider` in the responder can fail
    silently when the engine name and registry canonical name differ.
    """
    hit = registry.lookup(name)
    return hit.canonical_name if hit is not None else name


def _qualifies_as_near_miss(
    candidate: MatchResult,
    *,
    target_noc: str | None,
) -> bool:
    """Pure predicate: does this single low-band candidate qualify as
    a near-miss for the given target? See filter_near_miss_candidates
    for the full rule list.

    Kept separate so tests can pin the per-candidate logic without
    constructing whole input lists.
    """
    expl = candidate.score_explanation or {}

    # Condition (a): engine already declared a title match
    if bool(expl.get("title_match_override")):
        return True

    # Condition (b): high title similarity even without override
    # The engine's similarity is a float in [0, 1] or None when the
    # title comparison didn't run. Guard against None and non-float
    # explicitly -- a typo "0.85" string in the dict would otherwise
    # be coerced or crash.
    similarity = expl.get("title_match_similarity")
    if isinstance(similarity, (int, float)) and similarity >= TITLE_MATCH_SIMILARITY_THRESHOLD:
        return True

    # Condition (c): NOC match. None target_noc disables this rule.
    # Comparison is strict equality on the canonical NOC string.
    if target_noc and candidate.noc_code and candidate.noc_code == target_noc:
        return True

    return False
