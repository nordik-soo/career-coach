"""SSM region authority — matching-market and adjacency predicates.

Two public functions:

  is_ssm_region_job(job) -> bool
      Predicate for chat/adjacency code to check whether a
      `core.v_current_job` row is in Sault-Ste.-Marie-proper.

  normalize_declared_job_location(source_location_text) -> (loc, remote)
      Ingest-time normalizer that populates
      `core.job_posting.normalized_job_location`. Added Step 1A
      (2026-07-15) with exact-match-after-cleanup semantics. Also
      returns non-SSM cities truthfully ("Wawa", "Elliot Lake", ...).

Both share `_SSM_PROPER_LOCATION_ALIASES`. Region codes
(`_SSM_PROPER_REGION_CODES`) are consulted ONLY by `is_ssm_region_job`
via `job.region_code` — they aren't location strings and don't belong
in `normalize_declared_job_location`'s pipeline.

Aliases are matched at WORD BOUNDARIES in the pre-Step-1A code path
that is now retired; exact-match-after-cleanup is the current rule
(see `normalize_declared_job_location`). "Rossmore, Ontario" and
"Wawa / SSM" are both rejected under the new rule.

Why this is a dedicated module:
  - `match/engine.py _location_boost` is a scoring function returning a
    float; reusing it for boolean decisions would couple boost weights
    to scope filtering. (Step 1A cutover 2026-07-16: _location_boost
    deleted; SSM-only v_current_job is the market boundary now.)
  - `ingest/partners.py:584 _is_sccc_ssm_location(location, url)` is
    URL-aware ingestion code that uses the partner-feed URL pattern as
    a hint; chat-layer code can't supply a URL.
  - `config.py SCCC_INGEST_LOCALITIES` (renamed 2026-07-16 from
    LOCAL_CITIES) reads `.env:SCCC_INGEST_LOCALITIES` which by operator
    default INCLUDES Wawa, Blind River, Chapleau, Algoma and other
    communities. That is an INGESTION allowlist — the matching engine
    never consults it. The rename makes the historical misuse
    impossible: a symbol named SCCC_INGEST_LOCALITIES cannot be
    misread as a matching-market allowlist.
"""
from __future__ import annotations

import re
from typing import Any


# Locked SSM-proper scope. ANY widening here must also widen
# chat/prompts.py and chat/responder.py to match (the locked product
# scope says "ZERO data outside Sault Ste. Marie"). Do NOT add Wawa,
# Blind River, Chapleau, Algoma, or other Algoma-district communities
# unless that scope is explicitly relocked.
_SSM_PROPER_LOCATION_ALIASES: frozenset[str] = frozenset({
    "sault ste. marie",
    "sault ste marie",
    "ssm",
})

# Step 1A (2026-07-15): word-boundary substring path retired. The old
# `_SSM_PROPER_LOCATION_PATTERNS` compiled `\b<alias>\b` regexes and
# used `.search()`, which admitted "Wawa / SSM" and "North of SSM" as
# SSM matches (word-boundary substring — technically anchored, but
# still substring). The current rule is exact case-insensitive
# equality against the alias set AFTER a deterministic cleanup pass;
# see `normalize_declared_job_location`. `is_ssm_region_job` consults
# that function for its location fallback.

# Trailing suffixes stripped in normalization (order matters: Remote
# BEFORE ", ON" so "SSM, ON (Remote)" cleans fully).
_REMOTE_SUFFIX_RE = re.compile(r"\s*\(remote\)\s*$", re.IGNORECASE)
_ON_SUFFIX_RE = re.compile(r"\s*,\s*(ontario|on)\s*$", re.IGNORECASE)

# Verified region codes. The production fixtures use "3557011"
# (Statistics Canada CSD code for Sault Ste. Marie) -- see
# tests/test_match_strength.py:160, test_hard_gates.py:44,
# test_occupation_resolver.py:73, test_semantic_match.py:229,
# test_score_explanation_structure.py:55. One legacy fixture
# (test_cert_to_matcher_promotion.py:416) uses the literal "SSM".
# Both are accepted case-insensitively.
_SSM_PROPER_REGION_CODES: frozenset[str] = frozenset({
    "3557011",
    "ssm",
})


def normalize_declared_job_location(
    source_location_text: str | None,
) -> tuple[str | None, bool]:
    """Deterministic normalization for `normalized_job_location`.

    Populates the ingest-time schema field per Step 1A spec
    (docs/matching-revise/step-1-source-data-integrity.md §2c). Also
    returns non-SSM cities truthfully ("Wawa", "Elliot Lake", etc.);
    hence the general name.

    Reuses `_SSM_PROPER_LOCATION_ALIASES` (the location-alias registry
    shared with `is_ssm_region_job`). Matches EXACTLY against the
    cleaned string, NOT via word-boundary substring search — that
    would admit "Wawa / SSM" and "North of SSM" as SSM matches,
    contradicting the "no substring inference" rule.

    Does NOT consult `_SSM_PROPER_REGION_CODES`; those are scoped to
    `is_ssm_region_job`'s higher-precedence `region_code` path, and
    belong on `job.region_code`, not `job.location`.

    Returns:
        (normalized_location, remote_flag)
        - normalized_location is "Sault Ste. Marie" iff the cleaned
          string equals an SSM alias literal (case-insensitive).
        - Otherwise the cleaned, Title-Cased input (Wawa → "Wawa").
        - None when the input is blank or non-string.
        remote_flag is True iff "(Remote)" was stripped from the tail.

    Pipeline (order matters):
      1. Reject non-string / empty → (None, False).
      2. Trim; collapse internal whitespace to single spaces.
      3. Strip trailing "(Remote)" (case-insensitive); set
         remote_flag = True if stripped.
      4. Strip trailing ", ON" or ", Ontario" (case-insensitive).
      5. If lowercased result exactly equals an SSM alias →
         ("Sault Ste. Marie", remote_flag).
      6. Otherwise → (Title-Cased cleaned input, remote_flag).
    """
    if not isinstance(source_location_text, str):
        return None, False
    cleaned = source_location_text.strip()
    if not cleaned:
        return None, False
    # Collapse internal whitespace runs.
    cleaned = re.sub(r"\s+", " ", cleaned)
    # Strip (Remote) suffix and record the flag.
    remote_flag = False
    stripped = _REMOTE_SUFFIX_RE.sub("", cleaned)
    if stripped != cleaned:
        remote_flag = True
        cleaned = stripped.strip()
    # Strip trailing ", ON" / ", Ontario".
    cleaned = _ON_SUFFIX_RE.sub("", cleaned).strip()
    if not cleaned:
        return None, remote_flag
    # Exact case-insensitive equality against the alias set.
    if cleaned.lower() in _SSM_PROPER_LOCATION_ALIASES:
        return "Sault Ste. Marie", remote_flag
    # Non-SSM truthful normalization: Title Case for display parity.
    return cleaned.title(), remote_flag


def is_ssm_region_job(job: dict[str, Any]) -> bool:
    """True iff a `core.v_current_job` row is in Sault Ste. Marie proper.

    Decision precedence:
      1. region_code present AND lower(region_code) in
         `_SSM_PROPER_REGION_CODES`                  → True
      2. region_code present AND not matched         → False
         (explicit non-SSM region; never accept)
      3. region_code missing AND
         normalize_declared_job_location(location)[0]
             == "Sault Ste. Marie"                   → True
      4. otherwise                                   → False
         (CONSERVATIVE: missing location is treated as non-SSM.)

    Step 1A (2026-07-15) retrofit: precedence 3 now uses exact-match-
    after-cleanup via `normalize_declared_job_location` — was a word-
    boundary substring search. Behavior change: inputs like
    "Wawa / SSM" and "North of SSM" that previously matched now
    return False. That was a bug.

    Does NOT consult `SCCC_INGEST_LOCALITIES` (an ingestion-side
    allowlist). Algoma communities never enter the adjacency
    candidate pool.
    """
    code_raw = job.get("region_code")
    code = code_raw.strip().lower() if isinstance(code_raw, str) else ""
    if code:
        return code in _SSM_PROPER_REGION_CODES
    normalized, _remote = normalize_declared_job_location(job.get("location"))
    return normalized == "Sault Ste. Marie"
