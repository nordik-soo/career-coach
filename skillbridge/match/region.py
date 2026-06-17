"""SSM region predicate for adjacent recommendations (AR-1b).

Dedicated, neutral boolean predicate. Cookie/chat-layer code consumes
`is_ssm_region_job(job)` to decide whether a `core.v_current_job` row
is in the Sault-Ste.-Marie-proper service area.

Aliases are matched at WORD BOUNDARIES, not as unrestricted substrings,
so a location string like "Rossmore, Ontario" can NOT slip in via a
naïve `"ssm" in "rossmore"` check.

Why this is a dedicated module:
  - `match/engine.py:639 _location_boost(job_location, region_code)` is
    a scoring function returning a float; reusing it for boolean
    decisions would couple the boost weights to scope filtering.
  - `ingest/partners.py:296 _is_sccc_ssm_location(location, url)` is
    URL-aware ingestion code that uses the partner-feed URL pattern
    as a hint; chat-layer code can't supply a URL.
  - `config.py:152 LOCAL_CITIES` reads `.env:LOCAL_CITIES` which by
    operator default INCLUDES Wawa, Blind River, Chapleau, Algoma and
    other communities (the "local-boost" radius). Reusing it for
    adjacency scope would silently widen the candidate pool past the
    locked product scope (SSM proper only -- see chat/prompts.py:154,
    :483 and chat/responder.py:18 "ZERO data outside Sault Ste.
    Marie").

This module is dead code until AR-3 wires it into `retrieve_candidates`
and AR-6 activates the adjacency engine. See
`docs/adjacent-recommendations-design.md`.
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

# Word-boundary patterns derived from the alias set. Used by
# `is_ssm_region_job` so that a naive substring check can't admit
# unrelated locations like "Rossmore, Ontario" (which contains "ssm"
# as a substring of "rossmore"). `\b` is Python's "transition between
# a word character and a non-word character" anchor; alias text with
# punctuation (e.g. "sault ste. marie") still anchors correctly because
# only the word-character boundaries matter at the alias edges.
_SSM_PROPER_LOCATION_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(r"\b" + re.escape(alias) + r"\b")
    for alias in _SSM_PROPER_LOCATION_ALIASES
)

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


def is_ssm_region_job(job: dict[str, Any]) -> bool:
    """True iff a `core.v_current_job` row is in Sault Ste. Marie proper.

    Decision precedence:
      1. region_code present AND lower(region_code) in
         `_SSM_PROPER_REGION_CODES`                  → True
      2. region_code present AND not matched         → False
         (explicit non-SSM region; never accept)
      3. region_code missing AND lower(location) contains any alias
         in `_SSM_PROPER_LOCATION_ALIASES`           → True
      4. otherwise                                   → False
         (CONSERVATIVE: missing location is treated as non-SSM rather
         than admitted.)

    Does NOT consult `config.LOCAL_CITIES`. Algoma communities never
    enter the adjacency candidate pool.
    """
    code_raw = job.get("region_code")
    code = code_raw.strip().lower() if isinstance(code_raw, str) else ""
    if code:
        return code in _SSM_PROPER_REGION_CODES
    loc_raw = job.get("location")
    loc = loc_raw.strip().lower() if isinstance(loc_raw, str) else ""
    if not loc:
        return False
    return any(p.search(loc) for p in _SSM_PROPER_LOCATION_PATTERNS)
