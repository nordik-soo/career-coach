"""AR-9.feat.coach-tiers CP1 step 11 — tiered-matches policy gate.

The existing `responder._check_ungrounded_provider` builds its
grounded set from `training_by_job` providers only. Under the new
tiered-matches view, user-skill brands like "QuickBooks" can appear
in `skill_alignment` records but never in training (because they're
brand names the user has, not training entities). The old check
rejected those mentions as ungrounded.

Step 11's narrow fix builds a broader per-turn GROUNDED_TERMS set
from the tier view's records:
  - User skills (`skill_alignment[i].user_skill` across all tiers)
  - Job requirements the user matches (`skill_alignment[i].job_requirement`)
  - Job requirements the user is missing
      (`non_blocking_gaps[i].job_requirement`,
       `prioritized_gaps[i].job_requirement`,
       `important_gaps[i]`)
  - Employer names (one per tier record)
  - Actual training providers
      (`prioritized_gaps[i].training_options[j].provider`)
  - Transferable-pair `user_skill` and `applies_to` (Sideways tier)

A known-provider token from `_KNOWN_TRAINING_PROVIDERS` is exempted
ONLY when its normalized form exactly matches a member of
GROUNDED_TERMS. No sentence classification. No fuzzy exemptions.
Genuinely ungrounded provider names (e.g. an LLM inventing
"Transportation Association of Canada" with no grounding anywhere
in the view) stay rejected.

Salary handling is OUT of scope: step 11's locked decision (option B)
omits salary from both the LLM prompt and the deterministic fallback.
The existing `$`-rejection in `_policy_ok_v2` stays as a defense-
in-depth backstop and is not modified here.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from skillbridge.chat.training_provider_registry import (
    _KNOWN_TRAINING_PROVIDERS,
    _PROVIDER_ABBREVIATIONS,
)

if TYPE_CHECKING:
    from skillbridge.chat.url_views import SanitizedResponderView


def _normalize_term(s: str | None) -> str:
    """Lowercase + strip. Mirrors the normalization
    `responder._check_ungrounded_provider` already applies to its
    `grounded` set so set-membership comparisons line up exactly.
    """
    if not isinstance(s, str):
        return ""
    return s.strip().lower()


def build_grounded_terms(view: "SanitizedResponderView") -> frozenset[str]:
    """Build the per-turn GROUNDED_TERMS set from the tier view.

    Returns a frozenset of normalized (lowercase + stripped) tokens.
    Empty strings are silently excluded; an empty view yields an
    empty set.
    """
    out: set[str] = set()

    # ----- Apply-today tier -----
    for item in view.prompt_tiered_apply_today:
        out.add(_normalize_term(item.employer))
        for a in item.skill_alignment:
            out.add(_normalize_term(a.user_skill))
            out.add(_normalize_term(a.job_requirement))
        for g in item.non_blocking_gaps:
            out.add(_normalize_term(g.job_requirement))

    # ----- Worth-a-try tier -----
    for item in view.prompt_tiered_worth_a_try:
        out.add(_normalize_term(item.employer))
        for a in item.skill_alignment:
            out.add(_normalize_term(a.user_skill))
            out.add(_normalize_term(a.job_requirement))
        for g in item.prioritized_gaps:
            out.add(_normalize_term(g.job_requirement))
            for t in g.training_options:
                out.add(_normalize_term(t.provider))

    # ----- Sideways-move tier -----
    for item in view.prompt_tiered_sideways_move:
        out.add(_normalize_term(item.employer))
        for a in item.skill_alignment:
            out.add(_normalize_term(a.user_skill))
            out.add(_normalize_term(a.job_requirement))
        for g in item.important_gaps:
            out.add(_normalize_term(g))
        for p in item.transferable_pairs:
            out.add(_normalize_term(p.user_skill))
            out.add(_normalize_term(p.applies_to))

    out.discard("")
    return frozenset(out)


def check_ungrounded_provider_for_tiered_matches(
    reply: str,
    view: "SanitizedResponderView",
) -> str | None:
    """Tiered-matches equivalent of
    `responder._check_ungrounded_provider`.

    Detects each `_KNOWN_TRAINING_PROVIDERS` token mentioned in the
    reply via a word-boundary regex on the lowercased reply (same
    pattern the existing check uses). For each detection, the token
    is exempted ONLY when its normalized form exactly matches a member
    of GROUNDED_TERMS. The first ungrounded provider name is returned
    as the rejection reason; None when every mention is grounded (or
    no provider is mentioned).

    Abbreviation table from the existing check
    (`_PROVIDER_ABBREVIATIONS`) is honoured: when a canonical name is
    in GROUNDED_TERMS, every registered abbreviation is treated as
    grounded too. This preserves the live-observed SCCC↔"Sault
    Community Career Centre" grounding contract.

    Post-step-11 ownership cleanup: both `_KNOWN_TRAINING_PROVIDERS`
    and `_PROVIDER_ABBREVIATIONS` now live in
    `chat.training_provider_registry` (a leaf module). Imported
    statically at module load — no more lazy import from the future
    consumer `responder.py`.
    """
    grounded = set(build_grounded_terms(view))
    for canonical in list(grounded):
        for abbr in _PROVIDER_ABBREVIATIONS.get(canonical, frozenset()):
            grounded.add(abbr)

    reply_lower = reply.lower()
    for provider in _KNOWN_TRAINING_PROVIDERS:
        pattern = rf"\b{re.escape(provider)}\b"
        if re.search(pattern, reply_lower):
            if provider not in grounded:
                return provider
    return None
