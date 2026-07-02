"""Skill-alignment provenance — the attribution authority for the matcher.

`UserSkillRow` is the SINGLE source of truth for which user skills enter
the matching engine on any given turn. The (ids, names, canon) sets that
`_skill_match_strength` already consumes are DERIVED from rows in one
pass — never assembled independently — so scoring and attribution
cannot disagree about which user skills were eligible.

`SkillAlignment` is the matcher-output projection that names, for one
job-side requirement, which user skill produced the match and at which
public stage (`exact` | `fuzzy` | `semantic`). It carries
`is_normalized_equal` so the responder prompt can reserve strong
wording ("they ask for X, which you have") for cases of literal
normalized equality, and use "your X aligns with their Y" for every
other match — including alias and substring matches that share the
public `exact` stage but are not literal equalities.

Evidence-eligibility constants live here too. They previously lived in
`match/adjacent.py`; they were moved so this module can own user-skill
provenance end-to-end without a circular dependency between adjacency
and alignment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from skillbridge.match.aliases import _key, canonicalize_skill
from skillbridge.session.staging import StagedSkill


# ---------------------------------------------------------------- evidence
# Accepted StagedSkill.source values. Production sources today (per
# resume/derive.py and chat/handler.py) are {"resume", "chat"}.
_ACCEPTED_EVIDENCE_SOURCES: frozenset[str] = frozenset({"resume", "chat"})

# Minimum per-skill confidence for an evidence-eligible record.
_MIN_EVIDENCE_CONFIDENCE: float = 0.6


def _is_valid_normalized_score(x: Any) -> bool:
    """True iff x is a finite, non-boolean numeric in [0.0, 1.0].

    Rejects booleans (Python's bool subclasses int, so True would pass
    an isinstance(x, int) check), NaN, ±inf, and out-of-range values.
    """
    if isinstance(x, bool):
        return False
    if not isinstance(x, (int, float)):
        return False
    fx = float(x)
    if math.isnan(fx) or math.isinf(fx):
        return False
    return 0.0 <= fx <= 1.0


# ---------------------------------------------------------- attribution
@dataclass(frozen=True)
class UserSkillRow:
    """One evidence-eligible user skill, with attribution preserved.

    Fields:
      skill_id  — reference.skill.skill_id (canonical OaSIS taxonomy
                  ID) when the skill was resolved by the extractor
                  (chat path) or the resume derivation
                  (skillbridge.resume.derive, 2026-07-01 fix). None
                  for concrete resume vocabulary that has no
                  reference.skill entry AND for skills where fuzzy
                  matching was intentionally skipped. Independent of
                  whether the user has a persisted profile row --
                  anonymous users' staged skills carry resolved IDs
                  identically to signed-up users.
      text      — the ORIGINAL user-typed phrasing. This is what the
                  responder prompt should quote when surfacing "your X"
                  in the coach response. Preserved verbatim.
      name      — text.strip().lower(). Matches the semantics of the
                  legacy `user_skill_names` set so the matcher's
                  case-insensitive comparisons behave byte-stable.
      canon     — canonicalize_skill(text). Alias-folded canonical form
                  used by the matcher's canonical-equality rung.

    Two rows can share the same `name` (user typed "QB" and "qb" in
    the same session) — rows preserve every evidence-eligible entry
    so the semantic-argmax path can map back to a distinct row even
    when set-derivation would fold them.
    """
    skill_id: str | None
    text: str
    name: str
    canon: str


SkillAlignmentStage = Literal["exact", "fuzzy", "semantic"]
SkillAlignmentSource = Literal["required", "preferred"]


@dataclass(frozen=True)
class SkillAlignment:
    """One satisfied job-side requirement, with attribution.

    Fields:
      user_skill           — UserSkillRow.text of the winning row.
                             Preserved in original casing for display.
      job_requirement      — the job_skill name as the listing has it.
                             Preserved in original casing for display.
      stage                — public match stage: "exact" | "fuzzy" |
                             "semantic". Aliases and word-bounded
                             substring matches collapse into "exact"
                             because the matcher's existing taxonomy
                             groups them at the strongest strength tier.
                             The internal rung that resolves ties
                             (skill_id > name eq > alias > substring >
                             fuzzy > semantic) is NOT propagated here.
      source               — "required" | "preferred". Mirrors the
                             matcher's required/preferred bucket.
      is_normalized_equal  — _key(user_skill) == _key(job_requirement).
                             True ONLY for literal normalized equality
                             (lowercase + non-alphanum collapsed). NOT
                             true for alias folds, substring matches,
                             fuzzy, or semantic. The prompt allows
                             "they ask for X, which you have" only when
                             this flag is True; otherwise default to
                             "your X aligns with their Y".
    """
    user_skill: str
    job_requirement: str
    stage: SkillAlignmentStage
    source: SkillAlignmentSource
    is_normalized_equal: bool


# ------------------------------------------------------- rows authority
def build_user_skill_rows(skills: list[StagedSkill]) -> list[UserSkillRow]:
    """Build the ordered authoritative list of evidence-qualified
    user-skill rows for one matching turn.

    Eligibility gate (same as the legacy `build_user_skill_sets`):
      - isinstance(s, StagedSkill);
      - s.source in _ACCEPTED_EVIDENCE_SOURCES;
      - _is_valid_normalized_score(s.confidence);
      - s.confidence >= _MIN_EVIDENCE_CONFIDENCE;
      - s.skill_name is a non-empty str after .strip().

    Ordering: input order is preserved exactly. The semantic-argmax
    path relies on row order matching the encoded embedding-matrix
    row order, so this must be stable. Rows are NOT deduplicated —
    two rows sharing the same `name` are valid and distinct. The
    set-derivation helper (`derive_user_skill_sets`) is where folding
    happens.

    Defensive: a forged-cookie StagedSkill could carry skill_name as
    int/bool/None; canonicalize_skill would crash on .lower(). Guard
    here so engine helpers never see malformed rows.
    """
    out: list[UserSkillRow] = []
    for s in skills:
        if not isinstance(s, StagedSkill):
            continue
        if s.source not in _ACCEPTED_EVIDENCE_SOURCES:
            continue
        if not _is_valid_normalized_score(s.confidence):
            continue
        if s.confidence < _MIN_EVIDENCE_CONFIDENCE:
            continue
        if not isinstance(s.skill_name, str):
            continue
        text = s.skill_name.strip()
        if not text:
            continue
        name = text.lower()
        canon = canonicalize_skill(text) or ""
        skill_id = str(s.skill_id) if s.skill_id else None
        out.append(UserSkillRow(
            skill_id=skill_id,
            text=text,
            name=name,
            canon=canon,
        ))
    return out


def derive_user_skill_sets(
    rows: list[UserSkillRow],
) -> tuple[set[str], set[str], set[str]]:
    """Derive the (ids, names, canon) sets the matcher consumes from
    the authoritative rows list.

    Single-pass derivation. The sets the matcher reads are guaranteed
    to be exactly the union of attributes across these rows — there
    is no independent construction path that could disagree with
    attribution.

    Returns (skill_ids, skill_names, skill_canons).
    """
    ids: set[str] = set()
    names: set[str] = set()
    canons: set[str] = set()
    for r in rows:
        if r.skill_id:
            ids.add(r.skill_id)
        names.add(r.name)
        if r.canon:
            canons.add(r.canon)
    return ids, names, canons


def is_normalized_equal(user_skill: str, job_requirement: str) -> bool:
    """Literal normalized equality on the alignment provenance flag.

    Uses `_key` (lowercase + non-alphanum collapsed + apostrophe
    stripped) — NOT `canonicalize_skill`, which would fold aliases
    like "QB" ↔ "QuickBooks" and over-claim literal equality.

    The reserved phrasing "they ask for X, which you have" is gated
    on this being True. Every other match — aliases, substring,
    fuzzy, semantic — must use the default "your X aligns with
    their Y".
    """
    if not user_skill or not job_requirement:
        return False
    return _key(user_skill) == _key(job_requirement)
