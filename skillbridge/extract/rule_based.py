"""Rule-based skill extractor.

This is the no-LLM fallback. Strategy:
  - Build candidate phrases from the text using a small dictionary of common
    workplace skills + every alias/canonical name in reference.skill.
  - Match by substring + token-set fuzzy match.

Quality is lower than the LLM path. It's intended for:
  - bootstrapping before any Anthropic key is set
  - cost-zero pilots
  - deterministic regression tests
"""
from __future__ import annotations

import re
from collections import Counter

from rapidfuzz import fuzz

from skillbridge.extract.base import (
    ExtractedSkill,
    SkillExtractor,
    _load_reference_cache,
    resolve_many,
)
from skillbridge.versions import EXTRACTOR_VERSION_RULE

# A floor of common workplace/newcomer-relevant skills used when the
# reference.skill table is still empty (first boot).
SEED_SKILLS = [
    "customer service", "cash handling", "inventory", "forklift operation",
    "shipping and receiving", "workplace safety", "microsoft excel",
    "microsoft word", "data entry", "scheduling", "phone communication",
    "english", "french", "esl", "first aid", "whmis", "team work",
    "leadership", "problem solving", "time management", "cooking",
    "food safety", "food preparation", "cleaning", "patient care",
    "medical terminology", "carpentry", "welding", "electrical", "plumbing",
    "machine operation", "quality control", "warehouse", "retail",
    "sales", "marketing", "accounting", "bookkeeping", "computer skills",
    "typing", "driving", "class g license", "ds class license",
]


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-/+#]{2,}")


def _candidates_from_cache() -> list[str]:
    cache = _load_reference_cache()
    if cache:
        # Use canonical names only (avoid duplicating each skill via aliases).
        seen: set[str] = set()
        out: list[str] = []
        for _, (sid, canonical) in cache.items():
            if sid not in seen:
                seen.add(sid)
                out.append(canonical)
        return out
    return list(SEED_SKILLS)


def _find_skills(text: str) -> list[ExtractedSkill]:
    if not text:
        return []
    lowered = text.lower()
    tokens = _TOKEN_RE.findall(lowered)
    token_freq = Counter(tokens)
    candidates = _candidates_from_cache()

    hits: list[ExtractedSkill] = []
    seen: set[str] = set()
    for cand in candidates:
        cand_lower = cand.lower()
        if cand_lower in seen:
            continue
        # Substring hit gets high confidence.
        if cand_lower in lowered:
            hits.append(ExtractedSkill(
                skill_name=cand,
                raw_phrase=cand,
                confidence=0.85,
            ))
            seen.add(cand_lower)
            continue
        # Fuzzy hit on multi-word phrases (avoid one-token false positives).
        if " " in cand_lower:
            score = fuzz.token_set_ratio(cand_lower, lowered)
            if score >= 90:
                hits.append(ExtractedSkill(
                    skill_name=cand,
                    raw_phrase=cand,
                    confidence=0.7,
                ))
                seen.add(cand_lower)

    # Importance rank by raw token frequency within the text (rough proxy).
    for h in hits:
        first_word = h.skill_name.split()[0].lower()
        h.importance_rank = token_freq.get(first_word, 0) or None
    hits.sort(key=lambda s: -(s.importance_rank or 0))
    for i, h in enumerate(hits):
        h.importance_rank = i + 1
    return hits


class RuleBasedSkillExtractor(SkillExtractor):
    version = EXTRACTOR_VERSION_RULE

    def extract_from_job(self, *, title: str, description: str) -> list[ExtractedSkill]:
        text = f"{title}\n\n{description or ''}"
        return resolve_many(_find_skills(text))

    def extract_from_training(self, *, title: str, description: str) -> list[ExtractedSkill]:
        text = f"{title}\n\n{description or ''}"
        return resolve_many(_find_skills(text))

    def extract_from_user_text(self, text: str) -> list[ExtractedSkill]:
        return resolve_many(_find_skills(text))
