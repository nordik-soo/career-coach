"""SkillExtractor interface + reference-skill resolution helpers.

Every extractor must return a list of ExtractedSkill items. Skill IDs are
resolved against reference.skill via exact + alias + fuzzy match. Unresolved
skills are still returned (skill_id=None) so the matcher can use the name.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable

from rapidfuzz import fuzz, process

from config import SKILL_FUZZY_THRESHOLD
from skillbridge.db import sync_cursor


@dataclass
class ExtractedSkill:
    skill_name: str
    raw_phrase: str | None = None
    confidence: float = 0.7
    importance_rank: int | None = None
    skill_id: str | None = None
    # 'required' | 'preferred' | None. None means the extractor didn't label
    # it (legacy rows pre-Sprint-5 or training/user extractors that don't
    # carry the distinction). The matcher treats None as 'required'.
    skill_type: str | None = None


class SkillExtractor(ABC):
    """Implementations: rule_based.RuleBasedSkillExtractor,
    llm_based.LlmSkillExtractor."""

    version: str = "abstract"

    @abstractmethod
    def extract_from_job(self, *, title: str, description: str) -> list[ExtractedSkill]: ...

    @abstractmethod
    def extract_from_training(self, *, title: str, description: str) -> list[ExtractedSkill]: ...

    @abstractmethod
    def extract_from_user_text(self, text: str) -> list[ExtractedSkill]: ...


# --------------------------------------------------------- Reference resolver

_REF_CACHE: dict[str, tuple[str, str]] | None = None  # alias_lower -> (skill_id, canonical_name)


def _load_reference_cache() -> dict[str, tuple[str, str]]:
    """Build an in-memory lookup of every alias and canonical name -> skill_id.

    Cached for process lifetime. Reload by restarting the API or calling
    invalidate_reference_cache().
    """
    global _REF_CACHE
    if _REF_CACHE is not None:
        return _REF_CACHE
    cache: dict[str, tuple[str, str]] = {}
    try:
        with sync_cursor() as cur:
            cur.execute("SELECT skill_id, skill_name, aliases FROM reference.skill")
            for row in cur.fetchall():
                sid = row["skill_id"]
                name = row["skill_name"]
                cache[name.lower()] = (sid, name)
                for alias in row.get("aliases") or []:
                    cache[alias.lower()] = (sid, name)
    except Exception:
        # Reference table may not exist yet on first boot — caller handles empty.
        pass
    _REF_CACHE = cache
    return cache


def invalidate_reference_cache() -> None:
    global _REF_CACHE
    _REF_CACHE = None


def resolve_skill(name: str) -> tuple[str | None, str]:
    """Resolve free-text skill phrase -> (skill_id, canonical_name).

    Strategy:
      1. exact alias / canonical-name match (case-insensitive)
      2. fuzzy match (rapidfuzz token_set_ratio) above SKILL_FUZZY_THRESHOLD
      3. fallback: return (None, original_name)
    """
    cache = _load_reference_cache()
    if not cache:
        return None, name
    key = name.strip().lower()
    if key in cache:
        sid, canonical = cache[key]
        return sid, canonical
    match = process.extractOne(
        key,
        list(cache.keys()),
        scorer=fuzz.token_set_ratio,
        score_cutoff=SKILL_FUZZY_THRESHOLD * 100,
    )
    if match:
        sid, canonical = cache[match[0]]
        return sid, canonical
    return None, name


def resolve_many(skills: Iterable[ExtractedSkill]) -> list[ExtractedSkill]:
    out: list[ExtractedSkill] = []
    for s in skills:
        if s.skill_id is None:
            sid, canonical = resolve_skill(s.skill_name)
            s.skill_id = sid
            if sid:
                s.skill_name = canonical
        out.append(s)
    return out
