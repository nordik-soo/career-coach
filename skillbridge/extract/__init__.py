"""Skill extractor package.

Picks the best available extractor at import time, honouring LLM_ENABLED.
"""
from __future__ import annotations

from skillbridge.extract.base import ExtractedSkill, SkillExtractor
from skillbridge.extract.llm_based import LlmSkillExtractor
from skillbridge.extract.rule_based import RuleBasedSkillExtractor
from skillbridge.llm import is_enabled as _llm_enabled


def default_extractor() -> SkillExtractor:
    """Return the best extractor available given current configuration."""
    if _llm_enabled():
        return LlmSkillExtractor(fallback=RuleBasedSkillExtractor())
    return RuleBasedSkillExtractor()


__all__ = [
    "SkillExtractor",
    "ExtractedSkill",
    "RuleBasedSkillExtractor",
    "LlmSkillExtractor",
    "default_extractor",
]
