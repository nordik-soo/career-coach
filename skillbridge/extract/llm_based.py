"""LLM-based skill extractor using Haiku 4.5.

The LLM returns JSON; we validate and resolve against reference.skill.
If the call fails or returns invalid JSON, we fall back to the rule-based
extractor so the pipeline never stalls because the LLM is down.
"""
from __future__ import annotations

import logging
from typing import Any

from skillbridge.extract.base import (
    ExtractedSkill,
    SkillExtractor,
    resolve_many,
)
from skillbridge.llm import call_json
from skillbridge.versions import EXTRACTOR_VERSION_LLM

log = logging.getLogger(__name__)


JOB_SKILL_PROMPT = """You extract structured workplace skills from job postings.

Output ONLY valid JSON with this exact shape:
{
  "skills": [
    {"name": "<skill phrase>", "raw_phrase": "<the exact text snippet>", "skill_type": "required" | "preferred", "importance_rank": <1..N>, "confidence": <0..1>}
  ]
}

Rules:
- Extract 3-15 concrete workplace skills. Use short noun phrases ("forklift operation", "customer service").
- Do not invent skills not in the text.
- skill_type:
    * "required"  -- the posting frames this as mandatory. Cue words include
                     "required", "must have", "minimum qualifications",
                     "essential", "you will", "the candidate must".
                     Credentials/licences/certifications named without a
                     "preferred" qualifier (e.g. "Class G licence", "WHMIS",
                     "First Aid/CPR") are always required.
    * "preferred" -- the posting frames this as optional or bonus. Cue words
                     include "preferred", "asset", "nice to have", "bonus",
                     "considered an asset", "ideally", "we would love".
  When the posting is silent or ambiguous about whether a skill is mandatory,
  label it "required". Default to required when unsure -- this is safer for
  job matching than over-classifying as preferred.
- Importance rank 1 = most important for this role (within required first,
  then preferred). skill_type and importance_rank are independent.
- Confidence is your estimate of how clearly the text indicates this skill (0.0 - 1.0).
- No commentary, no markdown, JSON only.
"""

TRAINING_SKILL_PROMPT = """You extract structured workplace skills TAUGHT by a training program or workshop.

Output ONLY valid JSON with this exact shape:
{
  "skills": [
    {"name": "<skill phrase>", "raw_phrase": "<text snippet>", "confidence": <0..1>}
  ]
}

Rules:
- Only extract skills the program teaches or supports.
- 3-10 skills max.
- Short noun phrases.
- No commentary, JSON only.
"""

USER_SKILL_PROMPT = """You extract workplace skills, education, and experience from a person's own description.

Output ONLY valid JSON with this exact shape:
{
  "skills": [
    {"name": "<skill phrase>", "raw_phrase": "<phrase from the text>", "confidence": <0..1>}
  ]
}

Rules:
- Only extract skills the person actually claims.
- Be specific ("inventory tracking", not just "warehouse").
- No commentary, JSON only.
"""


def _parse_skills(
    payload: dict | None,
    default_confidence: float = 0.7,
    *,
    accept_skill_type: bool = False,
) -> list[ExtractedSkill]:
    if not payload or not isinstance(payload, dict):
        return []
    raw = payload.get("skills") or []
    if not isinstance(raw, list):
        return []
    out: list[ExtractedSkill] = []
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or "").strip()
        if not name:
            continue
        conf_raw: Any = item.get("confidence", default_confidence)
        try:
            confidence = float(conf_raw)
        except (TypeError, ValueError):
            confidence = default_confidence
        confidence = max(0.0, min(1.0, confidence))
        rank_raw = item.get("importance_rank")
        try:
            rank = int(rank_raw) if rank_raw is not None else i
        except (TypeError, ValueError):
            rank = i

        # skill_type is only consumed when the caller opted in (JD-side
        # extraction). User/training extractors don't carry it.
        # Conservative default: anything not exactly "preferred" becomes
        # "required" (matches the "default to required when unsure" rule).
        skill_type: str | None = None
        if accept_skill_type:
            raw_type = (item.get("skill_type") or "").strip().lower()
            skill_type = "preferred" if raw_type == "preferred" else "required"

        out.append(ExtractedSkill(
            skill_name=name,
            raw_phrase=item.get("raw_phrase") or name,
            confidence=confidence,
            importance_rank=rank,
            skill_type=skill_type,
        ))
    return out


class LlmSkillExtractor(SkillExtractor):
    version = EXTRACTOR_VERSION_LLM

    def __init__(self, fallback: SkillExtractor | None = None) -> None:
        self.fallback = fallback

    def _extract(
        self, prompt: str, user_text: str, *, accept_skill_type: bool = False,
        max_tokens: int | None = None,
    ) -> list[ExtractedSkill]:
        if not user_text:
            return []
        payload = call_json(prompt, user_text, max_tokens=max_tokens)
        skills = _parse_skills(payload, accept_skill_type=accept_skill_type)
        return resolve_many(skills)

    def extract_from_job(self, *, title: str, description: str) -> list[ExtractedSkill]:
        text = f"Job title: {title}\n\nJob description:\n{description or ''}"
        # JD-side responses are longer than user/training responses because
        # each skill now carries skill_type plus the existing fields.
        # 600 default tokens truncated multi-skill JSON mid-list (Sprint 5
        # activation); 2000 comfortably handles ~15 skills.
        skills = self._extract(
            JOB_SKILL_PROMPT, text,
            accept_skill_type=True, max_tokens=2000,
        )
        if not skills and self.fallback:
            log.info("LLM job extraction returned nothing; using rule-based fallback")
            return self.fallback.extract_from_job(title=title, description=description)
        return skills

    def extract_from_training(self, *, title: str, description: str) -> list[ExtractedSkill]:
        text = f"Program/workshop title: {title}\n\nDescription:\n{description or ''}"
        skills = self._extract(TRAINING_SKILL_PROMPT, text)
        if not skills and self.fallback:
            return self.fallback.extract_from_training(title=title, description=description)
        return skills

    def extract_from_user_text(self, text: str) -> list[ExtractedSkill]:
        skills = self._extract(USER_SKILL_PROMPT, text)
        if not skills and self.fallback:
            return self.fallback.extract_from_user_text(text)
        return skills
