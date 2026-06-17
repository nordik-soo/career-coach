"""Single source of truth for engine + extractor + dataset version strings.

These strings appear in every API response envelope and every analytics row.
Bump them following the rule in the design doc:
  major: schema or non-comparable formula change
  minor: formula tweak / new feature / reweighting
  patch: bug fix with no expected behaviour change
"""
from __future__ import annotations

from datetime import datetime, timezone

ENGINE_VERSION_JOB_MATCH = "job-match-v1.1.0"   # PR 10: + work_type_fit, shift_fit
ENGINE_VERSION_TRAINING_REC = "training-rec-v1.0.0"
EXTRACTOR_VERSION_RULE = "rule-extractor-v1.0.0"
EXTRACTOR_VERSION_LLM = "llm-haiku-extractor-v1.2.0"   # Sprint 5: + required/preferred skill_type labels
EXTRACTOR_VERSION_RESUME = "resume-haiku-v1"           # Sprint 1: resume facts extractor
CHAT_PROMPT_VERSION = "chat-prompt-v1.1.0"   # PR 10: NEXT_ACTION responder
CONFIDENCE_RULE_VERSION = "confidence-v1"


def dataset_version_today(prefix: str = "ssm-jobs") -> str:
    """Used by the pipeline when publishing a new dataset."""
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
