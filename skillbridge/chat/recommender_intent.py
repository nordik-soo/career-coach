"""LLM-based career intent classifier (locked 2026-06-22).

Classifies each user message into exactly ONE of seven career intents
via Anthropic tool_use. Tool output is strictly enum-validated; any
deviation falls back to "unclear" so the deterministic router can
handle the message safely.

This module is the deterministic input to the router; it does not
itself decide which engine fires. Router work is a separate step.

See [[project-recommender-peer-engine-locked]] for the design lock.
The pattern-based `_classify_intent` in `truth_summary.py` keeps
handling hard control signals (yes / no / decline / proceed /
correction / redirect). This classifier sits BESIDE it for rich
career intent. Neither replaces the other.

Cache key includes the small set of contextual fields that can
change classification of the same surface message:
  (message, pending_recommender_offer, target_role_text, last_assistant_move)

Cache is process-wide and unbounded for now; typical session
contains a small number of unique messages.
"""
from __future__ import annotations

import logging
from typing import Literal

import anthropic

from config import (
    ANTHROPIC_API_KEY,
    LLM_ENABLED,
    LLM_FALLBACK_MODEL,
    LLM_MODEL,
    LLM_TIMEOUT_SECONDS,
)

log = logging.getLogger(__name__)


# Closed enum -- locked per project_recommender_peer_engine_locked memo.
# Any change here is a design-lock break and must be co-decided with the
# router work, NOT silently added.
CareerIntent = Literal[
    "job_matching",
    "local_skill_gap",
    "training_recommendation",
    "noc_standard_comparison",
    "career_exploration",
    "application_help_out_of_scope",
    "unclear",
]


_VALID_INTENTS: frozenset[str] = frozenset({
    "job_matching",
    "local_skill_gap",
    "training_recommendation",
    "noc_standard_comparison",
    "career_exploration",
    "application_help_out_of_scope",
    "unclear",
})


_TOOL_NAME = "classify_career_intent"


# Anthropic tool schema. The "enum" constraint forces the model to pick
# from the seven values; we additionally validate in code as defense
# in depth.
_TOOL_SCHEMA: dict = {
    "name": _TOOL_NAME,
    "description": (
        "Classify the user's message into exactly one career intent. "
        "Choose the single best match. Return 'unclear' if the message "
        "is a bare yes/no, a greeting, off-topic, or genuinely "
        "ambiguous between multiple intents."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": list(sorted(_VALID_INTENTS)),
                "description": (
                    "One of the seven career intents."
                ),
            },
        },
        "required": ["intent"],
        "additionalProperties": False,
    },
}


_SYSTEM_PROMPT = """You are an intent classifier for SkillBridge SSM, a job-coaching system for Sault Ste. Marie, Ontario. Classify each user message into exactly ONE career intent from the closed enum.

You MUST call the `classify_career_intent` tool. Output NO prose. Output NO other tool call. The tool call IS your response.

INTENT DEFINITIONS:

- `job_matching`: user wants to see job postings or matches.
  Examples: "what jobs can I try", "show me admin work", "any openings in healthcare", "what's hiring in Sault", "I want to find work".

- `local_skill_gap`: user wants to know what specific skills they are missing for local roles, or what to improve to get hired locally.
  Examples: "what should I improve to get an admin job", "where am I weak for nursing", "what gaps do I have", "suggest me to improve", "what's missing from my profile for office work".

- `training_recommendation`: user wants training, courses, certifications, or learning paths.
  Examples: "what training should I take", "where can I learn QuickBooks", "what course do I need for admin", "what certifications would help".

- `noc_standard_comparison`: user wants to compare themselves to the Canadian / NOC occupational standard for a specific occupation.
  Examples: "compare me to the Canadian standard for accounting clerk", "am I meeting the NOC standard for admin", "check my skills against the standard for X".

- `career_exploration`: user wants to explore other careers or related roles given their background.
  Examples: "what else can I do with my background", "I was a delivery driver, what else fits", "what other careers should I consider", "what roles use the skills I already have".

- `application_help_out_of_scope`: user wants help with resume tailoring, cover letters, interview preparation, salary information, or other application coaching. SkillBridge does not handle these; the classifier surfaces them so the router can redirect politely.
  Examples: "help me write a cover letter", "what's the salary at X", "how should I tailor my resume", "interview tips", "is this offer fair".

- `unclear`: the message does not clearly fit any of the above intents. Includes bare yes/no/short replies, greetings, "ok", "thanks", off-topic chat, requests for clarification ABOUT the system, and genuinely ambiguous compound messages.

CONTEXT SIGNALS:

The user block includes optional context:
  PENDING_RECOMMENDER_OFFER: the recommender mode the system is waiting on a yes/no for (may be null)
  TARGET_ROLE_TEXT: the target occupation the user named (may be null)
  LAST_ASSISTANT_MOVE: the system's last action (may be null)

RULES:

1. Hard yes/no routing is handled OUTSIDE this classifier. A bare "yes" / "no" / "sure" / "no thanks" MUST classify as `unclear`.
2. When the user's message names an explicit intent, follow the message. Do NOT let pending context override an explicit message intent. Example: PENDING_RECOMMENDER_OFFER="local_gap_coach" and message="show me admin jobs" -> `job_matching`, not `local_skill_gap`.
3. When the message is ambiguous AND context is present, prefer `unclear` over guessing. The router will combine your enum with the hard-pattern signal and pending state.
4. Never invent intents. The seven values are exhaustive.
5. Intent classification does NOT require target context. If the user asks what to improve / what gaps they have / what training to take / how to compare to the standard / what else they can do, classify the intent EVEN WHEN `TARGET_ROLE_TEXT` is null. Substrate gating downstream will ask for the missing target. Examples (all valid with TARGET_ROLE_TEXT=null):
   - "what should I improve?" -> `local_skill_gap`
   - "what training should I take?" -> `training_recommendation`
   - "compare me to the standard" -> `noc_standard_comparison`
   - "what else can I do?" -> `career_exploration`
6. If `TARGET_ROLE_TEXT` contains a value that looks like the user's question echoed back (e.g. ends with "?", starts with a question word, or otherwise reads as a phrase rather than a role title), treat it as null for the purposes of this classification. A poisoned target context must not suppress a clear intent signal.
7. SLOT-FILL ANSWERS ARE NOT RECOMMENDER INTENTS. When `LAST_ASSISTANT_MOVE` names a substrate slot the system just asked for, and the user's message looks like they are answering that slot, classify as `unclear`. Slot fills are intake substrate, not recommender intent. The substrate gate and existing planner intake handle them downstream.

   Open-text slots the system asks for (treat as substrate context):
     - skills_text       (system asked: "tell me your skills" or "what software have you used")
     - target_role_text  (system asked: "what role" or "what kind of work")
     - experience_text   (system asked: "tell me about your work history")
     - education_text    (system asked: "tell me about your education")
     - transportation_text / availability_text / salary_expectation_text

   Examples of slot fills that MUST classify as `unclear`:
     - LAST_ASSISTANT_MOVE=skills_text, USER_MESSAGE="I've done bookkeeping, QuickBooks, Excel, payroll" -> `unclear`
     - LAST_ASSISTANT_MOVE=skills_text, USER_MESSAGE="forklift, customer service, cash handling" -> `unclear`
     - LAST_ASSISTANT_MOVE=target_role_text, USER_MESSAGE="accounting clerk" -> `unclear`
     - LAST_ASSISTANT_MOVE=target_role_text, USER_MESSAGE="something in healthcare" -> `unclear`
     - LAST_ASSISTANT_MOVE=experience_text, USER_MESSAGE="5 years at Walmart as a cashier" -> `unclear`

   EXCEPTION: if the user's message explicitly names a different intent ALONGSIDE the slot answer, classify the explicit intent. Example:
     - LAST_ASSISTANT_MOVE=skills_text, USER_MESSAGE="bookkeeping, QuickBooks -- and what training should I take for accounting?" -> `training_recommendation` (explicit recommender ask wins)

   The principle: slot fills go to substrate; intent questions go to recommender; mixed messages let the explicit intent win.
"""


# Cache: process-wide, keyed on the four context fields that can
# legitimately change classification of the same surface message.
_INTENT_CACHE: dict[
    tuple[str, str | None, str | None, str | None],
    CareerIntent,
] = {}


_client: anthropic.Anthropic | None = None


def _client_get() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY.startswith("PLACEHOLDER"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY missing or placeholder. "
                "Set LLM_ENABLED=false to disable career intent classification."
            )
        _client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY,
            timeout=LLM_TIMEOUT_SECONDS,
        )
    return _client


def classify_career_intent(
    *,
    message: str | None,
    pending_recommender_offer: str | None = None,
    target_role_text: str | None = None,
    last_assistant_move: str | None = None,
) -> CareerIntent:
    """Classify a user message into one of the seven career intents.

    Returns "unclear" when:
      - the message is empty / blank / not a string
      - the LLM is disabled at config level
      - the API call fails (network, auth, etc.)
      - the tool returns an invalid / missing enum value
      - the response carries no tool_use block

    Caching: same (message, pending_offer, target, last_move) tuple
    returns the cached enum without re-calling the LLM. Cache is
    process-wide; reset_cache() clears it (test helper).
    """
    if not isinstance(message, str):
        return "unclear"
    norm_msg = message.strip()
    if not norm_msg:
        return "unclear"

    cache_key = (
        norm_msg,
        pending_recommender_offer,
        target_role_text,
        last_assistant_move,
    )
    cached = _INTENT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if not LLM_ENABLED:
        _INTENT_CACHE[cache_key] = "unclear"
        return "unclear"

    try:
        raw_intent = _call_classifier_llm(
            message=norm_msg,
            pending_recommender_offer=pending_recommender_offer,
            target_role_text=target_role_text,
            last_assistant_move=last_assistant_move,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "recommender_intent classifier call failed (%s); returning unclear",
            type(exc).__name__,
        )
        _INTENT_CACHE[cache_key] = "unclear"
        return "unclear"

    if raw_intent not in _VALID_INTENTS:
        log.warning(
            "recommender_intent classifier returned invalid value=%r; "
            "returning unclear",
            raw_intent,
        )
        _INTENT_CACHE[cache_key] = "unclear"
        return "unclear"

    result: CareerIntent = raw_intent  # type: ignore[assignment]
    _INTENT_CACHE[cache_key] = result
    return result


def _build_user_block(
    *,
    message: str,
    pending_recommender_offer: str | None,
    target_role_text: str | None,
    last_assistant_move: str | None,
) -> str:
    """Serialize context for the LLM. Null fields are passed explicitly
    so the model can see absence vs presence."""
    return "\n".join([
        f"USER_MESSAGE: {message}",
        f"PENDING_RECOMMENDER_OFFER: {pending_recommender_offer or 'null'}",
        f"TARGET_ROLE_TEXT: {target_role_text or 'null'}",
        f"LAST_ASSISTANT_MOVE: {last_assistant_move or 'null'}",
    ])


def _call_classifier_llm(
    *,
    message: str,
    pending_recommender_offer: str | None,
    target_role_text: str | None,
    last_assistant_move: str | None,
) -> str:
    """Issue the tool_use call. Returns the raw intent string or raises.

    Caller is responsible for enum validation.
    """
    client = _client_get()
    user_block = _build_user_block(
        message=message,
        pending_recommender_offer=pending_recommender_offer,
        target_role_text=target_role_text,
        last_assistant_move=last_assistant_move,
    )
    system_blocks = [{
        "type": "text",
        "text": _SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }]

    def _do_call(model: str):
        # temperature=0 for routing stability. The schema constrains
        # the output shape but not choice stability; lower randomness
        # matters when this enum drives engine selection.
        return client.messages.create(
            model=model,
            max_tokens=200,
            temperature=0,
            system=system_blocks,
            tools=[_TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": _TOOL_NAME},
            messages=[{"role": "user", "content": user_block}],
        )

    try:
        resp = _do_call(LLM_MODEL)
    except anthropic.APIStatusError as e:
        if e.status_code in (429, 529, 503) and LLM_MODEL != LLM_FALLBACK_MODEL:
            log.warning(
                "career intent classifier overloaded on %s; falling back to %s",
                LLM_MODEL, LLM_FALLBACK_MODEL,
            )
            resp = _do_call(LLM_FALLBACK_MODEL)
        else:
            raise

    for block in resp.content:
        block_type = getattr(block, "type", None)
        if block_type == "tool_use":
            tool_name = getattr(block, "name", None)
            if tool_name != _TOOL_NAME:
                continue
            tool_input = getattr(block, "input", None)
            if not isinstance(tool_input, dict):
                continue
            intent = tool_input.get("intent")
            if isinstance(intent, str):
                return intent
    raise RuntimeError("LLM returned no usable tool_use block")


def reset_cache() -> None:
    """Test helper: drop the in-memory intent cache."""
    _INTENT_CACHE.clear()


def cache_size() -> int:
    """Test helper: current cache entry count."""
    return len(_INTENT_CACHE)
