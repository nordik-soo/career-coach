"""Deterministic reference resolver (slice 2 step 2.1, 2026-07-03).

Given a user message and the surface items derived from
`ConversationFrame.latest_surface_items`, decide whether the message
references one of those items, requires clarification, or does not
name a reference at all.

Deterministic rules only (LLM fallback lives in step 2.3):

  1. `surface_items` empty     -> no_reference (nothing to resolve to).
  2. Single item + pronoun     -> resolved.
     Pronouns: "that", "it", "this role", "this one".
  3. Ordinal in range          -> resolved by ordinal position.
     Word forms first..fifth; digits 1..N with optional "st|nd|rd|th"
     suffix; "the first one" etc.
  4. Full-label case-insensitive substring in message -> resolved
     iff unique. If multiple items' labels appear in the message,
     that is ambiguous -> clarification.
  5. Multi-item + pronoun-only (no ordinal, no name) -> clarification.
  6. Otherwise                 -> no_reference.

Explicitly NOT implemented (locked with lead 2026-07-03):
  - Prefix / partial-name matching. The existing drilldown resolver
    dropped loose partials for real ambiguity reasons; this module
    inherits that discipline.
  - Fuzzy / Levenshtein distance. Any near-miss ("admin secretary"
    vs "Administrative assistant") is handed to the LLM fallback in
    step 2.3, not guessed at deterministically.

Pure function. No LLM, no DB, no store. Reads staged only insofar as
the caller has already derived surface items from it.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

import anthropic

from config import (
    ANTHROPIC_API_KEY,
    LLM_ENABLED,
    LLM_FALLBACK_MODEL,
    LLM_MODEL,
    LLM_TIMEOUT_SECONDS,
)

from skillbridge.chat.conversation_frame import SurfaceItem


log = logging.getLogger(__name__)


ResolveStatus = Literal["resolved", "clarification", "no_reference"]


@dataclass(frozen=True, slots=True)
class ResolveOutcome:
    """Result of one reference-resolution attempt.

    `status`:
      - "resolved":       `item` names the referent; caller acts on it.
      - "clarification":  message referenced the surface but ambiguity
                          requires asking the user which they mean.
                          `item` is None.
      - "no_reference":   message does not name a reference. `item` is
                          None. Caller falls through to normal flow.

    `reason`: telemetry-grade short label naming which rule fired.
    Stable across releases so log analysis groups on it.
    """

    status: ResolveStatus
    item: SurfaceItem | None
    reason: str


# Pronouns that unambiguously reference a single item when only one is
# present. "the one" is deliberately NOT included: "the one about
# admin work" is a very different signal (partial name), and this
# module leaves partial-name to the LLM fallback.
_PRONOUN_PATTERN = re.compile(
    r"\b(?:that(?:\s+one|\s+role)?|it|this(?:\s+one|\s+role))\b",
    re.IGNORECASE,
)


# Ordinal word forms mapped to 1-based positions. Kept small and
# explicit; caller checks range against surface length.
_ORDINAL_WORDS: dict[str, int] = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
}


# Ordinal patterns:
#   - word forms (first / second / ...) with optional leading "the".
#   - digit forms (1 / 2 / 3) with optional ordinal suffix (1st / 2nd /
#     3rd / 4th) and optional leading "the".
# Both require word boundaries so "second" inside "secondary" does not
# false-match.
_ORDINAL_WORD_PATTERN = re.compile(
    r"\b(?:the\s+)?(first|second|third|fourth|fifth)(?:\s+one)?\b",
    re.IGNORECASE,
)
_ORDINAL_DIGIT_PATTERN = re.compile(
    r"\b(?:the\s+)?#?(\d+)(?:st|nd|rd|th)?(?:\s+one)?\b",
    re.IGNORECASE,
)


def _find_pronoun(message: str) -> bool:
    """True iff the message contains a resolver-relevant pronoun."""
    return bool(_PRONOUN_PATTERN.search(message))


def _find_ordinal(message: str) -> int | None:
    """Return the 1-based ordinal named by the message, or None.

    Word forms take precedence over digit forms so "the first one"
    doesn't accidentally return 1 via a digit-only fallback that
    misfires on something else in the same message. When both a word
    and a digit ordinal are present, word wins deterministically.
    """
    word_hit = _ORDINAL_WORD_PATTERN.search(message)
    if word_hit is not None:
        return _ORDINAL_WORDS[word_hit.group(1).lower()]
    digit_hit = _ORDINAL_DIGIT_PATTERN.search(message)
    if digit_hit is not None:
        try:
            return int(digit_hit.group(1))
        except ValueError:
            return None
    return None


def _find_label_matches(
    message: str,
    surface_items: tuple[SurfaceItem, ...],
) -> tuple[SurfaceItem, ...]:
    """Return the subset of items whose full label appears in the
    message as a case-insensitive substring.

    Full label only. No prefix rule (see module docstring). Empty /
    whitespace-only labels are skipped so a malformed surface entry
    cannot silently match a message with any content.
    """
    lower = message.lower()
    hits: list[SurfaceItem] = []
    for item in surface_items:
        label = (item.label or "").strip()
        if not label:
            continue
        if label.lower() in lower:
            hits.append(item)
    return tuple(hits)


def resolve_reference(
    message: str,
    surface_items: tuple[SurfaceItem, ...],
) -> ResolveOutcome:
    """Deterministic reference resolution.

    See module docstring for the locked rule set + rationale for what
    is deliberately NOT implemented.

    Never raises. Non-string message -> no_reference (defensive).
    """
    if not isinstance(message, str):
        return ResolveOutcome(
            status="no_reference", item=None, reason="non_string_message",
        )
    if not surface_items:
        return ResolveOutcome(
            status="no_reference", item=None, reason="no_surface",
        )
    msg = message.strip()
    if not msg:
        return ResolveOutcome(
            status="no_reference", item=None, reason="empty_message",
        )

    has_pronoun = _find_pronoun(msg)
    ordinal = _find_ordinal(msg)

    # Rule: ordinal in range wins. Ordinals are the most explicit signal
    # a user can give against a numbered surface, so they take precedence
    # over pronoun and name matching.
    if ordinal is not None:
        if 1 <= ordinal <= len(surface_items):
            return ResolveOutcome(
                status="resolved",
                item=surface_items[ordinal - 1],
                reason="ordinal",
            )
        # Out-of-range ordinal is NOT a clarification -- the user
        # named an item that doesn't exist. Treat as no_reference so
        # the caller can decide (fall through, or ask "we only have
        # N options"). This module doesn't invent that response.
        return ResolveOutcome(
            status="no_reference",
            item=None,
            reason="ordinal_out_of_range",
        )

    # Rule: full-label substring match. Unique -> resolved; ambiguous
    # (multiple items in message) -> clarification. Runs BEFORE the
    # multi-item-pronoun check so "match me to Administrative
    # assistant" resolves even if the message also contains "that".
    label_hits = _find_label_matches(msg, surface_items)
    if len(label_hits) == 1:
        return ResolveOutcome(
            status="resolved",
            item=label_hits[0],
            reason="label_match_unique",
        )
    if len(label_hits) >= 2:
        return ResolveOutcome(
            status="clarification",
            item=None,
            reason="label_match_ambiguous",
        )

    # Rule: single-item + pronoun -> resolved.
    if has_pronoun and len(surface_items) == 1:
        return ResolveOutcome(
            status="resolved",
            item=surface_items[0],
            reason="single_item_pronoun",
        )

    # Rule: multi-item + pronoun-only (no ordinal, no name match) ->
    # clarification. Pronoun alone against multiple items is ambiguous.
    if has_pronoun and len(surface_items) >= 2:
        return ResolveOutcome(
            status="clarification",
            item=None,
            reason="multi_item_pronoun",
        )

    return ResolveOutcome(
        status="no_reference", item=None, reason="no_signal",
    )


# ---------------------------------------------------------------- clarification


# Em dash character (U+2014). Locked with lead 2026-07-03 as the
# delimiter between the lead-in and the item list.
_EM_DASH = "—"

# Prompt lead-in. Kept short; the coach voice elaborates upstream if
# it wants. Downstream copy tests grep on this constant so a rename
# is a deliberate act.
_CLARIFICATION_LEAD = "Which one do you mean"


def build_clarification_prompt(
    items: tuple[SurfaceItem, ...],
) -> str:
    """Render a natural-language clarification prompt when the
    reference resolver returned status="clarification".

    Locked rendering (Step 2.2, 2026-07-03):
      - Em dash (U+2014) separator between the lead-in and the list.
      - Oxford comma before the final "or" for lists of 3+.
      - 2-item list uses plain "A or B" (no serial comma).
      - Blank labels are filtered defensively (a malformed surface
        entry should not produce " , or Foo").
      - Fewer than 2 usable labels after filtering returns an empty
        string. The resolver never emits "clarification" with fewer
        than 2 items, so this is a defensive short-circuit -- caller
        can treat empty as "nothing to ask; fall through".

    The prompt does NOT include a question mark inside the option
    list; the trailing "?" applies to the whole sentence.
    """
    labels: list[str] = []
    for it in items:
        label = (it.label or "").strip()
        if label:
            labels.append(label)

    if len(labels) < 2:
        return ""

    if len(labels) == 2:
        listing = f"{labels[0]} or {labels[1]}"
    else:
        # 3+ items: Oxford comma before the terminal "or".
        listing = ", ".join(labels[:-1]) + f", or {labels[-1]}"

    return f"{_CLARIFICATION_LEAD} {_EM_DASH} {listing}?"


# ---------------------------------------------------------------- LLM fallback (Step 2.3)


# Locked with lead 2026-07-03:
#   - reuse existing LLM_MODEL + LLM_FALLBACK_MODEL from config (no new
#     model knob for this task; keeps the resolver aligned with the
#     classifier stack)
#   - full-identity cache key including kind so a job posting and a
#     role/NOC with the same label never share a cached resolution
#   - message normalized to strip().lower() in the cache key only;
#     labels/ids pass through untouched so malformed surfaces stay
#     visible rather than being silently normalized
#   - enum-constrained tool_use, item_1..item_N + clarification +
#     no_match; NO free-text target invention
_TOOL_NAME = "select_referenced_item"


# Fully sorted for cache-friendliness of the system prompt (tools
# render before system per Anthropic's cache prefix ordering).
_LLM_SYSTEM_PROMPT = """You are a reference resolver for SkillBridge SSM. The user has just been shown a numbered list of items (job postings or occupational roles) by the coach. Your job is to decide which item the user's next message references.

You MUST call the `select_referenced_item` tool. Output NO prose. Output NO other tool call. The tool call IS your response.

SELECTION VALUES:

- `item_N`: the user's message clearly refers to item N (1-indexed). Use this for near-misses ("admin secretary" -> "Administrative assistant"), common abbreviations ("AP clerk" -> "Accounts payable clerk"), and paraphrases that unambiguously identify ONE item on the list.

- `clarification`: the user's message references SOMETHING on the list but is ambiguous between two or more items. Use this when a token could match multiple items (e.g. user says "admin" and the list has "Administrative assistant" AND "Administrative clerk").

- `no_match`: the user's message does not reference any item on the list. Includes: unrelated topics, general questions, requests for something outside the list.

RULES:

1. The deterministic pre-filter has already tried ordinals (first / 1st / #2) and exact substring name match. If either would have worked, this call would not fire. So you are handling near-misses, abbreviations, and paraphrases only.

2. Do NOT invent new target names. If the user mentions a role or job that is NOT on the list, that is `no_match`, not `clarification`. Never guess an item just because the user seems to want one.

3. Case-insensitive matching is fine. Common abbreviations ("admin" for administrative, "AP" for accounts payable) are legitimate reference signals.

4. Ambiguity between two or more items on the list is `clarification`, never a guess. Better to ask than to pick wrong.

5. `no_match` is the safe default when uncertain. The caller falls through to normal routing on `no_match`; a wrong item selection would silently pivot the user to a target they did not choose.
"""


# Cache: process-wide, keyed on
#   (normalized message, ((kind, label, id), ...) for each surface item)
# See module locked-design note for rationale.
_LLM_CACHE: dict[
    tuple[str, tuple[tuple[str, str, str | None], ...]],
    ResolveOutcome,
] = {}


_client: anthropic.Anthropic | None = None


def _client_get() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY.startswith("PLACEHOLDER"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY missing or placeholder. "
                "Set LLM_ENABLED=false to disable LLM reference resolution."
            )
        _client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY,
            timeout=LLM_TIMEOUT_SECONDS,
        )
    return _client


def _cache_key(
    message: str,
    surface_items: tuple[SurfaceItem, ...],
) -> tuple[str, tuple[tuple[str, str, str | None], ...]]:
    """Locked cache-key shape: normalized message + full-identity item
    tuples in surface order."""
    norm_msg = message.strip().lower()
    surface_tuple = tuple(
        (it.kind, it.label, it.id) for it in surface_items
    )
    return (norm_msg, surface_tuple)


def _build_tool_schema(n: int) -> dict:
    """Build the Anthropic tool_use schema with a dynamic enum sized to
    the current surface. N is the number of items in the surface (1..N).

    Enum values: item_1, item_2, ..., item_N, clarification, no_match.
    """
    item_values = [f"item_{i}" for i in range(1, n + 1)]
    enum_values = item_values + ["clarification", "no_match"]
    return {
        "name": _TOOL_NAME,
        "description": (
            "Select which item on the surfaced list the user's message "
            "references, or return clarification / no_match."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "selection": {
                    "type": "string",
                    "enum": enum_values,
                    "description": (
                        "One of item_1..item_N naming a specific item, "
                        "or clarification if ambiguous, or no_match "
                        "if the message doesn't reference any item."
                    ),
                },
            },
            "required": ["selection"],
            "additionalProperties": False,
        },
    }


def _build_user_block(
    message: str,
    surface_items: tuple[SurfaceItem, ...],
) -> str:
    """Serialize the user message + numbered surface list for the LLM."""
    lines = [f"USER_MESSAGE: {message}", "", "OPTIONS:"]
    for i, item in enumerate(surface_items, start=1):
        label = item.label or "(no label)"
        # Include kind + id so the model can distinguish role-vs-job
        # and NOC-vs-job-id without us pre-formatting.
        id_hint = f" [{item.id}]" if item.id else ""
        lines.append(f"{i}. {label}{id_hint}  (kind: {item.kind})")
    return "\n".join(lines)


def _interpret_llm_selection(
    raw: str,
    surface_items: tuple[SurfaceItem, ...],
) -> ResolveOutcome:
    """Map the tool_use enum value back to a ResolveOutcome.

    Defensive: unknown values or out-of-range item_N coerce to
    no_reference with a distinct `reason` for telemetry.
    """
    if raw == "clarification":
        return ResolveOutcome(
            status="clarification", item=None, reason="llm_clarification",
        )
    if raw == "no_match":
        return ResolveOutcome(
            status="no_reference", item=None, reason="llm_no_match",
        )
    match = re.fullmatch(r"item_(\d+)", raw)
    if match is not None:
        try:
            n = int(match.group(1))
        except ValueError:
            return ResolveOutcome(
                status="no_reference", item=None, reason="llm_invalid",
            )
        if 1 <= n <= len(surface_items):
            return ResolveOutcome(
                status="resolved",
                item=surface_items[n - 1],
                reason="llm_selected",
            )
        return ResolveOutcome(
            status="no_reference", item=None, reason="llm_out_of_range",
        )
    return ResolveOutcome(
        status="no_reference", item=None, reason="llm_invalid",
    )


def _call_reference_llm(
    message: str,
    surface_items: tuple[SurfaceItem, ...],
) -> str:
    """Issue the tool_use call. Returns the raw selection string
    (item_N | clarification | no_match) or raises. Caller is
    responsible for enum-value validation.

    Mirrors classify_career_intent's shape: same system-prompt cache
    control, same 429/529/503 fallback-model retry, temperature=0 for
    stability.
    """
    client = _client_get()
    user_block = _build_user_block(message, surface_items)
    system_blocks = [{
        "type": "text",
        "text": _LLM_SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }]
    tool_schema = _build_tool_schema(len(surface_items))

    def _do_call(model: str):
        return client.messages.create(
            model=model,
            max_tokens=200,
            temperature=0,
            system=system_blocks,
            tools=[tool_schema],
            tool_choice={"type": "tool", "name": _TOOL_NAME},
            messages=[{"role": "user", "content": user_block}],
        )

    try:
        resp = _do_call(LLM_MODEL)
    except anthropic.APIStatusError as e:
        if e.status_code in (429, 529, 503) and LLM_MODEL != LLM_FALLBACK_MODEL:
            log.warning(
                "reference_resolver LLM overloaded on %s; falling back to %s",
                LLM_MODEL, LLM_FALLBACK_MODEL,
            )
            resp = _do_call(LLM_FALLBACK_MODEL)
        else:
            raise

    for block in resp.content:
        if getattr(block, "type", None) != "tool_use":
            continue
        if getattr(block, "name", None) != _TOOL_NAME:
            continue
        tool_input = getattr(block, "input", None)
        if not isinstance(tool_input, dict):
            continue
        selection = tool_input.get("selection")
        if isinstance(selection, str):
            return selection
    raise RuntimeError("LLM returned no usable tool_use block")


def resolve_reference_via_llm(
    *,
    message: str,
    surface_items: tuple[SurfaceItem, ...],
) -> ResolveOutcome:
    """LLM fallback resolver (Step 2.3 of slice 2).

    Called by the composed resolver (Step 2.4) after `resolve_reference`
    returns `no_reference` with a non-empty surface. Handles near-misses
    ("admin secretary" vs "Administrative assistant"), abbreviations,
    and paraphrases via Anthropic tool_use with an enum-constrained
    output. The enum guarantees the LLM cannot invent a free-text
    target name.

    Defensive short-circuits (all return no_reference with a distinct
    `reason` for telemetry):
      - non-string / empty / whitespace-only message
      - empty surface (nothing to resolve to)
      - LLM_ENABLED=false at config level
      - API call fails (network, auth, retryable overload after
        fallback also fails)
      - tool returns an invalid / unrecognized selection
      - tool returns item_N where N is out of range

    Cache: process-wide, keyed on (normalized message, item-identity
    tuple). Failures are cached too, matching the existing intent
    classifier's pattern -- prevents retry storms and keeps behavior
    deterministic across the same input tuple.
    """
    if not isinstance(message, str):
        return ResolveOutcome(
            status="no_reference", item=None, reason="non_string_message",
        )
    if not message.strip():
        return ResolveOutcome(
            status="no_reference", item=None, reason="empty_message",
        )
    if not surface_items:
        return ResolveOutcome(
            status="no_reference", item=None, reason="no_surface",
        )

    key = _cache_key(message, surface_items)
    cached = _LLM_CACHE.get(key)
    if cached is not None:
        return cached

    if not LLM_ENABLED:
        result = ResolveOutcome(
            status="no_reference", item=None, reason="llm_disabled",
        )
        _LLM_CACHE[key] = result
        return result

    try:
        raw = _call_reference_llm(message.strip(), surface_items)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "reference_resolver llm_call_failed (%s); returning no_reference",
            type(exc).__name__,
        )
        result = ResolveOutcome(
            status="no_reference", item=None, reason="llm_error",
        )
        _LLM_CACHE[key] = result
        return result

    result = _interpret_llm_selection(raw, surface_items)
    _LLM_CACHE[key] = result
    return result


def reset_llm_cache() -> None:
    """Test helper: drop the in-memory LLM cache."""
    _LLM_CACHE.clear()


def llm_cache_size() -> int:
    """Test helper: current cache entry count."""
    return len(_LLM_CACHE)
