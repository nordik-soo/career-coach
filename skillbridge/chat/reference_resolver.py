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

import re
from dataclasses import dataclass
from typing import Literal

from skillbridge.chat.conversation_frame import SurfaceItem


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
