"""Test fixtures for AR-9.bug.2a sub-step 4 consumer migration.

Provides:
  - empty_view(): a SanitizedResponderView with all slots empty
  - view_with_prompt_urls(): synthetic view with a specific allowlist
  - _extract_training_json_objects(): robust TRAINING block parser
  - _extract_json_block(): single-block JSON parser keyed by prefix

These are test infrastructure, not production code. Tests that need a
realistic view from an inp should call build_sanitized_responder_view_*
directly (production builder = test source of truth). These helpers
exist for adversarial / focused tests that need a specific shape.
"""
from __future__ import annotations

import dataclasses
import json
from types import MappingProxyType
from typing import Any

from skillbridge.chat.url_views import SanitizedResponderView


def empty_view() -> SanitizedResponderView:
    """All slots empty. URL allowlists are empty frozensets. No
    rejected source URLs. Useful for tests that don't care about
    URL content (e.g. policy gate regex tests).
    """
    return SanitizedResponderView(
        prompt_results=(),
        fallback_results=(),
        prompt_present_matches_training_flat=(),
        prompt_present_matches_training_groups=(),
        fallback_present_matches_training_by_job=MappingProxyType({}),
        prompt_explain_gap_training_flat=(),
        fallback_explain_gap_training_flat=(),
        prompt_present_near_miss_training_flat=(),
        prompt_explain_remaining_gaps_training_flat=(),
        prompt_adjacent_recommendations=None,
        fallback_adjacent_recommendations=(),
        prompt_adjacent_role=None,
        fallback_adjacent_role=None,
        rejected_source_urls=(),
        prompt_urls=frozenset(),
        fallback_urls=frozenset(),
    )


def view_with_prompt_urls(canonical_urls: set[str]) -> SanitizedResponderView:
    """An empty view whose prompt_urls allowlist contains exactly the
    given canonical URLs. For policy-gate membership tests once
    sub-step 5 wires the check.
    """
    return dataclasses.replace(empty_view(), prompt_urls=frozenset(canonical_urls))


def view_with_fallback_urls(canonical_urls: set[str]) -> SanitizedResponderView:
    """An empty view whose fallback_urls allowlist contains the given
    canonical URLs.
    """
    return dataclasses.replace(empty_view(), fallback_urls=frozenset(canonical_urls))


def _extract_training_json_objects(out: str) -> list[dict]:
    """Parse the TRAINING block of a V2 prompt user_block.

    Each line after `TRAINING:` is one `json.dumps(t)` object. Stop on
    the first line that isn't valid JSON — that's the end of the block,
    regardless of what the next section header looks like
    (`NEXT_SKILL: ...`, `RESUME_FACTS:`, etc., none of which are JSON).

    Empty lines inside the block are skipped (defensive).
    """
    lines = out.split("\n")
    try:
        start = lines.index("TRAINING:")
    except ValueError:
        return []
    parsed: list[dict] = []
    for line in lines[start + 1:]:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            parsed.append(obj)
    return parsed


def _extract_json_block(out: str, prefix: str) -> dict | None:
    """Extract a single JSON-object block tagged with a header prefix
    like `ADJACENT_RECOMMENDATIONS:` or `ADJACENT_ROLE_DESCRIPTION:`.

    Current responder.py emits these as:
        prefix
        {...json on subsequent lines...}

    `json.dumps` without indent produces single-line JSON, but
    json.dumps with `indent=` would emit multi-line. We accept either
    by joining lines until json.loads succeeds.

    Returns the parsed dict, or None if the prefix isn't found.
    """
    lines = out.split("\n")
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            # Try to parse from the line after the prefix; accumulate
            # until valid JSON or section ends.
            remainder = "\n".join(lines[i + 1:])
            # The next block starts at a line ending with `:` and
            # followed by JSON / text. To keep it simple, try
            # progressively shorter prefixes of `remainder` until
            # json.loads succeeds — current responder's json.dumps
            # produces single-line JSON, so the first newline is the
            # boundary.
            #
            # Most common case: prefix line, then ONE json line.
            candidate = lines[i + 1] if i + 1 < len(lines) else ""
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
            # Fall back: try joining lines until success.
            buf = ""
            for j in range(i + 1, len(lines)):
                buf = buf + ("\n" if buf else "") + lines[j]
                try:
                    obj = json.loads(buf)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    continue
            return None
    return None
