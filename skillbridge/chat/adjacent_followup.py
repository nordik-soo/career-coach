"""AR-5 ordinal follow-up + describe_adjacent_role render.

Two related functions:

  - `resolve_adjacent_followup(message, snap, current_message_count)`:
    Pure resolver. Returns the snapshot item the user referenced
    ("the second one" / "#3" / "the welder role") -- or None on:
      * TTL-dead snapshot (lives only on the immediately-following
        user turn, `current_message_count == snap.created + 1`);
      * out-of-range ordinal / numeric;
      * ambiguous title-suffix (two items share the distinctive
        token);
      * no match.

  - `render_describe_adjacent_role(snapshot_item)`: live-fetches the
    job by id from `core.v_current_job` and combines with the
    snapshot's `evidence_summary` + `matched_skills`. Does NOT
    re-score. When the posting has expired (no longer active),
    returns `expired=True` so the responder can fall through to a
    deterministic "that role's no longer on the board" line.

This module is DEAD CODE until AR-6 wires both call sites. The
AR-1c activation-safety audit confirms no production caller exists
yet.
"""
from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------- normalize
_SMART_APOSTROPHES = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
})


def _normalize(message: str | None) -> str:
    """Lower-case, fold smart apostrophes, collapse whitespace, strip
    surrounding punctuation. Symmetric with chat/adjacent_intent._normalize."""
    if not isinstance(message, str) or not message:
        return ""
    s = message.translate(_SMART_APOSTROPHES).lower().strip()
    s = re.sub(r"^[\s.,!?;:'\"]+", "", s)
    s = re.sub(r"[\s.,!?;:'\"]+$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


# ---------------------------------------------------------------- ordinal
# (pattern, zero-based index). Word-boundary matching so "first" is
# recognized but "firstly something else" isn't ambiguous.
_ORDINAL_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\b(?:the\s+)?(?:first|1st)(?:\s+one)?\b"), 0),
    (re.compile(r"\b(?:the\s+)?(?:second|2nd)(?:\s+one)?\b"), 1),
    (re.compile(r"\b(?:the\s+)?(?:third|3rd)(?:\s+one)?\b"), 2),
)


# Sentinel for "format matched, but the user gave multiple
# contradictory references in the same format". Distinct from None
# (which means "format absent") so the resolver can short-circuit to
# clarification instead of letting another path's lone match win.
_AMBIGUOUS = object()


def _match_ordinal(norm: str):
    """Three-state result:
        None         -- no ordinal pattern matched
        int          -- exactly one ordinal pattern matched
        _AMBIGUOUS   -- more than one ordinal matched ("first and
                        second"); the user contradicted themselves
                        within the format.
    """
    matched: list[int] = []
    for pat, idx in _ORDINAL_PATTERNS:
        if pat.search(norm):
            matched.append(idx)
    if not matched:
        return None
    if len(matched) > 1:
        return _AMBIGUOUS
    return matched[0]


# ---------------------------------------------------------------- numeric
# Accepts: "#1", "# 1", "number 1", "item 1", "no. 1", "no 1".
# `-?` captures the minus sign so "welder #-1" / "number -1" surface
# as an explicit invalid reference (downstream `_match_numeric` maps
# any value < 1 to `_AMBIGUOUS`); otherwise the title path could
# silently pick the welder.
_NUMERIC_PATTERN = re.compile(
    r"(?:#\s*|\bnumber\s+|\bitem\s+|\bno\.?\s+)(-?\d+)"
)


def _match_numeric(norm: str):
    """Three-state result mirrors `_match_ordinal`:
        None         -- no numeric pattern matched
        int          -- exactly one numeric reference; zero-based
                        (a 1-indexed "#1" returns 0)
        _AMBIGUOUS   -- multiple distinct numeric references
                        ("compare #1 and #2"); user is contradicting
                        themselves in-format.
    """
    matches = _NUMERIC_PATTERN.findall(norm)
    if not matches:
        return None
    # Multiple matches that all point to the same number aren't
    # ambiguous -- the user might have repeated "#2" -- but distinct
    # values are. Dedup before counting.
    try:
        distinct = {int(m) for m in matches}
    except (TypeError, ValueError):
        return None
    if len(distinct) > 1:
        return _AMBIGUOUS
    n = next(iter(distinct))
    if n < 1:
        # "#0" / "#-1" are invalid -- numeric references are
        # 1-indexed. Return _AMBIGUOUS (rather than None for "absent")
        # so the title-suffix path can't silently override what is in
        # fact an explicit-but-invalid numeric reference.
        return _AMBIGUOUS
    return n - 1


# ---------------------------------------------------------------- title-suffix
# Common stopwords stripped from title tokens before matching. The
# title-suffix path is "did the user mention a distinctive token from
# exactly one item's title?". Generic role-words ("role", "job",
# "position") are stripped so they don't make every item look like a
# match.
_TITLE_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "of", "and", "or",
    "i", "ii", "iii", "iv", "v",
    "role", "roles", "job", "jobs", "position", "positions",
    "opening", "openings", "posting", "postings",
    "one", "tell", "more", "about", "show", "me", "with",
    "operator", "specialist", "associate", "worker",
})


def _significant_title_tokens(title: str) -> set[str]:
    """Lowercased tokens from a title with stopwords removed. Used by
    the title-suffix path so "the welder role" matches the "Welder"
    item without "role" also matching every other entry."""
    if not isinstance(title, str):
        return set()
    raw = re.sub(r"[^a-z0-9]+", " ", title.lower()).split()
    return {t for t in raw if t and t not in _TITLE_STOPWORDS}


def _match_by_title(norm: str, items: list[dict]) -> list[dict]:
    """Items whose title shares at least one significant token with
    the message. Multiple matches -> ambiguous (the caller returns
    None)."""
    norm_tokens = set(re.sub(r"[^a-z0-9]+", " ", norm).split())
    if not norm_tokens:
        return []
    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        sig = _significant_title_tokens(title) if isinstance(title, str) else set()
        if sig & norm_tokens:
            out.append(item)
    return out


# ---------------------------------------------------------------- resolver
def resolve_adjacent_followup(
    message: str | None,
    snap: dict | None,
    current_message_count: int,
) -> dict | None:
    """Return the snapshot item the user referenced, or None.

    Cascade (first deterministic answer wins):
      1. Ordinal pattern → zero-based index, bounded by len(items).
      2. Numeric pattern → 1-based number from message, zero-based
         output index.
      3. Title-suffix pattern → distinctive title-token match (must
         be EXACTLY one item).

    TTL: live iff `current_message_count == snap.created_message_count + 1`.
    Anything else -> None. The AR-1a snapshot-shift helper handles the
    scope-violation digression case in remaining_gaps.py STEP 7
    (created_message_count is advanced forward by 1 on scope-violation
    turns) so a single redirect_scope between the recommendation turn
    and the follow-up doesn't burn the resolver.
    """
    # TTL. `bool` subclasses `int`, so a forged
    # `created_message_count=True` with current=2 would coincidentally
    # satisfy True + 1 == 2 and resolve item 1. Reject booleans.
    # Negative counts are also rejected -- they mirror the
    # `_sanitize_adjacent_snapshot` contract (`created < 0` is
    # malformed) and avoid the `created=-1, current=0` coincidence.
    if not isinstance(snap, dict):
        return None
    created = snap.get("created_message_count")
    if (
        not isinstance(created, int)
        or isinstance(created, bool)
        or created < 0
    ):
        return None
    if (
        not isinstance(current_message_count, int)
        or isinstance(current_message_count, bool)
        or current_message_count < 0
    ):
        return None
    if current_message_count != created + 1:
        return None

    items = snap.get("items")
    if not isinstance(items, list) or not items:
        return None

    norm = _normalize(message)
    if not norm:
        return None

    # Cross-format conflict detection (AR-5 review round 2/3):
    # Each detection path produces ONE OF:
    #   - None        : format absent (doesn't constrain)
    #   - int         : single candidate (path contributes {idx})
    #   - _AMBIGUOUS  : path detected MULTIPLE contradictory in-format
    #                   references; resolver short-circuits to None
    #                   so the planner can ask for clarification.
    # Title-suffix contributes a candidate SET (may be >1); another
    # path can resolve the ambiguity by intersection if its set
    # narrows to a single shared item.
    candidate_sets: list[set[int]] = []

    # Ordinal.
    ord_result = _match_ordinal(norm)
    if ord_result is _AMBIGUOUS:
        return None
    if isinstance(ord_result, int):
        if not (0 <= ord_result < len(items)):
            return None
        candidate_sets.append({ord_result})

    # Numeric.
    num_result = _match_numeric(norm)
    if num_result is _AMBIGUOUS:
        return None
    if isinstance(num_result, int):
        if not (0 <= num_result < len(items)):
            return None
        candidate_sets.append({num_result})

    # Title-suffix. Multiple matches contribute the full ambiguous
    # set; the intersection with ordinal/numeric (when present) can
    # narrow it.
    title_matches = _match_by_title(norm, items)
    if title_matches:
        title_indices: set[int] = set()
        for i, item in enumerate(items):
            if any(item is t for t in title_matches):
                title_indices.add(i)
        if title_indices:
            candidate_sets.append(title_indices)

    if not candidate_sets:
        return None   # no references detected at all

    final = set.intersection(*candidate_sets)
    if len(final) != 1:
        return None   # zero (conflicting) or many (ambiguous) candidates

    idx = next(iter(final))
    it = items[idx]
    return it if isinstance(it, dict) else None


# ---------------------------------------------------------------- render
def render_describe_adjacent_role(snapshot_item: dict | None) -> dict:
    """Build the responder payload for a `describe_adjacent_role`
    outcome.

    Re-fetches the job by id from `core.v_current_job` (LIVE state,
    not snapshot) and combines with the snapshot's `evidence_summary`
    + `matched_skills`. NO re-scoring -- the resolver already picked
    the item, and the snapshot's evidence is what we narrate.

    Returns a dict matching the v5 design's
    `adjacent_role_description_payload` shape:
        {
            "job": {employer, location, url, posted_date, title} | None,
            "evidence_summary": str,
            "matched_skills": list[str],
            "expired": bool,
        }

    On `expired=True` the responder narration falls through to a
    deterministic "that role's no longer on the board" line.
    """
    payload: dict[str, Any] = {
        "job": None,
        "evidence_summary": "",
        "matched_skills": [],
        "expired": True,
    }

    if not isinstance(snapshot_item, dict):
        return payload

    evidence = snapshot_item.get("evidence_summary")
    if isinstance(evidence, str):
        payload["evidence_summary"] = evidence

    raw_matched = snapshot_item.get("matched_skills")
    if isinstance(raw_matched, list):
        payload["matched_skills"] = [
            s for s in raw_matched if isinstance(s, str) and s
        ]

    job_id = snapshot_item.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        return payload   # expired=True; no id to fetch

    job_row = _fetch_active_job_by_id(job_id)
    if job_row is None:
        return payload   # expired=True

    # Surface only the responder-relevant fields. The full job row
    # carries other columns we don't narrate, and serializing them
    # unfiltered would be a future-compat hazard.
    payload["job"] = {
        "title":       job_row.get("title"),
        "employer":    job_row.get("employer"),
        "location":    job_row.get("location"),
        "url":         job_row.get("url"),
        "posted_date": job_row.get("posted_date"),
    }
    payload["expired"] = False
    return payload


def _fetch_active_job_by_id(job_id: str) -> dict | None:
    """One small SQL fetch against `core.v_current_job`. Returns the
    row (Mapping) or None when the posting is no longer active.

    Lives here -- NOT in `match/adjacent.py` -- because the describe
    render is a chat-layer concern (responder payload), and the
    adjacency module already carries the full engine pipeline.
    """
    from skillbridge.db import sync_cursor

    with sync_cursor() as cur:
        cur.execute(
            "SELECT * FROM core.v_current_job WHERE job_id = %s",
            (job_id,),
        )
        row = cur.fetchone()
    return dict(row) if row is not None else None
