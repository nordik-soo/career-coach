"""Derived read-only view of session state (locked 2026-07-03).

Slice 1 Step 1.1 of the memory/routing refactor: a pure function over
StagedProfile that returns a normalized ConversationFrame the router
and responder can read without touching individual scattered fields.

Locked design contract:
  - Pure function. No writes to staged. No engine calls. No I/O.
  - No new persisted fields on StagedProfile in this step. The frame
    is derived on every read.
  - Pending precedence (specific > general):
      credential_confirmation
      recommender_offer
      adjacent_search_offer
      adjacent_offer
  - Surface items are normalized to the closed shape
    SurfaceItem(kind, label, id, ordinal). Raw snapshot payloads never
    leak through.

Step 1.2 will add per-surface message_count anchors so
`latest_surface_at_turn` becomes deterministic. Until then this module
uses a static precedence stopgap; the getattr-with-default reads on
the anchor fields make Step 1.2 a drop-in upgrade with no rewrite here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from skillbridge.session.staging import StagedProfile


SurfaceKind = Literal["job", "role"]
SurfaceType = Literal["matches", "adjacent_recs", "none"]
EngineName = Literal["matching", "recommender", "none"]


@dataclass(frozen=True, slots=True)
class SurfaceItem:
    """One item from the most recently rendered surface.

    `kind`: what class of thing this is -- a job posting or a role/NOC.
    `label`: user-facing title (e.g. "Administrative assistant").
    `id`: stable identifier when the source has one; None for surfaces
          that store only titles (last_presented_job_titles).
    `ordinal`: 1-based position in the surface, used by the future
               reference resolver for "the second one".
    """

    kind: SurfaceKind
    label: str
    id: str | None
    ordinal: int


@dataclass(frozen=True, slots=True)
class ConversationFrame:
    """Derived read-only view of the current conversation state.

    All fields computed from StagedProfile at derive_frame() call time.
    Never persisted. Never mutated. If a downstream reader wants a
    different shape, add a normalization here rather than reading raw
    staged fields directly -- the point of the frame is to keep the
    scattered state discoverable through one surface.
    """

    active_target_role: str | None
    active_target_noc: str | None
    active_pending_offer: str | None
    latest_surface_type: SurfaceType
    latest_surface_at_turn: int | None
    latest_surface_items: tuple[SurfaceItem, ...]
    last_engine_used: EngineName
    available_referents: tuple[str, ...]


def _derive_active_pending(staged: StagedProfile) -> str | None:
    """Locked precedence: specific pending action beats general one.

    Returns a stable string label (not the raw payload). Callers that
    need the underlying value read staged directly.
    """
    if staged.pending_credential_confirmation is not None:
        return "credential_confirmation"
    if staged.pending_recommender_offer is not None:
        # Carry the mode so the router/responder can distinguish
        # between local_gap_coach / target_noc_standard /
        # adjacent_noc_standard / adjacent_role_drilldown_select
        # without re-reading staged.
        return f"recommender:{staged.pending_recommender_offer}"
    if staged.pending_adjacent_search_offer:
        return "adjacent_search"
    if staged.pending_adjacent_offer:
        return "adjacent_offer"
    return None


def _rec_surface_items(staged: StagedProfile) -> tuple[SurfaceItem, ...]:
    """Normalize last_recommender_adjacent_surface (NOC/role list)."""
    out: list[SurfaceItem] = []
    for i, d in enumerate(staged.last_recommender_adjacent_surface or ()):
        if not isinstance(d, dict):
            continue
        title = d.get("title")
        code = d.get("noc_code")
        if not isinstance(title, str) or not title.strip():
            continue
        out.append(
            SurfaceItem(
                kind="role",
                label=title.strip(),
                id=code.strip() if isinstance(code, str) and code.strip() else None,
                ordinal=i + 1,
            )
        )
    return tuple(out)


def _adj_snapshot_items(
    staged: StagedProfile,
) -> tuple[tuple[SurfaceItem, ...], int | None]:
    """Normalize last_adjacent_snapshot (adjacent job postings).

    Returns (items, created_message_count). Anchor comes from the
    snapshot itself, so no companion field is needed on staged.
    """
    snap = staged.last_adjacent_snapshot
    if not isinstance(snap, dict):
        return ((), None)
    raw_items = snap.get("items")
    created = snap.get("created_message_count")
    at_turn = created if isinstance(created, int) else None
    if not isinstance(raw_items, list):
        return ((), at_turn)
    out: list[SurfaceItem] = []
    for i, d in enumerate(raw_items):
        if not isinstance(d, dict):
            continue
        title = d.get("title")
        job_id = d.get("job_id")
        if not isinstance(title, str) or not title.strip():
            continue
        out.append(
            SurfaceItem(
                kind="job",
                label=title.strip(),
                id=job_id.strip() if isinstance(job_id, str) and job_id.strip() else None,
                ordinal=i + 1,
            )
        )
    return (tuple(out), at_turn)


def _matches_items(staged: StagedProfile) -> tuple[SurfaceItem, ...]:
    """Normalize last_presented_job_titles (matching engine's rendered
    top-N titles). No id available in that field -- id is None."""
    out: list[SurfaceItem] = []
    for i, title in enumerate(staged.last_presented_job_titles or ()):
        if not isinstance(title, str) or not title.strip():
            continue
        out.append(
            SurfaceItem(
                kind="job",
                label=title.strip(),
                id=None,
                ordinal=i + 1,
            )
        )
    return tuple(out)


# Static precedence used ONLY as a slice-1 stopgap when message_count
# anchors are absent. Step 1.2 replaces this with deterministic
# ordering by max(*_at_turn). Higher number = wins.
_STATIC_PRECEDENCE_ADJACENT_RECS = 3
_STATIC_PRECEDENCE_ADJACENT_SNAPSHOT = 2
_STATIC_PRECEDENCE_MATCHES = 1


def _pick_latest_surface(
    staged: StagedProfile,
) -> tuple[SurfaceType, int | None, tuple[SurfaceItem, ...]]:
    """Choose the most recently rendered non-empty surface.

    Ordering rule (locked):
      1. Prefer the surface with the highest `*_at_turn` anchor.
      2. When two surfaces tie or both anchors are None, fall back to
         the static precedence recommender > adjacent_snapshot > matches.

    Step 1.2 will populate the anchor fields via getattr paths below.
    Until then, all anchors resolve to None and the static tiebreaker
    picks the winner. The frame's public shape does not change when
    Step 1.2 lands.
    """
    rec_at_turn = getattr(
        staged, "last_recommender_adjacent_surface_at_turn", None
    )
    match_at_turn = getattr(staged, "last_presented_at_turn", None)

    rec_items = _rec_surface_items(staged)
    adj_items, adj_at_turn = _adj_snapshot_items(staged)
    match_items = _matches_items(staged)

    candidates: list[
        tuple[int | None, int, SurfaceType, int | None, tuple[SurfaceItem, ...]]
    ] = []
    if rec_items:
        candidates.append(
            (
                rec_at_turn,
                _STATIC_PRECEDENCE_ADJACENT_RECS,
                "adjacent_recs",
                rec_at_turn,
                rec_items,
            )
        )
    if adj_items:
        candidates.append(
            (
                adj_at_turn,
                _STATIC_PRECEDENCE_ADJACENT_SNAPSHOT,
                "matches",
                adj_at_turn,
                adj_items,
            )
        )
    if match_items:
        candidates.append(
            (
                match_at_turn,
                _STATIC_PRECEDENCE_MATCHES,
                "matches",
                match_at_turn,
                match_items,
            )
        )

    if not candidates:
        return ("none", None, ())

    def _sort_key(c):
        anchor, prio, _stype, _at_turn, _items = c
        # None anchors sort below any int anchor. Ties broken by static
        # precedence.
        return (anchor if anchor is not None else -1, prio)

    candidates.sort(key=_sort_key, reverse=True)
    _anchor, _prio, stype, at_turn, items = candidates[0]
    return (stype, at_turn, items)


def _derive_last_engine(staged: StagedProfile) -> EngineName:
    """Best-effort inference of which engine last produced output.

    Slice 1 stopgap rules:
      - matching signals: last_match_snapshot OR last_presented_job_titles
        OR last_adjacent_snapshot. The adjacent snapshot is the matching
        engine's sideways_move tier output -- an adjacent-JOB surface,
        not a recommender surface -- and must count as a matching signal
        to keep last_engine_used consistent with the surface it produces.
      - recommender signals: last_recommender_adjacent_surface (Layer C's
        adjacent-NOC/role list).
    When both present, recommender wins (it is the newer engine
    architecturally; Step 1.2 anchors replace this with deterministic
    ordering).
    """
    has_matching_signal = (
        staged.last_match_snapshot is not None
        or bool(staged.last_presented_job_titles)
        or staged.last_adjacent_snapshot is not None
    )
    has_recommender_signal = bool(staged.last_recommender_adjacent_surface)
    if has_recommender_signal:
        return "recommender"
    if has_matching_signal:
        return "matching"
    return "none"


def derive_frame(staged: StagedProfile) -> ConversationFrame:
    """Compute the read-only conversation frame from staged.

    Pure function. Never writes to staged. Returns a frozen frame that
    the router / responder / telemetry can read without knowing which
    underlying StagedProfile field carries which piece of state.
    """
    stype, at_turn, items = _pick_latest_surface(staged)
    referents = tuple(item.label for item in items if item.label)
    return ConversationFrame(
        active_target_role=(staged.target_role_text or None),
        active_target_noc=(staged.target_noc or None),
        active_pending_offer=_derive_active_pending(staged),
        latest_surface_type=stype,
        latest_surface_at_turn=at_turn,
        latest_surface_items=items,
        last_engine_used=_derive_last_engine(staged),
        available_referents=referents,
    )
