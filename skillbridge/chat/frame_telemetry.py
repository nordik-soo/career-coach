"""Per-turn frame telemetry emitter (Step 1.4 of memory/routing refactor).

Emits one structured log line per terminal response path inside the
recommender routing / consume layer. Not turn-level (that would need a
handler-wide refactor per Step 1.4's locked scope). Each terminal
captures its own pending-state snapshot at entry and derives the
current frame at exit -- the log line answers "what changed here?"
rather than "what changed across the turn."

Format matches the existing `anon_chat session=%s key=val ...` style
so existing telemetry-parsing tools pick up frame lines without a
new schema. Field values are always machine-safe: never-null string
tokens with a canonical `none` sentinel for unset optional fields.

Locked fields (do not change without co-decision):
  session, path, msg_count, pattern_intent, career_intent,
  router_action, router_reason, pending_before, pending_after,
  active_target_noc, latest_surface, latest_surface_at_turn,
  last_engine

Privacy: NEVER emit user_message text or target_role_text (both are
free-text and could carry PII). Only stable enums, NOC codes, and
machine state pass through this surface.

Slice 2 will add `resolution_outcome` when the reference resolver
lands. Adding it now would be dishonest telemetry -- there is no
resolver to describe.
"""
from __future__ import annotations

import logging
from typing import Any

from skillbridge.chat.conversation_frame import derive_frame
from skillbridge.session.staging import StagedProfile


log = logging.getLogger(__name__)


_NULL_TOKEN = "none"


def snapshot_pending_flags(staged: StagedProfile) -> tuple[str, ...]:
    """Capture the set of active pending-consent flags at a point in
    time.

    Returned as a stable-ordered tuple so equality checks in tests
    (and set-diffs in log analysis) are deterministic. Order follows
    the same precedence used by ConversationFrame._derive_active_pending
    (credential > recommender:<mode> > adjacent_search > adjacent_offer),
    but ALL live flags are returned -- not just the top-precedence
    one -- so telemetry captures the full picture at snapshot time.

    Callers MUST capture at the entry to the terminal function, before
    any state mutation, if they want `pending_before` to reflect the
    original state.
    """
    out: list[str] = []
    if staged.pending_credential_confirmation is not None:
        out.append("credential_confirmation")
    if staged.pending_recommender_offer is not None:
        out.append(f"recommender:{staged.pending_recommender_offer}")
    if staged.pending_adjacent_search_offer:
        out.append("adjacent_search")
    if staged.pending_adjacent_offer:
        out.append("adjacent_offer")
    return tuple(out)


def _fmt_flags(flags: tuple[str, ...]) -> str:
    """Render a flag tuple as a pipe-joined string, or `none` when
    empty. Pipe separator avoids collision with the key=val comma
    convention downstream log parsers expect."""
    if not flags:
        return _NULL_TOKEN
    return "|".join(flags)


def _fmt_optional(value: Any) -> str:
    """Render an optional scalar. None becomes `none`; everything else
    passes through str(). Guarantees a non-empty token so log-line
    tokenizers don't see key=<empty>."""
    if value is None:
        return _NULL_TOKEN
    return str(value)


def emit_frame_telemetry(
    *,
    staged: StagedProfile,
    path: str,
    pending_before: tuple[str, ...],
    pattern_intent: str | None = None,
    career_intent: str | None = None,
    router_action: str | None = None,
    router_reason: str | None = None,
) -> None:
    """Emit one structured log line for a terminal response path this
    slice touches.

    `path` is a short, machine-grepable label for the exit site (e.g.
    "route_recommender_layer", "consume_yes", "drilldown_no"). Callers
    pick a stable label per exit site; downstream log analysis groups
    on `path`.

    `pending_before` MUST be captured at the entry to the terminal
    function before any state mutation. This emitter reads
    `pending_after` from staged directly, so the diff between the two
    reflects what the enclosing terminal changed.

    Optional intent/router fields render as `none` when not available
    at this terminal -- the consume-hook paths don't run the router,
    so pattern_intent/career_intent/router_action/router_reason are
    None there and emit as `none`.

    Never raises: telemetry is best-effort observability, not a
    blocking side effect. If the frame derivation itself throws
    (shouldn't; it's pure and defensive), the exception is logged and
    swallowed so telemetry never crashes a chat turn.
    """
    try:
        frame = derive_frame(staged)
        pending_after = snapshot_pending_flags(staged)
        session_prefix = (
            staged.session_id[:8] if staged.session_id else _NULL_TOKEN
        )
        log.info(
            "frame_telemetry session=%s path=%s msg_count=%d "
            "pattern_intent=%s career_intent=%s router_action=%s "
            "router_reason=%s pending_before=%s pending_after=%s "
            "active_target_noc=%s latest_surface=%s "
            "latest_surface_at_turn=%s last_engine=%s",
            session_prefix,
            path,
            staged.message_count,
            _fmt_optional(pattern_intent),
            _fmt_optional(career_intent),
            _fmt_optional(router_action),
            _fmt_optional(router_reason),
            _fmt_flags(pending_before),
            _fmt_flags(pending_after),
            _fmt_optional(frame.active_target_noc),
            frame.latest_surface_type,
            _fmt_optional(frame.latest_surface_at_turn),
            frame.last_engine_used,
        )
    except Exception:  # noqa: BLE001
        log.exception(
            "frame_telemetry emit_failed path=%s -- swallowed",
            path,
        )
