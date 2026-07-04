"""Recommender → matching handoff helper (slice 2 step 2.5, 2026-07-04).

When the reference resolver returns a resolved role-kind item from the
recommender's adjacent surface, this helper writes the target role +
NOC onto StagedProfile in the locked order so downstream matching runs
against the new target.

Deliberately separated from `reference_resolver.py` because the
resolver is read-only (pure functions, no side effects) and this
module MUTATES staged. Keeping the boundary explicit prevents callers
from accidentally invoking this helper from read-only code paths.

Locked mutation order (2026-07-03 sign-off):

  1. Capture label + id into local variables. StagedProfile's
     __setattr__ hook cascade-clears the recommender adjacent surface
     (+ anchor) the moment target_role_text is written, so any read of
     `resolved_item.label` AFTER the target write is a use-after-clear
     hazard when the item's source is the surface being cleared.

  2. Set staged.target_role_text = label. The __setattr__ hook at
     staging.py:501 cascades to clear:
       - target_noc
       - last_match_snapshot (+ companion credential state)
       - last_recommender_adjacent_surface (+ Step 1.2 anchor)
       - pending_recommender_offer
       - pending_adjacent_search_offer
       - last_adjacent_snapshot
       - last_adjacent_nocs
       - resume_upload_offered
       - deferred_career_intent (conditional on prior_target_existed)
     This clean-slate on target change is CORRECT lifecycle per the
     slice-2 handoff design lock -- the resolver treats the surface
     as read-only input and, once we've captured what we need, the
     surface itself becomes stale.

  3. If the captured id is exactly 5 digits (a NOC 2021 code), set
     staged.target_noc = id AFTER the target_role_text write, so the
     cascade doesn't wipe it. If id is None or non-5-digit, leave
     target_noc at None -- the matching engine's lazy
     `resolve_title_to_noc` handles the fallback via its normal cache.

Scope discipline (locked):

  - Only role-kind items. resolved_item.kind != "role" raises
    ValueError. Job-kind references are a different behavior
    (matching-job follow-up, not target-role switch) and are
    explicitly out of scope for slice 2.

  - Empty / whitespace-only label raises ValueError. A handoff to a
    blank target is a caller bug -- setting target_role_text = "" or
    "   " would leave the profile in a broken state that the engine
    would then have to guess around.
"""
from __future__ import annotations

from skillbridge.chat.conversation_frame import SurfaceItem
from skillbridge.session.staging import StagedProfile


def handoff_recommender_to_matching(
    staged: StagedProfile,
    resolved_item: SurfaceItem,
) -> None:
    """Write the resolved role's label + NOC onto staged as the new
    matching target.

    See module docstring for the locked mutation order and scope
    discipline. Never returns anything meaningful -- the caller then
    routes to the matching flow, which will read the fresh
    target_role_text (and target_noc if set) from staged.

    Raises:
      ValueError: `resolved_item.kind != "role"` (job-kind is out of
        scope for slice 2); or the item's label is empty / whitespace
        only (handoff to a blank target is a caller bug).
    """
    if resolved_item.kind != "role":
        raise ValueError(
            "handoff_recommender_to_matching only supports role-kind "
            f"items; got kind={resolved_item.kind!r}. Job-kind "
            "references are a different behavior (matching-job "
            "follow-up, not target-role switch) and are out of scope "
            "for slice 2."
        )

    # Step 1: capture. StagedProfile's __setattr__ cascade will clear
    # the recommender surface momentarily; any read after the write
    # is a use-after-clear hazard when the item came from that surface.
    label = resolved_item.label
    noc_id = resolved_item.id

    if not isinstance(label, str) or not label.strip():
        raise ValueError(
            f"handoff requires a non-empty label; got {label!r}"
        )

    # Step 2: set target_role_text. The __setattr__ hook at
    # staging.py:501 cascades to clear the recommender surface and
    # everything else that was scoped to the prior target.
    staged.target_role_text = label

    # Step 3: if id is a 5-digit NOC 2021 code, write it AFTER the
    # cascade so it survives. Otherwise leave target_noc at None --
    # the matching engine's lazy title-to-NOC resolver will handle
    # the fallback path via its normal cache; adding a warning here
    # would just add noise for a case that isn't broken.
    if isinstance(noc_id, str) and len(noc_id) == 5 and noc_id.isdigit():
        staged.target_noc = noc_id
