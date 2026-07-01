"""Slice 5 step 4 (2026-06-19) -- recommender response fallback.

Deterministic per-mode templates that the responder uses when the
LLM-first path can't render. Mirrors the existing
coach_tiers_fallback.py pattern -- coach voice, no policy logic, no
external dependencies. The fallback fires when:
  - recommendation_evidence is None (handler contract violation;
    error-log + best-effort empty narration).
  - LLM is disabled at config level.
  - LLM returns an empty / falsy reply.
  - LLM reply fails the recommender policy gate (when wired in
    responder.py).

Voice rules per the locked design:
  local_gap_coach (Layer B):
    deficit voice OK (the user genuinely lacks the skill for a
    specific posting). Surface verified training providers from
    the TRAINING list verbatim. Fall back to "ask the SCCC" when
    no verified course is attached.

  target_noc_standard (Layer A):
    occupation-standard voice ONLY. NEVER deficit voice. The
    development_area voice means "the occupation emphasizes X" /
    "you can strengthen X" / "you can demonstrate X." NEVER "you
    don't have X" / "you lack X."

  adjacent_noc_standard (Layer C):
    same development-area voice as Layer A, framed as exploratory
    ("if you wanted to move toward [NOC title], that role
    emphasizes..."). Per-NOC paragraph structure.

Chained closings are emitted by the templates the same way the
RECOMMENDER_RESPONDER_PROMPT instructs the LLM to emit them, so
the user experience is consistent across LLM-success and
fallback paths.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skillbridge.chat.gap_evidence import (
        GapEvidence,
        RecommenderEvidence,
        RoleDrilldownEvidence,
        RoleDrilldownSkillRow,
        TrainingResource,
    )


# Slice 2 (locked 2026-06-23): chain closings reassigned for the
# new chain B -> C -> END, A -> END (intent-only).
#
# Layer B closes by OFFERING related career paths (Layer C).
# Layer C closes natural -- no chain to A (A is intent-only).
# Layer A closes natural -- no chain in or out.
_CLOSE_LOCAL_GAP_COACH: str = (
    "Want me to show what related career paths your skills line up "
    "with?"
)
_CLOSE_ADJACENT_NOC_STANDARD_NATURAL: str = (
    "Want to dig into one of these in particular?"
)
# Slice 5 (2026-06-29): Layer C close becomes an EXPLICIT offer
# that maps to the new pending state adjacent_role_drilldown_select.
# Replaces _CLOSE_ADJACENT_NOC_STANDARD_NATURAL at all call sites
# that intend to enable the drilldown handoff.
_CLOSE_ADJACENT_NOC_STANDARD_OFFER_DRILLDOWN: str = (
    "Want a skill-by-skill comparison and training options for one "
    "of these? Say which one."
)
_CLOSE_TARGET_NOC_STANDARD_NATURAL: str = (
    "Anything in there you want to dig into?"
)


def render_recommender_fallback(
    rec_evidence: "RecommenderEvidence | None",
    mode: "str | None" = None,
    target_role_text: str | None = None,
) -> str:
    """Render a deterministic recommender response from the
    RecommenderEvidence wrapper. Returns coach-voice text.

    Args:
        rec_evidence: the wrapper from handler-side assembly. When
            None (handler contract violation), the function renders
            a minimal safe response with no chain close.
        mode: optional explicit mode override (for tests). When
            None, taken from rec_evidence.mode.
        target_role_text: user's stated role text. Used by Layer A
            to refer to the role the way the user phrased it,
            rather than as the OaSIS source_label which can differ
            (e.g. user says "accounting clerk" while OaSIS source
            label is "Cost clerk"). Slice 2 grounding fix.

    Returns:
        Plain-text response. No URLs are invented; only TrainingResource
        URLs from the wrapper appear in output. Voice rules per the
        locked design enforced by template structure.
    """
    if rec_evidence is None:
        return (
            "I can't render that recommendation right now. "
            "What else can I help with?"
        )

    active_mode = mode or rec_evidence.mode
    if active_mode == "local_gap_coach":
        return _render_local_gap_coach(rec_evidence)
    if active_mode == "target_noc_standard":
        return _render_target_noc_standard(rec_evidence, target_role_text)
    if active_mode == "adjacent_noc_standard":
        return _render_adjacent_noc_standard(rec_evidence)
    # Unknown mode -- defensive (shouldn't happen given the Literal).
    return (
        "I can't render that recommendation right now. "
        "What else can I help with?"
    )


def _render_local_gap_coach(rec: "RecommenderEvidence") -> str:
    """Layer B response: name the top gap, attach training if
    available, fall back to SCCC referral if not. End with the
    slice-2 chain close to adjacent_noc_standard (offer C)."""
    if not rec.evidence:
        # Slice 2: this branch is reached ONLY from non-empty Layer B
        # in the new dispatch (the empty paths are handled by canned
        # texts in handler.py before the responder is invoked). Kept
        # for defense / direct fallback callers.
        return (
            "I looked at the gaps for those postings but didn't surface "
            "a single top development area to focus on right now -- "
            "your skills line up well with these matches. "
            + _CLOSE_LOCAL_GAP_COACH
        )

    # Top-1 by design (CP4 produces one primary recommendation).
    top = rec.evidence[0]
    matching_training = [
        t for t in rec.training
        if _training_matches_gap(t, top)
    ]

    if matching_training:
        # Name the first verified training resource verbatim.
        first = matching_training[0]
        body = (
            f"Looking at the gaps across those postings, "
            f"{top.skill_name} is the biggest one to address. "
            f"{first.provider} offers {first.summary}: {first.url}. "
        )
    else:
        body = (
            f"Looking at the gaps across those postings, "
            f"{top.skill_name} is the biggest one to address. "
            "I don't have a verified course in my registry for that "
            "one; the Sault Community Career Centre can help you find "
            "a path. "
        )

    return body + _CLOSE_LOCAL_GAP_COACH


def _render_target_noc_standard(
    rec: "RecommenderEvidence",
    target_role_text: str | None = None,
) -> str:
    """Layer A response: name top development areas in development-area
    voice. NEVER deficit voice. Closes natural (no chain).

    Slice 2 grounding fix: when target_role_text is supplied, the
    response uses it verbatim instead of "this occupation" so the
    LLM/user-facing surface matches the user's stated role text
    (e.g. "accounting clerk" instead of OaSIS source_label
    "Cost clerk").
    """
    role_phrase = (target_role_text or "").strip() or "this occupation"

    if not rec.evidence:
        return (
            f"I don't have a Canadian/NOC standard skill profile loaded "
            f"for {role_phrase} yet. "
            + _CLOSE_TARGET_NOC_STANDARD_NATURAL
        )

    # The wrapper arrives already capped at top-3 by importance.
    skill_names = [g.skill_name for g in rec.evidence]
    if len(skill_names) == 1:
        names_phrase = skill_names[0]
    elif len(skill_names) == 2:
        names_phrase = f"{skill_names[0]} and {skill_names[1]}"
    else:
        names_phrase = (
            ", ".join(skill_names[:-1]) + f", and {skill_names[-1]}"
        )

    body = (
        f"The Canadian/NOC standard for {role_phrase} emphasizes "
        f"{names_phrase}. These are standard development areas for the "
        "occupation; you can strengthen and demonstrate them in how you "
        "describe your work. "
    )
    return body + _CLOSE_TARGET_NOC_STANDARD_NATURAL


def _render_adjacent_noc_standard(rec: "RecommenderEvidence") -> str:
    """Layer C response: per-NOC paragraph, development-area voice,
    exploratory framing. Chain ENDS HERE -- no further mode offer,
    just a natural follow-up question."""
    if not rec.evidence:
        return (
            "Nothing surfaced from adjacent roles in this session. "
            "Want to dig into anything specific from those original "
            "matches?"
        )

    # Group records by (source_id, source_label) preserving first-seen
    # order. The wrapper-assembly side already capped per-NOC entries
    # at top-3 by importance.
    grouped: dict[str, dict] = {}
    order: list[str] = []
    for g in rec.evidence:
        key = g.source_id
        if key not in grouped:
            grouped[key] = {"label": g.source_label or key, "skills": []}
            order.append(key)
        grouped[key]["skills"].append(g.skill_name)

    paragraphs: list[str] = []
    for noc in order:
        info = grouped[noc]
        label = info["label"]
        skills = info["skills"]
        if len(skills) == 1:
            skills_phrase = skills[0]
        elif len(skills) == 2:
            skills_phrase = f"{skills[0]} and {skills[1]}"
        else:
            skills_phrase = (
                ", ".join(skills[:-1]) + f", and {skills[-1]}"
            )
        paragraphs.append(
            f"If you wanted to move toward {label}, that role "
            f"emphasizes {skills_phrase}."
        )

    body = " ".join(paragraphs)
    # Slice 5 (2026-06-29): chain close changed from natural follow-up
    # to explicit drilldown offer. Pairs with the handler setting
    # pending_recommender_offer = "adjacent_role_drilldown_select" +
    # populating staged.last_recommender_adjacent_surface, so the user's
    # next-turn selection routes via the consume hook to the drilldown
    # dispatcher.
    return body + " " + _CLOSE_ADJACENT_NOC_STANDARD_OFFER_DRILLDOWN


def _training_matches_gap(
    training: "TrainingResource",
    gap: "GapEvidence",
) -> bool:
    """Match TrainingResource to GapEvidence by skill_id preferred,
    skill_name fallback. Mirrors the per-skill attachment contract
    locked in Slice 5 step 1."""
    if training.skill_id is not None and gap.skill_id is not None:
        return training.skill_id == gap.skill_id
    # Fall back to case-insensitive name match.
    return training.skill_name.strip().lower() == gap.skill_name.strip().lower()


# ===========================================================================
# Slice 5 (2026-06-29) -- role drilldown markdown table renderer
# ===========================================================================
#
# Per the locked design: the table is rendered DETERMINISTICALLY by
# Python, NOT by the LLM. The LLM only writes an optional short close
# AFTER the table. This eliminates hallucination risk in the "Your
# Skill" cell because that cell is a concrete data field, not LLM
# prose.
#
# Honest fallback: when RoleDrilldownEvidence has empty rows (OaSIS
# profile not loaded for the chosen NOC), render no table -- emit a
# short canned text inviting the user to pick a different surfaced
# NOC.
# ===========================================================================
_DRILLDOWN_EMPTY_OASIS_FALLBACK: str = (
    "I don't have a Canadian/NOC standard profile loaded for "
    "{role_title} yet. {other_options_clause}"
)


def render_role_drilldown_table(
    evidence: "RoleDrilldownEvidence",
    target_role_text: str | None = None,
) -> str:
    """Slice 5 deterministic markdown table renderer.

    Returns the heading + table markdown for the drilldown payload.
    The caller (responder) appends the LLM-written close after this
    block.

    Empty rows -> caller emits honest fallback via
    render_role_drilldown_empty_fallback (NOT this function).

    Args:
        evidence: RoleDrilldownEvidence with role_title + rows.
        target_role_text: NOT used by the table itself, but kept in
            the signature for parity with other renderers that need
            user-facing role text.

    Returns:
        Markdown string: `**Target role:** ...` heading + table.
    """
    role_label = evidence.role_title or "this role"
    noc_suffix = (
        f" (NOC {evidence.noc_code})" if evidence.noc_code else ""
    )
    heading = f"**Target role:** {role_label}{noc_suffix}\n\n"

    if not evidence.rows:
        # Empty rows shouldn't reach this renderer (caller should
        # have routed to fallback). Defensive: return just the
        # heading.
        return heading.rstrip()

    # Slice 8 (2026-06-30): column header renamed from "Your Skill"
    # (mechanical comma-list) to "Your Evidence" (LLM-written coach
    # prose when available). The cell content function below prefers
    # row.user_evidence over row.your_skill_names.
    table_lines: list[str] = [
        "| OaSIS Skill | Your Evidence | Status | Training Direction |",
        "|---|---|---|---|",
    ]
    for row in evidence.rows:
        table_lines.append(_render_drilldown_row(row))

    table_md = heading + "\n".join(table_lines)

    # Slice 9 (2026-06-30): append the Coach Training Guide if
    # attached. When evidence.coach_guide is None the table renders
    # alone (feature off, LLM unavailable, or empty-rows guard above).
    guide = getattr(evidence, "coach_guide", None)
    if guide is not None:
        return table_md + "\n\n" + render_coach_training_guide(guide)
    return table_md


def render_coach_training_guide(guide: "CoachTrainingGuide") -> str:
    """Slice 9 (2026-06-30) deterministic renderer for the Coach
    Training Guide section beneath the drilldown table.

    Structure:
      **Coach Training Guide**

      <opening_sentence>

      **1. <skill>**
      - Why it matters: <why_it_matters>
      - How to build it: <how_to_build>
      - Training direction: <training_direction> [Provider](URL)

      **2. <skill>** ... (only if 2+ priority_gaps)

      <closing_question>

    Zero-gap branch (priority_gaps == ()): only opening + closing
    render. No gap sections. The opening sentence carries the
    deterministic encouragement template written by
    _build_zero_gap_encouragement.

    Provider markdown link: appended verbatim ONLY when the guide's
    CoachGapGuide row carries a non-None training_provider AND
    training_url (attached by assembly from the registry, NOT the
    LLM). If only provider is set (URL suppressed / stale / referral),
    provider name is appended in plain text. When both are None, the
    LLM's training_direction prose stands alone (which for the
    "ask SCCC" case already names SCCC as guidance).
    """
    lines: list[str] = [
        "**Coach Training Guide**",
        "",
        guide.opening_sentence,
    ]
    if guide.priority_gaps:
        lines.append("")
        for i, gap in enumerate(guide.priority_gaps, start=1):
            lines.append(f"**{i}. {gap.skill}**")
            lines.append(f"- Why it matters: {gap.why_it_matters}")
            lines.append(f"- How to build it: {gap.how_to_build}")
            td = gap.training_direction
            if gap.training_provider and gap.training_url:
                td = (
                    f"{td} "
                    f"[{gap.training_provider}]({gap.training_url})"
                )
            elif gap.training_provider:
                td = f"{td} ({gap.training_provider})"
            lines.append(f"- Training direction: {td}")
            lines.append("")
    lines.append(guide.closing_question)
    return "\n".join(lines)


def _render_drilldown_row(row: "RoleDrilldownSkillRow") -> str:
    """Render a single row of the drilldown table.

    Cell formats per locked design:
      OaSIS Skill: row.oasis_skill_name verbatim
      Your Evidence (slice 8):
        matched=True with row.user_evidence (LLM-written): the LLM
          string verbatim (coach prose, ~150 chars max)
        matched=True without user_evidence (cosine-only fallback):
          up to 2 names from row.your_skill_names, comma-separated
        matched=False: "—" (em-dash)
      Status: "✓" if matched else "✗"
      Training Direction (matched=True): "already have"
      Training Direction (matched=False) + registry hit:
        markdown link "[provider](url)"
      Training Direction (matched=False) + registry miss:
        "ask SCCC"
    """
    # Slice 8: prefer LLM-written user_evidence when present; fall
    # back to your_skill_names list (slice 7a behavior, used when
    # LLM is unavailable or the fallback path fired).
    if not row.matched:
        your_skill = "—"
    elif row.user_evidence:
        your_skill = row.user_evidence
    elif row.your_skill_names:
        your_skill = ", ".join(row.your_skill_names)
    else:
        # Matched but no evidence content -- defensive fallback so
        # the cell isn't empty.
        your_skill = "—"

    status = "✓" if row.matched else "✗"

    if row.matched:
        training = "already have"
    elif row.training_provider and row.training_url:
        # Markdown link rendering; clickable in markdown surfaces,
        # readable as plain text in non-markdown chat surfaces.
        training = f"[{row.training_provider}]({row.training_url})"
    else:
        training = "ask SCCC"

    # Defensive escaping: a pipe character in a cell would break the
    # markdown table. Replace with the HTML-style escape that
    # renderers commonly accept; if the cell content can never
    # legitimately have a pipe (provider names from registry should
    # not), this is a no-op for valid data and a safety net for
    # malformed data.
    cells = [
        row.oasis_skill_name.replace("|", "\\|"),
        your_skill.replace("|", "\\|"),
        status,
        training.replace("|", "\\|"),
    ]
    return "| " + " | ".join(cells) + " |"


def render_role_drilldown_empty_fallback(
    evidence: "RoleDrilldownEvidence",
    other_surface_titles: tuple[str, ...] = (),
) -> str:
    """Slice 5 honest fallback when the OaSIS profile isn't loaded
    for the chosen NOC. Mirrors slice 2's _LAYER_A_EMPTY_HONEST
    shape -- never invents skills, never renders an empty table.

    Args:
        evidence: RoleDrilldownEvidence with role_title (possibly
            empty) and rows=().
        other_surface_titles: titles of the OTHER NOCs still in
            the user's adjacent surface (so they can pick one).
            Empty -> no alternative options offered.
    """
    role_label = evidence.role_title or "that role"
    if other_surface_titles:
        joined = (
            other_surface_titles[0] if len(other_surface_titles) == 1
            else (
                ", ".join(other_surface_titles[:-1])
                + " or "
                + other_surface_titles[-1]
            )
        )
        clause = f"Want to dig into {joined} instead?"
    else:
        clause = (
            "Want to look at a different target, or check what jobs "
            "are open in this field?"
        )
    return _DRILLDOWN_EMPTY_OASIS_FALLBACK.format(
        role_title=role_label,
        other_options_clause=clause,
    )


def render_role_drilldown_reprompt(
    surface: tuple[dict, ...],
) -> str:
    """Slice 5 re-prompt when the user's selection didn't resolve
    (consent='yes' without specifying a role, or multiple title
    matches). Asks the user to clarify with explicit options.

    Args:
        surface: the active last_recommender_adjacent_surface; each
            entry has noc_code + title.
    """
    if not surface:
        return (
            "I don't have any adjacent roles saved from this "
            "conversation. Want me to look at related career paths "
            "for your target?"
        )
    titles = [
        e.get("title", "")
        for e in surface
        if isinstance(e, dict) and e.get("title")
    ]
    if not titles:
        return (
            "I don't have any adjacent roles saved from this "
            "conversation. Want me to look at related career paths "
            "for your target?"
        )
    if len(titles) == 1:
        joined = titles[0]
    else:
        joined = (
            ", ".join(titles[:-1]) + ", or " + titles[-1]
        )
    return (
        f"Which one would you like to compare against -- {joined}?"
    )
