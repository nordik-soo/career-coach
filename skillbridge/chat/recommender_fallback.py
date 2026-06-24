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
        TrainingResource,
    )


# Locked verbatim chain closings (mirrors RECOMMENDER_RESPONDER_PROMPT).
_CLOSE_LOCAL_GAP_COACH: str = (
    "Want me to compare your skills with the Canadian/NOC standard "
    "for this occupation?"
)
_CLOSE_TARGET_NOC_STANDARD: str = (
    "Want me to show how to prepare for those related career paths?"
)


def render_recommender_fallback(
    rec_evidence: "RecommenderEvidence | None",
    mode: "str | None" = None,
) -> str:
    """Render a deterministic recommender response from the
    RecommenderEvidence wrapper. Returns coach-voice text.

    Args:
        rec_evidence: the wrapper from handler-side assembly. When
            None (handler contract violation), the function renders
            a minimal safe response with no chain close.
        mode: optional explicit mode override (for tests). When
            None, taken from rec_evidence.mode.

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
        return _render_target_noc_standard(rec_evidence)
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
    locked chain close to target_noc_standard."""
    if not rec.evidence:
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


def _render_target_noc_standard(rec: "RecommenderEvidence") -> str:
    """Layer A response: name top development areas in development-area
    voice. NEVER deficit voice. End with the locked chain close to
    adjacent_noc_standard."""
    if not rec.evidence:
        return (
            "I don't have a Canadian/NOC standard skill profile loaded "
            "for that occupation yet. "
            + _CLOSE_TARGET_NOC_STANDARD
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
        f"The Canadian/NOC standard for this occupation emphasizes "
        f"{names_phrase}. These are standard development areas for the "
        "occupation; you can strengthen and demonstrate them in how you "
        "describe your work. "
    )
    return body + _CLOSE_TARGET_NOC_STANDARD


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
    # Chain ENDS here -- natural follow-up only.
    return body + " Want to dig into one of these in particular?"


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
