"""AR-9.feat.coach-tiers CP1 step 10 — deterministic tier-structured
fallback renderer.

Used when the LLM is unavailable OR the policy gate rejects the LLM
response. Produces the same three sections as the LLM coach response,
from purely templated prose grounded in the view fields:
  - "Apply today — your skills line up"
  - "Worth a try — close, with gaps to address"
  - "Sideways move — same skills, different angle"

Hard rules (v5 lock + step-10 spec):
  - Reads ONLY `view.prompt_tiered_*` records — never the raw
    `TieredEvidence`, never a raw job dict, never `MatchResult`.
  - Strength phrasing is a closed-vocab lookup keyed on
    `strength_claim_text` — no generated prose, no invented wording.
  - URLs rendered are exactly the SanitizedURLs present on the tier
    records (job URLs + training URLs with valid SanitizedURL).
    Non-SanitizedURL slots contribute no URL.
  - No cards. No invented advice. No matching logic. The renderer is
    purely a string projection.
  - Empty tiers produce no header and no paragraph — the section
    disappears.
  - One closing question, picked from a closed set based on which
    tiers were rendered.
"""
from __future__ import annotations

from typing import Iterable

from skillbridge.chat.pipeline_snapshot import PipelineSnapshot
from skillbridge.chat.url_views import (
    PromptAdjacentJob,
    PromptStretchMatch,
    PromptStrongMatch,
    SanitizedResponderView,
    SanitizedURL,
)


# Locked v5 strength-phrase vocabulary. The prompt uses the same set;
# the fallback shares it so the user-visible voice matches across
# LLM-on / LLM-off paths.
_STRENGTH_PHRASES: dict[str, str] = {
    "competitive_match": "You'd be competitive for this one.",
    "strongest_current": "This is one of your strongest current matches.",
    "close_with_named_gap": "This one is close — there's a specific gap to close first.",
    "stretch_with_training_bridge": "This one is reachable with a step in between.",
    "transferable_lane": "Your skill set carries over here.",
}


# CP2 step 6.4 — repeat-opener machinery removed. The fallback is the
# LLM-unreachable last resort; the validator no longer demands a
# canonical phrase per record so the fallback can lead each record
# with the canonical phrase without robotic-repetition concerns. If
# the LLM is reachable the user sees natural prose; if it isn't, the
# fallback at least stays grounded.


# scoring-v6 (2026-06-17): heading rename — see closing-matrix v2 memory.
# The previous strings "**Apply today — your skills line up**" and
# "**Worth a try — close, with gaps to address**" were replaced with
# the 4-label vocabulary. The CONSTANT NAMES are kept (call sites
# elsewhere in this file remain unchanged) — only the user-facing
# strings updated to align with the new prompt headings.
# Two NEW constants — _HEADER_GOOD_MATCH and _HEADER_EXPLORE_LATER —
# are introduced for the two newly-distinct labels. The fallback
# renderer's per-tier render functions don't yet split Strong vs
# Good or surface Explore-later separately — that's a follow-on
# (the fallback degradation is acceptable; the LLM happy-path is
# the canonical path).
_HEADER_APPLY_TODAY = "**Strong match — apply today**"
_HEADER_GOOD_MATCH = "**Good match — solid fit**"
_HEADER_WORTH_A_TRY = "**Stretch — reachable with prep**"
_HEADER_EXPLORE_LATER = "**Explore later — not your main target**"
_HEADER_SIDEWAYS = "**Sideways move — same skills, different angle**"


# Closed closing-question set. Selection is deterministic from which
# tiers were rendered. The fallback always closes with one question.
_CLOSING_ALL_TIERS = "Which of these would you like to look at first?"
_CLOSING_APPLY_AND_STRETCH = "Which of these would you like to look at first?"
_CLOSING_APPLY_AND_SIDEWAYS = "Which of these would you like to look at first?"
_CLOSING_APPLY_ONLY = "Want me to dig into any of these further?"
_CLOSING_STRETCH_AND_SIDEWAYS = "Would the prep be doable, or should we look at the sideways options first?"
_CLOSING_STRETCH_ONLY = "Would the prep be doable, or should we look at other options?"
# Fix 3 (post-step-10 review): Sideways-only already surfaces live
# listings (with URLs), so asking to "pull live listings" was
# self-contradictory. Also: the empty closing was grammatically
# awkward ("Tell me more about the kind of role you're aiming for?"
# reads as a statement with a tacked-on question mark).
_CLOSING_SIDEWAYS_ONLY = "Which sideways option would you like to explore first?"
_CLOSING_EMPTY = "What kind of work are you aiming for?"


_EMPTY_BODY = "Nothing on the board matches yet."


def _compose_empty_body(snapshot: PipelineSnapshot | None) -> str:
    """Step 12: when a `PipelineSnapshot` is supplied, surface the
    grounded count and (when known) the last-refreshed timestamp.
    Otherwise return the generic empty body.

    Used ONLY for the empty-tier branch. The snapshot must not
    influence matching or tier selection — its single user is this
    one composition site.
    """
    if snapshot is None:
        return _EMPTY_BODY
    parts: list[str] = [_EMPTY_BODY]
    parts.append(
        f"The current view has {snapshot.total_active_jobs} active postings"
    )
    if snapshot.last_publish_at_text:
        parts[-1] += f", last refreshed {snapshot.last_publish_at_text}."
    else:
        parts[-1] += "."
    return " ".join(parts)


# =========================================================================
# Section helpers — small, single-purpose, return (text, urls)
# =========================================================================
def _facts_clause(item: object) -> str:
    """Compose a parenthesized fact clause from job_facts.
    Returns "" when no sourceable fact is present.

    Step 11 (post-step-10 review): salary_text is intentionally
    OMITTED from this clause. Salary grounding is more complex than
    `$`-matching (formatting varies, ranges/units differ, numbers
    can collide with unrelated text in the reply, and stale wording
    can sneak through). The locked decision (option B) is to keep
    salary_text on the evidence and view for future use, but never
    surface it in the LLM prompt or the deterministic fallback in
    this slice.
    """
    facts = getattr(item, "job_facts", None)
    if facts is None:
        return ""
    parts: list[str] = []
    et = getattr(facts, "employment_type", None)
    if isinstance(et, str) and et:
        parts.append(et)
    pda = getattr(facts, "posted_days_ago", None)
    if isinstance(pda, int):
        parts.append(f"posted {pda} days ago")
    # Salary deliberately omitted — see step-11 rationale above.
    if not parts:
        return ""
    return f"({', '.join(parts)})"


def _job_url_clause(item: object) -> tuple[str, str | None]:
    """Returns (clause, canonical_url). clause is the bare URL string
    suitable for inclusion in prose. Empty string + None when no URL."""
    url = getattr(item, "url", None)
    if isinstance(url, SanitizedURL):
        return url.raw, url.canonical
    return "", None


def _credential_note_clause(item: object) -> str:
    """One-clause occupational licensing note, if present."""
    text = getattr(item, "credential_warning_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    return ""


def _alignment_sentence(alignments: object) -> str:
    """Fix 2 (post-step-10 review): a grounded match-detail sentence
    sourced directly from `skill_alignment`. The previous Apply Today
    paragraph stopped at "competitive" and the employer/title, which
    repeated the old shallow-response problem (no evidence that this
    job actually matches the user). One short sentence drawn from
    the top 1–2 alignment records grounds the claim.

    Example outputs (templated, never invented):
        "Your QuickBooks aligns with their QuickBooks requirement."
        "Your bookkeeping and payroll align with their accounts-payable and reconciliation requirements."

    Returns "" when there are no alignments to summarize.
    """
    if not alignments:
        return ""
    items = list(alignments[:2])
    if not items:
        return ""
    user_skills = [a.user_skill for a in items]
    job_reqs = [a.job_requirement for a in items]
    if len(items) == 1:
        return f"Your {user_skills[0]} aligns with their {job_reqs[0]} requirement."
    return (
        f"Your {user_skills[0]} and {user_skills[1]} align with "
        f"their {job_reqs[0]} and {job_reqs[1]} requirements."
    )


def _render_strong_paragraph(item: PromptStrongMatch) -> tuple[str, set[str]]:
    urls: set[str] = set()
    sentences: list[str] = [_STRENGTH_PHRASES[item.strength_claim_text]]
    if item.employer:
        sentences.append(f"{item.employer} is hiring {item.title}.")
    else:
        sentences.append(f"{item.title}.")
    # Fix 2: grounded match detail directly after the title.
    align = _alignment_sentence(item.skill_alignment)
    if align:
        sentences.append(align)
    facts = _facts_clause(item)
    if facts:
        sentences.append(facts)
    cred = _credential_note_clause(item)
    if cred:
        sentences.append(cred)
    url_clause, canonical = _job_url_clause(item)
    if url_clause:
        sentences.append(url_clause)
        if canonical is not None:
            urls.add(canonical)
    return " ".join(sentences), urls


def _render_gap_and_training(
    gap: object, *, lead: bool,
) -> tuple[list[str], set[str]]:
    """Step-12 review Medium: shared gap-and-training renderer.

    Renders one PromptPrioritizedGap as 1–2 sentences:
      - A lead sentence naming the gap. When `lead=True`, "The gap
        is X." When `lead=False`, "Also a credential gap: X." (used
        only for additional credential blockers beyond the first gap.)
      - A training sentence, naming the first actionable training
        option (one with a SanitizedURL) or the honest fallback line
        "I don't have a verified training option for that gap yet."

    Returns (sentences, urls).
    """
    urls: set[str] = set()
    sentences: list[str] = []
    if lead:
        sentences.append(f"The gap is {gap.job_requirement}.")  # type: ignore[attr-defined]
    else:
        sentences.append(f"Also a credential gap: {gap.job_requirement}.")  # type: ignore[attr-defined]
    actionable = [
        t for t in gap.training_options  # type: ignore[attr-defined]
        if isinstance(t.url, SanitizedURL)
    ]
    if actionable:
        t = actionable[0]
        duration = (
            f" ({t.duration_text})" if t.duration_text else ""
        )
        sentences.append(
            f"{t.provider} offers {t.title}{duration}: {t.url.raw}"
        )
        urls.add(t.url.canonical)
    else:
        sentences.append(
            "I don't have a verified training option for that gap yet."
        )
    return sentences, urls


def _render_stretch_paragraph(item: PromptStretchMatch) -> tuple[str, set[str]]:
    urls: set[str] = set()
    sentences: list[str] = [_STRENGTH_PHRASES[item.strength_claim_text]]
    if item.employer:
        sentences.append(f"{item.employer} is hiring {item.title}.")
    else:
        sentences.append(f"{item.title}.")
    # Fix 2: name one aligned skill BEFORE naming the first gap.
    # Otherwise the fallback collapses to "close, with a gap" with
    # no evidence the user has any of the matching skills.
    align = _alignment_sentence(item.skill_alignment)
    if align:
        sentences.append(align)
    facts = _facts_clause(item)
    if facts:
        sentences.append(facts)
    # Name the first gap (engine emits gaps in importance order).
    if item.prioritized_gaps:
        first = item.prioritized_gaps[0]
        lead_sentences, lead_urls = _render_gap_and_training(first, lead=True)
        sentences.extend(lead_sentences)
        urls.update(lead_urls)
        # Step-12 review Medium: surface EVERY credential blocker that
        # wasn't the first gap. Without this, a credential at position
        # ≥ 1 (when a non-credential gap came first by importance order)
        # is silently hidden and the user never sees the blocker.
        for gap in item.prioritized_gaps[1:]:
            if gap.blocker:
                blocker_sentences, blocker_urls = _render_gap_and_training(
                    gap, lead=False,
                )
                sentences.extend(blocker_sentences)
                urls.update(blocker_urls)
    cred = _credential_note_clause(item)
    if cred:
        sentences.append(cred)
    url_clause, canonical = _job_url_clause(item)
    if url_clause:
        sentences.append(url_clause)
        if canonical is not None:
            urls.add(canonical)
    return " ".join(sentences), urls


def _render_sideways_paragraph(item: PromptAdjacentJob) -> tuple[str, set[str]]:
    urls: set[str] = set()
    sentences: list[str] = [_STRENGTH_PHRASES[item.strength_claim_text]]
    if item.employer:
        sentences.append(f"{item.employer} has a {item.title} posting.")
    else:
        sentences.append(f"{item.title}.")
    # First transferable pair only — keeps the fallback compact.
    if item.transferable_pairs:
        p = item.transferable_pairs[0]
        sentences.append(
            f"Your {p.user_skill} carries over to {p.applies_to}."
        )
    cred = _credential_note_clause(item)
    if cred:
        sentences.append(cred)
    url_clause, canonical = _job_url_clause(item)
    if url_clause:
        sentences.append(url_clause)
        if canonical is not None:
            urls.add(canonical)
    return " ".join(sentences), urls


def _render_section(
    header: str,
    paragraphs: Iterable[tuple[str, set[str]]],
) -> tuple[str, set[str]]:
    paragraphs = list(paragraphs)
    if not paragraphs:
        return "", set()
    urls: set[str] = set()
    body_lines: list[str] = [header]
    for text, paragraph_urls in paragraphs:
        body_lines.append(text)
        urls.update(paragraph_urls)
    return "\n\n".join(body_lines), urls


def _render_apply_today_section(
    items: tuple[PromptStrongMatch, ...],
) -> tuple[str, set[str]]:
    return _render_section(
        _HEADER_APPLY_TODAY,
        (_render_strong_paragraph(i) for i in items),
    )


def _render_worth_a_try_section(
    items: tuple[PromptStretchMatch, ...],
) -> tuple[str, set[str]]:
    return _render_section(
        _HEADER_WORTH_A_TRY,
        (_render_stretch_paragraph(i) for i in items),
    )


def _render_sideways_section(
    items: tuple[PromptAdjacentJob, ...],
) -> tuple[str, set[str]]:
    return _render_section(
        _HEADER_SIDEWAYS,
        (_render_sideways_paragraph(i) for i in items),
    )


def _closing_question(
    has_apply: bool, has_stretch: bool, has_sideways: bool,
) -> str:
    """Closed-set closing-question selector. Deterministic from the
    tier-presence flags. The fallback always ends with one question."""
    if not (has_apply or has_stretch or has_sideways):
        return _CLOSING_EMPTY
    if has_apply and has_stretch and has_sideways:
        return _CLOSING_ALL_TIERS
    if has_apply and has_stretch:
        return _CLOSING_APPLY_AND_STRETCH
    if has_apply and has_sideways:
        return _CLOSING_APPLY_AND_SIDEWAYS
    if has_apply:
        return _CLOSING_APPLY_ONLY
    if has_stretch and has_sideways:
        return _CLOSING_STRETCH_AND_SIDEWAYS
    if has_stretch:
        return _CLOSING_STRETCH_ONLY
    return _CLOSING_SIDEWAYS_ONLY


# =========================================================================
# Public entry points
# =========================================================================
def render_coach_tiers_fallback(
    view: SanitizedResponderView,
    snapshot: PipelineSnapshot | None = None,
) -> tuple[str, frozenset[str]]:
    """Render the deterministic three-section coach-tier fallback.

    Returns `(text, rendered_urls)`:
      - text is the templated prose body.
      - rendered_urls is the set of canonical URLs the text references.
        The view's `fallback_urls` is computed from this same set; the
        renderer is the authority for what counts as "rendered."

    Reads ONLY `view.prompt_tiered_apply_today`,
    `view.prompt_tiered_worth_a_try`, and
    `view.prompt_tiered_sideways_move`. Other view slots are not
    consulted — they belong to other moves.

    Step 12: an optional `PipelineSnapshot` enriches the empty-tier
    body with the grounded active-job count and (when known) the
    last-refreshed timestamp. The snapshot is used ONLY when every
    tier is empty — it cannot influence non-empty-tier rendering,
    matching, or tier selection.
    """
    apply_text, apply_urls = _render_apply_today_section(
        view.prompt_tiered_apply_today,
    )
    worth_text, worth_urls = _render_worth_a_try_section(
        view.prompt_tiered_worth_a_try,
    )
    side_text, side_urls = _render_sideways_section(
        view.prompt_tiered_sideways_move,
    )

    sections: list[str] = [t for t in (apply_text, worth_text, side_text) if t]
    has_apply = bool(view.prompt_tiered_apply_today)
    has_stretch = bool(view.prompt_tiered_worth_a_try)
    has_sideways = bool(view.prompt_tiered_sideways_move)

    if not sections:
        body = _compose_empty_body(snapshot)
    else:
        body = "\n\n".join(sections)

    closing = _closing_question(has_apply, has_stretch, has_sideways)
    text = f"{body}\n\n{closing}"
    rendered_urls = frozenset(apply_urls | worth_urls | side_urls)
    return text, rendered_urls


def collect_fallback_render_urls(
    view: SanitizedResponderView,
) -> frozenset[str]:
    """The set of canonical URLs the deterministic fallback would
    render given this view. Mirrors what `render_coach_tiers_fallback`
    produces — used by the view builder to populate `fallback_urls`
    so the allowlist is computed from exactly the rendered fields.

    The pipeline snapshot does not influence the URL set — the empty
    body is pure text — so this collector does not need a snapshot
    argument.
    """
    _, urls = render_coach_tiers_fallback(view)
    return urls
