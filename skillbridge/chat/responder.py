"""NEXT_ACTION-driven response composition — conversational, role-aware.

The backend has already decided everything important: which slot to ask
about (at most ONE per turn), whether to show matches, whether to
redirect, whether the profile is ready. The responder only NARRATES that
decision in natural language, never inventing new facts.

Hard rules enforced by the prompt + post-processing:
  - Every job title / employer / URL / training URL must come from the
    RESULTS or TRAINING blocks we pass in.
  - Never invent statistics, wages, or counts.
  - Never promise to remember information until consent has been granted.
  - Ask AT MOST ONE question per turn, woven into prose (no bullets).
  - Acknowledge what the user just said before the next question.
  - Use role-aware example phrasing from intake_priority.prompt_hint().

Sprint 3 — scope boundaries (policy_ok):
  - No offers to search outside Sault Ste. Marie ("nearby city", "try
    Toronto", "broaden the search", etc.).
  - No naming non-local Ontario cities as destinations.
  - No credential-equivalence claims ("equivalent to a Canadian X").
  - No immigration / Express Entry / RCIP-eligibility / PR advice.
  - Soft market observations (e.g. "in a smaller local market, software
    roles are less common") are allowed — but ONLY after anchoring to
    the dataset, which the prompt enforces.

If the LLM is disabled or its response fails the policy check, we fall
back to a deterministic single-question reply so the chat never breaks.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from skillbridge.chat import intake_priority
from skillbridge.chat.intake_state import (
    ACTION_ACKNOWLEDGE_AND_WAIT,
    ACTION_ASK_QUESTIONS,
    ACTION_CONFIRM_READY,
    ACTION_PRESENT_MATCHES,
    ACTION_PRESENT_RESUME_FACTS,
    ACTION_REDIRECT,
    Decision,
)
from skillbridge.chat.prompts import NEXT_ACTION_RESPONDER_PROMPT
from skillbridge.chat.url_policy import (
    Violation,
    check_url_membership,
    extract_url_candidates,
    safe_telemetry_fields,
)
from skillbridge.chat.url_views import (
    SanitizedResponderView,
    build_sanitized_responder_view_for_tiered_matches,
    build_sanitized_responder_view_v1,
    build_sanitized_responder_view_v2,
    serialize_adjacent_recommendations_for_prompt,
    serialize_prompt_adjacent_role,
    serialize_result_for_v1_prompt,
    serialize_result_for_v2_prompt,
    serialize_training_for_v1_prompt,
    serialize_training_for_v2_prompt,
)
from skillbridge.llm import call, is_enabled
from skillbridge.match.engine import is_credential_skill_name
# AR-9.feat.coach-tiers CP2 step 3: imports for the tiered-matches branch.
# All four modules are CP1 leaves — no cycle risk with responder.py.
from skillbridge.chat.coach_tiers_fallback import (
    _CLOSING_ALL_TIERS,
    _CLOSING_APPLY_AND_SIDEWAYS,
    _CLOSING_APPLY_AND_STRETCH,
    _CLOSING_APPLY_ONLY,
    _CLOSING_EMPTY,
    _CLOSING_SIDEWAYS_ONLY,
    _CLOSING_STRETCH_AND_SIDEWAYS,
    _CLOSING_STRETCH_ONLY,
    _HEADER_APPLY_TODAY,
    _HEADER_SIDEWAYS,
    _HEADER_WORTH_A_TRY,
    _STRENGTH_PHRASES,
    _closing_question,
    render_coach_tiers_fallback,
)
from skillbridge.chat.coach_tiers_policy import (
    check_ungrounded_provider_for_tiered_matches,
)
from skillbridge.chat.pipeline_snapshot import PipelineSnapshot
from skillbridge.chat.prompts import (
    COACH_TIERS_RESPONDER_PROMPT,
    RECOMMENDER_RESPONDER_PROMPT,
)
from skillbridge.chat.tiered_evidence import TieredEvidence
# Slice 5 step 4 (2026-06-19) -- recommender response dispatch.
from skillbridge.chat.gap_evidence import RecommenderEvidence
from skillbridge.chat.recommender_fallback import render_recommender_fallback
# Step 3 (2026-06-22): voice_hint carrier for intent-driven recommender
# dispatch. Plumbed only; prompt does not branch on it yet.
from skillbridge.chat.recommender_intent import CareerIntent

log = logging.getLogger(__name__)


@dataclass
class ResponderInput:
    user_message: str
    decision: Decision
    results: list[dict]
    training_by_job: dict[str, list[dict]]
    next_skill: tuple[str | None, int]
    # "strong_or_good" | "stretch_only" | "low_only" | "none".
    # "low_only" means engine found eligible candidates but all low-band;
    # results is empty and the responder narrates the no-match shape.
    band_signal: str
    requires_consent: bool
    # PR 10 UX tuning: target_role drives role-aware prompt hints so the
    # responder asks "what stack have you worked with?" for software vs
    # "what shifts work for you?" for warehouse.
    target_role_text: str | None = None
    # Sprint 1: post-suppression view of the user's parsed resume. Carried
    # on every turn (not just the PRESENT_RESUME_FACTS one) so the
    # responder can quote resume context when narrating matches in item 7.
    # None when the user hasn't uploaded a resume.
    resume_facts: dict[str, Any] | None = None


def compose_reply(inp: ResponderInput) -> str:
    """Compose an assistant message. LLM-first, deterministic fallback.

    Sub-step 4: builds the SanitizedResponderView at entry (before
    is_enabled()) so the LLM-disabled fallback path reads URLs from the
    same sanitized projection as the LLM path. View is constructed
    once per turn and threaded to every downstream helper.
    """
    view = build_sanitized_responder_view_v1(inp)
    if not is_enabled():
        return _fallback_reply(inp, view)

    user_block = _build_user_block(inp, view)
    reply = call(NEXT_ACTION_RESPONDER_PROMPT, user_block, max_tokens=500)
    if not reply:
        return _fallback_reply(inp, view)

    if not _policy_ok(reply, inp, view):
        log.warning("Responder reply failed policy check; falling back")
        return _fallback_reply(inp, view)
    return reply


# =========================================================================
# User block — structured payload the prompt narrates from
# =========================================================================
_NARRATION_SKILL_CAP = 3


def _narration_skill_view(skills: list[str] | None) -> list[str]:
    """Build the narration view of matched_skills / missing_skills.

    The matcher considers up to top-N (12) skills per JD plus credential
    carve-outs, but narrating all of them turns the chat into a checklist.
    Cap at `_NARRATION_SKILL_CAP` (3) by the engine's existing order
    (rank-then-confidence), and FORCE-include any credentials further
    down the list so the responder can always cite a missing Class G or
    WHMIS regardless of where it ranked.

    Sprint 5 slice 4c -- "matcher honest, chat clean" contract.
    """
    if not skills:
        return []
    top = list(skills[:_NARRATION_SKILL_CAP])
    seen = set(top)
    credentials_below_cap = [
        s for s in skills[_NARRATION_SKILL_CAP:]
        if is_credential_skill_name(s) and s not in seen
    ]
    return top + credentials_below_cap


def _narration_skill_view_with_indices(
    skills: list[str] | None,
) -> tuple[list[str], list[int]]:
    """Same logic as `_narration_skill_view` but also returns the list
    of source indices that were kept. Callers can use the index list
    to slice parallel arrays (strengths, stages) and stay aligned.

    Used by `_capped_score_explanation` to cap required_matched and
    its parallel required_match_stages / required_match_strengths
    arrays consistently.
    """
    if not skills:
        return [], []
    kept_indices = list(range(min(_NARRATION_SKILL_CAP, len(skills))))
    seen = set(skills[:_NARRATION_SKILL_CAP])
    for i in range(_NARRATION_SKILL_CAP, len(skills)):
        name = skills[i]
        if is_credential_skill_name(name) and name not in seen:
            kept_indices.append(i)
            seen.add(name)
    return [skills[i] for i in kept_indices], kept_indices


def _capped_score_explanation(
    se: dict | None,
) -> dict | None:
    """Step 6 review fix: cap the skill lists INSIDE score_explanation
    too -- not just the top-level matched/missing.

    Pre-fix the engine output's score_explanation.required_matched
    could be 12+ entries while the top-level matched_skills was 3.
    The LLM sees both, so the prompt cap was a leaky contract -- a
    chat could narrate the long uncapped list from inside
    score_explanation. This helper applies the same narration cap to
    every skill-name list in score_explanation, AND slices the parallel
    *_match_strengths / *_match_stages arrays to the same positions
    so they stay index-aligned.

    Engine output is unchanged (full data still in the engine's
    MatchResult.score_explanation); only the responder projection
    sees the capped view.

    Returns a shallow-copied dict with the lists replaced; falls back
    to the original when input is empty/None.
    """
    if not se:
        return se
    out = dict(se)  # shallow copy; we only mutate top-level list-typed keys

    # Top-level matched / missing duplicates -- cap as plain lists.
    if "matched_skills" in out:
        out["matched_skills"] = _narration_skill_view(out["matched_skills"])
    if "missing_skills" in out:
        out["missing_skills"] = _narration_skill_view(out["missing_skills"])
    if "required_missing" in out:
        out["required_missing"] = _narration_skill_view(out["required_missing"])
    if "preferred_missing" in out:
        out["preferred_missing"] = _narration_skill_view(out["preferred_missing"])

    # Matched lists with parallel arrays (strengths + stages). Cap them
    # using the kept-indices variant so we can slice the parallels.
    for prefix in ("required", "preferred"):
        matched_key = f"{prefix}_matched"
        if matched_key not in out:
            continue
        capped_names, kept_idx = _narration_skill_view_with_indices(
            out[matched_key],
        )
        out[matched_key] = capped_names
        for parallel in ("_match_strengths", "_match_stages"):
            pkey = f"{prefix}{parallel}"
            arr = out.get(pkey) or []
            if arr:
                out[pkey] = [arr[i] for i in kept_idx if i < len(arr)]

    return out


def _build_user_block(
    inp: ResponderInput, view: SanitizedResponderView,
) -> str:
    """Sub-step 4: reads RESULTS and TRAINING from the sanitized view's
    projected items via the v1 serializers. Non-URL payload fields
    (NEXT_ACTION, ROLE_CATEGORY, etc.) continue to read from inp.
    """
    d = inp.decision
    role_category = intake_priority.classify_role(inp.target_role_text)

    parts: list[str] = []
    parts.append(f"USER_MESSAGE:\n{inp.user_message}\n")
    parts.append(f"NEXT_ACTION: {d.action}")
    parts.append(f"ROLE_CATEGORY: {role_category}")
    if d.redirect_reason:
        parts.append(f"REDIRECT_REASON: {d.redirect_reason}")

    # Single slot per turn. The prompt_hint carries the role-aware examples
    # the LLM should weave into a natural question.
    if d.ask_slots:
        slot = d.ask_slots[0]
        hint = intake_priority.prompt_hint(slot, inp.target_role_text)
        parts.append("ASK_SLOT:\n" + json.dumps({"slot": slot, "prompt_hint": hint}))

    parts.append(f"BAND_SIGNAL: {inp.band_signal}")

    if d.show_matches and view.prompt_results:
        parts.append("RESULTS:")
        for r in view.prompt_results:
            parts.append(json.dumps(serialize_result_for_v1_prompt(r)))
    elif d.show_matches:
        parts.append("RESULTS:\n(no eligible matches in current SSM dataset)")

    if d.show_matches and view.prompt_present_matches_training_flat:
        parts.append("TRAINING:")
        for t in view.prompt_present_matches_training_flat:
            parts.append(json.dumps(serialize_training_for_v1_prompt(t)))

    # NEXT_SKILL hint is match-context; only surface it to the LLM when
    # we're actually presenting matches. Otherwise it can bleed into
    # ASK_QUESTIONS / CONFIRM_READY turns ("if you build X, N more jobs
    # could open") before the user has agreed to see matches.
    if d.show_matches:
        skill, count = inp.next_skill
        if skill and count:
            parts.append(
                f"NEXT_SKILL: {skill} would unlock {count} more current SSM jobs."
            )

    if inp.requires_consent and d.action == ACTION_PRESENT_MATCHES:
        parts.append(
            "CONSENT_STATE: anonymous. The user has NOT consented to save "
            "their profile. Show the matches but do not promise to remember "
            "across sessions."
        )

    # RESUME_FACTS block: surface the post-suppression view so the LLM can
    # quote concrete entries on every turn — confirmation summary AND
    # later PRESENT_MATCHES narration (e.g. "your forklift cert from
    # Acme lines up with..."). Always sent when the user has uploaded
    # a resume, so the LLM keeps consistent context across the chat.
    if inp.resume_facts and _resume_facts_summary_has_content(inp.resume_facts):
        parts.append("RESUME_FACTS:\n" + _resume_facts_summary_for_prompt(inp.resume_facts))

    return "\n".join(parts)


def _resume_facts_summary_has_content(facts: dict[str, Any]) -> bool:
    """True if the facts have at least one entry worth showing the LLM."""
    return any(
        bool(facts.get(group))
        for group in ("work_history", "education", "certifications", "skills", "languages")
    )


def _resume_facts_summary_for_prompt(facts: dict[str, Any]) -> str:
    """Compact, prompt-friendly view of the parsed resume.

    Includes only what the responder needs to acknowledge: titles,
    employers, dates, credentials, skill names. Evidence strings are
    dropped (they bloat the prompt and the LLM should not quote
    verbatim resume text in the chat anyway — that risks leaking PII
    if the resume contains contact info in section text).
    """
    blob: dict[str, Any] = {}

    work = facts.get("work_history") or []
    if work:
        blob["work_history"] = [
            {
                "title": w.get("title"),
                "employer": w.get("employer"),
                "start_year": w.get("start_year"),
                "end_year": w.get("end_year"),
                "is_current": w.get("is_current"),
            }
            for w in work
            if isinstance(w, dict)
        ]

    edu = facts.get("education") or []
    if edu:
        blob["education"] = [
            {
                "credential": e.get("credential"),
                "institution": e.get("institution"),
                "year": e.get("year"),
            }
            for e in edu
            if isinstance(e, dict)
        ]

    certs = facts.get("certifications") or []
    if certs:
        blob["certifications"] = [
            {"name": c.get("name"), "issuer": c.get("issuer"), "year": c.get("year")}
            for c in certs
            if isinstance(c, dict)
        ]

    skills = facts.get("skills") or []
    if skills:
        # Skill names only (no fact_ids in prompt — those are backend bookkeeping).
        blob["skills"] = [s.get("name") for s in skills if isinstance(s, dict) and s.get("name")]

    langs = facts.get("languages") or []
    if langs:
        blob["languages"] = list(langs)

    return json.dumps(blob)


# =========================================================================
# Sprint 3 — scope-boundary patterns
# =========================================================================
#
# These regexes block specific failure modes we observed in live chats.
# They are intentionally action-context-shaped: the goal is to catch
# the LLM OFFERING to do something out-of-scope, not to catch every
# mention of a city name (which would false-positive on legitimate
# RESUME_FACTS references like "University of Toronto").

# Out-of-region offers — generic phrasing.
_OUT_OF_REGION_PATTERNS = (
    re.compile(r"\bnearby (cit(?:y|ies)|towns?|regions?|areas?)\b", re.I),
    re.compile(r"\b(another|other|different) (cit(?:y|ies)|towns?|regions?)\b", re.I),
    re.compile(r"\b(broaden|broader|expand|widen)\b.*\b(search|net|geography|area)\b", re.I),
    re.compile(r"\bcheck (elsewhere|other (cit(?:y|ies)|places|areas|regions))\b", re.I),
    re.compile(r"\blook (elsewhere|outside (the )?(sault|ssm|algoma))\b", re.I),
    re.compile(r"\boutside (the )?(sault ste\.? marie|sault|ssm|algoma)\b", re.I),
    re.compile(r"\bsearch (in|across) other (cit|region|area|place)", re.I),
)

# Non-local Ontario cities that occasionally surface as recommendations.
# We only block when one appears with an *action* verb that implies
# "go look there" — "in Toronto", "try Sudbury", "consider Ottawa", etc.
# Bare references like "University of Toronto" pass through.
_NON_LOCAL_CITY = (
    r"(toronto|ottawa|sudbury|thunder bay|north bay|timmins|kingston|"
    r"hamilton|kitchener|mississauga|london|windsor|waterloo|barrie|"
    r"greater toronto|gta)"
)
_NON_LOCAL_CITY_OFFER_PATTERNS = (
    # Action verbs (with -ed / -ing variants) directly before a city.
    re.compile(rf"\btr(?:y|ied|ying)\s+{_NON_LOCAL_CITY}\b", re.I),
    re.compile(rf"\bconsider(?:ed|ing)?\s+{_NON_LOCAL_CITY}\b", re.I),
    re.compile(rf"\bcheck(?:ed|ing)?\s+{_NON_LOCAL_CITY}\b", re.I),
    re.compile(rf"\bsearch(?:ed|ing)?(?:\s+in)?\s+{_NON_LOCAL_CITY}\b", re.I),
    re.compile(rf"\blook(?:ed|ing)?\s+(?:in|at|for(?:\s+jobs?\s+in)?)\s+{_NON_LOCAL_CITY}\b", re.I),
    # Prepositional context: "in/to/near + city + presence/availability".
    re.compile(rf"\b(in|to|near|around)\s+{_NON_LOCAL_CITY}\b.*\b(might|may|could|has|offers|hiring|opportunities|jobs?)\b", re.I),
    # City-first construction: "Toronto might/has/offers …".
    re.compile(rf"\b{_NON_LOCAL_CITY}\s+(might|may|could|has|offers)\b.*\b(more|better|opportunities|jobs?|hiring)\b", re.I),
    # "hiring/jobs in <city>".
    re.compile(rf"\b(hiring|jobs?|opportunities|openings?) in {_NON_LOCAL_CITY}\b", re.I),
    # Relocation suggestions.
    re.compile(rf"\bmove (to|toward) {_NON_LOCAL_CITY}\b", re.I),
)

# Credential equivalence — WES territory.
_CREDENTIAL_EQUIVALENCE_PATTERNS = (
    re.compile(r"\b(equivalent|equivalence|equates?) (to|with) (a |an )?canadian\b", re.I),
    re.compile(r"\bcanadian (equivalent|equivalence)\b", re.I),
    re.compile(r"\bwes (certification|evaluation|assessment|process|report)\b", re.I),
    re.compile(r"\bcredential (equivalence|recognition|conversion)\b", re.I),
    re.compile(r"\b(your|the) (degree|diploma|credential) (counts|qualifies) as\b", re.I),
)

# Immigration / legal scope — specific programs and procedural language.
# RCIP is a designated SSM employer pathway, so mentioning the program
# name is OK (e.g. "RCIP-designated employers") — only eligibility /
# application advice is blocked.
# Post-step-11 ownership cleanup: the conservative training-provider
# deny list and abbreviation table now live in their own dependency-
# neutral module so both this path and the new tiered-matches policy
# path (`coach_tiers_policy.py`) import from the same source of truth.
from skillbridge.chat.training_provider_registry import (
    _KNOWN_TRAINING_PROVIDERS,
    _PROVIDER_ABBREVIATIONS,
)


def _check_ungrounded_provider(
    reply: str, training_by_job: dict[str, list[dict]],
) -> str | None:
    """Return the first provider name from the deny-list that appears
    in `reply` AND is NOT present in this turn's TRAINING block.

    Returns None when every provider mention is grounded (i.e. appears
    in the TRAINING block) or when no provider from the deny-list is
    mentioned at all.

    Abbreviation-aware: if a canonical name is grounded, its registered
    abbreviations (`_PROVIDER_ABBREVIATIONS`) are ALSO treated as
    grounded so the LLM can naturally write "SCCC" when TRAINING
    carries "Sault Community Career Centre". The grounding contract
    stays the same — the canonical name must actually be in TRAINING
    for the abbreviation to count.
    """
    # Set of provider names ACTUALLY in this turn's TRAINING block
    grounded: set[str] = set()
    for entries in (training_by_job or {}).values():
        for entry in entries:
            p = entry.get("provider", "")
            if isinstance(p, str) and p.strip():
                grounded.add(p.strip().lower())

    # Slice (2026-06-08) abbreviation expansion: for every grounded
    # canonical name we know shorthand for, add its abbreviations to
    # the grounded set. SCCC alone is the live-observed case; the table
    # is structured for future additions.
    for canonical in list(grounded):
        for abbr in _PROVIDER_ABBREVIATIONS.get(canonical, frozenset()):
            grounded.add(abbr)

    reply_lower = reply.lower()
    for provider in _KNOWN_TRAINING_PROVIDERS:
        # Word-boundary match to avoid "tac" matching inside "tactical".
        # re.escape handles periods in "st. john ambulance".
        pattern = rf"\b{re.escape(provider)}\b"
        if re.search(pattern, reply_lower):
            if provider not in grounded:
                return provider
    return None


_IMMIGRATION_LEGAL_PATTERNS = (
    re.compile(r"\bexpress entry\b", re.I),
    re.compile(r"\brcip (eligibility|application|requirements|criteria|process)\b", re.I),
    re.compile(r"\b(work permit|pr application|permanent residence|ircc|pnp)\b", re.I),
    re.compile(r"\byou (may|might|could) (qualify|be eligible) (for|under)\b.*\b(express entry|rcip|pnp|pr)\b", re.I),
    re.compile(r"\bconsult (a |an |your |with a )?(lawyer|attorney|immigration consultant)\b", re.I),
)


# Slice N (2026-06-05): patterns that MUST NOT appear in present_near_miss
# replies. A near-miss is "the role exists but you're not ready" -- calling
# it a match, fit, or stretch misrepresents the gap. Locked in the design
# doc's responder CAN/CANNOT contract; the prompt teaches the same
# constraint, and this regex catches the LLM violating it.
# R-5: patterns the LLM must NOT use on explain_remaining_gaps turns.
# The payload supplies gap NAMES only; any sentence speculating about
# how those gaps are typically closed is invented content (no TRAINING
# block backs it, no registry resource grounds the claim).
_REMAINING_GAPS_SPECULATION_PATTERNS = (
    # "usually come on the job", "typically come with experience"
    re.compile(
        r"\b(?:usually|typically|most often|generally)\s+come(?:s)?\s+"
        r"(?:on the job|with experience|with time|through)",
        re.I,
    ),
    # "best learned through", "best learned at", "learn(ed) on the job"
    re.compile(
        r"\bbest\s+(?:learned|developed|earned)\s+(?:through|at|on|in)",
        re.I,
    ),
    re.compile(r"\blearn(?:ed)?\s+on\s+the\s+job\b", re.I),
    # "usually a course", "typically a course", "usually takes a course"
    re.compile(
        r"\b(?:usually|typically|often)\s+(?:\w+\s+)?(?:a|an)\s+"
        r"(?:course|class|program|certification)",
        re.I,
    ),
    # "comes with time", "comes with experience"
    re.compile(r"\bcomes?\s+with\s+(?:time|experience|practice)\b", re.I),
    # "you'll pick that up at", "you'll learn that on the job"
    re.compile(
        r"\byou'?ll\s+(?:pick that up|learn that|develop that)\s+"
        r"(?:at|on|in|through)",
        re.I,
    ),
)


_NEAR_MISS_FORBIDDEN_PATTERNS = (
    # "good fit", "great match", "perfect match", "strong fit", etc.
    re.compile(r"\b(good|great|perfect|strong|solid) (fit|match)\b", re.I),
    # "you qualify", "you do qualify", "you would qualify"
    re.compile(r"\byou (do |would |may |might )?qualify\b", re.I),
    # "you're qualified", "you are qualified"
    re.compile(r"\byou'?re (qualified|a qualified|currently qualified)\b", re.I),
    re.compile(r"\byou are qualified\b", re.I),
    # "stretch match" -- distinct band; reserved for actual stretch-band results
    re.compile(r"\bstretch match\b", re.I),
    # "perfect for you" / "great for you" / "ideal for you" -- the
    # locked v11 §"Forbidden vocabulary" list explicitly names
    # "perfect for you" and "ideal role for you" as banned framings
    # for adjacency (it's eligibility-by-credential, NOT match-
    # quality certification).
    re.compile(
        r"\b(perfect|great|ideal)( (role|fit|match))? for you\b",
        re.I,
    ),
    # Candidate certification -- SEMANTIC rule, not grammar-by-grammar.
    # Adjacency, near-miss, and remaining-gaps outcomes all forbid
    # certifying the user as a candidate. Rather than chasing every
    # modal / contraction / perception verb / adverb-insertion
    # bypass ("you'd be", "you could potentially be", "you may well
    # be", "you appear likely to be", etc.), reject the prohibited
    # noun-phrase outright: any "<positive-adjective> candidate"
    # framing is unnecessary on these outcomes, so the broader rule
    # is safer and simpler. See AR-6c review round 6.
    re.compile(
        r"\b(?:strong|good|great|perfect|excellent|ideal)\s+candidate\b",
        re.I,
    ),
)


# =========================================================================
# Policy check on LLM output
# =========================================================================
def _policy_ok(
    reply: str, inp: ResponderInput, view: SanitizedResponderView,
) -> bool:
    """Cheap heuristic policy sweep. Returns False if reply is suspect.

    Sub-step 5: URL grounding check is active. Every URL in the reply
    must canonicalize to a member of view.prompt_urls (move-gated
    allowlist). Telemetry emitted via safe_telemetry_fields.
    """
    if not reply.strip():
        return False

    for candidate in extract_url_candidates(reply):
        result = check_url_membership(
            candidate.extracted_token, view.prompt_urls,
        )
        if isinstance(result, Violation):
            move_for_telemetry = getattr(inp.decision, "action", "unknown")
            fields = safe_telemetry_fields(
                result, move=str(move_for_telemetry),
            )
            log.warning(
                "policy: URL grounding violation "
                "code=%s move=%s scheme=%s host=%s hash=%s",
                fields["violation_code"], fields["move"],
                fields["scheme"], fields["host"], fields["url_hash"],
            )
            return False

    lower = reply.lower()

    if inp.requires_consent:
        bad_promises = (
            "i'll remember", "i will remember", "i'll save your profile",
            "i'll keep this for you", "next time i'll",
        )
        if any(p in lower for p in bad_promises):
            return False

    # No fabricated salary / hourly rate.
    if "$" in reply or "/hr" in reply or "/hour" in reply.lower():
        return False

    # No national-feed language.
    forbidden = ("job bank", "statistics canada", "statcan", "national average")
    if any(p in lower for p in forbidden):
        return False

    # Sprint 3 — out-of-region offers and non-local city recommendations.
    # Both block the same failure: implying we can search outside SSM
    # (we can't). RESUME_FACTS references to non-local cities pass
    # through because the offer patterns require an action verb.
    for pat in _OUT_OF_REGION_PATTERNS:
        if pat.search(reply):
            log.warning("policy: reply offers out-of-region search (pattern=%s)", pat.pattern)
            return False
    for pat in _NON_LOCAL_CITY_OFFER_PATTERNS:
        if pat.search(reply):
            log.warning("policy: reply suggests non-local city (pattern=%s)", pat.pattern)
            return False

    # Sprint 3 — credential equivalence is WES territory, not ours.
    for pat in _CREDENTIAL_EQUIVALENCE_PATTERNS:
        if pat.search(reply):
            log.warning("policy: reply makes credential equivalence claim")
            return False

    # Sprint 3 — immigration / legal / consultant-tier advice is
    # out-of-scope.
    for pat in _IMMIGRATION_LEGAL_PATTERNS:
        if pat.search(reply):
            log.warning("policy: reply gives immigration/legal-tier advice")
            return False

    # Hard policy: no bullet-list questions in ASK_QUESTIONS turns. One
    # bullet line is enough to declare it a checklist — we explicitly want
    # all intake questions woven into prose, never bulleted.
    if inp.decision.action == ACTION_ASK_QUESTIONS:
        bullet_lines = sum(
            1 for ln in reply.splitlines()
            if ln.lstrip().startswith(("•", "- ", "* ", "1.", "2.", "3."))
        )
        if bullet_lines >= 1:
            return False

    return True


# =========================================================================
# Deterministic fallback — single conversational question, never bullets
# =========================================================================
def _fallback_reply(
    inp: ResponderInput, view: SanitizedResponderView,
) -> str:
    """Sub-step 4: dispatcher threads `view` to the URL-bearing
    callee `_present_matches_fallback`. Other branches are URL-free.
    """
    d = inp.decision
    target = inp.target_role_text

    if d.action == ACTION_REDIRECT:
        line = (
            "I'm focused on helping you find work in Sault Ste. Marie. "
            "I can match you to local jobs and suggest skills to build."
        )
        if d.ask_slots:
            line += " " + _single_ask(d.ask_slots[0], target)
        return line

    if d.action == ACTION_ACKNOWLEDGE_AND_WAIT:
        line = "No problem — we can skip that."
        if d.ask_slots:
            line += " " + _single_ask(d.ask_slots[0], target)
        return line

    if d.action == ACTION_ASK_QUESTIONS:
        if d.ask_slots:
            return _single_ask(d.ask_slots[0], target)
        return "Tell me a bit more about what you're looking for."

    if d.action == ACTION_CONFIRM_READY:
        line = "Thanks — I have enough to start matching against current Sault Ste. Marie jobs."
        if d.ask_slots:
            hint = intake_priority.prompt_hint(d.ask_slots[0], target)
            line += f" One quick thing first: {hint}."
        line += " Want me to show you what I've got?"
        return line

    if d.action == ACTION_PRESENT_RESUME_FACTS:
        return _present_resume_facts_fallback(inp)

    # ACTION_PRESENT_MATCHES
    return _present_matches_fallback(inp, view)


def _present_resume_facts_fallback(inp: ResponderInput) -> str:
    """Deterministic resume-parsed summary when the LLM is off or fails.

    Reads from inp.resume_facts (post-suppression view). Mentions up to a
    few work entries + the credential + the top skills. No bullets.

    Updated 2026-06-29 (resume-confirm gate removal): does NOT ask for
    confirmation of parsed facts. Conditional close:
      - If inp.target_role_text is missing/empty: end with
        "What kind of work are you looking for right now?"
      - If inp.target_role_text is set: end with no question. The
        user drives the next turn.
    """
    facts = inp.resume_facts or {}
    work = facts.get("work_history") or []
    edu = facts.get("education") or []
    skills = facts.get("skills") or []

    pieces: list[str] = ["Thanks — I read your resume. Here's what stood out:"]

    # Most recent work entry (assume order is reverse-chronological).
    if work:
        w = work[0]
        title = (w.get("title") or "").strip()
        employer = (w.get("employer") or "").strip()
        sy, ey, current = w.get("start_year"), w.get("end_year"), w.get("is_current")
        years = ""
        if isinstance(sy, int) and current:
            years = f" {sy}-present"
        elif isinstance(sy, int) and isinstance(ey, int):
            years = f" {sy}-{ey}"
        elif isinstance(sy, int):
            years = f" {sy}"
        head = " at ".join(p for p in (title, employer) if p) or title or employer
        if head:
            pieces.append(f"{head}{years}.")

    # Top credential
    if edu:
        e = edu[0]
        cred = (e.get("credential") or "").strip()
        inst = (e.get("institution") or "").strip()
        if cred and inst:
            pieces.append(f"{cred} from {inst}.")
        elif cred:
            pieces.append(f"{cred}.")

    # Skills (up to 6, no bullets — comma-separated)
    if skills:
        names = [s.get("name") for s in skills[:6] if isinstance(s, dict) and s.get("name")]
        if names:
            pieces.append("Skills I picked up: " + ", ".join(names) + ".")

    if len(pieces) == 1:
        # Nothing parsed worth mentioning beyond the intro — bail to a
        # neutral prompt.
        return (
            "I read your resume but couldn't pull much from it. "
            "Could you tell me a bit about your background — what kind of "
            "work have you done?"
        )

    # Resume-confirm gate removed 2026-06-29. Conditional close based
    # on whether target is already known.
    target = (getattr(inp, "target_role_text", None) or "").strip()
    if not target:
        pieces.append(
            "What kind of work are you looking for right now?"
        )
    # else: target is set -- no question; user drives the next turn.
    return " ".join(pieces)


def _single_ask(slot: str, target_role_text: str | None) -> str:
    """One conversational question, no bullets, role-aware examples."""
    hint = intake_priority.prompt_hint(slot, target_role_text)
    # Capitalise the first character; turn "what kind of..." into "What kind of...?"
    capitalised = hint[:1].upper() + hint[1:] if hint else hint
    return f"{capitalised}?"


def _present_matches_fallback(
    inp: ResponderInput, view: SanitizedResponderView,
) -> str:
    """Sub-step 4: reads results + per-job training URLs from
    view.fallback_results and view.fallback_present_matches_training_by_job.
    URL rendering uses SanitizedURL.raw; invalid source URLs are stripped
    at projection.
    """
    results = view.fallback_results
    if inp.band_signal == "none" or not results:
        skill, count = inp.next_skill
        msg = (
            "I don't see a strong match in the current Sault Ste. Marie "
            "jobs yet. "
        )
        if skill and count:
            msg += (
                f"If you build {skill}, around {count} more current jobs "
                "could open up. "
            )
        msg += (
            "I'd recommend contacting Sault Community Career Centre for "
            "one-on-one help."
        )
        return msg

    lead = (
        "Here are the most relevant Sault Ste. Marie jobs I can see right now:"
        if inp.band_signal != "stretch_only"
        else "I don't see a strong match today, but here are some stretch matches worth considering:"
    )
    lines = [lead]
    training_by_job = view.fallback_present_matches_training_by_job
    for r in results:
        lines.append(
            f"• {r.title or 'this role'} at "
            f"{r.employer or 'employer unspecified'} — "
            f"{r.match_band or 'match'} match."
        )
        if r.url is not None:
            lines.append(f"   {r.url.raw}")
        if r.credential_warning:
            lines.append(f"   Note: {r.credential_warning}")
        if r.missing_skills:
            lines.append(
                f"   Skills to build: {', '.join(r.missing_skills[:3])}"
            )
        if r.job_id:
            for t in training_by_job.get(r.job_id, ()):
                tline = f"   Try: {t.title}"
                if t.provider:
                    tline += f" ({t.provider})"
                if t.url is not None:
                    tline += f" — {t.url.raw}"
                lines.append(tline)
    skill, count = inp.next_skill
    if skill and count:
        lines.append(
            f"\nIf you build {skill}, around {count} more current jobs "
            "could open up."
        )
    return "\n".join(lines)


# =========================================================================
# Chat orchestration v2 -- outcome-move responder (Slice 5)
# =========================================================================
# Consumes an ArbiterDecision from arbiter.py instead of a legacy
# state-machine Decision. The narrow input surface is the whole point:
# operational fields (arbiter_action, notes) MUST NOT enter the LLM
# prompt -- they are telemetry, not user-facing text. The
# `_build_user_block_v2` function whitelists exactly the fields the
# responder is allowed to narrate from. Tests assert the operational
# fields don't leak.
#
# Slice 5 ships planner.py + arbiter.py + this responder as DEAD CODE.
# Slice 6 will toggle between v1 (`compose_reply`) and v2
# (`compose_response_v2`) via the CHAT_ORCHESTRATOR env flag.
from skillbridge.chat.arbiter import ArbiterDecision
from skillbridge.chat.prompts import OUTCOME_RESPONDER_PROMPT


@dataclass(frozen=True)
class ConversationContext:
    """Compact short-session memory for the responder (Slice 8).

    Built from the staged profile each turn. Carries:
      - target_role_text: the user's stated job target, if known
      - last_presented_job_titles: titles shown on the most recent
        present_matches turn (empty if matches were never shown or the
        last v2 turn cleared them)
      - last_presented_caps_applied: caps that fired on the last match
        turn (flag-shaped, e.g. "band_capped_by_credential")
      - last_presented_credential_gaps: human-readable specific
        credential names that were missing (e.g. "310T technician
        certification")

    Used primarily by the deterministic fallback paths in
    `_fallback_reply_v2` -- the LLM happy path already weaves context
    on its own from RESULTS / RESUME_FACTS, so we don't surface this
    structure into the prompt yet. Slice 8 fixes the FALLBACK path,
    which is what users see when the LLM's reply fails the policy
    check.
    """
    target_role_text: str | None = None
    last_presented_job_titles: tuple[str, ...] = ()
    last_presented_caps_applied: tuple[str, ...] = ()
    last_presented_credential_gaps: tuple[str, ...] = ()

    def has_presented_context(self) -> bool:
        """True when at least one job was shown to the user previously.
        Drives whether the fallback can reference "those roles we just
        looked at" vs starting cold."""
        return bool(self.last_presented_job_titles)


@dataclass
class ResponderV2Input:
    """v2 responder input. Carries an `ArbiterDecision` instead of the
    legacy `Decision`. Only a subset of decision fields ever reach the
    prompt -- see `_build_user_block_v2` for the whitelist."""
    user_message: str
    decision: ArbiterDecision
    results: list[dict]
    training_by_job: dict[str, list[dict]]
    next_skill: tuple[str | None, int]
    # "strong_or_good" | "stretch_only" | "low_only" | "none".
    # "low_only" means engine found eligible candidates but all low-band;
    # results is empty and the responder narrates the no-match shape.
    band_signal: str
    requires_consent: bool
    target_role_text: str | None = None
    resume_facts: dict[str, Any] | None = None
    # Slice 8: short-session context used by fallback paths. None when
    # the handler hasn't built one (e.g. v1 callers, tests).
    conversation_context: ConversationContext | None = None
    # Slice N (2026-06-05): present_near_miss payload. Populated by the
    # handler ONLY when the arbiter resolves to present_near_miss --
    # carries the role name + employer + classified+capped gap lists
    # the responder narrates. None on any other final_move. The handler
    # owns the cap-3+3 + credential-first ordering; the responder is a
    # pure renderer.
    #
    # Expected shape when set:
    #   {
    #     "role": "Truck and Coach Technician",   # str, never None
    #     "employer": "Garden River First Nation",# str | None
    #     "job_count": 1,                         # int >= 1
    #     "credential_gaps": ["310T cert...", "Class G ..."],   # 0-3 items
    #     "core_skill_gaps": ["emergency repair", ...],         # 0-3 items
    #   }
    #
    # `training_by_job` (existing field) carries provider info when the
    # registry knew the credential; the fallback uses it to name
    # providers verbatim, never invented.
    near_miss_payload: dict[str, Any] | None = None
    # R-4 (remaining-gaps iteration): payload for explain_remaining_gaps
    # turns. Populated by the handler ONLY when the synthesis hook fires
    # for kind="subtract" or kind="retract". See
    # docs/remaining-gaps-design.md §9 for the structured REMAINING_GAPS
    # block shape.
    #
    # Expected shape when set:
    #   {
    #     "role": "310S Licensed Automotive Technician",
    #     "employer": "Great Lakes Honda" | None,
    #     "assumed_completed_credentials": [
    #       {"display": "...", "canonical": "...", "mode": "claimed"|"hypothetical"},
    #     ],
    #     "remaining_credentials": [
    #       {"display": "...", "canonical": "..."},
    #     ],
    #     "remaining_core_skills": ["Honda vehicle experience", ...],
    #     "any_hypothetical": bool,
    #   }
    #
    # `training_by_job` (existing field) is regrounded by the handler
    # for the LEAD remaining credential on this turn so the responder
    # can name providers verbatim.
    remaining_gaps_payload: dict[str, Any] | None = None
    # R-5 (remaining-gaps iteration): clarification payload for
    # ask_one_clarifying_question turns synthesized by the
    # remaining-gaps handler (kind="confirm" or kind="bootstrap").
    # When set, `compose_response_v2` EARLY-RETURNS the templated
    # text without calling the LLM -- the templates are trusted by
    # construction (no provider names, no URLs, no scope-violation
    # content, no "you qualify" framing). Round-9 design §11.
    #
    # Expected shapes (discriminated by "kind"):
    #   {
    #     "kind": "credential_completion_confirmation",
    #     "credential_canonical": "310S ..." | None,
    #     "credential_display":   "310S licence" | "",
    #     "action": "add" | "remove",
    #   }
    #   {
    #     "kind": "bootstrap_match_request",
    #   }
    clarification_payload: dict[str, Any] | None = None
    # AR-6c (adjacent-recommendations v11):
    # `adjacent_recommendations_payload` is populated by the handler
    # ONLY when `_try_adjacency_dispatch` synthesizes
    # `recommend_adjacent_roles`. Shape (locked v11 §"Locked StagedProfile
    # / ResponderV2Input additions"):
    #   {
    #     "recommendations": [
    #       {"job_id": str, "title": str, "employer": str | None,
    #        "location": str, "evidence_summary": str,
    #        "why_adjacent": "same_noc_minor_group" | "skill_evidence",
    #        "matched_skills": list[str]},
    #       ...                                # max 3 items
    #     ],
    #     "total_retrieved": int,
    #     "total_dropped_by_credential_gap": int,
    #     "total_dropped_by_coverage_floor": int,
    #     "total_dropped_by_transferable_floor": int,
    #     "total_dropped_by_no_required_non_credential_skills": int,
    #   }
    # The LLM uses the `recommendations` list VERBATIM (titles,
    # employers, evidence summaries). Empty list -> deterministic
    # provider-free fallback line.
    adjacent_recommendations_payload: dict[str, Any] | None = None
    # AR-6c: `adjacent_role_description_payload` is populated by the
    # handler ONLY when `_try_adjacency_dispatch` synthesizes
    # `describe_adjacent_role`. The handler re-fetches the job by id
    # from `core.v_current_job` and combines with the snapshot's
    # evidence. Shape:
    #   {
    #     "job":  {"title": str, "employer": str | None,
    #              "location": str | None, "url": str | None,
    #              "posted_date": str | date | None} | None,
    #     "evidence_summary": str,
    #     "matched_skills": list[str],
    #     "expired": bool,
    #   }
    # When `expired=True` the deterministic fallback renders "that
    # role's no longer on the board" without naming a provider.
    adjacent_role_description_payload: dict[str, Any] | None = None
    # AR-9.feat.coach-tiers CP2 step 3: tiered-matches inputs. Populated
    # by the handler ONLY when final_move == "present_tiered_matches".
    # The legacy `results`/`training_by_job` fields are not consulted on
    # that branch; the tier_evidence + pipeline_snapshot pair carries
    # everything the new prompt and the deterministic fallback need.
    #
    # The arbiter dispatch in `resolve_match_outcome` requires the
    # handler to supply `tiered_evidence_available=True` BEFORE
    # emitting `present_tiered_matches`. If the handler dispatches the
    # move without populating `tier_evidence` here, the responder
    # branch logs the contract violation and falls back to the empty-
    # state body (no LLM call).
    tier_evidence: TieredEvidence | None = None
    pipeline_snapshot: PipelineSnapshot | None = None
    # Resume-upload offer (2026-06-16). Handler sets True when:
    #   - final_move in {present_no_match} OR band_signal in
    #     {low_only, stretch_only}, AND
    #   - chat_skill_count < 5 (thin evidence), AND
    #   - resume_uploaded is False, AND
    #   - staged.resume_upload_offered is False (not already offered).
    # The responder uses this flag to weave a deterministic
    # "upload a CV/resume could unlock more matches" offer into the
    # no-match / low-band response. The handler also flips
    # staged.resume_upload_offered=True at the same time so the offer
    # doesn't re-fire on every subsequent thin-evidence turn.
    should_offer_resume_upload: bool = False
    # Step 11h (2026-06-17, closing-matrix v2): the CP4 primary
    # recommendation's canonical skill name when available, or None.
    # Populated by the handler before the responder is called, on
    # turns where the closing matrix can use it (currently:
    # `present_no_match` with resume_facts in play — the SHAPE 2 /
    # RELATED_ROLES_EXHAUSTED case, Movement C2 in
    # OUTCOME_RESPONDER_PROMPT). The LLM and the deterministic
    # fallback both quote this verbatim — no paraphrase, no
    # invention.
    cp4_primary_gap: str | None = None
    # Slice 5 step 4 (2026-06-19): conversational recommender payload.
    # Populated by the handler ONLY on chained recommender turns. When
    # set, compose_response_v2 EARLY-DISPATCHES to the recommender path
    # (the legacy v2 view builder + LLM happy path are skipped). One mode
    # per turn -- the wrapper's `mode` field discriminates which prompt
    # section the LLM activates. See
    # project_recommender_step4_implementation_lock memory.
    recommendation_evidence: RecommenderEvidence | None = None
    # Step 3 peer-engine wiring (2026-06-22): the original CareerIntent
    # that drove this turn's recommender dispatch. Plumbed for future
    # voice-branching in Layer B prompts (skill_gap vs training_
    # recommendation). Currently NOT read by the prompt; a follow-up
    # slice will wire it in. See project_recommender_peer_engine_locked.
    recommender_voice_hint: CareerIntent | None = None


def compose_response_v2(inp: ResponderV2Input) -> str:
    """Compose a v2 assistant message. LLM-first, deterministic fallback.

    Mirrors `compose_reply` but consumes an ArbiterDecision and feeds
    `OUTCOME_RESPONDER_PROMPT` instead of `NEXT_ACTION_RESPONDER_PROMPT`.
    Reuses the existing `_policy_ok` regex sweep -- those rules are
    output-level and apply to ANY chat reply regardless of which prompt
    produced it.

    When LLM is disabled OR the policy check fails, falls back to
    `_fallback_reply_v2` which maps each `OutcomeMove` to deterministic
    text.

    R-5 (remaining-gaps iteration): when `clarification_payload` is set
    the LLM is SKIPPED entirely -- the templated text is rendered
    deterministically. Policy regex is also bypassed because the
    templates are trusted by construction (no provider names, no URLs,
    no scope-violation content, no "you qualify" framing). See locked
    design §11 "trusted templated text" carve-out.
    """
    # ---- URL-free early returns (verified URL-free; above view-build) ----
    # Each branch below renders deterministic text without consulting any
    # URL-bearing payload, and does not call any of the bug.2a view-aware
    # helpers. The view-build below pays a small projection cost; we skip
    # it for these paths entirely.
    if inp.clarification_payload is not None:
        return _render_clarification(inp.clarification_payload)

    # Slice 5 step 4 (2026-06-19): conversational recommender. When the
    # handler has populated `recommendation_evidence`, the recommender
    # owns this turn end-to-end -- no legacy view build, no OUTCOME_
    # RESPONDER_PROMPT call. The local_gap_coach mode is the only one
    # that can carry TrainingResource URLs; those come from the
    # registry-verified wrapper and pass surface_url(today) by
    # construction.
    if inp.recommendation_evidence is not None:
        return _compose_recommender_response(inp)

    # Rule 3 (router): training_request without a registry entity gets
    # a deterministic static question rather than a role-aware skills
    # intake.
    if (
        inp.decision.final_move == "ask_one_clarifying_question"
        and inp.decision.reason_code == "training_request_no_entity"
    ):
        return _TRAINING_REQUEST_NO_ENTITY_QUESTION

    # Slice 6 (locked 2026-06-29, Option 1): LLM bypass for
    # present_no_match. The LLM has drifted repeatedly on this
    # surface ("I checked related roles" when recommender never
    # ran; "want training directions?" with no consume hook;
    # Sault College hallucinations caught by policy gate). The
    # deterministic _present_no_match_fallback_v2 is now the
    # SOURCE OF TRUTH for no_match responses, not just a safety
    # net. Skip the LLM entirely for this final_move.
    #
    # Resume-upload-offer branch within the fallback still fires
    # for no-resume cases (that's a distinct UX flow with its own
    # consume hook -- user uploads resume next turn).
    if inp.decision.final_move == "present_no_match":
        return _present_no_match_fallback_v2(inp)

    # AR-9.feat.coach-tiers CP2 step 3: tiered-matches surface uses its
    # own view builder, prompt, policy gate, and fallback. The
    # legacy v2 view builder is never invoked on this branch.
    if inp.decision.final_move == "present_tiered_matches":
        return _compose_tiered_matches_response(inp)

    # ---- Build view BEFORE any URL-bearing path ----
    view = build_sanitized_responder_view_v2(inp)

    # AR-8a: deterministic empty-adjacency. _recommend_adjacent_roles_fallback_v2
    # renders no URLs but its signature now requires view per the
    # locked sub-step 4 contract.
    if (
        inp.decision.final_move == "recommend_adjacent_roles"
        and not _valid_adjacent_recommendations(inp.adjacent_recommendations_payload)
    ):
        return _recommend_adjacent_roles_fallback_v2(inp, view)

    if not is_enabled():
        return _fallback_reply_v2(inp, view)

    user_block = _build_user_block_v2(inp, view)
    reply = call(OUTCOME_RESPONDER_PROMPT, user_block, max_tokens=500)
    if not reply:
        return _fallback_reply_v2(inp, view)

    if not _policy_ok_v2(reply, inp, view):
        log.warning("Responder v2 reply failed policy check; falling back")
        return _fallback_reply_v2(inp, view)
    return reply


# =========================================================================
# R-5 -- deterministic clarification renderer (LLM skipped, policy
# skipped; templates are trusted by construction per locked §11)
# =========================================================================
# R-5 round-3: strict allow-list validation, NOT deny-list stripping.
# Round-2's deny list let URI schemes (ftp://, javascript:, mailto:),
# bare domains (evil.ca/path), markup fragments, unknown provider names,
# and partial dollar/hour leakage through. The trusted-by-construction
# guarantee that justifies the policy bypass in compose_response_v2
# requires that NO unsafe substitution can reach the response. The
# correct contract is: validate the WHOLE string; reject it (return
# empty) on any suspect content; let the renderer fall back to the
# no-target "Which credential?" template. Falling back is safer than
# stripping fragments.
_DISPLAY_REJECT_PATTERNS = (
    # Any URI scheme: literal scheme:something. Catches http, https, ftp,
    # ftps, javascript, mailto, data, file, vbscript, and anything else
    # that looks like a scheme. Schemes are letters + digits / + . / -
    # per RFC 3986; we accept that shape but reject as soon as a colon
    # follows.
    re.compile(r"[A-Za-z][A-Za-z0-9+\-.]*:", re.I),
    # Bare domains: word.word with a 2-6 char TLD-like tail. Catches
    # "evil.ca/path", "drivetest.ca", "ontario.ca" with no protocol.
    # Credential displays don't legitimately contain TLD-shaped tokens.
    re.compile(r"\b[A-Za-z0-9-]+\.[A-Za-z]{2,6}\b", re.I),
    # Markup / interpolation delimiters: any HTML tag, markdown link,
    # bracketed reference, brace, backslash, or pipe. Credential
    # displays don't legitimately contain these.
    re.compile(r"[<>\[\]{}\\|`]"),
    # Email syntax (already partially caught by URI scheme via mailto:,
    # but explicitly reject @ symbols anywhere).
    re.compile(r"@"),
    # Any control character (we already filter these out below, but the
    # validation pass also rejects so the test of "the whole string is
    # safe" is decisive). Allowing tab/newline-class chars in a one-line
    # interpolation is never legitimate.
    re.compile(r"[\x00-\x1f\x7f]"),
    # Salary / hourly markers in ANY form: $ sign, "/hour", "/hr",
    # numeric per-hour patterns, "per hour" / "per year". A credential
    # display shouldn't quote rates.
    re.compile(r"\$"),
    re.compile(r"/\s*(?:hour|hr|yr|year)\b", re.I),
    re.compile(r"\b\d+\s*per\s+(?:hour|hr|yr|year|day|week|month)\b", re.I),
    # Scope-violation hooks the responder polices everywhere else.
    re.compile(
        r"\b(?:express entry|job bank|statistics canada|statcan|"
        r"national average|rcip|cra|cic|ircc|wes)\b",
        re.I,
    ),
    # Known training provider names (a credential DISPLAY is a gap name,
    # never a provider). The handler-side snapshot capture would never
    # produce these legitimately.
    re.compile(
        r"\b(?:drivetest|skilled\s+trades\s+ontario|sault\s+college|"
        r"sault\s+community\s+career\s+centre|sccc|wsib|ontario\s+public\s+service|"
        r"preytech)\b",
        re.I,
    ),
)


# Characters legitimately allowed in a credential display: letters,
# digits, spaces, apostrophes (driver's), forward slashes (G2/G),
# hyphens (310S-canon), periods (Class A.), commas (Class A, B). Any
# character outside this set rejects the whole display.
_DISPLAY_ALLOWED_CHARS = re.compile(r"^[A-Za-z0-9 '/\-.,]+$")


def _sanitize_credential_display(display: str) -> str:
    """Validate the WHOLE credential display string. Returns the input
    (with collapsed whitespace + length cap) when it's safe, or ""
    when ANY suspect content is present so the renderer falls back to
    the no-target "Which credential?" template.

    Reject (do not strip) on:
      - any URI scheme (http, https, ftp, javascript, mailto, data, ...)
      - any domain-like token (`foo.bar`, `evil.ca/path`)
      - any markup delimiter (`<`, `>`, `[`, `]`, `{`, `}`, `\\`, `|`)
      - any email `@`
      - any control character
      - any $, /hour, /hr, per-hour pattern
      - any scope-violation hook (Express Entry, Job Bank, IRCC, ...)
      - any known training provider name
      - any character outside the credential-display allow-list
    """
    if not isinstance(display, str) or not display:
        return ""
    # Strip control characters first so the allow-list check sees a
    # clean string; if any were present, the next allow-list pass would
    # reject anyway -- but doing this here means cleaner test diagnostics.
    if any(ord(ch) < 0x20 or ord(ch) == 0x7f for ch in display):
        return ""
    # Collapse whitespace so the allow-list pattern can apply.
    candidate = re.sub(r"\s+", " ", display).strip()
    if not candidate:
        return ""
    # Length cap: matches the MAX_CANONICAL_CHARS rule from staging.
    if len(candidate) > 80:
        return ""
    # Strict reject on any deny pattern.
    for pat in _DISPLAY_REJECT_PATTERNS:
        if pat.search(candidate):
            return ""
    # Final allow-list check: every character must be in the safe set.
    if not _DISPLAY_ALLOWED_CHARS.match(candidate):
        return ""
    return candidate


def _render_clarification(payload: dict[str, Any]) -> str:
    """Render the templated clarification text from a structured
    payload. Static text only -- no provider names, no URLs, no
    user-supplied substitutions outside the explicit credential display.

    Templates:
      kind="credential_completion_confirmation"
        action="add", canonical known:
          "Just to make sure -- have you completed your {display}, or
           are you still working toward it? Want to point you at the
           right next step."
        action="remove", canonical known:
          "Just to confirm -- you don't have your {display}? I'll
           recalculate against that."
        canonical=None:
          "Could you say which credential you mean -- 310S, G2/G, or
           something else? Just want to make sure I point you at the
           right next step."
      kind="bootstrap_match_request":
          "I haven't shown you any local matches yet -- want me to look
           for roles in your target field first, then we can walk
           through the gaps together?"
    """
    if not isinstance(payload, dict):
        return _GENERIC_CLARIFICATION_FALLBACK
    kind = payload.get("kind")
    if kind == "bootstrap_match_request":
        return (
            "I haven't shown you any local matches yet -- want me to "
            "look for roles in your target field first, then we can "
            "walk through the gaps together?"
        )
    if kind == "credential_completion_confirmation":
        canonical = payload.get("credential_canonical")
        raw_display = payload.get("credential_display") or ""
        action = payload.get("action") or "add"
        # R-5 round-4 (identity contract): the renderer MUST verify
        # the candidate display against a set of TRUSTED sources
        # before interpolating. Syntactic validation (round 3) is
        # insufficient because anything matching the allow-list could
        # still be hostile prose ("310S Evil Training Academy",
        # "310S 35 dollars an hour"). Semantic safety requires that
        # the interpolated string equal a value the handler resolved
        # against the snapshot's credential_gaps[*].display.
        #
        # `trusted_displays` is the list of snapshot displays the
        # handler attached to the payload. Display verification:
        #   1. raw_display ∈ trusted_displays → use raw_display
        #   2. canonical ∈ trusted_displays → use canonical
        #   3. else → fall back to no-target template
        if not isinstance(canonical, str) or not canonical:
            # Disambiguation branch: user said "got it" / "I have that"
            # with no resolvable entity. Ask which credential.
            return _CLARIFICATION_NO_TARGET_TEMPLATE
        # Round-22 hardening: trusted_displays MUST be a list/tuple
        # before iteration. Otherwise a payload that supplied a dict
        # (iterates keys) or a bare string (iterates characters)
        # would build a malformed `trusted` set and let single-char
        # strings pass the isinstance check.
        raw_trusted = payload.get("trusted_displays")
        if not isinstance(raw_trusted, (list, tuple)):
            raw_trusted = ()
        trusted: set[str] = {
            t for t in raw_trusted if isinstance(t, str) and t
        }
        verified: str | None = None
        if isinstance(raw_display, str) and raw_display in trusted:
            verified = raw_display
        elif canonical in trusted:
            verified = canonical
        if verified is None:
            # Neither display nor canonical was verified against the
            # trusted snapshot set. Forged or stale payload -- fall
            # back to the safe no-target template.
            return _CLARIFICATION_NO_TARGET_TEMPLATE
        # Defense in depth: even an identity-verified string passes
        # the syntactic allow-list. If a snapshot somehow stored bad
        # content, this catches it too.
        display = _sanitize_credential_display(verified)
        if not display:
            return _CLARIFICATION_NO_TARGET_TEMPLATE
        if action == "remove":
            return (
                f"Just to confirm -- you don't have your {display}? "
                f"I'll recalculate against that."
            )
        return (
            f"Just to make sure -- have you completed your {display}, "
            f"or are you still working toward it? Want to point you at "
            f"the right next step."
        )
    return _GENERIC_CLARIFICATION_FALLBACK


_GENERIC_CLARIFICATION_FALLBACK = (
    "Tell me a bit more about what you're looking for."
)


# Rule 3 (router) -- training_request_no_entity. Single source of truth
# for the locked phrasing per design decision #4. Used by both
# `compose_response_v2` (early-return) and `_fallback_reply_v2`
# (defense-in-depth).
_TRAINING_REQUEST_NO_ENTITY_QUESTION = (
    "Sure -- what skill or certificate do you want training for? "
    "For example Excel, WHMIS, forklift, Class G, or 310T."
)


_CLARIFICATION_NO_TARGET_TEMPLATE = (
    "Could you say which credential you mean -- 310S, G2/G, or "
    "something else? Just want to make sure I point you at the right "
    "next step."
)


def _build_user_block_v2(
    inp: ResponderV2Input, view: SanitizedResponderView,
) -> str:
    """Serialize the v2 input for the LLM. WHITELISTS exactly the
    decision fields that may reach the prompt.

    Fields from ArbiterDecision that are deliberately EXCLUDED:
      - arbiter_action: operational telemetry; would let the LLM
        narrate internals like "I overrode the planner"
      - notes: internal debugging output
    Tests assert these never appear in the resulting block.

    Sub-step 4: reads results, training, adjacent-recommendation, and
    adjacent-role-description payloads via the SanitizedResponderView's
    projected items (URLs sanitized; unknown raw fields dropped;
    score_explanation normalized). Non-URL payloads (NEAR_MISS_GAPS,
    REMAINING_GAPS) continue to read from inp directly.
    """
    d = inp.decision
    role_category = intake_priority.classify_role(inp.target_role_text)

    parts: list[str] = []
    parts.append(f"USER_MESSAGE:\n{inp.user_message}\n")

    # ---- Whitelisted decision fields ----
    parts.append(f"FINAL_MOVE: {d.final_move}")
    parts.append(f"TONE: {d.tone}")
    parts.append(f"REASON_CODE: {d.reason_code}")
    if d.caps_applied:
        parts.append("CAPS_APPLIED: " + json.dumps(list(d.caps_applied)))
    # NOTE: d.arbiter_action and d.notes are NOT serialized here.
    # See module comment + test_chat_responder_v2.py::test_*_no_leak.

    parts.append(f"ROLE_CATEGORY: {role_category}")

    # Resume-upload offer (Option B, 2026-06-16): when the handler
    # signals should_offer_resume_upload=True, surface the literal
    # target role and the offer flag in the user block so the LLM can
    # weave a role-aware no-match response naturally. The prompt's
    # `present_no_match` shape handles the rest. Without this, the
    # LLM had no signal that the upload offer was authorized and
    # tended to render a generic "no postings" + SCCC referral (which
    # was both misleading and blocked by the ungrounded-provider
    # policy gate).
    if inp.target_role_text:
        parts.append(f"TARGET_ROLE: {inp.target_role_text}")
    if inp.should_offer_resume_upload:
        parts.append("RESUME_UPLOAD_OFFER: yes")

    if d.ask_slot:
        hint = intake_priority.prompt_hint(d.ask_slot, inp.target_role_text)
        parts.append("ASK_SLOT:\n" + json.dumps(
            {"slot": d.ask_slot, "prompt_hint": hint},
        ))

    parts.append(f"BAND_SIGNAL: {inp.band_signal}")

    # ---- Match payloads (only when narrating a match outcome) ----
    show_matches = d.final_move in {"present_matches", "present_no_match"}
    if show_matches and view.prompt_results:
        parts.append("RESULTS:")
        for r in view.prompt_results:
            parts.append(json.dumps(serialize_result_for_v2_prompt(r)))
    elif show_matches:
        parts.append("RESULTS:\n(no eligible matches in current SSM dataset)")

    # ---- NEAR_MISS_GAPS payload (Slice N) ----
    # Only serialized on present_near_miss turns. The LLM uses these
    # gap lists VERBATIM -- the prompt's narration rule for
    # present_near_miss forbids inventing gaps or surfacing items
    # filtered out upstream (operational requirements).
    if d.final_move == "present_near_miss" and inp.near_miss_payload:
        parts.append("NEAR_MISS_GAPS:\n" + json.dumps({
            "role": inp.near_miss_payload.get("role"),
            "employer": inp.near_miss_payload.get("employer"),
            "job_count": inp.near_miss_payload.get("job_count"),
            "credential_gaps": list(inp.near_miss_payload.get("credential_gaps") or ()),
            "core_skill_gaps": list(inp.near_miss_payload.get("core_skill_gaps") or ()),
        }))

    # ---- REMAINING_GAPS payload (R-5) ----
    # Only serialized on explain_remaining_gaps turns. The LLM uses
    # these lists VERBATIM -- the prompt forbids inventing gaps,
    # explaining "why each gap matters", or naming providers absent
    # from TRAINING. Conditional tense is REQUIRED when
    # `any_hypothetical` is true.
    if d.final_move == "explain_remaining_gaps" and inp.remaining_gaps_payload:
        rg = inp.remaining_gaps_payload
        parts.append("REMAINING_GAPS:\n" + json.dumps({
            "role":     rg.get("role"),
            "employer": rg.get("employer"),
            "assumed_completed_credentials": list(
                rg.get("assumed_completed_credentials") or ()
            ),
            "remaining_credentials": list(
                rg.get("remaining_credentials") or ()
            ),
            "remaining_core_skills": list(
                rg.get("remaining_core_skills") or ()
            ),
            "any_hypothetical": bool(rg.get("any_hypothetical")),
        }))

    # ---- ADJACENT_RECOMMENDATIONS payload (AR-6c) ----
    # Only serialized on recommend_adjacent_roles turns. The LLM uses
    # the `recommendations` list VERBATIM (titles, employers, evidence
    # summaries, matched skills). The narration shape in
    # OUTCOME_RESPONDER_PROMPT forbids inventing roles or "you qualify"
    # framing. Empty list -> deterministic fallback.
    if d.final_move == "recommend_adjacent_roles" and view.prompt_adjacent_recommendations is not None:
        block = serialize_adjacent_recommendations_for_prompt(
            view.prompt_adjacent_recommendations,
        )
        parts.append("ADJACENT_RECOMMENDATIONS:\n" + json.dumps(block))

    # ---- ADJACENT_ROLE_DESCRIPTION payload (AR-6c) ----
    # Only serialized on describe_adjacent_role turns. Carries the
    # LIVE job row + the snapshot's evidence summary + matched skills.
    # `expired=True` triggers the deterministic "that role's no longer
    # on the board" fallback.
    if d.final_move == "describe_adjacent_role" and view.prompt_adjacent_role is not None:
        block = serialize_prompt_adjacent_role(view.prompt_adjacent_role)
        parts.append("ADJACENT_ROLE_DESCRIPTION:\n" + json.dumps(block))

    # ---- TRAINING payload ----
    # Sub-step 4 + bug.4: training is surfaced via the move-specific
    # projection slot on the view.
    #
    #   present_matches (and present_no_match defensive carryover):
    #       grouped by owning job_id (bug.4) via
    #       view.prompt_present_matches_training_groups. Each group
    #       serializes as one JSON line:
    #         {"job_id", "job_title", "resources": [TrainingView dicts]}
    #       The grouping makes per-job ownership structural — the LLM
    #       cannot ignore it the way it might an instruction.
    #
    #   explain_gap / present_near_miss / explain_remaining_gaps:
    #       flat training (single-role context, no multi-job
    #       attribution problem).
    #
    # In all cases: unknown raw fields are dropped per the 11-field
    # TrainingView allowlist; None fields are omitted (V2 serializer).
    if d.final_move in ("present_matches", "present_no_match"):
        groups = view.prompt_present_matches_training_groups
        if groups:
            parts.append("TRAINING:")
            for g in groups:
                parts.append(json.dumps({
                    "job_id": g.job_id,
                    "job_title": g.job_title,
                    "resources": [
                        serialize_training_for_v2_prompt(t)
                        for t in g.resources
                    ],
                }))
    else:
        training_flat: tuple = ()
        if d.final_move == "explain_gap":
            training_flat = view.prompt_explain_gap_training_flat
        elif d.final_move == "present_near_miss":
            training_flat = view.prompt_present_near_miss_training_flat
        elif d.final_move == "explain_remaining_gaps":
            training_flat = view.prompt_explain_remaining_gaps_training_flat
        if training_flat:
            parts.append("TRAINING:")
            for t in training_flat:
                parts.append(json.dumps(serialize_training_for_v2_prompt(t)))

    if show_matches:
        skill, count = inp.next_skill
        if skill and count:
            parts.append(
                f"NEXT_SKILL: {skill} would unlock {count} more current SSM jobs."
            )

    if inp.requires_consent and d.final_move == "present_matches":
        parts.append(
            "CONSENT_STATE: anonymous. The user has NOT consented to save "
            "their profile. Show the matches but do not promise to remember "
            "across sessions."
        )

    if inp.resume_facts and _resume_facts_summary_has_content(inp.resume_facts):
        parts.append("RESUME_FACTS:\n" + _resume_facts_summary_for_prompt(inp.resume_facts))

    # Step 11d (2026-06-17): pipe the SSM market snapshot through to the
    # LLM happy path on no-match turns. SHAPE 2 enhanced (Step 9) needs
    # `total_active_jobs`, `top_sectors`, and `top_employers` to weave
    # the "Sault Ste. Marie has 43 active postings — mostly in healthcare,
    # trades, and admin" panorama. Without this, the LLM defaults to the
    # legacy "I don't see one + SCCC referral" close — which the live
    # verify on 2026-06-17 surfaced as a dead-end for resume-uploaded
    # users at the bottom of the closing matrix.
    if inp.pipeline_snapshot is not None:
        parts.append("PIPELINE_SNAPSHOT:\n" + json.dumps({
            "total_active_jobs": inp.pipeline_snapshot.total_active_jobs,
            "last_publish_at_text": inp.pipeline_snapshot.last_publish_at_text,
            "top_sectors": list(inp.pipeline_snapshot.top_sectors),
            "top_employers": list(inp.pipeline_snapshot.top_employers),
        }))

    # Step 11e (2026-06-17, closing-matrix v2): signal to the LLM that
    # we already attempted the related-role (CP5) search on this turn's
    # engine run — either via Pattern 2 yes-consent or Pattern 3
    # auto-fire — and the search returned 0 results. Without this
    # signal, SHAPE 2's legacy "Optional: offer one alternative angle"
    # rule fires (the LLM thinks it might surface related roles) and
    # the closing reads "want me to look at related roles?" — but the
    # engine ALREADY tried, the search returned empty, and the system
    # is asking the user to re-request a path the system has already
    # exhausted. Infinite-offer loop. Discovered in live verify on
    # 2026-06-17.
    #
    # Trigger: present_no_match outcome AND resume facts on file. The
    # second condition is the discriminator — when no resume is on
    # file the path is Pattern 1 (upload ask, SHAPE 1), which is a
    # different closing branch entirely. With resume on file at
    # present_no_match, by construction the engine has run a full
    # adjacency lookup as part of `_build_tier_evidence_for_handler`
    # and concluded "no related-role bridge available for this profile."
    if (
        d.final_move == "present_no_match"
        and inp.resume_facts
        and _resume_facts_summary_has_content(inp.resume_facts)
    ):
        parts.append("RELATED_ROLES_EXHAUSTED: yes")

    # Step 11h (2026-06-17): when CP4 produced a primary recommendation
    # (canonical skill name), surface it for MOVEMENT C2 ("the one thing
    # that came up is [GAP]"). Quoted VERBATIM by both the LLM happy path
    # (per OUTCOME_RESPONDER_PROMPT grounding rules) and the
    # deterministic fallback. None when CP4 returned no recommendation
    # (Movement C2 then skips).
    if inp.cp4_primary_gap:
        parts.append(f"CP4_PRIMARY_GAP: {inp.cp4_primary_gap}")

    return "\n".join(parts)


def _policy_ok_v2(
    reply: str, inp: ResponderV2Input, view: SanitizedResponderView,
) -> bool:
    """v2 policy sweep. Reuses every shared output-level rule from
    `_policy_ok` plus a v2-specific bullet check that triggers on
    `ask_one_clarifying_question` final_move (instead of legacy
    ACTION_ASK_QUESTIONS).

    Sub-step 5: URL grounding check is now active. Every URL extracted
    from the reply must canonicalize to a member of view.prompt_urls
    (the move-gated allowlist the LLM was shown for this turn).
    Structural violations (URL_MALFORMED, URL_UNSUPPORTED_SCHEME, etc.)
    also reject. Telemetry emitted via safe_telemetry_fields.
    """
    if not reply.strip():
        return False

    # URL grounding: extract every URL-shaped token from the reply and
    # check membership against the per-turn prompt allowlist. Any
    # violation (structural or membership) rejects the reply.
    for candidate in extract_url_candidates(reply):
        result = check_url_membership(
            candidate.extracted_token, view.prompt_urls,
        )
        if isinstance(result, Violation):
            fields = safe_telemetry_fields(
                result, move=inp.decision.final_move,
            )
            log.warning(
                "policy v2: URL grounding violation "
                "code=%s move=%s scheme=%s host=%s hash=%s",
                fields["violation_code"], fields["move"],
                fields["scheme"], fields["host"], fields["url_hash"],
            )
            return False

    lower = reply.lower()

    if inp.requires_consent:
        bad_promises = (
            "i'll remember", "i will remember", "i'll save your profile",
            "i'll keep this for you", "next time i'll",
        )
        if any(p in lower for p in bad_promises):
            return False

    if "$" in reply or "/hr" in reply or "/hour" in reply.lower():
        return False

    forbidden = ("job bank", "statistics canada", "statcan", "national average")
    if any(p in lower for p in forbidden):
        return False

    for pat in _OUT_OF_REGION_PATTERNS:
        if pat.search(reply):
            log.warning("policy v2: reply offers out-of-region search (pattern=%s)", pat.pattern)
            return False
    for pat in _NON_LOCAL_CITY_OFFER_PATTERNS:
        if pat.search(reply):
            log.warning("policy v2: reply suggests non-local city (pattern=%s)", pat.pattern)
            return False
    for pat in _CREDENTIAL_EQUIVALENCE_PATTERNS:
        if pat.search(reply):
            log.warning("policy v2: reply makes credential equivalence claim")
            return False
    for pat in _IMMIGRATION_LEGAL_PATTERNS:
        if pat.search(reply):
            log.warning("policy v2: reply gives immigration/legal-tier advice")
            return False

    # Slice N: forbid "match" / "fit" / "qualify" framing on
    # present_near_miss turns. The whole point of near-miss is "the
    # role exists but you're NOT a match" -- the LLM saying "good fit"
    # here misrepresents the gap and undermines the gap-analysis frame.
    # R-5: SAME rule applies on explain_remaining_gaps turns -- the
    # user has only CLAIMED completion; we don't certify the match.
    # AR-6c: SAME rule applies on the two adjacency outcomes --
    # adjacency surfaces eligibility-by-credential, NOT match-quality
    # certification. The locked surface vocabulary is "roles worth
    # exploring" / "where some of your existing skills transfer";
    # "you qualify" / "good fit" / "perfect for you" undermine that
    # framing (locked v11 §"Forbidden vocabulary").
    if inp.decision.final_move in (
        "present_near_miss",
        "explain_remaining_gaps",
        "recommend_adjacent_roles",
        "describe_adjacent_role",
    ):
        for pat in _NEAR_MISS_FORBIDDEN_PATTERNS:
            if pat.search(reply):
                log.warning(
                    "policy v2: %s reply uses forbidden framing "
                    "(pattern=%s)", inp.decision.final_move, pat.pattern,
                )
                return False

    # R-5 (design §9 prompt rule): on explain_remaining_gaps turns the
    # LLM MUST NOT speculate about how non-credential gaps are typically
    # closed ("usually come on the job", "best learned through a
    # course", "comes with experience"). The payload supplies gap
    # NAMES only; any "why it matters" / "how it's earned" sentence is
    # invented content. This is a structural prohibition regardless of
    # TRAINING contents -- v1 doesn't populate TRAINING for skill gaps,
    # so the LLM can never ground these claims.
    if inp.decision.final_move == "explain_remaining_gaps":
        for pat in _REMAINING_GAPS_SPECULATION_PATTERNS:
            if pat.search(reply):
                log.warning(
                    "policy v2: explain_remaining_gaps reply speculates "
                    "about gap closure (pattern=%s)", pat.pattern,
                )
                return False

    # Bullet-list check keyed on OutcomeMove. Bullets are still
    # banned on ask_one_clarifying_question (checklist behavior we
    # explicitly want to avoid). They are NOT banned on explain_gap
    # turns -- training-resource lists are structured information and
    # bullets are the right UX shape per the prompt's explain_gap
    # narration rule. Match cards already use bullets too.
    if inp.decision.final_move == "ask_one_clarifying_question":
        bullet_lines = sum(
            1 for ln in reply.splitlines()
            if ln.lstrip().startswith(("•", "- ", "* ", "1.", "2.", "3."))
        )
        if bullet_lines >= 1:
            return False

    # Reject any reply that surfaces operational/arbiter terms. These
    # never belong in user-facing text.
    operational_leakage = (
        "arbiter_action", "overrode the planner", "the planner said",
        "the arbiter decided", "fallback_to_legacy",
    )
    for term in operational_leakage:
        if term in lower:
            log.warning("policy v2: reply leaks operational term %r", term)
            return False

    # Post-Slice-9 grounding fix: reject ungrounded training-provider
    # mentions. This is the architectural defense behind the "registry
    # decides, LLM narrates" promise. The check is conservative -- it
    # only fires on names in `_KNOWN_TRAINING_PROVIDERS` (which Haiku
    # has been seen to invent) that are NOT in this turn's TRAINING
    # block. Providers actually present in TRAINING are allowed.
    #
    # SCCC carve-out (Option B, 2026-06-16): "Sault Community Career
    # Centre" is an INSTITUTIONAL referral, not a training-provider
    # claim. On `present_no_match` turns it's the canonical fallback
    # the prompt explicitly authorises ("End by suggesting Sault
    # Community Career Centre."), and on `should_offer_resume_upload`
    # turns the prompt now instructs the LLM to weave SCCC as the
    # last-resort fallback. Without this carve-out the LLM correctly
    # follows the prompt and gets blocked by the policy gate, falling
    # back to the canned deterministic template the user complained
    # about. The carve-out is move-scoped — outside no-match turns
    # the original gate behaviour is preserved verbatim.
    sccc_allowed = (
        inp.decision.final_move == "present_no_match"
        or bool(inp.should_offer_resume_upload)
    )
    # Intake-ask carve-out (2026-06-16, evening): on
    # ask_one_clarifying_question turns the LLM is asking the user about
    # their background, not recommending training. It naturally reaches
    # for role-relevant tool names as example skills ("QuickBooks,
    # Excel, accounting software, etc."). Several of those tool names
    # — QuickBooks, Microsoft Office, etc. — happen to be in
    # `_KNOWN_TRAINING_PROVIDERS` because they're also registry-tracked
    # training providers. The original gate then blocked the LLM's
    # natural intake question, and the deterministic fallback fired
    # the canned single-line ask. Result: every accounting/admin/
    # bookkeeping first-turn intake felt robotic ("What office
    # software and tasks you've used — for example Microsoft Word,
    # Excel, scheduling, bookkeeping, QuickBooks?").
    #
    # On intake-ask turns the LLM is collecting evidence, not making
    # training claims. The gate is allowed to be permissive here. On
    # any turn where the LLM IS making a training claim
    # (explain_gap, present_matches with TRAINING, etc.) the original
    # strict behaviour is preserved.
    intake_ask_allowed = (
        inp.decision.final_move == "ask_one_clarifying_question"
    )
    ungrounded = _check_ungrounded_provider(reply, inp.training_by_job)
    if ungrounded is not None:
        is_sccc = "sault community career centre" in ungrounded.lower()
        if is_sccc and sccc_allowed:
            pass  # explicitly authorised institutional referral
        elif intake_ask_allowed:
            pass  # tool name used as a skill example during intake
        else:
            log.warning(
                "policy v2: reply names ungrounded training provider %r "
                "(not in this turn's TRAINING block)", ungrounded,
            )
            return False

    return True


def _fallback_reply_v2(
    inp: ResponderV2Input, view: SanitizedResponderView,
) -> str:
    """Deterministic per-OutcomeMove fallback. Returns a sensible reply
    when the LLM is disabled or its output failed the policy check.

    The chat never breaks: each OutcomeMove maps to a single-sentence
    or short-paragraph reply that respects the SCOPE BOUNDARIES.

    Sub-step 4: threads `view` to each URL-bearing callee. URL-free
    callees (redirect_scope, confirm_resume_summary, present_no_match,
    present_near_miss, present_remaining_gaps) receive only `inp`.
    """
    d = inp.decision
    target = inp.target_role_text
    move = d.final_move

    if move == "redirect_scope":
        return _redirect_scope_fallback_v2(inp)

    if move == "acknowledge_and_continue":
        return "Got it. What would you like to focus on next?"

    if move == "ask_one_clarifying_question":
        # Rule 3 defense-in-depth: if the LLM happy-path was skipped or
        # failed policy and we ended up here, still emit the locked
        # training-discovery question for this reason. The same string
        # lives in `compose_response_v2` as the primary early-return.
        if d.reason_code == "training_request_no_entity":
            return _TRAINING_REQUEST_NO_ENTITY_QUESTION
        if d.ask_slot:
            return _single_ask(d.ask_slot, target)
        return "Tell me a bit more about what you're looking for."

    if move == "explain_gap":
        return _explain_gap_fallback_v2(inp, view)

    if move == "offer_refinement":
        return (
            "Happy to narrow these down. Would you like to focus on a more "
            "specific role, a different work type, or a different skill set?"
        )

    if move == "confirm_resume_summary":
        return _confirm_resume_summary_fallback_v2(inp)

    if move == "present_no_match":
        return _present_no_match_fallback_v2(inp)

    if move == "present_near_miss":
        return _present_near_miss_fallback_v2(inp)

    if move == "explain_remaining_gaps":
        return _present_remaining_gaps_fallback_v2(inp)

    if move == "recommend_adjacent_roles":
        return _recommend_adjacent_roles_fallback_v2(inp, view)

    if move == "describe_adjacent_role":
        return _describe_adjacent_role_fallback_v2(inp, view)

    # present_matches (default)
    return _present_matches_fallback_v2(inp, view)


# =========================================================================
# AR-6c -- deterministic adjacency-recommendation fallback
# =========================================================================
def _valid_adjacent_recommendations(
    payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return the validity-filtered recommendations from the payload.

    A "valid" entry is a dict with a non-empty string `title`. Anything
    else (forged blob, broken upstream, missing title, empty title) is
    dropped before the renderer or the early-return gate sees it.

    Shared by `_recommend_adjacent_roles_fallback_v2` (which renders
    `valid_recs`) and the AR-8a early-return guard in
    `compose_response_v2` (which uses emptiness of this list to skip
    the LLM entirely). Keeping the rule in one place means "empty"
    means the same thing on both code paths -- new validity rules
    auto-propagate.
    """
    if not isinstance(payload, dict):
        return []
    recs = payload.get("recommendations")
    if not isinstance(recs, list):
        return []
    out: list[dict[str, Any]] = []
    for r in recs:
        if not isinstance(r, dict):
            continue
        title = r.get("title")
        if not isinstance(title, str) or not title.strip():
            continue
        out.append(r)
    return out


def _recommend_adjacent_roles_fallback_v2(
    inp: ResponderV2Input, view: SanitizedResponderView,
) -> str:
    """Render recommend_adjacent_roles deterministically when the LLM
    is disabled or its output failed the policy check.

    Three branches:
      1. Non-empty recommendations: list up to three roles with
         title + employer + evidence summary, close with a focused
         next-step question.
      2. Empty recommendations: provider-free "I'm not seeing other
         roles..." line (locked v11 §"Empty-result narration").
      3. Missing payload (defensive): same empty-result line.

    NEVER says "you qualify", "good fit", "perfect for you" --
    approved tokens are "roles worth exploring" and "where some of
    your existing skills transfer".

    Sub-step 4: reads filtered, validated recommendations from
    view.fallback_adjacent_recommendations (which already applies the
    dataclass-as-allowlist filtering — non-dict entries dropped,
    missing/empty titles dropped, url and unknown fields stripped).
    """
    fallback_recs = view.fallback_adjacent_recommendations

    if not fallback_recs:
        return (
            "From today's Sault Ste. Marie postings, I'm not seeing "
            "other roles where your current skills line up strongly "
            "enough to recommend. Want to look at the training path "
            "for your current target, or check back when more "
            "postings come in?"
        )

    lines: list[str] = [
        "Here are a few Sault Ste. Marie postings worth exploring "
        "with what you've got today:"
    ]
    for r in fallback_recs[:3]:
        title = r.title or "Role"
        head = f"- {title}"
        if r.employer:
            head += f" at {r.employer}"
        if r.evidence_summary:
            head += f" -- {r.evidence_summary}."
        lines.append(head)
    lines.append("Want me to look closer at any of these?")
    return "\n".join(lines)


# =========================================================================
# AR-6c -- deterministic adjacency-role-description fallback
# =========================================================================
def _describe_adjacent_role_fallback_v2(
    inp: ResponderV2Input, view: SanitizedResponderView,
) -> str:
    """Render describe_adjacent_role deterministically.

    Two branches:
      1. expired=True OR missing job: deterministic fallback line.
      2. expired=False: name the role + employer + location, surface
         the snapshot's evidence summary, mention matched skills,
         close with a next-step. NEVER invents a URL.

    Sub-step 4: reads payload-level fields from view.fallback_adjacent_role
    (job_id, title, employer, location, evidence_summary, matched_skills,
    expired) and uses view.fallback_adjacent_role.has_validated_url to
    gate the "Want the posting URL?" wording. The fallback never reads
    the URL string itself per the locked bug.2a contract.
    """
    role_view = view.fallback_adjacent_role
    if role_view is None:
        return "That role's no longer on the board -- want me to look again?"
    if role_view.expired:
        return "That role's no longer on the board -- want me to look again?"
    if not role_view.job_is_mapping:
        # Mirrors original responder.py:1787's `isinstance(job, dict)`
        # guard. A job dict with only employer/location populated is
        # accepted (job_is_mapping=True); only a missing/None/non-dict
        # value triggers the deterministic fallback line.
        return "That role's no longer on the board -- want me to look again?"
    title = role_view.title or "this role"
    location = role_view.location or "Sault Ste. Marie"
    head = f"{title}"
    if role_view.employer:
        head += f" at {role_view.employer}"
    head += f" -- {location}."

    evidence = role_view.evidence_summary or ""
    middle = ""
    if evidence:
        middle = f" {evidence}."
    if role_view.matched_skills:
        names = ", ".join(role_view.matched_skills[:3])
        if names:
            middle += f" Your {names} carry over."

    # Bug.2b: render the validated posting URL inline when available
    # so the user doesn't need a second round-trip to see it.
    # has_validated_url stays as a defensive cross-check; both fields
    # come from the same SanitizedURL via _project_fallback_adjacent_role.
    if role_view.url is not None:
        tail = (
            f" See the posting at {role_view.url.raw}."
            " Want me to look at the path to apply?"
        )
    else:
        tail = " Want me to look at the path to apply?"
    return (head + middle + tail).strip()


# =========================================================================
# R-5 -- deterministic remaining-gaps fallback
# =========================================================================
def _present_remaining_gaps_fallback_v2(inp: ResponderV2Input) -> str:
    """Render explain_remaining_gaps from `remaining_gaps_payload` only.
    Never invents gaps; never names providers absent from TRAINING; uses
    conditional tense when `any_hypothetical` is True (locked design §3).

    Three narration branches:
      1. remaining_credentials non-empty: name the LEAD credential,
         optionally mention a provider from TRAINING, close with a
         next-step offer.
      2. remaining_credentials empty AND remaining_core_skills non-empty:
         pivot to skill gaps with NO provider grounding (design §6).
      3. both empty: minimal close.
    """
    payload = inp.remaining_gaps_payload or {}
    role = payload.get("role") or "this role"
    remaining_credentials = list(payload.get("remaining_credentials") or ())
    remaining_skills = list(payload.get("remaining_core_skills") or ())
    any_hypothetical = bool(payload.get("any_hypothetical"))
    assumed = list(payload.get("assumed_completed_credentials") or ())

    # Opening: acknowledge the assumption. Conditional when hypothetical.
    opener_assumed_phrase = _assumed_phrase(assumed)
    if any_hypothetical:
        opener = f"If you've got {opener_assumed_phrase} in hand"
    else:
        opener = f"With {opener_assumed_phrase} done"
    if not assumed:
        # No accumulated state -- this is a generic "what else?" before
        # any claim. Skip the opener phrase shape.
        opener = "On the credentials for this role"

    parts: list[str] = []

    if remaining_credentials:
        lead = remaining_credentials[0]
        lead_display = lead.get("display") if isinstance(lead, dict) else None
        lead_display = lead_display or "the next required credential"
        # Provider grounding: ONLY from inp.training_by_job. The handler
        # populated this for the lead credential (flag-gated).
        provider_clause = _grounded_provider_clause(
            inp.training_by_job, lead_display,
        )
        sentence = (
            f"{opener}, the next required credential for {role} is "
            f"your {lead_display}"
        )
        if provider_clause:
            sentence += f" -- {provider_clause}"
        sentence += "."
        parts.append(sentence)

        if remaining_skills:
            skills_phrase = ", ".join(remaining_skills[:3])
            parts.append(
                f"Beyond that, there are experience and skill gaps: "
                f"{skills_phrase}."
            )
        parts.append(
            "Want me to look at the next step on that credential, or "
            "check what local shops are hiring while you work toward it?"
        )
        return " ".join(parts)

    # All credentials closed.
    if remaining_skills:
        skills_phrase = ", ".join(remaining_skills[:3])
        if any_hypothetical:
            parts.append(
                f"If you've got {opener_assumed_phrase} in hand, the "
                f"credentials would line up for {role}."
            )
        else:
            parts.append(
                f"With {opener_assumed_phrase} done, the credentials "
                f"line up for {role}."
            )
        parts.append(
            f"The remaining items are experience and skill gaps: "
            f"{skills_phrase}."
        )
        parts.append(
            "Want me to look at what local shops are hiring so you "
            "can apply with your current experience?"
        )
        return " ".join(parts)

    # Both closed: minimal close, no provider grounding (design §6).
    if any_hypothetical:
        return (
            f"If you've got {opener_assumed_phrase} in hand, the job "
            f"posting itself is the next step -- check the listing "
            f"and consider applying."
        )
    return (
        f"With {opener_assumed_phrase} done, the job posting itself "
        f"is the next step -- check the listing and consider applying."
    )


def _assumed_phrase(assumed: list[dict[str, Any]]) -> str:
    """Concatenate display names from assumed_completed_credentials,
    falling back to a neutral phrase when empty. Used by the fallback
    narrator's opener."""
    displays: list[str] = []
    for a in assumed:
        if isinstance(a, dict):
            d = a.get("display")
            if isinstance(d, str) and d:
                displays.append(d)
    if not displays:
        return "what you've shared"
    if len(displays) == 1:
        return f"your {displays[0]}"
    if len(displays) == 2:
        return f"your {displays[0]} and {displays[1]}"
    return f"your {', '.join(displays[:-1])}, and {displays[-1]}"


def _grounded_provider_clause(
    training_by_job: dict[str, list[dict]] | None,
    lead_display: str,
) -> str:
    """Return a one-line "Provider can point you at the next step"
    clause IFF the registry surfaced a resource WHOSE GAP KEY MATCHES
    the lead credential's display. Otherwise return "".

    Round-2 R-5 review: the helper now honours its `lead_display`
    argument. Without this guard the helper returned the first
    provider from ANY training list -- so a stale or unrelated entry
    in `training_by_job` (handler keeps it gated, but defense in
    depth) could ground the wrong credential.

    The handler keys training as `gap:<lead_display>`; we match that
    shape directly. Returns "" when no matching key exists or the
    matched resource list is empty.
    """
    if not training_by_job or not isinstance(lead_display, str) or not lead_display:
        return ""
    expected_key = f"gap:{lead_display}"
    resources = training_by_job.get(expected_key)
    if not isinstance(resources, list) or not resources:
        return ""
    first = resources[0]
    if not isinstance(first, dict):
        return ""
    provider = first.get("provider")
    if not isinstance(provider, str) or not provider:
        return ""
    return f"{provider} can point you at the next step"


# =========================================================================
# Slice 8 -- context-aware redirect_scope fallback
# =========================================================================
# When the responder LLM produces text that fails the policy check
# (typically by crossing into immigration / legal / national-feeds
# territory), `compose_response_v2` falls back to this deterministic
# text. Pre-Slice-8 the fallback was a generic "What kind of work are
# you looking for here?" -- safe but context-poor. Slice 8 wires the
# short-session ConversationContext so the fallback can reference
# matches the user just saw + the specific credential gaps that came
# up, instead of starting cold.
#
# Three context tiers, in order of strength:
#   1. last_presented_job_titles + last_presented_credential_gaps
#        -> "On the truck and coach roles we just looked at, the main
#            gap is still 310T technician certification..."
#   2. last_presented_job_titles only (no specific credential gap data)
#        -> "We can keep working on those roles..."
#   3. target_role_text only (no matches shown yet)
#        -> "We were looking at warehouse work..."
#   4. nothing -> the original generic line (preserved for cold sessions)
# =========================================================================
_SCOPE_REDIRECT_PREFIX = "I'm focused on helping you find work in Sault Ste. Marie."


def _redirect_scope_fallback_v2(inp: ResponderV2Input) -> str:
    """Pick the strongest available context and weave it into the
    redirect. Falls through to the generic line when no context
    exists. Stays SCOPE_BOUNDARIES-compliant by NEVER offering
    immigration / PR / legal *advice* -- but mentioning the Sault
    Community Career Centre as the right place to ask about
    immigration is a REFERRAL, not advice, and is the honest answer
    when a newcomer asks about PR. Live-test feedback (2026-06-05):
    a bare "I focus on jobs" line for an immigration question leaves
    the user without a next step.
    """
    ctx = inp.conversation_context
    is_immigration = inp.decision.reason_code == "scope_violation_immigration"

    # Tier 1: matches AND specific credential gaps
    if ctx and ctx.has_presented_context() and ctx.last_presented_credential_gaps:
        gap = ctx.last_presented_credential_gaps[0]
        role_phrase = _role_phrase_from_titles(ctx.last_presented_job_titles)
        return (
            f"{_SCOPE_REDIRECT_PREFIX} On {role_phrase} we just looked at, "
            f"the main gap is still {gap}. Want to keep working on that path, "
            f"or look at something else?"
        )

    # Tier 2: matches were shown but no specific gap names captured
    if ctx and ctx.has_presented_context():
        role_phrase = _role_phrase_from_titles(ctx.last_presented_job_titles)
        return (
            f"{_SCOPE_REDIRECT_PREFIX} We can keep working on "
            f"{role_phrase} you were looking at. "
            f"What would you like to focus on next?"
        )

    # Tier 3: target role known but no matches shown yet
    if ctx and ctx.target_role_text:
        return (
            f"{_SCOPE_REDIRECT_PREFIX} We were looking at "
            f"{ctx.target_role_text} work — want to stay with that, "
            f"or explore something else?"
        )

    # Tier 4a: cold session, IMMIGRATION scope -- SCCC referral.
    # The design doc's scenario #4 expects this exact shape: no
    # immigration *advice*, but the user gets a real next step
    # (the local agency that handles newcomer / PR / work-permit
    # questions). Keeps the redirect honest and useful.
    if is_immigration:
        return (
            f"{_SCOPE_REDIRECT_PREFIX} For immigration or PR questions, "
            "the Sault Community Career Centre is the right place to ask — "
            "they support newcomers with that side of things. "
            "In the meantime, what kind of work are you looking for here?"
        )

    # Tier 4b: cold session, OTHER scope reasons (wages, non-SSM,
    # off-topic) -- original generic line preserved. SCCC isn't the
    # right referral for these.
    return (
        f"{_SCOPE_REDIRECT_PREFIX} I can match you to local jobs and "
        f"suggest skills to build. What kind of work are you looking for here?"
    )


def _role_phrase_from_titles(titles: tuple[str, ...]) -> str:
    """Phrase the titles compactly for fallback prose.

    Single title: 'the {title} role'. Multiple: 'those roles' (we don't
    want to dump 5 titles inline). Empty: 'those roles' as a safe
    fallback (callers should check has_presented_context first).
    """
    titles_list = [t for t in titles if t]
    if len(titles_list) == 1:
        return f"the {titles_list[0]} role"
    return "those roles"


def _registry_grounded_explain_gap_fallback(
    inp: ResponderV2Input, view: SanitizedResponderView,
) -> str:
    """Deterministic fallback that narrates registry resources for the
    user's gap.

    Used when the LLM happy path failed policy (e.g. it tried to
    invent a provider). The fallback CANNOT introduce providers the
    LLM was rejected for naming -- it can only narrate what's in
    `view.fallback_explain_gap_training_flat`, which came from the
    curated registry via projection. That keeps the safety net on
    even when the LLM is taken out of the loop.

    Sub-step 4: signature drops the previous `flat_resources` parameter
    per the locked contract; reads sanitized training resources from
    view.fallback_explain_gap_training_flat. URL renders use
    SanitizedURL.raw.
    """
    training_flat = view.fallback_explain_gap_training_flat
    # Use the first entry's for_gap field for the lead-in; all entries
    # in a given training_by_job dict typically share the same gap.
    first = training_flat[0]
    gap_label = first.for_gap or "this credential"

    ctx = inp.conversation_context
    role_phrase = ""
    if ctx and ctx.last_presented_job_titles:
        titles = ctx.last_presented_job_titles
        if len(titles) == 1:
            role_phrase = f" for the {titles[0]} role we just looked at"
        else:
            role_phrase = " for those roles we just looked at"

    lines: list[str] = [
        f"For {gap_label}{role_phrase}, here are the next steps:",
    ]

    # The view's fallback_explain_gap_training_flat is already capped
    # at 3 per the locked sub-step 3 contract (matches the prompt's
    # bullet carve-out limit).
    for entry in training_flat:
        provider = (entry.provider or "").strip()
        summary = (entry.summary or "").strip()
        if not provider:
            continue
        bullet = f"- **{provider}**"
        if summary:
            bullet += f": {summary}"
        if entry.url is not None:
            bullet += f" ({entry.url.raw})"
        lines.append(bullet)

    lines.append(
        "Want me to look at related roles in the meantime, or focus on "
        "closing that gap first?"
    )
    return "\n".join(lines)


def _confirm_resume_summary_fallback_v2(inp: ResponderV2Input) -> str:
    """Deterministic resume-review summary when LLM is off.

    Mirrors the v1 `_present_resume_facts_fallback` pattern: reads
    `inp.resume_facts` (the post-suppression facts view from the
    handler), mentions the most recent work entry + top credential +
    a few skill names, ends with a "does that look right?" prompt.

    No bullets. No verbatim resume text (PII protection). If facts is
    empty / sparse, returns a neutral re-prompt asking the user to
    describe their background. Used by the resume_upload gate path in
    `_try_v2_path` (Slice 7 review fix).
    """
    facts = inp.resume_facts or {}
    work = facts.get("work_history") or []
    edu = facts.get("education") or []
    skills = facts.get("skills") or []

    pieces: list[str] = ["Thanks — I read your resume. Here's what stood out:"]

    if work:
        w = work[0]
        title = (w.get("title") or "").strip()
        employer = (w.get("employer") or "").strip()
        sy, ey, current = w.get("start_year"), w.get("end_year"), w.get("is_current")
        years = ""
        if isinstance(sy, int) and current:
            years = f" {sy}-present"
        elif isinstance(sy, int) and isinstance(ey, int):
            years = f" {sy}-{ey}"
        elif isinstance(sy, int):
            years = f" {sy}"
        head = " at ".join(p for p in (title, employer) if p) or title or employer
        if head:
            pieces.append(f"{head}{years}.")

    if edu:
        e = edu[0]
        cred = (e.get("credential") or "").strip()
        inst = (e.get("institution") or "").strip()
        if cred and inst:
            pieces.append(f"{cred} from {inst}.")
        elif cred:
            pieces.append(f"{cred}.")

    if skills:
        names = [
            s.get("name") for s in skills[:6]
            if isinstance(s, dict) and s.get("name")
        ]
        if names:
            pieces.append("Skills I picked up: " + ", ".join(names) + ".")

    if len(pieces) == 1:
        return (
            "I read your resume but couldn't pull much from it. "
            "Could you tell me a bit about your background — what kind of "
            "work have you done?"
        )

    pieces.append(
        "Does that look right? You can add anything I missed, or tell me "
        "to remove something that's not yours."
    )
    return " ".join(pieces)


def _explain_gap_fallback_v2(
    inp: ResponderV2Input, view: SanitizedResponderView,
) -> str:
    """Explain the credential / skill gap in plain language.

    Priority of data sources (post-grounding-fix order):
      1. view.fallback_explain_gap_training_flat (registry resources via
         sanitized projection) -- when populated, this is the BEST
         source: real provider names + summaries + optionally validated
         URLs.
      2. ConversationContext.last_presented_credential_gaps -- name the gap
      3. ConversationContext.last_presented_job_titles -- name the role
      4. decision.caps_applied -- name the cap category
      5. fall through to a neutral prompt

    Never lists job cards (that would defeat the explain_gap routing).
    Never invents specific training URLs or course names -- providers
    come from view.fallback_explain_gap_training_flat (which is sourced
    from the curated registry via the sub-step 3 projection).
    """
    # Priority 1: registry-grounded resources via sanitized view
    if view.fallback_explain_gap_training_flat:
        return _registry_grounded_explain_gap_fallback(inp, view)

    ctx = inp.conversation_context
    caps = inp.decision.caps_applied

    # ---- Priority 1: specific credential gap + role context (richest) ----
    if ctx and ctx.last_presented_credential_gaps:
        gap = ctx.last_presented_credential_gaps[0]
        if ctx.has_presented_context():
            titles = ctx.last_presented_job_titles
            role_phrase = (
                f"the {titles[0]} role"
                if len(titles) == 1
                else "those roles we just looked at"
            )
            return (
                f"For {role_phrase}, {gap} is the missing piece. "
                f"Your best move is to contact Sault Community Career "
                f"Centre — they can walk you through local training and "
                f"apprenticeship paths and tell you which counts toward "
                f"that credential. Sault College runs the trades programs "
                f"locally too. Want me to look at related roles in the "
                f"meantime, or focus on closing that gap first?"
            )
        # Gap known but no role context (rare path)
        return (
            f"The next step is {gap}. Sault Community Career Centre can "
            f"guide you through local training and apprenticeship paths. "
            f"Want me to suggest related roles you could pivot to while "
            f"you work on that?"
        )

    # ---- Priority 2: cap-flag fallback (no specific gap captured) ----
    if caps:
        cap = caps[0]
        cap_messages = {
            "band_capped_by_credential": (
                "The gap here is a required credential. Sault Community "
                "Career Centre can point you at local training and "
                "apprenticeship pathways — once you've got the credential, "
                "this lands closer to a strong match."
            ),
            "band_capped_by_no_experience": (
                "The skills line up, but I'm holding this at stretch until "
                "we have some work history on file. Happy to add anything "
                "you've done, or look at entry roles to build toward this."
            ),
            "band_capped_by_work_type_mismatch": (
                "The role's work type doesn't match what you mentioned "
                "wanting, so I'm flagging it as a stretch rather than a "
                "strong match. Want me to filter for your preferred work type?"
            ),
        }
        return cap_messages.get(cap, (
            f"This match has been demoted because of a known limitation "
            f"({cap}). Sault Community Career Centre can help you map a "
            f"path forward — tell me if you want to dig into it."
        ))

    # ---- Priority 3: nothing concrete to point at ----
    return (
        "Happy to walk through what's holding this back. Sault Community "
        "Career Centre can help you map a training path — tell me which "
        "gap you'd like to dig into."
    )


def _present_near_miss_fallback_v2(inp: ResponderV2Input) -> str:
    """Deterministic narrator for the present_near_miss outcome
    (Slice N, 2026-06-05).

    Shape:
      1. Open: name the role, anchor to SSM, state it WAS found.
      2. Be plain: it's not a realistic match yet.
      3. Surface gaps -- credentials first, then a short skill list.
         Use the payload's lists VERBATIM (handler already capped +
         ordered them per Q4).
      4. If `training_by_job` has an entry for the lead credential,
         name the provider verbatim. Never invent a provider here;
         the policy regex would reject it anyway.
      5. Offer to walk through the lead credential's path.

    When `near_miss_payload` is missing (shouldn't happen in
    production -- handler controls this), the function returns a
    safe fallback that points at SCCC without inventing gap data.
    """
    payload = inp.near_miss_payload or {}
    role = (payload.get("role") or "").strip()
    credentials = tuple(payload.get("credential_gaps") or ())
    skills = tuple(payload.get("core_skill_gaps") or ())
    job_count = int(payload.get("job_count") or 0)

    # Defensive fallback: missing payload -> deterministic safe text
    # that doesn't invent gaps. Still points at SCCC so the user has
    # a next step. The handler-level test in N-5 will assert payload
    # is always populated for this outcome.
    if not role or (not credentials and not skills):
        return (
            "I found a posting close to what you're looking for in Sault "
            "Ste. Marie, but it's not a realistic match yet -- some "
            "required skills and credentials aren't in your profile. "
            "Sault Community Career Centre can help map a path forward."
        )

    # ---- Sentence 1: anchor to the dataset + role ----
    if job_count > 1:
        opener = (
            f"I found {job_count} {role} postings in Sault Ste. Marie, "
            "but they're not a realistic match yet."
        )
    else:
        opener = (
            f"I found a {role} posting in Sault Ste. Marie, but it's not "
            "a realistic match yet."
        )

    # ---- Sentence 2: name the gaps ----
    # Credentials first per the locked Q4 ordering. If only one
    # category has entries, narrate only that one (don't fake the
    # other with empty filler).
    parts: list[str] = [opener]
    if credentials and skills:
        parts.append(
            "The main blockers are credentials -- "
            + _join_for_prose(credentials)
            + " -- plus some core skills like "
            + _join_for_prose(skills) + "."
        )
    elif credentials:
        parts.append(
            "The main blockers are credentials: "
            + _join_for_prose(credentials) + "."
        )
    elif skills:
        parts.append(
            "The main blockers are core skills: "
            + _join_for_prose(skills) + "."
        )

    # ---- Sentence 3 (optional): provider for the lead credential ----
    # Reuses inp.training_by_job (existing field) -- the handler in
    # Slice N-5 populates it for the lead credential gap, same way
    # explain_gap turns work. If the registry didn't know the
    # credential, we say nothing rather than inventing a provider.
    lead_credential = credentials[0] if credentials else None
    if lead_credential:
        provider = _find_grounded_provider(
            lead_credential, inp.training_by_job,
        )
        if provider:
            parts.append(
                f"For {lead_credential}, {provider} is where to start."
            )

    # ---- Closing: offer to walk through ----
    if lead_credential:
        parts.append(
            f"Want to walk through the {lead_credential} path first?"
        )
    else:
        # Only skill gaps -- offer to discuss the strongest one
        parts.append(
            f"Want to look at how to build up {skills[0]}?"
        )

    return " ".join(parts)


def _join_for_prose(items: tuple[str, ...]) -> str:
    """Comma-list with Oxford-style 'and' before the last item. Used
    for the gap narration -- bullets would feel checklist-y, prose
    keeps the tone warm. Empty -> empty string; the caller pre-checks
    so this defensive case shouldn't surface."""
    items = tuple(s for s in items if s)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _find_grounded_provider(
    credential: str, training_by_job: dict[str, list[dict]],
) -> str | None:
    """Look through training_by_job for an entry whose for_gap (or
    for_skill) matches the credential. Returns the first provider
    name found, or None when no registry entry exists for this
    credential.

    Why this lookup: the responder MUST NOT name a provider that
    isn't grounded in the registry (the policy regex would reject
    it anyway). If the handler populated training_by_job with the
    registry's entries for the lead credential, the provider name
    is safe to surface. If not, we stay silent rather than invent.
    """
    target = credential.lower().strip()
    for entries in training_by_job.values():
        for entry in entries:
            for_field = (
                (entry.get("for_gap") or entry.get("for_skill") or "")
                .lower().strip()
            )
            if for_field == target:
                provider = entry.get("provider")
                if isinstance(provider, str) and provider.strip():
                    return provider.strip()
    return None


_KEEP_GOING_EXAMPLES_BY_CATEGORY: dict[str, str] = {
    "software": (
        "languages, frameworks, or projects you've shipped"
    ),
    "warehouse": (
        "equipment you've operated (forklift, pallet jack), "
        "shifts you've worked, or shipping/receiving experience"
    ),
    "healthcare": (
        "patient care, clinical certifications (PSW, RPN), or "
        "shifts you've worked"
    ),
    "retail": (
        "POS systems, customer-facing tasks, or shift coverage"
    ),
    "admin": (
        "Office tools (Excel, QuickBooks, Outlook), specific tasks "
        "like payroll or reconciliation, and years of experience"
    ),
    "trades": (
        "trade tickets you hold (310T, 310S, welding endorsements), "
        "tools you've worked with, or apprenticeship hours"
    ),
    "other": (
        "tools, software, or specific tasks you've worked with"
    ),
}


def _role_phrase_for_offer(target_role_text: str | None) -> str:
    """Return a brief role-aware phrase for the upload-offer template
    (Option A, 2026-06-16). Falls back to a generic phrase when no
    target is set. Used to substitute the literal target name into the
    canned message so the responder feels role-aware rather than
    one-size-fits-all."""
    if isinstance(target_role_text, str) and target_role_text.strip():
        return target_role_text.strip()
    return "your target"


def _present_no_match_fallback_v2(inp: ResponderV2Input) -> str:
    """Canonical no-match shape from the prompt. Deterministic version.

    Resume-upload offer (2026-06-16): when the handler signals
    `should_offer_resume_upload=True`, the message is reframed to
    acknowledge the thin-evidence cause and offer the resume-upload
    path BEFORE the SCCC referral. This corrects the previously
    misleading "no postings exist" framing for cases where the
    underlying truth was "evidence too thin to score the postings
    that DO exist."

    Role substitution (Option A, 2026-06-16): the literal
    `target_role_text` and a role-category-specific "keep going"
    example are woven in so accounting-clerk and truck-driver
    sessions get DIFFERENT no-match messages. Pre-fix, both rendered
    identical text ("the roles in your target ... tools, software,
    or tasks") which read as a canned card.
    """
    skill, count = inp.next_skill

    if inp.should_offer_resume_upload:
        role_phrase = _role_phrase_for_offer(inp.target_role_text)
        role_category = intake_priority.classify_role(inp.target_role_text)
        keep_going_examples = _KEEP_GOING_EXAMPLES_BY_CATEGORY.get(
            role_category, _KEEP_GOING_EXAMPLES_BY_CATEGORY["other"],
        )
        msg = (
            f"Based on what you've shared, I couldn't find a strong "
            f"fit for {role_phrase} in today's Sault Ste. Marie "
            f"postings. {role_phrase.capitalize()} roles around here "
            f"usually ask for more specific evidence than what we've "
            f"covered yet."
            "\n\n"
            "If you've got a CV or resume handy, you can upload it "
            "and I'll read more skills out of it — that often "
            "unlocks matches that thin chat input misses."
            "\n\n"
            f"Or if you'd rather keep going here, tell me about "
            f"{keep_going_examples}."
        )
        if skill and count:
            msg += (
                f" (Also: if you can pick up {skill}, around {count} "
                "more current jobs could open up.)"
            )
        msg += (
            "\n\n"
            "If none of that surfaces something, Sault Community "
            "Career Centre has access to more sources and can flag "
            "openings as they come in."
        )
        return msg

    # Slice 6 (locked 2026-06-29, Option 1 narrow text-only unlock):
    # the matching engine's no-match response was REPEATEDLY making
    # false claims and hollow offers ("I checked for related roles"
    # when the recommender was never invoked; "Want training
    # directions?" with no consume hook). Live verify caught this
    # multiple times -- including with the existing policy gate
    # rejecting the LLM and the deterministic fallback ALSO emitting
    # the same offers.
    #
    # Locked replacement: minimal 2-sentence honest text. Absence
    # statement + SCCC referral. NO related-role claim, NO training
    # offer, NO "do you want?" dead-end, NO market panorama
    # (total_active_jobs / top_sectors / top_employers were editorial
    # padding that contributed nothing to the user's accounting-clerk
    # question and added more drift surface).
    #
    # Also LLM bypass for present_no_match (see compose_response_v2)
    # -- this fallback is now the SOURCE OF TRUTH for no_match
    # responses, not just the safety net.
    target = (inp.target_role_text or "").strip()
    if target:
        absence = (
            f"I don't see any {target} postings in Sault Ste. Marie "
            f"today."
        )
    else:
        absence = (
            "I don't see matching postings in Sault Ste. Marie today."
        )
    referral = (
        "The Sault Community Career Centre has access to more "
        "sources and can flag openings as they come up."
    )
    return f"{absence} {referral}"


def _format_top_sectors_phrase(top_sectors: tuple[str, ...]) -> str:
    """Format up to 3 sector names as a coach-voice "mostly in X, Y,
    and Z" phrase. Returns empty string when no sectors are available
    (caller substitutes a sectors-free variant)."""
    if not top_sectors:
        return ""
    sectors = list(top_sectors[:3])
    if len(sectors) == 1:
        return f"mostly in {sectors[0]}"
    if len(sectors) == 2:
        return f"mostly in {sectors[0]} and {sectors[1]}"
    return f"mostly in {sectors[0]}, {sectors[1]}, and {sectors[2]}"


def _format_top_employers_phrase(top_employers: tuple[str, ...]) -> str:
    """Format up to 3 employer names as a coach-voice "X, Y, and Z
    are actively hiring" phrase. Returns empty string when no
    employers are available."""
    if not top_employers:
        return ""
    employers = list(top_employers[:3])
    if len(employers) == 1:
        return f"{employers[0]} is actively hiring."
    if len(employers) == 2:
        return f"{employers[0]} and {employers[1]} are actively hiring."
    return (
        f"{employers[0]}, {employers[1]}, and {employers[2]} are "
        f"actively hiring."
    )


def _present_matches_fallback_v2(
    inp: ResponderV2Input, view: SanitizedResponderView,
) -> str:
    """Same shape as v1's _present_matches_fallback but cap-aware via
    inp.decision.caps_applied. Names the cap inline rather than letting
    a demoted match read like an ordinary stretch.

    Sub-step 4: reads results and per-job training URLs from the
    SanitizedResponderView (`view.fallback_results` and
    `view.fallback_present_matches_training_by_job`). URL fields use
    SanitizedURL.raw for rendering; invalid source URLs are stripped
    at projection.
    """
    results = view.fallback_results
    if inp.band_signal == "none" or not results:
        return _present_no_match_fallback_v2(inp)

    caps = inp.decision.caps_applied
    cap_lead = ""
    if caps:
        cap = caps[0]
        cap_lead_map = {
            "band_capped_by_credential": (
                "These are stretch matches — a required credential is "
                "missing in each, so they'd land stronger once you've got it."
            ),
            "band_capped_by_no_experience": (
                "I'm flagging these as stretch until we have some work "
                "history on file — the skills line up, just no confirmed experience yet."
            ),
            "band_capped_by_work_type_mismatch": (
                "These are flagged as stretch — the role's work type "
                "doesn't quite match what you mentioned wanting."
            ),
        }
        cap_lead = cap_lead_map.get(cap, "")

    lead = cap_lead or (
        "Here are the most relevant Sault Ste. Marie jobs I can see right now:"
        if inp.band_signal != "stretch_only"
        else "I don't see a strong match today, but here are some stretch matches worth considering:"
    )
    lines = [lead]
    training_by_job = view.fallback_present_matches_training_by_job
    for r in results:
        lines.append(
            f"• {r.title or 'this role'} at "
            f"{r.employer or 'employer unspecified'} — "
            f"{r.match_band or 'match'} match."
        )
        if r.url is not None:
            lines.append(f"   {r.url.raw}")
        if r.credential_warning:
            lines.append(f"   Note: {r.credential_warning}")
        if r.missing_skills:
            lines.append(
                f"   Skills to build: {', '.join(r.missing_skills[:3])}"
            )
        if r.job_id:
            for t in training_by_job.get(r.job_id, ()):
                tline = f"   Try: {t.title}"
                if t.provider:
                    tline += f" ({t.provider})"
                if t.url is not None:
                    tline += f" — {t.url.raw}"
                lines.append(tline)
    skill, count = inp.next_skill
    if skill and count:
        lines.append(
            f"\nIf you build {skill}, around {count} more current jobs "
            "could open up."
        )
    return "\n".join(lines)


# =========================================================================
# AR-9.feat.coach-tiers CP2 step 3 — tiered-matches responder branch
#
# Lives at the END of responder.py so the new branch is visibly separate
# from the legacy v2 dispatch. It does NOT touch:
#   - results / training_by_job (legacy fields; unused on this branch)
#   - build_sanitized_responder_view_v2 (legacy view builder)
#   - OUTCOME_RESPONDER_PROMPT (legacy outcome prompt)
#   - _policy_ok_v2 (legacy policy gate; tier-aware sibling below)
#   - _check_ungrounded_provider (legacy provider gate; tier-aware
#     sibling via coach_tiers_policy)
#
# It DOES read inp.tier_evidence + inp.pipeline_snapshot — handler-
# populated CP1 outputs — and projects them through the tier view
# builder + deterministic fallback.
# =========================================================================
def _empty_tiered_view() -> SanitizedResponderView:
    """A view with all three tiers empty. Used when tier_evidence is
    missing (handler contract violation) so the deterministic empty-
    state body still renders without an LLM call."""
    return build_sanitized_responder_view_for_tiered_matches(
        TieredEvidence(apply_today=(), worth_a_try=(), sideways_move=()),
    )


def _serialize_job_facts_for_tiered(facts: Any) -> dict[str, Any]:
    """Serialize PromptJobFacts. salary_text is INTENTIONALLY omitted
    per step-11 option B — salary is preserved on the view for future
    use but never surfaces in generated text."""
    return {
        "posted_date": (
            facts.posted_date.isoformat() if facts.posted_date else None
        ),
        "posted_days_ago": facts.posted_days_ago,
        "location": facts.location,
        "employment_type": facts.employment_type,
        # salary_text deliberately omitted — see step-11 option B.
    }


def _serialize_alignment_for_tiered(a: Any) -> dict[str, Any]:
    return {
        "user_skill": a.user_skill,
        "job_requirement": a.job_requirement,
        "stage": a.stage,
        "source": a.source,
        "is_normalized_equal": a.is_normalized_equal,
    }


def _serialize_training_option_for_tiered(t: Any) -> dict[str, Any]:
    return {
        "provider": t.provider,
        "title": t.title,
        "url": t.url.raw if t.url else None,
        "format": t.format,
        "duration_text": t.duration_text,
    }


def _serialize_prioritized_gap_for_tiered(g: Any) -> dict[str, Any]:
    return {
        "job_requirement": g.job_requirement,
        "category": g.category,
        "priority": g.priority,
        "blocker": g.blocker,
        "training_options": [
            _serialize_training_option_for_tiered(t)
            for t in g.training_options
        ],
    }


def _serialize_non_blocking_gap_for_tiered(g: Any) -> dict[str, Any]:
    return {"job_requirement": g.job_requirement, "material": g.material}


def _serialize_transferable_pair_for_tiered(p: Any) -> dict[str, Any]:
    return {
        "user_skill": p.user_skill,
        "applies_to": p.applies_to,
        "stage": p.stage,
    }


def _serialize_strong_for_tiered(item: Any) -> dict[str, Any]:
    return {
        "job_id": item.job_id,
        "title": item.title,
        "employer": item.employer,
        "location": item.location,
        "url": item.url.raw if item.url else None,
        "job_facts": _serialize_job_facts_for_tiered(item.job_facts),
        "skill_alignment": [
            _serialize_alignment_for_tiered(a) for a in item.skill_alignment
        ],
        "non_blocking_gaps": [
            _serialize_non_blocking_gap_for_tiered(g)
            for g in item.non_blocking_gaps
        ],
        "credential_warning_text": item.credential_warning_text,
        "strength_claim_text": item.strength_claim_text,
    }


def _serialize_stretch_for_tiered(item: Any) -> dict[str, Any]:
    return {
        "job_id": item.job_id,
        "title": item.title,
        "employer": item.employer,
        "location": item.location,
        "url": item.url.raw if item.url else None,
        "job_facts": _serialize_job_facts_for_tiered(item.job_facts),
        "skill_alignment": [
            _serialize_alignment_for_tiered(a) for a in item.skill_alignment
        ],
        "prioritized_gaps": [
            _serialize_prioritized_gap_for_tiered(g)
            for g in item.prioritized_gaps
        ],
        "credential_warning_text": item.credential_warning_text,
        "strength_claim_text": item.strength_claim_text,
    }


def _serialize_adjacent_for_tiered(item: Any) -> dict[str, Any]:
    return {
        "job_id": item.job_id,
        "title": item.title,
        "employer": item.employer,
        "location": item.location,
        "url": item.url.raw if item.url else None,
        "job_facts": _serialize_job_facts_for_tiered(item.job_facts),
        "skill_alignment": [
            _serialize_alignment_for_tiered(a) for a in item.skill_alignment
        ],
        "transferable_pairs": [
            _serialize_transferable_pair_for_tiered(p)
            for p in item.transferable_pairs
        ],
        "important_gaps": list(item.important_gaps),
        "credential_warning_text": item.credential_warning_text,
        "why_adjacent": item.why_adjacent,
        "strength_claim_text": item.strength_claim_text,
    }


def _build_user_block_for_tiered_matches(
    inp: ResponderV2Input, view: SanitizedResponderView,
) -> str:
    """Serialize the tiered-matches input for the LLM. Matches the
    EVIDENCE PACKAGE schema documented in COACH_TIERS_RESPONDER_PROMPT.
    Reads ONLY the view's prompt_tiered_* slots and the snapshot.

    Deliberately omitted:
      - results / training_by_job / band_signal — legacy fields
      - score_explanation / next_skill — legacy fields
      - arbiter_action / notes (per V2 prompt discipline)
      - decision.tone / reason_code (the new prompt does not consume
        these; the tier records' strength_claim_text drives the voice)
      - salary_text (option B)
    """
    parts: list[str] = []
    parts.append(f"USER_MESSAGE:\n{inp.user_message}\n")
    if inp.target_role_text:
        parts.append(f"TARGET_ROLE: {inp.target_role_text}")

    # Gap 1 (2026-06-16 evening, user-signed-off): when the engine
    # surfaces only stretch-tier matches AND the user has no resume,
    # the upload offer should weave into the response just like it
    # does on present_no_match turns. The handler-level
    # `_should_offer_resume_upload` gate is already band-aware
    # (fires on `band_signal in {low_only, stretch_only}` even when
    # results > 0). Threading the flag here lets the LLM mention
    # "uploading a resume might unlock a stronger match" alongside
    # the stretch-tier card.
    if inp.should_offer_resume_upload:
        parts.append("RESUME_UPLOAD_OFFER: yes")

    # scoring-v6 (2026-06-17): the apply_today slot still carries
    # both Strong (competitive_match) and Good (strongest_current)
    # band records, but the response renders them under DISTINCT
    # headings per the closing-matrix v2 design. Split them at the
    # serialization boundary so the LLM sees two clean sections
    # rather than one mixed bag — keeps the prompt's heading rules
    # straightforward (records in STRONG_MATCHES → Strong heading;
    # records in GOOD_MATCHES → Good heading).
    parts.append("STRONG_MATCHES:")
    for item in view.prompt_tiered_apply_today:
        if item.strength_claim_text == "competitive_match":
            parts.append(json.dumps(_serialize_strong_for_tiered(item)))

    parts.append("GOOD_MATCHES:")
    for item in view.prompt_tiered_apply_today:
        if item.strength_claim_text == "strongest_current":
            parts.append(json.dumps(_serialize_strong_for_tiered(item)))

    parts.append("STRETCH_MATCHES:")
    for item in view.prompt_tiered_worth_a_try:
        parts.append(json.dumps(_serialize_stretch_for_tiered(item)))

    # scoring-v6 (2026-06-17): the new fourth direct-target section.
    # Records the classifier labeled "explore_later" — surfaced under
    # the "Explore later — not your main target" heading rather than
    # hidden by the responder's old eligible-only-low branch.
    parts.append("EXPLORE_LATER:")
    for item in view.prompt_tiered_explore_later:
        parts.append(json.dumps(_serialize_stretch_for_tiered(item)))

    parts.append("ADJACENT_JOBS:")
    for item in view.prompt_tiered_sideways_move:
        parts.append(json.dumps(_serialize_adjacent_for_tiered(item)))

    if inp.pipeline_snapshot is not None:
        parts.append("PIPELINE_SNAPSHOT:\n" + json.dumps({
            "total_active_jobs": inp.pipeline_snapshot.total_active_jobs,
            "last_publish_at_text": (
                inp.pipeline_snapshot.last_publish_at_text
            ),
        }))

    return "\n".join(parts)


def _policy_ok_tiered_matches(
    reply: str, inp: ResponderV2Input, view: SanitizedResponderView,
) -> bool:
    """Tiered-matches policy gate (CP2 step 6.4 — minimal validator).

    Authorised the LLM to write the response from the evidence
    naturally; the validator stopped policing prose. The rules that
    remain are SAFETY rules — they prevent the LLM from making
    claims outside the evidence, not from choosing how to phrase a
    grounded one:

      1. URL grounding: every URL in the reply must canonicalize to
         a member of `view.prompt_urls`. No fake links.
      2. Ungrounded provider rejected (tier-view GROUNDED_TERMS).
      3. Consent promises rejected when requires_consent.
      4. Salary mentions rejected ($/`/hr`/`/hour`).
      5. Forbidden corpora rejected (job bank, statcan, etc.).
      6. Out-of-region / non-local city offers rejected.
      7. Credential-equivalence claims rejected.
      8. Immigration / legal advice rejected.
      9. Operational leakage rejected (arbiter_action, etc.).
     10. Internal evidence-package tokens rejected (field names like
         `skill_alignment` or enum values like `competitive_match`).
     11. Forbidden achievability words rejected (`perfect match`,
         `guaranteed`, `ideal candidate`, etc.).
     12. Reply must end with `?`.

    Heading exactness, strength-phrase exactness, paragraph
    ownership, gap-text exactness, training-sentence exactness,
    closing-from-closed-set, and per-tier blocker/first-gap
    enforcement are NO LONGER ENFORCED. The LLM composes naturally
    from the evidence package; the deterministic fallback is the
    last-ditch path when the LLM is unreachable.
    """
    if not reply.strip():
        return False

    # URL grounding — every URL in the reply must canonicalize to a
    # member of view.prompt_urls (move-gated allowlist).
    for candidate in extract_url_candidates(reply):
        result = check_url_membership(
            candidate.extracted_token, view.prompt_urls,
        )
        if isinstance(result, Violation):
            fields = safe_telemetry_fields(
                result, move=inp.decision.final_move,
            )
            log.warning(
                "policy tiered_matches: URL grounding violation "
                "code=%s move=%s scheme=%s host=%s hash=%s",
                fields["violation_code"], fields["move"],
                fields["scheme"], fields["host"], fields["url_hash"],
            )
            return False

    lower = reply.lower()

    if inp.requires_consent:
        bad_promises = (
            "i'll remember", "i will remember", "i'll save your profile",
            "i'll keep this for you", "next time i'll",
        )
        if any(p in lower for p in bad_promises):
            return False

    # Salary defense-in-depth (step-11 option B omits salary from the
    # surfaces, so the LLM shouldn't emit any pay reference).
    # Step-3 review High: parity with `_policy_ok_v2`'s salary
    # detection — `/hr` and `/hour` (the v2 source carries `\hr` /
    # `\hour` literals that never match real text; the intended
    # detection is the forward-slash form, which the new gate uses).
    if "$" in reply or "/hr" in lower or "/hour" in lower:
        return False

    forbidden = (
        "job bank", "statistics canada", "statcan", "national average",
    )
    if any(p in lower for p in forbidden):
        return False

    for pat in _OUT_OF_REGION_PATTERNS:
        if pat.search(reply):
            log.warning(
                "policy tiered_matches: reply offers out-of-region "
                "search (pattern=%s)", pat.pattern,
            )
            return False
    for pat in _NON_LOCAL_CITY_OFFER_PATTERNS:
        if pat.search(reply):
            log.warning(
                "policy tiered_matches: reply suggests non-local city "
                "(pattern=%s)", pat.pattern,
            )
            return False

    # Step-3 review High: credential-equivalence claims are out of
    # scope for the coach surface. Parity with `_policy_ok_v2`.
    for pat in _CREDENTIAL_EQUIVALENCE_PATTERNS:
        if pat.search(reply):
            log.warning(
                "policy tiered_matches: reply makes credential "
                "equivalence claim",
            )
            return False

    # Step-3 review High: immigration / legal scope is out of bounds.
    # Parity with `_policy_ok_v2`.
    for pat in _IMMIGRATION_LEGAL_PATTERNS:
        if pat.search(reply):
            log.warning(
                "policy tiered_matches: reply gives "
                "immigration/legal-tier advice",
            )
            return False

    # Step-3 review High: operational leakage. Parity with
    # `_policy_ok_v2`. The tiered surface also forbids the new move
    # token "resolved_to_tiered_matches."
    _OPERATIONAL_LEAKAGE = (
        "arbiter_action", "overrode the planner", "the planner said",
        "the arbiter decided", "fallback_to_legacy",
        "resolved_to_tiered_matches",
    )
    for term in _OPERATIONAL_LEAKAGE:
        if term in lower:
            log.warning(
                "policy tiered_matches: reply leaks operational term %r",
                term,
            )
            return False

    # CP2 step 6.4 — provider-grounding check removed. It was matching
    # the user's own skills (e.g. "QuickBooks") as if they were
    # hallucinated training-provider names, rejecting the LLM for
    # faithfully citing evidence. URL allowlist remains the primary
    # anti-hallucination defense: a fake training provider needs a
    # fake URL to be actionable, and the URL allowlist catches that.

    # CP2 step 6.4 — internal-token leakage: closed-vocab field names
    # and enum values from the evidence package never belong in user
    # prose. Case-insensitive sweep; the list is already lowercased.
    for token in _INTERNAL_TOKENS_FORBIDDEN_IN_REPLY:
        if token in lower:
            log.warning(
                "policy tiered_matches: reply leaks internal token "
                "(safe code only)",
            )
            return False

    # CP2 step 6.4 — forbidden achievability words. The LLM may not
    # promise outcomes ("guaranteed", "perfect match"); evidence
    # carries fit signals, not job promises.
    for word in _FORBIDDEN_ACHIEVABILITY_WORDS:
        if word in lower:
            log.warning(
                "policy tiered_matches: reply uses forbidden "
                "achievability word",
            )
            return False

    # CP2 step 6.4 — must end with a question (the surface is a
    # coach turn; the next-action is always a question to the user).
    if not reply.rstrip().endswith("?"):
        log.warning(
            "policy tiered_matches: reply does not end with a question",
        )
        return False

    return True


# All distinct closing-question strings. Three tier-presence profiles
# (all-three / apply+stretch / apply+sideways) collapse to the same
# closing string, so the DISTINCT set has 6 entries. Used by the
# structural validator to verify exactly one authorized closing
# appears in the reply.
_ALL_DISTINCT_CLOSINGS: frozenset[str] = frozenset({
    _CLOSING_ALL_TIERS,
    _CLOSING_APPLY_ONLY,
    _CLOSING_STRETCH_AND_SIDEWAYS,
    _CLOSING_STRETCH_ONLY,
    _CLOSING_SIDEWAYS_ONLY,
    _CLOSING_EMPTY,
})


# Internal tokens and field names that MUST NEVER appear literally in
# the response. Strength_claim_text values and why_adjacent values
# are the closed-vocab tokens the prompt routes on — their canonical
# wording lives in `_STRENGTH_PHRASES`. The other entries are field
# names from the evidence-package schema; the LLM is told these are
# data identifiers, not narration.
_INTERNAL_TOKENS_FORBIDDEN_IN_REPLY: tuple[str, ...] = (
    # strength_claim_text values
    "competitive_match",
    "strongest_current",
    "close_with_named_gap",
    "stretch_with_training_bridge",
    "transferable_lane",
    # why_adjacent values
    "same_noc_minor_group",
    "skill_evidence",
    # Evidence-package field names
    "skill_alignment",
    "prioritized_gaps",
    "non_blocking_gaps",
    "transferable_pairs",
    "important_gaps",
    "why_adjacent",
    "strength_claim_text",
    "job_facts",
    "is_normalized_equal",
)


# CP2 step 6.4 — forbidden achievability words. Short, closed list
# of phrases the LLM may never use because they make outcome
# promises the evidence does not support. Anything else — phrasing,
# hedging, voice — is the LLM's call.
_FORBIDDEN_ACHIEVABILITY_WORDS: tuple[str, ...] = (
    "perfect match",
    "guaranteed",
    "ideal candidate",
    "you'll get the job",
    "you will get the job",
    "definitely qualified",
    "definitely a fit",
    "100% match",
)


# CP2 step 6.4 — shape validation is gone. The LLM composes
# naturally from the evidence; the policy gate enforces only
# grounding + safety rules (URL allowlist, ungrounded provider,
# salary/region/legal/leakage, internal-token leakage, forbidden
# achievability words, ends-with-?). Old shape helpers removed.


def _compact_user_profile(inp: ResponderV2Input) -> dict[str, Any]:
    """Slice 2: derive a compact USER_PROFILE block for the
    recommender prompt. Carries the user's named skills, a flag for
    whether a resume is attached, and a short work-history summary
    when resume_facts has one.

    The LLM uses this to ground sentences like "your bookkeeping
    experience already lines up here" -- without it, the LLM has no
    evidence about what the user actually brings.
    """
    facts = inp.resume_facts or {}
    named_skills: list[str] = []
    raw_skills = facts.get("skills") or []
    if isinstance(raw_skills, list):
        for s in raw_skills:
            if isinstance(s, dict):
                name = s.get("name") or s.get("skill_name")
            else:
                name = s
            if isinstance(name, str) and name.strip():
                named_skills.append(name.strip())

    # Compact work history summary -- one short string per role with
    # title/employer/year-range so the LLM can reference real entries
    # without us shipping the full resume_facts blob.
    work_history_summary: list[str] = []
    raw_wh = facts.get("work_history") or []
    if isinstance(raw_wh, list):
        for entry in raw_wh:
            if not isinstance(entry, dict):
                continue
            title = entry.get("job_title") or entry.get("title") or ""
            employer = entry.get("employer") or entry.get("company") or ""
            start = entry.get("start_date") or entry.get("start") or ""
            end = entry.get("end_date") or entry.get("end") or ""
            parts = [p for p in (title, employer) if isinstance(p, str) and p.strip()]
            base = " at ".join(parts) if parts else ""
            if start or end:
                base = f"{base} ({start}-{end})" if base else f"({start}-{end})"
            if base:
                work_history_summary.append(base)

    has_resume = bool(
        facts.get("skills")
        or facts.get("work_history")
        or facts.get("education")
        or facts.get("certifications")
        or facts.get("projects")
        or facts.get("languages")
    )

    return {
        "named_skills": named_skills,
        "has_resume": has_resume,
        "work_history_summary": work_history_summary,
    }


def _build_user_block_for_recommender(inp: ResponderV2Input) -> str:
    """Serialize the recommender evidence into the prompt's EVIDENCE
    PACKAGE block.

    Slice 2 (locked 2026-06-23): richer per-layer shape so the LLM
    can REASON from evidence rather than restate a generic list:

      Always present:
        USER_MESSAGE     - user's current turn (verbatim)
        TARGET_ROLE_TEXT - user's stated role, e.g. "accounting clerk"
                           (NOT the OaSIS source_label/title -- those
                           are background context only)
        USER_PROFILE     - {named_skills, has_resume, work_history_summary}
                           the user's evidence to reason FROM
        MODE             - "local_gap_coach" | "target_noc_standard" | "adjacent_noc_standard"
        VOICE_HINT       - CareerIntent string from the router; lets
                           Layer B differentiate skill_gap vs
                           training_recommendation framing

      Mode-specific evidence:
        local_gap_coach (Layer B):
          LAYER_B_EVIDENCE = {primary_gap, lead_posting, user_overlapping_skills}
          LAYER_B_TRAINING = [{provider, summary, url}]
        target_noc_standard (Layer A):
          LAYER_A_EVIDENCE = {noc_code, oasis_title, top_development_areas}
        adjacent_noc_standard (Layer C):
          LAYER_C_EVIDENCE = [{noc_code, noc_title, development_areas}]

    The prompt's anti-template guards instruct the LLM to combine
    these fields in coach prose, not list them as bullets.
    """
    rec = inp.recommendation_evidence
    assert rec is not None  # caller guards

    profile = _compact_user_profile(inp)
    payload: dict[str, Any] = {
        "USER_MESSAGE": inp.user_message,
        "TARGET_ROLE_TEXT": inp.target_role_text or "",
        "USER_PROFILE": profile,
        "MODE": rec.mode,
        "VOICE_HINT": (
            inp.recommender_voice_hint
            if isinstance(inp.recommender_voice_hint, str)
            else None
        ),
    }

    if rec.mode == "local_gap_coach":
        # Layer B's GapEvidence record carries source_label (posting
        # title) and source_id (job_id) directly from the matched
        # posting. employer and noc_code are not on GapEvidence; the
        # LLM can reference "the local bookkeeper posting" without
        # them. user_overlapping_skills is left for the LLM to reason
        # from USER_PROFILE.named_skills + LAYER_B_EVIDENCE.primary_gap
        # (the prompt instructs this combination).
        primary_gap = rec.evidence[0] if rec.evidence else None
        layer_b_ev: dict[str, Any] = {
            "primary_gap": primary_gap.skill_name if primary_gap else None,
            "primary_gap_skill_id": (
                primary_gap.skill_id if primary_gap else None
            ),
            "lead_posting": (
                {
                    "title": primary_gap.source_label,
                    "job_id": primary_gap.source_id,
                }
                if primary_gap is not None
                else None
            ),
        }
        payload["LAYER_B_EVIDENCE"] = layer_b_ev
        payload["LAYER_B_TRAINING"] = [
            {
                "provider": t.provider,
                "summary": t.summary,
                "url": t.url,
            }
            for t in rec.training
        ]
    elif rec.mode == "target_noc_standard":
        # All evidence records share a NOC source; pick first for
        # noc_code/oasis_title context.
        first = rec.evidence[0] if rec.evidence else None
        payload["LAYER_A_EVIDENCE"] = {
            "noc_code": first.source_id if first else None,
            "oasis_title": first.source_label if first else None,
            "top_development_areas": [
                {
                    "name": g.skill_name,
                    "importance": g.importance,
                }
                for g in rec.evidence
            ],
        }
    elif rec.mode == "adjacent_noc_standard":
        # Group evidence records by NOC, preserving first-seen order
        # (assembly already capped per-NOC at top-3 by importance).
        grouped: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for g in rec.evidence:
            key = g.source_id or "unknown"
            if key not in grouped:
                grouped[key] = {
                    "noc_code": g.source_id,
                    "noc_title": g.source_label or key,
                    "development_areas": [],
                }
                order.append(key)
            grouped[key]["development_areas"].append(g.skill_name)
        payload["LAYER_C_EVIDENCE"] = [grouped[k] for k in order]

    return json.dumps(payload, ensure_ascii=False, indent=2)


def _compose_recommender_response(inp: ResponderV2Input) -> str:
    """Render the conversational recommender surface -- LLM-first with
    deterministic per-mode fallback.

    Falls back to render_recommender_fallback when:
      - LLM is disabled at config level;
      - LLM returns an empty / falsy reply.

    No policy gate in this slice: the recommender wrapper carries only
    registry-verified URLs (surface_url(today) is enforced at
    wrapper-assembly time), and the prompt's CRITICAL HARD RULE
    constrains cross-mode leakage. A dedicated recommender policy
    gate can be added in a later slice when live traffic surfaces
    failure modes worth catching at policy level.
    """
    rec = inp.recommendation_evidence
    if rec is None:
        log.error(
            "responder: _compose_recommender_response invoked without "
            "recommendation_evidence; handler contract violation.",
        )
        return render_recommender_fallback(None)

    if not is_enabled():
        return render_recommender_fallback(
            rec, target_role_text=inp.target_role_text,
        )

    user_block = _build_user_block_for_recommender(inp)
    reply = call(RECOMMENDER_RESPONDER_PROMPT, user_block, max_tokens=800)
    if not reply:
        return render_recommender_fallback(
            rec, target_role_text=inp.target_role_text,
        )

    return reply


def _compose_tiered_matches_response(inp: ResponderV2Input) -> str:
    """Render the present_tiered_matches surface — LLM-first with
    deterministic tier-structured fallback.

    Falls back to the deterministic renderer when:
      - tier_evidence is missing (handler contract violation);
      - LLM is disabled at config level;
      - LLM returns empty / falsy reply;
      - policy gate rejects the reply.
    """
    if inp.tier_evidence is None:
        log.error(
            "responder: present_tiered_matches dispatched without "
            "tier_evidence; handler contract violation. Rendering "
            "empty-state body.",
        )
        view = _empty_tiered_view()
        text, _ = render_coach_tiers_fallback(view, inp.pipeline_snapshot)
        return text

    view = build_sanitized_responder_view_for_tiered_matches(
        inp.tier_evidence,
    )

    if not is_enabled():
        text, _ = render_coach_tiers_fallback(view, inp.pipeline_snapshot)
        return text

    user_block = _build_user_block_for_tiered_matches(inp, view)
    reply = call(COACH_TIERS_RESPONDER_PROMPT, user_block, max_tokens=800)
    if not reply:
        text, _ = render_coach_tiers_fallback(view, inp.pipeline_snapshot)
        return text

    if not _policy_ok_tiered_matches(reply, inp, view):
        # Debug aid (2026-06-16 evening): log the actual rejected reply
        # tail so we can diagnose WHY the policy gate rejected it. The
        # specific warning above tells us WHICH rule fired; this tail
        # tells us what Haiku actually wrote so we can fix the prompt
        # if its closing instructions aren't landing.
        log.warning(
            "Responder tiered_matches reply failed policy check; "
            "falling back. last_120_chars_of_reply=%r",
            reply[-120:] if reply else "",
        )
        text, _ = render_coach_tiers_fallback(view, inp.pipeline_snapshot)
        return text

    return reply


__all__ = [
    "ResponderInput",
    "compose_reply",
    # Slice 5 -- v2 outcome-move responder
    "ResponderV2Input",
    "compose_response_v2",
]
