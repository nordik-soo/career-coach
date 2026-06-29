"""Chat message handler — PR 10 (Guided Intake + Profile-Based Matching).

Two paths, never crossed:

  ANONYMOUS PATH (pre-consent)
    1. Load StagedProfile from session store (or create new).
    2. Run evidence-bound extractor on user message; drop ungrounded fields.
    3. Merge fields + skills into staged blob (skipping declined slots).
    4. Run state-machine decide() to pick NEXT_ACTION.
    5. If action is PRESENT_MATCHES, compute in-memory match against
       core.v_current_job.
    6. Compose reply via NEXT_ACTION responder (LLM-narrated, deterministic
       fallback).
    7. Save staged blob back to the session store.
    8. Return reply + recommendations + session_id. WRITES NOTHING to
       profile.*, interaction.*, or analytics.*.

  AUTHENTICATED PATH (post-consent)
    Same loop, but state is implicit (profile_update_loop) and writes go
    to profile.user_profile / user_skill / interaction.chat_event /
    analytics.job_match.

The consent grant endpoint (routes/profiles.py via grant_consent below)
atomically flushes the StagedProfile to Postgres.
"""
from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # CareerIntent is a Literal type used only in annotations on the
    # intent dispatcher; runtime import lives inside the routing helper
    # to keep handler.py's import surface tight.
    from skillbridge.chat.recommender_intent import CareerIntent

from config import (
    CHAT_ORCHESTRATOR,
    MESSAGE_UNDERSTANDING_ENABLED,
    TRAINING_REGISTRY_ENABLED,
)
from skillbridge.chat import extractor as chat_extractor
from skillbridge.chat import gates as chat_gates
from skillbridge.chat import intake_state
from skillbridge.chat.arbiter import (
    ArbiterDecision,
    RunEngine,
    resolve_match_outcome,
    validate_planner_intent,
)
from skillbridge.chat.message_understanding import understand_message
from skillbridge.chat.planner import plan_next_move
from skillbridge.chat.routing import route_from_understanding
from skillbridge.match.near_miss import (
    build_near_miss_payload,
    filter_near_miss_candidates,
)
from skillbridge.chat.responder import (
    ConversationContext,
    ResponderInput,
    ResponderV2Input,
    compose_reply,
    compose_response_v2,
)
from skillbridge.chat.truth_summary import build_truth_summary
from skillbridge.db import sync_cursor
from skillbridge.extract import default_extractor
from skillbridge.extract.base import ExtractedSkill
from skillbridge.match import engine as match_engine
from skillbridge.match.recommend import suggest_for_skill
from skillbridge.resume import (
    derive_with_suppressions as resume_derive_with_suppressions,
    extract_resume_facts,
    parse_resume,
)
from skillbridge.session import get_store
from skillbridge.session.staging import (
    MAX_CANONICAL_CHARS as _stg_MAX_CANONICAL_CHARS,
    MAX_CRED_GAPS as _stg_MAX_CRED_GAPS,
    MAX_EMPLOYER_CHARS as _stg_MAX_EMPLOYER_CHARS,
    MAX_JOB_ID_CHARS as _stg_MAX_JOB_ID_CHARS,
    MAX_OTHER_JOBS as _stg_MAX_OTHER_JOBS,
    MAX_PRESENTED_JOB_IDS as _stg_MAX_PRESENTED_JOB_IDS,
    MAX_SKILL_GAPS as _stg_MAX_SKILL_GAPS,
    MAX_TITLE_CHARS as _stg_MAX_TITLE_CHARS,
    StagedProfile,
    StagedSkill,
)

log = logging.getLogger(__name__)

MAX_PREVIEW_JOBS = 5


# =========================================================================
# Resume upload helper — parse → extract → derive into staged profile
# =========================================================================
def _apply_resume_upload(
    staged: StagedProfile,
    file_bytes: bytes,
    filename: str | None,
) -> dict[str, Any]:
    """Run the resume pipeline and merge results into the staged profile.

    Returns a dict describing what happened (which the route surfaces in
    the API response so the frontend can show parse_warning to the user).

    On parser-level failure (oversize, unsupported, scan-only PDF,
    no_text, etc.) the staged profile's facts are NOT populated, but
    `resume_parse_warning` and `resume_filename` ARE written so the
    truth-summary classifier can distinguish failed-parse from
    no-upload across turns (chat orchestration v2 slice 1 review fix).
    A successful re-upload clears `resume_parse_warning` back to None.
    """
    parse_result = parse_resume(file_bytes, filename)

    if parse_result.parse_warning in ("too_large", "empty_input",
                                       "unsupported_format", "parse_failed"):
        log.info(
            "resume_upload session=%s parse_warning=%s file=%s bytes=%d",
            staged.session_id[:8], parse_result.parse_warning,
            parse_result.filename, parse_result.byte_count,
        )
        # Chat orchestration v2 slice 1 review fix: persist the warning
        # on the staged profile so the truth-summary classifier can
        # distinguish failed-parse from no-upload across turns. The
        # filename is recorded too so the responder can reference what
        # the user tried to upload.
        staged.resume_parse_warning = parse_result.parse_warning
        if parse_result.filename:
            staged.resume_filename = parse_result.filename
        return {
            "parsed": False,
            "parse_warning": parse_result.parse_warning,
            "filename": parse_result.filename,
        }

    # 'no_text' means we extracted but got too few characters (typically a
    # scanned PDF). Don't try to extract facts; ask the user to paste text.
    if parse_result.parse_warning == "no_text":
        log.info("resume_upload session=%s no_text file=%s",
                 staged.session_id[:8], parse_result.filename)
        staged.resume_parse_warning = "no_text"
        if parse_result.filename:
            staged.resume_filename = parse_result.filename
        return {
            "parsed": False,
            "parse_warning": "no_text",
            "filename": parse_result.filename,
        }

    # Successful text extraction — run the evidence-bound facts extractor.
    extraction = extract_resume_facts(parse_result.text)

    # Stash the raw text and parsed facts on the staged blob. In cookie
    # mode these get redacted out at serialize time (see staging.to_json).
    # In Redis mode they persist for the session TTL.
    from datetime import datetime, timezone
    staged.resume_text = parse_result.text
    staged.resume_filename = parse_result.filename
    staged.resume_parsed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    staged.resume_facts_json = extraction.facts
    # Successful upload supersedes any prior parse failure. Clear the
    # warning so the truth-summary classifier moves out of the "failed"
    # bucket on the next turn.
    staged.resume_parse_warning = None

    # New upload supersedes any prior parse. Suppressions from a previous
    # parse pointed at fact_ids that no longer exist, so drop them.
    # Preferences (declined_slots) survive — they're per-user, not per-parse.
    staged.suppressed_fact_ids = []

    # Derive flat slots from the facts JSON and merge into the staged
    # profile. The derive helper applies confidence floor + suppressions
    # (none right now, but future corrections feed through the same path).
    derived = resume_derive_with_suppressions(
        extraction.facts, staged.suppressed_fact_ids
    )
    # A new resume upload SUPERSEDES any prior resume's contribution.
    # Without this, uploading a second resume in the same session would
    # union the new skills with the previous resume's -- the matcher
    # would then score against the union, producing false-positive
    # "strong match" results (e.g. a data analyst's session that
    # previously held a truck-tech resume would still match truck jobs).
    # Chat-derived skills (source != "resume") are preserved, mirroring
    # the same pattern used by _refresh_derived_into_staged on suppression.
    staged.skills = [s for s in staged.skills if s.source != "resume"]
    _merge_derived_into_staged(staged, derived)

    log.info(
        "resume_upload session=%s file=%s bytes=%d "
        "text_chars=%d skills=%d work=%d edu=%d cert=%d langs=%d dropped=%d",
        staged.session_id[:8], parse_result.filename, parse_result.byte_count,
        len(parse_result.text),
        len(extraction.facts.get("skills") or []),
        len(extraction.facts.get("work_history") or []),
        len(extraction.facts.get("education") or []),
        len(extraction.facts.get("certifications") or []),
        len(extraction.facts.get("languages") or []),
        len(extraction.raw_keys_dropped),
    )

    return {
        "parsed": True,
        "parse_warning": extraction.parse_warning,  # may be None or "llm_disabled" etc.
        "filename": parse_result.filename,
        "content_type": parse_result.content_type,
        "facts_counts": {
            "skills": len(extraction.facts.get("skills") or []),
            "work_history": len(extraction.facts.get("work_history") or []),
            "education": len(extraction.facts.get("education") or []),
            "certifications": len(extraction.facts.get("certifications") or []),
            "projects": len(extraction.facts.get("projects") or []),
            "languages": len(extraction.facts.get("languages") or []),
        },
    }


# =========================================================================
# RESUME_REVIEW helpers — detect when the user has anything to confirm,
# and parse natural-language suppression requests against the parsed facts
# =========================================================================
def _resume_facts_have_content(facts: dict[str, Any] | None) -> bool:
    """True when the parsed facts JSON has at least one useful entry.

    LLM-off testing produces an empty-but-valid facts shape; we don't want
    to fire RESUME_REVIEW in that case — there's nothing to review.
    """
    if not facts:
        return False
    return any(
        bool(facts.get(group))
        for group in ("skills", "work_history", "education", "certifications", "projects")
    ) or bool(facts.get("languages"))


def _effective_facts_view(staged: StagedProfile) -> dict[str, Any] | None:
    """Return the post-suppression view of resume_facts_json.

    This is what the responder receives — the user's perspective on
    what we know about them after corrections, not the raw parser
    record. Suppressed fact_ids are filtered out of every group.
    Returns None if no resume has been uploaded.
    """
    facts = staged.resume_facts_json
    if not facts:
        return None
    suppressed = set(staged.suppressed_fact_ids or [])
    if not suppressed:
        return facts

    out: dict[str, Any] = {
        "version": facts.get("version"),
        "extractor_version": facts.get("extractor_version"),
        "languages": list(facts.get("languages") or []),
        "summary_signals": dict(facts.get("summary_signals") or {}),
    }
    for group in ("skills", "work_history", "education", "certifications", "projects"):
        out[group] = [
            entry for entry in (facts.get(group) or [])
            if isinstance(entry, dict) and entry.get("fact_id") not in suppressed
        ]
    return out


_REMOVAL_VERBS = (
    "remove", "delete", "drop ", " drop", "scrap", "take out", "take off",
    "not my", "not mine", "never worked", "didn't work", "did not work",
    "that's wrong", "thats wrong", "wrong about", "incorrect",
    "not me", "that isn't", "that isnt",
)


def _detect_resume_suppressions(
    staged: StagedProfile, message: str,
) -> list[str]:
    """Return fact_ids the user wants suppressed, by keyword heuristic.

    Looks for a removal verb anywhere in the message, then fuzzy-matches
    the rest against fact names / titles / employers / credentials.

    This is intentionally a soft layer:
      - If a removal is missed, user can repeat their request more
        specifically.
      - If a fact gets wrongly suppressed, the user can re-upload and
        the suppression list is reset (see _apply_resume_upload).

    A future version could replace this with a small LLM call that maps
    the message + fact list to specific fact_ids more reliably. v1 stays
    deterministic to avoid an extra LLM round-trip on every review turn.
    """
    if not staged.resume_facts_json:
        return []

    text = (message or "").lower()
    if not any(verb in text for verb in _REMOVAL_VERBS):
        return []

    facts = staged.resume_facts_json
    suppressed: set[str] = set()

    def _maybe_suppress(fact: dict[str, Any], *fields: str) -> None:
        fid = fact.get("fact_id")
        if not fid or fid in suppressed:
            return
        for field_name in fields:
            value = (fact.get(field_name) or "").strip().lower()
            # Require >=3 chars to avoid trivial matches ("at" matching
            # "data scientist") and skip very common single words.
            if value and len(value) >= 3 and value in text:
                suppressed.add(fid)
                return

    for s in facts.get("skills") or []:
        _maybe_suppress(s, "name")
    for w in facts.get("work_history") or []:
        _maybe_suppress(w, "title", "employer")
    for e in facts.get("education") or []:
        _maybe_suppress(e, "credential", "institution")
    for c in facts.get("certifications") or []:
        _maybe_suppress(c, "name", "issuer")
    for p in facts.get("projects") or []:
        _maybe_suppress(p, "name")

    return sorted(suppressed)


def _merge_derived_into_staged(
    staged: StagedProfile, derived: dict[str, Any],
) -> None:
    """Apply derived flat slots onto the staged profile, preserving any
    values the user has already given via chat.

    Resume-derived skills are merged into staged.skills via the existing
    merge_skills union semantics. Skills already present (e.g. from a
    chat turn before the upload) are not downgraded.
    """
    # Text slots: only set if not already filled by the user via chat.
    # The user's own words take precedence over the resume's framing.
    for slot in ("skills_text", "experience_text", "education_text"):
        if not getattr(staged, slot, None):
            value = derived.get(slot)
            if value:
                setattr(staged, slot, value)

    # Skills: union with existing chat-derived skills. resume-source skills
    # are tagged source="resume" so grant_consent persists their provenance
    # correctly in profile.user_skill (matters for audit + future "where
    # did this skill come from?" UI). raw_phrase carries the verbatim
    # evidence so the responder can cite the resume when narrating a match.
    derived_skills = derived.get("skills") or []
    if derived_skills:
        new_staged_skills = [
            StagedSkill(
                skill_name=d["skill_name"],
                raw_phrase=d.get("raw_phrase"),
                confidence=float(d.get("confidence") or 0.7),
                source="resume",
            )
            for d in derived_skills
        ]
        staged.merge_skills(new_staged_skills)


def _refresh_derived_into_staged(
    staged: StagedProfile, derived: dict[str, Any],
) -> None:
    """Replace resume-sourced contributions with the post-suppression view.

    Unlike _merge_derived_into_staged (which keeps chat values intact),
    this is used after the user has suppressed a fact: the matcher must
    no longer see the suppressed skill, so we drop all resume-source
    skills and re-add from the suppression-filtered derivation. Chat-
    source skills are preserved.

    Text slots (skills_text / experience_text / education_text) are
    intentionally left alone. We can't reliably tell whether the current
    value came from chat or resume, and overwriting could clobber user
    input. The skills list is what the matcher consumes for scoring;
    text slots are mostly used as prompt context, where staleness has
    low impact.
    """
    # Drop resume-source skills; keep chat-source skills.
    staged.skills = [s for s in staged.skills if s.source != "resume"]

    # Re-add the (now suppression-filtered) resume skills.
    derived_skills = derived.get("skills") or []
    for d in derived_skills:
        staged.skills.append(StagedSkill(
            skill_name=d["skill_name"],
            raw_phrase=d.get("raw_phrase"),
            confidence=float(d.get("confidence") or 0.7),
            source="resume",
        ))


# =========================================================================
# Profile extraction wrapper — evidence-bound LLM + rule-based skill fallback
# =========================================================================
def _extract(message: str, *, asked_slots: list[str]) -> chat_extractor.ExtractionResult:
    """Evidence-bound LLM extraction.

    If the LLM is disabled or returns nothing usable, fall back to the
    rule-based skill extractor so the chat still progresses on simple
    keyword messages.

    Slot-answer guard (post-acceptance live finding): the rule-based
    fallback is over-eager on short replies that are clearly answering
    a single slot. Live test showed "Truck and coach technician role"
    producing 4 phantom skills via substring matching of DB skill
    canonical names against the short role-name text. Those phantom
    skills made `chat_skill_count >= 3` -> `usable_evidence_present =
    True` -> `enough_to_match = True`, and the arbiter approved the
    engine on a profile with no real evidence. The LLM extractor
    correctly returned empty for that input; the rule-based fallback
    is the cause.

    Fix: skip the rule-based fallback when the message is plausibly
    a short slot answer (the previous turn asked for a slot AND the
    reply is short AND lacks a skill-list marker). Real skill claims
    are longer ("I have 3 years of forklift and welding experience"),
    span multiple comma-separated items, or arrive without a recent
    slot question -- those still hit the rule-based path.
    """
    result = chat_extractor.extract(message, asked_slots=asked_slots)
    if not result.skills:
        if not _is_likely_slot_answer(message, asked_slots):
            fallback = default_extractor().extract_from_user_text(message)
            if fallback:
                result.skills = fallback
                # If we got rule-based skills but no LLM fields, only the skills
                # path saves us from "off_topic = True".
                if result.fields or result.declined:
                    pass
                else:
                    result.off_topic = False
    return result


def _blank_direct_tiers_for_pattern_2(tier_evidence):
    """Pattern 2 yes-consent display projection (Step 8 / 2026-06-17).

    When the user consents to the related-roles offer ("yes" to
    "want me to also look at related roles?"), the next turn must
    surface ONLY the Sideways tier — the user already saw their
    direct-target matches on the prior turn; this turn is the
    related-roles pivot they requested.

    Returns a new `TieredEvidence` with the direct-target tiers
    (apply_today, worth_a_try, explore_later) blanked and
    sideways_move preserved. This routes through the existing
    arbiter logic exactly like Pattern 3's auto-fire path —
    `_tier_evidence_has_any_records` stays True iff sideways_move
    has records, so the arbiter still chooses
    `present_tiered_matches`. The responder then sees a tier
    bundle with only Sideways populated and the prompt's
    PATTERN 3 rule kicks in for framing.

    If sideways_move is ALSO empty (no related roles found for
    this user's profile), the blanked TieredEvidence has all four
    tiers empty. `_tier_evidence_has_any_records` returns False
    and the handler falls back to `present_no_match` — which Step 9
    will enhance with the SSM market summary as the terminal anchor.
    """
    from skillbridge.chat.tiered_evidence import TieredEvidence

    return TieredEvidence(
        apply_today=(),
        worth_a_try=(),
        sideways_move=tier_evidence.sideways_move,
        explore_later=(),
    )


def _classify_pattern_2_reply(message: str) -> str:
    """Pattern 2 consent classifier (closing-matrix v2, Step 7b /
    Step 11b refactor, 2026-06-17).

    Classifies the user's reply to a Pattern 2 closing question
    ("want me to also look at related roles?") into one of:
      - "yes":   user consented; Step 8's blanking hook fires
      - "no":    user declined; conversation ends naturally
      - "other": user changed topic / asked a question / corrected
                 something — clear the flag, route normally

    Implementation note (Step 11b, 2026-06-17): the v1 of this
    classifier used a locked frozenset of exact-match replies
    ("yes", "yes please", "go ahead", ...) plus a brittle
    normalization (`strip().lower().rstrip(".!?,")`). Live verify
    on 2026-06-17 showed the failure: a user typing "yes. go ahead"
    classifies as "other" because the trailing rstrip doesn't chew
    the period AFTER "yes" — and "yes. go ahead" isn't in the
    locked set as a whole.

    Fix: delegate to the existing `_classify_intent` regex
    classifier in `truth_summary`, which has been battle-tested on
    the same kinds of natural-language replies via the planner
    layer. It correctly identified "yes. go ahead" as
    `impatient_proceed`. Reusing it gives:
      - zero new vocabulary maintenance (single source of truth)
      - free upgrades when the intent classifier improves
      - no risk of two classifiers disagreeing on the same message

    Signal → consent mapping (locked):
      confirming         → yes  ("yes", "alright", "looks good")
      impatient_proceed  → yes  ("go ahead", "show me", "just do it")
      declining          → no   ("no", "skip", "not now")
      asking_question    → other (user asked something else)
      asking_about_gap   → other (user asked about a skill gap)
      correcting         → other (user pivoted: "actually, ...")
      redirecting        → other (user changed topic)
      neutral            → other (no strong signal)
    """
    # Lazy import — consistent with the file's other cross-module
    # imports; no actual circular here.
    from skillbridge.chat.truth_summary import _classify_intent

    if not isinstance(message, str):
        return "other"
    intent = _classify_intent(message)
    if intent in ("confirming", "impatient_proceed"):
        return "yes"
    if intent == "declining":
        return "no"
    return "other"


# Slice 5 step 4 (2026-06-19): conversational recommender chain
# consume dispatch. See project_recommender_step4_implementation_lock
# memory for the locked design.
_VALID_RECOMMENDER_MODES: frozenset[str] = frozenset({
    "local_gap_coach",
    "target_noc_standard",
    "adjacent_noc_standard",
})

# Locked next-mode mapping: each mode advances the chain. None means
# the chain ENDS HERE (adjacent_noc_standard is terminal).
# Slice 2 (locked 2026-06-23): chain reassigned per peer-engine
# locked design. B -> C (offer related career paths after Layer B).
# C -> END (Layer C terminal natural follow-up). A -> END (A is
# intent-only, never reached via chain).
_RECOMMENDER_NEXT_MODE: dict[str, str | None] = {
    "local_gap_coach":       "adjacent_noc_standard",  # B -> offer C
    "adjacent_noc_standard": None,                     # C -> END
    "target_noc_standard":   None,                     # A -> END (no chain in)
}


# Slice 1 follow-up (2026-06-23): which CareerIntent values can be
# deferred across a substrate-fill turn. job_matching is never
# deferred (matching engine has its own intake). unclear is never
# deferred (no intent to remember). The other five are deferrable.
_DEFERRABLE_CAREER_INTENTS: frozenset[str] = frozenset({
    "local_skill_gap",
    "training_recommendation",
    "noc_standard_comparison",
    "career_exploration",
    "application_help_out_of_scope",
})


def _classify_recommender_consent(message: str) -> str:
    """Classify the user's reply to a recommender chain offer. Wraps
    the existing `_classify_pattern_2_reply` -- same yes/no/other
    semantics, same battle-tested classifier authority, same coach
    vocabulary set. Kept as a separate name so future divergence (if
    the chain ever needs different semantics) is a one-place edit."""
    return _classify_pattern_2_reply(message)


def _dispatch_recommender_consume(
    *,
    staged: StagedProfile,
    user_message: str,
    store,
    resume_info: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Dispatch one turn of the conversational recommender chain.

    Returns a complete response dict when the chain handled the turn,
    or None to fall through to normal flow (consent="other" or any
    structural-precondition failure).

    Locked design contract (see project_recommender_step4_implementation_lock):
      - On consent="yes": re-run the engine if mode == local_gap_coach;
        build the per-mode RecommenderEvidence via the corresponding
        helper from recommender_assembly; dispatch the responder with
        recommendation_evidence set; advance the chain to the next mode
        (or None at adjacent_noc_standard).
      - On consent="no": consume the flag + clear last_adjacent_nocs;
        render a soft acknowledgment and return.
      - On consent="other": leave the flag set and return None so the
        message routes through normal flow. The chain-bound TTL on
        target_role_text change clears the orphaned flag eventually.
    """
    if staged.pending_recommender_offer is None:
        return None
    mode = staged.pending_recommender_offer
    if mode not in _VALID_RECOMMENDER_MODES:
        # Defensive: a forged or stale flag value clears safely.
        staged.pending_recommender_offer = None
        return None

    consent = _classify_recommender_consent(user_message)
    log.info(
        "anon_chat session=%s recommender_consent=%s mode=%s",
        staged.session_id[:8], consent, mode,
    )

    if consent == "other":
        return None  # fall through to normal flow

    if consent == "no":
        staged.pending_recommender_offer = None
        staged.last_adjacent_nocs = ()
        staged.touch()
        new_session_id = store.save(staged)
        return {
            "reply": (
                "Got it -- happy to help if you change your mind. "
                "What else can I look into?"
            ),
            "profile_id": None,
            "session_id": new_session_id,
            "intake_state": staged.intake_state,
            "asked_slots": [],
            "next_action": intake_state.ACTION_ACKNOWLEDGE_AND_WAIT,
            "recommended_jobs": [],
            "next_skill_suggestion": None,
            "resume_info": resume_info,
            "requires_consent": True,
        }

    # consent == "yes" -> build evidence for the active mode + dispatch.
    from datetime import date
    from skillbridge.chat.gap_evidence import RecommenderEvidence
    from skillbridge.chat.recommender_assembly import (
        build_recommender_evidence_adjacent_noc_standard,
        build_recommender_evidence_local_gap_coach,
        build_recommender_evidence_target_noc_standard,
        filter_matches_to_target_family,
    )
    from skillbridge.chat.development_plan import compute_primary_gap_name
    from skillbridge.match.engine import (
        build_user_skill_rows,
        compute_matches_in_memory,
        derive_user_skill_sets,
    )

    user_rows = build_user_skill_rows(staged.skills)
    user_skill_ids, _names, _canon = derive_user_skill_sets(user_rows)

    rec_evidence: RecommenderEvidence | None = None
    try:
        if mode == "local_gap_coach":
            # Slice 2: target-NOC family filter BEFORE CP4. Engine's
            # top-5 by skill overlap can include off-target NOCs; CP4
            # would then pick a primary gap from an off-target posting.
            # Filter anchors Layer B on target NOC postings only.
            in_memory_matches = compute_matches_in_memory(staged, top=5)
            filtered_matches = filter_matches_to_target_family(
                in_memory_matches, staged.target_noc,
            )
            primary = compute_primary_gap_name(
                staged=staged,
                user_message=user_message,
                truth_enough_to_match=True,
                truth_usable_evidence_present=True,
                engine_completed=True,
                in_memory_matches=filtered_matches,
                skill_adjacent_results=None,
                snapshot_usable=True,
                target_posting_count=len(filtered_matches),
            )
            registry = None
            if TRAINING_REGISTRY_ENABLED:
                try:
                    from skillbridge.training.registry import get_registry
                    registry = get_registry()
                except Exception:  # noqa: BLE001
                    log.warning(
                        "recommender_consume registry_load_failed; "
                        "training will be empty"
                    )
                    registry = None
            rec_evidence = build_recommender_evidence_local_gap_coach(
                match_results=filtered_matches,
                primary_gap_name=primary,
                registry=registry,
                today=date.today(),
            )
        elif mode == "target_noc_standard":
            rec_evidence = build_recommender_evidence_target_noc_standard(
                user_skill_ids=user_skill_ids,
                target_noc=staged.target_noc,
            )
        elif mode == "adjacent_noc_standard":
            # Slice 4 (2026-06-26): cold-start adjacency derivation.
            # If a prior matching turn populated last_adjacent_nocs,
            # use that (fast path). Otherwise invoke the matching
            # engine's adjacency pipeline READ-ONLY here to derive
            # adjacent NOCs ephemerally for this turn's Layer C
            # wrapper. Result is NOT persisted to staged --
            # ordinal followups + matching-turn lifecycle stay clean.
            from skillbridge.chat.recommender_assembly import (
                _compute_adjacent_nocs_for_recommender,
            )
            adjacent_nocs = staged.last_adjacent_nocs
            if not adjacent_nocs:
                adjacent_nocs = _compute_adjacent_nocs_for_recommender(staged)
            rec_evidence = build_recommender_evidence_adjacent_noc_standard(
                user_skill_ids=user_skill_ids,
                last_adjacent_nocs=adjacent_nocs,
            )
    except Exception:  # noqa: BLE001
        log.exception(
            "recommender_consume evidence_build_failed mode=%s", mode,
        )
        # Clear the flag so the chain doesn't loop on a failing mode.
        staged.pending_recommender_offer = None
        return None

    if rec_evidence is None:
        # Defensive: shouldn't happen given the mode guard above, but
        # if some future mode is added without a branch, fall through.
        return None

    # Slice 2 (locked 2026-06-23) follow-up: empty-evidence guards in
    # the consume path. The intent-driven dispatcher already emits
    # honest canned text when Layer A or Layer C evidence is empty,
    # but the consent path was falling through to compose_response_v2
    # with an empty wrapper -- the LLM then hallucinated plausible
    # but ungrounded role names (live verify 2026-06-26 surfaced this:
    # yes-consent to B's C-offer when last_adjacent_nocs was empty
    # produced "bookkeeper roles, financial analyst positions" out of
    # thin air). Mirror the intent-driven guards here so consent +
    # empty-evidence stays honest.
    if mode == "adjacent_noc_standard" and not rec_evidence.evidence:
        reply = _LAYER_C_EMPTY_HONEST
        staged.pending_recommender_offer = None
        staged.last_adjacent_nocs = ()
        staged.touch()
        new_session_id = store.save(staged)
        return {
            "reply": reply,
            "profile_id": None,
            "session_id": new_session_id,
            "intake_state": staged.intake_state,
            "asked_slots": [],
            "next_action": intake_state.ACTION_PRESENT_MATCHES,
            "recommended_jobs": [],
            "next_skill_suggestion": None,
            "resume_info": resume_info,
            "requires_consent": True,
        }
    if mode == "target_noc_standard" and not rec_evidence.evidence:
        # Defensive: A is intent-only in the new chain (slice 2 chain
        # B -> C -> END, A -> END with no chain in), so consent should
        # not normally land here. Stale pending state from legacy
        # cookies could still arrive; emit honest text.
        reply = _format_canned_with_target(
            _LAYER_A_EMPTY_HONEST, staged.target_role_text,
        )
        staged.pending_recommender_offer = None
        staged.last_adjacent_nocs = ()
        staged.touch()
        new_session_id = store.save(staged)
        return {
            "reply": reply,
            "profile_id": None,
            "session_id": new_session_id,
            "intake_state": staged.intake_state,
            "asked_slots": [],
            "next_action": intake_state.ACTION_PRESENT_MATCHES,
            "recommended_jobs": [],
            "next_skill_suggestion": None,
            "resume_info": resume_info,
            "requires_consent": True,
        }

    # Synthesize a passthrough ArbiterDecision. compose_response_v2's
    # recommender early-return ignores `decision` so this value is
    # never consulted downstream -- but the dataclass requires it.
    synth = ArbiterDecision(
        final_move="present_tiered_matches",
        reason_code="recommender_consume:" + mode,
        tone="neutral",
        arbiter_action="recommender",
        ask_slot=None,
    )
    inp = ResponderV2Input(
        user_message=user_message,
        decision=synth,
        results=[],
        training_by_job={},
        next_skill=(None, 0),
        band_signal="none",
        requires_consent=True,
        target_role_text=staged.target_role_text,
        resume_facts=_effective_facts_view(staged),
        recommendation_evidence=rec_evidence,
    )
    reply = compose_response_v2(inp)

    # Advance the chain.
    next_mode = _RECOMMENDER_NEXT_MODE[mode]
    staged.pending_recommender_offer = next_mode
    if next_mode is None:
        # Chain ENDS here. Clear the per-target Layer C cache.
        staged.last_adjacent_nocs = ()
    staged.touch()
    new_session_id = store.save(staged)
    return {
        "reply": reply,
        "profile_id": None,
        "session_id": new_session_id,
        "intake_state": staged.intake_state,
        "asked_slots": [],
        "next_action": intake_state.ACTION_PRESENT_MATCHES,
        "recommended_jobs": [],
        "next_skill_suggestion": None,
        "resume_info": resume_info,
        "requires_consent": True,
    }


# =========================================================================
# Step 3 peer-engine wiring (2026-06-22) -- intent-driven recommender
# dispatch + canned substrate/out-of-scope responses.
# =========================================================================
# Canned response placeholders. Wording deferred per step 3 lock --
# these are plain, terse, and may be re-voiced in a later slice without
# touching routing logic.

_ASK_TARGET_CANNED: str = (
    "Which kind of work are you thinking about? Name a role or field "
    "and I'll look at the gap from there."
)

_ASK_SKILLS_CANNED: str = (
    "Tell me a bit about what you've done -- any work history, "
    "training, or skills -- or upload your resume. I need that to "
    "say anything useful."
)

_ASK_BOTH_CANNED: str = (
    "Tell me what kind of work you're looking at and a bit about your "
    "background -- or upload your resume. I'll go from there."
)

_OUT_OF_SCOPE_CANNED: str = (
    "I help with finding local Sault Ste. Marie jobs and figuring out "
    "what skills to build for them. Resume tailoring, cover letters, "
    "and interview prep are a better fit for the Sault Community "
    "Career Centre. Want me to look at job matches or skill gaps for "
    "a target role?"
)


# Slice 2 (locked 2026-06-23): canned texts for the three Layer B
# branches and the Layer C/A direct-intent empty paths. All four
# substitute {target_role} from staged.target_role_text (fallback
# "that role" when None).
#
# Strict resume gate (Option A locked): when Layer B is empty AND
# no resume is uploaded, we ask SPECIFICALLY for the resume. No
# "or work history" alternative -- a resume is the only profile
# input that unlocks "OK, Layer B is honestly empty, offer Layer C."
# Reason: chat-typed work history can't be parsed into the same
# fact shape as resume_facts; if we accept it as equivalent we'd
# either over-promise (offer C without the same evidence floor) or
# silently keep asking for resume (confusing UX).
_ASK_RESUME_FOR_LAYER_B: str = (
    "I checked local {target_role} postings in Sault Ste. Marie, but "
    "to give you useful gap advice I need a fuller picture of your "
    "background. Could you upload your resume? With it I can give "
    "you targeted advice for local openings."
)

_OFFER_C_AFTER_EMPTY_B: str = (
    "I checked local {target_role} postings in Sault Ste. Marie but "
    "didn't find specific gaps to coach you on against your current "
    "profile. Want me to look at related career paths your skills "
    "line up with?"
)

_LAYER_C_EMPTY_HONEST: str = (
    "Nothing surfaced from related roles in this session. Want to "
    "try a different target, or check what jobs are open in this "
    "field?"
)

_LAYER_A_EMPTY_HONEST: str = (
    "I don't have a Canadian/NOC standard skill profile loaded for "
    "{target_role} yet. Want to look at this from a different angle?"
)


def _format_canned_with_target(template: str, target_role_text: str | None) -> str:
    """Substitute {target_role} into a canned-text template.

    Falls back to "that role" when staged.target_role_text is None
    or empty. Keeps single source of truth for the substitution rule
    across the four slice 2 canned texts.
    """
    role = (target_role_text or "").strip() or "that role"
    return template.format(target_role=role)


def _dispatch_recommender_from_intent(
    *,
    staged: StagedProfile,
    mode: str,
    voice_hint: "CareerIntent | None",
    user_message: str,
    store,
    resume_info: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Force-dispatch one recommender layer from a router verdict.

    Unlike `_dispatch_recommender_consume`, this entry point:
      - Does NOT classify consent (the router already decided this turn
        belongs to the recommender via career intent + substrate gate).
      - Takes the lead mode as input from the router; never re-decides.
      - Uses the SAME _RECOMMENDER_NEXT_MODE chain mapping the existing
        consume helper uses, so the prompt's closing offer and the
        next-turn pending state stay in sync (locked under Option A).

    Returns the response dict, or None if anything in the dispatch
    chain failed (caller falls through to normal flow).
    """
    if mode not in _VALID_RECOMMENDER_MODES:
        log.warning(
            "recommender_intent_dispatch invalid mode=%r; falling through",
            mode,
        )
        return None

    from datetime import date
    from skillbridge.chat.gap_evidence import RecommenderEvidence
    from skillbridge.chat.recommender_assembly import (
        _compute_adjacent_nocs_for_recommender,
        build_recommender_evidence_adjacent_noc_standard,
        build_recommender_evidence_local_gap_coach,
        build_recommender_evidence_target_noc_standard,
        filter_matches_to_target_family,
    )
    from skillbridge.chat.development_plan import compute_primary_gap_name
    from skillbridge.match.engine import (
        build_user_skill_rows,
        compute_matches_in_memory,
        derive_user_skill_sets,
    )

    user_rows = build_user_skill_rows(staged.skills)
    user_skill_ids, _names, _canon = derive_user_skill_sets(user_rows)

    rec_evidence: RecommenderEvidence | None = None
    try:
        if mode == "local_gap_coach":
            # Slice 2 (locked 2026-06-23): Layer B path with target-NOC
            # filter + 3-branch decision tree.
            #
            # Step 1: engine top-5 by skill overlap (unfiltered shape).
            # Step 2: filter to target NOC family (exact preferred,
            #         minor-group fallback). Without this, off-target
            #         postings (Communication Operator when target is
            #         accounting clerk) bleed into CP4's gap ranking.
            # Step 3: CP4 picks primary gap from FILTERED postings only.
            # Step 4: assemble Layer B evidence from FILTERED postings.
            # Step 5: three branches based on (evidence, has_resume):
            #   (a) evidence has content -> render Layer B + offer C
            #   (b) evidence empty + no resume -> ask resume, stop
            #   (c) evidence empty + has resume -> offer C, don't render C
            in_memory_matches = compute_matches_in_memory(staged, top=5)
            filtered_matches = filter_matches_to_target_family(
                in_memory_matches, staged.target_noc,
            )
            primary = compute_primary_gap_name(
                staged=staged,
                user_message=user_message,
                truth_enough_to_match=True,
                truth_usable_evidence_present=True,
                engine_completed=True,
                in_memory_matches=filtered_matches,
                skill_adjacent_results=None,
                snapshot_usable=True,
                target_posting_count=len(filtered_matches),
            )
            registry = None
            if TRAINING_REGISTRY_ENABLED:
                try:
                    from skillbridge.training.registry import get_registry
                    registry = get_registry()
                except Exception:  # noqa: BLE001
                    log.warning(
                        "recommender_intent_dispatch registry_load_failed; "
                        "training will be empty"
                    )
                    registry = None
            rec_evidence = build_recommender_evidence_local_gap_coach(
                match_results=filtered_matches,
                primary_gap_name=primary,
                registry=registry,
                today=date.today(),
            )

            # Slice 2 branches (b) and (c): when Layer B evidence is
            # empty, branch on resume presence. The evidence-build
            # returned a wrapper with mode set + evidence tuple empty.
            if not rec_evidence.evidence:
                has_resume = _resume_facts_have_content(
                    staged.resume_facts_json
                )
                target_phrase = staged.target_role_text
                if not has_resume:
                    # Branch (b): ask for resume. Persist deferred
                    # intent so slice 1 machinery re-routes after
                    # resume upload.
                    reply = _format_canned_with_target(
                        _ASK_RESUME_FOR_LAYER_B, target_phrase,
                    )
                    staged.deferred_career_intent = "local_skill_gap"
                    staged.last_asked_slots = []
                    staged.pending_recommender_offer = None
                    staged.touch()
                    new_session_id = store.save(staged)
                    return {
                        "reply": reply,
                        "profile_id": None,
                        "session_id": new_session_id,
                        "intake_state": staged.intake_state,
                        "asked_slots": [],
                        "next_action": intake_state.ACTION_ASK_QUESTIONS,
                        "recommended_jobs": [],
                        "next_skill_suggestion": None,
                        "resume_info": resume_info,
                        "requires_consent": True,
                    }
                # Branch (c): emit C offer (do NOT render C). Set
                # pending so next-turn "yes" routes via consume hook.
                reply = _format_canned_with_target(
                    _OFFER_C_AFTER_EMPTY_B, target_phrase,
                )
                staged.deferred_career_intent = None
                staged.pending_recommender_offer = "adjacent_noc_standard"
                staged.touch()
                new_session_id = store.save(staged)
                return {
                    "reply": reply,
                    "profile_id": None,
                    "session_id": new_session_id,
                    "intake_state": staged.intake_state,
                    "asked_slots": [],
                    "next_action": intake_state.ACTION_PRESENT_MATCHES,
                    "recommended_jobs": [],
                    "next_skill_suggestion": None,
                    "resume_info": resume_info,
                    "requires_consent": True,
                }
            # Branch (a): evidence has content. Fall through to the
            # normal render + chain-advance path below.
        elif mode == "target_noc_standard":
            rec_evidence = build_recommender_evidence_target_noc_standard(
                user_skill_ids=user_skill_ids,
                target_noc=staged.target_noc,
            )
            # Slice 2: Layer A direct-intent empty -> honest text.
            # NEVER cascade to another layer.
            if not rec_evidence.evidence:
                reply = _format_canned_with_target(
                    _LAYER_A_EMPTY_HONEST, staged.target_role_text,
                )
                staged.pending_recommender_offer = None
                staged.touch()
                new_session_id = store.save(staged)
                return {
                    "reply": reply,
                    "profile_id": None,
                    "session_id": new_session_id,
                    "intake_state": staged.intake_state,
                    "asked_slots": [],
                    "next_action": intake_state.ACTION_PRESENT_MATCHES,
                    "recommended_jobs": [],
                    "next_skill_suggestion": None,
                    "resume_info": resume_info,
                    "requires_consent": True,
                }
        elif mode == "adjacent_noc_standard":
            # Slice 4 (2026-06-26): cold-start adjacency derivation.
            # If a prior matching turn populated last_adjacent_nocs,
            # use that (fast path). Otherwise invoke the matching
            # engine's adjacency pipeline READ-ONLY via the recommender
            # helper to derive adjacent NOCs ephemerally for this
            # turn's Layer C wrapper. Result is NOT persisted to
            # staged -- ordinal-followup state and matching-turn
            # lifecycle remain owned by the matching engine.
            adjacent_nocs = staged.last_adjacent_nocs
            if not adjacent_nocs:
                adjacent_nocs = _compute_adjacent_nocs_for_recommender(staged)
            rec_evidence = build_recommender_evidence_adjacent_noc_standard(
                user_skill_ids=user_skill_ids,
                last_adjacent_nocs=adjacent_nocs,
            )
            # Slice 2: Layer C direct-intent empty -> honest text.
            # NEVER cascade to A.
            if not rec_evidence.evidence:
                reply = _LAYER_C_EMPTY_HONEST
                staged.pending_recommender_offer = None
                staged.touch()
                new_session_id = store.save(staged)
                return {
                    "reply": reply,
                    "profile_id": None,
                    "session_id": new_session_id,
                    "intake_state": staged.intake_state,
                    "asked_slots": [],
                    "next_action": intake_state.ACTION_PRESENT_MATCHES,
                    "recommended_jobs": [],
                    "next_skill_suggestion": None,
                    "resume_info": resume_info,
                    "requires_consent": True,
                }
    except Exception:  # noqa: BLE001
        log.exception(
            "recommender_intent_dispatch evidence_build_failed mode=%s",
            mode,
        )
        return None

    if rec_evidence is None:
        return None

    # Synthesize a passthrough ArbiterDecision; compose_response_v2's
    # recommender early-return ignores `decision` so this value is
    # never consulted downstream. Required by the dataclass shape.
    synth = ArbiterDecision(
        final_move="present_tiered_matches",
        reason_code="recommender_intent_dispatch:" + mode,
        tone="neutral",
        arbiter_action="recommender",
        ask_slot=None,
    )
    inp = ResponderV2Input(
        user_message=user_message,
        decision=synth,
        results=[],
        training_by_job={},
        next_skill=(None, 0),
        band_signal="none",
        requires_consent=True,
        target_role_text=staged.target_role_text,
        resume_facts=_effective_facts_view(staged),
        recommendation_evidence=rec_evidence,
        recommender_voice_hint=voice_hint,
    )
    reply = compose_response_v2(inp)

    # Slice 2 chain (locked): B -> C (offer); C -> END; A -> END.
    # The closing in the prompt/fallback matches whichever next_mode
    # is set here -- so the user's "yes" on the next turn routes
    # correctly via the consume helper.
    next_mode = _RECOMMENDER_NEXT_MODE.get(mode)
    staged.pending_recommender_offer = next_mode
    if next_mode is None:
        staged.last_adjacent_nocs = ()
    staged.touch()
    new_session_id = store.save(staged)
    return {
        "reply": reply,
        "profile_id": None,
        "session_id": new_session_id,
        "intake_state": staged.intake_state,
        "asked_slots": [],
        "next_action": intake_state.ACTION_PRESENT_MATCHES,
        "recommended_jobs": [],
        "next_skill_suggestion": None,
        "resume_info": resume_info,
        "requires_consent": True,
    }


def _maybe_route_recommender_from_intent(
    *,
    staged: StagedProfile,
    message: str,
    store,
) -> dict[str, Any] | None:
    """Apply step 3 peer-engine routing.

    Calls the LLM career-intent classifier and the deterministic
    router, then acts on the verdict:
      - matching_engine / default -> returns None (caller falls
        through to existing matching flow)
      - out_of_scope_canned       -> canned redirect
      - ask_substrate             -> ask for missing target/skills
      - recommender_layer         -> dispatch the forced mode via
                                     _dispatch_recommender_from_intent

    Defensive: any exception falls through to None so the existing
    matching flow handles the turn. Routing must never break the
    chat surface.
    """
    try:
        from skillbridge.chat.recommender_intent import classify_career_intent
        from skillbridge.chat.recommender_route import route_recommender
        from skillbridge.chat.truth_summary import _classify_intent
    except Exception:  # noqa: BLE001
        log.exception("recommender_routing import_failed; falling through")
        return None

    try:
        pattern_intent = _classify_intent(message)
        # Slice 1 (2026-06-23): thread last_asked_slot[0] as the
        # last_assistant_move context. Slot answers ("I've done X, Y, Z"
        # after the system asked for skills_text) used to misclassify as
        # recommender intent because the classifier had null context.
        last_asked_slot = (
            staged.last_asked_slots[0]
            if staged.last_asked_slots else None
        )
        career_intent = classify_career_intent(
            message=message,
            pending_recommender_offer=staged.pending_recommender_offer,
            target_role_text=staged.target_role_text,
            last_assistant_move=last_asked_slot,
        )
        chat_skill_count = len(staged.skills or [])
        has_resume = _resume_facts_have_content(staged.resume_facts_json)

        # Slice 1 (2026-06-23): consume deferred_career_intent.
        # When the current message classifies as `unclear` (e.g. a
        # bare slot fill) AND a prior turn deferred a recommender
        # intent because substrate was missing, route to the
        # deferred intent instead of letting it drop silently.
        # Explicit current intent (non-unclear) ALWAYS wins over
        # the deferred one -- the user has moved on.
        deferred = staged.deferred_career_intent
        if career_intent != "unclear":
            # Current intent wins; clear any deferred holdover.
            if deferred is not None:
                staged.deferred_career_intent = None
        elif deferred is not None and deferred in _DEFERRABLE_CAREER_INTENTS:
            # Current message has no clear intent; revive the
            # deferred one. The substrate gate below decides whether
            # it can fire NOW (target+skills present) or still has
            # to wait. We clear the deferred flag in either case --
            # if substrate is still missing, the router emits
            # ask_substrate again with deferred_intent re-set from
            # the verdict; if substrate is sufficient, the layer
            # dispatches and the intent has been honoured.
            career_intent = deferred  # type: ignore[assignment]
            staged.deferred_career_intent = None
            log.info(
                "recommender_routing session=%s deferred_intent_consumed=%s",
                staged.session_id[:8], deferred,
            )

        # Resolve target_role_text -> target_noc before substrate check.
        # target_noc is normally populated by the matching engine via
        # `resolve_title_to_noc` -- but the matching engine hasn't run
        # yet on this turn (the router decides whether it will). If the
        # extractor filled `target_role_text` from the current message
        # but `target_noc` is still None, resolve it here so the
        # substrate gate doesn't ask the user for a target they just
        # named.
        target_noc = staged.target_noc
        if not target_noc and staged.target_role_text:
            try:
                from skillbridge.match.occupation import resolve_title_to_noc
                resolved = resolve_title_to_noc(staged.target_role_text)
            except Exception:  # noqa: BLE001
                log.warning(
                    "recommender_routing resolve_title_to_noc_failed; "
                    "target stays None for this turn",
                )
                resolved = None
            if (
                isinstance(resolved, str)
                and len(resolved) == 5
                and resolved.isdigit()
            ):
                # Persist for downstream so the matching engine doesn't
                # repeat the lookup -- mirrors the matching engine's
                # own caching of resolution.
                staged.target_noc = resolved
                target_noc = resolved

        verdict = route_recommender(
            pattern_intent=pattern_intent,
            career_intent=career_intent,
            target_noc=target_noc,
            chat_skill_count=chat_skill_count,
            has_resume=has_resume,
        )
    except Exception:  # noqa: BLE001
        log.exception(
            "recommender_routing classify_or_route_failed; falling through",
        )
        return None

    log.info(
        "recommender_routing session=%s pattern=%s career=%s action=%s "
        "reason=%s",
        staged.session_id[:8], pattern_intent, career_intent,
        verdict.action, verdict.reason,
    )

    if verdict.action in ("matching_engine", "default"):
        return None  # let existing flow handle

    if verdict.action == "out_of_scope_canned":
        # No slot is being asked -- system is redirecting. Don't
        # pollute last_asked_slots with a target/skills hint.
        return _emit_canned_response(
            staged=staged, store=store,
            reply=_OUT_OF_SCOPE_CANNED, resume_info=None,
            asked_slot=None,
        )

    if verdict.action == "ask_substrate":
        # Slice 1 follow-up (2026-06-23): the ask carries an
        # explicit slot the next turn's reply is answering. Set
        # last_asked_slots so the extractor + classifier + fallback
        # fill all share the same understanding of what slot is
        # being filled. For "both missing", target is the first
        # gate (we ask for it first; substrate gate re-emits ask
        # for skills on the following turn after target fills).
        missing = set(verdict.missing)
        if missing == {"target"}:
            reply = _ASK_TARGET_CANNED
            asked_slot = "target_role_text"
        elif missing == {"skills"}:
            reply = _ASK_SKILLS_CANNED
            asked_slot = "skills_text"
        else:
            reply = _ASK_BOTH_CANNED
            asked_slot = "target_role_text"
        # Persist the deferred intent so the next turn (after the
        # user provides substrate) can route to it instead of
        # dropping silently. Only persist deferrable intents per
        # the closed set above.
        if (
            verdict.deferred_intent is not None
            and verdict.deferred_intent in _DEFERRABLE_CAREER_INTENTS
        ):
            staged.deferred_career_intent = verdict.deferred_intent
        return _emit_canned_response(
            staged=staged, store=store,
            reply=reply, resume_info=None,
            asked_slot=asked_slot,
        )

    if verdict.action == "recommender_layer":
        if verdict.recommender_mode is None:
            log.warning(
                "recommender_routing recommender_layer verdict missing "
                "recommender_mode; falling through",
            )
            return None
        return _dispatch_recommender_from_intent(
            staged=staged,
            mode=verdict.recommender_mode,
            voice_hint=verdict.voice_hint,
            user_message=message,
            store=store,
            resume_info=None,
        )

    # Defensive: unknown action. Fall through.
    log.warning(
        "recommender_routing unknown verdict action=%r; falling through",
        verdict.action,
    )
    return None


def _emit_canned_response(
    *,
    staged: StagedProfile,
    store,
    reply: str,
    resume_info: dict[str, Any] | None,
    asked_slot: str | None = None,
) -> dict[str, Any]:
    """Shared shape for canned-text responses (ask_substrate +
    out_of_scope_canned). Does NOT mutate intake_state or pending
    flags -- routing chose to bypass the engines this turn, but the
    user's session continues from its current state on the next turn.

    asked_slot (Slice 1 follow-up 2026-06-23): when set, updates
    `staged.last_asked_slots` so the next turn's extractor + bridge
    + fallback_fill all know which slot the system just asked for.
    Without this, an ask_substrate canned response would leave
    `last_asked_slots` stale (or empty) and the next turn's reply
    couldn't be recognised as a slot answer.
    Pass None for out-of-scope responses (no slot asked).
    """
    if asked_slot is not None:
        staged.last_asked_slots = [asked_slot]
    staged.touch()
    new_session_id = store.save(staged)
    return {
        "reply": reply,
        "profile_id": None,
        "session_id": new_session_id,
        "intake_state": staged.intake_state,
        "asked_slots": [asked_slot] if asked_slot else [],
        "next_action": intake_state.ACTION_ASK_QUESTIONS,
        "recommended_jobs": [],
        "next_skill_suggestion": None,
        "resume_info": resume_info,
        "requires_consent": True,
    }


def _should_offer_resume_upload(
    *, staged, final_move: str, band_signal: str,
) -> bool:
    """Resume-upload offer gate (Pattern 1 — closing-matrix v2,
    2026-06-17 LOCKED).

    Foundational principle (locked):
        [[project-user-always-gets-something]] — the system never ends
        a turn empty-handed when it could give a meaningful next step.
        Pattern 1 is the no-resume implementation of that principle:
        if the user hasn't uploaded a resume, the closing of any tier
        or no-match turn is a universal invitation to upload — framed
        as "so I can look at related roles your skills also fit"
        (broadening), NOT "to find a stronger match" (terminating).

    Pattern 1 fires when:
      - No resume has been uploaded this session (resume_facts_json
        is None or empty).

    That's it. The earlier v1 gate also required `final_move ==
    present_no_match` OR `band_signal in {low_only, stretch_only}`,
    so a Strong/Good match with no resume would fall through to
    either the (now-deleted) action closing or the generic fallback —
    neither of which offered the user broadening. Under the locked
    user-always-gets-something principle, those paths were wrong:
    the system never pushes a no-resume user toward "go apply"; it
    keeps offering more service via the upload entitlement.

    band_signal and final_move arguments are RETAINED in the
    signature (no caller changes) but no longer read by the gate.
    They're kept for telemetry / debug consistency and because
    Pattern 2 / Pattern 3 logic (resume + match → CP5 offer; resume
    + 0 match → auto-fire CP5) will be wired in separate handlers
    that read these signals — they should stay on the call shape.

    Note: the prior "once per target" gate
    (`staged.resume_upload_offered`) was DROPPED in 2026-06-16. The
    flag is left on StagedProfile for audit / telemetry but is not
    consulted here. The LLM happy-path varies phrasing turn-by-turn
    so the user doesn't feel pestered.
    """
    if staged.resume_facts_json:
        return False
    return True


def _filter_registry_gaps_by_have_skills(
    found_canonicals: list[str],
    staged_skills: list,
) -> tuple[list[str], list[str]]:
    """Drop registry-discovered gap canonicals the user already states
    they HOLD (Bug A part 2 fix, 2026-06-15).

    `registry.find_gaps_in_message` matches canonical/alias text at word
    boundaries — it is HAS/NEED blind. Without this filter, "I have my
    Class G license" surfaces Class G as a registry gap (because the
    phrase appears in the message), the router fires
    `rule_2_training_with_entity`, and the responder narrates a
    credential gap the user explicitly denied.

    Pure function. Returns (kept, suppressed) — both as canonical-name
    lists. Comparison is by `canonicalize_skill` output so JD-side
    aliases ("Class G driver's license") and user-side phrasings
    ("Class G license") that converge to the same canonical form
    suppress each other consistently.

    Negation is handled upstream: the credential patch in
    `chat_extractor` does NOT add a skill from "I don't have Class G",
    so `staged_skills` won't contain it, and the registry gap correctly
    remains — producing a true training-request intent on that turn.
    """
    from skillbridge.match.alignment import canonicalize_skill

    user_canon: set[str] = set()
    for s in staged_skills:
        name = getattr(s, "skill_name", None)
        if isinstance(name, str) and name.strip():
            user_canon.add(canonicalize_skill(name) or name.strip().lower())

    kept: list[str] = []
    suppressed: list[str] = []
    for gap_name in found_canonicals:
        if not isinstance(gap_name, str) or not gap_name.strip():
            continue
        gap_canon = canonicalize_skill(gap_name) or gap_name.strip().lower()
        if gap_canon in user_canon:
            suppressed.append(gap_name)
        else:
            kept.append(gap_name)
    return kept, suppressed


_SLOT_ANSWER_MAX_TOKENS = 8


def _is_likely_slot_answer(message: str, asked_slots: list[str]) -> bool:
    """Heuristic: was this message likely a short reply to a single
    slot question? Inputs that match should bypass the rule-based
    extractor fallback (the LLM extractor's empty result is taken at
    face value).

    Three conditions, all required:
      - The previous turn asked for at least one slot
      - The message is short (<= _SLOT_ANSWER_MAX_TOKENS tokens)
      - The message doesn't look like a comma-separated skill list

    Tuned for the live failure case "Truck and coach technician role"
    while preserving the rule-based fallback for genuine skill claims
    ("welding, forklift, shipping") and cold sessions (no asked slot).
    """
    if not asked_slots:
        return False
    if "," in message:
        return False
    token_count = len(message.strip().split())
    return token_count <= _SLOT_ANSWER_MAX_TOKENS


def _maybe_recover_skills_text_slot(
    *,
    staged: "StagedProfile",
    extraction: "chat_extractor.ExtractionResult",
    message: str,
) -> bool:
    """Back-fill `staged.skills_text` when the extractor dropped the
    slot-level value as ungrounded BUT individual skills from the same
    message grounded successfully.

    Why this exists (2026-06-17 live repro): the LLM extractor returns
    BOTH a `skills_text` slot field (with its own verbatim-evidence
    requirement) AND a per-skill list (with per-item evidence). The
    slot-level evidence is more brittle — Haiku tends to paraphrase
    or consolidate when summarizing what the user said, so the
    substring grounding check rejects the slot even when individual
    skills ground cleanly.

    On the live accounting-clerk turn, the user explicitly listed 12
    real skills. The per-skill grounding passed 12/12. But the slot-
    level skills_text was dropped (`raw_keys_dropped=['ungrounded:
    skills_text']`), leaving `staged.skills_text` empty. Change C's
    `skills_text_present` guard then read False and `enough_to_match`
    stayed False — so the engine refused to run and the user was
    re-asked despite having just listed real skills.

    The recovery only fires when ALL of these hold:
      1. The extractor signalled "user was listing skills" by
         attempting to fill skills_text (presence of
         'ungrounded:skills_text' in raw_keys_dropped). Phantom-skill
         messages ("Completed Truck and Coach apprenticeship at Sault
         College") do NOT trigger this because the extractor doesn't
         claim the user was listing skills in pure experience prose.
      2. >=3 per-skill items from THIS turn's extraction grounded
         successfully. This is stronger evidence than the cumulative
         chat_skill_count gate — we're asserting the CURRENT MESSAGE
         contained real skill claims, not pulling forward old state.
      3. `staged.skills_text` is currently empty. Recovery never
         overwrites a prior turn's slot value.

    Returns True iff staged.skills_text was filled by this call.
    """
    if "ungrounded:skills_text" not in extraction.raw_keys_dropped:
        return False
    if len(extraction.skills) < 3:
        return False
    existing = getattr(staged, "skills_text", None)
    if isinstance(existing, str) and existing.strip():
        return False
    msg_stripped = message.strip()
    if len(msg_stripped) < 3:
        return False
    staged.skills_text = msg_stripped[:500]
    return True


def _closed_vocab_reply(slot: str, message: str) -> str | None:
    """Interpret short replies to closed-vocabulary questions.

    Evidence-bound extraction intentionally rejects tiny evidence by default
    to avoid hallucinations, but real users answer shift/work-type questions
    with words like "day" or "full time". This helper is only used when the
    backend knows exactly which slot it asked on the previous turn.
    """
    m = message.strip().lower()
    if slot == "shift_preference":
        if re.fullmatch(r"(day|days|day shift|morning|mornings)", m):
            return "days"
        if re.fullmatch(r"(evening|evenings|evening shift)", m):
            return "evenings"
        if re.fullmatch(r"(night|nights|night shift|overnight)", m):
            return "nights"
        if re.fullmatch(r"(weekend|weekends|saturday|sunday)", m):
            return "weekends"
        if re.fullmatch(r"(any|flexible|whatever|open)", m):
            return "flexible"
    if slot == "work_type_preference":
        if re.fullmatch(r"(full time|full-time|fulltime|ft)", m):
            return "full-time"
        if re.fullmatch(r"(part time|part-time|parttime|pt)", m):
            return "part-time"
        if re.fullmatch(r"(contract|casual|seasonal|temporary|temp)", m):
            return "temporary" if m == "temp" else m
        if re.fullmatch(r"(any|flexible|open)", m):
            return "flexible"
    return None


# =========================================================================
# Reply composition — shared
# =========================================================================
def _build_results_block(match_results: list[Any]) -> tuple[list[dict], str]:
    """Pick top recommendations + signal whether we have real matches.

    Returns (jobs, band) where band is one of:
      'strong_or_good' — at least one strong/good match in the list
      'stretch_only'   — at least one stretch match shown (no strong/good)
      'low_only'       — eligible matches exist but all are low-band; the
                         presentation layer hides them, so `jobs` is empty.
                         Distinct from 'none' so logs reveal "engine found
                         candidates but they were filtered" vs "engine
                         found nothing at all."
      'none'           — no eligible matches at all (engine produced
                         nothing or all ineligible).

    Live-test feedback (2026-06-05): before this fix the function returned
    ('stretch_only', []) when only low-band matches existed, which was a
    misleading lie in the operational logs ("we had stretches" when we
    didn't). 'low_only' is the honest signal.
    """
    eligible = [m for m in match_results if m.match_eligible]
    eligible.sort(key=lambda r: r.match_score, reverse=True)
    if not eligible:
        return [], "none"
    strong_or_good = [m for m in eligible if m.match_band in {"strong", "good"}]
    stretch_matches = [m for m in eligible if m.match_band == "stretch"]
    if strong_or_good:
        chosen = strong_or_good[:MAX_PREVIEW_JOBS]
        band = "strong_or_good"
    elif stretch_matches:
        chosen = stretch_matches[:MAX_PREVIEW_JOBS]
        band = "stretch_only"
    else:
        # Eligible-only-low: engine ran, found candidates, but every one
        # is below stretch. Presentation hides them (today's product
        # call). Telemetry MUST distinguish this from "engine found
        # nothing" so future debugging knows where to look.
        chosen = []
        band = "low_only"
    return [
        {
            "job_id": m.job_id,
            "title": m.title,
            "employer": m.employer,
            "url": m.url,
            "location": m.location,
            "match_score": m.match_score,
            "match_band": m.match_band,
            "matched_skills": m.matched_skills,
            "missing_skills": m.missing_skills,
            "credential_warning": m.credential_warning,
            # Sprint 1: structured "why" payload — the responder may quote
            # any of these signals to ground a "because" clause, but
            # cannot invent causality outside of them.
            "score_explanation": m.score_explanation,
        }
        for m in chosen
    ], band


def _attach_training(results: list[dict]) -> dict[str, list[dict]]:
    seen: set[str] = set()
    by_job: dict[str, list[dict]] = {}
    for r in results:
        suggestions: list[dict] = []
        for skill in r.get("missing_skills", [])[:3]:
            for s in suggest_for_skill(skill, limit=2):
                # Drop the "no local match" fallback sentinel from the
                # payload. recommend.suggest_for_skill returns it when
                # core.training_resource has no rows matching the skill;
                # rendering it as a training card ("Speak with a career
                # counsellor — SCCC · Short program") misleads the user
                # into thinking we found them a course. The responder
                # narrates the no-training case in prose instead.
                if s.resource_id is None:
                    continue
                key = s.resource_id + s.title
                if key in seen:
                    continue
                seen.add(key)
                suggestions.append({
                    "provider": s.provider,
                    "title": s.title,
                    "url": s.url,
                    "duration_band": s.duration_band,
                    "resource_type": s.resource_type,
                    "for_skill": skill,
                    "reason": s.reason,
                })
                if len(suggestions) >= 4:
                    break
            if len(suggestions) >= 4:
                break
        by_job[r["job_id"]] = suggestions
    return by_job


# =========================================================================
# Chat orchestration v2 -- handler dispatch (Slice 6)
# =========================================================================
# Single function, deliberately boring. The order MUST be visible in
# the call sequence:
#
#   gates  ->  planner  ->  arbiter pass 1  ->  [engine?]  ->  arbiter pass 2  ->  responder v2
#
# Engine invocation lives in exactly one place (the `if isinstance(pass1, RunEngine)`
# branch). No other path in this module calls the match engine when
# CHAT_ORCHESTRATOR == "v2". Tests assert this exhaustively.
# =========================================================================
# =========================================================================
# AR-6b -- handler-side soft-offer append + flag setter
# =========================================================================
def _last_adjacency_was_empty(staged: StagedProfile) -> bool:
    """AR-8c: True iff the IMMEDIATELY PRIOR turn ran the adjacency
    engine and returned zero recommendations.

    `_run_adjacency_engine_and_persist` always writes
    `last_adjacent_snapshot = {"created_message_count": N,
    "items": [...]}` -- when `items` is the empty list, the engine
    ran but accepted no candidates (typical when the credential gate
    or coverage floor drops everything). The user has already seen
    the deterministic "I'm not seeing other roles" narration; re-
    offering the soft-offer line on the immediately following turn
    reads as nagging.

    Window: K=1 (one turn back). After K turns of unrelated
    activity, the offer becomes appropriate again -- the empty
    adjacency is no longer fresh.

    Note on scope digressions: `shift_adjacent_snapshot_ttl` advances
    `created_message_count` by 1 on scope-violation turns so the
    snapshot stays "fresh" for ordinal resolution. That keeps the
    K=1 distance check consistent across digressions too: the user's
    last engagement with adjacency was still empty, the suppression
    is still appropriate.
    """
    snap = staged.last_adjacent_snapshot
    if not isinstance(snap, dict):
        return False
    items = snap.get("items")
    if not isinstance(items, list) or items:
        return False
    created = snap.get("created_message_count")
    # bool-as-int guard: True/False would compare as 1/0.
    if not isinstance(created, int) or isinstance(created, bool):
        return False
    return staged.message_count - created == 1


def _maybe_append_soft_offer(
    *,
    reply: str,
    staged: StagedProfile,
    final: "ArbiterDecision",
    results: list[dict],
    pending_offer: bool,
    prior_empty_adjacency: bool = False,
) -> str:
    """Append the soft-offer line to `reply` when the standard match
    path emitted a credential-only-capped `present_matches` or a
    genuine `present_no_match` AND the user has usable skill
    evidence. Mutates `staged.pending_adjacent_offer = True` so the
    next turn's `detect_adjacent_intent` consumes the affirmative.

    Reoffer suppression (v11 lock): when `pending_offer` was True at
    handler entry, this consume-turn must NOT re-attach the offer
    even if the same outcome fires again. Otherwise a user who said
    "no thanks" would get the same offer back. The flag ends the
    turn False; later turns can re-offer because pending_offer will
    be False entering them.

    AR-8c empty-snapshot suppression: `prior_empty_adjacency` is the
    flag the caller captures BEFORE the AR-6c lifecycle clears
    `last_adjacent_snapshot`. When True, the soft offer is
    suppressed -- the user just walked the adjacency path and saw
    the empty-result narration; offering it again on the
    immediately following turn reads as nagging. The flag is the
    SOURCE OF TRUTH here; the helper does NOT re-read
    `staged.last_adjacent_snapshot` because the `present_matches`
    branch clears it before this function runs.

    Gated to Redis-mode + ADJACENCY_ACTIVATION_ENABLED via
    `_adjacency_enabled()`. Cookie-mode and flag-OFF users see the
    pre-AR-6 reply unchanged.

    Returns the (possibly extended) reply text. Side effect: sets
    `staged.pending_adjacent_offer = True` when applicable.
    """
    if pending_offer:
        return reply

    # AR-8c: empty-snapshot reoffer suppression. The caller in
    # `_try_v2_path` captures this state BEFORE the AR-6c lifecycle
    # clear so the suppression is reachable on `present_matches`
    # turns (the lifecycle clear there would otherwise hide the
    # empty snapshot from any direct snapshot peek done here).
    if prior_empty_adjacency:
        return reply

    # Step 11f + 11l (closing-matrix v2): suppress the AR-6b soft
    # offer ("If you'd like, I can also look for related roles...
    # just say what other roles?") on EVERY present_no_match turn,
    # regardless of resume state.
    #
    # Step 11f rationale (resume + present_no_match): the LLM's
    # response already acknowledges that the related-role search ran
    # via the RELATED_ROLES_EXHAUSTED prompt rule. Appending the
    # soft-offer creates an infinite-offer loop.
    #
    # Step 11l rationale (no-resume + present_no_match, 2026-06-18):
    # the OUTCOME_RESPONDER_PROMPT SHAPE 1 closing IS the Pattern 1
    # upload ask, framed around finding related roles. The AR-6b
    # soft-offer line then says "or say what other roles?" — two
    # offers competing for the user's attention, exactly the
    # "splits attention" anti-pattern Pattern 1's structural rules
    # forbid. Suppressing AR-6b here lets Pattern 1's upload ask
    # remain the single closing pivot.
    if getattr(final, "final_move", None) == "present_no_match":
        return reply

    from skillbridge.match.adjacent import (
        _SOFT_OFFER_LINE,
        _adjacency_enabled,
        should_emit_soft_offer_on_matches,
        should_emit_soft_offer_on_no_match,
    )

    if not _adjacency_enabled():
        return reply

    move = getattr(final, "final_move", None)
    emit = False
    if move == "present_matches":
        lead_result = results[0] if (results and isinstance(results[0], dict)) else None
        if lead_result is not None and should_emit_soft_offer_on_matches(
            lead_result, staged,
        ):
            emit = True
    elif move == "present_no_match":
        if should_emit_soft_offer_on_no_match(staged):
            emit = True

    if not emit:
        return reply

    staged.pending_adjacent_offer = True
    return reply.rstrip() + "\n\n" + _SOFT_OFFER_LINE


# =========================================================================
# AR-9.feat.coach-tiers CP2 step 4 — tier-evidence build helper.
#
# Wraps the four-step adjacency pipeline plus the tier builder behind a
# single call so `_try_v2_path` keeps a thin dispatch contract. The
# helper is the SOLE place the handler builds TieredEvidence; it must
# always run BEFORE the handler sets `tiered_evidence_available=True`
# on the arbiter re-dispatch (signed-off pin from step-2 review).
#
# Product framing for proactive adjacency (signed-off pin from step-3
# review): the Sideways tier exists to widen the user's view of roles
# where their skills transfer. The trigger is NOT `len(strong) < 3` —
# that was a numeric heuristic. The product need is: surface adjacency
# whenever the existing adjacency gates pass
# (ADJACENCY_ACTIVATION_ENABLED + Redis-mode + has_usable_skill_evidence
# floor + accept_candidates strict 5-gate AND). build_tiered_evidence's
# tier-exclusivity-by-job_id + 3-record cap then decides how many
# sideways records actually surface. Cookie-mode users degrade
# gracefully to an empty Sideways tier.
# =========================================================================
def _build_tier_evidence_for_handler(
    results: list[MatchResult],
    training_by_job: dict[str, list[dict]],
    staged: StagedProfile,
) -> "TieredEvidence":
    """Build the per-turn TieredEvidence package for the
    present_tiered_matches surface.

    Runs adjacency through the existing strict-AND gate when
    `_adjacency_enabled()` returns True. Catches adjacency-DB errors
    and degrades to an empty Sideways tier rather than failing the
    whole turn — direct-match coaching still happens via Strong /
    Stretch even if adjacency is offline.

    Returns a TieredEvidence object. Caller checks
    `tier_evidence.apply_today/worth_a_try/sideways_move` for tier
    presence before flipping `tiered_evidence_available`.
    """
    from skillbridge.chat.tiered_evidence import (
        TieredEvidence,
        build_tiered_evidence,
    )
    from skillbridge.match.adjacent import _adjacency_enabled
    from skillbridge.match.alignment import (
        build_user_skill_rows,
        derive_user_skill_sets,
    )
    from skillbridge.match.engine import MatchResult as _MatchResult

    # Defensive: in production `compute_matches_in_memory` returns
    # `list[MatchResult]`. Some legacy tests (notably
    # `test_chat_transcripts`) mock the engine to return raw dicts; the
    # tier builder accesses MatchResult dataclass attributes and would
    # crash on those. When the input doesn't carry MatchResult
    # objects, return an empty TieredEvidence so
    # `_tier_evidence_has_any_records` is False and the handler stays
    # on the legacy `present_matches` dispatch. Production paths are
    # unaffected.
    if results and not isinstance(results[0], _MatchResult):
        return TieredEvidence(
            apply_today=(), worth_a_try=(), sideways_move=(),
        )

    user_rows = build_user_skill_rows(staged.skills)
    user_ids, user_names, user_canon = derive_user_skill_sets(user_rows)

    # Step-4 review (Medium): gate the board load on BOTH
    # `_adjacency_enabled()` AND `has_usable_skill_evidence(staged)`.
    # The strict accept-candidates gate would drop every candidate
    # when the evidence floor fails, so loading the active-job board
    # and running retrieve_candidates first is wasted work
    # (DB query + per-job iteration) for a guaranteed-empty result.
    accepted_adjacent: list[dict] = []
    if _adjacency_enabled():
        try:
            from skillbridge.match.adjacent import (
                _load_active_jobs_with_skills,
                accept_candidates,
                has_usable_skill_evidence,
                retrieve_candidates,
            )
            if has_usable_skill_evidence(staged):
                all_jobs = _load_active_jobs_with_skills()
                retrieved = retrieve_candidates(
                    staged, snapshot=None, all_jobs=all_jobs,
                    user_ids=user_ids, user_names=user_names,
                    user_canon=user_canon,
                )
                accepted_adjacent, _drops = accept_candidates(
                    retrieved, staged, user_ids, user_names, user_canon,
                )
        except Exception:
            log.warning(
                "tier-evidence: adjacency pipeline raised; Sideways "
                "tier will be empty for this turn.",
                exc_info=True,
            )
            accepted_adjacent = []

    target_noc = (
        staged.target_noc
        if isinstance(staged.target_noc, str) and staged.target_noc
        else None
    )

    return build_tiered_evidence(
        results=results,
        accepted_adjacent=accepted_adjacent,
        user_rows=user_rows,
        user_skill_ids=user_ids,
        user_skill_names=user_names,
        user_skill_names_canon=user_canon,
        training_by_job=training_by_job,
        target_noc=target_noc,
    )


def _cp4_shadow_invocation(
    *,
    staged: StagedProfile,
    user_message: str | None,
    truth,
    in_memory_matches: list,
    tier_evidence,
) -> None:
    """CP4 shadow invocation (2026-06-15). Builds the inputs the
    diagnosis + DevelopmentPlan need, calls the shadow-trace emitter,
    returns nothing. NEVER modifies any user-facing state.

    Inputs derived from existing handler state — no new pipeline
    stages or schema migrations are required for this first
    increment. `target_posting_count` is derived only when the user's
    target NOC resolves to an exact 5-digit value; for vague or
    unresolved targets it is None.

    `snapshot_usable` is a thin read-only derivation: true iff the
    pipeline snapshot has non-zero `total_active_jobs` AND a non-null
    `last_publish_at_text`. Future work upgrades this to the full
    `MarketSnapshotContext` once `pipeline_snapshot` exposes raw
    timestamp + run status.
    """
    from skillbridge.chat.development_plan import emit_shadow_trace

    # Engine completion: if we reached this point with in_memory_matches
    # populated by compute_matches_in_memory (or even empty after a
    # successful no-match run), the engine completed. Engine failures
    # raise before we get here.
    engine_completed = True

    # Snapshot usability — thin derivation.
    snapshot_usable = _derive_snapshot_usable()

    # Target posting count — only for exact 5-digit target NOC.
    target_posting_count = _derive_target_posting_count(staged.target_noc)

    # Skill-adjacent results: use the existing tier evidence Sideways
    # records as the adjacency proxy. Empty list when no tier evidence
    # was built (the diagnosis treats that as "no skill-adjacent found").
    skill_adjacent_results: list = []
    if tier_evidence is not None:
        skill_adjacent_results = list(
            getattr(tier_evidence, "sideways_move", ()) or ()
        )

    # H2 + H1 Round 3 fix (2026-06-15): the locked diagnosis contract
    # says direct_match_results is scoped to the user's target NOC
    # when resolved. The previous Round-1 fix filtered the global
    # top-20 by NOC, but target-NOC postings ranked below 20 globally
    # were invisible. Round 3: when the target NOC resolves to an
    # exact 5-digit value, fetch and score the COMPLETE target-NOC
    # local posting set so CP4's candidate-gap collection sees every
    # underqualified target posting.
    target_noc_value = (
        staged.target_noc if isinstance(staged.target_noc, str) else None
    )
    if (
        isinstance(target_noc_value, str)
        and len(target_noc_value.strip()) == 5
        and target_noc_value.strip().isdigit()
    ):
        target_noc_value = target_noc_value.strip()
        direct_match_results_scoped = _score_target_noc_postings_for_shadow(
            staged, target_noc_value,
        )
        # Defensive fallback: if the target-scoped fetch returned []
        # for any reason (DB hiccup, no eligible jobs), fall back to
        # filtering the global top-20 by NOC so the shadow trace
        # still produces something rather than silently emit no
        # evidence. The fallback is logged so the discrepancy is
        # auditable.
        if not direct_match_results_scoped:
            direct_match_results_scoped = [
                r for r in in_memory_matches
                if isinstance(getattr(r, "noc_code", None), str)
                and r.noc_code == target_noc_value
            ]
    else:
        direct_match_results_scoped = list(in_memory_matches)

    emit_shadow_trace(
        staged=staged,
        user_message=user_message,
        truth_enough_to_match=bool(getattr(truth, "enough_to_match", False)),
        truth_usable_evidence_present=bool(
            getattr(truth, "usable_evidence_present", False),
        ),
        engine_completed=engine_completed,
        in_memory_matches=direct_match_results_scoped,
        skill_adjacent_results=skill_adjacent_results,
        snapshot_usable=snapshot_usable,
        target_posting_count=target_posting_count,
    )


def _derive_snapshot_usable() -> bool:
    """Thin read-only derivation. M2 fix (2026-06-15): a valid
    publication whose current-job count is ZERO is still a usable
    snapshot — that maps cleanly to NO_OPPORTUNITY_FOUND/LOCAL_INVENTORY_GAP
    at the diagnosis layer. Previously this conflated "no jobs found
    in this snapshot" with "snapshot unavailable", which forced
    MARKET_DATA_UNAVAILABLE on a valid empty board.

    Usability now depends solely on whether the publication has a
    timestamp. Best-effort; any exception → False (conservative).
    """
    try:
        from skillbridge.chat.pipeline_snapshot import fetch_pipeline_snapshot
        snap = fetch_pipeline_snapshot()
        return getattr(snap, "last_publish_at_text", None) is not None
    except Exception:  # noqa: BLE001
        return False


def _score_target_noc_postings_for_shadow(
    staged: StagedProfile, target_noc_value: str,
) -> list:
    """Score the user against EVERY currently-live local posting in
    the user's exact 5-digit target NOC.

    H1 fix Round 3 (2026-06-15): the handler previously filtered the
    global top-20 `in_memory_matches` to the target NOC. Target-NOC
    postings ranked below 20 globally were invisible, producing
    incomplete CP4 candidates and misleading shadow results.

    This helper bounded by `target_posting_count` (typically 0-15
    on the SSM board). Single fetch of postings, one
    `_score_one_job` call per posting. The output is sorted by
    `job_id` ascending so the shadow trace is deterministic.

    Best-effort: any exception returns []; the shadow caller falls
    back to the filtered global subset rather than break.
    """
    try:
        from skillbridge.db import sync_cursor
        from skillbridge.match.engine import (
            _fetch_job_skill_embeddings,
            _fetch_job_skills,
            _maybe_embed_user_skill_rows,
            _score_one_job,
        )
        from skillbridge.match.alignment import (
            build_user_skill_rows,
            derive_user_skill_sets,
        )
        from skillbridge.match.region import is_ssm_region_job
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "cp4_shadow target-scoped scoring import failed: %s",
            type(exc).__name__,
        )
        return []

    try:
        with sync_cursor() as cur:
            cur.execute(
                "SELECT * FROM core.v_current_job WHERE noc_code = %s",
                (target_noc_value,),
            )
            rows = list(cur.fetchall())
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "cp4_shadow target-scoped fetch failed: %s",
            type(exc).__name__,
        )
        return []

    local_rows = [r for r in rows if is_ssm_region_job(r)]
    if not local_rows:
        return []

    user_rows = build_user_skill_rows(staged.skills)
    user_skill_ids, user_skill_names, user_skill_names_canon = (
        derive_user_skill_sets(user_rows)
    )
    user_embeddings_matrix = _maybe_embed_user_skill_rows(user_rows)

    profile_dict = {
        "profile_id": staged.session_id,
        "preferred_location": staged.preferred_location,
        "target_role_text": staged.target_role_text,
        "target_noc": target_noc_value,
        "work_type_preference": staged.work_type_preference,
        "shift_preference": staged.shift_preference,
        "experience_text": staged.experience_text,
    }

    results: list = []
    for job in local_rows:
        try:
            job_skills = _fetch_job_skills(str(job["job_id"]))
            job_skill_embeddings = (
                _fetch_job_skill_embeddings(str(job["job_id"]))
                if user_embeddings_matrix is not None else None
            )
            m = _score_one_job(
                job, job_skills, user_skill_ids, user_skill_names,
                profile_dict,
                user_skill_names_canon=user_skill_names_canon,
                user_rows=user_rows,
                user_embeddings_matrix=user_embeddings_matrix,
                job_skill_embeddings=job_skill_embeddings,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "cp4_shadow target-scoped per-job score failed: %s",
                type(exc).__name__,
            )
            continue
        if m is not None:
            results.append(m)

    results.sort(key=lambda r: r.job_id)
    return results


def _derive_target_posting_count(target_noc) -> int | None:
    """Count of currently-live LOCAL (SSM-proper) postings in the
    user's exact 5-digit target NOC. Returns None when the target NOC
    is vague (4-digit or shorter), unresolved, or any other non-5-digit
    value.

    H4 fix (2026-06-15): the previous SQL counted every posting with
    the NOC regardless of region. Now uses `is_ssm_region_job()`
    Python-side as the single local-scope authority, matching the
    locked contract.
    """
    if not isinstance(target_noc, str):
        return None
    target_noc = target_noc.strip()
    if len(target_noc) != 5 or not target_noc.isdigit():
        return None
    try:
        from skillbridge.db import sync_cursor
        from skillbridge.match.region import is_ssm_region_job
        with sync_cursor() as cur:
            cur.execute(
                "SELECT job_id, region_code, location FROM core.v_current_job "
                "WHERE noc_code = %s",
                (target_noc,),
            )
            rows = cur.fetchall()
            return sum(1 for r in rows if is_ssm_region_job(r))
    except Exception:  # noqa: BLE001
        return None


def _tier_evidence_has_any_records(tier_evidence: "TieredEvidence") -> bool:
    """Centralised emptiness check. Used by the handler to decide
    whether to flip `tiered_evidence_available` on the arbiter
    re-dispatch."""
    return bool(
        tier_evidence.apply_today
        or tier_evidence.worth_a_try
        or tier_evidence.sideways_move
    )


def _build_adjacent_snapshot_from_sideways(
    staged: StagedProfile,
    sideways: tuple["AdjacentJob", ...],
) -> dict[str, Any]:
    """Project the Sideways tier into the `last_adjacent_snapshot`
    shape so ordinal follow-ups ("tell me about the second one")
    resolve against the records the responder just surfaced.

    Mirrors the shape produced by `_run_adjacency_engine_and_persist`
    (lines ~1086-1106) so `resolve_adjacent_followup` consumes both
    paths uniformly. CP2 step 6.1 — without this stamp the
    Sideways-only path would render correctly but the next-turn
    ordinal resolver would have nothing to bind against.

    `evidence_summary` reuses the "N of M required skills" convention
    from the legacy snapshot builder: N = transferable_pairs (skills
    that mapped to an adjacent requirement), M = transferable_pairs +
    important_gaps (non-credential required skills overall).
    Credentials are excluded from the denominator there too, matching
    the legacy builder's `_required_or_preferred` walk that skips
    credential rows.
    """
    from skillbridge.session.staging import (
        MAX_EVIDENCE_CHARS,
        MAX_JOB_ID_CHARS,
        MAX_MATCHED_SKILLS,
        MAX_SKILL_CHARS,
        MAX_TITLE_CHARS,
    )

    items: list[dict[str, Any]] = []
    for adj in sideways:
        matched_total = len(adj.transferable_pairs)
        required_total = matched_total + len(adj.important_gaps)
        evidence = (
            f"{matched_total} of {max(required_total, 1)} required skills"
        )[:MAX_EVIDENCE_CHARS]

        matched_display: list[str] = []
        for pair in adj.transferable_pairs[:MAX_MATCHED_SKILLS]:
            name = pair.applies_to
            if isinstance(name, str) and name.strip():
                matched_display.append(name[:MAX_SKILL_CHARS])

        items.append({
            "job_id": (adj.job_id or "")[:MAX_JOB_ID_CHARS],
            "title": (adj.title or "")[:MAX_TITLE_CHARS],
            "evidence_summary": evidence,
            "why_adjacent": str(adj.why_adjacent),
            "matched_skills": matched_display,
        })

    return {
        "created_message_count": staged.message_count,
        "items": items,
    }


# =========================================================================
# AR-6a -- adjacency dispatch (Redis-mode-gated short-circuit)
# =========================================================================
def _try_adjacency_dispatch(
    *,
    staged: StagedProfile,
    store,
    user_message: str,
    pending_adjacent_offer: bool,
    resume_info: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """When the adjacency layer can answer this turn, build the full v2
    response dict and return it. Otherwise return None and the caller
    (`_try_v2_path`) continues with the standard router/planner/engine
    flow.

    Two synthesis paths:
      1. Ordinal follow-up: the user resolved a reference like "the
         second one" against `staged.last_adjacent_snapshot`. We
         synthesize a `describe_adjacent_role` decision (the
         payload-shaped render lands in AR-6c).
      2. AdjacentIntent: the user explicitly asked for different
         roles. We run the AR-3 retrieve + accept + AR-4 rank +
         drop_excluded pipeline, persist `last_adjacent_snapshot`,
         and synthesize a `recommend_adjacent_roles` decision.
      3. NeedsEvidenceIntent: explicit phrasing matched but the user
         doesn't have evidence to anchor a recommendation. We
         synthesize a clarification asking for skills.

    Gated to Redis-mode sessions; cookie-mode users see the pre-AR-1
    experience.
    """
    from skillbridge.match.adjacent import _adjacency_enabled

    if not _adjacency_enabled():
        return None

    from skillbridge.chat.adjacent_followup import (
        render_describe_adjacent_role, resolve_adjacent_followup,
    )
    from skillbridge.chat.adjacent_intent import (
        AdjacentIntent, NeedsEvidenceIntent, detect_adjacent_intent,
    )
    from skillbridge.match.adjacent import (
        _synthesize_describe_adjacent_role_decision,
        _synthesize_recommend_adjacent_roles_decision,
        has_usable_skill_evidence,
    )

    # 1. Ordinal follow-up against the snapshot.
    followup_item = resolve_adjacent_followup(
        user_message,
        staged.last_adjacent_snapshot,
        staged.message_count,
    )
    if followup_item is not None:
        description_payload = render_describe_adjacent_role(followup_item)
        decision = _synthesize_describe_adjacent_role_decision()
        return _build_adjacency_short_circuit_response(
            staged=staged, store=store, user_message=user_message,
            decision=decision, resume_info=resume_info,
            adjacent_role_description_payload=description_payload,
        )

    # 2. Intent detection.
    intent = detect_adjacent_intent(
        message=user_message,
        staged=staged,
        user_has_evidence=has_usable_skill_evidence(staged),
        pending_offer=pending_adjacent_offer,
    )
    if isinstance(intent, AdjacentIntent):
        recommendations_payload = _run_adjacency_engine_and_persist(
            staged, trigger=intent.trigger,
        )
        decision = _synthesize_recommend_adjacent_roles_decision()
        return _build_adjacency_short_circuit_response(
            staged=staged, store=store, user_message=user_message,
            decision=decision, resume_info=resume_info,
            adjacent_recommendations_payload=recommendations_payload,
        )
    if isinstance(intent, NeedsEvidenceIntent):
        # Synthesize a clarification asking for skills/experience.
        # The reason_code reuses the existing
        # `insufficient_profile_evidence` ReasonCode -- the responder
        # narration already covers the "we need a bit more context"
        # shape. AR-6c may refine to a dedicated reason_code if the
        # responder needs to differentiate this path from the
        # standard intake clarification.
        clarification = ArbiterDecision(
            final_move="ask_one_clarifying_question",
            reason_code="insufficient_profile_evidence",
            tone="warm_supportive",
            arbiter_action="handler_synthesized_clarification",
            ask_slot="skills_text",
        )
        return _build_adjacency_short_circuit_response(
            staged=staged, store=store, user_message=user_message,
            decision=clarification, resume_info=resume_info,
        )
    return None


def _run_adjacency_engine_and_persist(
    staged: StagedProfile,
    *,
    trigger: str = "user_explicit",
) -> dict[str, Any]:
    """Run the AR-3 retrieve + accept + AR-4 rank + drop_excluded
    pipeline, persist `last_adjacent_snapshot`, and return the
    responder payload.

    Returns the `adjacent_recommendations_payload` shape locked in
    v11 §"Locked StagedProfile / ResponderV2Input additions". The
    snapshot fields and the responder payload share identical content
    so the LLM and the ordinal-followup resolver see the same data.

    Cap discipline (AR-6c review round 2):
      - `matched_skills` capped at `MAX_MATCHED_SKILLS` (4); the
        EVIDENCE COUNT is computed BEFORE the cap so a 6-of-6 match
        narrates as "6 of 6" instead of "4 of 6".
      - `matched_skills` strings capped at `MAX_SKILL_CHARS`.
      - `job_id` / `title` / `employer` / `evidence_summary`
        truncated to their staging constants.
    """
    from skillbridge.match.adjacent import (
        _load_active_jobs_with_skills,
        accept_candidates,
        build_user_skill_sets,
        drop_excluded,
        is_credential_skill_name,
        rank_adjacent,
        retrieve_candidates,
    )
    from skillbridge.match.engine import (
        _required_or_preferred, _skill_match_strength,
    )
    from skillbridge.session.staging import (
        MAX_EMPLOYER_CHARS,
        MAX_EVIDENCE_CHARS,
        MAX_JOB_ID_CHARS,
        MAX_MATCHED_SKILLS,
        MAX_SKILL_CHARS,
        MAX_TITLE_CHARS,
    )

    created_msg = staged.message_count
    target_minor = ""
    if isinstance(staged.target_noc, str):
        target_minor = staged.target_noc[:4]

    user_ids, user_names, user_canon = build_user_skill_sets(staged.skills)
    all_jobs = _load_active_jobs_with_skills()
    retrieved = retrieve_candidates(
        staged, snapshot=None, all_jobs=all_jobs,
        user_ids=user_ids, user_names=user_names, user_canon=user_canon,
    )
    accepted, drops = accept_candidates(
        retrieved, staged, user_ids, user_names, user_canon,
    )
    ranked = rank_adjacent(accepted, user_ids, user_names, user_canon)

    presented: tuple[str, ...] = ()
    if isinstance(staged.last_match_snapshot, dict):
        raw_presented = staged.last_match_snapshot.get("presented_job_ids") or ()
        if isinstance(raw_presented, (list, tuple)):
            presented = tuple(x for x in raw_presented if isinstance(x, str))

    filtered = drop_excluded(ranked, presented)
    top3 = [j for j in filtered[:3] if isinstance(j, dict)]

    snapshot_items: list[dict[str, Any]] = []
    payload_recs: list[dict[str, Any]] = []
    for j in top3:
        raw_job_id = j.get("job_id") if isinstance(j.get("job_id"), str) else ""
        raw_title = j.get("title") if isinstance(j.get("title"), str) else ""
        raw_employer = j.get("employer") if isinstance(j.get("employer"), str) else None

        # Walk the required non-credential skills. Count BEFORE
        # truncation so the evidence string reflects the real ratio
        # ("6 of 6 required skills") not the displayed-subset ratio.
        all_matched_unique: list[str] = []
        seen: set[str] = set()
        required_total = 0
        for s in j.get("skills") or []:
            if not isinstance(s, dict):
                continue
            name = s.get("skill_name")
            if not isinstance(name, str) or not name.strip():
                continue
            if is_credential_skill_name(name):
                continue
            bucket = (
                _required_or_preferred(s)
                if isinstance(s.get("skill_type"), (str, type(None)))
                else "required"
            )
            if bucket != "required":
                continue
            required_total += 1
            strength, _stage = _skill_match_strength(
                s, user_ids, user_names, user_canon,
            )
            if strength > 0.0 and name not in seen:
                seen.add(name)
                all_matched_unique.append(name)

        # Evidence count uses the FULL unique-match list, before cap.
        total_matched = len(all_matched_unique)
        # Display list capped at the contract limit.
        matched_display = [
            (n[:MAX_SKILL_CHARS] if isinstance(n, str) else "")
            for n in all_matched_unique[:MAX_MATCHED_SKILLS]
        ]
        matched_display = [n for n in matched_display if n]

        evidence = f"{total_matched} of {max(required_total, 1)} required skills"
        evidence = evidence[:MAX_EVIDENCE_CHARS]

        # Adjacency reason: NOC minor-group hit vs. skill-evidence hit.
        job_noc = j.get("noc_code") if isinstance(j.get("noc_code"), str) else ""
        why = (
            "same_noc_minor_group"
            if (target_minor and job_noc[:4] == target_minor)
            else "skill_evidence"
        )

        job_id = raw_job_id[:MAX_JOB_ID_CHARS]
        title = raw_title[:MAX_TITLE_CHARS]
        employer = (
            raw_employer[:MAX_EMPLOYER_CHARS]
            if isinstance(raw_employer, str) and raw_employer else None
        )

        snapshot_items.append({
            "job_id": job_id,
            "title": title,
            "evidence_summary": evidence,
            "why_adjacent": why,
            "matched_skills": list(matched_display),
        })
        payload_recs.append({
            "job_id": job_id,
            "title": title,
            "employer": employer,
            "location": "Sault Ste. Marie, ON",
            "evidence_summary": evidence,
            "why_adjacent": why,
            "matched_skills": list(matched_display),
        })

    staged.last_adjacent_snapshot = {
        "created_message_count": created_msg,
        "items": snapshot_items,
    }

    # AR-7 aggregate telemetry. Per-turn log line on
    # `skillbridge.chat.adjacent` -- enough signal to track
    # acceptance funnel without PII (no target_role_text, no
    # job titles, no employer names; just counts + trigger). Drop
    # buckets correspond 1:1 to accept_candidates' drops dict.
    from skillbridge.match.adjacent import ADJACENT_MIN_REQUIRED_COVERAGE
    log.info(
        "adjacent_recommendations turn=%d "
        "candidates_returned=%d candidate_pool=%d "
        "dropped_by_credential=%d dropped_by_coverage=%d "
        "dropped_by_transferable=%d "
        "dropped_no_required_non_credential_skills=%d "
        "trigger=%s adjacent_min_required_coverage=%.2f",
        staged.message_count,
        len(payload_recs), len(retrieved),
        drops.get("credential", 0),
        drops.get("coverage", 0),
        drops.get("transferable", 0),
        drops.get("no_required_non_credential_skills", 0),
        trigger,
        ADJACENT_MIN_REQUIRED_COVERAGE,
    )

    return {
        "recommendations": payload_recs,
        "total_retrieved": len(retrieved),
        "total_dropped_by_credential_gap": drops.get("credential", 0),
        "total_dropped_by_coverage_floor": drops.get("coverage", 0),
        "total_dropped_by_transferable_floor": drops.get("transferable", 0),
        "total_dropped_by_no_required_non_credential_skills": drops.get(
            "no_required_non_credential_skills", 0,
        ),
    }


def _build_adjacency_short_circuit_response(
    *,
    staged: StagedProfile,
    store,
    user_message: str,
    decision: "ArbiterDecision",
    resume_info: dict[str, Any] | None,
    adjacent_recommendations_payload: dict[str, Any] | None = None,
    adjacent_role_description_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose the response for a handler-synthesized adjacency turn.
    Skips planner / router / arbiter / engine. Mirrors
    `_build_remaining_gaps_short_circuit_response` (R-3).

    The responder payload extensions
    (`adjacent_recommendations_payload`,
    `adjacent_role_description_payload`) are passed through to the
    OUTCOME_RESPONDER_PROMPT narration shapes (AR-1c) so the LLM
    sees the structured data; the AR-6c fallback path narrates them
    deterministically when the LLM is disabled.
    """
    reply = compose_response_v2(ResponderV2Input(
        user_message=user_message,
        decision=decision,
        results=[],
        training_by_job={},
        next_skill=(None, 0),
        band_signal="none",
        requires_consent=True,
        target_role_text=staged.target_role_text,
        resume_facts=_effective_facts_view(staged),
        conversation_context=_build_conversation_context(staged),
        adjacent_recommendations_payload=adjacent_recommendations_payload,
        adjacent_role_description_payload=adjacent_role_description_payload,
    ))
    staged.last_asked_slots = [decision.ask_slot] if decision.ask_slot else []
    staged.touch()
    new_session_id = store.save(staged)
    log.info(
        "anon_chat_v2 session=%s adjacency action=%s final_move=%s",
        staged.session_id[:8], decision.arbiter_action, decision.final_move,
    )
    # Adjacency turns deliberately pass results=[]. Adjacent recommendations
    # and role descriptions are rendered as conversational prose with
    # validated inline links (per bug.2b for describe_adjacent_role).
    # recommended_jobs is reserved for direct matches that surface as
    # EmbeddedJobCard components; routing adjacency entries through it
    # would conflict with the locked product direction.
    return _build_v2_response(
        staged=staged, new_session_id=new_session_id, reply=reply,
        final_move=decision.final_move, ask_slot=decision.ask_slot,
        resume_info=resume_info,
        results=[], training_by_job={}, next_skill=(None, 0),
    )


def _try_v2_path(
    *,
    staged: StagedProfile,
    message: str | None,
    uploaded_file: bool,
    resume_info: dict[str, Any] | None,
    store,
    pending_adjacent_offer: bool = False,
    pattern_2_consent: str | None = None,
) -> dict[str, Any] | None:
    """v2 dispatch entry. Returns either a complete response dict (v2
    fully handled the turn) or None (explicit fallback_to_legacy
    signal -- caller drops to v1 path).

    Calls `staged.touch()` only on its terminal paths so the
    first_turn_greeting gate can observe message_count == 0 on a
    user's first message. If we return None, the v1 path's touch()
    runs in our place.
    """
    user_message = message or ""

    # ---- Step 1: gates (deterministic short-circuits) ----
    # `message_count` is read BEFORE touch() so the first_turn gate
    # fires correctly on turn zero. Both empty_input and resume_upload
    # gates are typically pre-empted earlier in handle_anonymous (empty
    # check at the top + the PRESENT_RESUME_FACTS short-circuit), but
    # we still evaluate them here for defense in depth and to keep the
    # gate dispatch surface uniform.
    gate = chat_gates.evaluate_gates(
        user_message=user_message,
        uploaded_file=uploaded_file,
        message_count=staged.message_count,
    )
    if gate is not None and gate.canned_response is not None:
        # Canned-response gate (empty_input or first_turn_greeting):
        # skip planner + arbiter + responder LLM entirely. This is the
        # whole point of the gate -- no decision to route.
        # Clear last_asked_slots so the next turn's slot-answer guard
        # sees the correct (empty) state -- gates don't ask for slots.
        staged.last_asked_slots = []
        staged.touch()
        new_session_id = store.save(staged)
        log.info(
            "anon_chat_v2 session=%s gate=%s canned",
            staged.session_id[:8], gate.gate_name,
        )
        return _build_v2_response(
            staged=staged, new_session_id=new_session_id,
            reply=gate.canned_response,
            final_move=gate.final_move,
            ask_slot=gate.ask_slot,
            resume_info=resume_info,
            results=[], training_by_job={}, next_skill=(None, 0),
        )
    if gate is not None:
        # Non-canned gate (resume_upload): planner + arbiter + engine
        # are SKIPPED because the gate has already decided the move,
        # but the responder v2 IS called -- the resume_upload outcome
        # needs to narrate the actual parsed facts, which the canned
        # path can't carry. We synthesize an ArbiterDecision from the
        # gate's GateDecision so the responder input shape is uniform.
        #
        # Note: in the current rollout, this branch is reachable only
        # when something upstream sets uploaded_file=True AFTER v1's
        # PRESENT_RESUME_FACTS short-circuit has already returned.
        # That doesn't happen through handle_anonymous today, but
        # `_try_v2_path` accepts uploaded_file directly so transcript
        # tests can drive this path. Wiring it through handle_anonymous
        # for production uploads is a deferred slice.
        synth = ArbiterDecision(
            final_move=gate.final_move,
            reason_code="gate:" + gate.gate_name,
            tone=gate.tone,
            arbiter_action="gate_fired",
            ask_slot=gate.ask_slot,
        )
        reply = compose_response_v2(ResponderV2Input(
            user_message=user_message,
            decision=synth,
            results=[], training_by_job={}, next_skill=(None, 0),
            band_signal="none",
            requires_consent=True,
            target_role_text=staged.target_role_text,
            resume_facts=_effective_facts_view(staged),
            conversation_context=_build_conversation_context(staged),
        ))
        # Non-canned gate (resume_upload) doesn't ask for a slot --
        # clear last_asked_slots like the canned-gate path does.
        staged.last_asked_slots = []
        staged.touch()
        new_session_id = store.save(staged)
        log.info(
            "anon_chat_v2 session=%s gate=%s final_move=%s",
            staged.session_id[:8], gate.gate_name, gate.final_move,
        )
        return _build_v2_response(
            staged=staged, new_session_id=new_session_id, reply=reply,
            final_move=gate.final_move, ask_slot=gate.ask_slot,
            resume_info=resume_info,
            results=[], training_by_job={}, next_skill=(None, 0),
        )

    # ---- Step 2: build truth summary (pure, no LLM) ----
    # Pre-compute registry-discovered gaps from the user message so
    # both truth_summary (for the planner's routing signal) and
    # _registry_training_for_gap (for the TRAINING payload) see the
    # same canonical list. Only when the flag is on; cheap pure-Python
    # scan otherwise.
    discovered_registry_gaps: list[str] = []
    if TRAINING_REGISTRY_ENABLED and user_message:
        try:
            from skillbridge.training.registry import get_registry as _get_reg
            _reg = _get_reg()
            all_found = [
                g.canonical_name
                for g in _reg.find_gaps_in_message(user_message)
            ]
            discovered_registry_gaps, suppressed = (
                _filter_registry_gaps_by_have_skills(all_found, staged.skills)
            )
            if suppressed:
                log.info(
                    "anon_chat_v2 session=%s "
                    "registry_gaps_suppressed_as_have_skills=%s",
                    staged.session_id[:8], suppressed,
                )
        except Exception as e:
            log.warning("training_registry message-scan failed: %s", e)
            discovered_registry_gaps = []

    truth = build_truth_summary(
        staged=staged,
        user_message=user_message,
        registry_gaps_in_message=discovered_registry_gaps,
    )
    truth_json = truth.to_planner_json()

    # AR-6a: scope-violated TTL shift. Runs IMMEDIATELY after the
    # truth summary is built, BEFORE any other dispatch (R-3,
    # adjacency, router, planner, engine). This guarantees that a
    # `redirect_scope` digression doesn't burn the
    # `last_adjacent_snapshot` TTL even when downstream paths
    # fall back to v1 (`return None`). See
    # docs/adjacent-recommendations-design.md v11 §"TTL forward-shift".
    scope_violated = bool(truth.scope_violations_detected)
    if scope_violated:
        from skillbridge.session.staging import shift_adjacent_snapshot_ttl
        shift_adjacent_snapshot_ttl(staged)

    # Slice 8 follow-up: surface the gating signals at INFO so live
    # acceptance can spot misclassifications (e.g. chat_skill_count
    # inflated by extraction, target_role_specificity not what we
    # expect). The fields named here are the ones that decide whether
    # the engine is allowed to run. Cheap log line; keep it.
    log.info(
        "anon_chat_v2 session=%s truth target_role_spec=%s "
        "resume_parse=%s chat_skill_count=%d enough_to_match=%s "
        "usable_evidence=%s scope_violations=%s intent=%s "
        "registry_gaps=%s",
        staged.session_id[:8],
        truth.target_role_specificity,
        truth.resume_parse_quality,
        sum(1 for s in staged.skills if s.source != "resume"),
        truth.enough_to_match,
        truth.usable_evidence_present,
        truth.scope_violations_detected,
        truth.user_intent_signal,
        truth.registry_gaps_in_message,
    )

    # Fresh-intake-on-target-change pillar (2026-06-15) — telemetry.
    # Only emitted when the alignment gate fires AND a GENUINE target
    # switch is present (the prior stamp points at a non-None target
    # that differs from the current target_role_text). Cold-profile
    # and partial-intake cases also fail `target_alignment_ok` but
    # they're normal intake flow — logging them would drown the
    # signal we actually care about (cross-target carry).
    if not truth.target_alignment_ok:
        prior_skills = staged.skills_collected_for_target
        prior_exp = staged.experience_collected_for_target
        current = staged.target_role_text

        def _is_real_drift(prior: str | None) -> bool:
            if not isinstance(prior, str) or not prior.strip():
                return False
            if not isinstance(current, str) or not current.strip():
                return False
            return prior.casefold().strip() != current.casefold().strip()

        if _is_real_drift(prior_skills) or _is_real_drift(prior_exp):
            chat_skill_count_now = sum(
                1 for s in staged.skills if s.source != "resume"
            )
            log.info(
                "anon_chat_v2 session=%s target_changed_intake_required "
                "prior_skills_target=%r prior_experience_target=%r "
                "new_target=%r skills_carried_over=%d "
                "experience_carried_over=%s first_misaligned_slot=%s",
                staged.session_id[:8],
                prior_skills, prior_exp, current,
                chat_skill_count_now,
                bool(staged.experience_text),
                truth.target_alignment_first_misaligned_slot,
            )

    # ---- Step 3.5: R-3 remaining-gaps handler-synthesis hook ----
    # When the detector returns a truthy RemainingGapsIntent the handler
    # bypasses planner / router / arbiter Pass 1 / engine / arbiter
    # Pass 2 entirely. Same control-flow pattern as gates.py. The hook
    # owns clearing staged.pending_credential_confirmation BEFORE
    # detection (the detector is pure and can't mutate StagedProfile);
    # _run_remaining_gaps_dispatch handles the try/except restore so a
    # detector failure preserves the pending question for next turn.
    #
    # Scope precedence (round-14 R-3 review): the hook MUST NOT run
    # when `truth.scope_violations_detected` is non-empty. Otherwise
    # a "what else about PR?" turn with a snapshot would synthesize
    # explain_remaining_gaps and silently hijack the immigration /
    # wages / off-topic / non-SSM redirect that the arbiter's Pass 1
    # Rule 2 owns. The snapshot is left intact across the redirect
    # (locked §10) so the user can return to the career conversation
    # immediately afterward.
    #
    # Pending one-turn-validity (round-15 R-3 review): the locked
    # contract is that `pending_credential_confirmation` is valid for
    # ONE turn only -- whatever the user does (answer, divert, ignore)
    # must consume or clear it. Normally `_run_remaining_gaps_dispatch`
    # owns the save-and-clear; on scope-diversion turns we bypass the
    # dispatch entirely, so we MUST clear pending here. Otherwise an
    # unrelated `yes` on a later turn could retroactively confirm the
    # stale pending question. Snapshot + accumulated +
    # last_discussed are PRESERVED per §10.
    #
    # See docs/remaining-gaps-design.md §R-3.
    if truth.scope_violations_detected:
        staged.pending_credential_confirmation = None
    else:
        _remaining_gaps_decision, _claims_count = _run_remaining_gaps_dispatch(
            staged, user_message,
        )
        if _remaining_gaps_decision is not None:
            return _build_remaining_gaps_short_circuit_response(
                staged=staged, store=store, user_message=user_message,
                final=_remaining_gaps_decision, resume_info=resume_info,
                current_turn_claims_count=_claims_count,
            )

    # AR-6a: adjacency dispatch (Redis-mode-gated). Runs AFTER R-3 so
    # same-role-gap phrasings stay R-3's territory; runs BEFORE the
    # router / planner / standard engine so a successful adjacency
    # synthesis bypasses them entirely. Skipped on:
    #   - scope-violated turns (the standard arbiter path emits
    #     redirect_scope; the TTL shift above already preserved the
    #     snapshot for the resolver's next on-topic turn);
    #   - cookie-mode sessions (per `_adjacency_enabled`, which gates
    #     the entire feature to Redis-backed stores).
    if not scope_violated:
        adjacency_response = _try_adjacency_dispatch(
            staged=staged, store=store, user_message=user_message,
            pending_adjacent_offer=pending_adjacent_offer,
            resume_info=resume_info,
        )
        if adjacency_response is not None:
            return adjacency_response

    # ---- Step 3a: deterministic router (chat orchestration v2.1) ----
    # Behind MESSAGE_UNDERSTANDING_ENABLED. When ON, the router decides
    # high-confidence routing (scope, training+entity, training-no-entity,
    # job_search+ready) and the planner LLM is SKIPPED. Medium/low
    # confidence falls through to the existing planner-first path.
    # When OFF (default during rollout) this block is a no-op and the
    # handler runs exactly as before.
    #
    # The router returns a PlannerDecision (or None) which slots directly
    # into the existing arbiter pass-1 contract -- no other code path
    # changes shape. See docs/message-understanding-design.md.
    planner_decision = None
    router_trace = None
    if MESSAGE_UNDERSTANDING_ENABLED and user_message:
        try:
            understanding = understand_message(
                user_message=user_message,
                registry_gaps_in_message=discovered_registry_gaps,
            )
            planner_decision, router_trace = route_from_understanding(
                understanding, truth_json,
            )
            log.info(
                "anon_chat_v2 session=%s router rule=%s skipped_planner=%s "
                "intent=%s confidence=%s entities=%s",
                staged.session_id[:8],
                router_trace.rule_fired,
                router_trace.planner_skipped,
                router_trace.understanding_intent,
                router_trace.understanding_confidence,
                router_trace.entity_canonical_names,
            )
        except Exception as e:
            # Router must NEVER block the handler. On any error, fall
            # through to the planner-first path -- exactly the OFF
            # behavior. Logged at WARNING so live tests catch it.
            log.warning(
                "message_understanding router failed: %s -- falling "
                "through to planner", e,
            )
            planner_decision = None
            router_trace = None

    # ---- Step 3b: planner LLM call (skipped when router decided) ----
    # If the router returned a decision (HIGH confidence rules 1-4), the
    # planner is NOT called -- this is the architectural promise from the
    # design doc. Otherwise the planner runs as today.
    if planner_decision is None:
        planner_decision = plan_next_move(truth_json)

    # ---- Step 4: arbiter pass 1 (pure, validates planner intent) ----
    pass1 = validate_planner_intent(planner_decision, truth_json)

    # ---- Step 4a: explicit fallback signal from arbiter ----
    if isinstance(pass1, ArbiterDecision) and pass1.arbiter_action == "fallback_to_legacy":
        # Drop to v1. Do NOT touch() here; the v1 path will do it.
        return None

    # Slice N: declared at top scope so both pass-1-terminal and
    # engine-ran branches can set it; the ResponderV2Input below
    # consumes it uniformly. Only the engine-ran path can populate
    # near_miss_payload; pass-1-terminal turns (gates, scope, etc.)
    # always leave it None.
    near_miss_payload: dict[str, Any] | None = None
    near_miss_candidates: list = []
    # AR-9.feat.coach-tiers CP2 step 4: same top-scope declaration so
    # the ResponderV2Input below can consume it on every path. Only
    # the engine-ran branch populates tier_evidence (after building
    # it pre-flag-flip and re-resolving the arbiter to
    # present_tiered_matches when at least one tier is populated).
    tier_evidence: "TieredEvidence | None" = None

    # CP4 shadow Round 3 fix (2026-06-15): `in_memory_matches` needs an
    # outer-scope default so ask-paths (where the engine doesn't run)
    # don't trigger UnboundLocalError when the CP4 shadow invocation
    # reads it. The engine-ran branch overwrites this with the real
    # MatchResult list. The diagnose() function correctly treats an
    # empty list combined with `enough_to_match=False` as
    # UNDETERMINED, so the shadow trace stays honest.
    in_memory_matches: list = []

    # AR-8c: same outer-scope declaration. Only the engine-ran
    # branch's lifecycle clear can hide the empty-adjacency state
    # from `_maybe_append_soft_offer`; pass-1-terminal turns never
    # touch `last_adjacent_snapshot`, so this stays False there.
    prior_empty_adjacency = False

    # ---- Step 5a: Pass 1 terminal -- NO engine invocation ----
    if isinstance(pass1, ArbiterDecision):
        final = pass1
        results: list[dict] = []
        training_by_job: dict[str, list[dict]] = {}
        next_skill: tuple[str | None, int] = (None, 0)
        band_signal = "none"
        # On explain_gap turns, populate TRAINING from the curated
        # registry when the feature flag is on. The registry's runtime
        # freshness check suppresses unverified URLs -- pending entries
        # come through as Resource objects with url=None so the
        # responder narrates the provider name + SCCC referral instead
        # of citing a URL we can't vouch for.
        if (
            CHAT_ORCHESTRATOR == "v2"
            and TRAINING_REGISTRY_ENABLED
            and final.final_move == "explain_gap"
        ):
            training_by_job = _registry_training_for_gap(
                staged, discovered_gaps=discovered_registry_gaps,
            )
    else:
        # ---- Step 5b: Pass 1 cleared the engine to run ----
        # The arbiter has independently verified usable_evidence_present,
        # enough_to_match, and no scope violations. This is the ONLY place
        # in the v2 path that calls compute_matches_in_memory.
        assert isinstance(pass1, RunEngine), (
            "Pass 1 must return ArbiterDecision or RunEngine"
        )
        in_memory_matches = match_engine.compute_matches_in_memory(staged, top=20)
        results, band_signal = _build_results_block(in_memory_matches)
        match_count = len(results)
        caps_applied = _collect_caps_applied(results)
        if match_count > 0:
            training_by_job = _attach_training(results)
            next_skill = match_engine.next_skill_to_unlock_in_memory(staged)
        else:
            training_by_job = {}
            next_skill = (None, 0)

        # ---- Step 5c: near-miss gap analysis (Slice N, 2026-06-05) ----
        # When match_count == 0 AND band_signal == "low_only" AND the
        # user has named a specific target AND has baseline evidence,
        # surface the LOCAL low-band candidates that title/NOC-match the
        # target as a "role exists, here are the blockers" outcome.
        # Otherwise fall through to present_no_match unchanged.
        #
        # Precondition gating (locked Q7) lives HERE in the handler
        # rather than in resolve_match_outcome -- keeps the arbiter a
        # thin outcome-selector and puts truth/staging-aware logic
        # next to the rest of the truth-aware code.
        near_miss_candidates, near_miss_payload = _compute_near_miss(
            match_count=match_count,
            band_signal=band_signal,
            in_memory_matches=in_memory_matches,
            staged=staged,
            truth=truth,
        )
        # If near-miss fired, also surface registry training for the
        # LEAD credential gap so the responder can name the provider
        # verbatim. Otherwise leave training_by_job as computed above.
        if near_miss_payload and near_miss_payload.get("credential_gaps"):
            lead_credential = near_miss_payload["credential_gaps"][0]
            near_miss_training = _registry_training_for_gap(
                staged, discovered_gaps=[lead_credential],
            )
            if near_miss_training:
                # Merge -- existing training_by_job is empty at this
                # point (match_count == 0 branch), but a defensive
                # update keeps the merge semantics explicit.
                training_by_job = {**training_by_job, **near_miss_training}

        # ---- Step 6: arbiter pass 2 (resolve to outcome move) ----
        final = resolve_match_outcome(
            match_count=match_count,
            caps_applied=tuple(caps_applied),
            near_miss_candidates=near_miss_candidates,
            planner_reason_code=pass1.planner_reason_code,
            planner_tone=pass1.planner_tone,
        )

        # AR-9.feat.coach-tiers CP2 step 4 — tier-evidence dispatch.
        #
        # When pass 2 would emit `present_matches`, build the tier
        # evidence package (with proactive adjacency) and re-resolve
        # the arbiter with `tiered_evidence_available=True` so the
        # responder takes the new three-tier surface. Signed-off pin:
        # tier evidence is built BEFORE the flag is set; the handler
        # is the SOLE authority for `tiered_evidence_available`.
        #
        # Defensive: when every tier comes back empty (no adjacency,
        # no Strong, no Stretch — implies match_count == 0 already,
        # which would not have produced present_matches here),
        # `tier_evidence` stays None and we stay on the legacy surface.
        # `tier_evidence` itself is declared at top scope above so the
        # pass-1-terminal branch and the ResponderV2Input below both
        # see a defined variable.
        if final.final_move == "present_matches":
            # NOTE: pass the raw `in_memory_matches` (list[MatchResult])
            # into the tier builder, NOT the `results` list (which
            # `_build_results_block` projected into dicts for the legacy
            # responder). The tier builder accesses MatchResult
            # attributes (.skill_alignment, .match_band, .match_eligible,
            # .score_explanation, ...).
            tier_evidence_candidate = _build_tier_evidence_for_handler(
                results=in_memory_matches,
                training_by_job=training_by_job,
                staged=staged,
            )
            if _tier_evidence_has_any_records(tier_evidence_candidate):
                tier_evidence = tier_evidence_candidate
                # Pattern 2 yes-consent display projection (Step 8,
                # closing-matrix v2, 2026-06-17): the prior turn rendered
                # the user's direct-target matches AND asked "want me to
                # also look at related roles?" — this turn's "yes" reply
                # was captured by the consume hook into `pattern_2_consent`.
                # On yes, suppress direct tiers so the surface is purely
                # the related-roles pivot the user asked for. Falls back
                # to present_no_match if sideways is also empty.
                if pattern_2_consent == "yes":
                    # Step 11j fix (2026-06-17): capture the
                    # pre-blank in_memory_matches BEFORE Step 11c
                    # clears it, so the later Step 11h CP4 fetch
                    # can run against the user's REAL match
                    # candidates (the ones that surfaced on the
                    # PRIOR turn before consent). Without this
                    # capture, Step 11h sees the empty post-blank
                    # in_memory_matches → CP4 diagnoses
                    # NO_OPPORTUNITY_FOUND → returns no
                    # recommendation → Movement C2 silently skips
                    # → user sees no gap callout. Live-verified
                    # symptom on 2026-06-17 (turn 5785a4a4).
                    _pre_blank_in_memory_matches = list(in_memory_matches)
                    tier_evidence = _blank_direct_tiers_for_pattern_2(
                        tier_evidence
                    )
                    # Step 11c (2026-06-17): also clear the raw
                    # MatchResults / training / caps / band-signal
                    # payload that drives the frontend's structured
                    # job-card render. The chat text comes from
                    # tier_evidence (just blanked) but the front-end
                    # job-card surface reads `results` / `caps_applied`
                    # / `training_by_job` / `band_signal` directly.
                    # Without this clearing, the chat says "no direct
                    # match" while the cards continue showing the
                    # prior turn's direct-target jobs — exactly the
                    # 2026-06-17 live-verify Pattern 2 round-trip bug.
                    # Mirrors the Bug B downgrade-to-no-match cleanup
                    # earlier in this same function (lines ~2222-2226).
                    results = []
                    in_memory_matches = []
                    match_count = 0
                    caps_applied = []
                    training_by_job = {}
                    band_signal = "none"
                    log.info(
                        "anon_chat_v2 session=%s "
                        "pattern_2_yes_suppressing_direct_tiers "
                        "sideways_count=%d",
                        staged.session_id[:8],
                        len(tier_evidence.sideways_move),
                    )
                    if not _tier_evidence_has_any_records(tier_evidence):
                        # No related roles either — fall through to
                        # present_no_match so the user gets the honest
                        # close (Step 9 will enhance it with SSM market
                        # summary).
                        final = resolve_match_outcome(
                            match_count=0,
                            caps_applied=(),
                            near_miss_candidates=near_miss_candidates,
                            planner_reason_code=pass1.planner_reason_code,
                            planner_tone=pass1.planner_tone,
                        )
                    else:
                        final = resolve_match_outcome(
                            match_count=match_count,
                            caps_applied=tuple(caps_applied),
                            near_miss_candidates=near_miss_candidates,
                            planner_reason_code=pass1.planner_reason_code,
                            planner_tone=pass1.planner_tone,
                            tiered_evidence_available=True,
                        )
                else:
                    final = resolve_match_outcome(
                        match_count=match_count,
                        caps_applied=tuple(caps_applied),
                        near_miss_candidates=near_miss_candidates,
                        planner_reason_code=pass1.planner_reason_code,
                        planner_tone=pass1.planner_tone,
                        tiered_evidence_available=True,
                    )
            elif (
                isinstance(staged.target_noc, str)
                and len(staged.target_noc) >= 4
            ):
                # Bug B fix (2026-06-15): the tier evidence is empty
                # AND a target NOC was resolved → the same-NOC-family
                # gate dropped every direct-tier candidate. The
                # engine's raw `in_memory_matches` are out-of-family
                # bleed (typically a prior target's skill matches that
                # survived a mid-session target change). Without this
                # branch, the legacy `present_matches` path would
                # surface those bled jobs as "stretch matches" via the
                # arbiter's pass-2 fallback at arbiter.py:604, which is
                # exactly the configuration CP4's diagnose contract
                # calls NO_OPPORTUNITY_FOUND. Downgrade to
                # `present_no_match` so the user-facing surface aligns
                # with CP4.
                #
                # CRITICAL: also clear `results`, `caps_applied`,
                # `training_by_job`, and `band_signal`. The arbiter only
                # decides the final_move; the response payload (which
                # the frontend renders as match-card tiles) is built
                # from `results` separately. Without clearing here, the
                # responder text says "no match" while the frontend
                # surfaces the bled jobs as Stretch-match cards — exactly
                # the live-2026-06-15 mixed-signal repro.
                log.info(
                    "anon_chat_v2 session=%s "
                    "downgrade_to_no_match_target_noc_family_empty "
                    "target_noc=%s raw_match_count=%d",
                    staged.session_id[:8], staged.target_noc, match_count,
                )
                results = []
                in_memory_matches = []
                match_count = 0
                caps_applied = []
                training_by_job = {}
                band_signal = "none"
                final = resolve_match_outcome(
                    match_count=0,
                    caps_applied=(),
                    near_miss_candidates=near_miss_candidates,
                    planner_reason_code=pass1.planner_reason_code,
                    planner_tone=pass1.planner_tone,
                )
        elif final.final_move == "present_no_match":
            # AR-9.feat.coach-tiers CP2 step 6.1 — Sideways-only path.
            # When the engine produced no Strong/Stretch but the user
            # has transferable skills, the proactive adjacency builder
            # may still surface Sideways records. Re-dispatch to
            # present_tiered_matches so the user sees the Adjacent
            # tier inline instead of the legacy "what other roles?"
            # soft offer. present_near_miss precedence is intact
            # because the arbiter chose present_no_match here, not
            # present_near_miss — a near-miss profile would have
            # bypassed this branch via the pass-2 ordering above.
            tier_evidence_candidate = _build_tier_evidence_for_handler(
                results=in_memory_matches,  # empty list; builder still tries adjacency
                training_by_job=training_by_job,
                staged=staged,
            )
            if _tier_evidence_has_any_records(tier_evidence_candidate):
                tier_evidence = tier_evidence_candidate
                final = resolve_match_outcome(
                    match_count=match_count,
                    caps_applied=tuple(caps_applied),
                    near_miss_candidates=near_miss_candidates,
                    planner_reason_code=pass1.planner_reason_code,
                    planner_tone=pass1.planner_tone,
                    tiered_evidence_available=True,
                )

        # AR-8c: capture empty-adjacency state BEFORE the AR-6c
        # lifecycle clear below. The clear on `present_matches` /
        # `present_near_miss` would otherwise hide the empty snapshot
        # from the soft-offer suppression check at line ~1488. By
        # snapshotting the bool here, AR-8c is reachable on every
        # outcome that runs the soft-offer hook.
        prior_empty_adjacency = _last_adjacency_was_empty(staged)

        # Slice 8 + Slice N: capture short-session context from this
        # match turn so the NEXT turn's responder fallback (esp. a
        # policy-rejected redirect) can reference it instead of starting
        # cold. Only update on present_matches; clear on present_no_match
        # AND present_near_miss so a cold redirect after either doesn't
        # reference a stale set.
        #
        # R-1 (remaining-gaps): runs in parallel with the legacy capture
        # until R-5 deprecates `last_presented_*`. Registry load is
        # decoupled from TRAINING_REGISTRY_ENABLED per design §4a -- we
        # attempt the load regardless of the flag because canonical
        # identity is needed by remaining-gaps reasoning even when
        # provider surfacing is off.
        #
        # AR-9.feat.coach-tiers CP2 step 4 (signed-off pin): the new
        # `present_tiered_matches` move shares the same session
        # snapshot lifecycle as legacy `present_matches`. Both capture
        # context + match snapshot and invalidate any standing
        # adjacent snapshot.
        if final.final_move in {"present_matches", "present_tiered_matches"}:
            _capture_presented_context(staged, results, caps_applied)
            _capture_match_snapshot(
                staged, results, registry=_try_load_registry_for_snapshot(),
            )
            # AR-6c lifecycle (locked v11 §"One-turn adjacent snapshot"):
            # a new direct-match decision invalidates any standing
            # adjacent recommendations. The user is back in the
            # standard match flow; ordinal follow-up on a stale
            # snapshot would surface roles they no longer asked about.
            #
            # CP2 step 6.1 — when the tiered surface includes a
            # populated Sideways tier, stamp `last_adjacent_snapshot`
            # with those records so ordinal follow-ups ("tell me about
            # the second one") still resolve. Otherwise clear: a fresh
            # direct-match-only response invalidates any prior adjacent
            # snapshot. This is what makes the Sideways-only path
            # round-trip-safe — without it the responder would surface
            # Sideways jobs but the next-turn resolver would have
            # nothing to bind ordinals against.
            if (
                final.final_move == "present_tiered_matches"
                and tier_evidence is not None
                and tier_evidence.sideways_move
            ):
                staged.last_adjacent_snapshot = (
                    _build_adjacent_snapshot_from_sideways(
                        staged, tier_evidence.sideways_move,
                    )
                )
            else:
                staged.last_adjacent_snapshot = None
        elif final.final_move in {"present_no_match", "present_near_miss"}:
            _clear_presented_context(staged)
            _clear_match_snapshot(staged)
            # AR-6c lifecycle: present_near_miss also clears the
            # snapshot (a near-miss is the user's current focus, NOT
            # a recommendation). present_no_match leaves it alone --
            # no new match means the existing snapshot might still be
            # the user's reference.
            if final.final_move == "present_near_miss":
                staged.last_adjacent_snapshot = None

    # ---- Step 7: responder v2 (LLM-narrated, deterministic fallback) ----
    # Slice N: near_miss_payload is None on every path except
    # present_near_miss; defined only inside the engine-ran branch
    # above. Defensive fall-back to None ensures the variable always
    # exists for the ResponderV2Input construction.
    #
    # AR-9.feat.coach-tiers CP2 step 4: tier_evidence is non-None only
    # when the dispatch above re-resolved to `present_tiered_matches`.
    # The responder's new branch reads only `tier_evidence` and
    # `pipeline_snapshot`; legacy fields are not consulted on that
    # path. We do not fetch a PipelineSnapshot here: the responder
    # branch only consults the snapshot on the empty-state fallback,
    # which is unreachable when `tier_evidence` has at least one
    # populated tier (the dispatch precondition above).
    # Resume-upload offer (2026-06-16): when the user has thin evidence
    # AND no resume AND the engine couldn't surface a strong/good
    # match, suggest uploading a CV/resume instead of falsely framing
    # the dataset as empty. See `_should_offer_resume_upload` for the
    # exact four-condition gate. Flip the staged flag here (BEFORE
    # rendering) so an LLM-render failure that falls back to template
    # still doesn't cause the offer to re-fire on the next thin-
    # evidence turn — once the user has heard it, they've heard it.
    offer_resume = _should_offer_resume_upload(
        staged=staged, final_move=final.final_move,
        band_signal=band_signal,
    )
    if offer_resume:
        staged.resume_upload_offered = True

    # Pattern 2 set hook (closing-matrix v2, Step 7b, 2026-06-17):
    # when the responder is about to render a tier surface with at
    # least one direct-target tier record AND the user has a resume
    # on file, the COACH_TIERS prompt emits the Pattern 2 closing
    # ("want me to also look at related roles?"). Flip the staged
    # flag here so the next turn's consume hook routes the user's
    # yes/no reply correctly.
    #
    # Slice 5 step 4 separation (2026-06-19): the recommender chain
    # used to also fire here, but it is now a fully-separate engine
    # with its own trigger. This SET only handles the matching
    # engine's Pattern 2 adjacency-consent offer.
    if (
        final.final_move == "present_tiered_matches"
        and staged.resume_facts_json
        and tier_evidence is not None
        and (
            tier_evidence.apply_today
            or tier_evidence.worth_a_try
            or tier_evidence.explore_later
        )
    ):
        staged.pending_adjacent_search_offer = True

    # Step 11d (2026-06-17): fetch the pipeline snapshot when the
    # responder will render a no-match-style outcome — needed by the
    # SHAPE 2 enhanced rule (Step 9) so the LLM has the SSM market
    # summary to weave when no direct + no adjacency surface. We
    # fetch on present_no_match AND present_tiered_matches with a
    # Sideways-only surface (Pattern 2 yes-consent / Pattern 3 auto-
    # fire), so the LLM can mention the dataset's freshness even when
    # showing related roles. Defensive try/except: a DB hiccup
    # shouldn't break the response; falls back to None and the
    # responder uses its existing legacy paths.
    _pipeline_snapshot = None
    if final.final_move in {"present_no_match", "present_tiered_matches"}:
        try:
            from skillbridge.chat.pipeline_snapshot import (
                fetch_pipeline_snapshot,
            )
            _pipeline_snapshot = fetch_pipeline_snapshot()
        except Exception as exc:  # noqa: BLE001
            log.info(
                "anon_chat_v2 session=%s pipeline_snapshot_fetch_failed=%s",
                staged.session_id[:8], type(exc).__name__,
            )

    # Step 11h (2026-06-17, closing-matrix v2): compute the CP4
    # primary recommendation's canonical skill name when the
    # responder will land on present_no_match AND the user has a
    # resume on file. The Movement C2 sub-rule of SHAPE 2 ENHANCED
    # RELATED_ROLES_EXHAUSTED (and the symmetric deterministic
    # fallback) quote this verbatim to give the user a concrete
    # gap callout instead of a generic training offer. CP4 is the
    # same engine already running as shadow telemetry (see
    # `_cp4_shadow_invocation` below) — this call invokes it for
    # the responder's benefit and discards the rest of the plan.
    # Defensive: any failure returns None and the responder reverts
    # to its non-personalized Movement C wording.
    _cp4_primary_gap = None
    if (
        final.final_move == "present_no_match"
        and staged.resume_facts_json
    ):
        try:
            from skillbridge.chat.development_plan import (
                compute_primary_gap_name,
            )
            _snapshot_usable_for_cp4 = _derive_snapshot_usable()
            _target_posting_count_for_cp4 = _derive_target_posting_count(
                staged.target_noc
            )
            _skill_adjacent_for_cp4 = []
            if tier_evidence is not None:
                _skill_adjacent_for_cp4 = list(
                    getattr(tier_evidence, "sideways_move", ()) or ()
                )
            # Step 11i fix (2026-06-17): `truth` is a TruthSummary
            # dataclass, NOT a dict — use attribute access. Live
            # verify on 2026-06-17 hit AttributeError because the
            # initial Step 11h code called truth.get(...). The
            # dataclass has the same field names; just read them
            # directly.
            # Step 11j fix (2026-06-17): when Pattern 2 yes-consent
            # fired this turn (Step 11c cleared in_memory_matches),
            # CP4 needs the PRE-BLANK matches to diagnose
            # PREPARATION_GAP and surface a real recommendation. Use
            # the captured `_pre_blank_in_memory_matches` when the
            # blanking ran; otherwise fall back to the live
            # in_memory_matches (Pattern 3 auto-fire and other
            # present_no_match paths). Without this, the diagnosis
            # is NO_OPPORTUNITY_FOUND on every yes-consent turn and
            # Movement C2 silently skips.
            _matches_for_cp4 = locals().get(
                "_pre_blank_in_memory_matches", in_memory_matches
            )
            if not _matches_for_cp4:
                _matches_for_cp4 = in_memory_matches
            _cp4_primary_gap = compute_primary_gap_name(
                staged=staged,
                user_message=user_message,
                truth_enough_to_match=bool(
                    getattr(truth, "enough_to_match", False)
                ),
                truth_usable_evidence_present=bool(
                    getattr(truth, "usable_evidence_present", False)
                ),
                engine_completed=True,
                in_memory_matches=_matches_for_cp4,
                skill_adjacent_results=_skill_adjacent_for_cp4,
                snapshot_usable=_snapshot_usable_for_cp4,
                target_posting_count=_target_posting_count_for_cp4,
            )
            if _cp4_primary_gap:
                log.info(
                    "anon_chat_v2 session=%s cp4_primary_gap=%r",
                    staged.session_id[:8], _cp4_primary_gap,
                )
        except Exception as exc:  # noqa: BLE001
            # Defensive — log type + first 80 chars of message so
            # future failure modes are diagnosable without spelunking.
            log.info(
                "anon_chat_v2 session=%s cp4_primary_gap_failed=%s msg=%s",
                staged.session_id[:8], type(exc).__name__, str(exc)[:80],
            )

    reply = compose_response_v2(ResponderV2Input(
        user_message=user_message,
        decision=final,
        results=results,
        training_by_job=training_by_job,
        next_skill=next_skill,
        band_signal=band_signal,
        requires_consent=True,
        target_role_text=staged.target_role_text,
        resume_facts=_effective_facts_view(staged),
        conversation_context=_build_conversation_context(staged),
        near_miss_payload=near_miss_payload,
        tier_evidence=tier_evidence,
        should_offer_resume_upload=offer_resume,
        pipeline_snapshot=_pipeline_snapshot,
        cp4_primary_gap=_cp4_primary_gap,
    ))

    # AR-6b: append the adjacency soft offer when the standard match
    # path emitted a credential-only-capped `present_matches` or a
    # genuine `present_no_match` AND the user has usable skill
    # evidence. Reoffer-suppressed via `pending_offer` (the entry-time
    # value of `pending_adjacent_offer`). Gated to Redis-mode +
    # ADJACENCY_ACTIVATION_ENABLED. Sets
    # `staged.pending_adjacent_offer = True` so the next turn's
    # `detect_adjacent_intent` consumes the affirmative.
    reply = _maybe_append_soft_offer(
        reply=reply,
        staged=staged,
        final=final,
        results=results,
        pending_offer=pending_adjacent_offer,
        prior_empty_adjacency=prior_empty_adjacency,
    )

    # ---- Step 7.5: persist last_asked_slots so the NEXT turn can see
    # which slot we just asked about. v1's intake_state.decide() loop
    # writes this; v2 had silently skipped it, which broke the
    # slot-answer guard in _extract (the guard reads
    # staged.last_asked_slots via the asked_slots parameter to decide
    # whether to run the rule-based fallback). Without this update,
    # a "role-name" reply on turn N had asked_slots=[] at turn N+1,
    # so the rule-based extractor fired and produced phantom skills.
    # Mirrors the v1 code at line ~1044-1050.
    if final.ask_slot:
        staged.last_asked_slots = [final.ask_slot]
        staged.mark_asked(final.ask_slot)
    else:
        staged.last_asked_slots = []

    # CP3 step 3 (2026-06-15) — pending_training_topic setter.
    # When the OUTGOING turn is Rule 3 ("what skill or certificate do
    # you want training for?"), flag the NEXT user turn so the
    # handler entry guard captures a bare topic answer ("Excel")
    # before normal extraction merges it as a profile skill. Every
    # other outgoing turn clears the flag defensively so stale state
    # cannot leak forward.
    staged.pending_training_topic = (
        final.reason_code == "training_request_no_entity"
    )

    # CP4 SHADOW INVOCATION (2026-06-15) — log-only, no user-facing
    # change. Builds the InventoryDiagnosis, runs CP4 when authorized,
    # emits sanitized telemetry. Wrapped in a defensive try/except so
    # the existing response path cannot be affected by any failure
    # in the shadow code.
    try:
        _cp4_shadow_invocation(
            staged=staged,
            user_message=user_message,
            truth=truth,
            in_memory_matches=in_memory_matches,
            tier_evidence=tier_evidence,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "cp4_shadow_invocation raised %s; suppressed (shadow-only)",
            type(exc).__name__,
        )

    # ---- Step 8: persist + return ----
    staged.touch()
    new_session_id = store.save(staged)
    log.info(
        "anon_chat_v2 session=%s final_move=%s reason=%s tone=%s "
        "arbiter_action=%s ask_slot=%s results=%d band=%s near_miss=%d",
        staged.session_id[:8], final.final_move, final.reason_code,
        final.tone, final.arbiter_action, final.ask_slot,
        len(results), band_signal, len(near_miss_candidates),
    )
    return _build_v2_response(
        staged=staged, new_session_id=new_session_id, reply=reply,
        final_move=final.final_move, ask_slot=final.ask_slot,
        resume_info=resume_info, results=results,
        training_by_job=training_by_job, next_skill=next_skill,
    )


# =========================================================================
# Slice 8 -- short-session conversation context
# =========================================================================
# These three helpers feed `ConversationContext` to the v2 responder so
# fallback paths (especially redirect_scope after a policy rejection)
# can reference what the user just saw. State is persisted on the
# StagedProfile across turns.
#
# Update rules:
#   - present_matches  -> capture titles + caps + credential gaps
#   - present_no_match -> clear (don't carry stale matches forward)
#   - any other move   -> leave previous context as-is (the user might
#                         redirect mid-stream and still want context
#                         from the prior match turn)
# =========================================================================
_PRESENTED_CONTEXT_CAP = 5  # how many titles / gaps to remember at most


def _capture_presented_context(
    staged: StagedProfile, results: list[dict], caps_applied: list[str],
) -> None:
    """After Pass 2 emits present_matches, capture compact context.

    Title list comes from the top results in display order. Caps list
    is the deduped union we already computed. Credential gap names are
    pulled from each result's `score_explanation.credential_gap_skills`
    -- a more user-friendly surface than the cap flag itself."""
    titles: list[str] = []
    for r in results[: _PRESENTED_CONTEXT_CAP]:
        t = r.get("title")
        if t and t not in titles:
            titles.append(t)
    staged.last_presented_job_titles = titles

    staged.last_presented_caps_applied = list(caps_applied)[: _PRESENTED_CONTEXT_CAP]

    gaps: list[str] = []
    for r in results[: _PRESENTED_CONTEXT_CAP]:
        se = r.get("score_explanation") or {}
        for gap in (se.get("credential_gap_skills") or []):
            if gap and gap not in gaps:
                gaps.append(gap)
            if len(gaps) >= _PRESENTED_CONTEXT_CAP:
                break
        if len(gaps) >= _PRESENTED_CONTEXT_CAP:
            break
    staged.last_presented_credential_gaps = gaps


def _clear_presented_context(staged: StagedProfile) -> None:
    """Reset the short-session context. Called when the engine ran
    but returned zero matches -- the user shouldn't see a fallback
    referencing roles that weren't actually surfaced this turn."""
    staged.last_presented_job_titles = []
    staged.last_presented_caps_applied = []
    staged.last_presented_credential_gaps = []


# =========================================================================
# R-1 -- remaining-gaps match snapshot
# =========================================================================
# Structured per-job snapshot of the most-recent present_matches turn.
# Lives on staged.last_match_snapshot and drives the follow-up reasoning
# layer added in R-2..R-6 (subtract / retract / clarify / bootstrap).
#
# This runs IN PARALLEL with _capture_presented_context until R-5 deprecates
# the legacy fields. Two separate state shapes during the transition is the
# safer migration path: existing fallbacks that read last_presented_* keep
# working; the new feature reads only last_match_snapshot.
#
# Registry policy (per design §4a): canonical alias resolution is
# decoupled from TRAINING_REGISTRY_ENABLED. The flag controls resource
# surfacing (provider names + URLs in TRAINING blocks); snapshot capture
# attempts the registry load regardless. Load failure falls through to
# Mode B (normalized-display canonicals); the snapshot is still built.
# =========================================================================
import re as _re_remaining_gaps


def _normalize_canonical(display: str) -> str:
    """Deterministic Mode-B / Mode-C canonical form for a display string.

    Lowercase, replace non-alphanumeric runs with a single space, strip
    surrounding whitespace, collapse internal whitespace. Used both as
    the snapshot's stored canonical when the registry is unavailable AND
    as the comparison key for the deterministic token-fallback matcher
    in R-2. Mirrors docs/remaining-gaps-design.md §4.1 / §4.3.
    """
    if not isinstance(display, str):
        return ""
    s = _re_remaining_gaps.sub(r"[^a-z0-9]+", " ", display.lower()).strip()
    return s


def _try_load_registry_for_snapshot():
    """Best-effort load of the training registry singleton. Returns None
    on any failure (file missing, YAML parse error, env var unset, import
    error). Caller falls back to Mode B canonical normalization."""
    try:
        from skillbridge.training.registry import get_registry
        return get_registry()
    except Exception as e:    # pragma: no cover - tested via Mode B path
        log.info("remaining_gaps_registry_unavailable_at_capture: %s", e)
        return None


def _capture_match_snapshot(
    staged: StagedProfile,
    results: list[dict],
    registry=None,
) -> None:
    """After present_matches, capture the structured snapshot that drives
    remaining-gaps reasoning on the next turn.

    Schema: docs/remaining-gaps-design.md §1. The lead_job is `results[0]`;
    other matches contribute minimal metadata only (job_id + title) so
    future job-pivot can address them without replicating the gap shape.

    Behavior:
      - clears `last_assumed_completed_credentials`,
        `last_discussed_credential_canonical`, and
        `pending_credential_confirmation` (a new snapshot resets the
        conversation state from the previous one)
      - resolves canonical aliases through `registry.lookup` when the
        registry is loaded (Mode A) or `_normalize_canonical` when it is
        not (Mode B)
      - dedupes `credential_gaps` by resolved canonical, preserving the
        first occurrence in engine order (round-9 R-1 invariant)
      - splits engine `missing_skills` into credentials
        (`score_explanation.credential_gap_skills`) vs the remainder
        (core_skill_gaps)
    """
    # AR-9.bug.1: capture prior presented_job_ids BEFORE we overwrite the
    # snapshot. The list accumulates across consecutive present_matches
    # turns so the adjacency engine can exclude EVERYTHING the user has
    # already seen, not just this turn's results.
    # Lifecycle: target-role change clears last_match_snapshot via
    # StagedProfile.__setattr__; present_no_match / present_near_miss
    # call _clear_match_snapshot. Both reset the accumulation, which is
    # what we want -- a fresh presentation context.
    prior_presented: list[str] = []
    if isinstance(staged.last_match_snapshot, dict):
        raw_prior = staged.last_match_snapshot.get("presented_job_ids")
        if isinstance(raw_prior, list):
            prior_presented = [x for x in raw_prior if isinstance(x, str) and x]

    # Atomic reset of companion state -- the prior snapshot's accumulated
    # assumptions belong to the prior match.
    staged.last_assumed_completed_credentials = []
    staged.last_discussed_credential_canonical = None
    staged.pending_credential_confirmation = None

    if not results:
        staged.last_match_snapshot = None
        return

    lead = results[0]
    score_explanation = lead.get("score_explanation") or {}
    credential_gap_names: list[str] = list(
        score_explanation.get("credential_gap_skills") or []
    )
    credential_gap_set = {n.lower() for n in credential_gap_names if isinstance(n, str)}
    all_missing: list[str] = list(lead.get("missing_skills") or [])
    core_skill_gap_names = [
        s for s in all_missing
        if isinstance(s, str) and s.lower() not in credential_gap_set
    ]

    def _resolve(display: str) -> str:
        if registry is not None:
            try:
                hit = registry.lookup(display)
            except Exception:    # pragma: no cover - defensive
                hit = None
            if hit is not None:
                return getattr(hit, "canonical_name", None) or _normalize_canonical(display)
        return _normalize_canonical(display)

    # Credential gaps: resolve canonical, dedupe by canonical, THEN cap.
    # The dedupe MUST happen BEFORE the cap; otherwise an early run of
    # duplicates (e.g. "G2 driver's licence" + "Class G driver's license"
    # both -> Class G canonical) would consume slots that should belong
    # to later unique credentials. Round-9 R-1 invariant + round-10
    # ordering fix.
    credential_gaps: list[dict[str, Any]] = []
    seen_canonicals: set[str] = set()
    for raw_display in credential_gap_names:
        if not isinstance(raw_display, str) or not raw_display:
            continue
        canonical = _resolve(raw_display)
        if not canonical:
            continue
        if canonical in seen_canonicals:
            continue
        seen_canonicals.add(canonical)
        credential_gaps.append({
            "display":   raw_display[: _stg_MAX_CANONICAL_CHARS],
            "canonical": canonical[: _stg_MAX_CANONICAL_CHARS],
        })
        if len(credential_gaps) >= _stg_MAX_CRED_GAPS:
            break

    # Core skill gaps: cap, truncate, display-only (v1 doesn't subtract
    # skills, so no canonicalisation is required here).
    core_skill_gaps: list[str] = []
    seen_skills: set[str] = set()
    for raw in core_skill_gap_names[: _stg_MAX_SKILL_GAPS * 2]:
        if not isinstance(raw, str) or not raw:
            continue
        key = raw.lower()
        if key in seen_skills:
            continue
        seen_skills.add(key)
        core_skill_gaps.append(raw[: _stg_MAX_CANONICAL_CHARS])
        if len(core_skill_gaps) >= _stg_MAX_SKILL_GAPS:
            break

    # Other-jobs metadata: minimal, capped, for future job-pivot.
    other_jobs_meta: list[dict[str, str]] = []
    for r in results[1:][: _stg_MAX_OTHER_JOBS]:
        ji = r.get("job_id")
        jt = r.get("title")
        if not isinstance(ji, str) or not isinstance(jt, str) or not ji or not jt:
            continue
        other_jobs_meta.append({
            "job_id": ji[: _stg_MAX_CANONICAL_CHARS],
            "title":  jt[: _stg_MAX_TITLE_CHARS],
        })

    # AR-9.bug.1: build presented_job_ids by accumulating THIS turn's
    # results (most recent first per the AR-1 staging contract) on top
    # of the prior list, deduped + capped. Empty in the snapshot when
    # results carry no usable job_ids; non-empty drives
    # `drop_excluded()` in adjacent.py so jobs the user already saw as
    # direct matches never reappear under "You Can Try."
    presented_job_ids: list[str] = []
    seen_ids: set[str] = set()
    for r in results:
        ji = r.get("job_id")
        if not isinstance(ji, str) or not ji:
            continue
        capped = ji[: _stg_MAX_JOB_ID_CHARS]
        if capped in seen_ids:
            continue
        seen_ids.add(capped)
        presented_job_ids.append(capped)
        if len(presented_job_ids) >= _stg_MAX_PRESENTED_JOB_IDS:
            break
    for prior_id in prior_presented:
        if len(presented_job_ids) >= _stg_MAX_PRESENTED_JOB_IDS:
            break
        capped = prior_id[: _stg_MAX_JOB_ID_CHARS]
        if not capped or capped in seen_ids:
            continue
        seen_ids.add(capped)
        presented_job_ids.append(capped)

    title_raw = lead.get("title")
    employer_raw = lead.get("employer")
    job_id_raw = lead.get("job_id")
    staged.last_match_snapshot = {
        "captured_at_turn": int(staged.message_count or 0),
        "lead_job": {
            "job_id":   (job_id_raw or "")[: _stg_MAX_CANONICAL_CHARS] if isinstance(job_id_raw, str) else "",
            "title":    (title_raw or "")[: _stg_MAX_TITLE_CHARS] if isinstance(title_raw, str) else "",
            "employer": (employer_raw[: _stg_MAX_EMPLOYER_CHARS]
                         if isinstance(employer_raw, str) else None),
            "credential_gaps": credential_gaps,
            "core_skill_gaps": core_skill_gaps,
        },
        "other_jobs_meta": other_jobs_meta,
        "presented_job_ids": presented_job_ids,
    }


def _clear_match_snapshot(staged: StagedProfile) -> None:
    """Atomically clear all four remaining-gaps fields. Called when no
    matches were presented this turn or when the snapshot is otherwise
    invalidated (target_role change is handled in StagedProfile.__setattr__)."""
    staged.last_match_snapshot = None
    staged.last_assumed_completed_credentials = []
    staged.last_discussed_credential_canonical = None
    staged.pending_credential_confirmation = None


# =========================================================================
# R-3 -- remaining-gaps handler-level synthesis
# =========================================================================
# When the detector returns a truthy RemainingGapsIntent, the handler
# synthesizes the ArbiterDecision DIRECTLY (planner / router /
# validate_planner_intent / engine / resolve_match_outcome all skip).
# Same control-flow pattern as gates.py.
#
# Identity contract: every canonical written to staged via these
# helpers MUST be a value the detector pulled VERBATIM from the
# snapshot's stored credential_gaps[*].canonical. The detector's
# §4.0 invariant guarantees this; these helpers DO NOT re-resolve.
# =========================================================================
def _snapshot_canonicals_from(staged: StagedProfile) -> set[str]:
    """Build the set of canonicals present in the current snapshot.
    Used as the identity authority (§4.0) at every write to
    `staged.last_assumed_completed_credentials`."""
    snap = staged.last_match_snapshot
    if not isinstance(snap, dict):
        return set()
    lead = snap.get("lead_job")
    if not isinstance(lead, dict):
        return set()
    out: set[str] = set()
    for g in (lead.get("credential_gaps") or []):
        if isinstance(g, dict):
            c = g.get("canonical")
            if isinstance(c, str):
                out.add(c)
    return out


def _sanitize_accumulated_against_snapshot(
    staged: StagedProfile,
) -> list[dict[str, Any]]:
    """Drop entries whose canonical isn't in the CURRENT snapshot AND
    dedupe duplicate canonicals (preserving first-position order,
    promoting hypothetical -> claimed if ANY duplicate is claimed),
    then write the sanitized list back to staged.

    Round-17 R-4 review: this MUST run before append/dedupe/cap so
    stale entries can't squeeze a valid current claim out via the
    cap. Round-18 R-4 review: the same write-time pass MUST dedupe
    duplicate canonicals. Otherwise a cookie that survived signature
    verification with `[{A, hypothetical}, {A, claimed}]` would:
      - narrate A twice in the responder block
      - keep `any_hypothetical=True` even though A is claimed
      - consume two cap slots
      - present conflicting provenance to telemetry

    Stale entries arise when a snapshot transition slips past the
    clearing rules (or state arrives from a forged / older cookie).
    The snapshot is the sole identity authority -- entries whose
    canonical isn't there are not actionable and are dropped.

    Logged at WARNING with the dropped canonical and current snapshot
    canonicals so the leak is visible in transcripts."""
    snapshot_canonicals = _snapshot_canonicals_from(staged)
    # First pass: drop wrong-type / wrong-snapshot entries while
    # preserving order of first occurrence.
    deduped: list[dict[str, Any]] = []
    seen_index: dict[str, int] = {}
    for a in list(staged.last_assumed_completed_credentials or []):
        if not isinstance(a, dict):
            continue
        c = a.get("canonical")
        mode = a.get("mode")
        if not isinstance(c, str) or mode not in {"claimed", "hypothetical"}:
            continue
        if c not in snapshot_canonicals:
            log.warning(
                "remaining_gaps_stale_accumulated_dropped canonical=%r "
                "snapshot_canonicals=%r", c, sorted(snapshot_canonicals),
            )
            continue
        if c in seen_index:
            # Duplicate canonical: promote to claimed if EITHER copy
            # is claimed; otherwise stay hypothetical. Position is
            # the first occurrence's position (preserved).
            existing = deduped[seen_index[c]]
            if existing["mode"] == "hypothetical" and mode == "claimed":
                existing["mode"] = "claimed"
            log.warning(
                "remaining_gaps_duplicate_accumulated_merged canonical=%r "
                "kept_mode=%r dropped_mode=%r",
                c, existing["mode"], mode,
            )
            continue
        seen_index[c] = len(deduped)
        deduped.append({"canonical": c, "mode": mode})
    staged.last_assumed_completed_credentials = deduped
    return deduped


def _synthesize_remaining_gaps_decision(
    staged: StagedProfile,
    intent,                                   # RemainingGapsIntent
    *,
    retracted: bool,
) -> "ArbiterDecision":
    """Mutate staged.last_assumed_completed_credentials per the intent
    (append-and-dedupe for `kind="subtract"`, filter-out for
    `kind="retract"`), then return the synthesized ArbiterDecision.

    The accumulation algorithm matches docs/remaining-gaps-design.md §3
    (ordered append-and-dedupe with hypothetical->claimed promotion).

    Round-17: the persisted accumulated list is sanitized against the
    CURRENT snapshot before append-and-cap so stale entries (snapshot
    transitions that escaped the clearing rules) can't consume the cap
    and squeeze out a valid current claim.
    """
    from skillbridge.chat.arbiter import (
        ARBITER_REASON_REMAINING_GAPS,
        ARBITER_REASON_REMAINING_GAPS_RETRACTED,
        ArbiterDecision,
    )
    # Round-17 R-4 review: snapshot-identity sanitization MUST come
    # first. Stale entries dropped here cannot fight for cap slots
    # against the current turn's claims.
    _sanitize_accumulated_against_snapshot(staged)

    if retracted:
        # Filter the named canonical out of accumulated; preserve order
        # of the remaining entries.
        target = intent.retract_canonical
        staged.last_assumed_completed_credentials = [
            a for a in staged.last_assumed_completed_credentials
            if a.get("canonical") != target
        ]
        reason = ARBITER_REASON_REMAINING_GAPS_RETRACTED
    else:
        # Ordered append-and-dedupe with hypothetical->claimed promotion.
        existing = list(staged.last_assumed_completed_credentials)
        existing_by_canonical = {
            a["canonical"]: a for a in existing if "canonical" in a
        }
        for claim in intent.current_turn_claims:
            entry = existing_by_canonical.get(claim.canonical)
            if entry is None:
                new_entry = claim.to_dict()
                existing.append(new_entry)
                existing_by_canonical[claim.canonical] = new_entry
            elif entry.get("mode") == "hypothetical" and claim.mode == "claimed":
                # Promote -- the same canonical was assumed hypothetically
                # earlier and is now explicitly claimed.
                entry["mode"] = "claimed"
        # Cap to MAX_CRED_GAPS (drop latest entries first; preserves
        # snapshot-order of first occurrence). R-1 invariant.
        staged.last_assumed_completed_credentials = existing[: _stg_MAX_CRED_GAPS]
        reason = ARBITER_REASON_REMAINING_GAPS

    return ArbiterDecision(
        final_move="explain_remaining_gaps",
        reason_code=reason,
        tone="warm_supportive",
        arbiter_action="handler_synthesized_remaining_gaps",
        ask_slot=None,
        caps_applied=(),
        notes=None,
    )


def _synthesize_clarification_decision(
    staged: StagedProfile,
    intent,                                   # RemainingGapsIntent
) -> "ArbiterDecision":
    """Synthesize the clarification ArbiterDecision and (for
    `kind="confirm"` with a non-empty canonical) set
    `staged.pending_credential_confirmation`.

    Round-13 contract: pending is set ONLY when
    `intent.confirmation_target_canonical` is a non-empty string.
    `kind="confirm"` with canonical=None (ambiguous disambiguation)
    explicitly clears pending so the next turn runs fresh detection
    on whatever credential the user names. `kind="bootstrap"` also
    leaves pending unset -- there's nothing to confirm.
    """
    from skillbridge.chat.arbiter import (
        ARBITER_REASON_BOOTSTRAP_MATCH,
        ARBITER_REASON_CONFIRM_CREDENTIAL,
        ArbiterDecision,
    )
    if intent.kind == "confirm":
        canonical = intent.confirmation_target_canonical
        if isinstance(canonical, str) and canonical:
            staged.pending_credential_confirmation = {
                "canonical": canonical,
                "action":    intent.pending_action or "add",
            }
            # Locked §2 lifecycle (round-14 R-3 review): the confirm
            # clarification sets last_discussed to the resolved
            # canonical so a next-turn anaphor ("yes I got it",
            # "what about that one?") resolves to the credential we
            # asked about, not to a stale one.
            staged.last_discussed_credential_canonical = canonical
        else:
            # Disambiguation question -- user will reply with a credential
            # NAME, not yes/no. Don't record an unusable pending entry.
            # last_discussed is left UNCHANGED here per locked §2:
            # the prior recency anchor is still the best fallback the
            # next-turn anaphor resolver has if the user replies with
            # something other than an entity (e.g. "the second one").
            staged.pending_credential_confirmation = None
        reason = ARBITER_REASON_CONFIRM_CREDENTIAL
    else:                                       # kind == "bootstrap"
        staged.pending_credential_confirmation = None
        reason = ARBITER_REASON_BOOTSTRAP_MATCH

    return ArbiterDecision(
        final_move="ask_one_clarifying_question",
        reason_code=reason,
        tone="warm_supportive",
        arbiter_action="handler_synthesized_clarification",
        ask_slot=None,
        caps_applied=(),
        notes=None,
    )


def _build_remaining_gaps_payload(
    staged: StagedProfile,
) -> "dict[str, Any] | None":
    """Compute the REMAINING_GAPS payload (design §9) from the current
    `staged.last_match_snapshot` + `staged.last_assumed_completed_credentials`.

    Returns None when the snapshot is missing or malformed (defensive --
    the dispatch hook only fires when a snapshot exists, but we still
    guard so a future caller that loses the invariant doesn't crash).

    Identity contract: every canonical value in the returned payload
    is pulled VERBATIM from the snapshot or the accumulated state. No
    fresh registry resolutions, no normalization.
    """
    snap = staged.last_match_snapshot
    if not isinstance(snap, dict):
        return None
    lead = snap.get("lead_job")
    if not isinstance(lead, dict):
        return None
    raw_credential_gaps = lead.get("credential_gaps")
    if not isinstance(raw_credential_gaps, list):
        return None

    # Build the snapshot identity authority FIRST -- accumulated entries
    # whose canonical isn't in the snapshot are stale and must be dropped
    # (round-16 R-4 review). The snapshot is the sole identity source per
    # §4.0; a surviving entry whose canonical is gone from the snapshot
    # means a snapshot transition slipped past the clearing rules. Log
    # the stale entry so the gap is visible.
    snapshot_canonical_to_display: dict[str, str] = {}
    for g in raw_credential_gaps:
        if isinstance(g, dict):
            c = g.get("canonical")
            d = g.get("display")
            if isinstance(c, str) and isinstance(d, str):
                snapshot_canonical_to_display[c] = d
    snapshot_canonicals = set(snapshot_canonical_to_display)

    raw_accumulated = list(staged.last_assumed_completed_credentials or [])
    accumulated: list[dict[str, Any]] = []
    for a in raw_accumulated:
        if not isinstance(a, dict):
            continue
        c = a.get("canonical")
        mode = a.get("mode")
        if not isinstance(c, str) or mode not in {"claimed", "hypothetical"}:
            continue
        if c not in snapshot_canonicals:
            log.warning(
                "remaining_gaps_stale_accumulated_dropped canonical=%r "
                "snapshot_canonicals=%r", c, sorted(snapshot_canonicals),
            )
            continue
        accumulated.append({"canonical": c, "mode": mode})
    accumulated_canonicals = {a["canonical"] for a in accumulated}

    assumed_completed_payload: list[dict[str, Any]] = [
        {
            "display":   snapshot_canonical_to_display[a["canonical"]],
            "canonical": a["canonical"],
            "mode":      a["mode"],
        }
        for a in accumulated
    ]

    remaining_credentials: list[dict[str, str]] = []
    for g in raw_credential_gaps:
        if not isinstance(g, dict):
            continue
        c = g.get("canonical")
        d = g.get("display")
        if not isinstance(c, str) or not isinstance(d, str):
            continue
        if c in accumulated_canonicals:
            continue
        remaining_credentials.append({"display": d, "canonical": c})

    # Round-16 defensive fix: core_skill_gaps is supposed to be a list,
    # but a forged cookie that survived signature verification could
    # supply a dict (the list-comprehension below would otherwise
    # iterate dict keys and produce phantom skills like ["bad"]).
    raw_core_skill_gaps = lead.get("core_skill_gaps")
    if not isinstance(raw_core_skill_gaps, list):
        raw_core_skill_gaps = []
    remaining_core_skills = [
        s for s in raw_core_skill_gaps if isinstance(s, str) and s
    ]

    any_hypothetical = any(a["mode"] == "hypothetical" for a in accumulated)

    role = lead.get("title") if isinstance(lead.get("title"), str) else ""
    employer = lead.get("employer") if isinstance(lead.get("employer"), str) else None

    return {
        "role":     role,
        "employer": employer,
        "assumed_completed_credentials": assumed_completed_payload,
        "remaining_credentials": remaining_credentials,
        "remaining_core_skills": remaining_core_skills,
        "any_hypothetical": any_hypothetical,
    }


def _reground_training_for_lead_remaining(
    staged: StagedProfile,
    remaining_credentials: list[dict[str, str]],
) -> dict[str, list[dict]]:
    """Populate `training_by_job` for the LEAD remaining credential per
    design §9 ("Beyond that..." narration shape) and §4a (resource
    surfacing is the ONE thing TRAINING_REGISTRY_ENABLED still gates).

    Returns the standard `gap:<name>` → [resources] shape so the
    responder consumes it the same way it does for explain_gap.

    Returns {} when:
      - the flag is off (resource surfacing disabled by config)
      - remaining_credentials is empty (all-closed branch -- design §6
        explicitly forbids naming providers because there's nothing to
        ground them against)
      - the registry isn't loadable
      - the registry doesn't know the lead canonical
    """
    if not TRAINING_REGISTRY_ENABLED or not remaining_credentials:
        return {}
    lead_display = remaining_credentials[0].get("display")
    if not isinstance(lead_display, str) or not lead_display:
        return {}
    # Round-16 fix: explicitly disable the carry-forward source so
    # completed credentials (which are in last_presented_credential_gaps
    # from the prior present_matches turn but in
    # last_assumed_completed_credentials now) don't leak resources back
    # to the responder. We query for EXACTLY one gap: the lead remaining.
    return _registry_training_for_gap(
        staged, discovered_gaps=[lead_display],
        include_carry_forward=False,
    )


def _update_last_discussed_after_recompute(
    staged: StagedProfile,
    remaining_credentials: list[dict[str, str]],
) -> None:
    """Set `staged.last_discussed_credential_canonical` to the LEAD
    remaining credential after subtract / retract recomputation, so a
    next-turn anaphoric reference ("what about that one?") resolves
    correctly. When the all-credentials-closed branch fires
    (no remaining credentials), clear the field -- there's nothing to
    anchor against.
    """
    if remaining_credentials:
        staged.last_discussed_credential_canonical = \
            remaining_credentials[0]["canonical"]
    else:
        staged.last_discussed_credential_canonical = None


def _build_clarification_payload(
    staged: StagedProfile,
    reason_code: str,
) -> dict[str, Any]:
    """R-5: build the deterministic clarification payload from the
    just-synthesized state.

    For credential confirmation (reason=ARBITER_REASON_CONFIRM_CREDENTIAL):
      - When canonical is known: the synthesizer wrote pending_credential_
        confirmation = {canonical, action}. Look up display from the
        snapshot so the template can name the credential.
      - When canonical is None (ambiguous disambiguation): pending is
        cleared, so we emit a credential_completion_confirmation with
        canonical=None / display="" and action="add". The renderer's
        no-target template asks which credential the user meant.

    For bootstrap (reason=ARBITER_REASON_BOOTSTRAP_MATCH): no entity
    payload; the renderer uses a static template.
    """
    from skillbridge.chat.arbiter import (
        ARBITER_REASON_BOOTSTRAP_MATCH,
        ARBITER_REASON_CONFIRM_CREDENTIAL,
    )
    if reason_code == ARBITER_REASON_BOOTSTRAP_MATCH:
        return {"kind": "bootstrap_match_request"}
    if reason_code != ARBITER_REASON_CONFIRM_CREDENTIAL:
        # Unknown clarification reason -- fall back to the
        # disambiguation template, which is the safest empty shape.
        return {
            "kind": "credential_completion_confirmation",
            "credential_canonical": None,
            "credential_display":   "",
            "action":               "add",
        }

    # Build the list of trusted snapshot displays. The R-5 round-4
    # contract requires the renderer to verify any candidate display
    # against this list before interpolation -- syntactic validation
    # alone can't prove the value is grounded in current conversation
    # state. Forged or stale payloads whose display doesn't appear in
    # this list fall back to the safe no-target template.
    trusted_displays: list[str] = []
    snap = staged.last_match_snapshot
    if isinstance(snap, dict):
        lead = snap.get("lead_job")
        if isinstance(lead, dict):
            for g in (lead.get("credential_gaps") or []):
                if (
                    isinstance(g, dict)
                    and isinstance(g.get("display"), str)
                    and g["display"]
                ):
                    trusted_displays.append(g["display"])

    pending = staged.pending_credential_confirmation
    if not isinstance(pending, dict):
        # canonical=None branch -- the synthesizer cleared pending.
        return {
            "kind": "credential_completion_confirmation",
            "credential_canonical": None,
            "credential_display":   "",
            "trusted_displays":     trusted_displays,
            "action":               "add",
        }
    canonical = pending.get("canonical")
    action = pending.get("action") or "add"
    display = canonical
    if isinstance(snap, dict):
        lead = snap.get("lead_job")
        if isinstance(lead, dict):
            for g in (lead.get("credential_gaps") or []):
                if (
                    isinstance(g, dict)
                    and g.get("canonical") == canonical
                    and isinstance(g.get("display"), str)
                ):
                    display = g["display"]
                    break
    return {
        "kind": "credential_completion_confirmation",
        "credential_canonical": canonical,
        "credential_display":   display or "",
        "trusted_displays":     trusted_displays,
        "action":               action if action in {"add", "remove"} else "add",
    }


def _build_remaining_gaps_short_circuit_response(
    *,
    staged: StagedProfile,
    store,
    user_message: str,
    final: "ArbiterDecision",
    resume_info,
    current_turn_claims_count: int = 0,
) -> dict:
    """Compose the response for a handler-synthesized remaining-gaps turn
    (kind="subtract" | "retract" | "confirm" | "bootstrap"). Skips the
    planner / router / arbiter / engine chain entirely.

    For the explain_remaining_gaps final_move (kind="subtract" |
    "retract"), R-4 builds the structured REMAINING_GAPS payload from
    `staged.last_match_snapshot` + the just-mutated accumulated state,
    regrounds `training_by_job` for the lead remaining credential
    (flag-gated), and updates `last_discussed` to the new lead. For the
    clarification final_move (kind="confirm" | "bootstrap") the payload
    and training stay empty -- those are handled by R-5's clarification
    renderer.
    """
    from skillbridge.chat.arbiter import (
        ARBITER_REASON_BOOTSTRAP_MATCH,
        ARBITER_REASON_CONFIRM_CREDENTIAL,
    )
    remaining_gaps_payload: dict[str, Any] | None = None
    clarification_payload: dict[str, Any] | None = None
    training_by_job: dict[str, list[dict]] = {}
    if final.final_move == "explain_remaining_gaps":
        remaining_gaps_payload = _build_remaining_gaps_payload(staged)
        if remaining_gaps_payload is not None:
            remaining = remaining_gaps_payload["remaining_credentials"]
            training_by_job = _reground_training_for_lead_remaining(
                staged, remaining,
            )
            _update_last_discussed_after_recompute(staged, remaining)
    elif final.final_move == "ask_one_clarifying_question":
        # R-5: build the deterministic clarification payload. The
        # synthesizer already wrote pending state (for confirm with
        # canonical) and last_discussed; this helper just packs the
        # shape the responder template renders from.
        clarification_payload = _build_clarification_payload(
            staged, final.reason_code,
        )

    reply = compose_response_v2(ResponderV2Input(
        user_message=user_message,
        decision=final,
        results=[],
        training_by_job=training_by_job,
        next_skill=(None, 0),
        band_signal="none",
        requires_consent=True,
        target_role_text=staged.target_role_text,
        resume_facts=_effective_facts_view(staged),
        conversation_context=_build_conversation_context(staged),
        near_miss_payload=None,
        remaining_gaps_payload=remaining_gaps_payload,
        clarification_payload=clarification_payload,
    ))
    # final.ask_slot is None on both synthesized branches; mirror the
    # Step 7.5 logic so staged.last_asked_slots is reset to [] cleanly.
    staged.last_asked_slots = []

    staged.touch()
    new_session_id = store.save(staged)
    # R-4 telemetry: surface the synthesis-path summary alongside the
    # standard v2 line so live transcripts can audit accumulation +
    # remaining-credential counts + provenance over time.
    _accum = staged.last_assumed_completed_credentials or []
    _remaining = (remaining_gaps_payload or {}).get("remaining_credentials") or []
    _skills = (remaining_gaps_payload or {}).get("remaining_core_skills") or []
    _any_hypo = (remaining_gaps_payload or {}).get("any_hypothetical", False)
    log.info(
        "anon_chat_v2 session=%s final_move=%s reason=%s tone=%s "
        "arbiter_action=%s ask_slot=None results=0 band=none near_miss=0 "
        "remaining_gaps_intent=%s current_turn_claims_count=%d "
        "accumulated_credentials_count=%d any_hypothetical=%s "
        "remaining_credentials_count=%d remaining_skills_count=%d "
        "pending_action=%s",
        staged.session_id[:8], final.final_move, final.reason_code,
        final.tone, final.arbiter_action,
        _intent_from_reason(final.reason_code),
        current_turn_claims_count,
        len(_accum), _any_hypo, len(_remaining), len(_skills),
        (staged.pending_credential_confirmation or {}).get("action") or "none",
    )
    return _build_v2_response(
        staged=staged, new_session_id=new_session_id, reply=reply,
        final_move=final.final_move, ask_slot=final.ask_slot,
        resume_info=resume_info, results=[],
        training_by_job=training_by_job, next_skill=(None, 0),
    )


def _intent_from_reason(reason_code: str) -> str:
    """Telemetry mapping from synthesized reason_code to the
    remaining_gaps_intent label used in transcript audits."""
    from skillbridge.chat.arbiter import (
        ARBITER_REASON_BOOTSTRAP_MATCH,
        ARBITER_REASON_CONFIRM_CREDENTIAL,
        ARBITER_REASON_REMAINING_GAPS,
        ARBITER_REASON_REMAINING_GAPS_RETRACTED,
    )
    return {
        ARBITER_REASON_REMAINING_GAPS:           "subtract",
        ARBITER_REASON_REMAINING_GAPS_RETRACTED: "retract",
        ARBITER_REASON_CONFIRM_CREDENTIAL:       "confirm",
        ARBITER_REASON_BOOTSTRAP_MATCH:          "bootstrap",
    }.get(reason_code, "none")


def _run_remaining_gaps_dispatch(
    staged: StagedProfile,
    user_message: str,
) -> "tuple[ArbiterDecision | None, int]":
    """Detect + synthesize. Returns (ArbiterDecision | None,
    current_turn_claims_count). Decision is non-None when the handler
    has bypassed planner/engine; the count is the number of claims the
    detector emitted THIS turn -- threaded to telemetry per R-4 design.

    Pending-clear ownership (design §2): the handler saves
    `staged.pending_credential_confirmation` to a local AND clears the
    StagedProfile field BEFORE calling the detector (which is pure and
    can't mutate StagedProfile). If detection raises, the saved value
    is restored so the user's pending question is preserved.
    """
    from skillbridge.chat.remaining_gaps import detect_remaining_gaps_intent

    saved_pending = staged.pending_credential_confirmation
    staged.pending_credential_confirmation = None

    try:
        intent = detect_remaining_gaps_intent(
            user_message,
            staged.last_match_snapshot,
            _try_load_registry_for_snapshot(),
            accumulated_credentials=staged.last_assumed_completed_credentials,
            pending_confirmation=saved_pending,
            last_discussed_canonical=staged.last_discussed_credential_canonical,
        )
    except Exception as exc:                    # pragma: deliberate
        log.exception("remaining_gaps_detection_failed: %s", exc)
        staged.pending_credential_confirmation = saved_pending
        intent = None

    if intent is None:
        return None, 0

    claims_count = (
        len(intent.current_turn_claims) if intent.kind == "subtract" else 0
    )
    match intent.kind:
        case "subtract":
            return (
                _synthesize_remaining_gaps_decision(
                    staged, intent, retracted=False,
                ),
                claims_count,
            )
        case "retract":
            return (
                _synthesize_remaining_gaps_decision(
                    staged, intent, retracted=True,
                ),
                claims_count,
            )
        case "confirm":
            return _synthesize_clarification_decision(staged, intent), 0
        case "bootstrap":
            return _synthesize_clarification_decision(staged, intent), 0
        case _:
            log.warning(
                "remaining_gaps_unknown_kind=%r", intent.kind,
            )
            return None, 0


def _build_conversation_context(staged: StagedProfile) -> ConversationContext:
    """Snapshot staged into the responder's input dataclass. Frozen
    so the responder can't accidentally mutate session state."""
    return ConversationContext(
        target_role_text=staged.target_role_text,
        last_presented_job_titles=tuple(staged.last_presented_job_titles),
        last_presented_caps_applied=tuple(staged.last_presented_caps_applied),
        last_presented_credential_gaps=tuple(staged.last_presented_credential_gaps),
    )


# =========================================================================
# Slice N (2026-06-05) -- near-miss gap-analysis compute
# =========================================================================
# Pure-ish helper that owns the present_near_miss preconditions and the
# filter -> classify -> cap pipeline. Returns a (candidates, payload)
# tuple ready for the arbiter + responder. Returns ([], None) when any
# precondition fails -- handler treats that identically to today's
# present_no_match path.
def _compute_near_miss(
    *,
    match_count: int,
    band_signal: str,
    in_memory_matches: list,
    staged: StagedProfile,
    truth: Any,
) -> tuple[list, dict[str, Any] | None]:
    """Decide whether this turn qualifies as a near-miss, and if so
    build the responder payload.

    Preconditions (all must hold; locked in design):
      1. match_count == 0                   -- no presented matches
      2. band_signal == "low_only"          -- engine found low candidates
      3. truth.target_role_specificity == "specific"
      4. baseline evidence: resume parsed OR >= 3 chat-source skills

    When any precondition fails, returns ([], None). The handler
    passes the empty list to `resolve_match_outcome`, which falls
    through to the existing `present_no_match` branch -- byte-identical
    behavior to pre-Slice-N for any turn that doesn't qualify.

    When all preconditions hold:
      - Subset in_memory_matches to (eligible, band == "low")
      - Run filter_near_miss_candidates with target_role + target_noc
      - If non-empty, call build_near_miss_payload(...) and return both

    The registry is loaded via the same `get_registry()` cache the
    other handler helpers use. A registry-load failure logs WARNING
    and degrades to ([], None) -- same fail-safe pattern as
    _registry_training_for_gap.
    """
    # Slice N-6 live-debug telemetry: at each early-return point we log
    # which precondition failed so the operator can read "band=low_only
    # near_miss=0" in the live log and immediately know which of the
    # four gates dropped it. Cheap (string formatting); removed only
    # after the v2.2 rollout is fully stable.
    if match_count != 0 or band_signal != "low_only":
        return [], None  # not even a near-miss-shaped turn; silent
    if truth.target_role_specificity != "specific":
        log.info(
            "near_miss skipped: target_role_specificity=%s (need 'specific')",
            truth.target_role_specificity,
        )
        return [], None
    # Baseline evidence check (resume parsed OR enough chat skills)
    has_resume = truth.resume_parse_quality in {"full", "partial"}
    chat_skill_count = sum(1 for s in staged.skills if s.source != "resume")
    has_baseline = has_resume or chat_skill_count >= 3
    if not has_baseline:
        log.info(
            "near_miss skipped: no baseline evidence (resume=%s, chat_skills=%d)",
            truth.resume_parse_quality, chat_skill_count,
        )
        return [], None

    from skillbridge.training.registry import get_registry
    try:
        registry = get_registry()
    except Exception as e:
        log.warning("near_miss registry load failed: %s", e)
        return [], None

    low_matches = [
        m for m in in_memory_matches
        if m.match_eligible and m.match_band == "low"
    ]
    if not low_matches:
        log.info(
            "near_miss skipped: no eligible low-band matches (band_signal=low_only "
            "but engine produced %d total candidates, 0 in eligible-low subset)",
            len(in_memory_matches),
        )
        return [], None

    candidates = filter_near_miss_candidates(
        low_matches,
        target_role_text=staged.target_role_text,
        target_noc=staged.target_noc,
    )
    if not candidates:
        # Capture WHY the filter rejected every low-band match -- log
        # the per-candidate title-similarity / NOC values so we can
        # diagnose live test failures without re-running the engine.
        sample = []
        for m in low_matches[:5]:
            expl = m.score_explanation or {}
            sample.append({
                "title": (m.title or "")[:40],
                "override": bool(expl.get("title_match_override")),
                "similarity": expl.get("title_match_similarity"),
                "noc_code": m.noc_code,
            })
        log.info(
            "near_miss filter returned 0 of %d low-band candidates; "
            "target_role=%r target_noc=%r; sample=%s",
            len(low_matches), staged.target_role_text, staged.target_noc,
            sample,
        )
        return [], None

    payload = build_near_miss_payload(candidates, registry)
    return candidates, payload


# =========================================================================
# Slice (2026-06-08) -- target_role anaphor resolution
# =========================================================================
# When the chat asks for a target role and the user replies with a
# pronominal phrase ("same role", "current job", "this"), the LLM
# extractor often misses the slot (no concrete role tokens to extract).
# The pre-existing fallback then stored the LITERAL phrase as
# target_role_text, which corrupted downstream title-match scoring --
# the live test of 2026-06-05 had `target_role_text = "same role"`
# with `title_match_similarity = 0.171` against the truck-tech job.
#
# These two helpers detect anaphors and resolve them to the resume's
# current job title (or most recent if none is marked current). The
# handler's fallback code branches to them BEFORE the literal-fill
# path; unresolvable anaphors leave target_role_text empty so the
# planner re-asks rather than storing a known-bad value.
_ANAPHORIC_TARGET_PATTERNS = (
    # "same", "same role", "the same role", "same kind of work"
    re.compile(
        r"^(the\s+)?same(\s+(role|job|field|kind|thing|one|work|position))?"
        r"(\s+of\s+\w+)?$",
        re.I,
    ),
    # "current role", "previous job", "prior position", "past field"
    re.compile(
        r"^(current|previous|prior|past)\s+(role|job|position|field|work|kind)$",
        re.I,
    ),
    # "this", "this one", "this role", "that", "that one", "it"
    re.compile(r"^(this|that|it)(\s+(one|role|job|position))?$", re.I),
)


def _is_target_role_anaphor(message: str) -> bool:
    """True when the message is a pronominal/anaphoric reference to a
    role. Conservative -- matches only the canonical phrases. Affirmatives
    like 'yes' are deliberately NOT included because they're ambiguous
    in many other contexts."""
    stripped = (message or "").strip()
    if not stripped:
        return False
    return any(p.match(stripped) for p in _ANAPHORIC_TARGET_PATTERNS)


# Post-live-test fix (2026-06-22): hygiene check for the
# target_role_text fallback-fill path. When the user replies to
# "what kind of work?" with a different question rather than a role
# answer, the fallback-fill binding poisons downstream state. This
# helper detects question-shaped messages so fallback_fill can skip.
_TARGET_QUESTION_STARTERS: tuple[str, ...] = (
    "what ", "what's ", "whats ",
    "how ", "how's ", "hows ",
    "why ", "why's ", "whys ",
    "where ", "where's ", "wheres ",
    "when ", "when's ", "whens ",
    "who ", "who's ", "whos ",
    "which ",
    "can ", "can't ", "cant ",
    "could ", "couldn't ", "couldnt ",
    "would ", "wouldn't ", "wouldnt ",
    "should ", "shouldn't ", "shouldnt ",
    "do ", "does ", "did ",
    "is ", "are ", "am ",
    "compare ",
    "tell me ", "show me ",
    "help me ",
    "suggest ",
)


def _is_target_role_question_shaped(message: str) -> bool:
    """True when the message looks like a question or recommender intent
    rather than a plausible target-role-name answer.

    Target role names are noun phrases ("accounting clerk", "nurse",
    "welder"). They don't end with "?" and they don't start with
    question words / conversational openers.

    Three signals (any one suffices):
      1. Message ends with "?"
      2. Message starts with a question word / conversational opener
         from _TARGET_QUESTION_STARTERS
      3. truth_summary._classify_intent returns asking_about_gap or
         asking_question (existing pattern classifier catches shapes
         the heuristics miss)
    """
    if not isinstance(message, str):
        return False
    stripped = message.strip()
    if not stripped:
        return False
    if stripped.endswith("?"):
        return True
    low = stripped.lower()
    for starter in _TARGET_QUESTION_STARTERS:
        if low.startswith(starter):
            return True
    try:
        from skillbridge.chat.truth_summary import _classify_intent
        intent = _classify_intent(stripped)
    except Exception:  # noqa: BLE001
        return False
    return intent in ("asking_about_gap", "asking_question")


def _resolve_target_role_anaphor(
    message: str, staged: StagedProfile,
) -> str | None:
    """Resolve an anaphoric target-role reply to the resume's current
    job title. Returns None when no resume work_history exists or no
    entry has a usable title.

    Source-of-truth order:
      1. work_history entry with is_current=True (any title)
      2. work_history entry with the highest start_year
      3. first work_history entry with a non-empty title

    Uses `_effective_facts_view(staged)` (NOT the raw
    `staged.resume_facts_json`) so suppressed entries are excluded.
    If the user has flagged the current job entry as "remove that"
    or similar, the suppression view filters it out, and this
    resolver falls through to the next entry -- matching what the
    responder/matcher already see.

    Returns the title string verbatim from the resume (the chat will
    later store it as target_role_text and the engine will compute
    title similarity against this string). No invention -- if the
    resume doesn't surface a title, return None.
    """
    facts = _effective_facts_view(staged) or {}
    work = facts.get("work_history") or []
    if not isinstance(work, list) or not work:
        return None

    # 1) is_current entries first
    current = next(
        (w for w in work
         if isinstance(w, dict) and w.get("is_current") and w.get("title")),
        None,
    )
    if current:
        return str(current["title"]).strip() or None

    # 2) sorted by start_year (most recent wins)
    dated = [
        w for w in work
        if isinstance(w, dict) and w.get("title") and w.get("start_year")
    ]
    if dated:
        dated.sort(
            key=lambda w: (w.get("start_year") or 0),
            reverse=True,
        )
        title = str(dated[0]["title"]).strip()
        if title:
            return title

    # 3) any entry with a title at all
    for w in work:
        if isinstance(w, dict) and w.get("title"):
            title = str(w["title"]).strip()
            if title:
                return title

    return None


# =========================================================================
# Training registry integration (gated behind TRAINING_REGISTRY_ENABLED)
# =========================================================================
# Called on `explain_gap` turns AND on `present_near_miss` turns (Slice N)
# so the lead credential gap has its provider available in TRAINING for
# the responder to name verbatim. Reads `staged.last_presented_credential_gaps`
# (populated in Slice 8 after a present_matches turn) and/or the explicit
# `discovered_gaps` argument, looks each gap up in the registry, and
# builds the TRAINING block payload the responder consumes.
#
# Runtime safety: pending/stale URLs are suppressed by
# `Resource.surface_url(today)`. The Resource object still comes through
# so the responder can narrate the provider + SCCC referral. No URL
# the chat could cite is ever fabricated; if the registry has nothing
# verified for this gap yet, the user sees a referral, not silence.
#
# Telemetry: the registry's `surface_resources` method logs
# `unknown_gap=...` at INFO when a gap is queried but missing from the
# registry. That log is the prioritized backlog for what to curate next.
def _registry_training_for_gap(
    staged: StagedProfile,
    *,
    discovered_gaps: list[str] | None = None,
    include_carry_forward: bool = True,
) -> dict[str, list[dict]]:
    """Build TRAINING-block dicts keyed by synthetic gap-name keys.

    Source priority:
      1. `discovered_gaps` -- canonical gap names already extracted
         from the user's current message by the caller (handler did
         this once when building truth_summary; we don't re-scan).
      2. Gaps surfaced in the previous match turn (Slice 8 carry-forward
         from `staged.last_presented_credential_gaps`). SKIPPED when
         `include_carry_forward=False` (R-4 round-2 fix): on
         explain_remaining_gaps turns the caller wants TRAINING for
         ONLY the lead remaining credential. Including the carry-forward
         would leak completed credentials (e.g., 310S after the user
         claimed it) back into the resources surfaced to the responder.
      3. If both sources empty, return {}.

    Deduped across sources; order preserved (message-discovered first).

    Returns dict[gap_name -> list of resource dicts] in the shape the
    responder's TRAINING block expects. Keys are `gap:<canonical_name>`.
    """
    from datetime import date as _date

    # Local import to avoid pulling yaml + registry boot on every chat
    # turn for flag-off sessions.
    from skillbridge.training.registry import get_registry

    try:
        registry = get_registry()
    except Exception as e:
        log.warning("training_registry load failed: %s", e)
        return {}

    # ---- Collect gap names, message-discovered first, deduped ----
    gaps_to_query: list[str] = []
    seen: set[str] = set()

    for name in (discovered_gaps or []):
        if name and name not in seen:
            seen.add(name)
            gaps_to_query.append(name)

    if include_carry_forward:
        for name in (staged.last_presented_credential_gaps or []):
            if name not in seen:
                seen.add(name)
                gaps_to_query.append(name)

    if not gaps_to_query:
        return {}

    today = _date.today()
    out: dict[str, list[dict]] = {}
    for gap_query in gaps_to_query[:3]:   # cap for prompt token budget
        resources = registry.surface_resources(gap_query, today=today, limit=3)
        if not resources:
            continue
        entries: list[dict] = []
        for r in resources:
            entries.append({
                "provider": r.provider,
                "title": f"{r.provider} — {gap_query}",
                "url": r.surface_url(today),    # pending/stale -> None
                "type": r.type,
                "for_gap": gap_query,
                "summary": r.summary,
                "verified": not r.is_pending and r.is_fresh(today),
            })
        out[f"gap:{gap_query}"] = entries
    return out


def _collect_caps_applied(results: list[dict]) -> list[str]:
    """Union of caps_applied across the top match results, preserving
    first-seen order. Pass 2 receives this as a tuple so the responder
    can name each cap in plain language."""
    seen: set[str] = set()
    out: list[str] = []
    for r in results[:5]:
        se = r.get("score_explanation") or {}
        for cap in (se.get("caps_applied") or []):
            if cap not in seen:
                seen.add(cap)
                out.append(cap)
    return out


def _build_v2_response(
    *,
    staged: StagedProfile,
    new_session_id: str,
    reply: str,
    final_move: str,
    ask_slot: str | None,
    resume_info: dict[str, Any] | None,
    results: list[dict],
    training_by_job: dict[str, list[dict]],
    next_skill: tuple[str | None, int],
) -> dict[str, Any]:
    """Shape the v2 response to match the existing handle_anonymous
    response dict so the frontend / route layer doesn't need to know
    which orchestrator handled the turn. Adds one new field
    (`final_move`) for analytics/transcript test consumers.

    AR-9.feat.coach-tiers CP2 step 4 (signed-off pin): the new
    `present_tiered_matches` move is ChatGPT-style prose-only. The
    locked design forbids cards. We therefore force
    `recommended_jobs=[]` and suppress the legacy next-skill
    suggestion fields for that move — the tier prose is the
    complete response surface.
    """
    is_tiered = final_move == "present_tiered_matches"
    if is_tiered:
        recommended_jobs: list[dict[str, Any]] = []
        next_skill_suggestion: str | None = None
        next_skill_jobs_unlocked = 0
    else:
        recommended_jobs = [
            {**r, "suggested_resources": training_by_job.get(r["job_id"], [])}
            for r in results
        ]
        next_skill_suggestion = (
            next_skill[0] if (results and next_skill[1]) else None
        )
        next_skill_jobs_unlocked = (
            next_skill[1] if (results and next_skill[1]) else 0
        )
    return {
        "reply": reply,
        "profile_id": None,
        "session_id": new_session_id,
        "intake_state": staged.intake_state,
        "asked_slots": [ask_slot] if ask_slot else [],
        "next_action": _final_move_to_legacy_action(final_move),
        "recommended_jobs": recommended_jobs,
        "next_skill_suggestion": next_skill_suggestion,
        "next_skill_jobs_unlocked": next_skill_jobs_unlocked,
        "resume_info": resume_info,
        "requires_consent": True,
        # v2-only addition. Backwards-compatible: v1 responses don't set it.
        "final_move": final_move,
    }


_FINAL_MOVE_TO_LEGACY_ACTION: dict[str, str] = {
    "ask_one_clarifying_question": intake_state.ACTION_ASK_QUESTIONS,
    "acknowledge_and_continue": intake_state.ACTION_ACKNOWLEDGE_AND_WAIT,
    "present_matches": intake_state.ACTION_PRESENT_MATCHES,
    "present_no_match": intake_state.ACTION_PRESENT_MATCHES,
    # AR-9.feat.coach-tiers CP2 step 2 (signed-off pin): the new
    # three-tier coach surface maps to the legacy PRESENT_MATCHES
    # action so the session-snapshot lifecycle and downstream
    # analytics consumers see it as a present-matches continuation.
    "present_tiered_matches": intake_state.ACTION_PRESENT_MATCHES,
    "redirect_scope": intake_state.ACTION_REDIRECT,
    "explain_gap": intake_state.ACTION_PRESENT_MATCHES,
    "offer_refinement": intake_state.ACTION_PRESENT_MATCHES,
    # Slice 7 review fix: gate-emitted outcome maps to the legacy v1
    # PRESENT_RESUME_FACTS action so downstream consumers (analytics,
    # legacy frontend code) see a familiar value.
    "confirm_resume_summary": intake_state.ACTION_PRESENT_RESUME_FACTS,
    # R-3 (remaining-gaps iteration): the new outcome maps to the same
    # legacy action as `present_matches` / `explain_gap` -- it's a
    # match-conversation continuation, not a topic-change ask. Without
    # this mapping, analytics / legacy frontend consumers reading
    # next_action would see ASK_QUESTIONS (the defensive default), which
    # misclassifies the conversation flow.
    "explain_remaining_gaps": intake_state.ACTION_PRESENT_MATCHES,
}


def _final_move_to_legacy_action(move: str) -> str:
    """Map an OutcomeMove to a v1 ACTION_* label for backward-compatibility
    with any client / analytics consumer reading `next_action`. Unknown
    moves fall back to ASK_QUESTIONS rather than crashing -- defensive."""
    return _FINAL_MOVE_TO_LEGACY_ACTION.get(move, intake_state.ACTION_ASK_QUESTIONS)


# =========================================================================
# Slice A2-α3 (2026-06-18): bare-yes ambiguity guard
# =========================================================================
# When the user replies with a bare yes/no but TWO OR MORE formal
# pending flags are simultaneously awaiting that response, the
# existing per-flag clearing code at the top of handle_anonymous
# would consume the first matching flag's `if` branch -- regardless
# of which question the user is actually answering. That's a hidden
# routing decision. A2-α3 short-circuits these turns with a soft
# re-ask BEFORE any pending-flag mutation, so the user's clarifying
# next message routes deterministically through existing per-flag
# handlers.
#
# Scope (locked with Nazmul 2026-06-18):
#   - Intercepts ONLY when entry pending_count >= 2 AND message is
#     bare yes/no/confirming-style. The 0-pending case is NOT
#     intercepted; existing routing handles natural "alright" /
#     "yes" replies to coach prose where there's an implicit
#     conversational frame.
#   - Skipped when uploaded_file is True: a file upload is a strong
#     user action that supersedes message-level ambiguity.
#   - Does NOT mutate staged. Pending flags stay set so the next
#     turn's existing routing can resolve them. `staged.touch()` +
#     `store.save(staged)` ARE called so message_count and timestamps
#     stay consistent.
_BARE_YES_NO_INTENTS: frozenset[str] = frozenset({
    "confirming", "declining", "impatient_proceed",
})
_BARE_MESSAGE_MAX_WORDS = 4  # "yes", "no", "alright", "yes please",
                              # "go ahead", "sure thing", "not now"


def _is_bare_yes_no_response(message: str, intent: str) -> bool:
    """True when the message is a short yes/no-style response. The
    intent guard rejects long messages with substantive content even
    if they happen to start with "yes"."""
    if intent not in _BARE_YES_NO_INTENTS:
        return False
    return len((message or "").strip().split()) <= _BARE_MESSAGE_MAX_WORDS


def _count_entry_pending_flags(staged: StagedProfile) -> int:
    """Counts the formal pending fields on staged. Called at
    handle_anonymous entry, BEFORE any clearing code runs. Same
    counting logic as turn_state._collect_pending_flags (Slice B);
    duplicated here to keep A2-α3 independent of Slice B per the
    locked decision not to thread DerivedTurnState through the
    handler lifecycle yet.

    Slice 5 step 2 (2026-06-18): pending_recommender_offer joins the
    count so a bare yes with this flag + ANY other pending flag
    triggers A2-α3's ambiguity guard. When this is the SOLE pending
    flag (count == 1), bare yes routes normally through the existing
    per-flag handler -- meaning Step 4 wiring becomes unambiguous."""
    count = 0
    if staged.pending_credential_confirmation is not None:
        count += 1
    if staged.pending_adjacent_offer:
        count += 1
    if staged.pending_training_topic:
        count += 1
    if staged.pending_adjacent_search_offer:
        count += 1
    if staged.pending_recommender_offer is not None:
        count += 1
    return count


def _ambiguous_yes_response(
    session_id: str, staged: StagedProfile,
) -> dict[str, Any]:
    """Build the A2-α3 short-circuit response. Preserves current
    intake_state and pending flags so the next turn's existing
    routing resolves cleanly. The reply phrasing is a soft re-ask
    that does NOT enumerate internal flag names (coach voice,
    no leaking of internal state)."""
    return {
        "reply": (
            "I want to make sure I'm answering the right question — "
            "can you say a bit more about what you'd like to do next?"
        ),
        "profile_id": None,
        "session_id": session_id,
        "intake_state": staged.intake_state,
        "asked_slots": [],
        "next_action": intake_state.ACTION_ASK_QUESTIONS,
        "recommended_jobs": [],
        "next_skill_suggestion": None,
        "requires_consent": True,
    }


# =========================================================================
# ANONYMOUS PATH — no DB writes for user data
# =========================================================================
def handle_anonymous(
    message: str,
    session_id: str | None,
    *,
    file_bytes: bytes | None = None,
    filename: str | None = None,
) -> dict[str, Any]:
    """Pre-consent chat. State machine drives ask/match cycle.

    file_bytes / filename: when the user uploaded a resume via the multipart
    route, the raw bytes are passed here. The handler parses + extracts +
    derives into staged before continuing with the normal chat flow.
    When no file is uploaded, behavior is identical to PR 10.

    INVARIANT: this function MUST NOT write to profile.*, interaction.*,
    or analytics.*. The consent-boundary test enforces that invariant.
    """
    # If the user uploaded a file but didn't say anything, treat as an
    # implicit "here's my resume" — we still need to load/create the
    # session so we can stash the parsed data.
    uploaded_file = file_bytes is not None
    if not uploaded_file and (not message or not message.strip()):
        return _empty_response(session_id, "Tell me a bit about the kind of work you're looking for and I'll find local matches.")

    store = get_store()
    staged = store.load(session_id) if session_id else None
    if staged is None:
        sid = store.new_session() if not session_id else session_id
        loaded = store.load(sid)
        staged = loaded if loaded is not None else StagedProfile.new(sid)

    # Slice A2-α3 (2026-06-18): bare-yes ambiguity guard. Runs BEFORE
    # any pending-flag clearing. Skipped on resume uploads (the file
    # action supersedes message-level ambiguity). See the helper-block
    # comment above for the full rationale + locked scope.
    if not uploaded_file:
        entry_pending_count = _count_entry_pending_flags(staged)
        if entry_pending_count >= 2:
            from skillbridge.chat.truth_summary import _classify_intent
            entry_intent = _classify_intent(message or "")
            if _is_bare_yes_no_response(message or "", entry_intent):
                log.info(
                    "anon_chat session=%s a2_intercept=ambiguous_yes "
                    "pending_count=%d intent=%s",
                    staged.session_id[:8], entry_pending_count, entry_intent,
                )
                staged.touch()
                new_session_id = store.save(staged)
                return _ambiguous_yes_response(new_session_id, staged)

    # AR-1: soft-offer save-and-clear. Runs IMMEDIATELY after session load
    # and BEFORE every downstream short-circuit (resume-upload review,
    # canned gates, _try_v2_path). Mirrors the pending_credential_confirmation
    # save-and-clear precedent: even if a downstream branch returns early,
    # the flag is consumed exactly once per "meaningful" turn -- a non-blank
    # message OR a resume upload, because uploaded_file=True bypasses the
    # blank short-circuit above. Pure blank input returned BEFORE session
    # load, so it never consumes the flag.
    #
    # AR-1 is INERT: the SETTER lives in AR-6's handler soft-offer wiring;
    # until then pending_adjacent_offer is provably never True in
    # production, and this hook is a no-op at every code path. See
    # docs/adjacent-recommendations-design.md §"Activation deferral".
    pending_adjacent_offer = bool(staged.pending_adjacent_offer)
    if pending_adjacent_offer:
        staged.pending_adjacent_offer = False

    # Pattern 2 consume hook (closing-matrix v2, Step 7b, 2026-06-17):
    # the prior turn rendered a Pattern 2 closing ("want me to also
    # look at related roles?") and set `pending_adjacent_search_offer`.
    # This turn's user message is the reply. Classify it as
    # yes / no / other; log the consent decision; clear the flag.
    # Routing-to-CP5 on a "yes" consent is Step 8 (= Sideways
    # infrastructure reuse) — Step 7b only decides consent.
    pattern_2_consent: str | None = None
    if staged.pending_adjacent_search_offer:
        pattern_2_consent = _classify_pattern_2_reply(message)
        staged.pending_adjacent_search_offer = False
        log.info(
            "anon_chat session=%s pattern_2_consent=%s",
            staged.session_id[:8], pattern_2_consent,
        )

    # Slice 5 step 4 (2026-06-19): conversational recommender consume
    # hook. Mirrors the Pattern 2 consume hook structure but routes to
    # the recommender dispatcher when `pending_recommender_offer` is set
    # AND the user's reply classifies as yes/no (on yes, the dispatcher
    # owns the turn end-to-end and returns a complete response dict).
    # See _dispatch_recommender_consume + project_recommender_step4_
    # implementation_lock memory.
    if not uploaded_file and staged.pending_recommender_offer is not None:
        recommender_reply = _dispatch_recommender_consume(
            staged=staged,
            user_message=message or "",
            store=store,
            resume_info=None,
        )
        if recommender_reply is not None:
            return recommender_reply

    # 0) If the user uploaded a resume, run the resume pipeline before
    # the regular chat extractor. The extracted facts feed staged.* slots,
    # so the rest of the handler "just sees" a richer profile.
    resume_info: dict[str, Any] | None = None
    if uploaded_file:
        resume_info = _apply_resume_upload(staged, file_bytes, filename)

    # 0b) Resume-review entry: if the upload produced something worth
    # confirming, short-circuit the rest of the handler. We narrate what
    # was parsed and let the user correct it on the next turn. Matching
    # and intake follow-ups wait until after the review.
    if (
        uploaded_file
        and resume_info
        and resume_info.get("parsed")
        and _resume_facts_have_content(staged.resume_facts_json)
    ):
        staged.intake_state = intake_state.STATE_RESUME_REVIEW
        staged.last_asked_slots = []
        decision = intake_state.Decision(
            next_state=intake_state.STATE_RESUME_REVIEW,
            action=intake_state.ACTION_PRESENT_RESUME_FACTS,
            ask_slots=[],
            show_matches=False,
        )
        staged.touch()

        reply = compose_reply(ResponderInput(
            user_message=message or "",
            decision=decision,
            results=[],
            training_by_job={},
            next_skill=(None, 0),
            band_signal="none",
            requires_consent=True,
            target_role_text=staged.target_role_text,
            resume_facts=_effective_facts_view(staged),
        ))

        new_session_id = store.save(staged)
        log.info(
            "anon_chat session=%s state=resume_review action=PRESENT_RESUME_FACTS "
            "skills=%d work=%d edu=%d",
            staged.session_id[:8],
            len(staged.resume_facts_json.get("skills") or []),
            len(staged.resume_facts_json.get("work_history") or []),
            len(staged.resume_facts_json.get("education") or []),
        )
        return {
            "reply": reply,
            "profile_id": None,
            "session_id": new_session_id,
            "intake_state": decision.next_state,
            "asked_slots": [],
            "next_action": decision.action,
            "recommended_jobs": [],
            "next_skill_suggestion": None,
            "resume_info": resume_info,
            "requires_consent": True,
        }

    # 0c) Resume-review exit: if the previous assistant turn was the
    # confirmation summary (state == RESUME_REVIEW) and the user has now
    # replied without uploading a new file, treat this turn as the
    # review reply. Detect any suppression requests, apply them to the
    # facts layer, then transition out of RESUME_REVIEW so the rest of
    # the handler runs the normal extract → state-machine → match path.
    # Any additions ("I also know Docker") are picked up by the regular
    # chat extractor in step 1.
    just_exited_resume_review = False
    if (
        not uploaded_file
        and staged.intake_state == intake_state.STATE_RESUME_REVIEW
    ):
        just_exited_resume_review = True
        suppressed = _detect_resume_suppressions(staged, message)
        if suppressed:
            staged.suppressed_fact_ids = sorted(
                set(staged.suppressed_fact_ids) | set(suppressed)
            )
            # Re-derive flat slots from the now-effective facts view
            # (facts minus suppressions) so matching reflects the user's
            # corrections immediately.
            derived = resume_derive_with_suppressions(
                staged.resume_facts_json, staged.suppressed_fact_ids,
            )
            _refresh_derived_into_staged(staged, derived)
            log.info(
                "anon_chat session=%s resume_review_suppressed=%s",
                staged.session_id[:8], suppressed,
            )
        # Whether or not the user suppressed anything, this turn closes
        # the review loop. Drop back into the normal intake_collecting
        # state so decide() picks the next move from completeness.
        staged.intake_state = intake_state.STATE_INTAKE_COLLECTING

    # =========================================================================
    # v2 EARLY-GATE SHORT-CIRCUIT (extractor-after-gates optimization)
    # =========================================================================
    # When CHAT_ORCHESTRATOR=v2 AND a CANNED-RESPONSE gate would fire
    # (empty_input or first_turn_greeting), skip the chat extractor and
    # planner entirely. These turns introduce no profile content to extract;
    # paying ~2-5s of Haiku for the extractor here was waste seen live in
    # acceptance testing.
    #
    # Tight scope:
    #   - Only when v2 is active (v1 path unchanged)
    #   - Only when no file was uploaded (uploads run their own pipeline
    #     in step 0/0b above; a non-canned resume_upload gate still
    #     wants the responder, which the normal v2 dispatch handles)
    #   - Only when NOT just exiting a resume review (that path needs the
    #     full extraction flow so corrections take effect)
    #   - Only when the gate has a `canned_response` (i.e. empty_input or
    #     first_turn_greeting). Content gates that need the responder
    #     fall through to the normal v2 dispatch below.
    #
    # The gate is then evaluated a second time inside `_try_v2_path` for any
    # non-short-circuited turn. That's pure-function work (~10us); the cost
    # is negligible vs the LLM call this branch skips. See Slice 7 review
    # findings + acceptance-testing log analysis for the motivating data.
    if (
        CHAT_ORCHESTRATOR == "v2"
        and not uploaded_file
        and not just_exited_resume_review
    ):
        early_gate = chat_gates.evaluate_gates(
            user_message=message or "",
            uploaded_file=False,
            message_count=staged.message_count,
        )
        if early_gate is not None and early_gate.canned_response is not None:
            # Gates don't ask for slots; clear last_asked_slots so the
            # slot-answer guard on a follow-up turn sees correct state.
            staged.last_asked_slots = []
            staged.touch()
            new_session_id = store.save(staged)
            log.info(
                "anon_chat_v2 session=%s gate=%s canned early_skip_extractor",
                staged.session_id[:8], early_gate.gate_name,
            )
            return _build_v2_response(
                staged=staged, new_session_id=new_session_id,
                reply=early_gate.canned_response,
                final_move=early_gate.final_move,
                ask_slot=early_gate.ask_slot,
                resume_info=resume_info,
                results=[], training_by_job={}, next_skill=(None, 0),
            )

    # CP3 step 3 (2026-06-15) — pending_training_topic capture.
    #
    # When the previous turn emitted Rule 3 ("what skill or certificate
    # do you want training for?"), `staged.pending_training_topic` is
    # True. We consume it here, BEFORE normal extraction merges, so
    # a bare topic answer ("Excel") never silently becomes a profile
    # skill claim.
    #
    # Lifecycle (verified design):
    #   * Blank input → flag survives untouched; the responder will
    #     re-ask via the normal intake path.
    #   * Meaningful turn with exactly one registry entity → consume
    #     flag, suppress extraction, synthesize explain_gap with the
    #     captured entity, render training options, return.
    #   * Meaningful turn with 2+ registry entities → keep flag set
    #     (one redirect for disambiguation), ask which one.
    #   * Meaningful turn with no registry entity → consume flag,
    #     fall through to normal routing (user has changed topic).
    if (
        bool(staged.pending_training_topic)
        and CHAT_ORCHESTRATOR == "v2"
        and not uploaded_file
        and message
        and message.strip()
    ):
        topic_entities: list[str] = []
        if TRAINING_REGISTRY_ENABLED:
            try:
                from skillbridge.training.registry import (
                    get_registry as _get_training_registry,
                )
                _reg = _get_training_registry()
                topic_entities = [
                    g.canonical_name
                    for g in _reg.find_gaps_in_message(message)
                ]
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "pending_training_topic registry scan failed: %s", exc,
                )
                topic_entities = []
        if len(topic_entities) == 1:
            # Single entity: consume flag, suppress extraction, route
            # to explain_gap with the captured topic.
            staged.pending_training_topic = False
            training_by_job = _registry_training_for_gap(
                staged, discovered_gaps=topic_entities,
                include_carry_forward=False,
            )
            synth_decision = ArbiterDecision(
                final_move="explain_gap",
                reason_code="credential_gap_present",
                tone="warm_supportive",
                arbiter_action="handler_synthesized_training_topic_capture",
            )
            from skillbridge.chat.responder import (
                ResponderV2Input,
                compose_response_v2,
            )
            reply = compose_response_v2(ResponderV2Input(
                user_message=message,
                decision=synth_decision,
                results=[],
                training_by_job=training_by_job,
                next_skill=(None, 0),
                band_signal="none",
                requires_consent=True,
                target_role_text=staged.target_role_text,
                resume_facts=_effective_facts_view(staged),
                conversation_context=_build_conversation_context(staged),
            ))
            staged.last_asked_slots = []
            staged.touch()
            new_session_id = store.save(staged)
            log.info(
                "anon_chat_v2 session=%s training_topic_captured=%s",
                staged.session_id[:8], topic_entities,
            )
            return _build_v2_response(
                staged=staged, new_session_id=new_session_id, reply=reply,
                final_move="explain_gap", ask_slot=None,
                resume_info=resume_info, results=[],
                training_by_job=training_by_job, next_skill=(None, 0),
            )
        if len(topic_entities) >= 2:
            # Multi-entity ambiguity: keep the flag SET (the user's
            # next single-entity reply consumes cleanly) and ask which.
            shown = topic_entities[:3]
            joined = ", ".join(shown[:-1]) + f", or {shown[-1]}" if len(shown) > 1 else shown[0]
            reply = (
                f"I can point you to training for {joined} — "
                "which one would you like to look at first?"
            )
            staged.last_asked_slots = []
            staged.touch()
            new_session_id = store.save(staged)
            log.info(
                "anon_chat_v2 session=%s training_topic_multi=%s",
                staged.session_id[:8], topic_entities,
            )
            return _build_v2_response(
                staged=staged, new_session_id=new_session_id, reply=reply,
                final_move="ask_one_clarifying_question", ask_slot=None,
                resume_info=resume_info, results=[],
                training_by_job={}, next_skill=(None, 0),
            )
        # No registry entity in a meaningful reply: consume the flag
        # and fall through to normal routing. The user may have
        # changed topics; normal extraction + planner handles it.
        staged.pending_training_topic = False

    # 1) Extract with evidence binding (skipping previously-declined slots).
    # Pass only the slots we asked on the PREVIOUS assistant turn so a
    # "skip that" reply doesn't decline every slot we've ever asked about.
    # When the user uploaded a file with no text message, skip chat
    # extraction entirely — the resume pipeline already populated the
    # staged profile and the off-topic detection would otherwise REDIRECT
    # us into "what kind of work are you looking for?" needlessly.
    asked_last_turn = list(staged.last_asked_slots)
    file_only_turn = uploaded_file and (not message or not message.strip())
    if file_only_turn:
        extraction = chat_extractor.ExtractionResult(
            fields={}, skills=[], declined=[], off_topic=False,
            raw_keys_dropped=[],
        )
    else:
        extraction = _extract(message, asked_slots=asked_last_turn)

    if extraction.raw_keys_dropped:
        log.info("anon_chat session=%s extractor_dropped=%s",
                 staged.session_id[:8], extraction.raw_keys_dropped)

    # Slice (2026-06-08) review-found gap: the anaphor resolver added to
    # the fallback_fill path only fires when the LLM extractor MISSED
    # the slot. If the extractor itself returned target_role_text="same
    # role" (it shouldn't per evidence-bound rules but can in edge cases),
    # the value was stored verbatim by merge_fields below, bypassing
    # the resolver entirely.
    #
    # This pre-merge normalization closes that bypass: if the extractor
    # populated target_role_text with an anaphor, we either resolve it
    # against the resume (matching the fallback_fill behavior) or remove
    # it from extraction.fields so the fallback_fill path handles it next.
    if extraction.fields and isinstance(extraction.fields, dict):
        extracted_target = extraction.fields.get("target_role_text")
        if (
            isinstance(extracted_target, str)
            and _is_target_role_anaphor(extracted_target)
        ):
            resolved = _resolve_target_role_anaphor(extracted_target, staged)
            if resolved:
                extraction.fields["target_role_text"] = resolved
                log.info(
                    "anon_chat session=%s anaphor_resolved=%r -> %r "
                    "(pre-merge normalization)",
                    staged.session_id[:8], extracted_target, resolved,
                )
            else:
                # Anaphor with no resume to anchor against: drop from
                # the extraction so the literal isn't merged. The
                # fallback_fill path below will see the slot as still
                # unfilled and apply its own (no-op for unresolvable
                # anaphor) handling.
                log.info(
                    "anon_chat session=%s anaphor_dropped_from_extraction=%r "
                    "(no resume work_history)",
                    staged.session_id[:8], extracted_target,
                )
                del extraction.fields["target_role_text"]

    # 2) Merge into staged blob.
    if extraction.fields:
        staged.merge_fields(extraction.fields)
    if extraction.skills:
        staged.merge_skills([
            StagedSkill(
                skill_name=s.skill_name,
                skill_id=s.skill_id,
                raw_phrase=s.raw_phrase,
                confidence=s.confidence,
                importance_rank=s.importance_rank,
            ) for s in extraction.skills
        ])
    for slot in extraction.declined:
        staged.mark_declined(slot)

    # 2.5) skills_text recovery (2026-06-17): when the LLM extractor
    # signalled "user was listing skills" (raw_keys_dropped contains
    # 'ungrounded:skills_text') but its slot-level evidence wasn't a
    # verbatim substring, the slot stays empty even though per-skill
    # grounding produced >=3 real skills. Change C's skills_text_present
    # guard then reads False and the engine refuses to run — user is
    # re-asked despite having just listed real skills. See helper
    # docstring for the gating rules and phantom-skill protection.
    if _maybe_recover_skills_text_slot(
        staged=staged, extraction=extraction, message=message,
    ):
        log.info(
            "anon_chat session=%s skills_text_recovered=%d_skills_grounded "
            "(slot-level grounding failed but per-skill grounding passed)",
            staged.session_id[:8], len(extraction.skills),
        )

    # 2a) Safety net: if we asked a single closed-vocabulary slot last
    # turn and the user answered with a short natural reply ("day",
    # "full time"), fill that slot deterministically.
    closed_vocab_filled = False
    if (
        len(asked_last_turn) == 1
        and asked_last_turn[0] not in extraction.fields
        and asked_last_turn[0] not in extraction.declined
        and asked_last_turn[0] not in staged.declined_slots
        and not getattr(staged, asked_last_turn[0], None)
    ):
        slot = asked_last_turn[0]
        normalized = _closed_vocab_reply(slot, message)
        if normalized:
            setattr(staged, slot, normalized)
            closed_vocab_filled = True
            log.info(
                "anon_chat session=%s closed_vocab_fill=%s",
                staged.session_id[:8], slot,
            )

    # 2b) Safety net: if we asked a single open-text slot last turn and the
    # evidence-bound extractor didn't fill it, but the user replied with
    # non-trivial content (not a decline), accept the reply itself as the
    # slot value. Bounded to open-text slots so we don't shove raw text
    # into closed-vocab fields like work_type_preference.
    _ANSWER_AS_VALUE_SLOTS = {
        "experience_text", "skills_text", "education_text",
        "target_role_text", "transportation_text", "availability_text",
        "salary_expectation_text",
    }
    msg_stripped = message.strip()
    if (
        len(asked_last_turn) == 1
        and asked_last_turn[0] in _ANSWER_AS_VALUE_SLOTS
        and asked_last_turn[0] not in extraction.fields
        and asked_last_turn[0] not in extraction.declined
        and asked_last_turn[0] not in staged.declined_slots
        and not getattr(staged, asked_last_turn[0], None)
        and len(msg_stripped) >= 3
    ):
        slot = asked_last_turn[0]
        # Slice (2026-06-08): anaphor resolution for target_role_text.
        # The live test of 2026-06-05 showed Michael's "same role" was
        # being stored as the literal string, which then fed an
        # incorrect title_match check downstream. When the user replies
        # with a pronominal/anaphoric phrase, try to resolve to the
        # resume's current job title. If unresolvable (no work_history),
        # leave the slot empty so the planner re-asks rather than
        # storing a literal we know is wrong.
        if slot == "target_role_text" and _is_target_role_anaphor(msg_stripped):
            resolved = _resolve_target_role_anaphor(msg_stripped, staged)
            if resolved:
                setattr(staged, slot, resolved)
                log.info(
                    "anon_chat session=%s anaphor_resolved=%r -> %r",
                    staged.session_id[:8], msg_stripped, resolved,
                )
            else:
                # Anaphor with no resume to anchor against -- log and
                # leave the slot empty. The truth_summary classifier
                # will report target_role_specificity=none, and the
                # planner asks again on the next turn.
                log.info(
                    "anon_chat session=%s anaphor_unresolved=%r "
                    "(no resume work_history)",
                    staged.session_id[:8], msg_stripped,
                )
        elif (
            slot == "target_role_text"
            and _is_target_role_question_shaped(msg_stripped)
        ):
            # Post-live-test fix (2026-06-22): the user's reply is
            # question-shaped (e.g. "what should I improve?", "what
            # training should I take?"). It is NOT a target role
            # answer; binding it to target_role_text poisons downstream
            # state (the matcher tries to resolve a question as a NOC
            # title and the recommender classifier reads a nonsense
            # role context). Skip the fallback fill -- the planner will
            # re-ask, OR the recommender router downstream will
            # classify the intent and route appropriately.
            log.info(
                "anon_chat session=%s fallback_fill_skipped=target_role_text "
                "reason=question_shaped",
                staged.session_id[:8],
            )
        else:
            setattr(staged, slot, msg_stripped[:500])
            log.info(
                "anon_chat session=%s fallback_fill=%s "
                "(extractor missed grounded reply)",
                staged.session_id[:8], slot,
            )

    # =========================================================================
    # v2 DISPATCH (Slice 6) -- hard rollback switch.
    #
    # Order is deliberately visible and boring:
    #   gates -> planner -> arbiter pass 1 -> [maybe engine] -> arbiter pass 2
    #   -> responder v2
    # The match engine is INVOKED ONLY when arbiter pass 1 explicitly returns
    # RunEngine. Scope overrides, "proceed but truth says no" overrides, and
    # passthrough moves all skip the engine entirely.
    #
    # `_try_v2_path` returns either a complete response dict (v2 fully handled
    # the turn) or None (explicit fallback_to_legacy signal from arbiter pass 1,
    # caller should drop into the v1 path below). The fallback is documented
    # graceful degradation, not a "half v1 + half v2" code path.
    #
    # The touch() that v1 runs at this point is deferred: v2 calls touch()
    # itself only on its terminal paths so the first_turn_greeting gate can
    # observe message_count == 0 on a user's very first message. If v2 falls
    # back, the v1 touch() below runs normally.
    # =========================================================================
    # Step 3 peer-engine routing (2026-06-22, relocated post-live-test
    # 2026-06-22). Runs AFTER all input-processing steps that can
    # legitimately update staged state in response to this turn's
    # message:
    #   - resume upload + resume-review (0 / 0b / 0c)
    #   - chat extraction + slot merge (1+)
    #   - canned-gates / training-topic consume / pattern-2 consume hooks
    # The router must see the post-extraction view of staged so that a
    # user's reply naming a target ("accounting clerk") or providing
    # skills is observed by the router BEFORE the substrate gate runs.
    # Original (broken) placement was pre-extraction and produced a
    # substrate-ask loop visible in live verify on 2026-06-22.
    #
    # Skipped on resume uploads and blank messages -- the existing
    # matching engine intake handles those naturally.
    if not uploaded_file and message and message.strip():
        intent_reply = _maybe_route_recommender_from_intent(
            staged=staged,
            message=message,
            store=store,
        )
        if intent_reply is not None:
            return intent_reply

    if CHAT_ORCHESTRATOR == "v2":
        v2_response = _try_v2_path(
            staged=staged,
            message=message,
            uploaded_file=uploaded_file,
            resume_info=resume_info,
            store=store,
            pending_adjacent_offer=pending_adjacent_offer,
            pattern_2_consent=pattern_2_consent,
        )
        if v2_response is not None:
            return v2_response
        # Explicit fallback_to_legacy from arbiter pass 1: drop to v1 below.
        log.info(
            "anon_chat session=%s v2_dispatch=fallback_to_legacy",
            staged.session_id[:8],
        )

    staged.touch()

    extracted_anything = (
        bool(extraction.fields)
        or bool(extraction.skills)
        or closed_vocab_filled
        or bool(asked_last_turn and getattr(staged, asked_last_turn[0], None) and asked_last_turn[0] in _ANSWER_AS_VALUE_SLOTS)
        # A fresh resume that parsed successfully counts as "the user told
        # us something" — don't let the state machine treat it as off-topic.
        or (resume_info is not None and resume_info.get("parsed") is True)
        # The user just confirmed / corrected their parsed resume. Even
        # if their reply was a bare "yes" / "looks good" that the chat
        # extractor flagged as off-topic, the previous turn was the
        # RESUME_REVIEW summary — the user is responding to that, not
        # going off-topic. Without this, decide() emits REDIRECT and the
        # responder asks the user what kind of work they're looking for,
        # which is a UX dead end after a successful upload.
        or just_exited_resume_review
    )

    # 3) Decide next action. Force off_topic=False on a resume-review
    # exit turn for the same reason as above — a bare confirmation isn't
    # a topic change; it closes the loop opened by the upload.
    effective_off_topic = extraction.off_topic and not just_exited_resume_review
    decision = intake_state.decide(
        staged,
        off_topic=effective_off_topic,
        extracted_anything=extracted_anything,
        declined_this_turn=extraction.declined,
        authenticated=False,
    )
    staged.intake_state = decision.next_state
    # Overwrite last-turn record (per-turn state) while appending to the
    # cumulative history (used by intake_priority to deprioritise re-asks).
    staged.last_asked_slots = list(decision.ask_slots)
    for slot in decision.ask_slots:
        staged.mark_asked(slot)

    # 4) Maybe compute matches.
    results: list[dict] = []
    training_by_job: dict[str, list[dict]] = {}
    next_skill: tuple[str | None, int] = (None, 0)
    band_signal = "none"

    # Title-match override: even when the intake band is too low to
    # trigger the normal show_matches path, run the matcher as soon as
    # target_role_text is filled. The engine's title-match fast path can
    # surface postings whose title strongly matches what the user typed
    # (e.g. user typed an exact SCCC job title with no skills yet). If
    # any real (stretch+) match exists, override the state machine to
    # PRESENT_MATCHES so we don't say "no matches" for a posting the
    # user named verbatim.
    should_compute = decision.show_matches or bool(staged.target_role_text)
    if should_compute:
        in_memory_matches = match_engine.compute_matches_in_memory(staged, top=20)
        results, band_signal = _build_results_block(in_memory_matches)
        has_real_match = bool(results)  # _build_results_block already drops low/ineligible

        if not decision.show_matches and has_real_match:
            from dataclasses import replace
            decision = replace(
                decision,
                action=intake_state.ACTION_PRESENT_MATCHES,
                next_state=intake_state.STATE_INTAKE_READY_FOR_CONSENT,
                show_matches=True,
                ask_slots=[],
            )
            staged.intake_state = decision.next_state
            staged.last_asked_slots = []
            log.info(
                "anon_chat session=%s title_match_override -> PRESENT_MATCHES (%d hits)",
                staged.session_id[:8], len(results),
            )

        if decision.show_matches:
            training_by_job = _attach_training(results)
            next_skill = match_engine.next_skill_to_unlock_in_memory(staged)
        else:
            # Engine ran but nothing >= stretch; suppress the empty result
            # set so the responder stays on the intake question path.
            results = []
            band_signal = "none"

    # 5) Compose reply. target_role_text drives role-aware prompt hints so
    # the responder asks software devs about stack rather than shifts.
    # resume_facts (post-suppression view) is passed on every turn — the
    # responder uses it on PRESENT_MATCHES to quote real resume context
    # in "because" clauses, not invent them.
    reply = compose_reply(ResponderInput(
        user_message=message,
        decision=decision,
        results=results,
        training_by_job=training_by_job,
        next_skill=next_skill,
        band_signal=band_signal,
        requires_consent=True,
        target_role_text=staged.target_role_text,
        resume_facts=_effective_facts_view(staged),
    ))

    # 6) Persist (only) the staged blob back to session store.
    new_session_id = store.save(staged)

    log.info(
        "anon_chat session=%s state=%s action=%s skills=%d filled=%d results=%d band=%s",
        staged.session_id[:8], decision.next_state, decision.action,
        len(staged.skills), len(staged.filled_slots()),
        len(results), band_signal,
    )

    return {
        "reply": reply,
        "profile_id": None,
        "session_id": new_session_id,
        "intake_state": decision.next_state,
        "asked_slots": list(decision.ask_slots),
        "next_action": decision.action,
        "recommended_jobs": [
            {**r, "suggested_resources": training_by_job.get(r["job_id"], [])}
            for r in results
        ],
        # next_skill is match-context: only surface it when we're actually
        # presenting matches. Otherwise the UI could render "build X to
        # unlock N jobs" while the assistant text is still asking intake
        # questions.
        "next_skill_suggestion": (
            next_skill[0] if (decision.show_matches and next_skill[1]) else None
        ),
        "next_skill_jobs_unlocked": (
            next_skill[1] if (decision.show_matches and next_skill[1]) else 0
        ),
        # resume_info is non-null only on the turn the user uploaded a file.
        # Frontend surfaces parse_warning (e.g. "no_text" → "I couldn't read
        # the file, please paste your resume text"). Counts are for UI display.
        "resume_info": resume_info,
        "requires_consent": True,
    }


def _empty_response(session_id: str | None, reply: str) -> dict[str, Any]:
    return {
        "reply": reply,
        "profile_id": None,
        "session_id": session_id,
        "intake_state": intake_state.STATE_ANONYMOUS_CHAT,
        "asked_slots": [],
        "next_action": intake_state.ACTION_ASK_QUESTIONS,
        "recommended_jobs": [],
        "next_skill_suggestion": None,
        "requires_consent": True,
    }


# =========================================================================
# AUTHENTICATED PATH — post-consent, normal persistence
# =========================================================================
def handle_authenticated(message: str, profile_id: str) -> dict[str, Any]:
    """Post-consent chat. Persists to profile.user_skill, interaction.chat_event,
    and analytics.job_match. profile_id resolved from bearer token by the route.
    """
    _log_event(profile_id, "user", message)

    # Load a synthetic staged profile from the persisted row so we can use
    # the same state machine + responder shape. This is read-only.
    # NOTE: authenticated mode doesn't persist per-turn intake state
    # (last_asked_slots), so blanket-decline ("skip that") detection is
    # effectively disabled here — slot-specific decline patterns still work.
    # That's the deliberate v1 trade-off: a logged-in user briefly being
    # re-asked is fine; mass-declining is not.
    synthetic = _load_synthetic_staged(profile_id)

    extraction = _extract(message, asked_slots=list(synthetic.last_asked_slots))

    if extraction.fields:
        synthetic.merge_fields(extraction.fields)
        _update_profile_fields(profile_id, extraction.fields)
    if extraction.skills:
        synthetic.merge_skills([
            StagedSkill(
                skill_name=s.skill_name,
                skill_id=s.skill_id,
                raw_phrase=s.raw_phrase,
                confidence=s.confidence,
            ) for s in extraction.skills
        ])
        _upsert_user_skills(profile_id, extraction.skills)
    synthetic.touch()

    extracted_anything = bool(extraction.fields) or bool(extraction.skills)
    decision = intake_state.decide(
        synthetic,
        off_topic=extraction.off_topic,
        extracted_anything=extracted_anything,
        declined_this_turn=extraction.declined,
        authenticated=True,
    )

    # Recompute matches (persists to analytics.job_match every turn — keeps
    # the dashboard fresh).
    matches = match_engine.compute_matches(profile_id)
    results, band_signal = _build_results_block(matches)
    training_by_job = _attach_training(results)
    next_skill = match_engine.next_skill_to_unlock(profile_id)

    reply = compose_reply(ResponderInput(
        user_message=message,
        decision=decision,
        results=results,
        training_by_job=training_by_job,
        next_skill=next_skill,
        band_signal=band_signal,
        requires_consent=False,
        target_role_text=synthetic.target_role_text,
        resume_facts=_effective_facts_view(synthetic),
    ))

    _log_event(profile_id, "assistant", reply, metadata={
        "results_count": len(results),
        "band_signal": band_signal,
        "next_skill": next_skill[0],
        "intake_state": decision.next_state,
        "next_action": decision.action,
    })
    # Gate the response payload on decision.show_matches. We still recompute
    # matches above (keeps analytics.job_match fresh for the dashboard), but
    # we do not leak them to the chat client when the state machine hasn't
    # decided to present them yet — otherwise the UI would show
    # recommendations while the assistant text is still asking questions.
    payload_jobs = (
        [
            {**r, "suggested_resources": training_by_job.get(r["job_id"], [])}
            for r in results
        ]
        if decision.show_matches else []
    )
    return {
        "reply": reply,
        "profile_id": profile_id,
        "session_id": None,
        "intake_state": decision.next_state,
        "asked_slots": list(decision.ask_slots),
        "next_action": decision.action,
        "recommended_jobs": payload_jobs,
        "next_skill_suggestion": (
            next_skill[0] if (decision.show_matches and next_skill[1]) else None
        ),
        "next_skill_jobs_unlocked": (
            next_skill[1] if (decision.show_matches and next_skill[1]) else 0
        ),
        "requires_consent": False,
    }


# =========================================================================
# Authenticated-path DB helpers
# =========================================================================
def _load_synthetic_staged(profile_id: str) -> StagedProfile:
    """Hydrate a StagedProfile shape from profile.user_profile + user_skill.

    Used so the state machine can reason about the authenticated user the
    same way it reasons about anonymous users. Not persisted anywhere —
    just an in-memory view of the database row.

    Sprint 1: also loads the persisted resume_facts_json + resume_filename
    + resume_parsed_at so the responder gets RESUME_FACTS context on the
    authenticated path. Source provenance on profile.user_skill is mirrored
    onto StagedSkill.source so resume-derived skills stay tagged.
    """
    with sync_cursor() as cur:
        cur.execute(
            """
            SELECT preferred_location, target_role_text, education_text,
                   experience_text, skills_text, work_type_preference,
                   language_preferences,
                   salary_expectation_text, shift_preference,
                   transportation_text, availability_text,
                   resume_filename, resume_parsed_at, resume_facts_json
              FROM profile.user_profile
             WHERE profile_id = %s AND deleted_at IS NULL
            """,
            (profile_id,),
        )
        prow = cur.fetchone() or {}
        cur.execute(
            "SELECT skill_id, skill_name, raw_phrase, source, confidence "
            "FROM profile.user_skill WHERE profile_id = %s",
            (profile_id,),
        )
        skill_rows = list(cur.fetchall())

    synthetic = StagedProfile.new(profile_id)
    for k in (
        "preferred_location", "target_role_text", "education_text",
        "experience_text", "skills_text", "work_type_preference",
        "salary_expectation_text", "shift_preference",
        "transportation_text", "availability_text",
        "resume_filename", "resume_parsed_at",
    ):
        val = prow.get(k)
        if val:
            setattr(synthetic, k, val)
    if prow.get("language_preferences"):
        synthetic.language_preferences = list(prow["language_preferences"])

    # resume_facts_json is a JSONB column; psycopg returns it as a dict
    # already, but be defensive about strings (older rows / migrations).
    facts = prow.get("resume_facts_json")
    if isinstance(facts, dict):
        synthetic.resume_facts_json = facts
    elif isinstance(facts, str):
        try:
            synthetic.resume_facts_json = json.loads(facts)
        except (ValueError, TypeError):
            synthetic.resume_facts_json = None

    synthetic.skills = [
        StagedSkill(
            skill_name=r["skill_name"],
            skill_id=r.get("skill_id"),
            raw_phrase=r.get("raw_phrase"),
            confidence=float(r.get("confidence") or 0.7),
            source=r.get("source") or "chat",
        ) for r in skill_rows
    ]
    return synthetic


def _log_event(profile_id: str, role: str, text: str,
               metadata: dict | None = None) -> None:
    with sync_cursor() as cur:
        cur.execute(
            """
            INSERT INTO interaction.chat_event (profile_id, role, message_text, metadata)
            VALUES (%s, %s, %s, %s)
            """,
            (profile_id, role, text,
             json.dumps(metadata, default=str) if metadata else None),
        )


def _update_profile_fields(profile_id: str, fields: dict[str, Any]) -> None:
    allowed = {
        "preferred_location", "target_role_text", "education_text",
        "experience_text", "skills_text", "work_type_preference",
        "language_preferences",
        "salary_expectation_text", "shift_preference",
        "transportation_text", "availability_text",
    }
    cols: list[str] = []
    params: list[Any] = []
    for k, v in fields.items():
        if k not in allowed or v in (None, "", []):
            continue
        cols.append(f"{k} = %s")
        params.append(v)
    if not cols:
        return
    cols.append("updated_at = NOW()")
    params.append(profile_id)
    sql = f"UPDATE profile.user_profile SET {', '.join(cols)} WHERE profile_id = %s"
    with sync_cursor() as cur:
        cur.execute(sql, tuple(params))


def _upsert_user_skills(profile_id: str, skills: list[ExtractedSkill]) -> int:
    if not skills:
        return 0
    n = 0
    with sync_cursor() as cur:
        for s in skills:
            cur.execute(
                """
                INSERT INTO profile.user_skill (profile_id, skill_id, skill_name, raw_phrase, source, confidence)
                VALUES (%s, %s, %s, %s, 'chat', %s)
                ON CONFLICT (profile_id, skill_name) DO UPDATE SET
                    skill_id   = COALESCE(EXCLUDED.skill_id, profile.user_skill.skill_id),
                    raw_phrase = EXCLUDED.raw_phrase,
                    confidence = GREATEST(EXCLUDED.confidence, profile.user_skill.confidence)
                """,
                (profile_id, s.skill_id, s.skill_name, s.raw_phrase, s.confidence),
            )
            n += 1
    return n


# =========================================================================
# Consent grant — atomic flush of StagedProfile -> Postgres
# =========================================================================
def grant_consent(session_id: str, consent_purposes: list[str]) -> dict[str, Any]:
    """Atomically materialize a staged session into a consented profile.

    PR 10: persists the four newcomer-intake text fields alongside
    everything PR 1 already persisted.

    Sprint 1: also flushes resume_text, resume_filename, resume_parsed_at,
    and resume_facts_json when the user uploaded a resume during the
    anonymous session. In cookie-mode sessions the heavy fields will
    have been redacted from the cookie payload (see staging.to_json
    docstring) — what's left here is whatever the live request still
    holds in memory or what Redis retained for the session TTL.

    The raw pre-consent chat transcript is NOT backfilled — the user
    consented to a profile, not to a transcript.
    """
    from config import CONSENT_PURPOSES, CONSENT_VERSION, PROFILE_TOKEN_TTL_HOURS
    from skillbridge.auth import hash_token, mint_token, token_expires_at

    valid_purposes = [p for p in consent_purposes if p in CONSENT_PURPOSES]
    if "profile_storage" not in valid_purposes:
        raise ValueError("consent_purposes must include 'profile_storage' to save a profile")

    store = get_store()
    staged = store.load(session_id)
    if staged is None:
        raise ValueError("Staged session not found or expired")

    raw_token = mint_token()
    token_hash = hash_token(raw_token)
    expires_at = token_expires_at(PROFILE_TOKEN_TTL_HOURS)

    # Postgres JSONB column expects a JSON string. Serialize once here;
    # NULL when the user didn't upload anything.
    resume_facts_param = (
        json.dumps(staged.resume_facts_json)
        if staged.resume_facts_json
        else None
    )

    with sync_cursor() as cur:
        cur.execute(
            """
            INSERT INTO profile.user_profile
                (session_id, preferred_location, target_role_text, education_text,
                 experience_text, skills_text, work_type_preference, language_preferences,
                 salary_expectation_text, shift_preference,
                 transportation_text, availability_text,
                 resume_text, resume_filename, resume_parsed_at, resume_facts_json,
                 consent_version, consent_purposes, consent_granted_at,
                 session_token_hash, session_token_expires_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW(),%s,%s)
            RETURNING profile_id
            """,
            (
                staged.session_id, staged.preferred_location, staged.target_role_text,
                staged.education_text, staged.experience_text, staged.skills_text,
                staged.work_type_preference, staged.language_preferences,
                staged.salary_expectation_text, staged.shift_preference,
                staged.transportation_text, staged.availability_text,
                staged.resume_text, staged.resume_filename,
                staged.resume_parsed_at, resume_facts_param,
                CONSENT_VERSION, valid_purposes, token_hash, expires_at,
            ),
        )
        profile_id = str(cur.fetchone()["profile_id"])
        for s in staged.skills:
            # Honor each StagedSkill's source so resume-derived skills
            # persist as source='resume' (and chat-derived as 'chat').
            # profile.user_skill.source is VARCHAR(20); the values we
            # currently emit are 'chat', 'resume', 'form', 'manual_update'.
            source_value = s.source if s.source in ("chat", "resume", "form", "manual_update") else "chat"
            cur.execute(
                """
                INSERT INTO profile.user_skill
                    (profile_id, skill_id, skill_name, raw_phrase, source, confidence)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (profile_id, skill_name) DO NOTHING
                """,
                (profile_id, s.skill_id, s.skill_name, s.raw_phrase, source_value, s.confidence),
            )

    # Compute matches against the now-persisted profile so the first
    # authenticated request returns immediately with results.
    match_engine.compute_matches(profile_id)

    # Drop the staged session — never persisted again.
    store.delete(session_id)

    log.info("consent_granted profile=%s purposes=%s skills=%d filled=%d",
             profile_id[:8], ",".join(valid_purposes),
             len(staged.skills), len(staged.filled_slots()))

    return {
        "profile_id": profile_id,
        "session_token": raw_token,
        "expires_at": expires_at,
        "consent_purposes": valid_purposes,
        "consent_version": CONSENT_VERSION,
    }
