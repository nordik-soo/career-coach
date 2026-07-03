"""Staged profile model.

A StagedProfile lives ONLY in the session store. It is never written to
Postgres unless and until the user grants consent (POST /v1/consent).

Hard rule: every field on this object is staged data — by definition it has
not been consented to for persistence. Don't tee it into logs.

R-1 (remaining-gaps iteration) additions:
  - last_match_snapshot: per-job structured snapshot of the most-recent
    present_matches turn (lead job + credential / core-skill gaps).
    Drives the remaining-gaps reasoning layer on follow-up turns.
  - last_assumed_completed_credentials: typed records
    `{canonical, mode}` that the candidate has claimed or hypothetically
    assumed during the snapshot's lifetime. Conversation state, NOT
    profile evidence: never reaches the matching engine.
  - last_discussed_credential_canonical: recency anchor for anaphora
    ("it", "that licence").
  - pending_credential_confirmation: typed `{canonical, action}` set
    when the handler emits a confirmation question; consumed by the
    next turn's detection layer.
  All four share a lifecycle and are atomically cleared with the
  snapshot. See docs/remaining-gaps-design.md §R-1 for the contract.

PR 10 additions:
  - Four newcomer-intake text fields (salary expectation, shift preference,
    transportation, availability). These match the new columns on
    profile.user_profile and are flushed on consent.
  - intake_state: drives the chat state machine (see chat/intake_state.py).
  - asked_slots: which fields the assistant has already explicitly asked
    about. Used by intake_priority to avoid re-asking declined fields.
  - declined_slots: fields the user has explicitly declined to answer
    (or said "I don't know" / "prefer not to say"). Never re-asked.
  - last_extracted_at: bookkeeping for the extractor.

Sprint 1 (resume) additions:
  - resume_text / resume_filename / resume_parsed_at / resume_facts_json:
    set when the user uploads a resume. See docs/resume-design.md §3 for
    the storage policy. In cookie session mode, `resume_text` is dropped
    and `resume_facts_json` is replaced with the compact form (evidence
    strings stripped; fact_ids and structural data preserved) so it fits
    inside the ~4 KB signed-cookie limit while still supporting
    suppression and RESUME_REVIEW across turns. In Redis mode everything
    persists for the session TTL. Both flush to Postgres on consent
    grant.
  - suppressed_fact_ids: fact IDs the user has explicitly suppressed
    during the RESUME_REVIEW turn. The matcher consumes facts MINUS
    suppressions (see docs/resume-design.md §4).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class StagedSkill:
    skill_name: str
    skill_id: str | None = None
    raw_phrase: str | None = None
    confidence: float = 0.7
    importance_rank: int | None = None
    source: str = "chat"


# Caps for the remaining-gaps snapshot. Bound the cookie payload so the
# full signed StagedProfile stays under the 3800-byte ceiling that leaves
# margin for Set-Cookie header attributes against the browser's 4 KB
# per-cookie limit. See docs/remaining-gaps-design.md §1.
#
# R-1 NOTE: the design speculated MAX_CANONICAL_CHARS=80 and
# MAX_OTHER_JOBS=3, with the test as "the authoritative gate." Measuring
# the actual worst-case showed the budget overflowed at those caps once
# compact_facts() overhead was applied. R-1 tightens to the values
# below, which leave the worst-case-state signed cookie under 3800
# bytes. Realistic credential canonicals are 20-45 chars (Class G
# Driver's License = 24, 310S Automotive Technician Certification = 40,
# WHMIS 2015 Certificate = 22), so a 50-char cap is comfortable.
# other_jobs_meta is deferred to 0 in v1 -- the field is for a future
# job-pivot feature and contributes nothing to remaining-gaps
# subtraction; saving its 300+ bytes is the largest single budget win.
MAX_TITLE_CHARS = 80
MAX_EMPLOYER_CHARS = 60
# 40 chars holds every canonical_name in data/training_registry.yaml at
# R-1 time (longest: "310S automotive technician certification" = 40).
# New registry entries longer than this would be truncated at the
# storage boundary -- add a registry-load-time check there if a longer
# canonical is ever added.
MAX_CANONICAL_CHARS = 40
MAX_CRED_GAPS = 5
MAX_SKILL_GAPS = 4
MAX_OTHER_JOBS = 0

# AR-1 (adjacent-recommendations) caps. The new fields land alongside the
# remaining-gaps snapshot and ride the same 3800-byte signed-cookie budget
# (see docs/adjacent-recommendations-design.md for the full contract).
# Worst-case combined size is exercised by the AR-1 cookie round-trip test.
#
# CAP TIGHTENING (post-design): the design v11 speculated higher caps
# (MAX_JOB_ID_CHARS=64, MAX_PRESENTED_JOB_IDS=24, MAX_EVIDENCE_CHARS=120,
# MAX_MATCHED_SKILLS=5). Measuring against the existing R-1 worst-case
# cookie test showed the combined budget overflowed at those values.
# Values below leave the signed cookie under 3800 bytes alongside the
# realistic resume_facts_json + full R-1 snapshot fixture. The cookie
# round-trip test is the authoritative gate; raise caps here only with
# a corresponding green run of that test.
MAX_PRESENTED_JOB_IDS = 16    # dedup-capped list on last_match_snapshot
# Align with MAX_CANONICAL_CHARS so job_ids inside last_match_snapshot
# and presented_job_ids share the same width contract.
MAX_JOB_ID_CHARS = MAX_CANONICAL_CHARS    # = 40
MAX_ADJACENT_ITEMS = 3        # recommendations cap = surface cap
MAX_EVIDENCE_CHARS = 100
MAX_MATCHED_SKILLS = 4
MAX_SKILL_CHARS = 32

# Accepted enumeration values on the typed records. Defensive
# deserialization drops entries whose mode/action is not in these sets.
_VALID_CREDENTIAL_MODES: frozenset[str] = frozenset({"hypothetical", "claimed"})
_VALID_PENDING_ACTIONS: frozenset[str] = frozenset({"add", "remove"})

# Slice 5 step 2 (2026-06-18): valid RecommenderMode strings for
# pending_recommender_offer. Locally enumerated here so the session
# module carries no import dependency on the chat recommender layer.
# This deliberate independence is enforced by the chat-recommender
# no-consumer guard test (which greps for cross-module references
# until Step 5 wires the consumer).
#
# Keep this set in sync with the RecommenderMode Literal in the
# chat recommender module. The three values are stable product
# modes (one offer per conversational turn).
_VALID_RECOMMENDER_MODES: frozenset[str] = frozenset({
    "local_gap_coach",
    "target_noc_standard",
    "adjacent_noc_standard",
    # Slice 5 (2026-06-29): pending-select state used while the user
    # is choosing which adjacent NOC to drill into. NOT a response
    # mode -- it's a routing signal for the consume hook. The actual
    # render payload is RoleDrilldownEvidence, dispatched separately.
    "adjacent_role_drilldown_select",
})

_VALID_WHY_ADJACENT: frozenset[str] = frozenset({
    "same_noc_minor_group", "skill_evidence",
})


# Slot names recognised by the intake state machine. Keep in sync with
# chat/intake_priority.PRIORITY_ORDER.
INTAKE_SLOTS: tuple[str, ...] = (
    "target_role_text",
    "experience_text",
    "skills_text",
    "education_text",
    "work_type_preference",
    "shift_preference",
    "preferred_location",
    "transportation_text",
    "availability_text",
    "salary_expectation_text",
    "language_preferences",
)


@dataclass
class StagedProfile:
    session_id: str
    created_at: str
    last_active_at: str
    message_count: int = 0
    preferred_location: str | None = None
    target_role_text: str | None = None
    # Matching v2 step 2: NOC 2021 code resolved from target_role_text.
    # The resolver runs lazily in compute_matches_in_memory. To prevent
    # a stale NOC from outliving a user changing their target role, the
    # __setattr__ override below clears target_noc whenever
    # target_role_text is set to a different value. Covers all mutation
    # paths uniformly (merge_fields, fallback_fill setattr, etc.).
    target_noc: str | None = None
    education_text: str | None = None
    experience_text: str | None = None
    skills_text: str | None = None
    work_type_preference: str | None = None
    language_preferences: list[str] = field(default_factory=list)
    skills: list[StagedSkill] = field(default_factory=list)

    # PR 10 intake fields
    salary_expectation_text: str | None = None
    shift_preference: str | None = None
    transportation_text: str | None = None
    availability_text: str | None = None

    # PR 10 intake state machine bookkeeping
    intake_state: str = "anonymous_chat"
    # Cumulative list of slots ever asked about (drives intake_priority,
    # which deprioritises already-asked slots when picking new questions).
    asked_slots: list[str] = field(default_factory=list)
    # Slots asked on the IMMEDIATELY PRECEDING assistant turn. Used by the
    # extractor's blanket-decline detector ("skip that" → only decline what
    # we asked *this* turn, not every slot we've ever asked about). Reset
    # on every turn.
    last_asked_slots: list[str] = field(default_factory=list)
    declined_slots: list[str] = field(default_factory=list)
    last_extracted_at: str | None = None

    # Sprint 1 — resume layer.
    # resume_text and resume_facts_json are gated by session mode:
    #   - Redis mode: both persist for the 30-min TTL.
    #   - Cookie mode: BOTH are redacted before signing (size + privacy).
    #     The user's flat slots and suppressed_fact_ids still persist so
    #     matching continues with the derived view.
    # See docs/resume-design.md §3 and StagedProfile.to_json(redact_for_cookie).
    resume_text: str | None = None
    resume_filename: str | None = None
    resume_parsed_at: str | None = None
    resume_facts_json: dict[str, Any] | None = None
    # Chat orchestration v2 slice 1 review fix: surface failed-upload state
    # across turns. When parse_resume() returns a warning ("too_large",
    # "empty_input", "unsupported_format", "parse_failed", "no_text"), the
    # handler leaves resume_facts_json untouched but sets this field so the
    # planner / truth_summary can distinguish "no upload happened" from
    # "upload happened but parser failed". Cleared on successful upload.
    resume_parse_warning: str | None = None
    # Fact IDs the user suppressed during the RESUME_REVIEW turn. Always
    # serialized so suppressions survive across turns even in cookie mode.
    suppressed_fact_ids: list[str] = field(default_factory=list)

    # Slice 8 -- short-session conversation context.
    # Captured AFTER a v2 present_matches turn so the next-turn responder
    # fallback (especially redirect_scope after a policy-rejection) can
    # reference the matches the user just saw instead of starting cold.
    # Cleared when no matches were presented. Capped to 5 entries each.
    # Tiny payload (~200 bytes); safe under the cookie 4KB ceiling.
    last_presented_job_titles: list[str] = field(default_factory=list)
    last_presented_caps_applied: list[str] = field(default_factory=list)
    last_presented_credential_gaps: list[str] = field(default_factory=list)

    # R-1 (remaining-gaps iteration) -- structured per-job snapshot of the
    # most-recent present_matches turn. Drives the follow-up reasoning
    # layer (subtract / retract / clarify / bootstrap) on subsequent turns.
    # Shape: see docs/remaining-gaps-design.md §1; canonical alias resolution
    # at capture time via training_registry. The four R-1 fields share a
    # lifecycle and are atomically cleared with the snapshot (see
    # _capture_match_snapshot / _clear_match_snapshot in handler.py + the
    # __setattr__ hook below for target_role_text changes).
    last_match_snapshot: dict[str, Any] | None = None
    # Typed records of credentials the user has claimed or hypothetically
    # assumed during the snapshot's lifetime. Each entry:
    #   {"canonical": str, "mode": "hypothetical" | "claimed"}
    # Ordered append-and-dedupe; conversation state, NOT profile evidence.
    last_assumed_completed_credentials: list[dict[str, Any]] = field(default_factory=list)
    # Recency anchor for credential anaphora ("it", "that licence").
    last_discussed_credential_canonical: str | None = None
    # Pending confirmation question state. Set when the handler emits a
    # `kind="confirm"` clarification; consumed and cleared by the NEXT
    # turn's detection layer (the handler clears BEFORE detection runs and
    # only re-sets if the new turn produces another confirm).
    # Shape: {"canonical": str, "action": "add" | "remove"}
    pending_credential_confirmation: dict[str, Any] | None = None

    # AR-1 (adjacent-recommendations) state.
    #
    # pending_adjacent_offer: True iff the previous responder turn appended
    # the soft "want me to look at related roles?" offer. Read + cleared at
    # the top of handle_anonymous (save-and-clear) and threaded into
    # detect_adjacent_intent. SETTER lives in the handler's soft-offer
    # wiring (lands in AR-6); AR-1's hook is a no-op until then. Survives
    # one cookie round-trip; defensive-deserialize collapses non-bool to
    # False.
    pending_adjacent_offer: bool = False

    # CP3 step 3 (2026-06-15) — pending_training_topic.
    # True iff the previous responder turn emitted Rule 3
    # ("what skill or certificate do you want training for?"). The
    # NEXT meaningful user turn is interpreted as the training topic
    # answer rather than a fresh skill claim. Consumed BEFORE the
    # normal extract/merge pass so a bare topic answer ("Excel")
    # does NOT silently become a profile skill.
    #
    # Lifecycle:
    #   - Set by the handler when Rule 3 fires (next-turn write).
    #   - Blank input does NOT consume it (intake will re-ask).
    #   - A meaningful turn consumes it (cleared at top of handler).
    #   - Scope violations follow their normal precedence; the flag
    #     stays set so the topic question persists across one
    #     redirect_scope digression, then naturally times out.
    #   - Survives the session-store round trip via the same
    #     bool-or-False sanitization used for pending_adjacent_offer.
    pending_training_topic: bool = False

    # last_adjacent_snapshot: ≤ 3 recommendations from the most-recent
    # recommend_adjacent_roles turn. Powers "tell me about the second one"
    # via resolve_adjacent_followup. Shape:
    #   {
    #     "created_message_count": int,           # captured BEFORE touch()
    #     "items": [                              # max MAX_ADJACENT_ITEMS
    #       {"job_id": str,                       # cap MAX_JOB_ID_CHARS
    #        "title": str,                        # cap MAX_TITLE_CHARS
    #        "evidence_summary": str,             # cap MAX_EVIDENCE_CHARS
    #        "why_adjacent": str,                 # _VALID_WHY_ADJACENT
    #        "matched_skills": list[str]},        # cap MAX_MATCHED_SKILLS
    #       ...
    #     ],
    #   }
    # TTL: live iff current_message_count == created + 1. Cleared on:
    #   - next present_matches / present_near_miss decision (handler);
    #   - target_role_text change (__setattr__);
    #   - natural TTL expiry (resolve_adjacent_followup).
    # On a scope-violation turn the handler shifts created_message_count
    # forward by 1 via shift_adjacent_snapshot_ttl so a single
    # redirect_scope digression doesn't burn the followup window.
    last_adjacent_snapshot: dict[str, Any] | None = None

    # Fresh-intake-on-target-change pillar (2026-06-15) — alignment
    # tracking. Each holds the literal target_role_text value that was
    # active when the corresponding evidence (skills / experience_text)
    # was most recently merged. None on a cold profile. The lifecycle:
    #
    #   - merge_skills with ≥1 new entry → skills_collected_for_target
    #     = current self.target_role_text.
    #   - merge_fields with a non-empty experience_text →
    #     experience_collected_for_target = current self.target_role_text.
    #   - target_role_text changes via __setattr__ → DO NOT clear these.
    #     The mismatch between the field's stored value and the new
    #     target_role_text IS the load-bearing signal that the truth
    #     summary uses to gate engine-run on a fresh intake.
    #
    # Why two fields instead of one: the locked original design says
    # the engine needs both skills evidence AND experience evidence
    # aligned with the current target. A profile with only one aligned
    # is still mid-intake; both being aligned (or both legitimately
    # empty under a target with no expectation) is the gate condition.
    skills_collected_for_target: str | None = None
    experience_collected_for_target: str | None = None

    # Resume-upload offer (2026-06-16) — once-PER-TARGET flag (Gap 3
    # fix, 2026-06-16). Set True when the responder has rendered the
    # "upload a CV to unlock more matches" offer. Reset to False on
    # target_role_text change via __setattr__ so a user who ignored
    # the offer for the prior target can hear it again for the new
    # target (the offer is a per-target product behaviour, not a
    # per-session one).
    resume_upload_offered: bool = False

    # Pattern 2 two-turn flag (closing-matrix v2, LOCKED 2026-06-17).
    # Set True on Turn N when the Pattern 2 closing fires — the LLM
    # has just asked "want me to also look at related roles your
    # skills fit?" and we expect the user to respond yes / no /
    # something-else on Turn N+1. Consumed by the planner / handler
    # on the next turn to route a yes-like reply into the CP5
    # adjacency search (= Sideways infrastructure reuse, Step 8) and
    # clear the flag on any other reply.
    #
    # Reset to False on target_role_text change via __setattr__: a
    # role switch invalidates any pending consent for the prior
    # target's offer. Same pattern as resume_upload_offered.
    pending_adjacent_search_offer: bool = False

    # Slice 5 step 2 (2026-06-18): conversational recommender pending
    # offer. When the system has emitted "want me to coach you on the
    # skill gaps for these postings / national standard / adjacent
    # roles?", the stored mode string tells the next-turn handler
    # which recommender flow to resume on a yes-consent reply.
    #
    # Stored as the RecommenderMode string (NOT a bool) so the field
    # carries both "is an offer pending?" and "which mode?" in one
    # slot. Mirrors the RecommenderMode Literal defined in the chat
    # recommender layer but does NOT import it -- the no-consumer
    # guard test pins zero cross-module references outside the chat
    # recommender module itself. The local _VALID_RECOMMENDER_MODES
    # frozenset enumerates the same three strings.
    #
    # Lifecycle (Step 2 plumbing):
    #   - Reset to None on target_role_text change via __setattr__.
    #   - Counted in A2-α3's _count_entry_pending_flags so a bare yes
    #     with this flag + another pending flag triggers the
    #     ambiguity guard.
    #   - Lossless minification in to_json(redact_for_cookie=True) so
    #     a default None value never eats cookie budget.
    #   - Defensive deserialization: invalid mode strings or non-str
    #     values are sanitized to None at from_json time.
    #
    # NOT included in Step 2:
    #   - SET point (which turn emits the offer) -- belongs in Step 4
    #     with the conversational flow design, to avoid conflict with
    #     the existing pending_adjacent_search_offer set point.
    #   - Consume / route logic on yes-consent -- belongs in Step 4
    #     alongside the recommender response.
    pending_recommender_offer: str | None = None

    # Slice 5 step 4 (2026-06-19): adjacent NOC codes captured at the
    # present_tiered_matches turn so the adjacent_noc_standard
    # recommender mode (locked design third in the chain) can compute
    # against the same NOCs the user saw earlier in the conversation.
    #
    # WHY a separate field and not last_adjacent_snapshot.items:
    #   last_adjacent_snapshot powers ordinal follow-up
    #   ("tell me about the second one") and has its own sanitizer,
    #   tests, and cookie budget. Its `items` shape carries
    #   job_id/title/evidence_summary/why_adjacent/matched_skills but
    #   NO noc_code -- a deliberate separation. Mixing noc_code into
    #   that shape would conflate two purposes (ordinal-follow-up vs
    #   per-NOC OaSIS comparison).
    #
    # SHAPE: tuple of unique exact-5-digit NOC codes, max 3.
    # Layer C's recommender fetches the human-readable NOC title
    # from OaSIS at compute time (reference.occupation via the
    # existing JOIN in gap_evidence._LAYER_A_SQL), so codes-only is
    # sufficient and minimises cookie footprint.
    #
    # LIFECYCLE (chain-bound, NOT time-based):
    #   - Captured: when present_tiered_matches emits AND
    #     tier_evidence.sideways_move has records. Extract unique
    #     noc_code from each AdjacentJob; cap at 3 by order received.
    #   - Cleared: target_role_text change (per-target hard reset);
    #     fresh tier-match turn overwrites the previous value;
    #     after adjacent_noc_standard recommender turn finishes
    #     (chain natural end -- the handler sets
    #     pending_recommender_offer = None AND clears this field
    #     simultaneously); user declines at any chain step (the
    #     handler clears the chain-bound state).
    #   - NO time-based TTL. The chain itself bounds the lifetime.
    #
    # COOKIE COST: 3 codes x 5 chars + JSON framing ~25-30 bytes
    # worst case. Lossless minification when empty drops the key
    # entirely. Defensive deserialization filters out non-string
    # entries and codes that fail _is_valid_noc_code (exact 5
    # digits).
    #
    # NOT included in this slice:
    #   - SET write point (handler.py:~2497 SET swap is part of
    #     Step 4 as a whole; this slice only adds the field/lifecycle
    #     plumbing on StagedProfile).
    #   - Layer C consumer logic (Step 4 dispatch branch).
    last_adjacent_nocs: tuple[str, ...] = field(default_factory=tuple)

    # Slice 1 follow-up (2026-06-23): deferred career intent.
    # When the router emits `ask_substrate` because target or skills
    # are missing, this field remembers WHICH recommender intent the
    # user was about to express, so the next turn (after substrate
    # fills) can route to that intent instead of dropping it.
    #
    # SHAPE: one of the CareerIntent literal values
    # (local_skill_gap | training_recommendation |
    #  noc_standard_comparison | career_exploration |
    #  application_help_out_of_scope) or None.
    # job_matching is NEVER deferred (matching engine has its own
    # intake; no substrate gating). unclear is NEVER deferred (no
    # intent to remember).
    #
    # LIFECYCLE:
    #   - Set: when _maybe_route_recommender_from_intent emits an
    #     ask_substrate verdict carrying a deferred_intent
    #   - Consumed: when the next turn brings the substrate to
    #     sufficient AND the current message classifies as `unclear`
    #     (user is just providing substrate, not naming a new
    #     intent). The bridge then routes to the deferred intent
    #     and clears the field.
    #   - Cleared (no consumption): target_role_text change
    #     (substrate invalidated); explicit non-unclear intent in
    #     the current message (the new intent supersedes the
    #     deferred one).
    deferred_career_intent: str | None = None

    # Slice 5 (2026-06-29): capture which adjacent NOCs were shown
    # in the most recent Layer C render. Lets the next-turn ordinal/
    # name resolver in _dispatch_recommender_consume map "the first
    # one" / "administrative secretary" / "13110" back to a concrete
    # NOC code for drilldown dispatch.
    #
    # SHAPE: tuple of dicts; each dict has exactly:
    #   {"noc_code": "13110", "title": "Administrative assistant"}
    # Capped at 3 entries (matches _MAX_RECOMMENDER_ADJACENT_NOCS in
    # the recommender_assembly slice 4 helper). Title is the OaSIS
    # noc_title from reference.noc_skill.
    #
    # LIFECYCLE (paired with pending_recommender_offer per slice 5
    # lock; both stay alive while user is in selection context, both
    # clear together):
    #   - Set: when Layer C renders adjacent NOCs in the recommender
    #     dispatcher (intent or consume path).
    #   - Kept after drilldown render: user can pick another from the
    #     same surface ("now the second one"). Pending also kept as
    #     "adjacent_role_drilldown_select".
    #   - Cleared:
    #       * target_role_text change (__setattr__ override below)
    #       * target_noc change (__setattr__ override below)
    #       * user declines (consume hook consent=="no")
    #       * user pivots to unrelated intent (consume hook consent
    #         =="other" with no surface match)
    #
    # COOKIE COST: capped at 3 entries; each ~60-80 bytes JSON
    # (noc_code 5 chars + title up to ~50 chars + framing). Worst-
    # case ~250 bytes. Same lossless-empty rule as last_adjacent_nocs.
    last_recommender_adjacent_surface: tuple[dict, ...] = field(
        default_factory=tuple,
    )

    # Step 1.2 (2026-07-03) — message_count anchor for the recommender's
    # Layer C adjacent surface. Stamped alongside
    # last_recommender_adjacent_surface at render time; cleared alongside
    # it. Consumed by ConversationFrame._pick_latest_surface to order
    # competing surfaces by recency (max message_count wins). Lives with
    # its surface: cleared on target_role_text change via __setattr__.
    # Lossless minification when None in cookie mode.
    last_recommender_adjacent_surface_at_turn: int | None = None

    # Step 1.2 (2026-07-03) — message_count anchor for the matching
    # engine's last_presented_job_titles surface. Stamped in
    # _capture_presented_context; cleared in _clear_presented_context.
    # Not cleared on target_role_text change (matches the existing
    # lifecycle of last_presented_job_titles, which is a one-turn
    # fallback field that survives a target switch until the next
    # present_matches decision overwrites or clears it).
    last_presented_at_turn: int | None = None

    # ----------------------------------------------- attribute interception
    def __setattr__(self, name: str, value: Any) -> None:
        """Invalidate cached target_noc when target_role_text changes.

        Matching v2 step 2 review fix. The engine caches target_noc on
        the staged profile (resolved lazily in compute_matches_in_memory)
        so we don't re-hit the DB every chat turn. But the cache must
        invalidate when the user changes their target role -- otherwise
        a stale NOC could keep boosting the wrong jobs:
          user: "software developer"   -> target_noc=21232
          user: "warehouse work"       -> target_noc still 21232 if not cleared
        Catching this in __setattr__ covers every mutation path uniformly
        (merge_fields, closed_vocab_fill setattr, fallback_fill setattr,
        and any future direct assignment).
        """
        if name == "target_role_text":
            current = self.__dict__.get("target_role_text")
            if value != current:
                # Use __dict__ to skip our own override and avoid recursion.
                self.__dict__["target_noc"] = None
                # R-1 lifecycle: the remaining-gaps snapshot and its three
                # companion fields belong to the prior target role. A role
                # change invalidates the snapshot (new role -> new gaps),
                # so clear all four together. See
                # docs/remaining-gaps-design.md §10.
                self.__dict__["last_match_snapshot"] = None
                self.__dict__["last_assumed_completed_credentials"] = []
                self.__dict__["last_discussed_credential_canonical"] = None
                self.__dict__["pending_credential_confirmation"] = None
                # AR-1: the adjacent-recommendations snapshot belongs to the
                # PRIOR target role -- a role change makes its ordinal
                # follow-up references meaningless. pending_adjacent_offer
                # is left alone (it's about UI state in flight, and the
                # one-turn handler-entry save-and-clear already bounds it).
                self.__dict__["last_adjacent_snapshot"] = None
                # Resume-upload offer (Gap 3 fix, 2026-06-16):
                # `resume_upload_offered` is per-target, not per-session.
                # A user who ignored the offer for the prior target can
                # hear it again for the new target.
                self.__dict__["resume_upload_offered"] = False
                # Pattern 2 pending-consent (closing-matrix v2,
                # 2026-06-17): target switch invalidates any pending
                # "want me to look at related roles?" consent — the
                # prior question was about the prior target's
                # adjacencies. Same per-target lifecycle as
                # resume_upload_offered.
                self.__dict__["pending_adjacent_search_offer"] = False
                # Slice 5 step 2 (2026-06-18): target switch
                # invalidates any pending recommender offer too --
                # the prior offer was about the prior target's gaps.
                # Same per-target lifecycle as pending_adjacent_search_offer.
                self.__dict__["pending_recommender_offer"] = None
                # Slice 5 step 4 (2026-06-19): target switch also
                # invalidates the adjacent-NOC list captured at the
                # prior target's tier-match turn. The new target will
                # surface a new sideways_move with potentially
                # different adjacent NOCs.
                self.__dict__["last_adjacent_nocs"] = ()
                # Slice 5 (2026-06-29): target switch invalidates
                # the recommender's adjacent-NOC surface snapshot --
                # the prior surface was computed against the prior
                # target's skill profile + adjacency pipeline. Drop
                # it so post-target-change selection can't pick from
                # the stale list.
                self.__dict__["last_recommender_adjacent_surface"] = ()
                # Step 1.2 (2026-07-03): companion anchor for the
                # recommender adjacent surface. Cleared alongside the
                # surface itself; keeping a live anchor to a surface
                # that has been dropped would make recency ordering
                # in ConversationFrame nonsensical.
                self.__dict__[
                    "last_recommender_adjacent_surface_at_turn"
                ] = None
                # Slice 1 follow-up (2026-06-23): clear deferred
                # career intent ONLY on a true target switch (prior
                # value existed and new differs). On a FIRST target
                # fill (current was None/empty), the deferred intent
                # was set with no target yet WAITING for substrate
                # to fulfill so the router can consume it on the
                # next turn. Clearing on first-fill would silently
                # drop the user's intent.
                prior_target_existed = (
                    isinstance(current, str) and current.strip() != ""
                )
                if prior_target_existed:
                    self.__dict__["deferred_career_intent"] = None
        # Fresh-intake-on-target-change pillar (2026-06-15) — stamp
        # experience alignment on ANY non-empty experience_text
        # assignment. Catching this here (not just in merge_fields)
        # covers every setter path uniformly: merge_fields,
        # fallback_fill direct setattr (see handler.py §2b
        # "open-text slot accept"), closed_vocab_fill, and any
        # future direct assignment. Without this, fallback_fill
        # filling experience_text bypassed the stamp and the
        # alignment gate kept firing even after experience was
        # filled (live-2026-06-16 repro).
        if name == "experience_text":
            if isinstance(value, str) and value.strip():
                self.__dict__["experience_collected_for_target"] = (
                    self.__dict__.get("target_role_text")
                )
        super().__setattr__(name, value)

    # ----------------------------------------------- factory / lifecycle
    @classmethod
    def new(cls, session_id: str) -> "StagedProfile":
        now = datetime.now(timezone.utc).isoformat()
        return cls(session_id=session_id, created_at=now, last_active_at=now)

    def touch(self) -> None:
        self.last_active_at = datetime.now(timezone.utc).isoformat()
        self.message_count += 1

    # ----------------------------------------------- field merging
    def merge_fields(self, fields: dict[str, Any]) -> None:
        """Apply LLM-extracted profile fields, only overwriting where we have a non-empty value.

        Declined slots are never overwritten by re-extraction; the user has
        already told us they don't want that field filled.
        """
        allowed = {
            "preferred_location", "target_role_text", "education_text",
            "experience_text", "skills_text", "work_type_preference",
            "language_preferences",
            "salary_expectation_text", "shift_preference",
            "transportation_text", "availability_text",
        }
        declined = set(self.declined_slots)
        for k, v in fields.items():
            if k not in allowed:
                continue
            if k in declined:
                continue
            if v in (None, "", []):
                continue
            setattr(self, k, v)
            # Fresh-intake-on-target-change pillar (2026-06-15):
            # the experience_text stamp is now in __setattr__ so it
            # covers fallback_fill / closed_vocab_fill paths too.
            # This site is left as a no-op stamp comment for traceability.

    def merge_skills(self, new_skills: list[StagedSkill]) -> int:
        """Union semantics — never lower confidence. Return count of new entries.

        Fresh-intake-on-target-change pillar (2026-06-15): when at least
        one new skill is appended (added > 0), stamp
        `skills_collected_for_target` with the current
        `target_role_text`. Confidence-only upgrades on already-present
        skills do NOT count as new evidence — the user has not
        re-affirmed the skill against the new target, just had its
        confidence floor lifted by a re-extraction pass.
        """
        existing = {s.skill_name.lower(): s for s in self.skills}
        added = 0
        for ns in new_skills:
            key = ns.skill_name.lower()
            if key in existing:
                prior = existing[key]
                if ns.confidence > prior.confidence:
                    prior.confidence = ns.confidence
                    prior.source = ns.source
                if ns.skill_id and not prior.skill_id:
                    prior.skill_id = ns.skill_id
                if ns.raw_phrase and not prior.raw_phrase:
                    prior.raw_phrase = ns.raw_phrase
            else:
                self.skills.append(ns)
                existing[key] = ns
                added += 1
        if added > 0:
            self.__dict__["skills_collected_for_target"] = (
                self.target_role_text
            )
        return added

    # ----------------------------------------------- intake helpers
    def mark_asked(self, slot: str) -> None:
        if slot not in self.asked_slots:
            self.asked_slots.append(slot)

    def mark_declined(self, slot: str) -> None:
        if slot not in self.declined_slots:
            self.declined_slots.append(slot)

    def filled_slots(self) -> set[str]:
        """Slots that have a usable non-empty value on the staged profile."""
        out: set[str] = set()
        if self.target_role_text:           out.add("target_role_text")
        if self.experience_text:            out.add("experience_text")
        if self.skills_text or self.skills: out.add("skills_text")
        if self.education_text:             out.add("education_text")
        if self.work_type_preference:       out.add("work_type_preference")
        if self.shift_preference:           out.add("shift_preference")
        if self.preferred_location:         out.add("preferred_location")
        if self.transportation_text:        out.add("transportation_text")
        if self.availability_text:          out.add("availability_text")
        if self.salary_expectation_text:    out.add("salary_expectation_text")
        if self.language_preferences:       out.add("language_preferences")
        return out

    # ----------------------------------------------- serialization
    def to_json(self, *, redact_for_cookie: bool = False) -> str:
        """Serialize for the session store.

        redact_for_cookie=True is set by the signed-cookie store, which has
        a ~4 KB hard limit on the signed payload. Transformations:
          - resume_text → None              (raw text routinely 10-50 KB)
          - resume_facts_json → compact_facts() form, which strips evidence
            strings and long descriptions while preserving fact_ids,
            names, dates, and source tags.
          - AR-1 adjacent-recommendations: the activation contract (locked
            in AR-1c / AR-6) gates the entire feature behind Redis-mode
            sessions, so in cookie mode the new fields are ALWAYS at
            their dataclass defaults (pending_adjacent_offer=False,
            last_adjacent_snapshot=None, last_match_snapshot has no
            "presented_job_ids" entry). The cookie path emits them as
            absent keys when they hold defaults -- a lossless JSON
            minification, NOT a value-bearing redaction. from_json's
            defensive deserialize reconstructs the defaults when keys
            are missing. The 30+ saved bytes are what keeps the R-1
            worst-case fixture under the 3800-byte signed ceiling.

        suppressed_fact_ids and the flat StagedProfile slots persist as-is
        so derived matching state and user corrections survive across turns.

        Hard invariant: cookie payloads NEVER carry raw resume text.
        """
        data = asdict(self)
        if redact_for_cookie:
            # Local import: derive.py imports from config.py which imports
            # this module via session_store factory at runtime. Defer to
            # break the cycle on package load.
            from skillbridge.resume.derive import compact_facts

            data["resume_text"] = None
            data["resume_facts_json"] = compact_facts(data.get("resume_facts_json"))

            # AR-1 lossless minification: drop adjacency keys that hold
            # their dataclass defaults. Cookie mode never sets non-default
            # values for these keys (Redis-gated activation in AR-6), so
            # the omission is lossless. If a non-default value DOES
            # appear here, it stays in the payload -- the cookie-size
            # test will then trip, surfacing the bug instead of hiding
            # it under a silent overwrite.
            if data.get("pending_adjacent_offer") is False:
                data.pop("pending_adjacent_offer", None)
            if data.get("pending_training_topic") is False:
                data.pop("pending_training_topic", None)
            if data.get("last_adjacent_snapshot") is None:
                data.pop("last_adjacent_snapshot", None)
            snap = data.get("last_match_snapshot")
            if isinstance(snap, dict) and not snap.get("presented_job_ids"):
                snap.pop("presented_job_ids", None)
            # Fresh-intake-on-target-change pillar (2026-06-15) +
            # resume-upload offer (2026-06-16): lossless minification.
            # All three new fields default to None / False on a cold
            # session and only carry a non-default value when the user
            # has actually merged evidence (skills/experience) or been
            # offered an upload. Cookie-mode profiles drop them when
            # default; `from_json`'s defensive deserialize reconstructs
            # the defaults from absent keys. Saves ~120 bytes worst
            # case (two strings + one bool key+value pair).
            if data.get("skills_collected_for_target") is None:
                data.pop("skills_collected_for_target", None)
            if data.get("experience_collected_for_target") is None:
                data.pop("experience_collected_for_target", None)
            if data.get("resume_upload_offered") is False:
                data.pop("resume_upload_offered", None)
            # Slice 5 step 2 (2026-06-18): lossless minification for
            # pending_recommender_offer. The default None value never
            # carries meaning; dropping it saves ~40 bytes per cookie
            # round-trip and keeps headroom against the 3800-byte
            # signed-cookie ceiling. from_json's defensive deserialize
            # reconstructs the default from absent keys.
            if data.get("pending_recommender_offer") is None:
                data.pop("pending_recommender_offer", None)
            # Slice 5 step 4 (2026-06-19): lossless minification for
            # last_adjacent_nocs. Empty tuple is the default; drop
            # the key. from_json reconstructs the default from
            # absence. Worst-case populated cost is ~25-30 bytes
            # (3 codes x 5 chars + JSON framing).
            adj_nocs = data.get("last_adjacent_nocs")
            if not adj_nocs:  # () or [] or None
                data.pop("last_adjacent_nocs", None)
            # Slice 1 follow-up (2026-06-23): lossless minification for
            # deferred_career_intent. None is the default; drop the key
            # so empty state pays zero bytes.
            if data.get("deferred_career_intent") is None:
                data.pop("deferred_career_intent", None)
            # Slice 5 (2026-06-29): lossless minification for
            # last_recommender_adjacent_surface. Empty tuple/list is
            # the default; drop the key.
            surface = data.get("last_recommender_adjacent_surface")
            if not surface:  # () or [] or None
                data.pop("last_recommender_adjacent_surface", None)
            # Step 1.2 (2026-07-03): lossless minification for the two
            # surface anchor fields. Both default to None; drop the key
            # when unset. Explicit `is None` check because 0 is a valid
            # anchor value (message_count starts at 0 on a fresh session).
            if data.get("last_recommender_adjacent_surface_at_turn") is None:
                data.pop("last_recommender_adjacent_surface_at_turn", None)
            if data.get("last_presented_at_turn") is None:
                data.pop("last_presented_at_turn", None)
        return json.dumps(data, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str | bytes) -> "StagedProfile":
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        data = json.loads(raw)
        skills_raw = data.pop("skills", []) or []
        skills = [StagedSkill(**s) for s in skills_raw if isinstance(s, dict)]
        # Backward-compat: older session blobs (pre-PR-10, pre-Sprint-1)
        # won't have the newer fields. dataclass defaults take care of
        # them, but if a key is present-but-wrong-type, drop it rather
        # than crash.
        for k in ("asked_slots", "last_asked_slots", "declined_slots",
                  "language_preferences", "suppressed_fact_ids"):
            if k in data and not isinstance(data[k], list):
                data[k] = []
        if "resume_facts_json" in data and not isinstance(
            data["resume_facts_json"], (dict, type(None))
        ):
            data["resume_facts_json"] = None

        # R-1 defensive deserialization for the four remaining-gaps fields.
        # The signed cookie is HMAC-verified but (a) keys rotate, (b) a
        # forged cookie that passes the signature check should not crash
        # the handler, (c) malformed legacy blobs from intermediate deploys
        # should degrade gracefully. Per-field discipline: wrong type ->
        # drop / default; unknown enum value -> drop the entry; out-of-cap
        # lists -> truncate.
        if "last_match_snapshot" in data:
            data["last_match_snapshot"] = _sanitize_snapshot(data["last_match_snapshot"])
        if "last_assumed_completed_credentials" in data:
            data["last_assumed_completed_credentials"] = _sanitize_accumulated(
                data["last_assumed_completed_credentials"]
            )
        if "last_discussed_credential_canonical" in data:
            data["last_discussed_credential_canonical"] = _sanitize_canonical_str(
                data["last_discussed_credential_canonical"]
            )
        if "pending_credential_confirmation" in data:
            data["pending_credential_confirmation"] = _sanitize_pending(
                data["pending_credential_confirmation"]
            )

        # AR-1 defensive deserialization for the adjacent-recommendations
        # fields. Wrong type -> safe default; never crash.
        if "pending_adjacent_offer" in data:
            data["pending_adjacent_offer"] = _sanitize_pending_adjacent_offer(
                data["pending_adjacent_offer"]
            )
        if "pending_training_topic" in data:
            data["pending_training_topic"] = _sanitize_pending_adjacent_offer(
                data["pending_training_topic"]
            )
        if "last_adjacent_snapshot" in data:
            data["last_adjacent_snapshot"] = _sanitize_adjacent_snapshot(
                data["last_adjacent_snapshot"]
            )
        # Slice 5 step 2 (2026-06-18): pending_recommender_offer must
        # be one of the canonical RecommenderMode strings or None. A
        # forged cookie with an arbitrary string would otherwise route
        # to a nonexistent recommender flow. Missing key -> default
        # None via the dataclass.
        if "pending_recommender_offer" in data:
            data["pending_recommender_offer"] = _sanitize_pending_recommender_offer(
                data["pending_recommender_offer"]
            )
        # Slice 5 step 4 (2026-06-19): last_adjacent_nocs is a tuple
        # of exact 5-digit NOC codes. Sanitizer drops malformed
        # entries (non-str, wrong length, non-digit) and caps at 3.
        # Empty tuple is the default when the key is absent.
        if "last_adjacent_nocs" in data:
            data["last_adjacent_nocs"] = _sanitize_last_adjacent_nocs(
                data["last_adjacent_nocs"]
            )
        # Slice 1 follow-up (2026-06-23): deferred_career_intent is one
        # of a closed enum of strings or None. Sanitizer accepts only
        # valid CareerIntent values (minus job_matching and unclear
        # which are never deferred); anything else coerces to None.
        if "deferred_career_intent" in data:
            data["deferred_career_intent"] = _sanitize_deferred_career_intent(
                data["deferred_career_intent"]
            )
        # Slice 5 (2026-06-29): last_recommender_adjacent_surface is a
        # tuple of dicts {noc_code, title}. Sanitizer drops malformed
        # entries and caps at 3.
        if "last_recommender_adjacent_surface" in data:
            data["last_recommender_adjacent_surface"] = (
                _sanitize_last_recommender_adjacent_surface(
                    data["last_recommender_adjacent_surface"]
                )
            )
        # Step 1.2 (2026-07-03): anchor fields must be int|None. A
        # forged cookie carrying a string or bool would otherwise be
        # written into the dataclass and blow up recency ordering. bool
        # is a subclass of int in Python so we exclude it explicitly.
        for _anchor_key in (
            "last_recommender_adjacent_surface_at_turn",
            "last_presented_at_turn",
        ):
            if _anchor_key in data:
                _v = data[_anchor_key]
                if isinstance(_v, bool) or not isinstance(_v, int) or _v < 0:
                    data[_anchor_key] = None
        return cls(**data, skills=skills)


# ---------------------------------------------------------------- R-1 helpers
def _truncate(s: Any, cap: int) -> str:
    """Return `s` coerced to str and truncated to `cap` chars. Non-str
    inputs come back as the empty string."""
    if not isinstance(s, str):
        return ""
    return s[:cap]


def _sanitize_canonical_str(value: Any) -> str | None:
    """Validate `last_discussed_credential_canonical`. Returns None when
    the value is not a non-empty string; truncates to MAX_CANONICAL_CHARS."""
    if not isinstance(value, str) or not value:
        return None
    return value[:MAX_CANONICAL_CHARS]


def _sanitize_pending(value: Any) -> dict[str, Any] | None:
    """Validate `pending_credential_confirmation`. Drops to None unless
    the value is a dict with a non-empty string `canonical` AND an
    `action` in {"add", "remove"}. Malformed pending state is the one
    case where coercion would be unsafe: we'd rather forget the question
    was asked than misinterpret the user's "yes"."""
    if not isinstance(value, dict):
        return None
    canonical = value.get("canonical")
    action = value.get("action")
    if not isinstance(canonical, str) or not canonical:
        return None
    if action not in _VALID_PENDING_ACTIONS:
        return None
    return {"canonical": canonical[:MAX_CANONICAL_CHARS], "action": action}


def _sanitize_accumulated(value: Any) -> list[dict[str, Any]]:
    """Validate `last_assumed_completed_credentials`. Each entry must be
    a dict with a non-empty string `canonical` AND a `mode` in
    {"hypothetical", "claimed"}. Unknown modes drop the ENTRY (we don't
    silently coerce — a future version's enum extension shouldn't sneak
    in here).

    Round-18: also dedupe duplicate canonicals at the cookie boundary
    (preserving first-position order, promoting hypothetical -> claimed
    when ANY duplicate is claimed). A cookie that survived signature
    verification with `[{A, hypothetical}, {A, claimed}]` shouldn't
    cross into handler land carrying that contradiction.

    Caps at MAX_CRED_GAPS preserving order (drop tail).
    """
    if not isinstance(value, list):
        return []
    # Round-19 fix: scan the FULL input first (dedupe + promote), THEN
    # cap. Breaking inside the loop after collecting MAX_CRED_GAPS
    # unique entries would skip a later duplicate's promotion --
    # `[A hypothetical, B, C, D, E, A claimed]` would have kept A as
    # hypothetical because the loop exited before seeing the second A.
    # Matches the R-1 invariant: dedupe before cap.
    out: list[dict[str, Any]] = []
    seen_index: dict[str, int] = {}
    for entry in value:
        if not isinstance(entry, dict):
            continue
        canonical = entry.get("canonical")
        mode = entry.get("mode")
        if not isinstance(canonical, str) or not canonical:
            continue
        if mode not in _VALID_CREDENTIAL_MODES:
            continue
        canonical = canonical[:MAX_CANONICAL_CHARS]
        if canonical in seen_index:
            existing = out[seen_index[canonical]]
            if existing["mode"] == "hypothetical" and mode == "claimed":
                existing["mode"] = "claimed"
            # else: keep the first occurrence's mode (claimed first
            # then hypothetical -> stay claimed; same-mode duplicates
            # are no-ops)
            continue
        seen_index[canonical] = len(out)
        out.append({"canonical": canonical, "mode": mode})
    return out[:MAX_CRED_GAPS]


def _sanitize_snapshot(value: Any) -> dict[str, Any] | None:
    """Validate `last_match_snapshot`. Returns None if the top-level
    shape isn't a dict with a usable `lead_job`. The schema follows
    docs/remaining-gaps-design.md §1 verbatim; caps + per-entry shape
    are enforced so a forged cookie can't blow the budget."""
    if not isinstance(value, dict):
        return None
    lead = value.get("lead_job")
    if not isinstance(lead, dict):
        return None

    title = _truncate(lead.get("title"), MAX_TITLE_CHARS)
    employer_raw = lead.get("employer")
    employer = employer_raw[:MAX_EMPLOYER_CHARS] if isinstance(employer_raw, str) else None
    job_id = _truncate(lead.get("job_id"), MAX_CANONICAL_CHARS)

    # Each list-typed field MUST pass an isinstance(..., list) check
    # before we slice it. A forged cookie that supplies a dict here would
    # otherwise raise KeyError on the `[:MAX_*]` slice (dict.__getitem__
    # doesn't accept slice keys). Defensive-deserialization rule: wrong
    # type -> empty list, never crash. (Round-10 R-1 review.)
    raw_cred_gaps = lead.get("credential_gaps")
    credential_gaps: list[dict[str, Any]] = []
    if isinstance(raw_cred_gaps, list):
        for g in raw_cred_gaps[:MAX_CRED_GAPS]:
            if not isinstance(g, dict):
                continue
            d = g.get("display")
            c = g.get("canonical")
            if not isinstance(d, str) or not isinstance(c, str) or not d or not c:
                continue
            credential_gaps.append({
                "display": d[:MAX_CANONICAL_CHARS],
                "canonical": c[:MAX_CANONICAL_CHARS],
            })

    raw_skill_gaps = lead.get("core_skill_gaps")
    core_skill_gaps: list[str] = []
    if isinstance(raw_skill_gaps, list):
        for s in raw_skill_gaps[:MAX_SKILL_GAPS]:
            if not isinstance(s, str) or not s:
                continue
            core_skill_gaps.append(s[:MAX_CANONICAL_CHARS])

    raw_other = value.get("other_jobs_meta")
    other_jobs_meta: list[dict[str, Any]] = []
    if isinstance(raw_other, list):
        for j in raw_other[:MAX_OTHER_JOBS]:
            if not isinstance(j, dict):
                continue
            ji = j.get("job_id")
            jt = j.get("title")
            if not isinstance(ji, str) or not isinstance(jt, str) or not ji or not jt:
                continue
            other_jobs_meta.append({
                "job_id": ji[:MAX_CANONICAL_CHARS],
                "title":  jt[:MAX_TITLE_CHARS],
            })

    captured_at_turn = value.get("captured_at_turn")
    if not isinstance(captured_at_turn, int):
        captured_at_turn = 0

    # AR-1: presented_job_ids ride alongside the existing snapshot. They
    # accumulate the full set of job_ids the user has already been shown
    # via present_matches / present_near_miss so the adjacency engine can
    # exclude them (drop_excluded in AR-4). Capped, deduped, deterministic
    # order (most-recent first).
    raw_presented = value.get("presented_job_ids")
    presented_job_ids: list[str] = []
    if isinstance(raw_presented, list):
        seen: set[str] = set()
        for ji in raw_presented:
            if not isinstance(ji, str) or not ji:
                continue
            capped = ji[:MAX_JOB_ID_CHARS]
            if capped in seen:
                continue
            seen.add(capped)
            presented_job_ids.append(capped)
            if len(presented_job_ids) >= MAX_PRESENTED_JOB_IDS:
                break

    return {
        "captured_at_turn": captured_at_turn,
        "lead_job": {
            "job_id":   job_id,
            "title":    title,
            "employer": employer,
            "credential_gaps":  credential_gaps,
            "core_skill_gaps":  core_skill_gaps,
        },
        "other_jobs_meta": other_jobs_meta,
        "presented_job_ids": presented_job_ids,
    }


# ---------------------------------------------------------------- AR-1 helpers
def _sanitize_pending_adjacent_offer(value: Any) -> bool:
    """Defensive-deserialize the soft-offer flag. Mirrors the precedent
    used by _sanitize_pending: anything other than literal True collapses
    to False. A forged cookie cannot trick the handler into believing an
    offer was issued when it wasn't."""
    return value is True


def _sanitize_pending_recommender_offer(value: Any) -> str | None:
    """Defensive-deserialize the conversational recommender pending
    offer mode. Returns the value verbatim only when it's one of the
    canonical strings in _VALID_RECOMMENDER_MODES; any other value
    -- including non-str, unknown strings, empty string -- collapses
    to None.

    As of slice 5 (2026-06-29) the valid set is:
      - local_gap_coach            -- Layer B response mode
      - target_noc_standard        -- Layer A response mode
      - adjacent_noc_standard      -- Layer C response mode
      - adjacent_role_drilldown_select -- pending-select state used
        while the user is choosing which adjacent NOC to drill into

    Slice 5 step 2 invariant: a forged cookie cannot inject an
    arbitrary string that would route to a nonexistent recommender
    flow."""
    if isinstance(value, str) and value in _VALID_RECOMMENDER_MODES:
        return value
    return None


_MAX_LAST_ADJACENT_NOCS: int = 3


def _sanitize_last_adjacent_nocs(value: Any) -> tuple[str, ...]:
    """Defensive-deserialize last_adjacent_nocs. Returns a tuple of
    exact-5-digit NOC code strings, capped at _MAX_LAST_ADJACENT_NOCS.
    Drops any entry that:
      - is not a string
      - is not exactly 5 characters after strip
      - has any non-digit character

    A forged cookie cannot inject malformed NOC codes that would
    fail the Layer C SQL fetch or route to nonexistent
    occupations. Slice 5 step 4 invariant; mirrors the validation
    in gap_evidence._is_valid_noc_code so the data shape matches
    what Layer C expects to read."""
    if not isinstance(value, (list, tuple)):
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for entry in value:
        if not isinstance(entry, str):
            continue
        code = entry.strip()
        if len(code) != 5 or not code.isdigit():
            continue
        if code in seen:
            continue
        seen.add(code)
        out.append(code)
        if len(out) >= _MAX_LAST_ADJACENT_NOCS:
            break
    return tuple(out)


# Slice 5 (2026-06-29): cap + sanitizer for the recommender's
# adjacent-surface snapshot. Mirrors _MAX_LAST_ADJACENT_NOCS.
_MAX_RECOMMENDER_ADJACENT_SURFACE: int = 3
_MAX_SURFACE_TITLE_CHARS: int = 80


def _sanitize_last_recommender_adjacent_surface(value: Any) -> tuple[dict, ...]:
    """Defensive-deserialize last_recommender_adjacent_surface.

    Returns a tuple of dicts {"noc_code", "title"}, capped at
    _MAX_RECOMMENDER_ADJACENT_SURFACE. Drops any entry that:
      - is not a dict
      - has no string noc_code OR a noc_code that fails the 5-digit
        all-digit validation
      - has no string title OR an empty title

    Cookie protection: a forged cookie cannot inject malformed
    entries that would route the next-turn drilldown resolver to a
    nonexistent NOC. Same validation discipline as
    _sanitize_last_adjacent_nocs.
    """
    if not isinstance(value, (list, tuple)):
        return ()
    out: list[dict] = []
    seen: set[str] = set()
    for entry in value:
        if not isinstance(entry, dict):
            continue
        noc = entry.get("noc_code")
        title = entry.get("title")
        if not isinstance(noc, str) or not isinstance(title, str):
            continue
        code = noc.strip()
        if len(code) != 5 or not code.isdigit():
            continue
        title_stripped = title.strip()
        if not title_stripped:
            continue
        if code in seen:
            continue
        seen.add(code)
        out.append({
            "noc_code": code,
            "title": title_stripped[:_MAX_SURFACE_TITLE_CHARS],
        })
        if len(out) >= _MAX_RECOMMENDER_ADJACENT_SURFACE:
            break
    return tuple(out)


# Slice 1 follow-up (2026-06-23): deferred career intents we accept.
# Mirrors the CareerIntent literal in recommender_intent.py minus
# the values that are never deferrable: `job_matching` (matching
# engine has its own intake) and `unclear` (no intent to defer).
_VALID_DEFERRED_CAREER_INTENTS: frozenset[str] = frozenset({
    "local_skill_gap",
    "training_recommendation",
    "noc_standard_comparison",
    "career_exploration",
    "application_help_out_of_scope",
})


def _sanitize_deferred_career_intent(value: Any) -> str | None:
    """Defensive-deserialize deferred_career_intent. Returns the
    string if it's one of the deferrable CareerIntent literal
    values; None otherwise.

    A forged cookie cannot inject an arbitrary intent string that
    the router would then act on -- only the closed set above is
    accepted. Mirrors the validation pattern in
    _sanitize_pending_recommender_offer.
    """
    if not isinstance(value, str):
        return None
    if value not in _VALID_DEFERRED_CAREER_INTENTS:
        return None
    return value


def _sanitize_adjacent_snapshot(value: Any) -> dict[str, Any] | None:
    """Defensive-deserialize last_adjacent_snapshot.

    Shape (see StagedProfile.last_adjacent_snapshot docstring):
      - created_message_count: non-negative int (else drop snapshot)
      - items: list of dicts, max MAX_ADJACENT_ITEMS, each requiring
        non-empty job_id + title; why_adjacent must be in
        _VALID_WHY_ADJACENT or it's coerced to "" so the renderer
        falls through to a deterministic label; matched_skills is
        capped MAX_MATCHED_SKILLS × MAX_SKILL_CHARS.

    Whole-snapshot drop on top-level shape failure; per-item drop on
    item shape failure; never crash.
    """
    if not isinstance(value, dict):
        return None
    created = value.get("created_message_count")
    # `bool` is a subclass of `int`, so `isinstance(True, int)` is True.
    # Reject booleans explicitly so a forged blob with
    # `created_message_count=True` cannot resolve the TTL on the
    # immediately-following turn (True + 1 == 2 by coincidence).
    if not isinstance(created, int) or isinstance(created, bool) or created < 0:
        return None
    raw_items = value.get("items")
    if not isinstance(raw_items, list):
        return None

    out_items: list[dict[str, Any]] = []
    for it in raw_items[:MAX_ADJACENT_ITEMS]:
        if not isinstance(it, dict):
            continue
        job_id = _truncate(it.get("job_id"), MAX_JOB_ID_CHARS)
        title = _truncate(it.get("title"), MAX_TITLE_CHARS)
        if not job_id or not title:
            continue
        evidence = _truncate(it.get("evidence_summary"), MAX_EVIDENCE_CHARS)
        why_raw = it.get("why_adjacent")
        why = why_raw if isinstance(why_raw, str) and why_raw in _VALID_WHY_ADJACENT else ""
        raw_matched = it.get("matched_skills")
        matched: list[str] = []
        if isinstance(raw_matched, list):
            for m in raw_matched[:MAX_MATCHED_SKILLS]:
                if isinstance(m, str) and m:
                    matched.append(m[:MAX_SKILL_CHARS])
        out_items.append({
            "job_id": job_id,
            "title": title,
            "evidence_summary": evidence,
            "why_adjacent": why,
            "matched_skills": matched,
        })

    return {
        "created_message_count": created,
        "items": out_items,
    }


def shift_adjacent_snapshot_ttl(staged: "StagedProfile") -> None:
    """Defensive state helper. Shifts last_adjacent_snapshot's
    `created_message_count` forward by 1 so the ordinal-followup TTL
    (`current_message_count == created + 1`) survives a single
    scope-violation digression.

    Mutates the StagedProfile in place; safe no-op when no snapshot is
    live or its created_message_count is malformed. Called from
    `_try_v2_path` on scope-violation turns BEFORE any hook or dispatch
    so it survives every downstream branch (including fallback_to_legacy
    at handler.py:848). AR-1 ships the helper; AR-6 wires the call site.
    """
    snap = staged.last_adjacent_snapshot
    if not isinstance(snap, dict):
        return
    created = snap.get("created_message_count")
    # `bool` is a subclass of `int`; the shifter must reject booleans
    # so a malformed `created_message_count=True` cannot be silently
    # mutated into True + 1 == 2. Negative values are also rejected
    # to match the `_sanitize_adjacent_snapshot` contract (`created
    # < 0` is malformed at the cookie boundary).
    if (
        not isinstance(created, int)
        or isinstance(created, bool)
        or created < 0
    ):
        return
    snap["created_message_count"] = created + 1
    staged.last_adjacent_snapshot = snap
