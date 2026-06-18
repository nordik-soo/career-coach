"""Truth summary -- the deterministic input the chat planner consumes.

Chat orchestration v2 slice 1. See docs/chat-orchestration-v2-design.md
section 3 for the design. This module produces a compact dict that
captures every fact the planner is allowed to reason about: profile
shape, resume parse quality, intent signals, match readiness, scope
violations. No LLM calls here -- this is pure Python.

The truth layer is what stays strict in v2: any fact the planner uses
must be computed by this module from existing staged-profile / chat
context, never inferred by the LLM. The LLM then decides the
conversational MOVE from this fixed substrate.

Key contract: `enough_to_match` is computed here, NOT by the planner.
The threshold logic lives in Python so it's testable and tunable
without prompt edits. See design doc §3 for the full rule.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from skillbridge.session.staging import StagedProfile


# =========================================================================
# Closed enums -- planner reads these, must not invent values outside them
# =========================================================================
ResumeParseQuality = Literal[
    "no_resume",       # user never uploaded
    "full",            # skills + work_history + (education or certs)
    "skills_only",     # skills present, work_history empty
    "work_only",       # work_history present, skills empty (rare)
    "partial",         # at least one fact group present, but incomplete
    "failed",          # upload happened but parse_warning fired (scan, etc.)
]

UserIntentSignal = Literal[
    "asking_question",     # user asked a question (?, "what about", "tell me")
    "asking_about_gap",    # "how do I get 310T", "where can I learn welding"
    "impatient_proceed",   # "show jobs", "see my cv", "just match", "same role"
    "declining",           # "no", "skip", "not now"
    "confirming",          # "yes", "alright", "good", "looks right"
    "correcting",          # "actually", "no I meant", "wait"
    "redirecting",         # user changes topic
    "neutral",             # default when no strong signal
]

TargetRoleSpecificity = Literal[
    "none",        # target_role_text unset
    "vague",       # "any job", "something local", short non-specific
    "specific",    # named role with role-domain words
]


# =========================================================================
# Truth summary data shape
# =========================================================================
@dataclass
class ResumeFactsSummary:
    """Counts only -- not the verbatim facts. Planner doesn't need
    the source data; it needs to know whether there's enough to act on."""
    skill_count: int = 0
    work_history_count: int = 0
    certifications_count: int = 0
    education_count: int = 0


@dataclass
class TruthSummary:
    """The full deterministic input the planner sees. Built once per
    turn from the staged profile + the current user message.

    See docs/chat-orchestration-v2-design.md §3 for the field-by-field
    contract. The serialized JSON form is what gets fed to Haiku (slice 3).
    """
    # The current turn's user message and prior assistant context
    user_message: str
    last_assistant_move: str | None = None
    last_asked_slot: str | None = None
    message_count: int = 0

    # Resume layer
    resume_uploaded: bool = False
    resume_parse_quality: ResumeParseQuality = "no_resume"
    resume_facts_summary: ResumeFactsSummary = field(default_factory=ResumeFactsSummary)

    # Profile / intake state
    target_role_text: str | None = None
    target_role_specificity: TargetRoleSpecificity = "none"
    work_type_preference: str | None = None
    filled_slots: list[str] = field(default_factory=list)
    declined_slots: list[str] = field(default_factory=list)

    # Match-readiness (deterministic, NOT planner-inferred)
    enough_to_match: bool = False
    enough_to_match_reason: str | None = None
    usable_evidence_present: bool = False
    missing_critical: list[str] = field(default_factory=list)

    # Fresh-intake-on-target-change pillar (2026-06-15).
    #
    # `skills_aligned_with_target` is True when staged.skills are
    # evidence collected for the CURRENT target_role_text (or when no
    # target is set yet, i.e. cold profile). False when the user
    # switched target mid-session and skills still point at the prior
    # target — that's the signal forcing fresh intake.
    #
    # `experience_aligned_with_target` mirrors the same for
    # staged.experience_text. The locked design says the engine needs
    # BOTH aligned before running; the arbiter pass 1 fires
    # ask_one_clarifying_question on the first misaligned slot.
    #
    # `target_alignment_ok` is the AND of both — convenience boolean
    # the arbiter reads. Default True so cold-profile / no-target paths
    # don't accidentally trigger the gate.
    skills_aligned_with_target: bool = True
    experience_aligned_with_target: bool = True
    target_alignment_ok: bool = True
    # Diagnostic: which slot to ask first when alignment fails.
    # "skills_text", "experience_text", or None.
    target_alignment_first_misaligned_slot: str | None = None

    # Match state (populated by the arbiter pass 2 -- starts empty here)
    match_count: int = 0
    best_match_band: str | None = None
    caps_applied: list[str] = field(default_factory=list)

    # User intent (deterministic via regex/keyword match, NOT LLM)
    user_intent_signal: UserIntentSignal = "neutral"

    # Scope violations (already detected by existing handler invariants)
    scope_violations_detected: list[str] = field(default_factory=list)

    # Canonical names of training-registry gaps detected in the current
    # user_message. Populated by the handler before calling
    # build_truth_summary. Used by the planner as a routing signal --
    # NOT an unconditional override (the user might say "I have Excel,
    # find me jobs" where Excel is a claimed skill, not a training
    # request). The planner combines this with user_intent_signal to
    # decide explain_gap vs proceed_to_match vs ask.
    registry_gaps_in_message: list[str] = field(default_factory=list)

    def to_planner_json(self) -> dict[str, Any]:
        """Serialize for the planner prompt. Keep keys snake_case, values
        primitive types (str / int / bool / list of those). No nested
        objects beyond resume_facts_summary so the planner prompt stays
        token-small."""
        return {
            "user_message": self.user_message,
            "last_assistant_move": self.last_assistant_move,
            "last_asked_slot": self.last_asked_slot,
            "message_count": self.message_count,
            "resume_uploaded": self.resume_uploaded,
            "resume_parse_quality": self.resume_parse_quality,
            "resume_facts_summary": {
                "skill_count": self.resume_facts_summary.skill_count,
                "work_history_count": self.resume_facts_summary.work_history_count,
                "certifications_count": self.resume_facts_summary.certifications_count,
                "education_count": self.resume_facts_summary.education_count,
            },
            "target_role_text": self.target_role_text,
            "target_role_specificity": self.target_role_specificity,
            "work_type_preference": self.work_type_preference,
            "filled_slots": self.filled_slots,
            "declined_slots": self.declined_slots,
            "enough_to_match": self.enough_to_match,
            "enough_to_match_reason": self.enough_to_match_reason,
            "usable_evidence_present": self.usable_evidence_present,
            "missing_critical": self.missing_critical,
            "skills_aligned_with_target": self.skills_aligned_with_target,
            "experience_aligned_with_target": self.experience_aligned_with_target,
            "target_alignment_ok": self.target_alignment_ok,
            "target_alignment_first_misaligned_slot": (
                self.target_alignment_first_misaligned_slot
            ),
            "match_count": self.match_count,
            "best_match_band": self.best_match_band,
            "caps_applied": self.caps_applied,
            "user_intent_signal": self.user_intent_signal,
            "scope_violations_detected": self.scope_violations_detected,
            "registry_gaps_in_message": self.registry_gaps_in_message,
        }


# =========================================================================
# User intent classification -- regex/keyword, same alias-map discipline
# =========================================================================
# Each pattern set is curated and small. Add to it ONLY when real chats
# expose a miss -- not speculatively. The classifier returns the first
# match in evaluation order; ordering reflects "which signal we should
# treat as primary when multiple keywords appear."

_IMPATIENT_PATTERNS = (
    r"\b(show( me)? (?:jobs?|matches?|results?|what you('ve)? got))\b",
    r"\b(see (my )?(cv|resume))\b",
    r"\b(just (?:show|match|find))\b",
    r"\b(same role)\b",
    r"\b(same field)\b",
    r"\b(let'?s go)\b",
    r"\b(go ahead)\b",
    r"\b(match (?:me )?now)\b",
)

_DECLINING_PATTERNS = (
    r"\b(no thanks?|nope|not (?:now|today|interested))\b",
    r"\b(skip( it)?)\b",
    r"\b(i (?:don'?t|do not) want)\b",
)

_CONFIRMING_PATTERNS = (
    r"^(yes|yep|yeah|yup|ya|sure|ok(?:ay)?|alright|sounds good|that'?s right|looks (?:right|good))\b",
    r"\b(confirmed?|correct|exactly|that works)\b",
)

_CORRECTING_PATTERNS = (
    r"\b(actually|wait|no i meant|i meant to say|sorry,? i meant)\b",
    r"\b(that'?s not|not quite|not really)\b",
)

_REDIRECTING_PATTERNS = (
    r"\b(change of plans?|nevermind|forget that)\b",
)

# Slice 9: gap-help questions. The bug-driving turn was after matches
# were shown with a credential cap; the user asked "310T technician
# certification, how can get this" -- a question about HOW to acquire
# a credential, not a request to see jobs again. Without this signal,
# the planner saw enough_to_match=True and routed back to proceed_to_match,
# re-rendering the same match cards. These patterns let the planner
# route to explain_gap instead.
#
# Conservative coverage: only patterns that clearly mean "tell me how
# to acquire / where to study / what training / do I need X / what is
# this cert". Pure "what about X" remains in _QUESTION_PATTERNS.
_ASKING_ABOUT_GAP_PATTERNS = (
    # "how do/can/to (I) get/earn/obtain/acquire/learn/find/study/improve/build/upgrade/develop"
    # 'improve|build|upgrade|develop' added so "how can improve excel" routes correctly.
    r"\bhow (do|can|to)( i)? (get|earn|obtain|acquire|learn|find|study(?:\s+for)?|prepare for|qualify for|complete|finish|take|improve|build|upgrade|develop)\b",
    # "where do/can/to (I) get/learn/study/take/enroll/sign up/apply for ..."
    r"\bwhere (do|can|to)( i)? (get|earn|obtain|learn|study|find|take|enroll|sign up|apply for)\b",
    # "how to X" forms (no leading I)
    r"\bhow to (get|learn|obtain|acquire|earn|study|complete|finish|prepare for|qualify for|take|enroll|improve|build|develop)\b",
    # "what (is|are|does) X (certification|certificate|credential|license|licence|cert|ticket)"
    r"\bwhat (is|are|does)\b.*\b(certification|certificate|credential|licen[sc]e|ticket|cert)\b",
    # Capability questions about a specific item -- "do I need X", "is X required"
    r"\bdo i (need|require|have to (?:get|have))\b",
    r"\bis (a |an |the )?[a-z0-9]+ (required|needed|necessary|mandatory)\b",
    # Casual phrasings the live test surfaced ("how can get", missing "I")
    r"\bhow (can|do) get\b",
    # "what training/courses/programs/classes/certifications/requirements ..."
    r"\bwhat (training|courses?|programs?|classes|certifications?|requirements?|tickets?)\b",
    # "how long / how much (time) to get/complete/earn the cert"
    r"\bhow (long|much (?:time|longer))\b.*\b(get|complete|earn|finish|take|obtain|qualify)\b",
    # "how (can|do) (I) improve" -- I now optional ("how can improve excel skill")
    r"\bhow (can|do)( i)? improve\b",
    # "(can|how) (i) upgrade/build" -- I now optional
    r"\b(can|how)( i)? (upgrade|build)\b",
    # ----- post-live-test additions -----
    # "do/can/would you recommend ..." (recommendation request)
    r"\b(do|can|would) you recommend\b",
    # "any/some/which/what (training|courses|programs|classes|certifications)"
    r"\b(any|some|which|what) (training|courses?|programs?|classes|certifications?)\b",
    # "(course|training|program|class) (to|for) (improve|build|learn|...)"
    r"\b(courses?|training|programs?|classes?) (to|for) (improve|build|learn|develop|upgrade|study|take|complete|earn|finish)\b",
    # "give me (a|the)? link/course/training" -- direct ask for a resource
    r"\bgive me (a |the )?(link|course|training|program)\b",
    # "I want to learn/improve/build/develop/upgrade/study"
    r"\bi want to (learn|improve|build|develop|upgrade|study|take|enroll)\b",
    # "online course/training/class/program" or "free course/..." -- resource request
    r"\b(online|free) (course|training|class|program)\b",
)


_QUESTION_PATTERNS = (
    r"\?",
    r"^(what|why|when|where|how|who|which|tell me|explain|can you|could you)\b",
)


def _classify_intent(message: str) -> UserIntentSignal:
    """Single-pass regex classifier. Cheap; runs once per turn."""
    if not message:
        return "neutral"
    m = message.strip().lower()
    if not m:
        return "neutral"

    # Order matters: impatient + confirming can both fire on "alright,
    # let's go". We treat impatient as the stronger signal (the user
    # wants action), so check it first.
    for pat in _IMPATIENT_PATTERNS:
        if re.search(pat, m):
            return "impatient_proceed"
    for pat in _DECLINING_PATTERNS:
        if re.search(pat, m):
            return "declining"
    for pat in _CORRECTING_PATTERNS:
        if re.search(pat, m):
            return "correcting"
    for pat in _REDIRECTING_PATTERNS:
        if re.search(pat, m):
            return "redirecting"
    for pat in _CONFIRMING_PATTERNS:
        if re.search(pat, m):
            return "confirming"
    # Slice 9: asking_about_gap is more specific than asking_question, so
    # it's checked first. The planner uses this signal to route to
    # explain_gap instead of proceed_to_match after matches have been
    # shown.
    for pat in _ASKING_ABOUT_GAP_PATTERNS:
        if re.search(pat, m):
            return "asking_about_gap"
    for pat in _QUESTION_PATTERNS:
        if re.search(pat, m):
            return "asking_question"
    return "neutral"


# =========================================================================
# Resume parse quality classification
# =========================================================================
def _classify_resume_parse_quality(
    staged: StagedProfile,
) -> tuple[ResumeParseQuality, ResumeFactsSummary]:
    """Inspect resume_facts_json + parse warnings to bucket the resume
    extractor's output. Returns the bucket plus per-group counts.

    No verbatim facts leak into the summary -- just counts. The planner
    decides intent from counts; the responder gets the verbatim facts
    separately via the existing RESUME_FACTS prompt block.

    Failure modes the classifier distinguishes:
      - "failed": parser-level failure (too_large / unsupported / no_text /
        empty / parse_failed). Surfaced via staged.resume_parse_warning,
        set by handler._apply_resume_upload's early-return paths.
      - "failed" (post-extraction): text extracted but the LLM facts
        extractor returned an empty payload. Detected via filename set
        but every fact group empty.
      - "skills_only" / "work_only" / "partial" / "full": various
        degrees of successful extraction.
    """
    # Parser-level failure persists across turns via resume_parse_warning.
    # Catch it before anything else so a scanned PDF on turn N still reads
    # as "failed" on turn N+1, not "no_resume".
    if staged.resume_parse_warning:
        return "failed", ResumeFactsSummary()

    facts = staged.resume_facts_json
    if not facts:
        return "no_resume", ResumeFactsSummary()

    skills = facts.get("skills") or []
    work_history = facts.get("work_history") or []
    certifications = facts.get("certifications") or []
    education = facts.get("education") or []
    counts = ResumeFactsSummary(
        skill_count=len(skills),
        work_history_count=len(work_history),
        certifications_count=len(certifications),
        education_count=len(education),
    )

    # Post-extraction empty: file was uploaded, text extracted, but
    # the LLM facts extractor returned nothing structured. Same bucket
    # as parser-level failure because the downstream effect (no
    # usable evidence) is identical.
    total_facts = (counts.skill_count + counts.work_history_count
                   + counts.certifications_count + counts.education_count)
    if staged.resume_filename and total_facts == 0:
        return "failed", counts

    # Full = at least skills + work_history + (education OR certs)
    if (counts.skill_count >= 3
            and counts.work_history_count >= 1
            and (counts.education_count >= 1 or counts.certifications_count >= 1)):
        return "full", counts

    # Skills-only is the case we saw in live test 3 -- big skill list,
    # zero work history. Worth a dedicated bucket because the planner
    # may decide to proceed anyway.
    if counts.skill_count >= 3 and counts.work_history_count == 0:
        return "skills_only", counts

    # Work-only is rare but possible (some resumes lead with experience
    # narrative and have no flat skills list). Also worth a bucket.
    if counts.work_history_count >= 1 and counts.skill_count == 0:
        return "work_only", counts

    # Anything else with at least one fact -> partial.
    return "partial", counts


# =========================================================================
# Target-role specificity classification
# =========================================================================
# Short non-specific tokens that on their own don't pick a direction.
# Includes work-type vocabulary ("full", "time", "part", "flexible",
# "contract") defensively: if the slot extractor ever misroutes work-type
# preferences into target_role_text, "full time" alone shouldn't mark as
# a specific target. Real role phrases like "part-time pharmacist" still
# classify as specific because "pharmacist" is outside this set.
_VAGUE_TARGET_TOKENS = {
    "any", "anything", "something", "job", "jobs", "work", "role",
    "career", "employment", "local", "general", "various",
    "full", "time", "part", "flexible", "contract",
    # Slice (2026-06-08): anaphoric / pronominal phrases. Without
    # these, "same role" / "current job" / "this" tokenized to one
    # vague + one non-vague token, leaving a non-empty `significant`
    # set, producing specificity=specific from a pure anaphor. The
    # handler-level resolver tries to swap the anaphor for the
    # resume's current title before this classifier sees it; this
    # set is the safety net for any anaphor that survives without
    # being resolved.
    "same", "current", "previous", "prior", "past",
    "this", "that", "it",
    "position", "field", "kind",
    # Articles -- never carry role meaning on their own. "the same"
    # without this entry tokenized to {the, same} where only "same"
    # was vague, leaving {the} as significant => specific.
    "the", "a", "an",
}


def _classify_target_role(text: str | None) -> TargetRoleSpecificity:
    if not text:
        return "none"
    cleaned = text.strip().lower()
    if not cleaned:
        return "none"
    tokens = {t for t in re.split(r"[^a-z0-9+]+", cleaned) if t}
    if not tokens:
        return "vague"
    # If every non-stopword token is in the vague set, it's vague.
    significant = tokens - _VAGUE_TARGET_TOKENS
    if not significant:
        return "vague"
    # Otherwise the user named a real role.
    return "specific"


# =========================================================================
# enough_to_match -- the load-bearing deterministic gate
# =========================================================================
def _compute_enough_to_match(
    target_role_specificity: TargetRoleSpecificity,
    resume_parse_quality: ResumeParseQuality,
    counts: ResumeFactsSummary,
    chat_skill_count: int,
    user_intent_signal: UserIntentSignal,
    *,
    skills_text_present: bool = False,
) -> tuple[bool, str | None, bool]:
    """See design doc §3. Returns (enough, reason_code, usable_evidence).

    `enough` is true when the engine has a real chance of producing
    meaningful matches. The threshold is tunable in Python; the
    planner just consumes the boolean.

    `usable_evidence_present` is the guard against the failure mode
    flagged in Step 5 review (failed-scan resume + impatient user +
    target role would otherwise mark enough=true with no actual
    evidence).

    Change C (no-final-no-without-resume rule, 2026-06-16):
        `skills_text_present` is the new explicit-skills guard. The
        prior `chat_skill_count >= 3` branch fired whenever the
        extractor scraped any 3 skill-like tokens, INCLUDING phantom
        skills lifted from experience prose ("Completed Truck and
        Coach Technician apprenticeship at Sault College" yields
        "truck and coach technician" / "Sault College" as skill-
        shaped tokens). That tripped enough_to_match=True on what
        the user clearly intended as experience-only input, the
        engine ran prematurely, and the upload offer fired on the
        wrong turn — leaving subsequent skill claims to render the
        canned legacy template. With `skills_text_present` required,
        the engine waits until the user has explicitly stated skills
        (skills_text slot non-empty) AND the chat skill count is at
        threshold — phantom skills alone no longer suffice.
        Resume-skills and resume-work-history paths are unchanged:
        those are objective resume evidence, not chat scraping.
    """
    # Step 1: usable evidence guard. Failed parses without chat-derived
    # skills don't count as evidence even if the user is impatient.
    usable_evidence_present = (
        resume_parse_quality not in {"failed", "no_resume"}
        or chat_skill_count >= 3
    )

    # Step 2: target role specificity check.
    #
    # CP3 step 1 (2026-06-15): a user with strong skill evidence and NO
    # specific target should still match. Real users don't always
    # phrase a target role ("what jobs match my QuickBooks + payroll
    # skills?" is a legitimate query). When skill evidence clears the
    # same threshold we use for chat_skills_sufficient OR the resume-
    # skills threshold, run the engine in skills-only mode — the engine
    # scores by skill overlap regardless of target_noc, so no NOC
    # anchor is required.
    if target_role_specificity != "specific":
        # A1 (2026-06-18): missing/vague target now defaults to "ask"
        # for the target. Prior behavior ran the engine in skills-only
        # mode on any neutral/confirming turn with decent evidence,
        # which produced CP4 silent failures (no target NOC -> no
        # recommendation possible) and cascading degradation in the
        # closing rules. The skills-only path is preserved ONLY when
        # the user EXPLICITLY asked to skip target-setting ("show
        # jobs based on my skills", "see my cv", "just match me") --
        # captured by the existing `impatient_proceed` intent.
        #
        # Caution for Slice B/C: `impatient_proceed` is broad. It also
        # matches "go ahead", "let's go", "same role". For A1 that is
        # acceptable because the branch only fires when the target is
        # missing/vague AND evidence is strong -- a narrow combination.
        # If live tests show false positives (user got matched without
        # really wanting skills-only mode), B/C should introduce a
        # tighter explicit-skills-only signal.
        if user_intent_signal == "impatient_proceed" and (
            chat_skill_count >= 3 or counts.skill_count >= 5
        ):
            return True, "skills_only_explicit_request", usable_evidence_present
        return False, "missing_target", usable_evidence_present

    # Step 3: evidence guard must pass.
    if not usable_evidence_present:
        return False, "no_usable_evidence", usable_evidence_present

    # Step 4: one of the evidence thresholds must clear. Each branch
    # has a distinct reason_code so the arbiter and tests can switch.
    if counts.skill_count >= 5:
        return True, "resume_skills_sufficient", usable_evidence_present
    # Change C (2026-06-16): chat-skills branch now ALSO requires the
    # `skills_text` slot to be explicitly filled. Phantom skills
    # scraped from experience prose no longer trigger engine-run.
    # The engine waits until the user has CLAIMED skills, not just
    # described their work history.
    if chat_skill_count >= 3 and skills_text_present:
        return True, "chat_skills_sufficient", usable_evidence_present
    if counts.work_history_count >= 1:
        return True, "resume_work_history_present", usable_evidence_present
    if user_intent_signal == "impatient_proceed":
        return True, "user_explicitly_asked_to_match", usable_evidence_present

    return False, "insufficient_skill_evidence", usable_evidence_present


# =========================================================================
# Training-action vocabulary -- broad signal for "user wants training"
# =========================================================================
# Combined with a registry-entity match in the same message, this is
# the decisive cold-session training-intent signal. The rule is
# applied AFTER base intent classification in `build_truth_summary`,
# only upgrading neutral/asking_question -- never overriding stronger
# signals (impatient, declining, etc.).
#
# Design principle: entity alone is not enough (could be a skill
# claim: "I have Excel"); training words alone are not enough (no
# specific gap to recommend for). Both required.
#
# The vocabulary is broad on purpose -- it covers the natural range
# of phrasings without per-phrase pattern bloat. New entries are
# cheap; the entity guard prevents false positives.
_TRAINING_ACTION_WORDS: frozenset[str] = frozenset({
    # Direct training nouns
    "course", "courses", "training", "trainings",
    "class", "classes", "program", "programs",
    "tutorial", "tutorials", "workshop", "workshops",
    # Credential nouns
    "certificate", "certificates", "certification", "certifications",
    "licence", "license", "licences", "licenses",
    "ticket", "tickets",
    # Acquisition verbs (overlap with intent patterns is fine -- this
    # is the broader-coverage layer)
    "learn", "improve", "upgrade", "build", "study", "develop",
    "earn", "obtain", "acquire", "complete", "finish", "qualify",
    # Resource-request words
    "link", "recommend", "suggest",
    "online", "free",
    # Enrollment / next-step verbs
    "enroll", "register",
})


def _has_training_action_words(message: str) -> bool:
    """True when the message contains any token from the training-action
    vocabulary. Tokenized on word boundaries, case-folded.
    Combined with registry_gaps_in_message to upgrade ambiguous intent
    (neutral / asking_question) to asking_about_gap. Both signals
    required -- pure entity ("I have Excel") or pure action word
    ("any free thing") alone is not enough."""
    if not message:
        return False
    tokens = re.findall(r"[a-z]+", message.lower())
    return any(t in _TRAINING_ACTION_WORDS for t in tokens)


# =========================================================================
# Scope violation detection (Slice 10)
# =========================================================================
# Pre-Slice-10 the arbiter's `scope_violations_detected non-empty ->
# redirect_scope` rule (Slice 4) was correct architecturally but
# practically dead -- nothing populated the signal. Live test
# surfaced this: a user said "this job can help for PR" mid-conversation
# and the system re-pitched the same matches because no scope flag
# fired. Slice 10 closes that gap by detecting scope-relevant keywords
# in the user message itself.
#
# Tags emitted match the strings the arbiter's `_scope_reason_code`
# helper maps to planner reason codes (immigration, national_wages,
# non_ssm_city, off_topic).
#
# Coverage rule: catch STATEMENTS and QUESTIONS alike. The bug-driving
# input "this job can help for PR" is declarative, not interrogative.
# Word-boundary regexes don't care about ? vs . -- same pattern fires
# either way.
#
# Bias: prefer false-positive redirects over false-negative leaks.
# Redirecting a borderline case is safe (we just point at SCCC); not
# redirecting an immigration question crosses the SCOPE_BOUNDARIES rule.

_SCOPE_IMMIGRATION_PATTERNS = (
    # Permanent residence (bare "PR" is the live bug case)
    r"\bpr\b",
    r"\bpermanent residenc[ey]",
    r"\bpr (?:application|status|eligib(?:le|ility)|sponsor)",
    # Express Entry
    r"\bexpress entry",
    # Provincial / regional immigration programs
    r"\brcip\b",                            # Rural and Northern Immigration Pilot
    r"\bpnp\b",                             # Provincial Nominee Program
    # Federal body
    r"\bircc\b",
    # Permit / visa (immigration context only -- avoid Visa-card collisions)
    r"\bwork permit\b",
    r"\bopen work permit\b",
    r"\bclosed work permit\b",
    r"\bstudy permit\b",
    r"\bvisa (?:application|status|holder|sponsor|category)\b",
    r"\bwork visa\b",
    r"\bstudy visa\b",
    r"\bvisitor visa\b",
    # Citizenship + status
    r"\bcitizenship",
    r"\blanded immigrant\b",
    r"\brefugee (?:status|claim|sponsor)\b",
    r"\basylum",
    # Newcomer-specific immigration framings
    r"\bsponsor(?:ed|ing)? (?:my )?(?:wife|husband|spouse|family|parent|child)\b",
    r"\bcoming to canada\b",
)

_SCOPE_NATIONAL_WAGES_PATTERNS = (
    # National-feed references the product explicitly does NOT use.
    r"\bnational average\b",
    r"\bstatcan\b",
    r"\bstatistics canada\b",
    r"\bjob bank\b",                        # the federal feed, not the chat
    r"\bcanadian average (?:wage|salary|pay)\b",
    r"\bnational (?:wage|salary|pay) (?:average|rate)\b",
)

_NON_LOCAL_CITY_RE = (
    r"(toronto|ottawa|sudbury|thunder bay|north bay|timmins|kingston|"
    r"hamilton|kitchener|mississauga|london|windsor|waterloo|barrie|"
    r"greater toronto|gta)"
)

# Mirror the responder's existing _NON_LOCAL_CITY_OFFER_PATTERNS shape:
# fire only when the city appears with an "action" verb that means
# "go there for work." Bare references like "University of Toronto"
# don't fire.
_SCOPE_NON_LOCAL_CITY_PATTERNS = (
    rf"\b(?:any|more|find|look(?:ing)? for|search(?:ing)? for|need|want) (?:a )?(?:jobs?|work|positions?|openings?|opportunities)\s+(?:in|near|around|at)\s+(?:the )?{_NON_LOCAL_CITY_RE}",
    rf"\b(?:move|moving|relocate|relocating|going) to (?:the )?{_NON_LOCAL_CITY_RE}",
    rf"\b(?:jobs?|work|positions?|openings?|opportunities) in (?:the )?{_NON_LOCAL_CITY_RE}",
    rf"\bcan i (?:find|get) (?:a )?(?:job|work) in (?:the )?{_NON_LOCAL_CITY_RE}",
    rf"\b(?:apply|applying) (?:to|in) (?:the )?{_NON_LOCAL_CITY_RE}",
)


def _detect_scope_violations(message: str) -> list[str]:
    """Return scope-violation tags surfaced by the user's text.

    The returned tags map to the planner's scope reason codes via
    `arbiter._scope_reason_code`:
        "immigration"   -> scope_violation_immigration
        "national_wages" -> scope_violation_wages
        "non_ssm_city"   -> scope_violation_non_ssm

    Returns an empty list when nothing matches. De-duped while
    preserving first-seen order so the arbiter's
    `scope_violations[0]` selection is deterministic.
    """
    if not message:
        return []
    lowered = message.lower()

    detected: list[str] = []

    def _check(tag: str, patterns: tuple[str, ...]) -> None:
        if tag in detected:
            return
        for pat in patterns:
            if re.search(pat, lowered):
                detected.append(tag)
                return

    _check("immigration", _SCOPE_IMMIGRATION_PATTERNS)
    _check("national_wages", _SCOPE_NATIONAL_WAGES_PATTERNS)
    _check("non_ssm_city", _SCOPE_NON_LOCAL_CITY_PATTERNS)

    return detected


def _normalize_target_role(value: str | None) -> str | None:
    """Casefold + collapse whitespace + strip. Returns None for falsy
    inputs. Used by the alignment helper so "Truck Driver" and
    "truck driver" compare equal, but "truck driver" and "long-haul
    truck driver" do NOT (different evidence required).
    """
    if not isinstance(value, str):
        return None
    norm = " ".join(value.split()).strip().casefold()
    return norm or None


def _compute_target_alignment(
    staged: StagedProfile,
) -> tuple[bool, bool, str | None]:
    """Fresh-intake-on-target-change pillar (2026-06-15).

    Returns (skills_aligned, experience_aligned, first_misaligned_slot).

    Alignment rule (per locked design):
      - target_role_text is None (cold profile, no target) →
        both aligned. The other intake gates still apply.
      - target_role_text is set AND the corresponding field's
        `*_collected_for_target` matches it (normalized) → aligned.
      - target_role_text is set AND the field's `*_collected_for_target`
        is None OR different normalized value → misaligned. The user
        switched target and the prior evidence still points at the
        prior target.

    `first_misaligned_slot` is "skills_text" if skills are misaligned,
    else "experience_text" if experience is misaligned, else None.
    Locked priority order: skills always asked first (primary signal).
    """
    target_norm = _normalize_target_role(staged.target_role_text)
    if target_norm is None:
        return True, True, None

    skills_for_target_norm = _normalize_target_role(
        staged.skills_collected_for_target
    )
    experience_for_target_norm = _normalize_target_role(
        staged.experience_collected_for_target
    )

    skills_aligned = skills_for_target_norm == target_norm
    experience_aligned = experience_for_target_norm == target_norm

    first_misaligned: str | None = None
    if not skills_aligned:
        first_misaligned = "skills_text"
    elif not experience_aligned:
        first_misaligned = "experience_text"
    return skills_aligned, experience_aligned, first_misaligned


# =========================================================================
# Public entry point
# =========================================================================
def build_truth_summary(
    *,
    staged: StagedProfile,
    user_message: str,
    last_assistant_move: str | None = None,
    last_asked_slot: str | None = None,
    resume_uploaded_this_turn: bool = False,
    scope_violations_detected: list[str] | None = None,
    registry_gaps_in_message: list[str] | None = None,
) -> TruthSummary:
    """The handler calls this once per turn, BEFORE the planner.

    All inputs are already-deterministic state from the existing
    handler (staged profile, prior turn's decision, the current user
    message). No LLM calls happen inside this function.
    """
    intent = _classify_intent(user_message)

    # Post-pattern intent upgrade: entity + training-action words.
    # When base intent is ambiguous (neutral / asking_question), we
    # check whether the message names a registry-known credential AND
    # uses any training-action word. Both signals together = decisive
    # training-intent. Catches the live failure case "online Excel
    # course" where the regex layer alone missed the wording.
    # Stronger intents (impatient_proceed, declining, etc.) are NOT
    # downgraded -- this only promotes from the two ambiguous classes.
    if intent in ("neutral", "asking_question"):
        rgim = list(registry_gaps_in_message or [])
        if rgim and _has_training_action_words(user_message):
            intent = "asking_about_gap"

    target_role_specificity = _classify_target_role(staged.target_role_text)
    resume_quality, counts = _classify_resume_parse_quality(staged)

    # "Resume uploaded" is sticky -- once a user has uploaded, every
    # turn from then on reports True. The handler may also flag
    # uploaded_this_turn (for the gate evaluation), but that's
    # separate from the persistent fact that a resume exists.
    resume_uploaded = bool(staged.resume_facts_json) or resume_uploaded_this_turn

    chat_skill_count = sum(
        1 for s in staged.skills if s.source != "resume"
    )

    # Change C (2026-06-16) — explicit-skills guard. `skills_text_present`
    # is True iff the user has filled the `skills_text` slot with a
    # non-empty string. Phantom skills lifted from experience prose
    # by the extractor don't populate this slot.
    skills_text_present = (
        isinstance(staged.skills_text, str) and bool(staged.skills_text.strip())
    )

    enough, reason, usable_evidence = _compute_enough_to_match(
        target_role_specificity=target_role_specificity,
        resume_parse_quality=resume_quality,
        counts=counts,
        chat_skill_count=chat_skill_count,
        user_intent_signal=intent,
        skills_text_present=skills_text_present,
    )

    # Fresh-intake-on-target-change pillar (2026-06-15) — compute the
    # alignment fields. The arbiter pass 1 reads these to force a
    # skills_text / experience_text intake before running the engine
    # when the user has switched target mid-session.
    (
        skills_aligned,
        experience_aligned,
        first_misaligned_slot,
    ) = _compute_target_alignment(staged)
    target_alignment_ok = skills_aligned and experience_aligned

    # Slice 10: scope-violation detection from the user message itself.
    # The handler can also pass server-side detections via
    # `scope_violations_detected`; we union both with caller-provided
    # tags first so they take precedence in `arbiter._scope_reason_code`'s
    # "first violation" selection.
    message_scope_violations = _detect_scope_violations(user_message)
    caller_scope_violations = list(scope_violations_detected or [])
    merged_scope_violations: list[str] = []
    for tag in caller_scope_violations + message_scope_violations:
        if tag not in merged_scope_violations:
            merged_scope_violations.append(tag)

    return TruthSummary(
        user_message=user_message,
        last_assistant_move=last_assistant_move,
        last_asked_slot=last_asked_slot,
        message_count=staged.message_count,
        resume_uploaded=resume_uploaded,
        resume_parse_quality=resume_quality,
        resume_facts_summary=counts,
        target_role_text=staged.target_role_text,
        target_role_specificity=target_role_specificity,
        work_type_preference=staged.work_type_preference,
        filled_slots=sorted(staged.filled_slots()),
        declined_slots=list(staged.declined_slots),
        enough_to_match=enough,
        enough_to_match_reason=reason,
        usable_evidence_present=usable_evidence,
        missing_critical=[],  # populated by handler when known
        match_count=0,         # populated by arbiter pass 2
        best_match_band=None,
        caps_applied=[],
        user_intent_signal=intent,
        scope_violations_detected=merged_scope_violations,
        registry_gaps_in_message=list(registry_gaps_in_message or []),
        skills_aligned_with_target=skills_aligned,
        experience_aligned_with_target=experience_aligned,
        target_alignment_ok=target_alignment_ok,
        target_alignment_first_misaligned_slot=first_misaligned_slot,
    )
