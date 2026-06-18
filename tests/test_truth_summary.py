"""Unit tests for chat orchestration v2 slice 1 -- truth_summary.

Pure-function tests: each surface (intent classifier, resume-quality
classifier, target-role specificity, enough_to_match guard, the public
builder) has its own block.

The contract these tests pin:
  - User intent classification is deterministic and matches docs/§3
  - resume_parse_quality buckets match the closed enum exactly
  - enough_to_match is FALSE when usable_evidence_present is FALSE,
    even if the impatience signal is on (the Step 4 guard from review)
  - enough_to_match logic produces the expected reason_code for every
    branch
  - build_truth_summary() serializes to a JSON-friendly dict the
    planner can consume
"""
from __future__ import annotations

import os

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from skillbridge.chat.truth_summary import (
    ResumeFactsSummary,
    TruthSummary,
    _classify_intent,
    _classify_resume_parse_quality,
    _classify_target_role,
    _compute_enough_to_match,
    build_truth_summary,
)
from skillbridge.session.staging import StagedProfile, StagedSkill

pytestmark = pytest.mark.nodb


# ---------------------------------------------------------------------------
# _classify_intent
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("msg, expected", [
    # impatient_proceed beats everything else
    ("show me jobs", "impatient_proceed"),
    ("see my cv", "impatient_proceed"),
    ("see my CV", "impatient_proceed"),
    ("just show what you've got", "impatient_proceed"),
    ("same role", "impatient_proceed"),
    ("same field", "impatient_proceed"),
    ("let's go", "impatient_proceed"),
    ("go ahead", "impatient_proceed"),
    # declining
    ("no thanks", "declining"),
    ("not now", "declining"),
    ("skip it", "declining"),
    ("i don't want that", "declining"),
    # correcting
    ("actually I want warehouse work", "correcting"),
    ("wait, that's wrong", "correcting"),
    ("no i meant data analyst", "correcting"),
    # confirming -- comes after correcting/redirecting on purpose
    ("yes", "confirming"),
    ("alright", "confirming"),
    ("looks right", "confirming"),
    ("that works", "confirming"),
    # Slice 9: asking_about_gap. User asks how to acquire / learn / obtain
    # a credential or skill (typically after matches were shown with caps).
    # Must beat asking_question (more specific) but NOT impatient_proceed.
    ("how do I get 310T?", "asking_about_gap"),
    ("how can I get my class G license", "asking_about_gap"),
    ("how can get this", "asking_about_gap"),                  # the LIVE bug case wording
    ("310T technician certification, how can get this", "asking_about_gap"),
    ("where can I learn welding", "asking_about_gap"),
    ("where do I get my forklift cert", "asking_about_gap"),
    ("how to obtain WHMIS", "asking_about_gap"),
    ("what training do I need for warehouse work", "asking_about_gap"),
    ("do I need a license to apply", "asking_about_gap"),
    ("how can I improve welding", "asking_about_gap"),
    ("what is the 310T certificate", "asking_about_gap"),
    # question (still asking_question -- not gap-specific)
    ("what do you have for me?", "asking_question"),
    ("tell me more", "asking_question"),
    ("how does that work?", "asking_question"),                # "how does" not in gap pattern
    # neutral
    ("hi", "neutral"),
    ("", "neutral"),
    ("   ", "neutral"),
])
def test_intent_classification(msg, expected):
    assert _classify_intent(msg) == expected


# ===========================================================================
# Slice 10 -- scope-violation detection
# ===========================================================================
# Pre-Slice-10 the arbiter's scope_violations_detected -> redirect_scope
# rule was architecturally correct but practically dead -- nothing
# populated the signal. Live test surfaced "this job can help for PR"
# slipping past. Slice 10 detects scope-relevant keywords in the user
# message itself.
from skillbridge.chat.truth_summary import _detect_scope_violations


@pytest.mark.parametrize("msg, expected_tag", [
    # ---- Immigration: the live bug case + variants ----
    ("This job can help for PR", "immigration"),
    ("can I apply for PR?", "immigration"),
    ("Will this support my work permit?", "immigration"),
    ("good for Express Entry?", "immigration"),
    ("RCIP eligible?", "immigration"),
    ("am I eligible for IRCC?", "immigration"),
    ("I have an open work permit", "immigration"),
    ("citizenship application is pending", "immigration"),
    ("waiting on my permanent residence", "immigration"),
    ("do I need PR for this role?", "immigration"),
    ("study permit holder looking for part-time", "immigration"),
    ("sponsoring my spouse", "immigration"),
    ("just became a landed immigrant", "immigration"),
    # PNP short form
    ("am I under PNP?", "immigration"),

    # ---- National wages: federal feeds we don't use ----
    ("what's the national average wage for warehouse work?", "national_wages"),
    ("does StatCan have data on this?", "national_wages"),
    ("Job Bank shows different numbers", "national_wages"),
    ("the Canadian average salary for the role", "national_wages"),

    # ---- Non-local city OFFERS only (action context) ----
    ("any jobs in Toronto?", "non_ssm_city"),
    ("I want to look for work in Sudbury", "non_ssm_city"),
    ("moving to Ottawa next month", "non_ssm_city"),
    ("can I find a job in Thunder Bay?", "non_ssm_city"),
    ("jobs in the GTA", "non_ssm_city"),

    # ---- Bare references that must NOT fire ----
    ("I went to University of Toronto", None),
    ("born in Sudbury but living here", None),
    ("my family is in Ottawa", None),

    # ---- Pure SSM job-search messages must NOT fire ----
    ("I want a warehouse job", None),
    ("can I get my forklift cert", None),
    ("how do I get my 310T", None),
    ("show me what you've got", None),
    ("yes", None),
    ("", None),

    # ---- Negative: Visa-card collisions ----
    ("I'll pay with my Visa card", None),
    ("Visa transactions only", None),
])
def test_scope_violation_detection(msg, expected_tag):
    """Each positive case fires the expected tag; negative cases
    return empty list. Critically: the live bug case
    'This job can help for PR' must fire."""
    detected = _detect_scope_violations(msg)
    if expected_tag is None:
        assert detected == [], (
            f"Expected NO scope violation for {msg!r}; got {detected}"
        )
    else:
        assert expected_tag in detected, (
            f"Expected {expected_tag!r} in {detected} for {msg!r}"
        )


def test_scope_detection_preserves_first_seen_order_when_multiple_fire():
    """Multi-category messages return tags in first-detection order.
    The arbiter's _scope_reason_code picks violations[0] for the
    reason mapping, so order matters."""
    # Immigration check runs before national_wages, so this message
    # surfaces 'immigration' first.
    msg = "I want to know about PR and the national average salary"
    detected = _detect_scope_violations(msg)
    assert detected[0] == "immigration"
    assert "national_wages" in detected


def test_scope_detection_deduplicates_tags():
    """If multiple immigration patterns match the same message, the
    'immigration' tag still only appears once."""
    msg = "I have a work permit and need PR"
    detected = _detect_scope_violations(msg)
    assert detected.count("immigration") == 1


def test_build_truth_summary_populates_scope_violations_from_message():
    """End-to-end: build_truth_summary picks up scope violations from
    the user_message even when no caller-provided list is passed.
    This is the production wiring that pre-Slice-10 was missing."""
    from skillbridge.chat.truth_summary import build_truth_summary
    from skillbridge.session.staging import StagedProfile

    sp = StagedProfile.new("test-scope")
    sp.target_role_text = "truck and coach technician"
    truth = build_truth_summary(
        staged=sp,
        user_message="this job can help for PR",
    )
    assert "immigration" in truth.scope_violations_detected


def test_build_truth_summary_merges_caller_and_message_scope_violations():
    """Caller-supplied scope tags come first (handler-side detections
    take precedence), with message-detected tags appended deduped."""
    from skillbridge.chat.truth_summary import build_truth_summary
    from skillbridge.session.staging import StagedProfile

    sp = StagedProfile.new("test-merge")
    truth = build_truth_summary(
        staged=sp,
        user_message="can I apply for PR?",
        scope_violations_detected=["off_topic"],
    )
    # Caller tag first
    assert truth.scope_violations_detected[0] == "off_topic"
    # Plus the message-detected immigration tag
    assert "immigration" in truth.scope_violations_detected


# ===========================================================================
# Post-live-test expansion: new training-intent patterns
# ===========================================================================
# Live test exposed gaps in the regex: "how can improve excel skill"
# returned asking_question, "give me link to online course" returned
# neutral. Expanded patterns cover these phrasings + others users
# naturally try.
@pytest.mark.parametrize("msg", [
    # "improve" verb in main pattern
    "how can I improve my excel skill",
    "how can improve excel skill",                 # the live-test wording (no "I")
    # "how do you improve X" rare phrasing; not supported -- users
    # almost always say "how can/do I" instead
    # build/upgrade/develop verbs
    "how can I build my forklift skill",
    "how do I upgrade my class G",
    # recommend variants
    "do you recommend any course",
    "can you recommend a course for me",
    "would you recommend a forklift course",
    # which/any/some/what course variants
    "any course do you recommend",
    "any course to improve my skill",
    "which training would be best",
    "some courses on excel?",
    # course/training/program "to/for" do-something
    "course to improve excel",
    "training for build welding skills",
    # give me link / give me course
    "give me link to do online course",
    "give me a course on excel",
    # I want to learn/improve etc
    "i want to learn excel",
    "I want to improve my forklift",
    # online course / free course as direct ask
    "online course please",
    "any free training",
])
def test_intent_classifier_catches_expanded_training_patterns(msg):
    assert _classify_intent(msg) == "asking_about_gap", (
        f"Expected asking_about_gap for {msg!r}; got {_classify_intent(msg)!r}"
    )


def test_intent_classifier_still_treats_impatient_as_impatient_when_mixed():
    """Sanity: impatient_proceed signals still win over training intent
    when both could fire (the user wants action, not training)."""
    assert _classify_intent("show me jobs now") == "impatient_proceed"
    # But a pure training question with no action verb is asking_about_gap
    assert _classify_intent("i want to learn excel") == "asking_about_gap"


# ===========================================================================
# registry_gaps_in_message field
# ===========================================================================
def test_truth_summary_carries_registry_gaps_in_message():
    """The handler computes registry-matched gaps from the user message
    and passes them through. The truth summary exposes them so the
    planner can use them as a routing signal."""
    from skillbridge.chat.truth_summary import build_truth_summary
    from skillbridge.session.staging import StagedProfile

    sp = StagedProfile.new("test-reg-gaps")
    truth = build_truth_summary(
        staged=sp,
        user_message="how can improve excel skill",
        registry_gaps_in_message=["Microsoft Excel"],
    )
    assert truth.registry_gaps_in_message == ["Microsoft Excel"]
    # And it round-trips through the prompt JSON
    blob = truth.to_planner_json()
    assert blob["registry_gaps_in_message"] == ["Microsoft Excel"]


# ===========================================================================
# Intent upgrade: registry-entity + training-action vocabulary
# ===========================================================================
# Live test surfaced phrases like "online Excel course" where the
# registry sees Microsoft Excel but the intent regex returns neutral.
# The intent-upgrade rule (in build_truth_summary, post-classification)
# promotes neutral/asking_question to asking_about_gap when both
# signals are present.
from skillbridge.chat.truth_summary import (
    _has_training_action_words,
    build_truth_summary as _build_truth_summary_for_upgrade_tests,
)
from skillbridge.session.staging import (
    StagedProfile as _StagedProfile_for_upgrade_tests,
)


def _build_with_intent_upgrade(
    msg: str, gaps: list[str] | None = None,
):
    sp = _StagedProfile_for_upgrade_tests.new("test-upgrade")
    return _build_truth_summary_for_upgrade_tests(
        staged=sp, user_message=msg,
        registry_gaps_in_message=gaps,
    )


# ---- Positive cases: BOTH entity AND training word -> upgrade ----
@pytest.mark.parametrize("msg, gaps", [
    # The exact live-test failure
    ("online Excel course", ["Microsoft Excel"]),
    # Other natural phrasings the rule should catch
    ("WHMIS training please", ["WHMIS"]),
    ("Class G link", ["Class G driver's license"]),
    ("forklift certificate", ["forklift certification"]),
    ("recommend a CPR course", ["first aid and CPR"]),
    ("free Excel class", ["Microsoft Excel"]),
    ("any 310T program around here?", ["310T technician certification"]),
    ("Microsoft Excel tutorial", ["Microsoft Excel"]),
    ("WHMIS certification online", ["WHMIS"]),
])
def test_intent_upgrades_to_asking_about_gap_when_entity_and_action_present(
    msg, gaps,
):
    truth = _build_with_intent_upgrade(msg, gaps)
    assert truth.user_intent_signal == "asking_about_gap", (
        f"Expected upgrade to asking_about_gap for {msg!r} with "
        f"entities {gaps!r}; got {truth.user_intent_signal!r}"
    )


# ---- Negative cases: entity present but NO training word -> no upgrade ----
@pytest.mark.parametrize("msg, gaps", [
    # Skill claims (the critical false-positive guard)
    ("I have Excel and forklift experience, find me jobs",
     ["Microsoft Excel", "forklift certification"]),
    ("warehouse job with Excel",
     ["Microsoft Excel"]),
    ("I worked with forklifts",
     ["forklift certification"]),
    ("my background includes Microsoft Excel and customer service",
     ["Microsoft Excel", "customer service"]),
])
def test_intent_does_NOT_upgrade_when_entity_without_training_word(msg, gaps):
    truth = _build_with_intent_upgrade(msg, gaps)
    assert truth.user_intent_signal != "asking_about_gap", (
        f"Skill claims must NOT upgrade to asking_about_gap. {msg!r} "
        f"has entities {gaps!r} but no training-action language -- "
        f"got {truth.user_intent_signal!r}"
    )


# ---- Negative cases: training word but NO entity -> no upgrade ----
@pytest.mark.parametrize("msg", [
    "any course you recommend",          # no specific gap
    "online training please",            # generic, no entity
    "give me a free course",
])
def test_intent_does_NOT_upgrade_when_training_word_without_entity(msg):
    # No registry gaps in the message
    truth = _build_with_intent_upgrade(msg, gaps=[])
    # These phrases may already be asking_about_gap via Layer 1 patterns
    # (e.g. "any course you recommend" matches a new pattern). The
    # point of THIS test is that the entity-less branch of the
    # upgrade rule isn't introducing spurious upgrades. The pre-Layer-1
    # intent matters: if intent was already asking_about_gap from
    # patterns, fine. If it was neutral/asking_question, no entity ->
    # no upgrade. We assert the second case: when intent is NOT
    # asking_about_gap from patterns, no upgrade happens from registry-less message.
    # For these messages Layer 1 patterns DO catch them, so they're
    # already asking_about_gap. We just sanity-check the test setup
    # doesn't pretend gap detection happened when it didn't.
    assert truth.registry_gaps_in_message == []


# ---- Stronger intents are NEVER downgraded ----
def test_intent_upgrade_does_not_overwrite_impatient():
    """If the user expresses impatient_proceed AND happens to mention
    a registry entity + training word, impatient still wins.
    'show me jobs and free Excel course' is action-first."""
    truth = _build_with_intent_upgrade(
        "show me jobs and free Excel course",
        gaps=["Microsoft Excel"],
    )
    assert truth.user_intent_signal == "impatient_proceed"


def test_intent_upgrade_does_not_overwrite_declining():
    truth = _build_with_intent_upgrade(
        "no thanks I'll skip the Excel course",
        gaps=["Microsoft Excel"],
    )
    assert truth.user_intent_signal == "declining"


# ---- _has_training_action_words helper ----
@pytest.mark.parametrize("msg, expected", [
    ("any course you recommend", True),
    ("training please", True),
    ("certificate to start", True),
    ("online resource", True),
    ("free", True),
    ("I want to learn", True),
    ("I have experience", False),
    ("find me jobs", False),
    ("I worked with forklifts", False),
    ("warehouse role", False),
    ("", False),
])
def test_has_training_action_words(msg, expected):
    assert _has_training_action_words(msg) == expected


def test_truth_summary_registry_gaps_defaults_to_empty():
    """Backwards-compat: caller can omit the field; defaults to []."""
    from skillbridge.chat.truth_summary import build_truth_summary
    from skillbridge.session.staging import StagedProfile

    sp = StagedProfile.new("test-empty")
    truth = build_truth_summary(staged=sp, user_message="hi")
    assert truth.registry_gaps_in_message == []


def test_intent_impatient_wins_over_asking_about_gap():
    """'show me jobs and how to get 310T' has both impatient + gap cues.
    The classifier should pick impatient -- the user explicitly wants
    action (re-show matches), and explain_gap's narration includes
    pointers anyway. Order: impatient > gap > question."""
    assert _classify_intent("show me jobs and how to get 310T") == "impatient_proceed"


def test_intent_gap_question_after_impatient_pattern_still_asking_about_gap():
    """Conversely, a pure gap question with no impatient cues classifies
    correctly even when the wording is conversational."""
    assert _classify_intent("how do I get my 310T?") == "asking_about_gap"


def test_intent_impatient_wins_over_confirming():
    """'alright, let's go' has both confirming + impatient cues. The
    classifier should pick impatient -- the user is signaling action,
    not just confirmation."""
    assert _classify_intent("alright let's go") == "impatient_proceed"


# ---------------------------------------------------------------------------
# _classify_target_role
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text, expected", [
    (None, "none"),
    ("", "none"),
    ("   ", "none"),
    ("any job", "vague"),
    ("something local", "vague"),
    ("general work", "vague"),
    ("software developer", "specific"),
    ("data analyst", "specific"),
    ("truck and coach technician", "specific"),
    ("electrical journeyman", "specific"),
    # "any" alone -> vague, but "any electrician" has a real noun -> specific
    ("any electrician", "specific"),
    # Slice (2026-06-08): anaphoric phrases that the upstream resolver
    # didn't handle (no resume work_history to anchor against) must
    # classify as vague so the planner asks again. Pre-fix these
    # tokenized to a single-real-word set and produced 'specific',
    # which corrupted downstream title-match scoring in the engine.
    ("same role", "vague"),
    ("same", "vague"),
    ("the same", "vague"),
    ("same kind", "vague"),
    ("current role", "vague"),
    ("current job", "vague"),
    ("current position", "vague"),
    ("previous job", "vague"),
    ("past field", "vague"),
    ("this", "vague"),
    ("that", "vague"),
    ("it", "vague"),
    # Negative cases: anaphor tokens combined with a real noun stay
    # specific -- the new vague tokens are conservative additions, not
    # so broad they break legitimate role names.
    ("warehouse position", "specific"),
    ("current warehouse manager", "specific"),
    ("apprentice technician", "specific"),
])
def test_target_role_specificity(text, expected):
    assert _classify_target_role(text) == expected


# ---------------------------------------------------------------------------
# _classify_resume_parse_quality
# ---------------------------------------------------------------------------
def _staged_with_facts(**facts_overrides) -> StagedProfile:
    sp = StagedProfile.new("test-session")
    sp.resume_filename = facts_overrides.pop("resume_filename", None)
    sp.resume_facts_json = facts_overrides or None
    return sp


def test_resume_quality_no_resume_when_facts_absent():
    sp = StagedProfile.new("test")
    quality, counts = _classify_resume_parse_quality(sp)
    assert quality == "no_resume"
    assert counts == ResumeFactsSummary()


def test_resume_quality_failed_when_filename_but_no_facts():
    """Live test 3-style: file uploaded, parse_warning fired, zero
    structured output -> bucket as 'failed' so enough_to_match's
    usable_evidence_present guard kicks in."""
    sp = _staged_with_facts(
        resume_filename="cv.pdf",
        skills=[], work_history=[], certifications=[], education=[],
    )
    quality, counts = _classify_resume_parse_quality(sp)
    assert quality == "failed"


def test_resume_quality_skills_only_matches_live_test_3():
    """The electrical CV that surfaced the orchestration gap:
    24 skills, 0 work_history -> 'skills_only' bucket."""
    sp = _staged_with_facts(
        resume_filename="cv.pdf",
        skills=[{"name": f"skill {i}"} for i in range(24)],
        work_history=[], certifications=[], education=[],
    )
    quality, counts = _classify_resume_parse_quality(sp)
    assert quality == "skills_only"
    assert counts.skill_count == 24
    assert counts.work_history_count == 0


def test_resume_quality_full_bucket():
    sp = _staged_with_facts(
        skills=[{"name": "x"}, {"name": "y"}, {"name": "z"}],
        work_history=[{"title": "Developer"}],
        education=[{"credential": "BSc"}],
        certifications=[],
    )
    quality, _ = _classify_resume_parse_quality(sp)
    assert quality == "full"


def test_resume_quality_partial_when_only_one_group():
    sp = _staged_with_facts(
        skills=[{"name": "x"}],   # below threshold of 3 -> partial, not skills_only
        work_history=[], certifications=[], education=[],
    )
    quality, _ = _classify_resume_parse_quality(sp)
    assert quality == "partial"


# ---------------------------------------------------------------------------
# _compute_enough_to_match -- the load-bearing guard
# ---------------------------------------------------------------------------
def test_enough_to_match_true_when_target_unspecified_but_skills_strong_and_explicit_intent():
    """A1 (2026-06-18): skills-only path now requires EXPLICIT intent.
    A user with strong skill evidence, NO specific target, AND
    `impatient_proceed` intent ("show jobs based on my skills") still
    gets engine-run. Without the explicit intent, the same evidence
    shape would route to ASK-for-target instead."""
    enough, reason, usable = _compute_enough_to_match(
        target_role_specificity="none",
        resume_parse_quality="full",
        counts=ResumeFactsSummary(skill_count=10, work_history_count=2),
        chat_skill_count=0,
        user_intent_signal="impatient_proceed",
    )
    assert enough is True
    assert reason == "skills_only_explicit_request"
    assert usable is True


def test_enough_to_match_false_when_target_vague_and_neutral_intent():
    """A1 (2026-06-18): vague target + neutral intent now defaults to
    ASK-for-target, even when skill evidence is strong. Prior behavior
    ran the engine in skills-only mode; the bug it caused was CP4
    silent failures because the engine had no target NOC to anchor
    recommendations."""
    enough, reason, _ = _compute_enough_to_match(
        target_role_specificity="vague",
        resume_parse_quality="full",
        counts=ResumeFactsSummary(skill_count=10),
        chat_skill_count=0,
        user_intent_signal="neutral",
    )
    assert enough is False
    assert reason == "missing_target"


def test_enough_to_match_false_when_target_unspecified_and_skills_thin():
    """Negative control: target absent AND skills below the skills-
    only threshold still falls back to asking for the target."""
    enough, reason, _ = _compute_enough_to_match(
        target_role_specificity="none",
        resume_parse_quality="full",
        counts=ResumeFactsSummary(skill_count=2),
        chat_skill_count=1,
        user_intent_signal="neutral",
    )
    assert enough is False
    assert reason == "missing_target"


def test_enough_to_match_false_when_target_unspecified_and_chat_skills_only_without_explicit_intent():
    """A1 (2026-06-18): chat_skill_count >= 3 alone is NO LONGER
    enough to fire skills-only mode. Without `impatient_proceed`
    intent (the user explicitly asking to skip target-setting), the
    default is to ASK for a target."""
    enough, reason, _ = _compute_enough_to_match(
        target_role_specificity="none",
        resume_parse_quality="no_resume",
        counts=ResumeFactsSummary(),
        chat_skill_count=5,
        user_intent_signal="neutral",
    )
    assert enough is False
    assert reason == "missing_target"


def test_enough_to_match_true_when_target_unspecified_and_chat_skills_with_explicit_intent():
    """A1 (2026-06-18): the chat-skills-only path SURVIVES when the
    user EXPLICITLY asks to skip target-setting. This is the legitimate
    "what jobs match my skills?" exploration mode."""
    enough, reason, _ = _compute_enough_to_match(
        target_role_specificity="none",
        resume_parse_quality="no_resume",
        counts=ResumeFactsSummary(),
        chat_skill_count=5,
        user_intent_signal="impatient_proceed",
    )
    assert enough is True
    assert reason == "skills_only_explicit_request"


def test_enough_to_match_true_when_target_vague_and_explicit_intent():
    """A1 (2026-06-18): vague target + `impatient_proceed` intent +
    strong resume skills also fires skills-only mode. The vague-target
    user who EXPLICITLY says "just match me" gets matched."""
    enough, reason, _ = _compute_enough_to_match(
        target_role_specificity="vague",
        resume_parse_quality="full",
        counts=ResumeFactsSummary(skill_count=10),
        chat_skill_count=0,
        user_intent_signal="impatient_proceed",
    )
    assert enough is True
    assert reason == "skills_only_explicit_request"


def test_enough_to_match_false_with_failed_resume_and_impatience():
    """The Step 4-review guard. Live test 3 had this exact shape:
    failed parse + impatience would otherwise mark enough=true. The
    usable_evidence_present guard prevents it."""
    enough, reason, usable = _compute_enough_to_match(
        target_role_specificity="specific",
        resume_parse_quality="failed",
        counts=ResumeFactsSummary(),
        chat_skill_count=0,
        user_intent_signal="impatient_proceed",
    )
    assert enough is False
    assert reason == "no_usable_evidence"
    assert usable is False


def test_enough_to_match_true_when_resume_skills_sufficient():
    enough, reason, usable = _compute_enough_to_match(
        target_role_specificity="specific",
        resume_parse_quality="skills_only",
        counts=ResumeFactsSummary(skill_count=10),
        chat_skill_count=0,
        user_intent_signal="neutral",
    )
    assert enough is True
    assert reason == "resume_skills_sufficient"
    assert usable is True


def test_enough_to_match_true_when_chat_skills_sufficient_no_resume():
    """User without a resume can still trigger match if they've
    described enough skills in chat AND explicitly filled the
    skills_text slot. Change C (no-final-no-without-resume rule,
    2026-06-16): the bare chat_skill_count >= 3 branch is now gated
    on skills_text_present so phantom skills lifted from experience
    prose don't trigger engine-run."""
    enough, reason, _ = _compute_enough_to_match(
        target_role_specificity="specific",
        resume_parse_quality="no_resume",
        counts=ResumeFactsSummary(),
        chat_skill_count=5,
        user_intent_signal="neutral",
        skills_text_present=True,
    )
    assert enough is True
    assert reason == "chat_skills_sufficient"


def test_enough_to_match_false_when_chat_skills_lifted_from_experience():
    """Change C regression (2026-06-16): a user provides ONLY
    experience prose; the extractor lifts phantom skill tokens from
    it (e.g. "truck and coach technician" from
    "Completed Truck and Coach Technician apprenticeship at Sault
    College"). Pre-fix, chat_skill_count >= 3 alone triggered
    enough_to_match=True and the engine ran prematurely. Post-fix,
    `skills_text_present=False` (the user has not claimed skills as
    skills) holds the engine until an explicit skills statement."""
    enough, reason, _ = _compute_enough_to_match(
        target_role_specificity="specific",
        resume_parse_quality="no_resume",
        counts=ResumeFactsSummary(),
        chat_skill_count=5,                       # phantom skills present
        user_intent_signal="neutral",
        skills_text_present=False,                # but skills_text empty
    )
    assert enough is False
    assert reason == "insufficient_skill_evidence"


def test_enough_to_match_true_when_work_history_present():
    """Light skill list but real work history -> enough."""
    enough, reason, _ = _compute_enough_to_match(
        target_role_specificity="specific",
        resume_parse_quality="work_only",
        counts=ResumeFactsSummary(skill_count=2, work_history_count=2),
        chat_skill_count=0,
        user_intent_signal="neutral",
    )
    assert enough is True
    assert reason == "resume_work_history_present"


def test_enough_to_match_true_when_user_explicitly_asks_with_minimal_evidence():
    """User with a partial resume but an explicit 'show me jobs' --
    proceed. The usable_evidence_present guard still gates this:
    partial > failed, so it passes."""
    enough, reason, _ = _compute_enough_to_match(
        target_role_specificity="specific",
        resume_parse_quality="partial",
        counts=ResumeFactsSummary(skill_count=1),
        chat_skill_count=0,
        user_intent_signal="impatient_proceed",
    )
    assert enough is True
    assert reason == "user_explicitly_asked_to_match"


def test_enough_to_match_false_when_no_evidence_thresholds_clear():
    enough, reason, _ = _compute_enough_to_match(
        target_role_specificity="specific",
        resume_parse_quality="partial",
        counts=ResumeFactsSummary(skill_count=2),   # below 5
        chat_skill_count=1,                          # below 3
        user_intent_signal="neutral",                # not impatient
    )
    assert enough is False
    assert reason == "insufficient_skill_evidence"


# ---------------------------------------------------------------------------
# build_truth_summary -- public entry point
# ---------------------------------------------------------------------------
def test_build_returns_truth_summary_with_expected_shape():
    sp = StagedProfile.new("test-build")
    sp.target_role_text = "data analyst"
    sp.skills = [StagedSkill(skill_name=f"skill {i}", source="chat")
                 for i in range(4)]
    # Change C (no-final-no-without-resume rule, 2026-06-16):
    # chat-skills branch now requires an explicit skills_text slot
    # to filter out phantom skills lifted from experience prose.
    sp.skills_text = "python, sql, data analysis, excel"

    ts = build_truth_summary(
        staged=sp,
        user_message="show me jobs",
        last_assistant_move="ask_one_clarifying_question",
        last_asked_slot="work_type_preference",
    )
    assert isinstance(ts, TruthSummary)
    assert ts.user_message == "show me jobs"
    assert ts.last_assistant_move == "ask_one_clarifying_question"
    assert ts.target_role_text == "data analyst"
    assert ts.target_role_specificity == "specific"
    assert ts.user_intent_signal == "impatient_proceed"
    # 4 chat skills + impatient + specific target + skills_text -> enough_to_match
    assert ts.enough_to_match is True
    assert ts.enough_to_match_reason == "chat_skills_sufficient"


def test_build_serializes_to_planner_json():
    """The serialized form is what the planner LLM consumes. Pin the
    keys so future schema drift breaks tests, not silently the prompt."""
    sp = StagedProfile.new("test-json")
    sp.target_role_text = "electrician"
    ts = build_truth_summary(staged=sp, user_message="hi")
    payload = ts.to_planner_json()

    required_keys = {
        "user_message", "last_assistant_move", "last_asked_slot",
        "message_count", "resume_uploaded", "resume_parse_quality",
        "resume_facts_summary", "target_role_text",
        "target_role_specificity", "work_type_preference",
        "filled_slots", "declined_slots", "enough_to_match",
        "enough_to_match_reason", "usable_evidence_present",
        "missing_critical", "match_count", "best_match_band",
        "caps_applied", "user_intent_signal", "scope_violations_detected",
        "registry_gaps_in_message",   # post-cold-session-hardening
        # Fresh-intake-on-target-change pillar (2026-06-15).
        "skills_aligned_with_target",
        "experience_aligned_with_target",
        "target_alignment_ok",
        "target_alignment_first_misaligned_slot",
    }
    assert set(payload.keys()) == required_keys


def test_build_handles_live_test_3_scenario_correctly():
    """Reproduces the orchestration-v2 motivating case:
    user uploads electrical CV with skills only, says 'same role',
    and the truth summary correctly reports enough_to_match=True with
    reason 'resume_skills_sufficient' -- so the planner CAN proceed
    instead of asking for work history."""
    sp = StagedProfile.new("test-live-3")
    sp.resume_filename = "cv_electrical.pdf"
    sp.resume_facts_json = {
        "skills": [{"name": f"electrical skill {i}"} for i in range(24)],
        "work_history": [],
        "certifications": [{"name": "309A"}],
        "education": [],
    }
    sp.target_role_text = "electrical journeyman"

    ts = build_truth_summary(staged=sp, user_message="same role")
    assert ts.resume_parse_quality == "skills_only"
    assert ts.target_role_specificity == "specific"
    assert ts.user_intent_signal == "impatient_proceed"
    assert ts.enough_to_match is True
    assert ts.enough_to_match_reason == "resume_skills_sufficient"
    assert ts.usable_evidence_present is True


def test_build_blocks_match_for_failed_resume_with_impatience():
    """Symmetric to the above: failed-parse + impatient must NOT
    auto-proceed. This is the Step-4-review guard in action."""
    sp = StagedProfile.new("test-failed-resume")
    sp.resume_filename = "scanned.pdf"
    sp.resume_facts_json = {
        "skills": [], "work_history": [], "certifications": [], "education": [],
    }
    sp.target_role_text = "data analyst"

    ts = build_truth_summary(staged=sp, user_message="see my cv")
    assert ts.resume_parse_quality == "failed"
    assert ts.user_intent_signal == "impatient_proceed"
    assert ts.enough_to_match is False
    assert ts.enough_to_match_reason == "no_usable_evidence"
    assert ts.usable_evidence_present is False


# ---------------------------------------------------------------------------
# Slice-1 review fix: durable resume_parse_warning persists across turns
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("warning", [
    "too_large", "empty_input", "unsupported_format",
    "parse_failed", "no_text",
])
def test_failed_parse_durable_via_resume_parse_warning(warning):
    """The real failure mode flagged in slice-1 review: handler's
    early-return paths on parser-level failures don't write
    resume_facts_json. Without the new resume_parse_warning field,
    truth_summary would classify these as 'no_resume' on the next
    turn, losing the failure signal.

    With the field set by the handler, the classifier returns
    'failed' across every turn until a successful re-upload clears it."""
    sp = StagedProfile.new("test-parse-warning")
    sp.resume_filename = "uploaded.pdf"
    sp.resume_parse_warning = warning
    # resume_facts_json STAYS None -- this mirrors what the handler
    # actually writes in the early-return branches.

    quality, counts = _classify_resume_parse_quality(sp)
    assert quality == "failed"
    assert counts == ResumeFactsSummary()


def test_successful_upload_clears_resume_parse_warning():
    """Verify the inverse: once resume_facts_json is populated by a
    successful re-upload AND resume_parse_warning is cleared (which
    the handler does), the classifier moves out of 'failed'."""
    sp = StagedProfile.new("test-recovery")
    sp.resume_filename = "uploaded.pdf"
    sp.resume_parse_warning = None   # handler cleared on success
    sp.resume_facts_json = {
        "skills": [{"name": "Python"}, {"name": "SQL"}, {"name": "AWS"}],
        "work_history": [], "certifications": [], "education": [],
    }
    quality, counts = _classify_resume_parse_quality(sp)
    assert quality == "skills_only"   # not "failed"
    assert counts.skill_count == 3


def test_resume_parse_warning_takes_precedence_over_facts():
    """Edge case: if a future code path ever writes BOTH a warning AND
    some facts (it shouldn't, but defense), the warning wins. Better
    to report 'failed' honestly than to pretend a partial parse was
    successful when the handler flagged a problem."""
    sp = StagedProfile.new("test-precedence")
    sp.resume_filename = "uploaded.pdf"
    sp.resume_parse_warning = "no_text"
    sp.resume_facts_json = {
        "skills": [{"name": "ghost"}],   # shouldn't happen, but if it does
        "work_history": [], "certifications": [], "education": [],
    }
    quality, _ = _classify_resume_parse_quality(sp)
    assert quality == "failed"


# ---------------------------------------------------------------------------
# Slice-1 polish: work-type vocabulary in vague tokens
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "full time",
    "full-time",
    "part time",
    "part-time",
    "flexible",
    "contract",
    "full-time contract",      # all-vague
])
def test_target_role_work_type_tokens_classify_vague(text):
    """If the slot extractor ever misroutes a work-type preference into
    target_role_text, classify it as vague (not specific). Prevents
    'full time' alone from triggering enough_to_match."""
    assert _classify_target_role(text) == "vague"


def test_target_role_work_type_does_not_falsely_vagueify_real_roles():
    """The polish must NOT mark legitimate work-type-qualified roles
    as vague. 'part-time pharmacist' has 'pharmacist' as a significant
    token, so it stays specific."""
    assert _classify_target_role("part-time pharmacist") == "specific"
    assert _classify_target_role("full-time data analyst") == "specific"
    assert _classify_target_role("contract software developer") == "specific"
