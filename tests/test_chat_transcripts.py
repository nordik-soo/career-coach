"""Chat orchestration v2 slice 7 -- transcript regression suite.

This is the slice that proves v2 was worth building. Each scenario is a
real multi-turn user flow driven end-to-end through the v2 dispatch.
Each turn captures:

  - The final_move that reaches the responder (architecture proof)
  - Whether the engine was actually invoked (no-hidden-matching proof)
  - The deterministic responder text (UX regression lock)

Assertions:

  1. expected_final_move_sequence -- the final moves the arbiter
     emits, in order, across all turns of the scenario. Exact match.
  2. expected_responder_NOT_to_contain -- substrings that must NEVER
     appear in any turn's reply. Combines a global ALWAYS_FORBIDDEN
     list (operational leakage, scope violations, v1 robotic phrases)
     with scenario-specific additions.
  3. engine_called per turn -- the engine runs ONLY on proceed turns
     that the arbiter has independently approved.

Test harness: planner LLM is mocked per-turn (we script what the
planner should return for each user message); match engine is mocked
per-turn (we script what the engine should return when arbiter
approves); responder runs in its deterministic fallback path
(LLM_ENABLED=false). The fallback path is what the chat ACTUALLY
returns when the LLM is unavailable -- so these assertions lock the
worst-case UX. A separate manual-flag test re-runs the same scenarios
against real Haiku to validate the happy-path narration.

Scope note: the upload turn in some design-doc scenarios is handled
by v1's existing PRESENT_RESUME_FACTS short-circuit, not by v2. We
adapt those scenarios to start from the post-upload state (staged
pre-populated, just like a returning user). The v2 resume_upload
gate is wired but not reachable through handle_anonymous in this
rollout -- a follow-up slice can route uploads through v2 after the
post-upload flows are validated.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from skillbridge.chat import handler
from skillbridge.chat import truth_summary as ts_mod
from skillbridge.chat.planner import PlannerDecision
from skillbridge.session.staging import StagedProfile, StagedSkill

pytestmark = pytest.mark.nodb


# ===========================================================================
# Test infrastructure
# ===========================================================================
class _FakeStore:
    """In-memory session store. Does not persist between scenarios."""
    def __init__(self):
        self.saved = []

    def new_session(self) -> str:
        return "test-sid"

    def load(self, sid):
        return None

    def save(self, staged):
        self.saved.append(staged)
        return staged.session_id or "test-sid"

    def delete(self, sid):
        pass


def _planner(
    move: str,
    reason_code: str,
    ask_slot: str | None = None,
    tone: str = "brief_confident",
) -> PlannerDecision:
    return PlannerDecision.model_validate({
        "move": move, "reason_code": reason_code,
        "ask_slot": ask_slot, "tone": tone,
    })


@dataclass
class Turn:
    """One turn of a scripted scenario."""
    user_message: str
    # The PlannerDecision the mocked planner should return for this
    # turn. None means "the gate is expected to fire and the planner
    # should never be called this turn."
    planner_decision: PlannerDecision | None
    # The engine output for this turn. None means "engine MUST NOT
    # be called this turn." A list (possibly empty) means "engine is
    # expected to run and return these results."
    engine_results: list[dict] | None
    # Expected final_move reaching the responder.
    expected_final_move: str
    # Optional truth_summary field overrides applied AFTER the
    # deterministic builder runs. Used to inject scope violations and
    # similar test-only signals.
    truth_overrides: dict[str, Any] = field(default_factory=dict)
    uploaded_file: bool = False


@dataclass
class Scenario:
    name: str
    description: str
    # Builder for a fresh StagedProfile (one per scenario run).
    initial_staged: Callable[[], StagedProfile]
    turns: list[Turn]
    # Substrings forbidden in every turn's reply IN ADDITION to the
    # global ALWAYS_FORBIDDEN list below.
    extra_forbidden: list[str] = field(default_factory=list)


# ===========================================================================
# Global forbidden phrases -- never allowed in any v2 reply
# ===========================================================================
# These come from three categories:
#   - Operational leakage (the Slice 4/5 hard boundary)
#   - v1 robotic phrases the architecture exists to retire
#   - Active product-rule violations (out-of-region, dollar amounts,
#     national feeds, immigration advice)
ALWAYS_FORBIDDEN: list[str] = [
    # Operational/architecture leakage -- should never reach the user
    "the planner said",
    "the arbiter decided",
    "arbiter_action",
    "I overrode",
    "fallback_to_legacy",
    # Live v1 robotic phrases the design doc §8 names explicitly
    "Before we find the right match",
    "Can you walk me through your previous jobs",
    "I hear you, but",
    # Out-of-region offers (Sprint 3 product boundary)
    "try Toronto",
    "search elsewhere",
    "broaden the search",
    "check nearby cities",
    # Dollar amounts and national feeds
    "$",
    "/hr",
    "/hour",
    "Job Bank",
    "Statistics Canada",
    "national average",
    # Immigration / legal -- out-of-scope
    "Express Entry",
    "RCIP eligibility",
    "PR application",
]


# ===========================================================================
# Scenario runner
# ===========================================================================
def _run_scenario(monkeypatch, scenario: Scenario) -> list[dict]:
    """Drive a scenario through v2 dispatch, return per-turn results.

    Each turn:
      - Mocks plan_next_move to return the turn's scripted PlannerDecision
        (or fails if engine_results=None but planner was called).
      - Mocks compute_matches_in_memory to return the turn's engine_results,
        or fails fast if called when engine_results=None.
      - Captures: response dict, whether engine was called, whether
        planner was called.
    """
    monkeypatch.setattr(handler, "CHAT_ORCHESTRATOR", "v2")

    # Per-turn counters
    turn_idx = [0]
    planner_calls = [0]
    engine_calls = [0]

    # ---- Planner mock: returns scripted decision for current turn ----
    def fake_plan(truth_json):
        planner_calls[0] += 1
        current = scenario.turns[turn_idx[0]]
        return current.planner_decision

    monkeypatch.setattr(handler, "plan_next_move", fake_plan)

    # ---- Engine mock: returns scripted results OR fails fast ----
    def fake_engine(staged, top=20):
        engine_calls[0] += 1
        current = scenario.turns[turn_idx[0]]
        if current.engine_results is None:
            pytest.fail(
                f"Scenario {scenario.name!r} turn {turn_idx[0]} "
                f"(user_message={current.user_message!r}): match engine "
                f"was called on a turn where engine_results=None. This "
                f"means v2 ran the engine on a path the arbiter shouldn't "
                f"have approved -- the 'no hidden matching' invariant has "
                f"been violated."
            )
        return current.engine_results

    monkeypatch.setattr(
        handler.match_engine, "compute_matches_in_memory", fake_engine,
    )

    # ---- Surrounding match-pipeline stubs ----
    def fake_build_results_block(matches):
        if not matches:
            return ([], "none")
        return (list(matches), "strong_or_good")

    monkeypatch.setattr(handler, "_build_results_block", fake_build_results_block)
    monkeypatch.setattr(handler, "_attach_training", lambda r: {})
    monkeypatch.setattr(
        handler.match_engine, "next_skill_to_unlock_in_memory",
        lambda staged: (None, 0),
    )

    # ---- Truth-summary override hook ----
    # When a turn specifies truth_overrides, apply them to the built
    # TruthSummary BEFORE to_planner_json() is called so the planner
    # AND the arbiter both see the overridden values.
    real_build = ts_mod.build_truth_summary

    def overriding_build(*, staged, user_message, **kw):
        result = real_build(staged=staged, user_message=user_message, **kw)
        current = scenario.turns[turn_idx[0]]
        for k, v in current.truth_overrides.items():
            setattr(result, k, v)
        return result

    monkeypatch.setattr(handler, "build_truth_summary", overriding_build)

    # ---- Drive the turns ----
    staged = scenario.initial_staged()
    store = _FakeStore()
    results: list[dict] = []

    for i, turn in enumerate(scenario.turns):
        turn_idx[0] = i
        engine_before = engine_calls[0]
        planner_before = planner_calls[0]

        response = handler._try_v2_path(
            staged=staged, message=turn.user_message,
            uploaded_file=turn.uploaded_file,
            resume_info=None, store=store,
        )

        results.append({
            "turn_index": i,
            "user_message": turn.user_message,
            "response": response,
            "engine_called": engine_calls[0] > engine_before,
            "planner_called": planner_calls[0] > planner_before,
            "expected_final_move": turn.expected_final_move,
            "expects_engine": turn.engine_results is not None,
            "expects_planner": turn.planner_decision is not None,
        })

    return results


# ===========================================================================
# Staged-profile factories
# ===========================================================================
def _fresh_session(sid: str = "test-session") -> StagedProfile:
    return StagedProfile.new(sid)


def _staged_with_warehouse_evidence(sid: str = "test-warehouse") -> StagedProfile:
    """A user who has built up a usable profile: target role + 5 skills.
    Simulates a few prior turns. Used for impatient-proceed scenarios."""
    sp = StagedProfile.new(sid)
    sp.message_count = 4
    sp.target_role_text = "warehouse worker"
    sp.skills_text = "forklift, inventory, shipping"
    sp.skills = [
        StagedSkill(skill_name=name, source="chat", confidence=0.9)
        for name in ("forklift", "inventory", "shipping", "receiving", "picking")
    ]
    sp.experience_text = "2 years warehouse work at a Sault Ste. Marie distribution centre."
    # Fresh-intake-on-target-change pillar (2026-06-15): direct list
    # assignment to sp.skills bypasses merge_skills, so the alignment
    # stamp doesn't fire. Set it manually to reflect the fixture's
    # intent: a user fully intaked for this target.
    sp.skills_collected_for_target = sp.target_role_text
    # experience_text setter already stamped via __setattr__.
    return sp


def _staged_after_truck_matches_with_310T_gap(
    sid: str = "test-post-match",
) -> StagedProfile:
    """Slice 9 scenario fixture: simulates a user whose previous turn
    surfaced two stretch-match truck/coach roles capped by the 310T
    credential. The Slice 8 ConversationContext carry-forward fields
    are pre-populated so the planner sees the same state it would on a
    real follow-up turn."""
    sp = StagedProfile.new(sid)
    sp.message_count = 8
    sp.target_role_text = "truck and coach technician"
    sp.skills = [
        StagedSkill(skill_name=name, source="resume", confidence=0.9)
        for name in (
            "diesel engine diagnosis", "computerized diagnostic tools",
            "hydraulic systems maintenance", "troubleshooting and diagnostics",
            "welding",
        )
    ]
    sp.resume_facts_json = {
        "skills": [{"name": "diesel engine diagnosis"}],
        "work_history": [{"title": "Apprentice Truck and Coach Technician"}],
        "education": [{"credential": "Truck and Coach Technician Apprenticeship",
                       "institution": "Sault College"}],
        "certifications": [],
        "languages": ["English"],
    }
    # Slice 8 carry-forward state -- the previous turn presented these.
    sp.last_presented_job_titles = [
        "Truck and Coach Technician (Licensed and Apprentice)",
        "Truck and Coach Technician",
    ]
    sp.last_presented_caps_applied = ["band_capped_by_credential"]
    sp.last_presented_credential_gaps = ["310T technician certification"]
    # Fresh-intake-on-target-change pillar (2026-06-15): direct list
    # assignment + resume-only evidence path; stamp alignment fields
    # to reflect the fixture's intent (post-match follow-up turn).
    sp.skills_collected_for_target = sp.target_role_text
    sp.experience_collected_for_target = sp.target_role_text
    return sp


def _staged_with_truck_profile(sid: str = "test-truck") -> StagedProfile:
    """A truck mechanic candidate missing the 310T credential. Used
    for the credential-cap scenario."""
    sp = StagedProfile.new(sid)
    sp.message_count = 6
    sp.target_role_text = "truck mechanic"
    sp.skills = [
        StagedSkill(skill_name=name, source="chat", confidence=0.9)
        for name in (
            "engine repair", "diagnostics", "preventive maintenance",
            "hydraulics", "electrical systems",
        )
    ]
    sp.experience_text = "3 years working as a truck mechanic apprentice."
    # Fresh-intake-on-target-change pillar (2026-06-15): direct list
    # assignment to sp.skills bypasses merge_skills, so the alignment
    # stamp doesn't fire. See _staged_with_warehouse_evidence.
    sp.skills_collected_for_target = sp.target_role_text
    return sp


def _staged_after_honda_310s_present_matches(
    sid: str = "test-honda-post-match",
) -> StagedProfile:
    """R-6 scenario fixture: simulates a user who just had a Honda
    310S role presented. The R-1 snapshot is pre-populated so the
    remaining-gaps detection layer fires on the user's next turn.
    """
    sp = StagedProfile.new(sid)
    sp.message_count = 4
    sp.target_role_text = "automotive technician"
    sp.skills = [
        StagedSkill(skill_name=name, source="chat", confidence=0.8)
        for name in (
            "tire changes", "oil changes", "basic diagnostics",
        )
    ]
    sp.last_match_snapshot = {
        "captured_at_turn": 3,
        "lead_job": {
            "job_id": "honda-1",
            "title": "310S Licensed Automotive Technician",
            "employer": "Great Lakes Honda",
            "credential_gaps": [
                {"display":   "310S Automotive Technician License",
                 "canonical": "310S automotive technician certification"},
                {"display":   "G2/G driver's license",
                 "canonical": "Class G driver's license"},
            ],
            "core_skill_gaps": ["Honda vehicle experience"],
        },
        "other_jobs_meta": [],
    }
    # Fresh-intake-on-target-change pillar (2026-06-15): stamp
    # alignment fields for fixtures that build a fully-intaked profile
    # via direct list assignment (bypassing merge_skills).
    sp.skills_collected_for_target = sp.target_role_text
    sp.experience_collected_for_target = sp.target_role_text
    return sp


# ===========================================================================
# Scenarios
# ===========================================================================
SCENARIOS: list[Scenario] = [
    # ---------- 1. First-turn bare greeting (gate 3 fires) ----------
    Scenario(
        name="first_turn_bare_greeting",
        description=(
            "User opens chat with 'hi'. Gate 3 (first_turn_greeting) "
            "fires; planner NOT called; engine NOT called; canned "
            "welcome returned. Locks the gates-short-circuit-before-LLM "
            "contract from Slice 2."
        ),
        initial_staged=_fresh_session,
        turns=[
            Turn(
                user_message="hi",
                planner_decision=None,    # gate fires -- planner never called
                engine_results=None,
                expected_final_move="acknowledge_and_continue",
            ),
        ],
    ),

    # ---------- 2. First-turn job intent (Slice 2 regression) ----------
    Scenario(
        name="first_turn_with_job_intent",
        description=(
            "Slice 2 review regression: 'I'm looking for warehouse work' "
            "on turn 1 must NOT fire the greeting gate. The planner is "
            "called and emits ask_one_clarifying_question (no usable "
            "evidence yet). Engine MUST NOT run."
        ),
        initial_staged=_fresh_session,
        turns=[
            Turn(
                user_message="I'm looking for warehouse work",
                planner_decision=_planner(
                    move="ask_one_clarifying_question",
                    reason_code="insufficient_profile_evidence",
                    ask_slot="skills_text",
                    tone="warm_supportive",
                ),
                engine_results=None,
                expected_final_move="ask_one_clarifying_question",
            ),
        ],
        extra_forbidden=[
            # Even though the user opened with job intent, the canned
            # greeting must not appear. Slice 2 review catch.
            "I help folks in Sault Ste. Marie find local work",
        ],
    ),

    # ---------- 3. Impatient proceed WITH evidence -> matches ----------
    Scenario(
        name="impatient_proceed_with_evidence_finds_matches",
        description=(
            "User has built profile (target + 5 skills). They type "
            "'just match me'. With MESSAGE_UNDERSTANDING_ENABLED on, the "
            "router's Rule 4 (job_search + truth ready) emits "
            "proceed_to_match deterministically and the planner is "
            "SKIPPED -- hence planner_decision=None. Arbiter pass 1 "
            "still verifies independently; engine runs; pass 2 emits "
            "present_matches with the router-supplied tone preserved."
        ),
        initial_staged=_staged_with_warehouse_evidence,
        turns=[
            Turn(
                user_message="just match me",
                # Slice D (2026-06-05): router Rule 4 pre-empts; planner
                # is not called. Behavior reaching the responder is the
                # same; the LLM is just no longer in this loop.
                planner_decision=None,
                engine_results=[
                    {
                        "job_id": "j1",
                        "title": "Warehouse Associate",
                        "employer": "Acme Logistics",
                        "url": "https://example.com/j1",
                        "match_band": "strong",
                        "matched_skills": ["forklift", "inventory"],
                        "missing_skills": [],
                        "score_explanation": {"caps_applied": []},
                    },
                ],
                expected_final_move="present_matches",
            ),
        ],
    ),

    # ---------- 4. Impatient proceed NO evidence -> override to ask ----------
    Scenario(
        name="impatient_proceed_no_evidence_overridden_to_ask",
        description=(
            "User says 'match me' but has no profile (no target_role, no "
            "skills). Even if planner emits proceed_to_match, arbiter "
            "pass 1's independent re-check overrides to "
            "ask_one_clarifying_question. Engine MUST NOT run. This is "
            "the 'LLM proposes, backend disposes' rule made testable."
        ),
        initial_staged=_fresh_session,
        turns=[
            Turn(
                user_message="just match me",
                planner_decision=_planner(
                    move="proceed_to_match",
                    reason_code="user_explicitly_asked_to_match",
                    tone="brief_confident",
                ),
                engine_results=None,   # MUST NOT be called
                expected_final_move="ask_one_clarifying_question",
            ),
        ],
    ),

    # ---------- 5. Scope redirect: immigration question ----------
    Scenario(
        name="scope_redirect_immigration",
        description=(
            "User asks about PR application while job-searching. With "
            "MESSAGE_UNDERSTANDING_ENABLED on, the router's Rule 1 "
            "(scope_violation) catches 'PR' deterministically and emits "
            "redirect_scope -- planner SKIPPED. Engine MUST NOT run. "
            "Reply must not contain Express Entry / PR advice."
        ),
        initial_staged=_staged_with_warehouse_evidence,
        turns=[
            Turn(
                user_message="can I apply for PR while looking for work here?",
                # Slice D (2026-06-05): router Rule 1 pre-empts; planner
                # is not called. The redirect_scope decision is synthesized
                # deterministically by the router from the scope keyword.
                planner_decision=None,
                engine_results=None,
                expected_final_move="redirect_scope",
            ),
        ],
        extra_forbidden=[
            "consult a lawyer",
            "you may qualify",
            "you might be eligible",
        ],
    ),

    # ---------- 6. Scope override BEATS proceed request ----------
    Scenario(
        name="scope_override_wins_over_proceed_request",
        description=(
            "User has a built profile AND asks about wages outside SSM. "
            "With MESSAGE_UNDERSTANDING_ENABLED on, the router's Rule 1 "
            "(scope_violation, national_wages category) emits "
            "redirect_scope FIRST -- the planner never gets a chance to "
            "wrongly emit proceed_to_match here. Engine MUST NOT run. "
            "This is the same precedence guarantee as the Slice 4 review, "
            "now enforced at the router layer instead of arbiter override."
        ),
        initial_staged=_staged_with_warehouse_evidence,
        turns=[
            Turn(
                user_message="what's the national average wage for warehouse work?",
                # Slice D (2026-06-05): router Rule 1 pre-empts; planner
                # is not called. truth_overrides retained as a marker of
                # what the legacy path would have surfaced, but the router
                # reads scope from the message directly (not from truth).
                planner_decision=None,
                truth_overrides={
                    "scope_violations_detected": ["national_wages"],
                },
                engine_results=None,   # router must block the engine path
                expected_final_move="redirect_scope",
            ),
        ],
    ),

    # ---------- 7. Credential cap match preserves planner tone ----------
    Scenario(
        name="credential_cap_match_preserves_planner_tone",
        description=(
            "Truck mechanic candidate is missing 310T. Engine returns "
            "matches with caps_applied=['band_capped_by_credential']. "
            "Slice 4 review tightening: pass 2 preserves planner_tone "
            "(warm_supportive in this case) instead of force-flattening "
            "to honest_redirect. caps_applied surfaces as a separate "
            "field; the responder names the cap in plain language."
        ),
        initial_staged=_staged_with_truck_profile,
        turns=[
            Turn(
                user_message="what jobs do you have for me?",
                planner_decision=_planner(
                    move="proceed_to_match",
                    reason_code="user_explicitly_asked_to_match",
                    tone="warm_supportive",   # NOT honest_redirect
                ),
                engine_results=[
                    {
                        "job_id": "j1",
                        "title": "Truck Mechanic",
                        "employer": "Local Garage",
                        "url": "https://example.com/j1",
                        "match_band": "stretch",
                        "matched_skills": ["engine repair", "diagnostics"],
                        "missing_skills": ["310T"],
                        "score_explanation": {
                            "caps_applied": ["band_capped_by_credential"],
                        },
                    },
                ],
                expected_final_move="present_matches",
            ),
        ],
    ),

    # ---------- 8. No-matches pivot with SCCC suggestion ----------
    Scenario(
        name="no_matches_pivot_with_honest_redirect",
        description=(
            "User has profile, asks for matches; engine returns 0. "
            "With MESSAGE_UNDERSTANDING_ENABLED on, the router's Rule 4 "
            "(job_search + truth ready) emits proceed_to_match; engine "
            "runs and returns []; pass 2 emits present_no_match + "
            "honest_redirect tone. Responder fallback uses the canonical "
            "no-match shape: anchor to dataset, suggest Sault Community "
            "Career Centre. Reply must not offer to look in other cities."
        ),
        initial_staged=_staged_with_warehouse_evidence,
        turns=[
            Turn(
                user_message="show me what you've got",
                # Slice D (2026-06-05): router Rule 4 pre-empts; planner
                # is not called. Engine still runs because router emits
                # proceed_to_match.
                planner_decision=None,
                engine_results=[],   # zero matches
                expected_final_move="present_no_match",
            ),
        ],
        extra_forbidden=[
            # Out-of-region offers covered in ALWAYS_FORBIDDEN; this
            # scenario specifically locks the SCCC-redirect shape.
            "I'll look in other regions",
        ],
    ),

    # ---------- 11. PR statement mid-conversation (Slice 10) ----------
    Scenario(
        name="pr_statement_after_matches_triggers_scope_detection",
        description=(
            "User saw matches, then said 'this job can help for PR'. "
            "With MESSAGE_UNDERSTANDING_ENABLED on, the router's Rule 1 "
            "catches 'PR' as immigration scope from the message itself "
            "and emits redirect_scope BEFORE the planner runs. Engine "
            "must not run. Pre-Slice-10 this fell through and re-rendered "
            "the same match cards; post-Slice-10 the arbiter caught it; "
            "post-Slice-B-v2.1 the router catches it deterministically."
        ),
        initial_staged=lambda: _staged_after_truck_matches_with_310T_gap(
            sid="test-pr-statement",
        ),
        turns=[
            Turn(
                user_message="this job can help for PR",
                # Slice D (2026-06-05): router Rule 1 pre-empts; planner
                # is not called.
                planner_decision=None,
                engine_results=None,    # engine MUST NOT run
                expected_final_move="redirect_scope",
            ),
        ],
        extra_forbidden=[
            # No immigration advice, no re-pitched matches.
            "Express Entry",
            "PR application",
            "Stretch match",
            "match band",
        ],
    ),

    # ---------- 10. Post-match gap question (Slice 9) ----------
    Scenario(
        name="post_match_gap_question_routes_to_explain_gap_not_proceed",
        description=(
            "User already saw matches with a credential cap. They follow "
            "up with 'how do I get my 310T?'. With "
            "MESSAGE_UNDERSTANDING_ENABLED on, the router's Rule 2 "
            "(training + registry entity) catches '310T' and emits "
            "explain_gap deterministically -- planner SKIPPED, engine "
            "MUST NOT run. Pre-Slice-9 the planner saw enough_to_match=True "
            "and routed to proceed_to_match, re-rendering matches; "
            "post-Slice-9 the planner picked explain_gap; post-Slice-B-v2.1 "
            "the router emits it without the LLM in the loop."
        ),
        initial_staged=lambda: _staged_after_truck_matches_with_310T_gap(),
        turns=[
            Turn(
                user_message="how do I get my 310T?",
                # Slice D (2026-06-05): router Rule 2 pre-empts; planner
                # is not called.
                planner_decision=None,
                engine_results=None,     # engine MUST NOT run
                expected_final_move="explain_gap",
            ),
        ],
        extra_forbidden=[
            # Should NOT re-pitch the matches as a list
            "Stretch match",
            "match band",
        ],
    ),

    # ---------- 9. Resume upload gate (Slice 7 review fix) ----------
    Scenario(
        name="resume_upload_shows_review_without_planner_or_engine",
        description=(
            "User uploads a resume. The resume_upload gate (gate 2) "
            "fires; planner is NOT called; engine is NOT called; the "
            "responder narrates the parsed facts via the existing "
            "RESUME_FACTS context. final_move=confirm_resume_summary. "
            "This locks the third deterministic gate from §4 of the "
            "design doc -- the one Slice 7 v1 left uncovered."
        ),
        initial_staged=lambda: _staged_with_uploaded_resume_facts(),
        turns=[
            Turn(
                user_message="",                  # file-only turn
                uploaded_file=True,               # triggers gate 2
                planner_decision=None,            # planner MUST NOT be called
                engine_results=None,              # engine MUST NOT be called
                expected_final_move="confirm_resume_summary",
            ),
        ],
        extra_forbidden=[
            # The v2 fallback should reference parsed facts, not ask
            # robotic intake questions or invent gaps.
            "Before we find the right match",
            "What kind of work are you looking for",
        ],
    ),

    # ---------- R-6: remaining-gaps end-to-end (310S Honda) ----------
    Scenario(
        name="remaining_gaps_310s_automotive_weak",
        description=(
            "R-6 (remaining-gaps iteration). User has a Honda 310S "
            "snapshot pre-populated from a prior present_matches turn. "
            "Four turns exercise the synthesis pipeline:\n"
            "  Turn 1: 'if I had 310S, what else?' -> kind=subtract "
            "          (hypothetical) -> explain_remaining_gaps with "
            "          G2/G remaining + conditional tense.\n"
            "  Turn 2: 'and if I had G2 too?' -> kind=subtract "
            "          (additional hypothetical) -> all credentials "
            "          assumed-closed; pivot to core skill gaps.\n"
            "  Turn 3: 'actually I don't have 310S' -> 310S in "
            "          accumulated -> kind=confirm pending_action=remove "
            "          -> ask_one_clarifying_question (retract confirm).\n"
            "  Turn 4: 'yes that's right' -> pending consumed -> "
            "          kind=retract -> explain_remaining_gaps with 310S "
            "          re-emerged as the lead remaining credential.\n"
            "Pins: planner / engine NEVER called on any of these turns "
            "(the remaining-gaps hook short-circuits before Pass 1). "
            "Snapshot is preserved across all four turns."
        ),
        initial_staged=_staged_after_honda_310s_present_matches,
        turns=[
            # Turn 1 -- explicit hypothetical
            Turn(
                user_message="if I had 310S, what else for this job?",
                planner_decision=None,         # hook short-circuits
                engine_results=None,           # no engine
                expected_final_move="explain_remaining_gaps",
            ),
            # Turn 2 -- additional hypothetical
            Turn(
                user_message="and if I had my G2 too?",
                planner_decision=None,
                engine_results=None,
                expected_final_move="explain_remaining_gaps",
            ),
            # Turn 3 -- explicit retraction triggers confirm
            Turn(
                user_message="actually I don't have 310S",
                planner_decision=None,
                engine_results=None,
                expected_final_move="ask_one_clarifying_question",
            ),
            # Turn 4 -- yes consumes pending; retract executes
            Turn(
                user_message="yes that's right",
                planner_decision=None,
                engine_results=None,
                expected_final_move="explain_remaining_gaps",
            ),
        ],
        extra_forbidden=[
            # Round-9 / R-5 contract: explain_remaining_gaps must NEVER
            # certify the match -- the user has only CLAIMED completion.
            "you qualify",
            "you're qualified",
            "you are qualified",
            "good fit",
            "good match",
            "great match",
            "perfect match",
            "stretch match",
            # R-5 prompt rule: no speculation about how non-credential
            # gaps are typically closed without TRAINING data.
            "usually come on the job",
            "typically come",
            "best learned through",
            "you'll pick that up on the job",
            "comes with time",
            "comes with experience",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Helper for scenario 9 -- a staged profile with pre-parsed resume facts.
# In production the upload pipeline populates these; we synthesize them
# directly so the transcript test stays focused on the v2 dispatch
# behavior and doesn't depend on the parser working.
# ---------------------------------------------------------------------------
def _staged_with_uploaded_resume_facts(sid: str = "test-upload") -> StagedProfile:
    sp = StagedProfile.new(sid)
    sp.message_count = 0   # first turn
    sp.resume_facts_json = {
        "work_history": [
            {
                "title": "Warehouse Associate",
                "employer": "Acme Logistics",
                "start_year": 2021, "end_year": 2024,
                "is_current": False,
            },
        ],
        "education": [
            {"credential": "High School Diploma", "institution": "Sault Collegiate"},
        ],
        "skills": [
            {"name": "forklift"}, {"name": "inventory"}, {"name": "shipping"},
        ],
        "certifications": [],
        "languages": ["English"],
    }
    return sp


# ===========================================================================
# Parametrized tests -- one assertion shape per concern
# ===========================================================================
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_scenario_final_move_sequence(scenario, monkeypatch):
    """expected_final_move_sequence: the arbiter's chosen moves across
    all turns of the scenario, in order, exact match. This is the
    primary architectural assertion."""
    results = _run_scenario(monkeypatch, scenario)
    actual = [r["response"]["final_move"] for r in results]
    expected = [r["expected_final_move"] for r in results]
    assert actual == expected, (
        f"Scenario {scenario.name!r}: final_move sequence mismatch.\n"
        f"  Description: {scenario.description}\n"
        f"  Expected: {expected}\n"
        f"  Actual:   {actual}\n"
        f"  Per-turn details: {[(r['user_message'], r['response']['final_move']) for r in results]}"
    )


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_scenario_engine_call_invariant(scenario, monkeypatch):
    """Per-turn engine invariant: the engine runs ONLY on turns where
    the script declared it should. Catches 'hidden matching' on any
    code path the arbiter shouldn't have approved."""
    results = _run_scenario(monkeypatch, scenario)
    for r in results:
        assert r["engine_called"] == r["expects_engine"], (
            f"Scenario {scenario.name!r} turn {r['turn_index']} "
            f"(message={r['user_message']!r}): engine_called="
            f"{r['engine_called']} but the script expected "
            f"engine={r['expects_engine']}. This means either the "
            f"arbiter approved an unsafe proceed, or it blocked one "
            f"that should have gone through."
        )


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_scenario_responder_never_uses_forbidden_phrases(scenario, monkeypatch):
    """expected_responder_NOT_to_contain: every turn's reply must
    avoid the global forbidden list AND the scenario-specific extras.

    With LLM_ENABLED=false, the responder runs its deterministic
    fallback. These assertions lock the WORST-CASE UX -- if the LLM
    is unavailable or fails policy, the canned text is what users
    actually see, and it must respect the same boundaries the prompt
    enforces in the happy path."""
    results = _run_scenario(monkeypatch, scenario)
    forbidden = ALWAYS_FORBIDDEN + scenario.extra_forbidden

    for r in results:
        reply = r["response"]["reply"]
        # Allow $ in URLs (none of our test fixtures use it, but defensive).
        # The actual check is conservative: any forbidden substring fails.
        for phrase in forbidden:
            assert phrase not in reply, (
                f"Scenario {scenario.name!r} turn {r['turn_index']} "
                f"(message={r['user_message']!r}): reply contains "
                f"forbidden phrase {phrase!r}.\n"
                f"  Full reply: {reply!r}"
            )


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_scenario_planner_call_invariant(scenario, monkeypatch):
    """Companion to the engine invariant: the planner is called ONLY
    on turns where no gate fires. Gates 1+3 (empty / first_turn_greeting)
    short-circuit before the planner runs."""
    results = _run_scenario(monkeypatch, scenario)
    for r in results:
        assert r["planner_called"] == r["expects_planner"], (
            f"Scenario {scenario.name!r} turn {r['turn_index']}: "
            f"planner_called={r['planner_called']} but the script "
            f"expected planner={r['expects_planner']}."
        )


# ===========================================================================
# Scenario coverage tripwire
# ===========================================================================
def test_at_least_nine_scenarios_present():
    """Post-Slice 7 review contract: 9 scripted scenarios covering all
    three deterministic gates plus the planner/arbiter/engine outcomes.
    The ninth (`resume_upload_shows_review_without_planner_or_engine`)
    was added in the Slice 7 review fix; if the list shrinks below 9,
    a load-bearing scenario was removed without a doc update."""
    assert len(SCENARIOS) >= 9, (
        f"Expected at least 9 transcript scenarios; found {len(SCENARIOS)}. "
        f"The 9-scenario set covers all three gates + key arbiter paths. "
        f"Check what was removed and update the design doc if intentional."
    )


def test_scenario_names_are_unique():
    """Scenarios are referenced by name in logs and pytest IDs;
    duplicates would silently shadow each other."""
    names = [s.name for s in SCENARIOS]
    assert len(names) == len(set(names)), (
        f"Duplicate scenario names: {[n for n in names if names.count(n) > 1]}"
    )


def test_every_scenario_has_a_description():
    """Per-scenario `description` documents what behavior the scenario
    locks. Mandatory for transcript-test readability."""
    for s in SCENARIOS:
        assert s.description.strip(), (
            f"Scenario {s.name!r} has no description. Document what "
            f"v2 behavior this scenario locks before merging."
        )


def test_every_scenario_has_at_least_one_turn():
    for s in SCENARIOS:
        assert len(s.turns) >= 1, (
            f"Scenario {s.name!r} has zero turns. Empty scenarios "
            f"don't exercise anything."
        )


# ===========================================================================
# R-6 acceptance: per-turn STATE + NARRATION assertions
# ===========================================================================
# The generic scaffolding above (final-move sequence, planner/engine
# call invariants, forbidden-phrase sweep) is necessary but not
# sufficient for the remaining-gaps iteration -- the scenario's value
# comes from the per-turn staged-state transitions and positive
# narration content. Without these, a regression that broke (e.g.)
# the retraction lifecycle or conditional tense would still pass the
# generic checks. The round-23 review caught this acceptance-test gap.
# This test drives the four-turn scenario in isolation, capturing the
# StagedProfile state AND the responder reply after each turn, and
# pins:
#   - accumulated credential modes (hypothetical / claimed, ordered)
#   - pending_credential_confirmation lifecycle (set / cleared at the
#     right turn boundaries)
#   - snapshot preservation across all four turns
#   - last_discussed anchor following the lead remaining credential
#   - conditional ("If you've got...") vs past-tense narration
#   - retracted credential re-emerges as remaining on Turn 4
#   - provider grounding follows the lead credential per turn
# ===========================================================================
def _drive_one_turn(monkeypatch, staged, message):
    """Drive a single turn through `_try_v2_path` with planner /
    engine fixtures that fail if called (the remaining-gaps hook
    short-circuits before either)."""
    real_build = ts_mod.build_truth_summary

    def _build_clean(*, staged, user_message, **kw):
        truth = real_build(staged=staged, user_message=user_message, **kw)
        object.__setattr__(truth, "scope_violations_detected", [])
        return truth
    monkeypatch.setattr(handler, "build_truth_summary", _build_clean)
    monkeypatch.setattr(handler, "CHAT_ORCHESTRATOR", "v2")

    def _no_planner(_truth):
        pytest.fail(
            f"plan_next_move called on a remaining-gaps turn "
            f"(message={message!r}). Hook must short-circuit before "
            f"the planner."
        )
    monkeypatch.setattr(handler, "plan_next_move", _no_planner)

    def _no_engine(staged, top=20):
        pytest.fail(
            f"compute_matches_in_memory called on a remaining-gaps "
            f"turn (message={message!r}). Engine must NEVER run on "
            f"hook-handled turns."
        )
    monkeypatch.setattr(
        handler.match_engine, "compute_matches_in_memory", _no_engine,
    )

    return handler._try_v2_path(
        staged=staged, message=message,
        uploaded_file=False, resume_info=None, store=_FakeStore(),
    )


def test_remaining_gaps_310s_scenario_state_and_narration_at_each_turn(monkeypatch):
    """Round-23 R-6 acceptance: per-turn state + reply content pins.

    Four-turn sequence:
      1. Explicit hypothetical -> 310S accumulated; lead remaining is
         G2/G; conditional narration; pending=None.
      2. Additional hypothetical -> both accumulated; all-closed branch;
         pivot to skill gaps; still conditional.
      3. Retraction trigger -> ask_one_clarifying_question with
         pending_action=remove; accumulated UNCHANGED (retract not
         executed yet); last_discussed anchors to 310S.
      4. Affirmative consumes pending -> 310S removed from accumulated;
         re-emerges as lead remaining; conditional preserved (G2 still
         hypothetical).
    """
    import copy
    staged = _staged_after_honda_310s_present_matches()
    # Round-24 R-6 review: a shallow `dict(...)` would share the nested
    # `lead_job` dict and `credential_gaps` list with the live snapshot.
    # An in-place mutation would then leave both values equal and the
    # assertion would silently pass. Deep-copy so any nested change
    # surfaces.
    initial_snapshot = copy.deepcopy(staged.last_match_snapshot)

    # ---- Turn 1: explicit hypothetical ----
    r1 = _drive_one_turn(
        monkeypatch, staged, "if I had 310S, what else for this job?",
    )
    assert r1["final_move"] == "explain_remaining_gaps"
    # Accumulated: only 310S, hypothetical
    assert staged.last_assumed_completed_credentials == [
        {"canonical": "310S automotive technician certification",
         "mode": "hypothetical"},
    ]
    # Pending: cleared (subtract path doesn't set pending)
    assert staged.pending_credential_confirmation is None
    # Last discussed: lead remaining is Class G after 310S subtraction
    assert staged.last_discussed_credential_canonical == \
        "Class G driver's license"
    # Snapshot intact
    assert staged.last_match_snapshot == initial_snapshot
    # Narration: conditional tense (any_hypothetical=True), names G2/G
    # as the lead remaining, does NOT use match-certification framing.
    reply1 = r1["reply"]
    assert "If you've got" in reply1
    assert "G2/G driver's license" in reply1
    # Round-24 finding 2: pin provider grounding to the LEAD remaining
    # credential. Turn 1's lead is G2/G -> DriveTest (registry's primary
    # resource for Class G driver's license). A regression that grounded
    # 310S's provider here would otherwise pass.
    assert "DriveTest" in reply1, (
        f"Turn 1 lead-remaining is G2/G; reply must name its provider "
        f"(DriveTest). Got: {reply1!r}"
    )
    # And it must NOT name 310S's provider here -- that would be
    # cross-wired training.
    assert "Skilled Trades Ontario" not in reply1, (
        f"Turn 1 lead-remaining is G2/G; reply must NOT cross-wire "
        f"310S's provider (Skilled Trades Ontario). Got: {reply1!r}"
    )
    for forbidden in ("good fit", "good match", "you qualify", "qualified"):
        assert forbidden not in reply1.lower(), (
            f"Turn 1 reply contains forbidden framing {forbidden!r}: "
            f"{reply1!r}"
        )

    # ---- Turn 2: additional hypothetical -> all credentials closed ----
    r2 = _drive_one_turn(monkeypatch, staged, "and if I had my G2 too?")
    assert r2["final_move"] == "explain_remaining_gaps"
    # Accumulated: both, both hypothetical, order preserved
    assert staged.last_assumed_completed_credentials == [
        {"canonical": "310S automotive technician certification",
         "mode": "hypothetical"},
        {"canonical": "Class G driver's license",
         "mode": "hypothetical"},
    ]
    assert staged.pending_credential_confirmation is None
    # Last discussed cleared -- no remaining credentials to anchor on
    assert staged.last_discussed_credential_canonical is None
    # Snapshot still intact across the second turn
    assert staged.last_match_snapshot == initial_snapshot
    reply2 = r2["reply"]
    assert "If you've got" in reply2          # still conditional
    # All-closed branch should pivot to skill gaps
    assert "Honda vehicle experience" in reply2
    # Round-24 finding 2 + design §6: with no remaining credentials,
    # NO provider may be named (training_by_job is empty on this
    # branch). Cross-check that no leak from the prior turn survives.
    for provider in (
        "DriveTest", "Skilled Trades Ontario",
        "Sault College", "Sault Community Career Centre",
    ):
        assert provider not in reply2, (
            f"All-closed branch must not name any provider; "
            f"{provider!r} leaked into reply: {reply2!r}"
        )

    # ---- Turn 3: explicit retraction trigger ----
    r3 = _drive_one_turn(
        monkeypatch, staged, "actually I don't have 310S",
    )
    assert r3["final_move"] == "ask_one_clarifying_question"
    # Accumulated UNCHANGED -- the retract isn't executed until the
    # user confirms on Turn 4. The handler only ASKS here.
    assert staged.last_assumed_completed_credentials == [
        {"canonical": "310S automotive technician certification",
         "mode": "hypothetical"},
        {"canonical": "Class G driver's license",
         "mode": "hypothetical"},
    ]
    # Pending: set to remove 310S
    assert staged.pending_credential_confirmation == {
        "canonical": "310S automotive technician certification",
        "action":    "remove",
    }
    # Last discussed: the credential we're about to walk back
    assert staged.last_discussed_credential_canonical == \
        "310S automotive technician certification"
    # Snapshot preserved
    assert staged.last_match_snapshot == initial_snapshot
    reply3 = r3["reply"]
    assert "Just to confirm" in reply3
    assert "310S Automotive Technician License" in reply3
    assert "recalculate" in reply3

    # ---- Turn 4: yes consumes pending -> retract executes ----
    r4 = _drive_one_turn(monkeypatch, staged, "yes that's right")
    assert r4["final_move"] == "explain_remaining_gaps"
    # Accumulated: ONLY G2 left -- 310S filtered out by the retract
    # path. Hypothetical preserved.
    assert staged.last_assumed_completed_credentials == [
        {"canonical": "Class G driver's license",
         "mode": "hypothetical"},
    ]
    # Pending cleared after consumption
    assert staged.pending_credential_confirmation is None
    # Last discussed follows the re-emerged credential (which is the
    # new lead remaining).
    assert staged.last_discussed_credential_canonical == \
        "310S automotive technician certification"
    # Snapshot still preserved across all four turns
    assert staged.last_match_snapshot == initial_snapshot
    reply4 = r4["reply"]
    # 310S re-emerges as the lead remaining credential
    assert "310S Automotive Technician License" in reply4
    # G2 is no longer in remaining (it's accumulated)
    # The narration stays conditional because G2 is still hypothetical
    assert "If you've got" in reply4
    # Round-24 finding 2: Turn 4's lead-remaining is now 310S ->
    # Skilled Trades Ontario (registry's primary resource for the
    # 310S apprenticeship). The retract path MUST reground the
    # provider to match the re-emerged credential, NOT carry over
    # the prior turn's DriveTest.
    assert "Skilled Trades Ontario" in reply4, (
        f"Turn 4 lead-remaining is 310S; reply must name its provider "
        f"(Skilled Trades Ontario) after the retract regrounded "
        f"training. Got: {reply4!r}"
    )
    assert "DriveTest" not in reply4, (
        f"Turn 4 lead-remaining is 310S; the DriveTest grounding from "
        f"Turn 1 must NOT carry over. Got: {reply4!r}"
    )
    # No match-certification framing
    for forbidden in ("good fit", "good match", "you qualify", "qualified"):
        assert forbidden not in reply4.lower(), (
            f"Turn 4 reply contains forbidden framing {forbidden!r}: "
            f"{reply4!r}"
        )


# ===========================================================================
# Live Haiku integration test -- manual flag
# ===========================================================================
@pytest.mark.skipif(
    os.environ.get("RUN_TRANSCRIPTS_LIVE") != "1",
    reason=(
        "Real Haiku transcript run -- set RUN_TRANSCRIPTS_LIVE=1 to run. "
        "Validates that the planner LLM emits sensible decisions on the "
        "scripted user messages, complementing the mocked unit tests."
    ),
)
def test_scenarios_with_real_planner_lm():
    """Run a subset of scenarios with the REAL planner (no plan_next_move
    mock). Manual-flag gated. Asserts the response dict has a final_move
    in the expected closed set and that no ALWAYS_FORBIDDEN phrase
    appears in the reply.

    Doesn't pin the exact final_move per turn -- the live LLM might
    legitimately pick different reasonable moves on different days.
    What we pin is the boundary: the LLM never produces operational
    leakage or out-of-scope offers regardless of which move it picks.
    """
    from skillbridge.chat.arbiter import OutcomeMove
    from typing import get_args

    live_scenarios = [
        s for s in SCENARIOS
        # Skip scenarios that need truth_overrides (those depend on
        # specific test injection that doesn't exist in live mode).
        if not any(t.truth_overrides for t in s.turns)
    ]

    for scenario in live_scenarios:
        staged = scenario.initial_staged()
        store = _FakeStore()

        for turn in scenario.turns:
            # Real planner; mocked engine (still scripted per turn).
            # We don't drive real DB matches in this test -- the goal
            # is to validate the planner's output shape, not the
            # engine's quality.
            response = handler._try_v2_path(
                staged=staged, message=turn.user_message,
                uploaded_file=turn.uploaded_file,
                resume_info=None, store=store,
            )
            if response is None:
                # Fallback signal -- acceptable in live mode (LLM
                # may have failed). Skip the rest of this scenario.
                break

            # Boundary checks: the LLM never produced forbidden phrases.
            for phrase in ALWAYS_FORBIDDEN:
                assert phrase not in response["reply"], (
                    f"LIVE: {scenario.name!r} reply contained "
                    f"{phrase!r}: {response['reply']!r}"
                )
            # final_move is in the closed set.
            assert response["final_move"] in get_args(OutcomeMove)
