"""Slice 2 (locked 2026-06-23) -- recommender resume-gated Layer B
flow + new chain (B -> C -> END, A intent-only) + richer responder
evidence package + prompt anti-template guards.

What this exercises (per design memo section 11):

  1. _user_profile_is_rich replaced by has_resume gate (no rich-profile
     heuristic; resume is the only gate).
  2. _format_canned_with_target substitution helper.
  3. Four canned text constants:
       _ASK_RESUME_FOR_LAYER_B (B empty + no resume)
       _OFFER_C_AFTER_EMPTY_B  (B empty + has resume)
       _LAYER_C_EMPTY_HONEST   (C direct intent + empty)
       _LAYER_A_EMPTY_HONEST   (A direct intent + empty)
  4. Layer B 3-branch decision tree in _dispatch_recommender_from_intent:
       (a) evidence has content -> render Layer B + offer C
       (b) evidence empty + no resume -> ASK_RESUME canned + deferred
       (c) evidence empty + has resume -> OFFER_C canned + pending=C
  5. Layer C direct intent empty -> _LAYER_C_EMPTY_HONEST (no cascade).
  6. Layer A direct intent empty -> _LAYER_A_EMPTY_HONEST (no cascade).
  7. Responder evidence package (slice 2 wider shape):
       TARGET_ROLE_TEXT, USER_PROFILE, MODE, VOICE_HINT,
       LAYER_B_EVIDENCE / LAYER_A_EVIDENCE / LAYER_C_EVIDENCE
  8. Prompt anti-template structural guards (no "Output format:";
     has reason-from-evidence instruction; has "don't use section
     headers" anti-template guard).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from skillbridge.chat import handler as h
from skillbridge.chat.gap_evidence import (
    GapEvidence,
    RecommenderEvidence,
    TrainingResource,
)
from skillbridge.session.staging import StagedProfile

pytestmark = pytest.mark.nodb


class _StubStore:
    def __init__(self):
        self.saved: dict[str, StagedProfile] = {}

    def save(self, staged: StagedProfile) -> str:
        self.saved[staged.session_id] = staged
        return staged.session_id


def _make_staged(
    *,
    target_role: str | None = "accounting clerk",
    target_noc: str | None = "14200",
    skills: tuple[str, ...] = (
        "bookkeeping", "QuickBooks", "Excel", "payroll",
        "invoicing", "accounts payable",
    ),
    has_resume: bool = False,
) -> StagedProfile:
    sp = StagedProfile.new("sess-test")
    sp.target_role_text = target_role
    sp.target_noc = target_noc
    from skillbridge.session.staging import StagedSkill
    sp.skills = [
        StagedSkill(skill_name=n, raw_phrase=n, confidence=1.0, source="chat")
        for n in skills
    ]
    if has_resume:
        sp.resume_facts_json = {"skills": list(skills)}
    return sp


# ===========================================================================
# _format_canned_with_target -- substitution helper
# ===========================================================================
def test_format_canned_substitutes_target_role():
    template = "Hi about {target_role}, want to know more?"
    out = h._format_canned_with_target(template, "accounting clerk")
    assert out == "Hi about accounting clerk, want to know more?"


def test_format_canned_falls_back_when_target_none():
    template = "Hi about {target_role}, want to know more?"
    out = h._format_canned_with_target(template, None)
    assert "that role" in out
    assert "{target_role}" not in out  # fully substituted


def test_format_canned_falls_back_when_target_blank():
    template = "Hi about {target_role}, want to know more?"
    out = h._format_canned_with_target(template, "   ")
    assert "that role" in out


# ===========================================================================
# The 4 canned text constants exist + have {target_role} where required
# ===========================================================================
def test_ask_resume_for_layer_b_constant_exists_with_target_placeholder():
    assert "{target_role}" in h._ASK_RESUME_FOR_LAYER_B
    # Strict gate -- no "or work history" alternative.
    assert "work history" not in h._ASK_RESUME_FOR_LAYER_B.lower()
    # Resume is named.
    assert "resume" in h._ASK_RESUME_FOR_LAYER_B.lower()
    assert "upload" in h._ASK_RESUME_FOR_LAYER_B.lower()


def test_offer_c_after_empty_b_constant_offers_related_career_paths():
    assert "{target_role}" in h._OFFER_C_AFTER_EMPTY_B
    assert "related career paths" in h._OFFER_C_AFTER_EMPTY_B.lower()


def test_layer_c_empty_honest_constant_does_not_cascade_to_a():
    assert "Canadian/NOC standard" not in h._LAYER_C_EMPTY_HONEST
    # No cascade offer to another mode.
    assert "different target" in h._LAYER_C_EMPTY_HONEST.lower() or \
        "different field" in h._LAYER_C_EMPTY_HONEST.lower()


def test_layer_a_empty_honest_constant_uses_target_role_placeholder():
    assert "{target_role}" in h._LAYER_A_EMPTY_HONEST
    # No cascade offer to other modes.
    assert "related career paths" not in h._LAYER_A_EMPTY_HONEST


# ===========================================================================
# Layer B 3-branch decision tree -- intent-driven dispatcher
# ===========================================================================
def _stub_branch_b_empty(monkeypatch):
    """Stubs that yield empty Layer B evidence (engine [] / CP4 None)."""
    monkeypatch.setattr(
        "skillbridge.chat.recommender_intent.classify_career_intent",
        lambda **kwargs: "local_skill_gap",
    )
    monkeypatch.setattr(
        "skillbridge.match.engine.compute_matches_in_memory",
        lambda staged, top=5: [],
    )
    monkeypatch.setattr(
        "skillbridge.match.engine.build_user_skill_rows",
        lambda skills: [],
    )
    monkeypatch.setattr(
        "skillbridge.match.engine.derive_user_skill_sets",
        lambda rows: (set(), set(), set()),
    )
    monkeypatch.setattr(
        "skillbridge.chat.development_plan.compute_primary_gap_name",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "skillbridge.chat.gap_evidence._fetch_noc_skill_rows",
        lambda noc: [],
    )
    monkeypatch.setattr(
        "skillbridge.chat.responder.is_enabled", lambda: False,
    )


def _stub_branch_b_with_content(monkeypatch):
    """Stubs that yield non-empty Layer B evidence (branch (a) fires)."""
    monkeypatch.setattr(
        "skillbridge.chat.recommender_intent.classify_career_intent",
        lambda **kwargs: "local_skill_gap",
    )

    @dataclass
    class _M:
        noc_code: str = "14200"
        job_id: str = "job-1"
        title: str = "Bookkeeper"

    monkeypatch.setattr(
        "skillbridge.match.engine.compute_matches_in_memory",
        lambda staged, top=5: [_M()],
    )
    monkeypatch.setattr(
        "skillbridge.match.engine.build_user_skill_rows",
        lambda skills: [],
    )
    monkeypatch.setattr(
        "skillbridge.match.engine.derive_user_skill_sets",
        lambda rows: (set(), set(), set()),
    )
    monkeypatch.setattr(
        "skillbridge.chat.development_plan.compute_primary_gap_name",
        lambda **kwargs: "QuickBooks",
    )
    fake_gap = GapEvidence(
        layer="local_posting",
        source_id="job-1",
        source_label="Bookkeeper",
        skill_id="S_QB",
        skill_name="QuickBooks",
        blocker=False,
        importance=None,
        source="extracted.job_skill",
    )
    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly.build_recommender_evidence_local_gap_coach",
        lambda **kwargs: RecommenderEvidence(
            mode="local_gap_coach",
            evidence=(fake_gap,),
            training=(),
        ),
    )
    monkeypatch.setattr(
        "skillbridge.chat.responder.is_enabled", lambda: False,
    )


def test_branch_a_layer_b_has_content_renders_and_offers_c(monkeypatch):
    """Branch (a): Layer B evidence non-empty -> render Layer B body
    + chain to adjacent_noc_standard (offer C)."""
    _stub_branch_b_with_content(monkeypatch)
    sp = _make_staged(has_resume=False)
    out = h._maybe_route_recommender_from_intent(
        staged=sp, message="what should I improve",
        store=_StubStore(),
    )
    assert out is not None
    assert isinstance(out["reply"], str)
    # Reply rendered via fallback (LLM disabled). Should contain the
    # gap name (verified from the responder render path).
    assert "QuickBooks" in out["reply"]
    # Chain advances to C.
    assert sp.pending_recommender_offer == "adjacent_noc_standard"
    # Branch (a) does NOT set deferred intent.
    assert sp.deferred_career_intent is None


def test_branch_b_empty_no_resume_asks_resume(monkeypatch):
    """Branch (b): Layer B empty + no resume -> ASK_RESUME canned
    text emitted; deferred intent persisted; pending cleared."""
    _stub_branch_b_empty(monkeypatch)
    sp = _make_staged(has_resume=False)
    out = h._maybe_route_recommender_from_intent(
        staged=sp, message="what should I improve",
        store=_StubStore(),
    )
    assert out is not None
    # Canned ASK_RESUME emitted with target_role substituted.
    assert "upload your resume" in out["reply"]
    assert "accounting clerk" in out["reply"]
    # Deferred persisted so slice 1 re-routes after resume upload.
    assert sp.deferred_career_intent == "local_skill_gap"
    # last_asked_slots emptied (the canned ask is not a slot ask).
    assert sp.last_asked_slots == []
    # Pending cleared -- branch (b) doesn't offer C until resume arrives.
    assert sp.pending_recommender_offer is None


def test_branch_c_empty_with_resume_offers_c(monkeypatch):
    """Branch (c): Layer B empty + resume uploaded -> OFFER_C
    canned text + pending = adjacent_noc_standard.
    The C layer is NOT rendered yet; only OFFERED."""
    _stub_branch_b_empty(monkeypatch)
    sp = _make_staged(has_resume=True)
    out = h._maybe_route_recommender_from_intent(
        staged=sp, message="what should I improve",
        store=_StubStore(),
    )
    assert out is not None
    # Canned OFFER_C emitted with target_role substituted.
    assert "related career paths" in out["reply"]
    assert "accounting clerk" in out["reply"]
    # Branch (c) does NOT render Layer C content this turn.
    # (No NOC titles or development areas should appear.)
    # Pending set so user's next "yes" routes to C via consume hook.
    assert sp.pending_recommender_offer == "adjacent_noc_standard"
    # Deferred intent cleared (we're not waiting for resume).
    assert sp.deferred_career_intent is None


def test_branch_b_no_cascade_to_a(monkeypatch):
    """Branch (b) and (c) must NEVER mention Layer A / NOC standard.
    Slice 2 rule: A is intent-only, never a fallback."""
    _stub_branch_b_empty(monkeypatch)

    # Branch (b): no resume.
    sp_b = _make_staged(has_resume=False)
    out_b = h._maybe_route_recommender_from_intent(
        staged=sp_b, message="what should I improve",
        store=_StubStore(),
    )
    assert "Canadian/NOC standard" not in out_b["reply"]
    assert "NOC profile" not in out_b["reply"]

    # Branch (c): resume.
    sp_c = _make_staged(has_resume=True)
    out_c = h._maybe_route_recommender_from_intent(
        staged=sp_c, message="what should I improve",
        store=_StubStore(),
    )
    assert "Canadian/NOC standard" not in out_c["reply"]


# ===========================================================================
# Layer C direct-intent empty -> honest text (no cascade to A)
# ===========================================================================
def test_layer_c_direct_intent_empty_emits_honest_text(monkeypatch):
    """career_exploration intent fires Layer C directly. When evidence
    is empty (last_adjacent_nocs not populated), the dispatcher emits
    the honest no-recommendation text -- NEVER cascades to Layer A."""
    monkeypatch.setattr(
        "skillbridge.chat.recommender_intent.classify_career_intent",
        lambda **kwargs: "career_exploration",
    )
    # Layer C reads from staged.last_adjacent_nocs; empty tuple yields
    # empty evidence per the assembly contract.
    monkeypatch.setattr(
        "skillbridge.match.engine.build_user_skill_rows",
        lambda skills: [],
    )
    monkeypatch.setattr(
        "skillbridge.match.engine.derive_user_skill_sets",
        lambda rows: (set(), set(), set()),
    )
    monkeypatch.setattr(
        "skillbridge.chat.gap_evidence._fetch_noc_skill_rows",
        lambda noc: [],
    )
    monkeypatch.setattr(
        "skillbridge.chat.responder.is_enabled", lambda: False,
    )
    sp = _make_staged()
    # No last_adjacent_nocs populated -> Layer C empty.
    assert sp.last_adjacent_nocs == ()

    out = h._maybe_route_recommender_from_intent(
        staged=sp, message="what else can I do",
        store=_StubStore(),
    )
    assert out is not None
    # Honest empty-C text emitted.
    assert "Nothing surfaced" in out["reply"] or "related roles" in out["reply"]
    # No cascade to Layer A.
    assert "Canadian/NOC standard" not in out["reply"]
    # Pending cleared.
    assert sp.pending_recommender_offer is None


# ===========================================================================
# Layer A direct-intent empty -> honest text (no cascade)
# ===========================================================================
def test_layer_a_direct_intent_empty_emits_honest_text(monkeypatch):
    """noc_standard_comparison intent fires Layer A directly. When the
    OaSIS reference profile for that NOC isn't loaded (evidence empty),
    the dispatcher emits the honest no-NOC-profile text with
    target_role substituted -- NEVER cascades elsewhere."""
    monkeypatch.setattr(
        "skillbridge.chat.recommender_intent.classify_career_intent",
        lambda **kwargs: "noc_standard_comparison",
    )
    monkeypatch.setattr(
        "skillbridge.match.engine.build_user_skill_rows",
        lambda skills: [],
    )
    monkeypatch.setattr(
        "skillbridge.match.engine.derive_user_skill_sets",
        lambda rows: (set(), set(), set()),
    )
    monkeypatch.setattr(
        "skillbridge.chat.gap_evidence._fetch_noc_skill_rows",
        lambda noc: [],  # empty -> Layer A evidence empty
    )
    monkeypatch.setattr(
        "skillbridge.chat.responder.is_enabled", lambda: False,
    )
    sp = _make_staged()
    out = h._maybe_route_recommender_from_intent(
        staged=sp, message="compare to NOC standard",
        store=_StubStore(),
    )
    assert out is not None
    # Honest empty-A text with target_role substituted.
    assert "accounting clerk" in out["reply"]
    assert "NOC standard" in out["reply"] or "skill profile" in out["reply"]
    # Pending cleared.
    assert sp.pending_recommender_offer is None


# ===========================================================================
# Filter integration: handler's Layer B path applies filter before CP4
# ===========================================================================
def test_handler_applies_target_noc_filter_before_cp4(monkeypatch):
    """The handler's Layer B path must filter engine matches to the
    target NOC family BEFORE calling CP4. Otherwise CP4 ranks gaps
    across off-target postings.

    We stub compute_matches_in_memory to return mixed-NOC matches and
    verify compute_primary_gap_name receives ONLY the target-NOC
    subset."""

    @dataclass
    class _M:
        noc_code: str
        job_id: str = "x"

    monkeypatch.setattr(
        "skillbridge.chat.recommender_intent.classify_career_intent",
        lambda **kwargs: "local_skill_gap",
    )
    monkeypatch.setattr(
        "skillbridge.match.engine.compute_matches_in_memory",
        lambda staged, top=5: [
            _M(noc_code="14200", job_id="a"),  # target -- should survive
            _M(noc_code="13110", job_id="b"),  # off-target -- should drop
            _M(noc_code="14404", job_id="c"),  # off-target -- should drop
            _M(noc_code="14200", job_id="d"),  # target -- should survive
        ],
    )
    monkeypatch.setattr(
        "skillbridge.match.engine.build_user_skill_rows",
        lambda skills: [],
    )
    monkeypatch.setattr(
        "skillbridge.match.engine.derive_user_skill_sets",
        lambda rows: (set(), set(), set()),
    )

    captured: dict = {}

    def fake_cp4(**kwargs):
        captured["in_memory_matches"] = kwargs.get("in_memory_matches")
        captured["target_posting_count"] = kwargs.get("target_posting_count")
        return None  # CP4 returns no primary -> Layer B evidence empty

    monkeypatch.setattr(
        "skillbridge.chat.development_plan.compute_primary_gap_name",
        fake_cp4,
    )
    monkeypatch.setattr(
        "skillbridge.chat.responder.is_enabled", lambda: False,
    )

    sp = _make_staged(has_resume=True)  # ensure we reach the dispatch
    h._maybe_route_recommender_from_intent(
        staged=sp, message="what should I improve",
        store=_StubStore(),
    )

    # CP4 received ONLY the target-NOC matches (a and d), not the
    # off-target ones (b and c).
    received = captured["in_memory_matches"]
    assert len(received) == 2
    job_ids = sorted([m.job_id for m in received])
    assert job_ids == ["a", "d"]
    # target_posting_count tracks the filtered count.
    assert captured["target_posting_count"] == 2


# ===========================================================================
# Responder evidence package (slice 2 wider shape)
# ===========================================================================
def _build_responder_input(
    *,
    mode: str,
    rec: RecommenderEvidence,
    target_role_text: str | None = "accounting clerk",
    voice_hint: str | None = "local_skill_gap",
    resume_facts: dict | None = None,
) -> Any:
    from skillbridge.chat.responder import ResponderV2Input
    from skillbridge.chat.arbiter import ArbiterDecision
    decision = ArbiterDecision(
        final_move="present_tiered_matches",
        reason_code="test",
        tone="neutral",
        arbiter_action="recommender",
        ask_slot=None,
    )
    return ResponderV2Input(
        user_message="what should I improve",
        decision=decision,
        results=[],
        training_by_job={},
        next_skill=(None, 0),
        band_signal="none",
        requires_consent=True,
        target_role_text=target_role_text,
        resume_facts=resume_facts,
        recommendation_evidence=rec,
        recommender_voice_hint=voice_hint,
    )


def test_responder_block_layer_b_carries_full_evidence_shape():
    """Slice 2 responder block must carry TARGET_ROLE_TEXT, USER_PROFILE,
    MODE, VOICE_HINT, LAYER_B_EVIDENCE, LAYER_B_TRAINING."""
    import json
    from skillbridge.chat.responder import _build_user_block_for_recommender

    gap = GapEvidence(
        layer="local_posting",
        source_id="job-1",
        source_label="Bookkeeper at Diamond J Farms",
        skill_id="S_QB",
        skill_name="QuickBooks Desktop",
        blocker=False,
        importance=None,
        source="extracted.job_skill",
    )
    training = TrainingResource(
        skill_id="S_QB",
        skill_name="QuickBooks Desktop",
        provider="Sault College",
        type="local_training",
        url="https://saultcollege.ca/quickbooks",
        summary="Bookkeeping fundamentals course",
    )
    rec = RecommenderEvidence(
        mode="local_gap_coach",
        evidence=(gap,),
        training=(training,),
    )
    inp = _build_responder_input(
        mode="local_gap_coach", rec=rec,
        resume_facts={"skills": ["bookkeeping", "QuickBooks"]},
    )
    block = json.loads(_build_user_block_for_recommender(inp))

    # Common fields.
    assert block["TARGET_ROLE_TEXT"] == "accounting clerk"
    assert block["MODE"] == "local_gap_coach"
    assert block["VOICE_HINT"] == "local_skill_gap"
    # USER_PROFILE structured.
    assert "USER_PROFILE" in block
    assert block["USER_PROFILE"]["has_resume"] is True
    assert "QuickBooks" in block["USER_PROFILE"]["named_skills"]
    # Layer B-specific fields.
    assert block["LAYER_B_EVIDENCE"]["primary_gap"] == "QuickBooks Desktop"
    assert block["LAYER_B_EVIDENCE"]["primary_gap_skill_id"] == "S_QB"
    assert block["LAYER_B_EVIDENCE"]["lead_posting"]["title"] == "Bookkeeper at Diamond J Farms"
    assert block["LAYER_B_EVIDENCE"]["lead_posting"]["job_id"] == "job-1"
    # Training surfaced with provider + URL.
    assert len(block["LAYER_B_TRAINING"]) == 1
    assert block["LAYER_B_TRAINING"][0]["provider"] == "Sault College"
    assert block["LAYER_B_TRAINING"][0]["url"] == "https://saultcollege.ca/quickbooks"
    # Layer A/C-specific fields NOT present in Layer B block.
    assert "LAYER_A_EVIDENCE" not in block
    assert "LAYER_C_EVIDENCE" not in block


def test_responder_block_layer_a_carries_development_areas():
    """Slice 2 Layer A block must carry LAYER_A_EVIDENCE with noc_code,
    oasis_title, top_development_areas."""
    import json
    from skillbridge.chat.responder import _build_user_block_for_recommender

    gaps = (
        GapEvidence(
            layer="target_noc_standard",
            source_id="14200", source_label="Cost clerk",
            skill_id="F.01", skill_name="Reading Comprehension",
            blocker=False, importance=4.5,
            source="reference.noc_skill",
        ),
        GapEvidence(
            layer="target_noc_standard",
            source_id="14200", source_label="Cost clerk",
            skill_id="F.02", skill_name="Critical Thinking",
            blocker=False, importance=4.0,
            source="reference.noc_skill",
        ),
    )
    rec = RecommenderEvidence(
        mode="target_noc_standard",
        evidence=gaps,
        training=(),
    )
    inp = _build_responder_input(
        mode="target_noc_standard", rec=rec,
        voice_hint="noc_standard_comparison",
    )
    block = json.loads(_build_user_block_for_recommender(inp))

    assert block["TARGET_ROLE_TEXT"] == "accounting clerk"
    assert block["MODE"] == "target_noc_standard"
    assert "LAYER_A_EVIDENCE" in block
    assert block["LAYER_A_EVIDENCE"]["noc_code"] == "14200"
    # OaSIS title is background context (NOT for user-facing role naming).
    assert block["LAYER_A_EVIDENCE"]["oasis_title"] == "Cost clerk"
    areas = block["LAYER_A_EVIDENCE"]["top_development_areas"]
    assert len(areas) == 2
    assert areas[0]["name"] == "Reading Comprehension"
    assert areas[0]["importance"] == 4.5


def test_responder_block_layer_c_groups_by_noc():
    """Slice 2 Layer C block must group evidence records by NOC,
    preserving first-seen order, with noc_code + noc_title +
    development_areas list per NOC."""
    import json
    from skillbridge.chat.responder import _build_user_block_for_recommender

    gaps = (
        GapEvidence(
            layer="adjacent_noc_standard",
            source_id="13110", source_label="Administrative assistant",
            skill_id="F.A", skill_name="Coordinating",
            blocker=False, importance=4.5,
            source="reference.noc_skill",
        ),
        GapEvidence(
            layer="adjacent_noc_standard",
            source_id="13110", source_label="Administrative assistant",
            skill_id="F.B", skill_name="Writing",
            blocker=False, importance=4.0,
            source="reference.noc_skill",
        ),
        GapEvidence(
            layer="adjacent_noc_standard",
            source_id="13100", source_label="Business services officer",
            skill_id="F.X", skill_name="Critical Thinking",
            blocker=False, importance=4.2,
            source="reference.noc_skill",
        ),
    )
    rec = RecommenderEvidence(
        mode="adjacent_noc_standard",
        evidence=gaps,
        training=(),
    )
    inp = _build_responder_input(
        mode="adjacent_noc_standard", rec=rec,
        voice_hint="career_exploration",
    )
    block = json.loads(_build_user_block_for_recommender(inp))

    assert "LAYER_C_EVIDENCE" in block
    layer_c = block["LAYER_C_EVIDENCE"]
    # Two NOCs, grouped, first-seen order preserved.
    assert len(layer_c) == 2
    assert layer_c[0]["noc_code"] == "13110"
    assert layer_c[0]["noc_title"] == "Administrative assistant"
    assert layer_c[0]["development_areas"] == ["Coordinating", "Writing"]
    assert layer_c[1]["noc_code"] == "13100"
    assert layer_c[1]["development_areas"] == ["Critical Thinking"]


def test_responder_block_target_role_text_distinct_from_oasis_title():
    """Critical slice 2 grounding: TARGET_ROLE_TEXT must hold the user's
    phrasing ('accounting clerk') while LAYER_A_EVIDENCE.oasis_title
    can differ ('Cost clerk'). The prompt instructs the LLM to use
    TARGET_ROLE_TEXT, not oasis_title."""
    import json
    from skillbridge.chat.responder import _build_user_block_for_recommender

    gap = GapEvidence(
        layer="target_noc_standard",
        source_id="14200", source_label="Cost clerk",  # OaSIS title
        skill_id="F.01", skill_name="Reading Comprehension",
        blocker=False, importance=4.5,
        source="reference.noc_skill",
    )
    rec = RecommenderEvidence(
        mode="target_noc_standard", evidence=(gap,), training=(),
    )
    inp = _build_responder_input(
        mode="target_noc_standard", rec=rec,
        target_role_text="accounting clerk",  # user's phrasing
    )
    block = json.loads(_build_user_block_for_recommender(inp))

    assert block["TARGET_ROLE_TEXT"] == "accounting clerk"
    assert block["LAYER_A_EVIDENCE"]["oasis_title"] == "Cost clerk"
    # They're distinct -- the prompt uses TARGET_ROLE_TEXT in user-
    # facing prose; oasis_title is background only.
    assert block["TARGET_ROLE_TEXT"] != block["LAYER_A_EVIDENCE"]["oasis_title"]


# ===========================================================================
# Prompt structural assertions -- anti-template guards present
# ===========================================================================
def test_prompt_has_no_output_format_directive():
    """Slice 2 anti-template rule: the prompt must NOT include
    'Output format:' or other slot-filling instructions. The loan-
    example diagnosis showed that 'Output format: Recommendation: /
    Reason: / Gap:' produces templated output. We avoid that here."""
    from skillbridge.chat.prompts import RECOMMENDER_RESPONDER_PROMPT
    # No "Output format:" anywhere.
    assert "Output format:" not in RECOMMENDER_RESPONDER_PROMPT
    # No labeled output sections enumerated as schema.
    # (The string "Recommendation:" appears INSIDE the anti-template
    # negative instruction "do NOT use section headers like
    # 'Recommendation:'" which is fine.)


def test_prompt_has_reason_from_evidence_instruction():
    """Slice 2 reasoning rule: the prompt must explicitly instruct the
    LLM to reason from evidence (combine facts), not just list them."""
    from skillbridge.chat.prompts import RECOMMENDER_RESPONDER_PROMPT
    body = RECOMMENDER_RESPONDER_PROMPT.lower()
    # Some form of "reason from the evidence" instruction.
    assert "reason from the evidence" in body
    # Combine-facts directive.
    assert "combine" in body


def test_prompt_has_anti_template_guards():
    """Slice 2 anti-template guards: prompt must say don't use section
    headers and don't bullet-list reasoning."""
    from skillbridge.chat.prompts import RECOMMENDER_RESPONDER_PROMPT
    body = RECOMMENDER_RESPONDER_PROMPT.lower()
    assert "section headers" in body
    assert "bullet-list" in body or "bullet list" in body


def test_prompt_uses_target_role_text_grounding():
    """Slice 2 grounding rule: prompt must instruct the LLM to use
    TARGET_ROLE_TEXT, not OaSIS oasis_title, when referring to the
    user's role."""
    from skillbridge.chat.prompts import RECOMMENDER_RESPONDER_PROMPT
    assert "TARGET_ROLE_TEXT" in RECOMMENDER_RESPONDER_PROMPT
    # Anti-leak: oasis_title NOT to be used as the user-facing role.
    body = RECOMMENDER_RESPONDER_PROMPT.lower()
    assert "oasis_title" in body  # referenced in anti-leak instruction


def test_prompt_has_no_citation_tag_pattern():
    """Slice 2 prose rule: the prompt instructs the LLM to NOT attach
    citation tags like [E1] or [F1]. This is the loan-example fix --
    inline naming in prose, not bullet-attached labels."""
    from skillbridge.chat.prompts import RECOMMENDER_RESPONDER_PROMPT
    body = RECOMMENDER_RESPONDER_PROMPT.lower()
    # The prompt says "don't attach citation tags like [E1] or [F1]".
    assert "citation tag" in body


def test_prompt_per_layer_voice_rules():
    """Slice 2 per-layer voice: Layer A and Layer C must NEVER use
    deficit voice on OaSIS competencies. The prompt enforces this."""
    from skillbridge.chat.prompts import RECOMMENDER_RESPONDER_PROMPT
    body = RECOMMENDER_RESPONDER_PROMPT.lower()
    # Development-area voice for A.
    assert "development-area" in body
    # Forbidden deficit phrasing quoted in anti-leak instruction.
    # Normalize whitespace so line-wraps in the prompt don't break
    # the assertion.
    normalized = " ".join(body.split())
    assert "you don't have" in normalized
