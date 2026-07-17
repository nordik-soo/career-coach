"""Sprint 5 step 6 -- responder narrates from explanation only.

Two surfaces are exercised here:

  1. **Prompt contract** -- the NEXT_ACTION_RESPONDER_PROMPT must name
     every score_explanation field added in Sprint 5 steps 3-5, must
     contain the broader CAPS APPLIED rule (not the old credential-only
     one), and must require any cited number to trace back to
     score_explanation. These are weak in isolation (just string
     containment) but they catch a refactor that drops the rule.

  2. **Projection wire-through** -- `_build_user_block` must serialize
     the full score_explanation dict (including score_components and
     caps_applied) into the user message the LLM receives. If the
     projection drops those fields, the prompt's allow-list would
     reference data the LLM doesn't actually see.

No DB and no Anthropic calls; the existing nodb marker covers both.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from skillbridge.chat.intake_state import ACTION_PRESENT_MATCHES, Decision
from skillbridge.chat.prompts import NEXT_ACTION_RESPONDER_PROMPT
from skillbridge.chat.responder import (
    ResponderInput,
    _NARRATION_SKILL_CAP,
    _build_user_block,
    _capped_score_explanation,
    _narration_skill_view,
    _narration_skill_view_with_indices,
)
from skillbridge.chat.url_views import (
    build_sanitized_responder_view_v1 as _v_v1,
)


def _iv1(inp):
    """Return (inp, view) for v1 — reuses one inp instance."""
    return inp, _v_v1(inp)

pytestmark = pytest.mark.nodb


# ---------------------------------------------------------------------------
# Prompt contract -- string containment checks. Weak but locks the rule.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("token", [
    # Sprint 5 step 3 additions (required / preferred split)
    "required_matched",
    "required_missing",
    "preferred_matched",
    "preferred_missing",
    # Sprint 5 step 5 additions (structured breakdown)
    "score_components",
    "skill_base",
    "score_pre_caps",
    "score_post_caps",
    "caps_applied",
    # Cap rule must name all three caps (not just credential)
    "band_capped_by_credential",
    "band_capped_by_no_experience",
    "band_capped_by_work_type_mismatch",
    # Matching v2 step 5 + step 6 additions (semantic re-ranker)
    "required_match_stages",
    "preferred_match_stages",
    "target_noc_match",
])
def test_prompt_allows_listing_includes_step_3_to_5_fields(token):
    assert token in NEXT_ACTION_RESPONDER_PROMPT, (
        f"prompt missing reference to {token!r} -- the LLM won't know it's "
        f"allowed to cite this field"
    )


def test_prompt_has_caps_applied_rule_not_just_credential():
    """The pre-Step-6 prompt only mentioned the credential cap. After
    Step 6 the rule must cover all three caps under one heading."""
    text = NEXT_ACTION_RESPONDER_PROMPT
    # The new heading replaced CREDENTIAL-CAPPED MATCHES with one that
    # covers all three caps.
    assert "CAPS APPLIED" in text or "caps_applied" in text
    # And it must mention every cap-flag name with a narration example.
    assert "band_capped_by_no_experience" in text
    assert "band_capped_by_work_type_mismatch" in text


def test_prompt_requires_numbers_to_trace_to_score_explanation():
    """Step 6 hard rule: any number, ratio, or count in the reply must
    trace to a specific score_explanation field. The exact wording can
    drift -- this assertion just locks the intent."""
    text = NEXT_ACTION_RESPONDER_PROMPT.lower()
    # Both halves of the rule must be present.
    assert "score_explanation" in text
    # The numbers-must-trace clause was added on top of the existing
    # "never invent statistics" rule.
    assert "must trace" in text or "trace to" in text


def test_prompt_distinguishes_semantic_stage_from_possession():
    """Matching v2 step 6: when a matched skill's stage is 'semantic',
    the responder must phrase it as related background, never as
    possession ("you have X"). Otherwise the chat tells data analysts
    they have welding because the embedding said the concepts are
    adjacent. This is the line between honest matching and hallucinated
    competence."""
    text = NEXT_ACTION_RESPONDER_PROMPT
    # Stage labels documented
    assert "semantic" in text
    assert "exact" in text
    assert "fuzzy" in text
    # The MATCH STAGES rule explicitly forbids "you have X" framing for
    # semantic-stage matches.
    text_lower = text.lower()
    assert "related" in text_lower
    # Forbidden phrasing called out
    assert "❌" in text   # the rule uses the cross-mark for forbidden examples


def test_prompt_describes_target_noc_match_boost():
    """Matching v2 step 6: occupation-match boost (target_noc_match)
    should be explained so the responder can narrate occupation
    alignment without inventing the concept."""
    text = NEXT_ACTION_RESPONDER_PROMPT
    assert "target_noc_match" in text
    # The doc explains that >0 means same NOC; 0 means no occupation signal
    text_lower = text.lower()
    assert "occupation" in text_lower


# ---------------------------------------------------------------------------
# Projection wire-through -- _build_user_block must serialize the full
# score_explanation including the new nested fields.
# ---------------------------------------------------------------------------
def _present_matches_input(result: dict) -> ResponderInput:
    """Minimal ResponderInput driving the PRESENT_MATCHES path."""
    decision = Decision(
        next_state="present_matches",
        action=ACTION_PRESENT_MATCHES,
        ask_slots=[],
        show_matches=True,
    )
    return ResponderInput(
        user_message="show me jobs",
        decision=decision,
        results=[result],
        training_by_job={},
        next_skill=(None, 0),
        band_signal="strong_or_good",
        requires_consent=False,
        target_role_text="customer service representative",
        resume_facts=None,
    )


def _sample_result() -> dict:
    """A result shaped like the engine returns it, including all the
    Sprint 5 step 3-5 score_explanation additions."""
    return {
        "title": "Front Desk Agent",
        "employer": "Quality Inn",
        "url": "https://example.test/job/123",
        "location": "Sault Ste. Marie, ON",
        "match_band": "stretch",
        "matched_skills": ["customer service", "communication"],
        "missing_skills": ["guest check-in"],
        "credential_warning": None,
        "score_explanation": {
            "matched_skills": ["customer service", "communication"],
            "missing_skills": ["guest check-in"],
            "skill_match_ratio": 0.667,
            "required_matched": ["customer service", "communication"],
            "required_missing": ["guest check-in"],
            "required_match_ratio": 0.667,
            "required_total": 3,
            "preferred_matched": [],
            "preferred_missing": [],
            "preferred_match_ratio": 0.0,
            "preferred_total": 0,
            "title_match_similarity": 0.85,
            "title_match_override": False,
            "recency_days": 4,
            "work_type_fit": "matched",
            "shift_fit": "no_signal",
            "credential_warning_present": False,
            "caps_applied": ["band_capped_by_no_experience"],
            "band_capped_by_no_experience": True,
            "score_components": {
                "skill_base": {
                    "value": 0.667,
                    "mode": "required_only",
                    "required_match_ratio": 0.667,
                    "required_weight": 0.8,
                    "preferred_match_ratio": 0.0,
                    "preferred_weight": 0.2,
                },
                "boosts": {
                    "recency": 0.03,
                    "target_role": 0.02,
                    "target_noc_match": 0.10,   # Step 2 occupation boost
                    "work_type_fit": 0.05,
                    "shift_fit": 0.0,
                },
                "title_match": {
                    "applied": False,
                    "raw_similarity": 0.85,
                },
                "score_pre_caps": 0.807,
                "score_post_caps": 0.41,
            },
        },
    }


def _parse_results_block(user_block: str) -> list[dict]:
    """Extract the JSON lines that follow the RESULTS: header."""
    lines = user_block.splitlines()
    if "RESULTS:" not in lines:
        return []
    start = lines.index("RESULTS:") + 1
    out: list[dict] = []
    for line in lines[start:]:
        line = line.strip()
        if not line or not line.startswith("{"):
            break
        out.append(json.loads(line))
    return out


def test_user_block_serializes_full_score_explanation():
    """The dict that lands in the LLM's user message must contain
    score_components and caps_applied -- otherwise the prompt's
    allow-list references fields the LLM can't see."""
    inp = _present_matches_input(_sample_result())
    user_block = _build_user_block(*_iv1(inp))
    blocks = _parse_results_block(user_block)
    assert len(blocks) == 1
    se = blocks[0]["score_explanation"]

    # Step 3 split fields
    assert "required_matched" in se
    assert "preferred_matched" in se
    assert "required_match_ratio" in se

    # Step 5 nested structure
    assert "score_components" in se
    sc = se["score_components"]
    assert sc["skill_base"]["mode"] == "required_only"
    assert sc["skill_base"]["value"] == 0.667
    assert "boosts" in sc
    assert "title_match" in sc
    assert sc["score_pre_caps"] == 0.807
    assert sc["score_post_caps"] == 0.41

    # caps_applied list and cap-specific flag both present
    assert se["caps_applied"] == ["band_capped_by_no_experience"]
    assert se["band_capped_by_no_experience"] is True


def test_user_block_preserves_caps_applied_ordering():
    """If multiple caps stack, the user block must preserve their order
    so the prompt's 'lead with the most actionable' rule has stable
    input."""
    r = _sample_result()
    r["score_explanation"]["caps_applied"] = [
        "band_capped_by_credential",
        "band_capped_by_no_experience",
        "band_capped_by_work_type_mismatch",
    ]
    inp = _present_matches_input(r)
    user_block = _build_user_block(*_iv1(inp))
    blocks = _parse_results_block(user_block)
    assert blocks[0]["score_explanation"]["caps_applied"] == [
        "band_capped_by_credential",
        "band_capped_by_no_experience",
        "band_capped_by_work_type_mismatch",
    ]


# ---------------------------------------------------------------------------
# Sprint 5 slice 4c -- narration cap on matched/missing skill lists.
# Matcher considers up to top-N+credential-carve-out; narration shows the
# top 3 PLUS any credentials further down the list.
# ---------------------------------------------------------------------------
def test_narration_cap_keeps_first_three_non_credential_skills():
    skills = ["customer service", "phone communication", "computer systems",
              "attention to detail", "communication skills"]
    view = _narration_skill_view(skills)
    assert view == ["customer service", "phone communication", "computer systems"]
    assert len(view) == _NARRATION_SKILL_CAP


def test_narration_cap_force_includes_credential_below_cutoff():
    """The credential at index 5 must survive the cap so the responder
    can narrate the missing-licence gap honestly."""
    skills = [
        "truck service and maintenance",
        "diesel engine diagnosis and repair",
        "emergency repair",
        "computerized diagnostic tools",
        "motor vehicle inspection",
        "Class G driver's license",   # rank 6+, must still appear
    ]
    view = _narration_skill_view(skills)
    assert view[:3] == skills[:3]
    assert "Class G driver's license" in view


def test_narration_cap_force_includes_multiple_credentials_below_cutoff():
    skills = [
        "customer service",
        "communication",
        "teamwork",
        "first aid",
        "CPR-C",
        "WHMIS 2015",
    ]
    view = _narration_skill_view(skills)
    # First 3 by order
    assert view[:3] == ["customer service", "communication", "teamwork"]
    # All three credentials force-included (order preserved)
    assert "first aid" in view
    assert "CPR-C" in view
    assert "WHMIS 2015" in view


def test_narration_cap_does_not_duplicate_credential_already_in_top_three():
    """If a credential is already in the first 3, the force-include
    pass must not append it again."""
    skills = ["Class G driver's license", "welding", "diesel repair",
              "First Aid certification"]
    view = _narration_skill_view(skills)
    assert view.count("Class G driver's license") == 1


def test_narration_cap_empty_and_none_inputs():
    assert _narration_skill_view(None) == []
    assert _narration_skill_view([]) == []


def test_narration_cap_skills_under_cap_pass_through():
    skills = ["welding", "driving"]
    assert _narration_skill_view(skills) == skills


def test_user_block_applies_narration_cap_to_matched_and_missing():
    """End-to-end: _build_user_block's serialization must use the cap."""
    r = _sample_result()
    r["matched_skills"] = [
        "customer service", "phone communication", "computer systems",
        "attention to detail",
    ]
    r["missing_skills"] = [
        "guest check-in", "reservation management",
        "property management system operation", "Class G driver's license",
    ]
    inp = _present_matches_input(r)
    user_block = _build_user_block(*_iv1(inp))
    blocks = _parse_results_block(user_block)
    block = blocks[0]
    # Matched capped at 3 (no credentials present to force-include)
    assert block["matched_skills"] == [
        "customer service", "phone communication", "computer systems",
    ]
    # Missing capped at 3 + force-included credential
    assert block["missing_skills"][:3] == [
        "guest check-in", "reservation management",
        "property management system operation",
    ]
    assert "Class G driver's license" in block["missing_skills"]


def test_user_block_includes_required_top_level_fields_for_results():
    """Sanity: the per-result block still carries everything the
    responder traditionally narrated from (band, matched/missing, URL,
    employer, location, credential_warning)."""
    inp = _present_matches_input(_sample_result())
    user_block = _build_user_block(*_iv1(inp))
    blocks = _parse_results_block(user_block)
    block = blocks[0]
    assert block["title"] == "Front Desk Agent"
    assert block["employer"] == "Quality Inn"
    assert block["url"] == "https://example.test/job/123"
    assert block["match_band"] == "stretch"
    assert "customer service" in block["matched_skills"]
    assert "guest check-in" in block["missing_skills"]


# ---------------------------------------------------------------------------
# Step 6 review fix: narration cap leak through score_explanation.
# Engine's score_explanation can carry up to top-N (12) skills, but the
# responder must see the same capped view as the top-level fields --
# otherwise the LLM sees a long list nested and narrates from it.
# ---------------------------------------------------------------------------
def test_narration_view_with_indices_returns_aligned_positions():
    """The _with_indices helper must return source-index list parallel
    to the kept-skill list -- callers slice strengths/stages by these."""
    skills = ["python", "sql", "docker", "kubernetes",
              "Class G driver's license"]
    view, indices = _narration_skill_view_with_indices(skills)
    # First 3 always kept by position 0, 1, 2
    assert indices[:3] == [0, 1, 2]
    # Credential below the cap (position 4) force-included
    assert 4 in indices
    # Names match indices
    assert view == [skills[i] for i in indices]


def test_capped_score_explanation_caps_top_level_lists():
    """matched_skills / missing_skills inside score_explanation must
    receive the same cap as the top-level fields."""
    se = {
        "matched_skills": ["a", "b", "c", "d", "e"],   # 5 entries
        "missing_skills": ["f", "g", "h", "i"],         # 4 entries
    }
    capped = _capped_score_explanation(se)
    assert len(capped["matched_skills"]) == 3   # _NARRATION_SKILL_CAP
    assert len(capped["missing_skills"]) == 3


def test_capped_score_explanation_aligns_strengths_and_stages():
    """When required_matched is capped, the parallel
    required_match_strengths and required_match_stages arrays must be
    sliced to the same indices so positions still correspond."""
    se = {
        "required_matched": [
            "python", "sql", "docker", "kubernetes", "graphql",
        ],
        "required_match_strengths": [1.0, 1.0, 1.0, 0.85, 0.75],
        "required_match_stages": ["exact", "exact", "exact", "fuzzy", "semantic"],
    }
    capped = _capped_score_explanation(se)
    assert capped["required_matched"] == ["python", "sql", "docker"]
    # Parallel arrays must align to the same positions
    assert capped["required_match_strengths"] == [1.0, 1.0, 1.0]
    assert capped["required_match_stages"] == ["exact", "exact", "exact"]


def test_capped_score_explanation_force_includes_credentials_in_required_matched():
    """A credential ranked below the cap must be force-included in
    required_matched AND its strength/stage carry through."""
    se = {
        "required_matched": [
            "diesel engine repair",         # rank 0  exact
            "vehicle inspection",           # rank 1
            "truck maintenance",            # rank 2
            "emissions testing preparation",  # rank 3 (below cap)
            "Class G driver's license",     # rank 4 credential -- forced
        ],
        "required_match_strengths": [1.0, 1.0, 0.85, 0.85, 1.0],
        "required_match_stages": ["exact", "exact", "fuzzy", "fuzzy", "exact"],
    }
    capped = _capped_score_explanation(se)
    assert "Class G driver's license" in capped["required_matched"]
    # Index of the credential in the capped output
    cred_idx = capped["required_matched"].index("Class G driver's license")
    # Strength/stage at that position must come from the source index 4
    assert capped["required_match_strengths"][cred_idx] == 1.0
    assert capped["required_match_stages"][cred_idx] == "exact"


def test_capped_score_explanation_handles_none_and_empty():
    """None / empty inputs pass through without crashing."""
    assert _capped_score_explanation(None) is None
    assert _capped_score_explanation({}) == {}


def test_capped_score_explanation_preserves_non_list_fields():
    """ratios, counts, caps, boosts must be left alone -- only the
    skill-name lists are capped."""
    se = {
        "required_matched": ["a", "b", "c", "d"],
        "required_match_ratio": 0.5,
        "required_total": 8,
        "caps_applied": ["band_capped_by_no_experience"],
        "score_components": {
            "skill_base": {"value": 0.5, "mode": "required_only"},
            "boosts": {"target_noc_match": 0.10},
        },
    }
    capped = _capped_score_explanation(se)
    assert capped["required_match_ratio"] == 0.5
    assert capped["required_total"] == 8
    assert capped["caps_applied"] == ["band_capped_by_no_experience"]
    assert capped["score_components"]["skill_base"]["value"] == 0.5
    assert capped["score_components"]["boosts"]["target_noc_match"] == 0.10


def test_user_block_does_not_leak_uncapped_lists_through_score_explanation():
    """End-to-end regression: a job with 12 matched required skills
    must appear capped at 3 (plus credentials) BOTH at the top level
    AND inside score_explanation. The LLM should never see the long
    list."""
    r = _sample_result()
    # Bloat score_explanation with 12 skills to simulate post-Step 4a
    # extracted job_skill rows.
    bloated = [
        "skill 1", "skill 2", "skill 3", "skill 4",
        "skill 5", "skill 6", "skill 7", "skill 8",
        "skill 9", "skill 10", "skill 11", "skill 12",
    ]
    r["score_explanation"]["required_matched"] = list(bloated)
    r["score_explanation"]["required_match_strengths"] = [1.0] * 12
    r["score_explanation"]["required_match_stages"] = ["exact"] * 12
    # And the top-level
    r["matched_skills"] = list(bloated)

    inp = _present_matches_input(r)
    user_block = _build_user_block(*_iv1(inp))
    blocks = _parse_results_block(user_block)
    block = blocks[0]
    # Top-level matched_skills capped (already covered by earlier test)
    assert len(block["matched_skills"]) <= _NARRATION_SKILL_CAP + 1
    # The Step-6 review-fix invariant: nested lists also capped
    se = block["score_explanation"]
    assert len(se["required_matched"]) <= _NARRATION_SKILL_CAP + 1, (
        "score_explanation.required_matched leaked uncapped to the responder"
    )
    assert len(se["required_match_strengths"]) == len(se["required_matched"])
    assert len(se["required_match_stages"]) == len(se["required_matched"])
