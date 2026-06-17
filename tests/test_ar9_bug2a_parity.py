"""AR-9.bug.2a sub-step 4 parity tests.

Each expected output below is a reviewed literal — no generator script.
Failures here mean the migrated function diverges from the pre-migration
behavior on a non-URL field (URL divergence is permitted only for
invalid source URLs being stripped, per the locked parity contract).

Tests document the projection-normalization rules locked in sub-step 4
revisions 4-5:
  - score_explanation {} -> "score_explanation": null
  - V2 training drops unknown raw fields and omits None fields
  - Adjacent role job={} expands to 6 null fields
  - Adjacent recommendation filtering (non-dict / empty title / url field)
  - posted_date numeric scalars stringified
  - URL rendering shapes per path (match indented, training em-dash,
    registry parenthesized)
"""
from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.nodb


from skillbridge.chat.arbiter import ArbiterDecision
from skillbridge.chat.responder import (
    ResponderV2Input,
    _build_user_block_v2,
    _present_matches_fallback_v2,
    _registry_grounded_explain_gap_fallback,
)
from skillbridge.chat.url_views import (
    build_sanitized_responder_view_v2 as _v_v2,
)
from tests._view_fixtures import (
    _extract_json_block,
    _extract_training_json_objects,
)


# =========================================================================
# Helpers
# =========================================================================
def _decision(move: str, **kw) -> ArbiterDecision:
    return ArbiterDecision(
        final_move=move,
        reason_code=kw.get("reason_code", "x"),
        tone=kw.get("tone", "brief_confident"),
        arbiter_action=kw.get("arbiter_action", "passed_planner_through"),
        ask_slot=kw.get("ask_slot"),
        caps_applied=kw.get("caps_applied", ()),
    )


def _v2_input(
    move: str,
    results=None,
    training_by_job=None,
    band_signal="strong_or_good",
    **payloads,
) -> ResponderV2Input:
    return ResponderV2Input(
        user_message="hi",
        decision=_decision(move),
        results=results or [],
        training_by_job=training_by_job or {},
        next_skill=(None, 0),
        band_signal=band_signal,
        requires_consent=False,
        target_role_text="warehouse worker",
        resume_facts=None,
        conversation_context=None,
        **payloads,
    )


# =========================================================================
# Score-explanation normalization: {} -> null in result JSON
# =========================================================================
def test_score_explanation_empty_dict_normalizes_to_null():
    """The result serializer always includes the 'score_explanation' key
    (parity with current behavior at responder.py:1389). When the raw
    value is {}, the projected view is None, so the serialized output
    is "score_explanation": null. Documents the {} -> None normalization
    at the JSON layer.
    """
    inp = _v2_input(
        "present_matches",
        results=[{
            "title": "T", "employer": "E", "url": "https://x.com/y",
            "score_explanation": {},
            "job_id": "j", "matched_skills": [], "missing_skills": [],
            "match_band": "good",
        }],
    )
    view = _v_v2(inp)
    out = _build_user_block_v2(inp, view)
    assert '"score_explanation": null' in out


def test_score_explanation_partial_dict_gains_empty_list_fields():
    """A partial score_explanation dict (only one tuple field present)
    gains empty-list fields for the other always-list keys per the
    locked projection-normalization contract.
    """
    inp = _v2_input(
        "present_matches",
        results=[{
            "title": "T", "employer": "E", "url": "https://x.com/y",
            "job_id": "j", "match_band": "good",
            "matched_skills": [], "missing_skills": [],
            "score_explanation": {"matched_skills": ["a"]},
        }],
    )
    view = _v_v2(inp)
    out = _build_user_block_v2(inp, view)
    # Locate the RESULTS JSON to inspect score_explanation
    lines = out.split("\n")
    res_idx = lines.index("RESULTS:")
    result_obj = json.loads(lines[res_idx + 1])
    se = result_obj["score_explanation"]
    assert se is not None
    assert se["matched_skills"] == ["a"]
    # All other tuple fields are present as empty lists
    for empty_key in (
        "missing_skills", "required_matched", "required_missing",
        "preferred_matched", "preferred_missing",
    ):
        assert se[empty_key] == [], empty_key


# =========================================================================
# V2 training: drop unknown raw fields, omit None
# =========================================================================
def test_v2_training_drops_unknown_raw_fields():
    """A raw training dict with a field not in TrainingView produces
    JSON omitting that key — dataclass-as-allowlist enforcement.

    Bug.4: on present_matches turns, TRAINING is grouped per-job:
    each top-level object is {"job_id", "job_title", "resources": [...]}.
    The 11-field TrainingView allowlist is enforced INSIDE each
    `resources[i]` dict.
    """
    inp = _v2_input(
        "present_matches",
        results=[{
            "title": "T", "employer": "E", "url": "https://x.com/y",
            "job_id": "j", "match_band": "good",
            "matched_skills": [], "missing_skills": [],
        }],
        training_by_job={"j": [{
            "provider": "P", "title": "T",
            "url": "https://x.com/training",
            "for_skill": "X",
            "experimental_field": "should_disappear",
            "another_unknown": 42,
        }]},
    )
    view = _v_v2(inp)
    out = _build_user_block_v2(inp, view)
    groups = _extract_training_json_objects(out)
    assert len(groups) == 1
    g = groups[0]
    # Group-level shape
    assert g["job_id"] == "j"
    assert g["job_title"] == "T"
    assert len(g["resources"]) == 1
    obj = g["resources"][0]
    # Unknown fields dropped from the resource entry
    assert "experimental_field" not in obj
    assert "another_unknown" not in obj
    assert "should_disappear" not in obj.values()
    # Approved fields present
    assert obj["provider"] == "P"
    assert obj["url"] == "https://x.com/training"


def test_v2_training_omits_none_fields():
    """V2 training serializer omits None fields (vs V1 which emits null)."""
    inp = _v2_input(
        "explain_gap",
        training_by_job={"j": [{
            "provider": "P", "title": "T",
            "url": "https://x.com/training",
            # for_skill omitted -> None in view -> key omitted in JSON
        }]},
    )
    view = _v_v2(inp)
    out = _build_user_block_v2(inp, view)
    objects = _extract_training_json_objects(out)
    assert len(objects) == 1
    obj = objects[0]
    # for_skill, duration_band, etc. all omitted (not present as null)
    for omitted in (
        "for_skill", "duration_band", "resource_type", "reason",
        "type", "for_gap", "summary", "verified",
    ):
        assert omitted not in obj, omitted


# =========================================================================
# Adjacent role: job={} expands to 6 null fields
# =========================================================================
def test_adjacent_role_empty_job_dict_expands_to_six_null_fields():
    """{} for the job dict triggers job_is_mapping=True and the
    serializer emits the full 6-field allowlist with all nulls.
    Documented divergence from current `dict(job)` which emits `{}`.
    """
    inp = _v2_input(
        "describe_adjacent_role",
        adjacent_role_description_payload={
            "job": {}, "evidence_summary": "ev",
            "matched_skills": [], "expired": False,
        },
    )
    view = _v_v2(inp)
    out = _build_user_block_v2(inp, view)
    block = _extract_json_block(out, "ADJACENT_ROLE_DESCRIPTION:")
    assert block is not None
    job = block["job"]
    assert job is not None
    assert set(job.keys()) == {
        "job_id", "title", "employer", "location", "url", "posted_date",
    }
    assert all(v is None for v in job.values())


def test_adjacent_role_missing_job_is_null():
    """payload with job=None -> serialized as "job": null."""
    inp = _v2_input(
        "describe_adjacent_role",
        adjacent_role_description_payload={
            "job": None, "evidence_summary": "ev",
            "matched_skills": [], "expired": False,
        },
    )
    view = _v_v2(inp)
    out = _build_user_block_v2(inp, view)
    block = _extract_json_block(out, "ADJACENT_ROLE_DESCRIPTION:")
    assert block is not None
    assert block["job"] is None


# =========================================================================
# Adjacent recommendation filtering
# =========================================================================
def test_adjacent_recommendation_filtering_documented():
    """Filters malformed entries, drops url and unknown fields,
    normalizes matched_skills element types.
    """
    inp = _v2_input(
        "recommend_adjacent_roles",
        adjacent_recommendations_payload={
            "recommendations": [
                # Canonical entry
                {"job_id": "j1", "title": "Welder", "employer": "ACME",
                 "location": "loc", "evidence_summary": "3 of 5",
                 "why_adjacent": "skill_evidence",
                 "matched_skills": ["welding"]},
                # Non-dict — filtered out
                "garbage",
                # Empty title — filtered out
                {"title": ""},
                # Extra url field + non-string mid in matched_skills
                {"job_id": "j2", "title": "Tech", "employer": "B",
                 "location": "loc", "evidence_summary": "ev",
                 "why_adjacent": "same_noc_minor_group",
                 "matched_skills": ["x", 42, "y"],
                 "url": "https://leaked.example.com"},
            ],
            "total_retrieved": 4,
        },
    )
    view = _v_v2(inp)
    out = _build_user_block_v2(inp, view)
    block = _extract_json_block(out, "ADJACENT_RECOMMENDATIONS:")
    assert block is not None
    recs = block["recommendations"]
    # Two canonical-or-recovered entries survive
    assert len(recs) == 2
    # url stripped from the second
    assert "url" not in recs[1]
    # Non-string filtered from matched_skills
    assert recs[1]["matched_skills"] == ["x", "y"]
    # total_retrieved preserved as int
    assert block["total_retrieved"] == 4


# =========================================================================
# posted_date numeric stringification (intentional normalization)
# =========================================================================
def test_posted_date_numeric_stringification_documented():
    """The view normalizes non-string posted_date scalars to str.
    Documented divergence from current `isinstance(pd, (str, int, float,
    bool))` defensive preservation.
    """
    inp = _v2_input(
        "describe_adjacent_role",
        adjacent_role_description_payload={
            "job": {"title": "T", "posted_date": 123},
            "evidence_summary": "", "matched_skills": [], "expired": False,
        },
    )
    view = _v_v2(inp)
    out = _build_user_block_v2(inp, view)
    block = _extract_json_block(out, "ADJACENT_ROLE_DESCRIPTION:")
    assert block is not None
    assert block["job"]["posted_date"] == "123"   # stringified


# =========================================================================
# URL rendering shapes — per-path locked forms
# =========================================================================
def test_present_matches_fallback_v2_url_rendering_indented_separate_line():
    """Match URL renders on an indented separate line, NO parentheses
    (responder.py:830-831, 2503-2504).
    Training URL renders with em-dash inline append (responder.py:842, 2515).
    """
    inp = _v2_input(
        "present_matches",
        results=[{
            "title": "Welder", "employer": "ACME",
            "url": "https://example.com/jobs/123",
            "job_id": "j1", "match_band": "good",
            "matched_skills": [], "missing_skills": [],
        }],
        training_by_job={"j1": [{
            "title": "T",
            "provider": "P",
            "url": "https://example.com/training",
            "for_skill": "X",
        }]},
    )
    view = _v_v2(inp)
    out = _present_matches_fallback_v2(inp, view)
    # Match URL on indented separate line (3-space prefix, no parens)
    assert "   https://example.com/jobs/123" in out
    assert "(https://example.com/jobs/123)" not in out
    # Training URL em-dash inline
    assert "— https://example.com/training" in out


def test_present_matches_fallback_v2_omits_url_when_absent():
    """Absent URL on result -> the URL line is skipped entirely
    (not rendered as null or empty)."""
    inp = _v2_input(
        "present_matches",
        results=[{
            "title": "Welder", "employer": "ACME",
            # url absent
            "job_id": "j1", "match_band": "good",
            "matched_skills": [], "missing_skills": [],
        }],
    )
    view = _v_v2(inp)
    out = _present_matches_fallback_v2(inp, view)
    # Output has the result line but no URL line follows
    assert "Welder at ACME" in out
    assert "https://" not in out
    assert "null" not in out


def test_registry_grounded_explain_gap_url_rendering_parenthesized():
    """Registry URL renders parenthesized inline (responder.py:2131-2132)."""
    inp = _v2_input(
        "explain_gap",
        training_by_job={"j1": [{
            "provider": "Sault College",
            "summary": "12-week truck training",
            "for_gap": "Class G licence",
            "url": "https://example.com/sault-truck",
        }]},
    )
    view = _v_v2(inp)
    out = _registry_grounded_explain_gap_fallback(inp, view)
    # Parenthesized URL form
    assert "(https://example.com/sault-truck)" in out


# =========================================================================
# Invalid source URL deliberately stripped from rendered output
# =========================================================================
def test_invalid_source_url_stripped_from_fallback():
    """A source URL that fails structural validation (ftp:// scheme)
    is stripped at projection. The migrated fallback omits the URL
    line rather than rendering the invalid URL.

    This is the intentional divergence at the heart of bug.2a:
    invalid URLs in payload data disappear from output.
    """
    inp = _v2_input(
        "present_matches",
        results=[{
            "title": "Welder", "employer": "ACME",
            "url": "ftp://bad.example.com",  # unsupported scheme
            "job_id": "j1", "match_band": "good",
            "matched_skills": [], "missing_skills": [],
        }],
    )
    view = _v_v2(inp)
    out = _present_matches_fallback_v2(inp, view)
    assert "Welder at ACME" in out
    # Invalid URL stripped — not rendered in any form
    assert "ftp://" not in out
    assert "bad.example.com" not in out
