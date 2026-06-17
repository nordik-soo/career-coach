"""AR-9.bug.4: V2 present_matches training grouped by owning job.

Bug.4 added a PromptJobTrainingGroup wrapper so the V2 prompt carries
per-job training attribution STRUCTURALLY rather than as an instruction
the LLM might ignore. This file pins the V2 grouping semantics and
the V1 invariance (V1 path stays flat, byte-for-byte unchanged).
"""
from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.nodb


from skillbridge.chat.arbiter import ArbiterDecision
from skillbridge.chat.intake_state import ACTION_PRESENT_MATCHES, Decision
from skillbridge.chat.prompts import OUTCOME_RESPONDER_PROMPT
from skillbridge.chat.responder import (
    ResponderInput,
    ResponderV2Input,
    _build_user_block,
    _build_user_block_v2,
)
from skillbridge.chat.url_views import (
    PromptJobTrainingGroup,
    build_sanitized_responder_view_v1,
    build_sanitized_responder_view_v2,
)
from tests._view_fixtures import _extract_training_json_objects


# =========================================================================
# Helpers
# =========================================================================
def _v2_input(
    results=None, training_by_job=None, band_signal="strong_or_good",
) -> ResponderV2Input:
    return ResponderV2Input(
        user_message="hi",
        decision=ArbiterDecision(
            final_move="present_matches",
            reason_code="x",
            tone="brief_confident",
            arbiter_action="passed_planner_through",
            ask_slot=None,
            caps_applied=(),
        ),
        results=results or [],
        training_by_job=training_by_job or {},
        next_skill=(None, 0),
        band_signal=band_signal,
        requires_consent=False,
        target_role_text="warehouse worker",
        resume_facts=None,
        conversation_context=None,
    )


def _v1_input(
    results=None, training_by_job=None, band_signal="strong_or_good",
) -> ResponderInput:
    return ResponderInput(
        user_message="hi",
        decision=Decision(
            next_state="present_matches",
            action=ACTION_PRESENT_MATCHES,
            ask_slots=[],
            show_matches=True,
        ),
        results=results or [],
        training_by_job=training_by_job or {},
        next_skill=(None, 0),
        band_signal=band_signal,
        requires_consent=False,
        target_role_text="warehouse worker",
        resume_facts=None,
    )


def _good_training(suffix: str = "/x") -> dict:
    return {
        "provider": "P", "title": "T",
        "url": f"https://example.com{suffix}",
        "for_skill": "skill",
    }


# =========================================================================
# Prompt grounding rule
# =========================================================================
def test_outcome_responder_prompt_contains_training_grouping_rule():
    """The V2 prompt must carry the locked rule wording so the LLM is
    told to narrate training only with its owning job.
    """
    # The wording spans multiple lines in the docstring so substring
    # checks ignore the line breaks the text has when authored.
    flattened = " ".join(OUTCOME_RESPONDER_PROMPT.split())
    assert "TRAINING is grouped by owning job" in flattened
    assert "ONLY in the context of its owning job" in flattened
    assert "Do not list training globally" in flattened
    assert "do not invent training-to-role mappings" in flattened


# =========================================================================
# V2 grouped structure
# =========================================================================
def test_v2_training_grouping_emits_one_object_per_job_with_resources_array():
    inp = _v2_input(
        results=[
            {"title": "Forklift Operator", "url": "https://j.com/1",
             "job_id": "j1", "match_band": "good",
             "matched_skills": [], "missing_skills": []},
            {"title": "Warehouse Lead", "url": "https://j.com/2",
             "job_id": "j2", "match_band": "good",
             "matched_skills": [], "missing_skills": []},
        ],
        training_by_job={
            "j1": [_good_training("/whmis")],
            "j2": [_good_training("/forklift")],
        },
    )
    view = build_sanitized_responder_view_v2(inp)
    out = _build_user_block_v2(inp, view)
    groups = _extract_training_json_objects(out)
    assert len(groups) == 2
    assert groups[0]["job_id"] == "j1"
    assert groups[0]["job_title"] == "Forklift Operator"
    assert isinstance(groups[0]["resources"], list)
    assert groups[1]["job_id"] == "j2"
    assert groups[1]["job_title"] == "Warehouse Lead"


def test_v2_orphan_training_dropped():
    """Training entries for a job_id not in results[:5] are dropped.
    Their URLs do not appear in prompt_urls either.
    """
    inp = _v2_input(
        results=[{
            "title": "Real Job", "url": "https://r.com/x",
            "job_id": "real-job", "match_band": "good",
            "matched_skills": [], "missing_skills": [],
        }],
        training_by_job={
            "real-job": [_good_training("/keep")],
            "orphan-job": [_good_training("/drop")],
        },
    )
    view = build_sanitized_responder_view_v2(inp)
    out = _build_user_block_v2(inp, view)
    # Only one group (the real job's)
    groups = _extract_training_json_objects(out)
    assert len(groups) == 1
    assert groups[0]["job_id"] == "real-job"
    # Orphan URL not in allowlist
    assert "https://example.com/drop" not in view.prompt_urls
    assert "https://example.com/keep" in view.prompt_urls


def test_v2_resource_cap_is_6_across_groups():
    """6-resource cap applies across groups in result order.
    4 jobs × 3 trainings = 12 total; only 6 reach the prompt.
    """
    inp = _v2_input(
        results=[
            {"title": f"T{i}", "url": f"https://j.com/{i}",
             "job_id": f"j{i}", "match_band": "good",
             "matched_skills": [], "missing_skills": []}
            for i in range(4)
        ],
        training_by_job={
            f"j{i}": [_good_training(f"/r{i}-{k}") for k in range(3)]
            for i in range(4)
        },
    )
    view = build_sanitized_responder_view_v2(inp)
    groups = view.prompt_present_matches_training_groups
    total = sum(len(g.resources) for g in groups)
    assert total == 6
    # j0 takes 3, j1 takes 3, j2 and j3 get nothing
    assert len(groups) == 2
    assert groups[0].job_id == "j0"
    assert len(groups[0].resources) == 3
    assert groups[1].job_id == "j1"
    assert len(groups[1].resources) == 3


def test_v2_groups_in_result_order():
    """Group order matches results[:5] order, even if training_by_job
    dict has a different insertion order.
    """
    inp = _v2_input(
        results=[
            {"title": "A", "url": "https://j.com/a",
             "job_id": "ja", "match_band": "good",
             "matched_skills": [], "missing_skills": []},
            {"title": "B", "url": "https://j.com/b",
             "job_id": "jb", "match_band": "good",
             "matched_skills": [], "missing_skills": []},
        ],
        # Insert jb BEFORE ja to verify result-order wins
        training_by_job={
            "jb": [_good_training("/tb")],
            "ja": [_good_training("/ta")],
        },
    )
    view = build_sanitized_responder_view_v2(inp)
    groups = view.prompt_present_matches_training_groups
    assert len(groups) == 2
    assert groups[0].job_id == "ja"  # result order
    assert groups[1].job_id == "jb"


def test_v2_empty_groups_dropped():
    """A result whose job_id has no training entries contributes no group."""
    inp = _v2_input(
        results=[
            {"title": "A", "url": "https://j.com/a",
             "job_id": "ja", "match_band": "good",
             "matched_skills": [], "missing_skills": []},
            {"title": "B", "url": "https://j.com/b",
             "job_id": "jb", "match_band": "good",
             "matched_skills": [], "missing_skills": []},
        ],
        training_by_job={"ja": [_good_training("/ta")]},  # jb has nothing
    )
    view = build_sanitized_responder_view_v2(inp)
    groups = view.prompt_present_matches_training_groups
    assert len(groups) == 1
    assert groups[0].job_id == "ja"


# =========================================================================
# V2 prompt_urls — bug.4 changes its content intentionally
# =========================================================================
def test_v2_prompt_urls_excludes_orphans_and_capped_trainings():
    """prompt_urls now reflects what the V2 prompt actually surfaces:
    orphans excluded, cap-excluded resources excluded.
    """
    inp = _v2_input(
        results=[{
            "title": "Real", "url": "https://r.com/x",
            "job_id": "j1", "match_band": "good",
            "matched_skills": [], "missing_skills": [],
        }],
        training_by_job={
            "j1": [_good_training(f"/keep-{k}") for k in range(8)],  # 8; cap drops 2
            "orphan": [_good_training("/orphan-drop")],
        },
    )
    view = build_sanitized_responder_view_v2(inp)
    # First 6 j1 trainings present
    for k in range(6):
        assert f"https://example.com/keep-{k}" in view.prompt_urls
    # Last 2 j1 trainings excluded (cap)
    for k in range(6, 8):
        assert f"https://example.com/keep-{k}" not in view.prompt_urls
    # Orphan excluded
    assert "https://example.com/orphan-drop" not in view.prompt_urls


# =========================================================================
# V1 invariance — bug.4 does NOT change v1 behavior
# =========================================================================
def test_v1_training_still_flat_after_bug4():
    """V1 builder populates prompt_present_matches_training_flat with
    legacy first-6-across-training_by_job.values() iteration. The new
    `prompt_present_matches_training_groups` slot is empty for v1.
    """
    inp = _v1_input(
        results=[
            {"title": "T1", "employer": "E", "url": "https://j.com/1",
             "job_id": "j1", "match_band": "good",
             "matched_skills": [], "missing_skills": []},
        ],
        training_by_job={
            "j1": [_good_training(f"/r{k}") for k in range(3)],
            # Orphan training under a non-result job_id — v1 INCLUDES
            # these (legacy flat behavior); v2 would drop them
            "orphan-job": [_good_training("/orphan")],
        },
    )
    view = build_sanitized_responder_view_v1(inp)
    # Flat slot populated; groups slot empty
    assert len(view.prompt_present_matches_training_flat) == 4  # 3 + 1 orphan
    assert view.prompt_present_matches_training_groups == ()
    # Orphan URL IS in v1 prompt_urls (legacy behavior preserved)
    assert "https://example.com/orphan" in view.prompt_urls


def test_v1_prompt_emits_flat_training_json_lines_not_groups():
    """V1 prompt's TRAINING block remains the legacy flat-line shape.
    Each line is a single training entry JSON, not a group object.
    """
    inp = _v1_input(
        results=[
            {"title": "T", "employer": "E", "url": "https://j.com/1",
             "job_id": "j1", "match_band": "good",
             "matched_skills": [], "missing_skills": []},
        ],
        training_by_job={"j1": [_good_training("/r")]},
    )
    view = build_sanitized_responder_view_v1(inp)
    out = _build_user_block(inp, view)
    objects = _extract_training_json_objects(out)
    assert len(objects) == 1
    obj = objects[0]
    # Legacy flat shape — provider at top level, no "resources" key
    assert obj["provider"] == "P"
    assert obj["url"] == "https://example.com/r"
    assert "resources" not in obj
    assert "job_id" not in obj  # V1 flat shape doesn't carry job_id
