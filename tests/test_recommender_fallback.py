"""Slice 5 step 4 (2026-06-19) -- recommender_fallback structural pins.

Tests verify deterministic-template output for each of the three
recommender modes. The fallback fires when the LLM-first path can't
render; tests assert the locked voice rules + chain closings hold.
"""
from __future__ import annotations

import pytest

from skillbridge.chat.gap_evidence import (
    GapEvidence,
    RecommenderEvidence,
    TrainingResource,
)
from skillbridge.chat.recommender_fallback import render_recommender_fallback

pytestmark = pytest.mark.nodb


# ---------------------------------------------------------------------------
# Defensive / contract violation
# ---------------------------------------------------------------------------
def test_render_with_none_evidence_returns_safe_text():
    """Handler contract violation case: rec_evidence is None. The
    fallback returns a minimal safe response with no chain close."""
    text = render_recommender_fallback(None)
    assert text  # non-empty
    assert "can't render" in text.lower() or "cannot render" in text.lower()


def test_render_with_unknown_mode_returns_safe_text():
    """Defensive: an unknown mode override (shouldn't happen given the
    Literal) still produces a safe response."""
    rec = RecommenderEvidence(
        mode="local_gap_coach", evidence=(), training=(),
    )
    text = render_recommender_fallback(rec, mode="some_unknown_mode")
    assert text
    assert "can't render" in text.lower() or "cannot render" in text.lower()


# ---------------------------------------------------------------------------
# local_gap_coach
# ---------------------------------------------------------------------------
def _gap(layer, skill_id, skill_name, importance=None, blocker=False,
         source="reference.noc_skill", source_id="14200",
         source_label="Accounting clerk"):
    return GapEvidence(
        layer=layer, source_id=source_id, source_label=source_label,
        skill_id=skill_id, skill_name=skill_name, blocker=blocker,
        importance=importance, source=source,
    )


def test_local_gap_coach_empty_evidence():
    """Slice 2: when Layer B evidence is empty AND the fallback is
    invoked directly (defensive path -- handler's slice 2 branches
    normally emit canned text BEFORE reaching the responder), the
    fallback honestly says nothing surfaced and emits the new
    B->C chain close."""
    rec = RecommenderEvidence(
        mode="local_gap_coach", evidence=(), training=(),
    )
    text = render_recommender_fallback(rec)
    assert "didn't surface" in text.lower() or "don't have" in text.lower()
    # Slice 2 chain close: B offers C, not A.
    assert "related career paths" in text


def test_local_gap_coach_with_evidence_and_no_training():
    """Layer B has a gap but no verified TrainingResource for it.
    Fallback honestly says ask the SCCC -- never invents a provider."""
    gap = _gap(
        layer="local_posting", skill_id="S_BANK",
        skill_name="bank reconciliation",
        source="extracted.job_skill",
        source_id="job-abc-123",
        source_label="Accounting Clerk @ Diamond J Farms",
    )
    rec = RecommenderEvidence(
        mode="local_gap_coach", evidence=(gap,), training=(),
    )
    text = render_recommender_fallback(rec)
    assert "bank reconciliation" in text
    assert "Sault Community Career Centre" in text
    # Slice 2 chain close: B offers C.
    assert "related career paths" in text
    # No invented provider names.
    assert "Sault College" not in text
    assert "https://" not in text  # no URL when no training


def test_local_gap_coach_with_evidence_and_matching_training():
    """Layer B has a gap and a matching TrainingResource. Fallback
    names the provider + summary + URL verbatim from the training
    record."""
    gap = _gap(
        layer="local_posting", skill_id="S_BANK",
        skill_name="bank reconciliation",
        source="extracted.job_skill",
        source_id="job-abc-123",
        source_label="Accounting Clerk @ Diamond J Farms",
    )
    training = TrainingResource(
        skill_id="S_BANK",
        skill_name="bank reconciliation",
        provider="Sault College",
        type="local_training",
        url="https://saultcollege.ca/bookkeeping",
        summary="Bookkeeping fundamentals course",
    )
    rec = RecommenderEvidence(
        mode="local_gap_coach", evidence=(gap,), training=(training,),
    )
    text = render_recommender_fallback(rec)
    assert "Sault College" in text
    assert "Bookkeeping fundamentals course" in text
    assert "https://saultcollege.ca/bookkeeping" in text
    # Slice 2 chain close: B offers C.
    assert "related career paths" in text


def test_local_gap_coach_training_attached_by_skill_id_preferred():
    """When skill_id matches, that takes precedence over name."""
    gap = _gap(
        layer="local_posting", skill_id="S_BANK",
        skill_name="bank reconciliation",
        source="extracted.job_skill",
    )
    training_wrong_name = TrainingResource(
        skill_id="S_BANK",
        skill_name="completely different name",
        provider="Sault College",
        type="local_training",
        url="https://saultcollege.ca/bookkeeping",
        summary="Bookkeeping fundamentals",
    )
    rec = RecommenderEvidence(
        mode="local_gap_coach",
        evidence=(gap,),
        training=(training_wrong_name,),
    )
    text = render_recommender_fallback(rec)
    assert "Sault College" in text  # skill_id won the match


def test_local_gap_coach_training_falls_back_to_name_when_skill_id_null():
    """When the gap has skill_id=None (Layer B with unresolved
    extractor name), training attaches by case-insensitive name
    match."""
    gap = _gap(
        layer="local_posting", skill_id=None,
        skill_name="QuickBooks Desktop",
        source="extracted.job_skill",
    )
    training = TrainingResource(
        skill_id=None,
        skill_name="quickbooks desktop",  # different case
        provider="Sault College",
        type="local_training",
        url="https://saultcollege.ca/quickbooks",
        summary="QuickBooks Desktop course",
    )
    rec = RecommenderEvidence(
        mode="local_gap_coach", evidence=(gap,), training=(training,),
    )
    text = render_recommender_fallback(rec)
    assert "Sault College" in text


# ---------------------------------------------------------------------------
# target_noc_standard
# ---------------------------------------------------------------------------
def test_target_noc_standard_empty_evidence():
    """Slice 2: Layer A is intent-only and closes NATURAL (no chain).
    Empty evidence is rare for A in production but the defensive
    fallback should not advertise a chain."""
    rec = RecommenderEvidence(
        mode="target_noc_standard", evidence=(), training=(),
    )
    text = render_recommender_fallback(rec)
    assert "don't have" in text.lower() or "no" in text.lower()
    assert "standard skill profile" in text.lower()
    # Slice 2: natural close, no chain offer.
    assert "Anything in there" in text
    # No chain offer to other layers.
    assert "related career paths" not in text


def test_target_noc_standard_single_skill():
    gap = _gap(
        layer="target_noc_standard",
        skill_id="F.01.b.01",
        skill_name="Reading Comprehension",
        importance=4.0,
        source_id="14200",
        source_label="Accounting clerk",
    )
    rec = RecommenderEvidence(
        mode="target_noc_standard", evidence=(gap,), training=(),
    )
    text = render_recommender_fallback(rec)
    assert "Reading Comprehension" in text
    # development-area voice -- NEVER deficit voice
    assert "emphasizes" in text.lower()
    body = text.lower()
    assert "you don't have" not in body
    assert "you lack" not in body
    assert "you're missing" not in body
    assert "you can't" not in body
    assert "is a gap" not in body
    # Slice 2: A closes natural (no chain).
    assert "Anything in there" in text
    assert "related career paths" not in text


def test_target_noc_standard_three_skills_with_oxford_comma():
    """Top-3 by importance is the locked cap. Three names get the
    Oxford comma."""
    gaps = (
        _gap(layer="target_noc_standard", skill_id="F.01.b.01",
             skill_name="Reading Comprehension", importance=4.5,
             source_id="14200", source_label="Accounting clerk"),
        _gap(layer="target_noc_standard", skill_id="F.02.a.01",
             skill_name="Critical Thinking", importance=4.0,
             source_id="14200", source_label="Accounting clerk"),
        _gap(layer="target_noc_standard", skill_id="F.01.c.01",
             skill_name="Numeracy", importance=3.8,
             source_id="14200", source_label="Accounting clerk"),
    )
    rec = RecommenderEvidence(
        mode="target_noc_standard", evidence=gaps, training=(),
    )
    text = render_recommender_fallback(rec)
    assert "Reading Comprehension" in text
    assert "Critical Thinking" in text
    assert "Numeracy" in text
    assert ", and Numeracy" in text  # Oxford comma
    assert "emphasizes" in text.lower()
    # Slice 2: A closes natural (no chain).
    assert "Anything in there" in text
    assert "related career paths" not in text


def test_target_noc_standard_no_forbidden_deficit_phrases():
    """Critical voice contract: never emit forbidden deficit
    phrasings on OaSIS broad competencies."""
    gaps = (
        _gap(layer="target_noc_standard", skill_id="F.01.b.01",
             skill_name="Reading Comprehension", importance=4.5,
             source_id="14200", source_label="Accounting clerk"),
    )
    rec = RecommenderEvidence(
        mode="target_noc_standard", evidence=gaps, training=(),
    )
    text = render_recommender_fallback(rec).lower()
    for forbidden in (
        "you don't have", "you lack", "you're missing", "you can't",
        "is a gap", "you need to learn", "you should improve",
    ):
        assert forbidden not in text, (
            f"forbidden deficit phrase {forbidden!r} appeared in "
            f"target_noc_standard fallback"
        )


# ---------------------------------------------------------------------------
# adjacent_noc_standard
# ---------------------------------------------------------------------------
def test_adjacent_noc_standard_empty_evidence():
    """No adjacency surfaced -- fallback acknowledges honestly and
    closes naturally (chain ENDS HERE)."""
    rec = RecommenderEvidence(
        mode="adjacent_noc_standard", evidence=(), training=(),
    )
    text = render_recommender_fallback(rec)
    assert "nothing surfaced" in text.lower()
    # Chain ENDS here -- no further mode offer.
    assert "Canadian/NOC standard" not in text  # not chained
    assert "related career paths" not in text  # not chained
    # But a natural follow-up close.
    assert "?" in text


def test_adjacent_noc_standard_single_noc():
    gaps = (
        _gap(layer="adjacent_noc_standard", skill_id="F.01.b.02",
             skill_name="Writing", importance=4.0,
             source_id="13110", source_label="Administrative assistant"),
        _gap(layer="adjacent_noc_standard", skill_id="F.05.a.01",
             skill_name="Coordinating", importance=3.5,
             source_id="13110", source_label="Administrative assistant"),
    )
    rec = RecommenderEvidence(
        mode="adjacent_noc_standard", evidence=gaps, training=(),
    )
    text = render_recommender_fallback(rec)
    assert "Administrative assistant" in text
    assert "Writing" in text
    assert "Coordinating" in text
    assert "If you wanted to move toward" in text  # exploratory voice
    assert "emphasizes" in text.lower()  # development-area voice
    # Slice 5 (2026-06-29): Layer C close changed from natural
    # follow-up to explicit drilldown offer.
    assert "skill-by-skill comparison" in text.lower()
    assert "say which one" in text.lower()


def test_adjacent_noc_standard_multiple_nocs_grouped():
    """Layer C fan-out: records grouped by source_id, one paragraph
    per unique NOC, first-seen order preserved."""
    gaps = (
        # NOC 14200 group
        _gap(layer="adjacent_noc_standard", skill_id="F.01.c.01",
             skill_name="Numeracy", importance=4.5,
             source_id="14200", source_label="Cost clerk"),
        # NOC 13100 group
        _gap(layer="adjacent_noc_standard", skill_id="F.05.a.01",
             skill_name="Coordinating", importance=4.0,
             source_id="13100", source_label="Business services officer"),
        # Second skill back in NOC 14200 group (should join 14200's paragraph)
        _gap(layer="adjacent_noc_standard", skill_id="F.02.a.01",
             skill_name="Critical Thinking", importance=3.5,
             source_id="14200", source_label="Cost clerk"),
    )
    rec = RecommenderEvidence(
        mode="adjacent_noc_standard", evidence=gaps, training=(),
    )
    text = render_recommender_fallback(rec)
    assert "Cost clerk" in text
    assert "Business services officer" in text
    # NOC 14200's two skills should both appear in its paragraph.
    cost_idx = text.find("Cost clerk")
    biz_idx = text.find("Business services officer")
    assert cost_idx < biz_idx  # first-seen order
    cost_chunk = text[cost_idx:biz_idx]
    assert "Numeracy" in cost_chunk
    assert "Critical Thinking" in cost_chunk  # grouped with same NOC
    # NOC 13100's skill appears in its paragraph.
    biz_chunk = text[biz_idx:]
    assert "Coordinating" in biz_chunk
    # Slice 5 (2026-06-29): close is the explicit drilldown offer.
    assert "skill-by-skill comparison" in text.lower()


def test_adjacent_noc_standard_no_forbidden_deficit_phrases():
    gaps = (
        _gap(layer="adjacent_noc_standard", skill_id="F.01.b.01",
             skill_name="Reading Comprehension", importance=4.5,
             source_id="13110", source_label="Administrative assistant"),
    )
    rec = RecommenderEvidence(
        mode="adjacent_noc_standard", evidence=gaps, training=(),
    )
    text = render_recommender_fallback(rec).lower()
    for forbidden in (
        "you don't have", "you lack", "you're missing", "you can't",
        "is a gap", "you need to learn", "you should improve",
    ):
        assert forbidden not in text, (
            f"forbidden deficit phrase {forbidden!r} appeared in "
            f"adjacent_noc_standard fallback"
        )


def test_adjacent_noc_standard_does_NOT_chain_to_another_mode():
    """Chain ENDS at adjacent. The fallback must NOT include the
    locked chain-close strings from earlier modes."""
    gaps = (
        _gap(layer="adjacent_noc_standard", skill_id="F.01.b.01",
             skill_name="Reading Comprehension", importance=4.5,
             source_id="13110", source_label="Administrative assistant"),
    )
    rec = RecommenderEvidence(
        mode="adjacent_noc_standard", evidence=gaps, training=(),
    )
    text = render_recommender_fallback(rec)
    assert "Canadian/NOC standard" not in text
    assert "related career paths" not in text
