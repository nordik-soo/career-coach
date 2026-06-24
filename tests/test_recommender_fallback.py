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
    """Slice 2.5: Layer B's fallback now closes with the
    related-career-paths offer (B -> C in the new chain)."""
    rec = RecommenderEvidence(
        mode="local_gap_coach", evidence=(), training=(),
    )
    text = render_recommender_fallback(rec)
    assert "didn't surface" in text.lower() or "don't have" in text.lower()
    assert "related career paths" in text  # B -> C locked chain close


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
    # Slice 2.5: B -> C chain close.
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
    # Slice 2.5: B -> C chain close.
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
    """Slice 2.5: Layer A is now the chain TERMINAL. Empty fallback
    closes with a natural follow-up, NOT a chain offer to another
    mode. (Body text may reference 'Canadian/NOC standard' as the
    descriptive name of what's missing -- that's not a chain offer.)"""
    rec = RecommenderEvidence(
        mode="target_noc_standard", evidence=(), training=(),
    )
    text = render_recommender_fallback(rec)
    assert "don't have" in text.lower() or "no" in text.lower()
    assert "standard skill profile" in text.lower()
    # No chain-offer phrasings (the close-offer would say
    # "Want me to ..." with one of the other modes).
    assert "Want me to show how to prepare" not in text  # not B's offer
    assert "Want me to compare your skills" not in text  # not C's offer
    # Has a natural follow-up question.
    assert "?" in text


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
    # Slice 2.5: Layer A is terminal -- no chain offer.
    assert "related career paths" not in text
    assert "dig into" in text.lower()  # natural follow-up


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
    # Slice 2.5: Layer A is terminal -- natural follow-up, no chain.
    assert "related career paths" not in text
    assert "dig into" in text.lower()


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
    """Slice 2.5: Layer C now offers Layer A in its chain close
    (C -> A in the new chain). Empty fallback acknowledges
    honestly, then offers the Canadian/NOC standard."""
    rec = RecommenderEvidence(
        mode="adjacent_noc_standard", evidence=(), training=(),
    )
    text = render_recommender_fallback(rec)
    assert "nothing surfaced" in text.lower()
    # Slice 2.5: C -> A chain close.
    assert "Canadian/NOC standard" in text
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
    # Slice 2.5: C -> A chain close.
    assert "Canadian/NOC standard" in text


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
    # Slice 2.5: C -> A chain close.
    assert "Canadian/NOC standard" in text


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


def test_adjacent_noc_standard_chains_to_layer_a():
    """Slice 2.5: Layer C now offers Layer A (Canadian/NOC standard)
    as its chain close. Layer C is no longer terminal."""
    gaps = (
        _gap(layer="adjacent_noc_standard", skill_id="F.01.b.01",
             skill_name="Reading Comprehension", importance=4.5,
             source_id="13110", source_label="Administrative assistant"),
    )
    rec = RecommenderEvidence(
        mode="adjacent_noc_standard", evidence=gaps, training=(),
    )
    text = render_recommender_fallback(rec)
    # Slice 2.5: C -> A chain close.
    assert "Canadian/NOC standard" in text
    # No B -> A leak (B's old close).
    assert "related career paths" not in text
