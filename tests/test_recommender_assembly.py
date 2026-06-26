"""Slice 5 step 4 (2026-06-19) -- recommender_assembly structural pins.

Tests cover the three build helpers' contracts without hitting the DB
or the live training registry. Layer A/C detector calls are patched
via the module-level `_fetch_noc_skill_rows`; Layer B detector is
exercised through `compute_local_posting_gaps` against duck-typed
MatchResult shims (same pattern as test_gap_evidence.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest

from skillbridge.chat.gap_evidence import GapEvidence
from skillbridge.chat.recommender_assembly import (
    build_recommender_evidence_adjacent_noc_standard,
    build_recommender_evidence_local_gap_coach,
    build_recommender_evidence_target_noc_standard,
)
from skillbridge.training.models import Gap, Resource
from skillbridge.training.registry import TrainingRegistry

pytestmark = pytest.mark.nodb


# ---------------------------------------------------------------------------
# Shared shims
# ---------------------------------------------------------------------------
@dataclass
class _MR:
    job_id: str
    title: str
    employer: str
    missing_skills: list[str]
    missing_skill_ids: list[str | None]


def _make_registry_with_one_gap(
    *,
    canonical: str,
    aliases: tuple[str, ...] = (),
    resources: tuple[Resource, ...] = (),
) -> TrainingRegistry:
    """Build a minimal TrainingRegistry in-memory via from_dict."""
    today = date(2026, 6, 21)
    res_dicts: list[dict] = []
    for r in resources:
        d: dict = {
            "provider": r.provider,
            "type": r.type,
            "url": r.url,
            "summary": r.summary,
        }
        if r.verified_at is not None:
            d["verified_at"] = r.verified_at.isoformat()
        if r.verified_by is not None:
            d["verified_by"] = r.verified_by
        res_dicts.append(d)

    # Registry requires a non-empty aliases list; default to a
    # lower-case copy of the canonical when caller didn't supply.
    alias_list = list(aliases) if aliases else [canonical.lower()]
    gap_dict = {
        "canonical_name": canonical,
        "category": "skill",
        "description": "test gap",
        "aliases": alias_list,
        "resources": res_dicts,
    }
    raw = {
        "version": 1,
        "registry_verified_at": today.isoformat(),
        "gaps": [gap_dict],
    }
    return TrainingRegistry.from_dict(raw)


# ===========================================================================
# Layer B helper
# ===========================================================================
def test_layer_b_empty_when_primary_gap_name_is_none():
    """When CP4 returns no primary recommendation, the helper returns
    empty evidence -- no Layer B record passes the canonical filter."""
    mr = _MR(
        job_id="job-1", title="Bookkeeper", employer="Acme",
        missing_skills=["QuickBooks Desktop"],
        missing_skill_ids=[None],
    )
    rec = build_recommender_evidence_local_gap_coach(
        match_results=[mr],
        primary_gap_name=None,
        registry=None,
        today=date(2026, 6, 21),
    )
    assert rec.mode == "local_gap_coach"
    assert rec.evidence == ()
    assert rec.training == ()


def test_layer_b_empty_when_primary_gap_name_is_blank():
    mr = _MR(
        job_id="job-1", title="Bookkeeper", employer="Acme",
        missing_skills=["QuickBooks Desktop"],
        missing_skill_ids=[None],
    )
    rec = build_recommender_evidence_local_gap_coach(
        match_results=[mr],
        primary_gap_name="   ",
        registry=None,
        today=date(2026, 6, 21),
    )
    assert rec.evidence == ()


def test_layer_b_filters_to_primary_gap_name_via_canonicalizer():
    """The filter uses canonicalize_skill on BOTH the primary gap name
    and the Layer B record's skill_name -- not a raw lowercase
    compare. Different spellings of the same canonical skill match."""
    mr = _MR(
        job_id="job-1", title="Bookkeeper", employer="Acme",
        missing_skills=["QuickBooks Desktop", "Excel"],
        missing_skill_ids=[None, "S_EXCEL"],
    )
    rec = build_recommender_evidence_local_gap_coach(
        match_results=[mr],
        primary_gap_name="quickbooks desktop",  # different casing
        registry=None,
        today=date(2026, 6, 21),
    )
    assert len(rec.evidence) == 1
    assert rec.evidence[0].skill_name == "QuickBooks Desktop"


def test_layer_b_top_1_only():
    """Even if multiple postings have the same gap, top-1 by design."""
    mr1 = _MR(
        job_id="job-1", title="Bookkeeper", employer="Acme",
        missing_skills=["QuickBooks Desktop"], missing_skill_ids=[None],
    )
    mr2 = _MR(
        job_id="job-2", title="Bookkeeper", employer="Beta",
        missing_skills=["QuickBooks Desktop"], missing_skill_ids=[None],
    )
    rec = build_recommender_evidence_local_gap_coach(
        match_results=[mr1, mr2],
        primary_gap_name="QuickBooks Desktop",
        registry=None,
        today=date(2026, 6, 21),
    )
    # Top-1 even though two postings carry it.
    assert len(rec.evidence) == 1
    # First-seen wins.
    assert rec.evidence[0].source_id == "job-1"


def test_layer_b_no_match_returns_empty():
    """When the primary gap name doesn't match any Layer B record,
    return empty -- don't fall back to a different gap."""
    mr = _MR(
        job_id="job-1", title="Bookkeeper", employer="Acme",
        missing_skills=["Excel"], missing_skill_ids=["S_EXCEL"],
    )
    rec = build_recommender_evidence_local_gap_coach(
        match_results=[mr],
        primary_gap_name="QuickBooks Desktop",
        registry=None,
        today=date(2026, 6, 21),
    )
    assert rec.evidence == ()


def test_layer_b_with_verified_training_resource():
    """Registry lookup returns a fresh URL; emit TrainingResource."""
    fresh = Resource(
        provider="Sault College",
        type="local_training",
        url="https://saultcollege.ca/quickbooks",
        summary="QuickBooks Desktop course",
        verified_at=date(2026, 5, 1),  # ~50 days old; well within 180
        verified_by="ops",
    )
    registry = _make_registry_with_one_gap(
        canonical="QuickBooks Desktop",
        resources=(fresh,),
    )
    mr = _MR(
        job_id="job-1", title="Bookkeeper", employer="Acme",
        missing_skills=["QuickBooks Desktop"], missing_skill_ids=[None],
    )
    rec = build_recommender_evidence_local_gap_coach(
        match_results=[mr],
        primary_gap_name="QuickBooks Desktop",
        registry=registry,
        today=date(2026, 6, 21),
    )
    assert len(rec.training) == 1
    t = rec.training[0]
    assert t.provider == "Sault College"
    assert t.url == "https://saultcollege.ca/quickbooks"
    assert t.summary == "QuickBooks Desktop course"


def test_layer_b_excludes_stale_resources():
    """Stale resources fail surface_url(today); excluded from training."""
    stale = Resource(
        provider="Sault College",
        type="local_training",
        url="https://saultcollege.ca/quickbooks",
        summary="QuickBooks Desktop course",
        verified_at=date(2025, 1, 1),  # > 180 days old
        verified_by="ops",
    )
    registry = _make_registry_with_one_gap(
        canonical="QuickBooks Desktop",
        resources=(stale,),
    )
    mr = _MR(
        job_id="job-1", title="Bookkeeper", employer="Acme",
        missing_skills=["QuickBooks Desktop"], missing_skill_ids=[None],
    )
    rec = build_recommender_evidence_local_gap_coach(
        match_results=[mr],
        primary_gap_name="QuickBooks Desktop",
        registry=registry,
        today=date(2026, 6, 21),
    )
    assert rec.training == ()


def test_layer_b_excludes_referral_only_resources():
    """referral_only resources have null URL by construction;
    excluded from training."""
    referral = Resource(
        provider="SCCC",
        type="referral_only",
        url=None,
        summary="ask the SCCC",
        verified_at=date(2026, 5, 1),
        verified_by="ops",
    )
    registry = _make_registry_with_one_gap(
        canonical="QuickBooks Desktop",
        resources=(referral,),
    )
    mr = _MR(
        job_id="job-1", title="Bookkeeper", employer="Acme",
        missing_skills=["QuickBooks Desktop"], missing_skill_ids=[None],
    )
    rec = build_recommender_evidence_local_gap_coach(
        match_results=[mr],
        primary_gap_name="QuickBooks Desktop",
        registry=registry,
        today=date(2026, 6, 21),
    )
    assert rec.training == ()


def test_layer_b_no_registry_still_emits_evidence():
    """When registry is None (load failed), evidence still emits but
    training is empty -- the fallback will narrate honestly."""
    mr = _MR(
        job_id="job-1", title="Bookkeeper", employer="Acme",
        missing_skills=["QuickBooks Desktop"], missing_skill_ids=[None],
    )
    rec = build_recommender_evidence_local_gap_coach(
        match_results=[mr],
        primary_gap_name="QuickBooks Desktop",
        registry=None,
        today=date(2026, 6, 21),
    )
    assert len(rec.evidence) == 1
    assert rec.training == ()


# ===========================================================================
# Layer A helper
# ===========================================================================
def test_layer_a_empty_when_no_target_noc(monkeypatch):
    rec = build_recommender_evidence_target_noc_standard(
        user_skill_ids=set(),
        target_noc=None,
    )
    assert rec.mode == "target_noc_standard"
    assert rec.evidence == ()
    assert rec.training == ()


def test_layer_a_caps_at_top_3_by_importance(monkeypatch):
    """Detector returns rows ordered by importance DESC; helper caps at 3."""
    rows = [
        {"skill_id": "F.01", "skill_name": "Reading Comprehension",
         "importance": 4.5, "noc_title": "Accounting clerk"},
        {"skill_id": "F.02", "skill_name": "Critical Thinking",
         "importance": 4.0, "noc_title": "Accounting clerk"},
        {"skill_id": "F.03", "skill_name": "Numeracy",
         "importance": 3.8, "noc_title": "Accounting clerk"},
        {"skill_id": "F.04", "skill_name": "Coordinating",
         "importance": 3.5, "noc_title": "Accounting clerk"},
        {"skill_id": "F.05", "skill_name": "Active Listening",
         "importance": 3.2, "noc_title": "Accounting clerk"},
    ]
    monkeypatch.setattr(
        "skillbridge.chat.gap_evidence._fetch_noc_skill_rows",
        lambda noc: rows,
    )
    rec = build_recommender_evidence_target_noc_standard(
        user_skill_ids=set(),
        target_noc="14200",
    )
    assert len(rec.evidence) == 3
    names = [g.skill_name for g in rec.evidence]
    assert names == [
        "Reading Comprehension", "Critical Thinking", "Numeracy"
    ]
    assert rec.training == ()


def test_layer_a_empty_when_no_reference_profile(monkeypatch):
    monkeypatch.setattr(
        "skillbridge.chat.gap_evidence._fetch_noc_skill_rows",
        lambda noc: [],
    )
    rec = build_recommender_evidence_target_noc_standard(
        user_skill_ids=set(),
        target_noc="99999",
    )
    assert rec.evidence == ()
    assert rec.training == ()


def test_layer_a_skips_skills_user_already_has(monkeypatch):
    rows = [
        {"skill_id": "F.01", "skill_name": "Reading Comprehension",
         "importance": 4.5, "noc_title": "Accounting clerk"},
        {"skill_id": "F.02", "skill_name": "Critical Thinking",
         "importance": 4.0, "noc_title": "Accounting clerk"},
    ]
    monkeypatch.setattr(
        "skillbridge.chat.gap_evidence._fetch_noc_skill_rows",
        lambda noc: rows,
    )
    rec = build_recommender_evidence_target_noc_standard(
        user_skill_ids={"F.01"},  # already has Reading Comprehension
        target_noc="14200",
    )
    assert len(rec.evidence) == 1
    assert rec.evidence[0].skill_name == "Critical Thinking"


# ===========================================================================
# Layer C helper
# ===========================================================================
def test_layer_c_empty_when_no_persisted_nocs():
    rec = build_recommender_evidence_adjacent_noc_standard(
        user_skill_ids=set(),
        last_adjacent_nocs=(),
    )
    assert rec.mode == "adjacent_noc_standard"
    assert rec.evidence == ()
    assert rec.training == ()


def test_layer_c_skips_blank_and_non_string_nocs():
    rec = build_recommender_evidence_adjacent_noc_standard(
        user_skill_ids=set(),
        last_adjacent_nocs=("", "   ", None),  # type: ignore[arg-type]
    )
    assert rec.evidence == ()


def test_layer_c_fans_out_per_noc_and_caps_top_3_per_noc(monkeypatch):
    """Layer C runs the OaSIS query for each NOC and caps each at top-3.
    Order is preserved (first NOC's records before second NOC's)."""
    def fake_fetch(noc):
        if noc == "13110":
            return [
                {"skill_id": "F.A", "skill_name": "Writing",
                 "importance": 4.5, "noc_title": "Admin assistant"},
                {"skill_id": "F.B", "skill_name": "Coordinating",
                 "importance": 4.0, "noc_title": "Admin assistant"},
                {"skill_id": "F.C", "skill_name": "Active Listening",
                 "importance": 3.8, "noc_title": "Admin assistant"},
                {"skill_id": "F.D", "skill_name": "Reading",
                 "importance": 3.5, "noc_title": "Admin assistant"},
            ]
        if noc == "13100":
            return [
                {"skill_id": "F.X", "skill_name": "Numeracy",
                 "importance": 4.2, "noc_title": "Business officer"},
            ]
        return []
    monkeypatch.setattr(
        "skillbridge.chat.gap_evidence._fetch_noc_skill_rows",
        fake_fetch,
    )
    rec = build_recommender_evidence_adjacent_noc_standard(
        user_skill_ids=set(),
        last_adjacent_nocs=("13110", "13100"),
    )
    # First NOC contributes 3 (capped), second contributes 1.
    assert len(rec.evidence) == 4
    # First-seen NOC's records come first.
    assert rec.evidence[0].source_id == "13110"
    assert rec.evidence[1].source_id == "13110"
    assert rec.evidence[2].source_id == "13110"
    assert rec.evidence[3].source_id == "13100"
    # Top-3 by importance per NOC.
    names_13110 = [g.skill_name for g in rec.evidence[:3]]
    assert names_13110 == ["Writing", "Coordinating", "Active Listening"]


def test_layer_c_includes_noc_label(monkeypatch):
    def fake_fetch(noc):
        return [
            {"skill_id": "F.A", "skill_name": "Writing",
             "importance": 4.5, "noc_title": "Admin assistant"},
        ]
    monkeypatch.setattr(
        "skillbridge.chat.gap_evidence._fetch_noc_skill_rows",
        fake_fetch,
    )
    rec = build_recommender_evidence_adjacent_noc_standard(
        user_skill_ids=set(),
        last_adjacent_nocs=("13110",),
    )
    assert len(rec.evidence) == 1
    assert rec.evidence[0].source_label == "Admin assistant"


# ===========================================================================
# Slice 2 (re-introduced 2026-06-23): Layer B target-NOC family filter
# ===========================================================================
from skillbridge.chat.recommender_assembly import filter_matches_to_target_family


@dataclass
class _MockMatch:
    """Minimal MatchResult-like object with just noc_code."""
    noc_code: str | None
    job_id: str = "job-x"


def test_filter_returns_all_when_target_noc_none():
    """No target -> no anchor -> return all matches unchanged."""
    matches = [
        _MockMatch(noc_code="14200"),
        _MockMatch(noc_code="13110"),
        _MockMatch(noc_code="14404"),
    ]
    out = filter_matches_to_target_family(matches, None)
    assert len(out) == 3


@pytest.mark.parametrize("bad_target", [
    "", "  ", "13", "13110abc", "ABC12", "131100",
])
def test_filter_returns_all_when_target_noc_invalid(bad_target):
    """Malformed target NOC -> no filter (preserves current behavior)."""
    matches = [
        _MockMatch(noc_code="14200"),
        _MockMatch(noc_code="13110"),
    ]
    out = filter_matches_to_target_family(matches, bad_target)
    assert len(out) == 2


def test_filter_returns_exact_noc_match_only():
    """When postings include the exact target NOC, return ONLY those.
    Don't dilute with minor-group / off-target postings."""
    matches = [
        _MockMatch(noc_code="14200"),
        _MockMatch(noc_code="13110"),
        _MockMatch(noc_code="14200"),  # second exact match
        _MockMatch(noc_code="14404"),
    ]
    out = filter_matches_to_target_family(matches, "14200")
    assert len(out) == 2
    assert all(m.noc_code == "14200" for m in out)


def test_filter_falls_back_to_minor_group_when_no_exact():
    """When no exact NOC match exists, return same-minor-group
    postings (first 4 digits match)."""
    matches = [
        _MockMatch(noc_code="14201"),  # same minor group as 14200
        _MockMatch(noc_code="14202"),  # same minor group
        _MockMatch(noc_code="13110"),  # different minor group
        _MockMatch(noc_code="14404"),  # different minor group
    ]
    out = filter_matches_to_target_family(matches, "14200")
    assert len(out) == 2
    assert all(m.noc_code in ("14201", "14202") for m in out)


def test_filter_returns_empty_when_no_exact_and_no_minor_group():
    """If neither exact nor minor-group match exists, return empty."""
    matches = [
        _MockMatch(noc_code="13110"),
        _MockMatch(noc_code="14404"),
        _MockMatch(noc_code="62024"),
    ]
    out = filter_matches_to_target_family(matches, "14200")
    assert out == []


def test_filter_skips_matches_with_no_noc_code():
    matches = [
        _MockMatch(noc_code=None),
        _MockMatch(noc_code="14200"),
        _MockMatch(noc_code=None),
    ]
    out = filter_matches_to_target_family(matches, "14200")
    assert len(out) == 1
    assert out[0].noc_code == "14200"


def test_filter_skips_matches_with_invalid_noc_code_length():
    matches = [
        _MockMatch(noc_code="1420"),  # too short
        _MockMatch(noc_code="142001"),  # too long
        _MockMatch(noc_code="14201"),  # valid same-minor
    ]
    out = filter_matches_to_target_family(matches, "14200")
    assert len(out) == 1
    assert out[0].noc_code == "14201"


def test_filter_empty_match_list():
    out = filter_matches_to_target_family([], "14200")
    assert out == []


def test_filter_accepts_any_iterable():
    matches_tuple = (
        _MockMatch(noc_code="14200"),
        _MockMatch(noc_code="13110"),
    )
    out = filter_matches_to_target_family(matches_tuple, "14200")
    assert len(out) == 1


def test_filter_preserves_match_order():
    matches = [
        _MockMatch(noc_code="14200", job_id="a"),
        _MockMatch(noc_code="14201", job_id="b"),
        _MockMatch(noc_code="14200", job_id="c"),
    ]
    out = filter_matches_to_target_family(matches, "14200")
    assert [m.job_id for m in out] == ["a", "c"]
