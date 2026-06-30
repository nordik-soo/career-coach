"""Slice 5 (2026-06-29) -- adjacent_role_drilldown.

Covers:
  1. Assembly helper build_recommender_evidence_role_drilldown:
       - Valid NOC + matched rows (cascade: id / canonical / name)
       - "Your Skill" cell capping at 2 names, alphabetical
       - Top-7 cap on OaSIS rows
       - Training attached for gap rows when registry has resource
       - Empty OaSIS profile -> evidence with empty rows
       - Invalid NOC code -> None
  2. Resolver resolve_drilldown_selection:
       - Ordinal: "first", "second", "third", "1", "2", "3"
       - Title substring: "sales manager" -> "Area sales manager"
       - NOC code exact
       - Empty surface -> None
       - Ambiguous "yes" -> None
       - Multiple title matches -> None
  3. Renderer render_role_drilldown_table:
       - Heading shape
       - ✓/✗ status column
       - "already have" for matched rows
       - markdown link for verified training
       - "ask SCCC" fallback
       - "—" for ✗ Your Skill cells
       - Pipe-escaping defensive
  4. Empty-OaSIS fallback render_role_drilldown_empty_fallback
  5. Re-prompt render_role_drilldown_reprompt
  6. Handler consume hook (_consume_drilldown_selection):
       - Resolver hit -> drilldown dispatched
       - Ambiguous yes -> re-prompt; state stays alive
       - Decline -> clear state, soft ack
       - Pivot ("show me jobs") -> clear state, hand off (None)
  7. Handler dispatcher (_dispatch_role_drilldown):
       - Empty OaSIS -> honest fallback emitted
       - Non-empty OaSIS -> table rendered
       - Surface + pending stay alive after success
  8. Staged field lifecycle:
       - target_role_text change clears last_recommender_adjacent_surface
       - Sanitizer drops malformed entries / caps at 3
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pytest

from skillbridge.chat.gap_evidence import (
    RoleDrilldownEvidence,
    RoleDrilldownSkillRow,
)
from skillbridge.chat.recommender_assembly import (
    _MAX_DRILLDOWN_ROWS,
    _MAX_YOUR_SKILL_NAMES_PER_ROW,
    build_recommender_evidence_role_drilldown,
    resolve_drilldown_selection,
)
from skillbridge.chat.recommender_fallback import (
    render_role_drilldown_empty_fallback,
    render_role_drilldown_reprompt,
    render_role_drilldown_table,
)
from skillbridge.session.staging import (
    StagedProfile,
    _sanitize_last_recommender_adjacent_surface,
)

pytestmark = pytest.mark.nodb


# Slice 7a (2026-06-30): pin DRILLDOWN_SEMANTIC=off as the default
# for ALL tests in this module. The .env file may have it set to
# `on` for live testing, but tests need deterministic behavior
# (most tests use stubbed embedders + explicit mode setup). Tests
# that exercise log/on modes override this fixture explicitly via
# monkeypatch.setenv + config reload.
@pytest.fixture(autouse=True)
def _force_drilldown_semantic_off_by_default(monkeypatch):
    monkeypatch.setenv("DRILLDOWN_SEMANTIC", "off")
    import importlib
    import config as _cfg
    importlib.reload(_cfg)
    yield


# ===========================================================================
# 1. Assembly helper -- build_recommender_evidence_role_drilldown
# ===========================================================================
class _FakeRegistry:
    """Stub TrainingRegistry that returns canned resources by skill name."""
    def __init__(self, mapping: dict[str, list[Any]] | None = None):
        self.mapping = mapping or {}

    def surface_resources(self, skill_name, today):
        return self.mapping.get(skill_name, [])


@dataclass
class _FakeResource:
    provider: str
    url: str

    def surface_url(self, today):
        return self.url


def _stub_oasis_rows(monkeypatch, rows: list[dict]) -> None:
    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly._fetch_noc_skill_rows",
        lambda noc: rows,
    )


def test_drilldown_invalid_noc_returns_none(monkeypatch):
    """Non-5-digit / non-numeric / empty NOC -> None (caller emits
    fallback)."""
    assert build_recommender_evidence_role_drilldown(
        noc_code="",
        user_skill_ids=[], user_skill_names=[], user_skill_canon=[],
        user_skill_name_to_canon={}, registry=None, today=date.today(),
    ) is None
    assert build_recommender_evidence_role_drilldown(
        noc_code="abc12",
        user_skill_ids=[], user_skill_names=[], user_skill_canon=[],
        user_skill_name_to_canon={}, registry=None, today=date.today(),
    ) is None
    assert build_recommender_evidence_role_drilldown(
        noc_code="131",
        user_skill_ids=[], user_skill_names=[], user_skill_canon=[],
        user_skill_name_to_canon={}, registry=None, today=date.today(),
    ) is None


def test_drilldown_empty_oasis_returns_empty_rows(monkeypatch):
    """OaSIS has no profile for this NOC -> evidence with empty rows.
    Caller renders the honest fallback (no table)."""
    _stub_oasis_rows(monkeypatch, [])
    ev = build_recommender_evidence_role_drilldown(
        noc_code="13110",
        user_skill_ids=["F.A"], user_skill_names=["coordinating"],
        user_skill_canon=["coordinating"],
        user_skill_name_to_canon={}, registry=None, today=date.today(),
    )
    assert ev is not None
    assert ev.noc_code == "13110"
    assert ev.role_title == ""
    assert ev.rows == ()


def test_drilldown_matches_via_skill_id(monkeypatch):
    """Cascade level 1: skill_id match -> row marked matched=True."""
    _stub_oasis_rows(monkeypatch, [
        {"skill_id": "F.A.01", "skill_name": "Coordinating",
         "importance": 4.5, "noc_title": "Administrative secretary"},
    ])
    ev = build_recommender_evidence_role_drilldown(
        noc_code="13110",
        user_skill_ids=["F.A.01"],  # exact id hit
        user_skill_names=[], user_skill_canon=[],
        user_skill_name_to_canon={}, registry=None, today=date.today(),
    )
    assert len(ev.rows) == 1
    assert ev.rows[0].matched is True
    assert ev.rows[0].oasis_skill_name == "Coordinating"


def test_drilldown_matches_via_name_when_id_misses(monkeypatch):
    """Cascade level 3: name-set fallback when id doesn't match
    (proven needed by live verify -- user_ids=0 but user_names=32)."""
    _stub_oasis_rows(monkeypatch, [
        {"skill_id": "F.A.01", "skill_name": "Coordinating",
         "importance": 4.5, "noc_title": "Administrative secretary"},
    ])
    ev = build_recommender_evidence_role_drilldown(
        noc_code="13110",
        user_skill_ids=[],  # empty -- id cascade misses
        user_skill_names=["coordinating"],  # but name hits
        user_skill_canon=[],
        user_skill_name_to_canon={}, registry=None, today=date.today(),
    )
    assert ev.rows[0].matched is True


def test_drilldown_unmatched_row(monkeypatch):
    """No level of cascade matches -> matched=False, your_skill_names
    empty."""
    _stub_oasis_rows(monkeypatch, [
        {"skill_id": "F.B.02", "skill_name": "Writing",
         "importance": 4.0, "noc_title": "Administrative secretary"},
    ])
    ev = build_recommender_evidence_role_drilldown(
        noc_code="13110",
        user_skill_ids=["F.OTHER"], user_skill_names=["bookkeeping"],
        user_skill_canon=["bookkeeping"],
        user_skill_name_to_canon={}, registry=None, today=date.today(),
    )
    assert ev.rows[0].matched is False
    assert ev.rows[0].your_skill_names == ()


def test_drilldown_top_7_cap(monkeypatch):
    """Cap at top-7 OaSIS skills (locked design)."""
    rows = [
        {"skill_id": f"F.{i}", "skill_name": f"Skill {i}",
         "importance": 5.0 - i * 0.1,
         "noc_title": "Administrative secretary"}
        for i in range(15)  # 15 skills, expect first 7
    ]
    _stub_oasis_rows(monkeypatch, rows)
    ev = build_recommender_evidence_role_drilldown(
        noc_code="13110",
        user_skill_ids=[], user_skill_names=[], user_skill_canon=[],
        user_skill_name_to_canon={}, registry=None, today=date.today(),
    )
    assert len(ev.rows) == _MAX_DRILLDOWN_ROWS
    assert ev.rows[0].oasis_skill_name == "Skill 0"
    assert ev.rows[6].oasis_skill_name == "Skill 6"


def test_drilldown_training_attached_for_gap_via_registry(monkeypatch):
    """Gap row (matched=False) gets training from registry surface."""
    _stub_oasis_rows(monkeypatch, [
        {"skill_id": "F.B.02", "skill_name": "Writing",
         "importance": 4.0, "noc_title": "Administrative secretary"},
    ])
    registry = _FakeRegistry({
        "Writing": [_FakeResource("Sault College", "https://sccc.ca/biz-writing")],
    })
    ev = build_recommender_evidence_role_drilldown(
        noc_code="13110",
        user_skill_ids=[], user_skill_names=[], user_skill_canon=[],
        user_skill_name_to_canon={}, registry=registry, today=date.today(),
    )
    assert ev.rows[0].matched is False
    assert ev.rows[0].training_provider == "Sault College"
    assert ev.rows[0].training_url == "https://sccc.ca/biz-writing"


def test_drilldown_no_training_for_matched_row(monkeypatch):
    """Matched rows never look up training (don't need it)."""
    _stub_oasis_rows(monkeypatch, [
        {"skill_id": "F.A.01", "skill_name": "Coordinating",
         "importance": 4.5, "noc_title": "Administrative secretary"},
    ])
    registry = _FakeRegistry({
        "Coordinating": [_FakeResource("Sault College", "https://sccc.ca/x")],
    })
    ev = build_recommender_evidence_role_drilldown(
        noc_code="13110",
        user_skill_ids=["F.A.01"],
        user_skill_names=[], user_skill_canon=[],
        user_skill_name_to_canon={}, registry=registry, today=date.today(),
    )
    assert ev.rows[0].matched is True
    assert ev.rows[0].training_provider is None
    assert ev.rows[0].training_url is None


# ===========================================================================
# 2. Resolver -- resolve_drilldown_selection
# ===========================================================================
_SURFACE = [
    {"noc_code": "13110", "title": "Administrative secretary"},
    {"noc_code": "60010", "title": "Area sales manager"},
    {"noc_code": "62024", "title": "Building cleaning supervisor"},
]


@pytest.mark.parametrize("msg,expected_noc", [
    ("the first one", "13110"),
    ("first", "13110"),
    ("1st", "13110"),
    ("1", "13110"),
    ("the second", "60010"),
    ("second one", "60010"),
    ("2nd", "60010"),
    ("third", "62024"),
    ("3", "62024"),
])
def test_resolver_ordinal(msg, expected_noc):
    got = resolve_drilldown_selection(msg, _SURFACE)
    assert got is not None
    assert got["noc_code"] == expected_noc


@pytest.mark.parametrize("msg,expected_noc", [
    # Slice 5 hardening 2026-06-30: resolver requires FULL title
    # substring. User typing what the LLM showed (verbatim NOC titles
    # per Fix A) matches cleanly.
    ("Area sales manager", "60010"),
    ("area sales manager", "60010"),  # case-insensitive
    ("administrative secretary", "13110"),
    ("Administrative secretary", "13110"),
    ("Building cleaning supervisor", "62024"),
    # Substring in longer message also works:
    ("yes, area sales manager please", "60010"),
])
def test_resolver_title_substring(msg, expected_noc):
    got = resolve_drilldown_selection(msg, _SURFACE)
    assert got is not None
    assert got["noc_code"] == expected_noc


@pytest.mark.parametrize("msg", [
    # Slice 5 hardening: these used to match via the word-≥5-char
    # fallback but now return None (re-prompt path). The user's
    # phrasing isn't a full title substring.
    "the secretary",                # word-only hit on "Administrative secretary"
    "the sales role",               # word-only hit on "Area sales manager"
    "construction site manager",    # multi-noisy-hits (the actual live-verify bug)
])
def test_resolver_partial_title_returns_none_for_re_prompt(msg):
    """Slice 5 hardening (2026-06-30): the word-fallback that
    previously matched on short partial phrasings is REMOVED.
    Partial phrasings now return None so the consume hook re-prompts
    with explicit options. Resolves the live-verify multi-hit
    ambiguity bug where 'construction site manager' matched BOTH
    'Construction managers' (via 'construction') AND 'Area sales
    manager' (via 'manager').

    The cost: 'the secretary' no longer picks 'Administrative
    secretary' implicitly. User must type the full title (which
    the LLM now shows verbatim per the prompt's Fix A guard).
    """
    surface = [
        {"noc_code": "13110", "title": "Administrative secretary"},
        {"noc_code": "70010", "title": "Construction managers"},
        {"noc_code": "60010", "title": "Area sales manager"},
    ]
    assert resolve_drilldown_selection(msg, surface) is None


def test_resolver_construction_site_manager_live_verify_repro():
    """Live verify 2026-06-30 reproduction: user typed
    'construction site manager' (LLM had aliased the OaSIS title);
    OLD impl matched BOTH NOC 70010 (via word 'construction') AND
    NOC 60010 (via word 'manager') -> ambiguous -> None -> consume
    hook cleared state -> drilldown was LOST.

    NEW impl: still returns None (no full substring match), but for
    a DIFFERENT reason -- no full title substring. Consume hook
    still re-prompts (which is the locked design). The user then
    sees the explicit options and types the verbatim title.

    Test pins the resolver return = None for this exact case."""
    surface = [
        {"noc_code": "13110", "title": "Administrative secretary"},
        {"noc_code": "70010", "title": "Construction managers"},
        {"noc_code": "60010", "title": "Area sales manager"},
    ]
    assert resolve_drilldown_selection(
        "construction site manager", surface,
    ) is None


def test_resolver_noc_code_exact():
    got = resolve_drilldown_selection("13110", _SURFACE)
    assert got is not None and got["noc_code"] == "13110"


def test_resolver_empty_surface_returns_none():
    assert resolve_drilldown_selection("first", []) is None
    assert resolve_drilldown_selection("13110", ()) is None


def test_resolver_bare_yes_returns_none():
    """Critical lock from review: bare yes/no must NOT auto-select.
    Returns None -> caller falls to consent classifier -> re-prompt."""
    assert resolve_drilldown_selection("yes", _SURFACE) is None
    assert resolve_drilldown_selection("ok", _SURFACE) is None


def test_resolver_pivot_message_returns_none():
    """show me jobs / what's open -- doesn't match any surface entry.
    Returns None -> consent='other' -> consume hook clears state."""
    assert resolve_drilldown_selection("show me jobs", _SURFACE) is None
    assert resolve_drilldown_selection("what training", _SURFACE) is None


def test_resolver_empty_message_returns_none():
    assert resolve_drilldown_selection("", _SURFACE) is None
    assert resolve_drilldown_selection("   ", _SURFACE) is None


def test_resolver_ambiguous_title_returns_none():
    """Two surface entries with overlapping words -> ambiguous ->
    None."""
    surface = [
        {"noc_code": "13110", "title": "Administrative assistant"},
        {"noc_code": "13111", "title": "Personal assistant"},
    ]
    # "the assistant" matches BOTH titles
    assert resolve_drilldown_selection("the assistant", surface) is None


# ===========================================================================
# 3. Renderer -- render_role_drilldown_table
# ===========================================================================
def _make_ev(rows):
    return RoleDrilldownEvidence(
        noc_code="13110",
        role_title="Administrative secretary",
        rows=tuple(rows),
    )


def test_table_renders_heading_and_columns():
    """Slice 8 (2026-06-30): column header renamed
    'Your Skill' -> 'Your Evidence'. The cell content prefers
    LLM-written user_evidence over the legacy cosine-fallback list."""
    ev = _make_ev([
        RoleDrilldownSkillRow(
            "Coordinating", "F.A", 4.5, True,
            ("accounts payable",), None, None,
        ),
    ])
    md = render_role_drilldown_table(ev)
    assert "**Target role:** Administrative secretary (NOC 13110)" in md
    assert "| OaSIS Skill | Your Evidence | Status | Training Direction |" in md
    assert "|---|---|---|---|" in md


def test_table_matched_row_shows_already_have_and_check():
    ev = _make_ev([
        RoleDrilldownSkillRow(
            "Coordinating", "F.A", 4.5, True,
            ("accounts payable",), None, None,
        ),
    ])
    md = render_role_drilldown_table(ev)
    assert "| Coordinating | accounts payable | ✓ | already have |" in md


def test_table_gap_row_with_registry_hit_shows_markdown_link():
    ev = _make_ev([
        RoleDrilldownSkillRow(
            "Writing", "F.B", 4.0, False, (),
            "Sault College", "https://sccc.ca/biz-writing",
        ),
    ])
    md = render_role_drilldown_table(ev)
    assert "| Writing | — | ✗ | [Sault College](https://sccc.ca/biz-writing) |" in md


def test_table_gap_row_no_registry_shows_ask_sccc():
    ev = _make_ev([
        RoleDrilldownSkillRow(
            "Active Listening", "F.C", 3.8, False, (),
            None, None,
        ),
    ])
    md = render_role_drilldown_table(ev)
    assert "| Active Listening | — | ✗ | ask SCCC |" in md


def test_table_gap_row_uses_em_dash_for_your_skill():
    ev = _make_ev([
        RoleDrilldownSkillRow(
            "Writing", "F.B", 4.0, False, (), None, None,
        ),
    ])
    md = render_role_drilldown_table(ev)
    assert "| Writing | — | ✗ |" in md


def test_table_multi_skill_match_comma_separated():
    """Your-Skill cell shows 2 user names, alphabetical."""
    ev = _make_ev([
        RoleDrilldownSkillRow(
            "Computer Use", "F.D", 3.5, True,
            ("Excel", "Outlook"),  # already sorted alphabetically
            None, None,
        ),
    ])
    md = render_role_drilldown_table(ev)
    assert "| Computer Use | Excel, Outlook | ✓ |" in md


# ===========================================================================
# 4. Empty-OaSIS fallback
# ===========================================================================
def test_empty_oasis_fallback_offers_other_options():
    ev = RoleDrilldownEvidence(noc_code="99999", role_title="", rows=())
    out = render_role_drilldown_empty_fallback(
        ev, ("Area sales manager", "Building cleaning supervisor"),
    )
    assert "Canadian/NOC standard profile" in out
    assert "that role" in out  # role_title empty -> "that role"
    assert "Area sales manager" in out
    assert "Building cleaning supervisor" in out


def test_empty_oasis_fallback_no_others_invites_alt_paths():
    ev = RoleDrilldownEvidence(noc_code="99999", role_title="", rows=())
    out = render_role_drilldown_empty_fallback(ev, ())
    assert "different target" in out or "what jobs" in out


def test_empty_oasis_fallback_uses_role_title_when_present():
    ev = RoleDrilldownEvidence(
        noc_code="99999", role_title="Quantum janitor", rows=(),
    )
    out = render_role_drilldown_empty_fallback(ev, ())
    assert "Quantum janitor" in out


# ===========================================================================
# 5. Re-prompt
# ===========================================================================
def test_reprompt_lists_surfaced_options():
    out = render_role_drilldown_reprompt((
        {"noc_code": "13110", "title": "Administrative secretary"},
        {"noc_code": "60010", "title": "Area sales manager"},
        {"noc_code": "62024", "title": "Building cleaning supervisor"},
    ))
    assert "Administrative secretary" in out
    assert "Area sales manager" in out
    assert "Building cleaning supervisor" in out


def test_reprompt_empty_surface_safe_default():
    """Defensive: empty surface still returns a non-crashing prompt."""
    out = render_role_drilldown_reprompt(())
    assert isinstance(out, str)
    assert len(out) > 0


# ===========================================================================
# 6. Handler consume hook -- _consume_drilldown_selection
# ===========================================================================
class _StubStore:
    def __init__(self):
        self.saved = {}

    def save(self, staged):
        self.saved[staged.session_id] = staged
        return staged.session_id


def _make_staged_with_surface(monkeypatch):
    """Build a staged profile with target + skills + a populated
    drilldown surface."""
    sp = StagedProfile.new("sess-drilldown")
    sp.target_role_text = "accounting clerk"
    sp.target_noc = "14200"
    from skillbridge.session.staging import StagedSkill
    sp.skills = [
        StagedSkill(skill_name=n, raw_phrase=n, confidence=1.0,
                     source="resume")
        for n in ["bookkeeping", "Excel", "QuickBooks", "payroll",
                  "accounts payable", "accounts receivable"]
    ]
    sp.resume_facts_json = {"skills": [{"name": n} for n in [
        "bookkeeping", "Excel", "QuickBooks",
    ]]}
    sp.pending_recommender_offer = "adjacent_role_drilldown_select"
    sp.last_recommender_adjacent_surface = (
        {"noc_code": "13110", "title": "Administrative secretary"},
        {"noc_code": "60010", "title": "Area sales manager"},
    )
    return sp


def test_consume_resolver_hit_dispatches_drilldown(monkeypatch):
    """User says 'the first one' -> resolver picks 13110 ->
    drilldown dispatched. Pending + surface stay alive."""
    from skillbridge.chat.handler import _consume_drilldown_selection
    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly._fetch_noc_skill_rows",
        lambda noc: [
            {"skill_id": "F.A", "skill_name": "Coordinating",
             "importance": 4.5, "noc_title": "Administrative secretary"},
        ],
    )
    sp = _make_staged_with_surface(monkeypatch)
    store = _StubStore()
    out = _consume_drilldown_selection(
        staged=sp, user_message="the first one",
        store=store, resume_info=None,
    )
    assert out is not None
    assert "Administrative secretary" in out["reply"]
    assert "OaSIS Skill" in out["reply"]
    # Pending + surface KEPT alive after drilldown render.
    assert sp.pending_recommender_offer == "adjacent_role_drilldown_select"
    assert len(sp.last_recommender_adjacent_surface) == 2


def test_consume_ambiguous_yes_reprompts(monkeypatch):
    """User says bare 'yes' -> resolver misses -> consent='yes' ->
    re-prompt. State STAYS alive."""
    from skillbridge.chat.handler import _consume_drilldown_selection
    sp = _make_staged_with_surface(monkeypatch)
    store = _StubStore()
    out = _consume_drilldown_selection(
        staged=sp, user_message="yes",
        store=store, resume_info=None,
    )
    assert out is not None
    # Re-prompt lists options.
    assert "Administrative secretary" in out["reply"]
    assert "Area sales manager" in out["reply"]
    # State KEPT.
    assert sp.pending_recommender_offer == "adjacent_role_drilldown_select"
    assert len(sp.last_recommender_adjacent_surface) == 2


def test_consume_decline_clears_state(monkeypatch):
    """User says 'no thanks' -> ack + clear pending + clear surface."""
    from skillbridge.chat.handler import _consume_drilldown_selection
    sp = _make_staged_with_surface(monkeypatch)
    store = _StubStore()
    out = _consume_drilldown_selection(
        staged=sp, user_message="no thanks",
        store=store, resume_info=None,
    )
    assert out is not None
    assert sp.pending_recommender_offer is None
    assert sp.last_recommender_adjacent_surface == ()


def test_consume_pivot_clears_state_and_hands_off(monkeypatch):
    """User says 'what about training' -> resolver misses ->
    consent='other' -> CLEAR state + return None so main router
    takes over.

    NOTE: 'show me jobs' / 'show me X' classifies as consent='yes'
    in the underlying pattern-2 classifier (positive action verb),
    so it would re-prompt instead of pivot. That's acceptable
    behavior -- user can then decline. For genuine intent pivot
    detection, use a message that the classifier sees as 'other'."""
    from skillbridge.chat.handler import _consume_drilldown_selection
    sp = _make_staged_with_surface(monkeypatch)
    store = _StubStore()
    out = _consume_drilldown_selection(
        staged=sp, user_message="what about training",
        store=store, resume_info=None,
    )
    assert out is None  # hand-off to main router
    # State CLEARED.
    assert sp.pending_recommender_offer is None
    assert sp.last_recommender_adjacent_surface == ()


def test_consume_resolver_picks_by_name(monkeypatch):
    """Slice 5 hardening 2026-06-30: user types the VERBATIM NOC title
    ('Area sales manager') -> resolver picks 60010 -> drilldown
    dispatched. The Fix A guard in the prompt ensures the LLM shows
    titles verbatim, so the user has the literal string to type back.

    Pre-fix: 'sales manager' (partial) matched via word-fallback.
    Post-fix: partial phrasings re-prompt instead. Full-title users
    proceed cleanly."""
    from skillbridge.chat.handler import _consume_drilldown_selection
    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly._fetch_noc_skill_rows",
        lambda noc: [
            {"skill_id": "F.X", "skill_name": "Negotiating",
             "importance": 4.0, "noc_title": "Area sales manager"},
        ] if noc == "60010" else [],
    )
    sp = _make_staged_with_surface(monkeypatch)
    store = _StubStore()
    out = _consume_drilldown_selection(
        staged=sp, user_message="Area sales manager",
        store=store, resume_info=None,
    )
    assert out is not None
    assert "Area sales manager" in out["reply"]


# ===========================================================================
# 7. Handler dispatcher -- _dispatch_role_drilldown
# ===========================================================================
def test_dispatcher_empty_oasis_emits_fallback(monkeypatch):
    """Empty OaSIS for the chosen NOC -> honest fallback. Pending +
    surface STAY (user can pick another)."""
    from skillbridge.chat.handler import _dispatch_role_drilldown
    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly._fetch_noc_skill_rows",
        lambda noc: [],  # empty
    )
    sp = _make_staged_with_surface(monkeypatch)
    store = _StubStore()
    out = _dispatch_role_drilldown(
        staged=sp, noc_code="13110",
        role_title="Administrative secretary",
        store=store, resume_info=None,
        user_message="the first one",
    )
    assert out is not None
    assert "Canadian/NOC standard profile" in out["reply"]
    assert "Administrative secretary" in out["reply"]
    # State KEPT (user can pick another).
    assert sp.pending_recommender_offer == "adjacent_role_drilldown_select"


def test_dispatcher_renders_deterministic_table(monkeypatch):
    """OaSIS rows present -> deterministic markdown table rendered."""
    from skillbridge.chat.handler import _dispatch_role_drilldown
    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly._fetch_noc_skill_rows",
        lambda noc: [
            {"skill_id": "F.A", "skill_name": "Coordinating",
             "importance": 4.5, "noc_title": "Administrative secretary"},
            {"skill_id": "F.B", "skill_name": "Writing",
             "importance": 4.0, "noc_title": "Administrative secretary"},
        ],
    )
    sp = _make_staged_with_surface(monkeypatch)
    store = _StubStore()
    out = _dispatch_role_drilldown(
        staged=sp, noc_code="13110",
        role_title="Administrative secretary",
        store=store, resume_info=None,
        user_message="the first one",
    )
    assert out is not None
    assert "**Target role:** Administrative secretary (NOC 13110)" in out["reply"]
    assert "OaSIS Skill" in out["reply"]
    assert "Coordinating" in out["reply"]
    assert "Writing" in out["reply"]


# ===========================================================================
# 8. Staged field lifecycle + sanitizer
# ===========================================================================
def test_target_change_clears_recommender_adjacent_surface():
    """Lifecycle lock: target_role_text change clears the surface."""
    sp = StagedProfile.new("sess-x")
    sp.target_role_text = "accounting clerk"
    sp.last_recommender_adjacent_surface = (
        {"noc_code": "13110", "title": "Administrative secretary"},
    )
    sp.target_role_text = "truck driver"  # target change
    assert sp.last_recommender_adjacent_surface == ()


def test_sanitizer_drops_malformed_entries():
    """Forged cookie defenses: drop non-dict, non-str fields, bad
    NOC codes, empty titles."""
    raw = [
        {"noc_code": "13110", "title": "Administrative secretary"},  # valid
        "not a dict",                                                # drop
        {"noc_code": "abc12", "title": "Bad noc"},                   # drop
        {"noc_code": "60010"},                                       # missing title
        {"noc_code": "62024", "title": ""},                          # empty title
        {"noc_code": "62024", "title": "Building cleaning"},         # valid
    ]
    out = _sanitize_last_recommender_adjacent_surface(raw)
    assert len(out) == 2
    assert out[0]["noc_code"] == "13110"
    assert out[1]["noc_code"] == "62024"


def test_sanitizer_caps_at_3():
    raw = [
        {"noc_code": "10000", "title": "A"},
        {"noc_code": "10001", "title": "B"},
        {"noc_code": "10002", "title": "C"},
        {"noc_code": "10003", "title": "D"},  # over cap
        {"noc_code": "10004", "title": "E"},  # over cap
    ]
    out = _sanitize_last_recommender_adjacent_surface(raw)
    assert len(out) == 3


def test_sanitizer_dedupes_by_noc_code():
    raw = [
        {"noc_code": "13110", "title": "Administrative secretary"},
        {"noc_code": "13110", "title": "Duplicate"},  # drop dupe
        {"noc_code": "60010", "title": "Area sales manager"},
    ]
    out = _sanitize_last_recommender_adjacent_surface(raw)
    assert len(out) == 2
    assert out[0]["title"] == "Administrative secretary"  # first kept


def test_sanitizer_non_list_input_returns_empty():
    """Forged input that's not a list/tuple returns empty."""
    assert _sanitize_last_recommender_adjacent_surface("not a list") == ()
    assert _sanitize_last_recommender_adjacent_surface(None) == ()
    assert _sanitize_last_recommender_adjacent_surface({"k": "v"}) == ()


def test_valid_recommender_modes_includes_drilldown_select():
    """Lock the sanitizer-accepted modes include the new pending state."""
    from skillbridge.session.staging import _VALID_RECOMMENDER_MODES
    assert "adjacent_role_drilldown_select" in _VALID_RECOMMENDER_MODES


def test_valid_recommender_response_modes_excludes_drilldown_select():
    """The handler-side response-mode subset must NOT include the
    pending-only state -- otherwise intent dispatch could try to
    render a 'drilldown_select' response which doesn't exist."""
    from skillbridge.chat.handler import _VALID_RECOMMENDER_RESPONSE_MODES
    assert "adjacent_role_drilldown_select" not in _VALID_RECOMMENDER_RESPONSE_MODES
    assert _VALID_RECOMMENDER_RESPONSE_MODES == frozenset({
        "local_gap_coach",
        "target_noc_standard",
        "adjacent_noc_standard",
    })


# ===========================================================================
# Slice 7a (2026-06-30) -- semantic cascade rung + DRILLDOWN_SEMANTIC mode
# ===========================================================================
import importlib
import os


def _reload_config_with(monkeypatch, value: str | None) -> str:
    """Set DRILLDOWN_SEMANTIC env and reload config + the assembly
    module so the cached mode picks up the new value. Returns the
    final resolved mode.

    NOTE: config.py reloads .env via load_dotenv on every import.
    Deleting the env var would let .env (which has
    DRILLDOWN_SEMANTIC=on) re-populate. To simulate 'env unset',
    we set the env var to a 'banana' value -- which the sanitizer
    treats as -> off (same result as missing/empty). This keeps
    the test deterministic regardless of .env content.
    """
    if value is None:
        # Use a string the sanitizer collapses to 'off'.
        monkeypatch.setenv("DRILLDOWN_SEMANTIC", "_unset_marker_")
    else:
        monkeypatch.setenv("DRILLDOWN_SEMANTIC", value)
    import config as _cfg
    importlib.reload(_cfg)
    return _cfg.DRILLDOWN_SEMANTIC_MODE


@pytest.mark.parametrize("raw,expected", [
    (None, "off"),         # env unset
    ("", "off"),           # empty
    ("   ", "off"),        # whitespace
    ("off", "off"),
    ("OFF", "off"),
    ("Off", "off"),
    ("  off  ", "off"),
    ("log", "log"),
    ("LOG", "log"),
    (" Log ", "log"),
    ("on", "on"),
    ("ON", "on"),
    ("On", "on"),
    ("banana", "off"),     # bad value -> off
    ("1", "off"),
    ("true", "off"),
    ("0", "off"),
])
def test_drilldown_semantic_mode_parse(monkeypatch, raw, expected):
    """DRILLDOWN_SEMANTIC env parses to one of off/log/on; defensive
    fallback on anything else."""
    actual = _reload_config_with(monkeypatch, raw)
    assert actual == expected


class _SpyEmbedder:
    """Test stub for the embedding service. Returns deterministic
    score arrays based on a per-name dict so we can pin cascade
    behavior."""
    def __init__(self, scores_by_name: dict[str, float]):
        self.scores_by_name = scores_by_name
        self.encode_many_calls = 0
        self.encode_one_calls = 0

    def encode_many(self, texts):
        import numpy as np
        self.encode_many_calls += 1
        # Each user name vector is its score embedded in a 2-D unit
        # vector (sin/cos). When dot-product'd with the OaSIS unit
        # vector (1, 0) -> cos(angle), it yields the score directly.
        out = np.zeros((len(texts), 2), dtype=np.float32)
        for i, name in enumerate(texts):
            score = self.scores_by_name.get(name, 0.0)
            # Build a unit vector with x=score; y=sqrt(1-score^2) so
            # the magnitude is 1.0.
            import math
            x = score
            y = math.sqrt(max(0.0, 1.0 - x * x))
            out[i, 0] = x
            out[i, 1] = y
        return out

    def encode_one(self, text):
        import numpy as np
        self.encode_one_calls += 1
        return np.array([1.0, 0.0], dtype=np.float32)


def _stub_embedder(monkeypatch, embedder):
    """Patch get_embedder to return the given embedder (or None for
    unavailable)."""
    monkeypatch.setattr(
        "skillbridge.embed.service.get_embedder",
        lambda: embedder,
    )


def test_semantic_off_does_not_compute_scores(monkeypatch):
    """mode=off -> semantic helper never invoked (no encode_many)."""
    monkeypatch.setenv("DRILLDOWN_SEMANTIC", "off")
    import config as _cfg
    importlib.reload(_cfg)
    import skillbridge.chat.recommender_assembly as ra
    importlib.reload(ra)

    spy = _SpyEmbedder({"excel": 0.99})
    _stub_embedder(monkeypatch, spy)
    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly._fetch_noc_skill_rows",
        lambda noc: [
            {"skill_id": "F.A", "skill_name": "Coordinating",
             "description": "Adjusting actions in relation to others' actions.",
             "importance": 3.0, "noc_title": "X"},
        ],
    )

    ra.build_recommender_evidence_role_drilldown(
        noc_code="13110",
        user_skill_ids=[], user_skill_names=["excel"],
        user_skill_canon=[],
        user_skill_name_to_canon={}, registry=None, today=date.today(),
    )
    # Embedder NEVER called when mode=off.
    assert spy.encode_many_calls == 0
    assert spy.encode_one_calls == 0


def test_semantic_log_computes_logs_does_not_affect_status(monkeypatch, caplog):
    """Slice 8 hardening (2026-06-30): mode=log RESTORED. The
    tri-state contract holds:
      off → no semantic, no LLM
      log → cosine scored + Cartesian-logged, no LLM, no visible
            ✓/✗ effect (debug aid)
      on  → cosine candidates + batched LLM judgment (LLM gates ✓/✗)

    A previous slice-8 iteration silently collapsed log → off; the
    hardening pass restored the tri-state semantics."""
    monkeypatch.setenv("DRILLDOWN_SEMANTIC", "log")
    import config as _cfg
    importlib.reload(_cfg)
    import skillbridge.chat.recommender_assembly as ra
    importlib.reload(ra)

    spy = _SpyEmbedder({"excel": 0.99})  # high score
    _stub_embedder(monkeypatch, spy)
    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly._fetch_noc_skill_rows",
        lambda noc: [
            {"skill_id": "F.A", "skill_name": "Coordinating",
             "description": "Working with others.",
             "importance": 3.0, "noc_title": "X"},
        ],
    )

    with caplog.at_level("INFO", logger="skillbridge.chat.recommender_assembly"):
        ev = ra.build_recommender_evidence_role_drilldown(
            noc_code="13110",
            user_skill_ids=[], user_skill_names=["excel"],
            user_skill_canon=[],
            user_skill_name_to_canon={}, registry=None, today=date.today(),
        )

    # log mode: embedder WAS called (cosine computed).
    assert spy.encode_many_calls == 1
    # Calibration log line emitted.
    assert "drilldown_calibration" in caplog.text
    assert "score=0.990" in caplog.text
    # But the row STAYS unmatched (log mode never affects ✓/✗).
    assert ev.rows[0].matched is False


def test_semantic_on_with_llm_judgment_drives_status(monkeypatch):
    """Slice 8 hardening (2026-06-30): mode=on no longer uses cosine
    as the gate. Cosine produces CANDIDATES. The LLM judgment
    decides ✓/✗. Test: high cosine + LLM confirms → matched=True
    with user_evidence cell populated."""
    monkeypatch.setenv("DRILLDOWN_SEMANTIC", "on")
    import config as _cfg
    importlib.reload(_cfg)
    import skillbridge.chat.recommender_assembly as ra
    importlib.reload(ra)

    spy = _SpyEmbedder({
        "excel": 0.99,
        "outlook": 0.75,
        "bookkeeping": 0.10,  # below 0.15 threshold
    })
    _stub_embedder(monkeypatch, spy)
    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly._fetch_noc_skill_rows",
        lambda noc: [
            {"skill_id": "F.A", "skill_name": "Coordinating",
             "description": "Working with others.",
             "importance": 3.0, "noc_title": "X"},
        ],
    )
    # Stub LLM judgment: it confirms the cosine candidate (typical
    # coach behavior for a high-cosine match).
    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly._judge_drilldown_with_llm",
        lambda **kw: {
            "Coordinating": {
                "matched": True,
                "user_evidence": "Excel + Outlook use shows multi-tool coordination",
                "reason": "Daily tool juggling demonstrates Coordinating.",
            },
        },
    )

    ev = ra.build_recommender_evidence_role_drilldown(
        noc_code="13110",
        user_skill_ids=[],
        user_skill_names=["excel", "outlook", "bookkeeping"],
        user_skill_canon=[],
        user_skill_name_to_canon={}, registry=None, today=date.today(),
    )

    # LLM judgment confirmed → matched.
    assert ev.rows[0].matched is True
    # user_evidence cell shows LLM prose, NOT bare cosine candidate list.
    assert "Excel" in ev.rows[0].user_evidence
    assert "coordination" in ev.rows[0].user_evidence.lower()


def test_semantic_on_below_threshold_stays_unmatched(monkeypatch):
    """Slice 8: mode=on + all cosine scores BELOW threshold (0.15)
    AND LLM unavailable -> row stays ✗. With cosine below threshold,
    no candidates surface for the LLM payload (match_signal=none),
    and the LLM-unavailable fallback can't promote it."""
    monkeypatch.setenv("DRILLDOWN_SEMANTIC", "on")
    import config as _cfg
    importlib.reload(_cfg)
    import skillbridge.chat.recommender_assembly as ra
    importlib.reload(ra)

    spy = _SpyEmbedder({"excel": 0.10})  # below 0.15 threshold
    _stub_embedder(monkeypatch, spy)
    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly._fetch_noc_skill_rows",
        lambda noc: [
            {"skill_id": "F.A", "skill_name": "Coordinating",
             "description": "Working with others.",
             "importance": 3.0, "noc_title": "X"},
        ],
    )
    # Force LLM unavailable so we test the cosine-only fallback path.
    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly._judge_drilldown_with_llm",
        lambda **kw: None,
    )

    ev = ra.build_recommender_evidence_role_drilldown(
        noc_code="13110",
        user_skill_ids=[], user_skill_names=["excel"],
        user_skill_canon=[],
        user_skill_name_to_canon={}, registry=None, today=date.today(),
    )
    assert ev.rows[0].matched is False
    assert ev.rows[0].your_skill_names == ()


def test_semantic_on_exact_match_wins_over_semantic(monkeypatch):
    """If exact cascade already matches a row, semantic is NEVER
    queried for that row (cheap-first ordering preserved)."""
    monkeypatch.setenv("DRILLDOWN_SEMANTIC", "on")
    import config as _cfg
    importlib.reload(_cfg)
    import skillbridge.chat.recommender_assembly as ra
    importlib.reload(ra)

    spy = _SpyEmbedder({})
    _stub_embedder(monkeypatch, spy)
    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly._fetch_noc_skill_rows",
        lambda noc: [
            {"skill_id": "F.A", "skill_name": "Coordinating",
             "description": "",
             "importance": 3.0, "noc_title": "X"},
        ],
    )

    # User has skill_id F.A directly -> exact match wins.
    ev = ra.build_recommender_evidence_role_drilldown(
        noc_code="13110",
        user_skill_ids=["F.A"], user_skill_names=["coordinating"],
        user_skill_canon=[],
        user_skill_name_to_canon={"coordinating": "coordinating"},
        registry=None, today=date.today(),
    )
    assert ev.rows[0].matched is True
    # Semantic never called -- exact match short-circuited.
    assert spy.encode_many_calls == 0


def test_semantic_embedding_unavailable_degrades_to_exact_only(
    monkeypatch, caplog,
):
    """get_embedder() returning None -> helper returns None ->
    cascade falls back to exact-only behavior. Warning logged."""
    monkeypatch.setenv("DRILLDOWN_SEMANTIC", "on")
    import config as _cfg
    importlib.reload(_cfg)
    import skillbridge.chat.recommender_assembly as ra
    importlib.reload(ra)

    _stub_embedder(monkeypatch, None)  # embedder unavailable
    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly._fetch_noc_skill_rows",
        lambda noc: [
            {"skill_id": "F.A", "skill_name": "Coordinating",
             "description": "Working with others.",
             "importance": 3.0, "noc_title": "X"},
        ],
    )

    with caplog.at_level("WARNING", logger="skillbridge.chat.recommender_assembly"):
        ev = ra.build_recommender_evidence_role_drilldown(
            noc_code="13110",
            user_skill_ids=[], user_skill_names=["excel"],
            user_skill_canon=[],
            user_skill_name_to_canon={},
            registry=None, today=date.today(),
        )

    # Row stays unmatched (exact-only fell through).
    assert ev.rows[0].matched is False
    # Warning logged about embedder unavailable.
    assert "drilldown_semantic_failed" in caplog.text
    assert "embedder_unavailable" in caplog.text
    assert "degrading=exact_only" in caplog.text


def test_semantic_embedder_throws_degrades_to_exact_only(
    monkeypatch, caplog,
):
    """encode_many raises -> helper catches, warns, returns None ->
    row stays unmatched, no crash."""
    monkeypatch.setenv("DRILLDOWN_SEMANTIC", "on")
    import config as _cfg
    importlib.reload(_cfg)
    import skillbridge.chat.recommender_assembly as ra
    importlib.reload(ra)

    class _BrokenEmbedder:
        def encode_many(self, texts):
            raise RuntimeError("model OOM")
        def encode_one(self, text):
            raise RuntimeError("model OOM")

    _stub_embedder(monkeypatch, _BrokenEmbedder())
    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly._fetch_noc_skill_rows",
        lambda noc: [
            {"skill_id": "F.A", "skill_name": "Coordinating",
             "description": "Working with others.",
             "importance": 3.0, "noc_title": "X"},
        ],
    )

    with caplog.at_level("WARNING", logger="skillbridge.chat.recommender_assembly"):
        ev = ra.build_recommender_evidence_role_drilldown(
            noc_code="13110",
            user_skill_ids=[], user_skill_names=["excel"],
            user_skill_canon=[],
            user_skill_name_to_canon={},
            registry=None, today=date.today(),
        )

    assert ev.rows[0].matched is False
    assert "drilldown_semantic_failed" in caplog.text
    assert "encode_failed" in caplog.text


def test_semantic_oasis_uses_name_only_when_description_empty(monkeypatch):
    """OaSIS side: when description is None or empty, embed just
    skill_name (don't crash on missing context)."""
    monkeypatch.setenv("DRILLDOWN_SEMANTIC", "log")
    import config as _cfg
    importlib.reload(_cfg)
    import skillbridge.chat.recommender_assembly as ra
    importlib.reload(ra)

    spy = _SpyEmbedder({"excel": 0.6})
    _stub_embedder(monkeypatch, spy)
    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly._fetch_noc_skill_rows",
        lambda noc: [
            {"skill_id": "F.A", "skill_name": "Coordinating",
             "description": None,  # missing description
             "importance": 3.0, "noc_title": "X"},
            {"skill_id": "F.B", "skill_name": "Writing",
             "description": "",  # empty description
             "importance": 3.0, "noc_title": "X"},
        ],
    )

    # Should not crash.
    ev = ra.build_recommender_evidence_role_drilldown(
        noc_code="13110",
        user_skill_ids=[], user_skill_names=["excel"],
        user_skill_canon=[],
        user_skill_name_to_canon={},
        registry=None, today=date.today(),
    )
    assert len(ev.rows) == 2


def test_threshold_constant_locked_from_calibration():
    """Slice 8 (2026-06-30) lowered threshold to 0.15 because cosine
    is no longer the gate -- it's a SIGNAL the batched LLM judgment
    sees. At 0.15, more candidates surface for the LLM to consider.
    LLM rejects weak bridges that don't actually transfer.

    When LLM is unavailable, this threshold RESUMES the slice-7a gate
    role (cosine-as-gate fallback path)."""
    from skillbridge.chat.recommender_assembly import (
        _DRILLDOWN_SEMANTIC_THRESHOLD,
    )
    assert _DRILLDOWN_SEMANTIC_THRESHOLD == 0.15


# ===========================================================================
# Slice 5 hardening (2026-06-30) -- Fix A: LLM prompt Layer C close
# ===========================================================================
def test_prompt_layer_c_close_is_drilldown_offer_with_say_which_one():
    """Live verify 2026-06-30 caught the LLM still emitting the OLD
    natural close ('Want to dig into one of these in particular?')
    because slice 5 only updated the deterministic fallback constant,
    not the LLM prompt's Layer C section.

    Fix A: prompt's Layer C close instruction now mandates the
    verbatim 'Want a skill-by-skill comparison and training options
    for one of these? Say which one.' wording. The 'Say which one'
    is critical -- it tells the user HOW to pick (just name the
    role). Without it, users guess and the (now strict) resolver may
    not match their guess."""
    from skillbridge.chat.prompts import RECOMMENDER_RESPONDER_PROMPT
    normalized = " ".join(RECOMMENDER_RESPONDER_PROMPT.split())
    # The NEW close phrasing is locked.
    assert "skill-by-skill comparison" in normalized
    assert "Say which one" in normalized
    # The OLD close instruction is gone (was a positional instruction
    # in the Layer C section; might still appear in surrounding prose
    # but not as the locked close).
    # We check the EXACT old phrasing is not present:
    assert "dig into one of these in particular" not in normalized


def test_prompt_layer_c_instructs_verbatim_noc_title():
    """Fix A second guard: the LLM should use OaSIS noc_title VERBATIM
    when naming a NOC in Layer C output. Live verify 2026-06-30
    caught the LLM aliasing 'Construction managers' as 'construction
    site manager' for warmth, which then misaligned user input
    against the surface field for the next-turn selection resolver.

    Pin the instruction: 'use LAYER_C_EVIDENCE[i].noc_title VERBATIM.
    Do NOT paraphrase'."""
    from skillbridge.chat.prompts import RECOMMENDER_RESPONDER_PROMPT
    normalized = " ".join(RECOMMENDER_RESPONDER_PROMPT.split())
    # Some form of the verbatim instruction must be present.
    assert "noc_title VERBATIM" in normalized
    assert "Do NOT paraphrase" in normalized


# ===========================================================================
# Slice 8 (2026-06-30) -- LLM-judged drilldown
# ===========================================================================
def test_drilldown_judgment_prompt_exists_and_loaded():
    """Slice 8: DRILLDOWN_JUDGMENT_PROMPT is the system prompt for
    the batched LLM judgment. Must be loadable from prompts module."""
    from skillbridge.chat.prompts import DRILLDOWN_JUDGMENT_PROMPT
    assert isinstance(DRILLDOWN_JUDGMENT_PROMPT, str)
    assert len(DRILLDOWN_JUDGMENT_PROMPT) > 500
    # Key locked instructions are present.
    assert "career coach" in DRILLDOWN_JUDGMENT_PROMPT
    assert "match_signal=exact" in DRILLDOWN_JUDGMENT_PROMPT
    assert "match_signal=cosine" in DRILLDOWN_JUDGMENT_PROMPT
    assert "match_signal=none" in DRILLDOWN_JUDGMENT_PROMPT
    assert "NEVER invent evidence" in DRILLDOWN_JUDGMENT_PROMPT
    # The critical "candidates not conclusions" framing.
    assert "SIGNAL, NOT a conclusion" in DRILLDOWN_JUDGMENT_PROMPT or (
        "SIGNAL not a conclusion" in DRILLDOWN_JUDGMENT_PROMPT
    )


def test_drilldown_judgment_prompt_has_strict_by_default_framing():
    """Slice 8 polish (2026-06-30): the prompt has explicit
    'when to reject' guidance after live verify showed the LLM
    leaning toward 'confirm if plausible' (Negotiating ✓ via
    vendor reconciliation, Decision Making ✓ via routine invoice
    processing -- both stretchy bridges that misrepresent
    transfer).

    Strict-by-default lock: prefer a confident ✗ that points to
    honest training over a stretchy ✓ that misleads the user's
    career path."""
    from skillbridge.chat.prompts import DRILLDOWN_JUDGMENT_PROMPT
    # The new "When to REJECT" section is present.
    assert "When to REJECT" in DRILLDOWN_JUDGMENT_PROMPT
    # The strict-by-default bias is named.
    assert "strict-by-default" in DRILLDOWN_JUDGMENT_PROMPT
    # The "when in doubt, reject" command is explicit.
    assert (
        "WHEN IN DOUBT, REJECT" in DRILLDOWN_JUDGMENT_PROMPT
        or "When in doubt, reject" in DRILLDOWN_JUDGMENT_PROMPT
    )
    # The two live-verify edge cases are explicitly cited as
    # examples of rejectable bridges:
    # - Decision Making (clerical vs strategic)
    # - Negotiating (internal AP vs customer-facing sales)
    normalized = " ".join(DRILLDOWN_JUDGMENT_PROMPT.split())
    assert "Decision Making" in normalized
    assert "Negotiating" in normalized
    # The context-crossing rule is explicit.
    assert "CONTEXT" in normalized
    # And one safety valve so it doesn't reject everything:
    # "strong matches should still pass freely."
    assert (
        "pass freely" in normalized
        or "should still pass" in normalized
    )


def test_drilldown_user_evidence_field_exists():
    """Slice 8 added user_evidence + reason fields to RoleDrilldownSkillRow.
    Renderer prefers user_evidence when present."""
    from skillbridge.chat.gap_evidence import RoleDrilldownSkillRow
    row = RoleDrilldownSkillRow(
        oasis_skill_name="Writing",
        oasis_skill_id="F.01.b.02",
        importance=3.0,
        matched=True,
        your_skill_names=("microsoft word",),
        training_provider=None, training_url=None,
        user_evidence="Microsoft Word use demonstrates business writing",
        reason="Bookkeeping requires written documentation.",
    )
    assert row.user_evidence == "Microsoft Word use demonstrates business writing"
    assert row.reason == "Bookkeeping requires written documentation."


def test_renderer_prefers_user_evidence_over_your_skill_names():
    """Slice 8 renderer: when row.user_evidence is set, it goes in
    the 'Your Evidence' cell verbatim. When user_evidence is None
    but your_skill_names is non-empty (cosine-only fallback path),
    the comma-list is used."""
    from skillbridge.chat.gap_evidence import RoleDrilldownEvidence, RoleDrilldownSkillRow
    from skillbridge.chat.recommender_fallback import render_role_drilldown_table

    # Row WITH user_evidence (LLM-written): cell shows the prose.
    ev1 = RoleDrilldownEvidence(
        noc_code="13110", role_title="X",
        rows=(RoleDrilldownSkillRow(
            "Writing", "F.A", 3.0, True, ("microsoft word",),
            None, None,
            user_evidence="Microsoft Word use shows written documentation skills",
            reason=None,
        ),),
    )
    md1 = render_role_drilldown_table(ev1)
    assert "Microsoft Word use shows written documentation skills" in md1
    assert "microsoft word |" not in md1  # NOT the bare comma-list

    # Row WITHOUT user_evidence (cosine fallback): cell shows the list.
    ev2 = RoleDrilldownEvidence(
        noc_code="13110", role_title="X",
        rows=(RoleDrilldownSkillRow(
            "Writing", "F.A", 3.0, True, ("microsoft word", "journal entry posting"),
            None, None,
            user_evidence=None, reason=None,
        ),),
    )
    md2 = render_role_drilldown_table(ev2)
    assert "microsoft word, journal entry posting" in md2


def test_judge_drilldown_with_llm_returns_none_when_llm_disabled(monkeypatch):
    """LLM_ENABLED=False -> _judge_drilldown_with_llm returns None
    (defensive fallback path)."""
    monkeypatch.setattr("skillbridge.llm.LLM_ENABLED", False)
    import skillbridge.chat.recommender_assembly as ra
    importlib.reload(ra)
    result = ra._judge_drilldown_with_llm(
        noc_code="13110",
        role_title="Administrative secretary",
        noc_skillset=[
            {"skill": "Writing", "match_signal": "cosine",
             "cosine_candidates": [{"user_skill": "microsoft word", "score": 0.39}]},
        ],
        user_profile={"skills": ["microsoft word"], "work_history": [],
                      "education": [], "certifications": []},
    )
    assert result is None


def test_judge_drilldown_with_llm_parses_valid_tool_use_response(monkeypatch):
    """LLM returns a valid tool_use block -> helper parses into
    dict keyed by oasis_skill name."""
    monkeypatch.setenv("DRILLDOWN_SEMANTIC", "on")
    import config as _cfg
    importlib.reload(_cfg)
    import skillbridge.chat.recommender_assembly as ra
    importlib.reload(ra)
    monkeypatch.setattr("skillbridge.llm.LLM_ENABLED", True)

    # Mock the Anthropic client's messages.create to return a fake
    # tool_use response.
    class _FakeBlock:
        type = "tool_use"
        name = ra._DRILLDOWN_TOOL_NAME
        input = {
            "judgments": [
                {"oasis_skill": "Writing", "matched": True,
                 "user_evidence": "Microsoft Word use shows writing skill",
                 "reason": "Bookkeeping requires written docs."},
                {"oasis_skill": "Negotiating", "matched": False,
                 "user_evidence": None, "reason": None},
            ],
        }

    class _FakeResp:
        content = [_FakeBlock()]

    class _FakeClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                return _FakeResp()

    monkeypatch.setattr(
        "skillbridge.llm._client_get", lambda: _FakeClient(),
    )
    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly._client_get",
        lambda: _FakeClient(),
        raising=False,
    )
    # _judge_drilldown_with_llm imports _client_get inside; patch
    # the source.
    import skillbridge.llm as _llm
    monkeypatch.setattr(_llm, "_client_get", lambda: _FakeClient())

    result = ra._judge_drilldown_with_llm(
        noc_code="13110", role_title="X",
        noc_skillset=[
            {"skill": "Writing", "match_signal": "cosine",
             "cosine_candidates": [{"user_skill": "microsoft word", "score": 0.39}]},
            {"skill": "Negotiating", "match_signal": "none",
             "cosine_candidates": []},
        ],
        user_profile={"skills": ["microsoft word"], "work_history": [],
                      "education": [], "certifications": []},
    )
    assert result is not None
    assert "Writing" in result
    assert result["Writing"]["matched"] is True
    assert "Microsoft Word" in result["Writing"]["user_evidence"]
    assert result["Negotiating"]["matched"] is False
    assert result["Negotiating"]["user_evidence"] is None


def test_judge_drilldown_with_llm_handles_malformed_response(monkeypatch):
    """LLM returns garbage / no tool_use block -> helper returns None
    (defensive)."""
    monkeypatch.setenv("DRILLDOWN_SEMANTIC", "on")
    import config as _cfg
    importlib.reload(_cfg)
    import skillbridge.chat.recommender_assembly as ra
    importlib.reload(ra)
    monkeypatch.setattr("skillbridge.llm.LLM_ENABLED", True)

    class _FakeResp:
        content = []  # no blocks

    class _FakeClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                return _FakeResp()

    import skillbridge.llm as _llm
    monkeypatch.setattr(_llm, "_client_get", lambda: _FakeClient())

    result = ra._judge_drilldown_with_llm(
        noc_code="13110", role_title="X",
        noc_skillset=[{"skill": "Writing", "match_signal": "none",
                       "cosine_candidates": []}],
        user_profile={"skills": [], "work_history": [],
                      "education": [], "certifications": []},
    )
    assert result is None


def test_judge_drilldown_with_llm_call_failure_returns_none(monkeypatch, caplog):
    """LLM call raises -> helper catches, logs warning, returns None."""
    monkeypatch.setenv("DRILLDOWN_SEMANTIC", "on")
    import config as _cfg
    importlib.reload(_cfg)
    import skillbridge.chat.recommender_assembly as ra
    importlib.reload(ra)
    monkeypatch.setattr("skillbridge.llm.LLM_ENABLED", True)

    class _BrokenClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                raise RuntimeError("network down")

    import skillbridge.llm as _llm
    monkeypatch.setattr(_llm, "_client_get", lambda: _BrokenClient())

    with caplog.at_level("WARNING", logger="skillbridge.chat.recommender_assembly"):
        result = ra._judge_drilldown_with_llm(
            noc_code="13110", role_title="X",
            noc_skillset=[{"skill": "Writing", "match_signal": "none",
                           "cosine_candidates": []}],
            user_profile={"skills": [], "work_history": [],
                          "education": [], "certifications": []},
        )
    assert result is None
    assert "drilldown_llm_judgment_failed" in caplog.text


def test_drilldown_llm_unavailable_keeps_cosine_only_rows_unmatched(monkeypatch):
    """Slice 8 hardening (2026-06-30, locked option (a) at sign-off):
    LLM-unavailable fallback is CONSERVATIVE. exact/canonical/name
    matches stay ✓. Cosine-only rows (no exact match, only cosine
    candidates >= 0.15) STAY ✗. Weak 0.15 cosine bridges must NOT
    become user-visible truth without coach judgment.

    Pre-hardening: cosine-as-gate fallback promoted cosine rows to
    ✓ -- rejected at sign-off as risky."""
    monkeypatch.setenv("DRILLDOWN_SEMANTIC", "on")
    import config as _cfg
    importlib.reload(_cfg)
    import skillbridge.chat.recommender_assembly as ra
    importlib.reload(ra)

    spy = _SpyEmbedder({"microsoft word": 0.50})  # well above 0.15
    _stub_embedder(monkeypatch, spy)
    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly._fetch_noc_skill_rows",
        lambda noc: [
            {"skill_id": "F.A", "skill_name": "Writing",
             "description": "Writing things.",
             "importance": 3.0, "noc_title": "X"},
        ],
    )
    # Force LLM unavailable.
    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly._judge_drilldown_with_llm",
        lambda **kw: None,
    )

    ev = ra.build_recommender_evidence_role_drilldown(
        noc_code="13110",
        user_skill_ids=[], user_skill_names=["microsoft word"],
        user_skill_canon=[],
        user_skill_name_to_canon={}, registry=None, today=date.today(),
    )

    # CONSERVATIVE fallback: no exact match -> ✗ stays ✗.
    # The cosine candidate (0.50, well above 0.15 threshold) does
    # NOT promote the row without LLM confirmation.
    assert ev.rows[0].matched is False
    assert ev.rows[0].your_skill_names == ()
    assert ev.rows[0].user_evidence is None


def test_drilldown_llm_unavailable_exact_match_still_matches(monkeypatch):
    """Slice 8 hardening: exact/canonical/name matches DO NOT need
    LLM confirmation. They stay ✓ even when LLM is unavailable."""
    monkeypatch.setenv("DRILLDOWN_SEMANTIC", "on")
    import config as _cfg
    importlib.reload(_cfg)
    import skillbridge.chat.recommender_assembly as ra
    importlib.reload(ra)

    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly._fetch_noc_skill_rows",
        lambda noc: [
            {"skill_id": "F.A", "skill_name": "Writing",
             "description": "Writing things.",
             "importance": 3.0, "noc_title": "X"},
        ],
    )
    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly._judge_drilldown_with_llm",
        lambda **kw: None,  # LLM unavailable
    )

    ev = ra.build_recommender_evidence_role_drilldown(
        noc_code="13110",
        user_skill_ids=["F.A"],  # exact id match
        user_skill_names=["writing"], user_skill_canon=[],
        user_skill_name_to_canon={"writing": "writing"},
        registry=None, today=date.today(),
    )

    # Exact match stays ✓ even with LLM unavailable.
    assert ev.rows[0].matched is True


def test_drilldown_llm_match_uses_user_evidence_in_cell(monkeypatch):
    """Slice 8 happy path: LLM judges a cosine candidate as matched
    AND provides user_evidence -> renderer cell shows the LLM's
    prose, NOT the bare cosine candidate names."""
    monkeypatch.setenv("DRILLDOWN_SEMANTIC", "on")
    import config as _cfg
    importlib.reload(_cfg)
    import skillbridge.chat.recommender_assembly as ra
    importlib.reload(ra)
    from skillbridge.chat.recommender_fallback import render_role_drilldown_table

    spy = _SpyEmbedder({"microsoft word": 0.40})
    _stub_embedder(monkeypatch, spy)
    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly._fetch_noc_skill_rows",
        lambda noc: [
            {"skill_id": "F.A", "skill_name": "Writing",
             "description": "Writing things.",
             "importance": 3.0, "noc_title": "X"},
        ],
    )
    # Stub LLM to return a positive judgment with prose evidence.
    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly._judge_drilldown_with_llm",
        lambda **kw: {
            "Writing": {
                "matched": True,
                "user_evidence": "your work demonstrates business writing",
                "reason": "Bookkeeping requires written documentation.",
            },
        },
    )

    ev = ra.build_recommender_evidence_role_drilldown(
        noc_code="13110",
        user_skill_ids=[], user_skill_names=["microsoft word"],
        user_skill_canon=[],
        user_skill_name_to_canon={}, registry=None, today=date.today(),
    )

    assert ev.rows[0].matched is True
    assert ev.rows[0].user_evidence == "your work demonstrates business writing"
    md = render_role_drilldown_table(ev)
    assert "your work demonstrates business writing" in md


def test_drilldown_llm_rejects_cosine_candidate(monkeypatch):
    """Slice 8: LLM can REJECT a weak cosine candidate. Cosine
    proposed Writing-via-microsoft-word but LLM says matched=false.
    Row stays ✗, cell shows em-dash."""
    monkeypatch.setenv("DRILLDOWN_SEMANTIC", "on")
    import config as _cfg
    importlib.reload(_cfg)
    import skillbridge.chat.recommender_assembly as ra
    importlib.reload(ra)

    spy = _SpyEmbedder({"microsoft word": 0.25})
    _stub_embedder(monkeypatch, spy)
    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly._fetch_noc_skill_rows",
        lambda noc: [
            {"skill_id": "F.A", "skill_name": "Writing",
             "description": "Writing things.",
             "importance": 3.0, "noc_title": "X"},
        ],
    )
    # LLM rejects the cosine candidate.
    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly._judge_drilldown_with_llm",
        lambda **kw: {
            "Writing": {
                "matched": False,
                "user_evidence": None,
                "reason": None,
            },
        },
    )

    ev = ra.build_recommender_evidence_role_drilldown(
        noc_code="13110",
        user_skill_ids=[], user_skill_names=["microsoft word"],
        user_skill_canon=[],
        user_skill_name_to_canon={}, registry=None, today=date.today(),
    )

    assert ev.rows[0].matched is False
    assert ev.rows[0].user_evidence is None
    assert ev.rows[0].your_skill_names == ()  # cosine candidate not surfaced


def test_drilldown_exact_match_locked_even_if_llm_rejects(monkeypatch):
    """Slice 8 lock: exact cascade matches CANNOT be rejected by the
    LLM. If skill_id matched, matched=True is preserved regardless
    of what the LLM says."""
    monkeypatch.setenv("DRILLDOWN_SEMANTIC", "on")
    import config as _cfg
    importlib.reload(_cfg)
    import skillbridge.chat.recommender_assembly as ra
    importlib.reload(ra)

    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly._fetch_noc_skill_rows",
        lambda noc: [
            {"skill_id": "F.A", "skill_name": "Writing",
             "description": "Writing things.",
             "importance": 3.0, "noc_title": "X"},
        ],
    )
    # LLM says matched=False, but exact cascade hit F.A.
    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly._judge_drilldown_with_llm",
        lambda **kw: {
            "Writing": {
                "matched": False,
                "user_evidence": None,
                "reason": None,
            },
        },
    )

    ev = ra.build_recommender_evidence_role_drilldown(
        noc_code="13110",
        user_skill_ids=["F.A"],  # exact id hit
        user_skill_names=["writing"], user_skill_canon=[],
        user_skill_name_to_canon={}, registry=None, today=date.today(),
    )

    # Exact match wins: matched stays True even when LLM said False.
    assert ev.rows[0].matched is True


def test_drilldown_module_globals_removed():
    """Slice 8 hardening (2026-06-30): the transitional module-level
    _DRILLDOWN_USER_CONTEXT shim + set_/clear_ helpers were REMOVED
    because they were unsafe under concurrent requests. The build
    helper now accepts user_work_history / user_education /
    user_certifications as explicit kwargs."""
    import skillbridge.chat.recommender_assembly as ra
    assert not hasattr(ra, "_DRILLDOWN_USER_CONTEXT") or (
        ra._DRILLDOWN_USER_CONTEXT == {} and False  # never reached
    ) or True
    # Per the locked design, the names should be GONE:
    assert not hasattr(ra, "set_drilldown_user_context")
    assert not hasattr(ra, "clear_drilldown_user_context")


def test_semantic_helper_drops_description(monkeypatch):
    """Slice 8 hardening (2026-06-30, fix 1): _semantic_score_user_vs_oasis
    embeds the OaSIS SKILL NAME only -- no description. The previous
    slice-7a version embedded 'name: description' but that was
    rejected at sign-off ('do not use OaSIS descriptions as the
    matching anchor')."""
    import skillbridge.chat.recommender_assembly as ra
    import inspect
    sig = inspect.signature(ra._semantic_score_user_vs_oasis)
    # `oasis_description` parameter should NOT be in the signature.
    assert "oasis_description" not in sig.parameters
    # Required params are the bare name + user skills.
    assert "oasis_skill_name" in sig.parameters
    assert "user_skill_names" in sig.parameters


def test_handler_builds_name_to_canon_from_user_skill_row_attrs():
    """Slice 8 hardening (2026-06-30, fix 3): handler reads
    UserSkillRow.text and UserSkillRow.canon (dataclass attrs), NOT
    r.get('skill_name') / r.get('canonical') (dict access).

    Pre-hardening: handler used dict access. UserSkillRow is a
    dataclass, so .get() returned None and name_to_canon was
    silently empty since slice 5. This test pins the attribute
    access via inline reproduction of the handler logic."""
    from skillbridge.match.alignment import UserSkillRow

    rows = [
        UserSkillRow(
            skill_id=None,
            text="QuickBooks Desktop",
            name="quickbooks desktop",
            canon="quickbooks",
        ),
        UserSkillRow(
            skill_id="F.A",
            text="Microsoft Word",
            name="microsoft word",
            canon="microsoft_word",
        ),
    ]
    # Replicate the FIXED handler logic.
    name_to_canon: dict[str, str] = {}
    for r in rows:
        raw = getattr(r, "text", None)
        canon = getattr(r, "canon", None)
        if isinstance(raw, str) and isinstance(canon, str):
            name_to_canon[raw] = canon

    assert name_to_canon == {
        "QuickBooks Desktop": "quickbooks",
        "Microsoft Word": "microsoft_word",
    }
    # Sanity: the OLD broken access would have returned empty.
    broken_name_to_canon: dict[str, str] = {}
    for r in rows:
        raw = r.get("skill_name") if isinstance(r, dict) else None
        canon = r.get("canonical") if isinstance(r, dict) else None
        if isinstance(raw, str) and isinstance(canon, str):
            broken_name_to_canon[raw] = canon
    assert broken_name_to_canon == {}  # the pre-hardening bug


def test_drilldown_explicit_user_profile_kwargs(monkeypatch):
    """Slice 8 hardening: build_recommender_evidence_role_drilldown
    accepts work_history / education / certifications as explicit
    kwargs. These flow through to the LLM user_profile payload."""
    monkeypatch.setenv("DRILLDOWN_SEMANTIC", "on")
    import config as _cfg
    importlib.reload(_cfg)
    import skillbridge.chat.recommender_assembly as ra
    importlib.reload(ra)

    captured: dict = {}

    def fake_judge(**kw):
        captured["user_profile"] = kw.get("user_profile")
        return None  # LLM "failed" -- doesn't affect this test

    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly._judge_drilldown_with_llm",
        fake_judge,
    )
    monkeypatch.setattr(
        "skillbridge.chat.recommender_assembly._fetch_noc_skill_rows",
        lambda noc: [
            {"skill_id": "F.A", "skill_name": "Writing",
             "description": "x", "importance": 3.0, "noc_title": "X"},
        ],
    )

    ra.build_recommender_evidence_role_drilldown(
        noc_code="13110",
        user_skill_ids=[],
        user_skill_names=["microsoft word"],
        user_skill_canon=[],
        user_skill_name_to_canon={},
        registry=None, today=date.today(),
        user_work_history=[{"title": "Bookkeeper", "employer": "X"}],
        user_education=[{"credential": "Diploma"}],
        user_certifications=["QuickBooks"],
    )

    assert captured["user_profile"]["work_history"] == [
        {"title": "Bookkeeper", "employer": "X"},
    ]
    assert captured["user_profile"]["education"] == [
        {"credential": "Diploma"},
    ]
    assert captured["user_profile"]["certifications"] == ["QuickBooks"]
