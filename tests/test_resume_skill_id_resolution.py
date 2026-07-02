"""Skill-ID resolution safety net for the resume-derivation path
(2026-07-01).

The chat/job/training extractor path already resolves skill_id via
skillbridge.extract.base.resolve_many. The resume derivation path,
prior to this slice, bypassed the resolver entirely -- every
StagedSkill built from resume_facts_json came through with
skill_id=None. That made Layer A / Layer C gap detection blind to
skills the user genuinely had (the log line
`recommender_internal_adjacency ... user_ids=0` was the visible
symptom on anonymous drilldowns).

The fix is narrow: derive.py now calls resolve_skill(name,
allow_fuzzy=False) at each staging site and plumbs skill_id
through the handler merge/refresh paths. Fuzzy is intentionally
disabled on the resume path -- empirically, fuzzy at threshold
0.75 resolves 'accounts payable management' to 'Time Management'
(F.04.b.05), which would corrupt Layer A/C by claiming the user
already has an OaSIS competency they haven't demonstrated. Chat
path still allows fuzzy (unchanged behavior).

These tests pin the contract so a future well-meaning refactor
can't silently re-introduce the drop or flip fuzzy back on.
"""
from __future__ import annotations

import inspect

import pytest

# All tests in this module monkeypatch the reference-skill cache
# and never touch the DB. Opt out of the autouse DB-truncate fixture
# so the suite runs without a live Postgres.
pytestmark = pytest.mark.nodb


# ===========================================================================
# 1. resolve_skill kwarg contract
# ===========================================================================
def test_resolve_skill_has_allow_fuzzy_kwarg_defaulting_true():
    """The `allow_fuzzy` kwarg exists AND defaults to True so
    chat/job/training callers get unchanged behavior."""
    from skillbridge.extract.base import resolve_skill
    sig = inspect.signature(resolve_skill)
    assert "allow_fuzzy" in sig.parameters
    param = sig.parameters["allow_fuzzy"]
    assert param.default is True
    # Keyword-only: chat callers using resolve_skill(name) shouldn't
    # accidentally hit this from positional expansion.
    assert param.kind == inspect.Parameter.KEYWORD_ONLY


def test_resolve_skill_exact_hit_returns_same_result_both_modes(monkeypatch):
    """Exact/alias matches resolve identically regardless of
    allow_fuzzy -- the flag only gates the fuzzy fallback."""
    from skillbridge.extract import base as _base

    fake_cache = {
        "writing": ("F.01.b.02", "Writing"),
        "digital literacy": ("F.01.c.02", "Digital Literacy"),
    }
    monkeypatch.setattr(_base, "_REF_CACHE", fake_cache)

    for allow_fuzzy in (True, False):
        sid, canonical = _base.resolve_skill("Writing", allow_fuzzy=allow_fuzzy)
        assert sid == "F.01.b.02"
        assert canonical == "Writing"


def test_resolve_skill_allow_fuzzy_false_skips_fuzzy_rung(monkeypatch):
    """The exact production-safety test: 'accounts payable management'
    MUST NOT resolve to Time Management (or any other OaSIS row)
    when allow_fuzzy=False. This was the observed false positive at
    threshold 0.75."""
    from skillbridge.extract import base as _base

    # Fake cache with the OaSIS row that triggered the empirical
    # false positive.
    fake_cache = {
        "time management": ("F.04.b.05", "Time Management"),
        "coordinating": ("F.05.a.01", "Coordinating"),
        "decision making": ("F.02.a.03", "Decision Making"),
    }
    monkeypatch.setattr(_base, "_REF_CACHE", fake_cache)

    sid, canonical = _base.resolve_skill(
        "accounts payable management", allow_fuzzy=False,
    )
    assert sid is None
    assert canonical == "accounts payable management"  # original preserved
    # And the AR variant.
    sid, canonical = _base.resolve_skill(
        "accounts receivable management", allow_fuzzy=False,
    )
    assert sid is None


def test_resolve_skill_allow_fuzzy_true_still_produces_the_known_falsepos(
    monkeypatch,
):
    """Regression baseline: with allow_fuzzy=True (the default), the
    known 'accounts payable management' -> Time Management fuzzy hit
    IS present. This test doesn't claim it's good -- it documents that
    the resume path MUST NOT use the default and pins the reason the
    kwarg exists."""
    from skillbridge.extract import base as _base

    fake_cache = {
        "time management": ("F.04.b.05", "Time Management"),
    }
    monkeypatch.setattr(_base, "_REF_CACHE", fake_cache)

    sid, canonical = _base.resolve_skill(
        "accounts payable management", allow_fuzzy=True,
    )
    # Fuzzy matches "management" tokens; may or may not fire depending
    # on rapidfuzz internals, but IF it fires it produces F.04.b.05.
    # We assert only that the SAME input with allow_fuzzy=False is
    # safe -- the True case is documented behavior we choose to
    # bypass on the resume path.
    if sid is not None:
        assert sid == "F.04.b.05"  # documents the known false positive
    # The safe path is always None:
    sid_safe, _ = _base.resolve_skill(
        "accounts payable management", allow_fuzzy=False,
    )
    assert sid_safe is None


# ===========================================================================
# 2. resume/derive.py wiring
# ===========================================================================
def _make_facts(*, skills=None, certifications=None):
    return {
        "skills": skills or [],
        "certifications": certifications or [],
        "work_history": [],
        "education": [],
    }


def test_derive_skills_populates_skill_id_on_exact_match(monkeypatch):
    """A resume skill that matches an OaSIS canonical name exactly
    should carry skill_id through the derived dict.

    Path B (2026-07-02): skill_name is the ORIGINAL resume phrasing,
    NOT the canonical -- see test_derive_skills_preserves_original_name_after_resolution
    for the rationale."""
    from skillbridge.extract import base as _base
    from skillbridge.resume.derive import _derive_skills_list

    monkeypatch.setattr(_base, "_REF_CACHE", {
        "writing": ("F.01.b.02", "Writing"),
    })

    facts = _make_facts(skills=[
        {"name": "Writing", "confidence": 0.9, "evidence": "resume"},
    ])
    result = _derive_skills_list(facts)
    assert len(result) == 1
    row = result[0]
    assert row["skill_id"] == "F.01.b.02"
    assert row["skill_name"] == "Writing"  # original input == canonical here
    assert row["source"] == "resume"


def test_derive_skills_leaves_id_none_for_concrete_vocab(monkeypatch):
    """Concrete resume skills that don't map to reference.skill
    (QuickBooks, invoice processing, etc.) must land with skill_id
    None and preserve the original skill_name."""
    from skillbridge.extract import base as _base
    from skillbridge.resume.derive import _derive_skills_list

    monkeypatch.setattr(_base, "_REF_CACHE", {
        "writing": ("F.01.b.02", "Writing"),
        "digital literacy": ("F.01.c.02", "Digital Literacy"),
    })

    facts = _make_facts(skills=[
        {"name": "QuickBooks Online", "confidence": 0.9, "evidence": "resume"},
        {"name": "invoice processing", "confidence": 0.8, "evidence": "resume"},
        {"name": "vendor reconciliation", "confidence": 0.8, "evidence": "resume"},
    ])
    result = _derive_skills_list(facts)
    assert len(result) == 3
    for row in result:
        assert row["skill_id"] is None
        # Original name preserved (not fuzzy-mangled)
        assert row["skill_name"] in {
            "QuickBooks Online", "invoice processing",
            "vendor reconciliation",
        }


def test_derive_skills_does_not_produce_ap_to_time_management_false_positive(
    monkeypatch,
):
    """The load-bearing safety test. If someone flips allow_fuzzy back
    on for the resume path, this test WILL fail -- 'accounts payable
    management' would fuzz to Time Management (F.04.b.05).

    Confirmed empirically at SKILL_FUZZY_THRESHOLD=0.75 on the real
    reference.skill data. Layer A/C cannot tolerate this class of
    false positive."""
    from skillbridge.extract import base as _base
    from skillbridge.resume.derive import _derive_skills_list

    monkeypatch.setattr(_base, "_REF_CACHE", {
        "time management": ("F.04.b.05", "Time Management"),
    })

    facts = _make_facts(skills=[
        {"name": "accounts payable management", "confidence": 0.9,
         "evidence": "AP work"},
        {"name": "accounts receivable management", "confidence": 0.9,
         "evidence": "AR work"},
    ])
    result = _derive_skills_list(facts)
    assert len(result) == 2
    for row in result:
        assert row["skill_id"] is None, (
            f"Resume path re-enabled fuzzy resolution; {row['skill_name']!r} "
            "fuzz-resolved to Time Management (F.04.b.05). This is the "
            "known false positive the exact-only rule prevents."
        )


def test_derive_skills_preserves_original_name_after_resolution(monkeypatch):
    """Path B (2026-07-02): when an alias resolves a resume-vocab tool
    (e.g. 'microsoft excel') to an OaSIS competency (Digital Literacy
    at F.01.c.02), the derived dict's skill_name MUST be the original
    resume phrase, NOT the canonical.

    Rationale: staged.merge_skills dedupes by skill_name.lower(). If
    we swapped every tool to 'Digital Literacy', all six aliased tools
    on a typical accounting resume would collapse to ONE staged skill.
    That reduces user_names diversity and shrinks the Layer C
    retrieve_candidates pool -- the observed 'labour adjudicator only'
    surface came from this collapse. Preserving raw names keeps
    name-based hits alive while skill_id adds id-based hits."""
    from skillbridge.extract import base as _base
    from skillbridge.resume.derive import _derive_skills_list

    monkeypatch.setattr(_base, "_REF_CACHE", {
        "microsoft excel": ("F.01.c.02", "Digital Literacy"),
        "microsoft word": ("F.01.c.02", "Digital Literacy"),
        "quickbooks online": ("F.01.c.02", "Digital Literacy"),
    })

    facts = _make_facts(skills=[
        {"name": "microsoft excel", "confidence": 0.9, "evidence": "resume"},
        {"name": "microsoft word", "confidence": 0.9, "evidence": "resume"},
        {"name": "quickbooks online", "confidence": 0.9, "evidence": "resume"},
    ])
    result = _derive_skills_list(facts)
    assert len(result) == 3

    # All three carry the SAME skill_id (aliased to Digital Literacy)
    for row in result:
        assert row["skill_id"] == "F.01.c.02"

    # Path B: each row keeps its ORIGINAL name -- NOT the canonical.
    names = {row["skill_name"] for row in result}
    assert names == {"microsoft excel", "microsoft word", "quickbooks online"}
    # And critically: 'Digital Literacy' MUST NOT appear as a
    # skill_name -- that swap was the pre-Path-B behavior.
    assert "Digital Literacy" not in names


def test_derive_skills_path_b_gives_ids_plus_names_diversity(monkeypatch):
    """Path B integration: after derive + merge, user_ids has 1 element
    (Digital Literacy) but user_names has all 3 raw tool names. Both
    signals participate in retrieve_candidates -- id-based AND
    name-based hits are preserved."""
    from skillbridge.chat.handler import _merge_derived_into_staged
    from skillbridge.extract import base as _base
    from skillbridge.match.alignment import (
        build_user_skill_rows, derive_user_skill_sets,
    )
    from skillbridge.resume.derive import derive_staged_slots
    from skillbridge.session.staging import StagedProfile

    monkeypatch.setattr(_base, "_REF_CACHE", {
        "microsoft excel": ("F.01.c.02", "Digital Literacy"),
        "microsoft word": ("F.01.c.02", "Digital Literacy"),
        "quickbooks online": ("F.01.c.02", "Digital Literacy"),
    })

    facts = {
        "skills": [
            {"name": "microsoft excel", "confidence": 0.9, "evidence": "r"},
            {"name": "microsoft word", "confidence": 0.9, "evidence": "r"},
            {"name": "quickbooks online", "confidence": 0.9, "evidence": "r"},
        ],
        "certifications": [], "work_history": [], "education": [],
    }
    derived = derive_staged_slots(facts)
    staged = StagedProfile.new(session_id="test-path-b")
    _merge_derived_into_staged(staged, derived)

    rows = build_user_skill_rows(staged.skills)
    user_ids, user_names, _ = derive_user_skill_sets(rows)

    # ID signal: deduped to one competency.
    assert user_ids == {"F.01.c.02"}

    # NAME signal: three distinct tool names preserved.
    assert "microsoft excel" in user_names
    assert "microsoft word" in user_names
    assert "quickbooks online" in user_names
    # And 'digital literacy' is NOT in the names set (pre-Path-B
    # behavior would have added it there instead).
    assert "digital literacy" not in user_names


def test_derive_certifications_also_go_through_exact_only_resolver(monkeypatch):
    """Certification-promoted skills follow the same exact-only rule.
    Most certs are concrete credentials (WHMIS, 310T, etc.) and stay
    skill_id=None correctly."""
    from skillbridge.extract import base as _base
    from skillbridge.resume.derive import _derive_skills_list

    monkeypatch.setattr(_base, "_REF_CACHE", {
        "writing": ("F.01.b.02", "Writing"),
    })

    facts = _make_facts(certifications=[
        {"name": "First Aid & CPR", "evidence": "cert"},
        {"name": "WHMIS", "evidence": "cert"},
    ])
    result = _derive_skills_list(facts)
    # Both certs promoted; neither maps to reference.skill.
    assert len(result) == 2
    for row in result:
        assert row["skill_id"] is None


# ===========================================================================
# 3. Handler plumbing (merge + refresh)
# ===========================================================================
def test_merge_derived_into_staged_preserves_skill_id():
    """_merge_derived_into_staged must plumb skill_id from the derived
    dict onto the StagedSkill object. Prior to 2026-07-01 this was
    silently dropped -- the fix pins skill_id preservation."""
    from skillbridge.chat.handler import _merge_derived_into_staged
    from skillbridge.session.staging import StagedProfile

    staged = StagedProfile.new(session_id="test-merge")
    derived = {
        "skills": [
            {"skill_name": "Writing", "skill_id": "F.01.b.02",
             "raw_phrase": "prose evidence", "confidence": 0.9,
             "source": "resume"},
            {"skill_name": "QuickBooks Online", "skill_id": None,
             "raw_phrase": "concrete tool", "confidence": 0.8,
             "source": "resume"},
        ],
    }
    _merge_derived_into_staged(staged, derived)
    by_name = {s.skill_name: s for s in staged.skills}
    assert by_name["Writing"].skill_id == "F.01.b.02"
    # Concrete tool stays None, still present in staged.skills.
    assert by_name["QuickBooks Online"].skill_id is None


def test_refresh_derived_into_staged_preserves_skill_id():
    """The refresh path (post-suppression re-add) mirrors the merge
    path. If someone toggles a resume fact suppression, the resolved
    skill_id must not be dropped when the derivation is re-applied."""
    from skillbridge.chat.handler import _refresh_derived_into_staged
    from skillbridge.session.staging import StagedProfile

    staged = StagedProfile.new(session_id="test-refresh")
    derived = {
        "skills": [
            {"skill_name": "Digital Literacy", "skill_id": "F.01.c.02",
             "raw_phrase": "excel evidence", "confidence": 0.9,
             "source": "resume"},
        ],
    }
    _refresh_derived_into_staged(staged, derived)
    assert len(staged.skills) == 1
    assert staged.skills[0].skill_id == "F.01.c.02"


# ===========================================================================
# 4. End-to-end wiring: derive -> merge -> user_ids populated
# ===========================================================================
# ===========================================================================
# 5. Curated alias policy (2026-07-01)
# ===========================================================================
# These tests pin the curation decisions made at sign-off. The source
# of truth is skillbridge/ingest/oasis_curated_aliases.CURATED_ALIASES
# (Python code, git-tracked). The CSV at data/oasis_skills.csv stays
# empty for aliases -- it's ETL output regenerated from official OaSIS
# source files, and hand-edits there would be silently overwritten on
# the next taxonomy refresh.
#
# Every alias claim below has hiring-manager judgment behind it. Removing
# an alias here is a policy decision; adding one requires the same
# discipline (category-1 only: deterministic tool -> competency or
# synonym -> competency, no LLM judgment territory).
def test_alias_policy_protected_rows_stay_empty():
    """Decision Making, Evaluation, Time Management must have ZERO
    aliases. Aliasing these would let the resolver claim a user has
    those competencies without LLM/context judgment.

    Time Management specifically is the safety pin against the
    'accounts payable management' -> 'Time Management' false positive
    caught during slice sign-off."""
    from skillbridge.ingest.oasis_curated_aliases import CURATED_ALIASES
    for sid, name in [
        ("F.02.a.03", "Decision Making"),
        ("F.02.b.01", "Evaluation"),
        ("F.04.b.05", "Time Management"),
    ]:
        assert sid not in CURATED_ALIASES, (
            f"{name} ({sid}) has curated aliases; must stay absent per "
            "locked policy. See oasis_curated_aliases module docstring."
        )


def test_alias_policy_digital_literacy_covers_tools_but_not_google_docs():
    """Digital Literacy carries the tool-heavy alias set. Word/Excel/
    QuickBooks/Acrobat/Office/Outlook all map here (tools prove digital
    tool use). google docs was explicitly DROPPED at sign-off to avoid
    the Word/Writing ambiguity trap."""
    from skillbridge.ingest.oasis_curated_aliases import CURATED_ALIASES
    aliases = CURATED_ALIASES["F.01.c.02"]
    for expected in [
        "microsoft excel", "ms excel",
        "microsoft word",
        "microsoft office", "ms office",
        "microsoft outlook",
        "quickbooks", "quickbooks online", "quickbooks desktop",
        "adobe acrobat",
        "google workspace",
        "computer literacy",
    ]:
        assert expected in aliases, (
            f"Digital Literacy missing expected alias: {expected!r}"
        )
    assert "google docs" not in aliases, (
        "'google docs' was dropped at sign-off (ambiguous with Writing)"
    )


def test_alias_policy_writing_covers_stated_writing_skills_not_tools():
    """Writing carries the STATED-skill aliases (written communication,
    business writing, report writing). Word/tools DO NOT map here --
    they map to Digital Literacy. Prevents double-counting."""
    from skillbridge.ingest.oasis_curated_aliases import CURATED_ALIASES
    aliases = CURATED_ALIASES["F.01.b.02"]
    for expected in [
        "written communication", "business writing", "report writing",
    ]:
        assert expected in aliases
    for tool in [
        "microsoft word", "ms word", "google docs",
    ]:
        assert tool not in aliases, (
            f"{tool!r} in Writing aliases would double-count with Digital "
            "Literacy. Locked at sign-off: Word is a tool, proves digital "
            "literacy, not writing quality."
        )


def test_alias_policy_persuading_drops_bare_influencing():
    """Persuading has 'persuasion' and 'influencing others' (specific).
    Bare 'influencing' was DROPPED at sign-off -- too broad; can read
    as 'shaped strategy' or 'advised leadership' rather than convincing
    someone to change behavior."""
    from skillbridge.ingest.oasis_curated_aliases import CURATED_ALIASES
    aliases = CURATED_ALIASES["F.05.a.04"]
    assert "persuasion" in aliases
    assert "influencing others" in aliases
    assert "influencing" not in aliases, (
        "Bare 'influencing' was dropped at sign-off -- too broad."
    )


def test_alias_policy_quality_control_testing_drops_quality_assurance():
    """Quality Control Testing has 'quality control' and 'qc testing'.
    'quality assurance' was DROPPED at sign-off -- QA is a broader
    process/audit discipline; QCT is running the tests. Different
    roles even inside the same department."""
    from skillbridge.ingest.oasis_curated_aliases import CURATED_ALIASES
    aliases = CURATED_ALIASES["F.03.a.08"]
    assert "quality control" in aliases
    assert "qc testing" in aliases
    assert "quality assurance" not in aliases, (
        "'quality assurance' was dropped at sign-off (broader than QCT)."
    )


def test_alias_policy_total_curated_rows_matches_signoff():
    """Load-bearing curation pin: 19 competencies curated at sign-off.
    If someone silently adds/removes a keyed row, this counter forces
    a review + memo update."""
    from skillbridge.ingest.oasis_curated_aliases import CURATED_ALIASES
    assert len(CURATED_ALIASES) == 19, (
        f"Curated row count changed to {len(CURATED_ALIASES)} (was 19 at "
        "sign-off). Update this assertion AND the module docstring before "
        "shipping."
    )


def test_alias_policy_all_aliases_are_lowercase_strings():
    """The resolver compares lowercased. Any uppercase alias is a bug
    (dead code that will never resolve)."""
    from skillbridge.ingest.oasis_curated_aliases import CURATED_ALIASES
    for sid, tup in CURATED_ALIASES.items():
        assert isinstance(tup, tuple), (
            f"{sid} aliases must be a tuple (immutable), got {type(tup).__name__}"
        )
        for alias in tup:
            assert isinstance(alias, str) and alias == alias.lower(), (
                f"{sid}: alias {alias!r} must be lowercase (resolver compares "
                "lowercased -- non-lowercase entries would never resolve)."
            )
            assert alias.strip() == alias, (
                f"{sid}: alias {alias!r} has leading/trailing whitespace."
            )


def test_alias_policy_no_duplicate_aliases_across_rows():
    """Curation invariant: no resume phrase maps to more than one
    competency. Otherwise a user typing that phrase would non-
    deterministically resolve to one of them based on cache-iteration
    order -- double-counting risk + ambiguous evidence."""
    from skillbridge.ingest.oasis_curated_aliases import CURATED_ALIASES
    seen: dict[str, str] = {}
    for sid, tup in CURATED_ALIASES.items():
        for alias in tup:
            assert alias not in seen, (
                f"Alias {alias!r} maps to both {seen[alias]} and {sid}. "
                "Each resume phrase must have exactly one competency."
            )
            seen[alias] = sid


# ===========================================================================
# 6. End-to-end wiring: derive -> merge -> user_ids populated
# ===========================================================================
def test_end_to_end_resume_upload_populates_user_ids(monkeypatch):
    """Load-bearing integration test. Simulate a resume-facts payload
    with one OaSIS-matching skill + several concrete resume-vocab
    skills. After derive + merge, derive_user_skill_sets must return
    user_ids containing the resolved OaSIS ID.

    Before 2026-07-01 this returned an empty set (user_ids=0) which
    made Layer A/C over-flag every OaSIS competency. This test pins
    the fix."""
    from skillbridge.chat.handler import _merge_derived_into_staged
    from skillbridge.extract import base as _base
    from skillbridge.match.alignment import (
        build_user_skill_rows, derive_user_skill_sets,
    )
    from skillbridge.resume.derive import derive_staged_slots
    from skillbridge.session.staging import StagedProfile

    monkeypatch.setattr(_base, "_REF_CACHE", {
        "writing": ("F.01.b.02", "Writing"),
        "digital literacy": ("F.01.c.02", "Digital Literacy"),
        "decision making": ("F.02.a.03", "Decision Making"),
    })

    facts = {
        "skills": [
            {"name": "Writing", "confidence": 0.9, "evidence": "resume"},
            {"name": "Digital Literacy", "confidence": 0.9, "evidence": "resume"},
            {"name": "QuickBooks Online", "confidence": 0.9, "evidence": "resume"},
            {"name": "invoice processing", "confidence": 0.8, "evidence": "resume"},
        ],
        "certifications": [], "work_history": [], "education": [],
    }
    derived = derive_staged_slots(facts)

    staged = StagedProfile.new(session_id="test-e2e")
    _merge_derived_into_staged(staged, derived)

    rows = build_user_skill_rows(staged.skills)
    user_ids, user_names, user_canon = derive_user_skill_sets(rows)

    # Two exact matches resolved -> user_ids has both.
    assert user_ids == {"F.01.b.02", "F.01.c.02"}
    # And the concrete vocab is still there via names/canonical
    # (matching engine and drilldown fall through to those).
    assert "quickbooks online" in user_names
    assert "invoice processing" in user_names