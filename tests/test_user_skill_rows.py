"""AR-9.feat.coach-tiers CP1 — UserSkillRow attribution authority.

Pins:
  - the evidence-eligibility gate on `build_user_skill_rows` exactly
    matches what `build_user_skill_sets` used to do byte-for-byte;
  - rows preserve input order (the semantic-argmax path relies on this);
  - rows preserve the ORIGINAL user-typed text (attribution display);
  - `derive_user_skill_sets` produces the same (ids, names, canon)
    triple the legacy direct construction did, so the matcher reads
    identical sets through the new authority;
  - `is_normalized_equal` is literal _key()-equality only — NOT
    alias-folded, NOT substring, NOT fuzzy/semantic.
"""
from __future__ import annotations

import pytest

from skillbridge.match.alignment import (
    SkillAlignment,
    UserSkillRow,
    build_user_skill_rows,
    derive_user_skill_sets,
    is_normalized_equal,
)
from skillbridge.match.adjacent import build_user_skill_sets
from skillbridge.session.staging import StagedSkill

pytestmark = pytest.mark.nodb


def _ss(name, *, conf=0.8, source="resume", skill_id=None):
    return StagedSkill(
        skill_name=name, skill_id=skill_id,
        confidence=conf, source=source,
    )


# =========================================================================
# Evidence-eligibility gate parity
# =========================================================================
def test_rows_admit_resume_chat_sources_at_threshold():
    rows = build_user_skill_rows([
        _ss("Python", source="resume", conf=0.8),
        _ss("Excel", source="chat", conf=0.6),
    ])
    assert [r.text for r in rows] == ["Python", "Excel"]


def test_rows_reject_off_source():
    rows = build_user_skill_rows([
        _ss("Python", source="inferred", conf=0.9),
        _ss("Excel", source="", conf=0.9),
    ])
    assert rows == []


def test_rows_reject_below_confidence_floor():
    rows = build_user_skill_rows([
        _ss("Python", conf=0.59),
        _ss("Excel", conf=0.0),
    ])
    assert rows == []


def test_rows_reject_invalid_confidence_types():
    rows = build_user_skill_rows([
        _ss("Python", conf=True),       # bool — rejected even though 1.0
        _ss("Excel", conf=float("nan")),
        _ss("Pandas", conf=float("inf")),
        _ss("NumPy", conf=1.5),         # out of [0, 1]
    ])
    assert rows == []


def test_rows_reject_non_str_skill_name():
    # StagedSkill is a runtime-permissive dataclass — forged cookies can
    # smuggle non-str into skill_name. Gate must reject defensively.
    bad = StagedSkill(skill_name=7, confidence=0.9, source="resume")  # type: ignore
    rows = build_user_skill_rows([bad])
    assert rows == []


def test_rows_reject_empty_after_strip():
    rows = build_user_skill_rows([_ss("   ", conf=0.9)])
    assert rows == []


def test_rows_reject_non_stagedskill_entries():
    rows = build_user_skill_rows([
        "Python",                          # type: ignore
        {"skill_name": "Excel"},           # type: ignore
        _ss("Real", conf=0.9),
    ])
    assert [r.text for r in rows] == ["Real"]


# =========================================================================
# Order preservation (the semantic-argmax path depends on this)
# =========================================================================
def test_rows_preserve_input_order():
    rows = build_user_skill_rows([
        _ss("Zoo"), _ss("Apple"), _ss("Mango"),
    ])
    assert [r.text for r in rows] == ["Zoo", "Apple", "Mango"]


def test_rows_do_not_dedup_same_normalized_name():
    # Two rows can share .name — the semantic-argmax path needs both
    # embedding positions distinct. Set-derivation is where folding
    # happens.
    rows = build_user_skill_rows([_ss("QB"), _ss("qb")])
    assert len(rows) == 2
    assert rows[0].text == "QB" and rows[1].text == "qb"
    assert rows[0].name == "qb" and rows[1].name == "qb"


# =========================================================================
# Attribution preservation
# =========================================================================
def test_row_text_is_original_user_input():
    rows = build_user_skill_rows([_ss("QuickBooks Online")])
    assert rows[0].text == "QuickBooks Online"   # original casing
    assert rows[0].name == "quickbooks online"   # match-lookup form


def test_row_text_strips_outer_whitespace_but_keeps_casing():
    rows = build_user_skill_rows([_ss("  QuickBooks Online  ")])
    assert rows[0].text == "QuickBooks Online"   # outer whitespace gone
    assert rows[0].name == "quickbooks online"


def test_row_canon_folds_known_alias():
    rows = build_user_skill_rows([_ss("PSW")])
    # canonicalize_skill maps "psw" → "personal support worker"
    assert rows[0].canon == "personal support worker"


def test_row_canon_falls_through_when_no_alias():
    rows = build_user_skill_rows([_ss("Forklift")])
    # No alias — canonicalize returns the _key() form
    assert rows[0].canon == "forklift"


def test_row_skill_id_str_coerced_when_present():
    rows = build_user_skill_rows([_ss("Python", skill_id="abc-123")])
    assert rows[0].skill_id == "abc-123"


def test_row_skill_id_is_none_when_absent():
    rows = build_user_skill_rows([_ss("Python")])
    assert rows[0].skill_id is None


# =========================================================================
# Sets derived from rows — single authority
# =========================================================================
def test_derive_sets_matches_legacy_build_user_skill_sets():
    """`build_user_skill_sets` is now a thin wrapper over rows. The
    triple it returns must equal `derive_user_skill_sets(rows)` for
    the same input — there is no parallel construction path."""
    skills = [
        _ss("QuickBooks", skill_id="qb-1"),
        _ss("AP", conf=0.7),
        _ss("PSW", source="chat"),
        _ss("noise", conf=0.3),               # dropped
    ]
    rows = build_user_skill_rows(skills)
    derived = derive_user_skill_sets(rows)
    legacy = build_user_skill_sets(skills)
    assert derived == legacy


def test_derive_sets_folds_duplicate_names():
    rows = build_user_skill_rows([_ss("QB"), _ss("qb")])
    ids, names, canons = derive_user_skill_sets(rows)
    assert names == {"qb"}                    # set folds the dup
    assert canons == {"qb"}


def test_derive_sets_collects_skill_ids_when_present():
    rows = build_user_skill_rows([
        _ss("Python", skill_id="p-1"),
        _ss("Excel"),                          # no id
    ])
    ids, _, _ = derive_user_skill_sets(rows)
    assert ids == {"p-1"}


# =========================================================================
# is_normalized_equal — the strong-phrasing gate
# =========================================================================
def test_is_normalized_equal_true_for_literal_match_modulo_case():
    assert is_normalized_equal("QuickBooks", "quickbooks") is True
    assert is_normalized_equal("Class G License", "class g license") is True


def test_is_normalized_equal_true_through_punctuation_normalization():
    # _key strips apostrophes and collapses non-alphanum to single space
    assert is_normalized_equal("driver's license", "drivers license") is True


def test_is_normalized_equal_false_for_alias_pair():
    # Alias-folded equality is NOT literal equality. The reserved
    # phrasing "they ask for X, which you have" must NOT fire here.
    assert is_normalized_equal("QB", "QuickBooks") is False
    assert is_normalized_equal("PSW", "personal support worker") is False


def test_is_normalized_equal_false_for_substring_pair():
    # "Excel" appears inside "Microsoft Excel" but they are not literally
    # equal once normalized — the substring rung lives at the matcher,
    # not at the prompt-wording boundary.
    assert is_normalized_equal("Excel", "Microsoft Excel") is False
    assert is_normalized_equal("Microsoft Excel", "Excel") is False


def test_is_normalized_equal_false_for_empty_inputs():
    assert is_normalized_equal("", "") is False
    assert is_normalized_equal("Python", "") is False
    assert is_normalized_equal("", "Python") is False


# =========================================================================
# SkillAlignment dataclass smoke
# =========================================================================
def test_skill_alignment_is_frozen():
    a = SkillAlignment(
        user_skill="QB", job_requirement="QuickBooks",
        stage="exact", source="required",
        is_normalized_equal=False,
    )
    with pytest.raises((AttributeError, Exception)):
        a.stage = "fuzzy"  # type: ignore


def test_skill_alignment_carries_required_fields():
    a = SkillAlignment(
        user_skill="QuickBooks", job_requirement="QuickBooks",
        stage="exact", source="required",
        is_normalized_equal=True,
    )
    assert a.user_skill == "QuickBooks"
    assert a.job_requirement == "QuickBooks"
    assert a.stage == "exact"
    assert a.source == "required"
    assert a.is_normalized_equal is True


def test_user_skill_row_is_frozen():
    r = UserSkillRow(skill_id=None, text="QB", name="qb", canon="quickbooks")
    with pytest.raises((AttributeError, Exception)):
        r.text = "different"  # type: ignore
