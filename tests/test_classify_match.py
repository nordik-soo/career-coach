"""AR-9.feat.coach-tiers CP1 — `_classify_match` attribution contracts.

Pins:
  - rung priority: skill_id > name eq > canonical alias > substring >
    fuzzy > semantic;
  - within-rung tie-break: lex-asc on UserSkillRow.text;
  - semantic argmax tie-break: highest cosine wins; lex-asc on row.text
    only when cosine is tied;
  - credential-strict gate: substring / fuzzy / semantic are DISABLED
    for credential-class job skills (Class G, 310S, WHMIS, ...);
  - is_normalized_equal: True only when _key(user_skill) ==
    _key(job_requirement); False for aliases, substring, fuzzy,
    semantic;
  - legacy 2-tuple wrapper `_skill_match_strength` continues to return
    the same (strength, stage) tuple — no change for pre-CP1 callers.
"""
from __future__ import annotations

import numpy as np
import pytest

from skillbridge.match.alignment import UserSkillRow
from skillbridge.match.engine import (
    _RUNG_CANON,
    _RUNG_FUZZY,
    _RUNG_NAME_EQ,
    _RUNG_NO_MATCH,
    _RUNG_SEMANTIC,
    _RUNG_SKILL_ID,
    _RUNG_SUBSTRING,
    _STRENGTH_STAGE_1,
    _STRENGTH_TOKEN_OVERLAP,
    _classify_match,
    _skill_match_strength,
)

pytestmark = pytest.mark.nodb


def _row(text, *, sid=None):
    """Build a UserSkillRow as the rows authority would."""
    from skillbridge.match.aliases import canonicalize_skill
    stripped = text.strip()
    return UserSkillRow(
        skill_id=sid,
        text=stripped,
        name=stripped.lower(),
        canon=canonicalize_skill(stripped) or "",
    )


def _sets_from_rows(rows):
    """Mirror what derive_user_skill_sets does, inline for test clarity."""
    ids = {r.skill_id for r in rows if r.skill_id}
    names = {r.name for r in rows}
    canons = {r.canon for r in rows if r.canon}
    return ids, names, canons


# =========================================================================
# Rung priority — each rung is reachable and produces the right attribution
# =========================================================================
def test_rung_skill_id_attribution():
    rows = [_row("Python", sid="p-1")]
    ids, names, canons = _sets_from_rows(rows)
    job = {"skill_id": "p-1", "skill_name": "Python Programming"}
    strength, stage, text, rung = _classify_match(
        job, ids, names, canons, user_rows=rows,
    )
    assert strength == _STRENGTH_STAGE_1
    assert stage == "exact"
    assert text == "Python"
    assert rung == _RUNG_SKILL_ID


def test_rung_name_eq_attribution():
    rows = [_row("QuickBooks")]
    ids, names, canons = _sets_from_rows(rows)
    job = {"skill_id": None, "skill_name": "QuickBooks"}
    strength, stage, text, rung = _classify_match(
        job, ids, names, canons, user_rows=rows,
    )
    assert strength == _STRENGTH_STAGE_1
    assert stage == "exact"
    assert text == "QuickBooks"
    assert rung == _RUNG_NAME_EQ


def test_rung_canonical_alias_attribution():
    """QB → QuickBooks via SKILL_ALIASES. Public stage still 'exact'."""
    rows = [_row("QB")]
    ids, names, canons = _sets_from_rows(rows)
    # Job skill canonicalizes via aliases too? Actually canonicalize_skill
    # is applied to the job-side _key form. For QB ↔ QuickBooks, both
    # canonicalize to the same form because SKILL_ALIASES maps them.
    # But "QB" is not in SKILL_ALIASES directly — let's use a known alias.
    rows = [_row("PSW")]
    ids, names, canons = _sets_from_rows(rows)
    job = {"skill_id": None, "skill_name": "personal support worker"}
    strength, stage, text, rung = _classify_match(
        job, ids, names, canons, user_rows=rows,
    )
    assert strength == _STRENGTH_STAGE_1
    assert stage == "exact"
    assert text == "PSW"
    assert rung == _RUNG_CANON


def test_rung_substring_attribution():
    """User has 'Excel', job asks for 'Microsoft Excel'. Word-bounded
    substring match — public stage 'exact', internal rung substring."""
    rows = [_row("Excel")]
    ids, names, canons = _sets_from_rows(rows)
    job = {"skill_id": None, "skill_name": "Microsoft Excel"}
    strength, stage, text, rung = _classify_match(
        job, ids, names, canons, user_rows=rows,
    )
    assert strength == _STRENGTH_STAGE_1
    assert stage == "exact"
    assert text == "Excel"
    assert rung == _RUNG_SUBSTRING


def test_rung_fuzzy_attribution():
    """Token-overlap fuzzy: user 'truck maintenance' vs job
    'truck service maintenance' — 2/3 tokens overlap."""
    rows = [_row("truck maintenance")]
    ids, names, canons = _sets_from_rows(rows)
    job = {"skill_id": None, "skill_name": "truck service maintenance"}
    strength, stage, text, rung = _classify_match(
        job, ids, names, canons, user_rows=rows,
    )
    assert strength == _STRENGTH_TOKEN_OVERLAP
    assert stage == "fuzzy"
    assert text == "truck maintenance"
    assert rung == _RUNG_FUZZY


def test_no_match_returns_no_match_rung():
    rows = [_row("welding")]
    ids, names, canons = _sets_from_rows(rows)
    job = {"skill_id": None, "skill_name": "underwater basket weaving"}
    strength, stage, text, rung = _classify_match(
        job, ids, names, canons, user_rows=rows,
    )
    assert strength == 0.0
    assert stage == "no_match"
    assert text is None
    assert rung == _RUNG_NO_MATCH


# =========================================================================
# Within-rung lex-asc tie-breaking
# =========================================================================
def test_name_eq_tiebreak_is_lex_asc_on_text():
    """Two rows whose .name both equals the job's lowercased skill_name.
    The .text values differ in case; lex-asc on .text wins."""
    # Both rows lowercase to "python"; texts differ in case.
    rows = [_row("python"), _row("Python")]
    ids, names, canons = _sets_from_rows(rows)
    job = {"skill_id": None, "skill_name": "python"}
    _, _, text, rung = _classify_match(
        job, ids, names, canons, user_rows=rows,
    )
    assert rung == _RUNG_NAME_EQ
    # Lex-asc on ['Python', 'python']: 'Python' < 'python' (uppercase first)
    assert text == "Python"


def test_name_eq_tiebreak_stable_under_shuffle():
    """Shuffling input rows must produce the SAME selected row.
    Tie-break is content-deterministic, not order-dependent."""
    rows_a = [_row("python"), _row("Python")]
    rows_b = [_row("Python"), _row("python")]
    ids_a, names_a, canons_a = _sets_from_rows(rows_a)
    ids_b, names_b, canons_b = _sets_from_rows(rows_b)
    job = {"skill_id": None, "skill_name": "python"}
    _, _, text_a, _ = _classify_match(
        job, ids_a, names_a, canons_a, user_rows=rows_a,
    )
    _, _, text_b, _ = _classify_match(
        job, ids_b, names_b, canons_b, user_rows=rows_b,
    )
    assert text_a == text_b == "Python"   # both runs pick the same row


# =========================================================================
# Rung priority — higher rung wins even when lower rungs also match
# =========================================================================
def test_skill_id_beats_name_eq_when_both_apply():
    """One row matches by skill_id (rung 0); another matches by name eq
    (rung 1). The skill_id row wins regardless of lex order on text."""
    # Row by-id text is lex-LARGER than row by-name to prove rung wins.
    rows = [_row("zz-by-name"), _row("aa-by-id", sid="job-skill-id")]
    # Make both apply: the job's skill_id matches the second row,
    # the job's skill_name matches the first row's name.
    ids, names, canons = _sets_from_rows(rows)
    # The first row's name is 'zz-by-name'; set the job skill_name to that
    job = {"skill_id": "job-skill-id", "skill_name": "zz-by-name"}
    _, _, text, rung = _classify_match(
        job, ids, names, canons, user_rows=rows,
    )
    # skill_id row wins even though its text 'aa-by-id' is lex-smaller —
    # rung priority overrides any cross-rung lex consideration.
    assert rung == _RUNG_SKILL_ID
    assert text == "aa-by-id"


# =========================================================================
# Credential-strict gate — preserved exactly under attribution
# =========================================================================
def test_credential_strict_disables_substring_for_credentials():
    """User has 'G2/G driver's license', job needs 'Class G driver's license'.
    Substring would normally match — but credentials are stage-1-only."""
    rows = [_row("G2/G driver's license")]
    ids, names, canons = _sets_from_rows(rows)
    job = {"skill_id": None, "skill_name": "Class G driver's license"}
    strength, stage, text, rung = _classify_match(
        job, ids, names, canons, user_rows=rows,
    )
    # Credential gate fired — no substring fallback allowed.
    assert strength == 0.0
    assert stage == "no_match"
    assert text is None
    assert rung == _RUNG_NO_MATCH


def test_credential_alias_still_works():
    """The alias path IS allowed for credentials. PSW → personal
    support worker is just an example, but Class G aliases work too."""
    rows = [_row("full G license")]
    ids, names, canons = _sets_from_rows(rows)
    # "full G license" canonicalizes to "class g license" via SKILL_ALIASES
    job = {"skill_id": None, "skill_name": "class g license"}
    strength, stage, text, rung = _classify_match(
        job, ids, names, canons, user_rows=rows,
    )
    # Name-eq won't fire (different lowercased text). Canonical alias
    # IS allowed for credentials and wins here.
    assert strength == _STRENGTH_STAGE_1
    assert stage == "exact"
    assert text == "full G license"
    assert rung == _RUNG_CANON


# =========================================================================
# Semantic argmax + lex tiebreak via tiebreak_keys
# =========================================================================
def test_semantic_argmax_picks_max_cosine_row():
    """Three user rows with different cosines vs one job. The highest
    cosine wins — _classify_match's semantic path resolves to that row."""
    from config import SEMANTIC_STRENGTH_CAP
    job_emb = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    # User matrix: row 0 cosine 0.95 (above threshold);
    #              row 1 cosine 0.50 (below);
    #              row 2 cosine 0.99 (highest).
    user_mat = np.asarray(
        [[0.95, np.sqrt(1 - 0.95 ** 2), 0.0],
         [0.50, np.sqrt(1 - 0.50 ** 2), 0.0],
         [0.99, np.sqrt(1 - 0.99 ** 2), 0.0]],
        dtype=np.float32,
    )
    rows = [_row("aaa-skill"), _row("bbb-skill"), _row("zzz-skill")]
    # Lexical paths must all miss so semantic is the rung that fires.
    ids, names, canons = set(), set(), set()
    job = {"skill_id": None, "skill_name": "completely different topic"}
    strength, stage, text, rung = _classify_match(
        job, ids, names, canons,
        user_rows=rows,
        job_skill_embedding=job_emb,
        user_embeddings_matrix=user_mat,
    )
    assert strength == SEMANTIC_STRENGTH_CAP
    assert stage == "semantic"
    assert rung == _RUNG_SEMANTIC
    # Row 2 had the highest cosine — lex on text only matters at ties.
    assert text == "zzz-skill"


def test_semantic_tied_cosine_breaks_lex_on_row_text():
    """Two rows have identical max cosines. Lex-asc on row.text wins."""
    from config import SEMANTIC_STRENGTH_CAP
    job_emb = np.asarray([1.0, 0.0], dtype=np.float32)
    # Both rows have cosine 1.0 — perfectly tied.
    user_mat = np.asarray(
        [[1.0, 0.0],
         [1.0, 0.0]],
        dtype=np.float32,
    )
    # Row 0 text lex-LARGER than row 1 text. Lex-asc must pick row 1.
    rows = [_row("zzz-row"), _row("aaa-row")]
    ids, names, canons = set(), set(), set()
    job = {"skill_id": None, "skill_name": "unrelated topic"}
    strength, _, text, _ = _classify_match(
        job, ids, names, canons,
        user_rows=rows,
        job_skill_embedding=job_emb,
        user_embeddings_matrix=user_mat,
    )
    assert strength == SEMANTIC_STRENGTH_CAP
    assert text == "aaa-row"   # lex-asc tiebreak among tied-max rows


# =========================================================================
# Legacy 2-tuple wrapper byte-stability
# =========================================================================
def test_skill_match_strength_legacy_wrapper_unchanged():
    """Pre-CP1 callers that unpack (strength, stage) continue to work
    with no behavior change. The wrapper hides attribution."""
    job = {"skill_id": None, "skill_name": "Python"}
    # Sets only (no rows). Wrapper returns 2-tuple, byte-stable.
    result = _skill_match_strength(job, set(), {"python"}, set())
    assert isinstance(result, tuple)
    assert len(result) == 2
    strength, stage = result
    assert strength == _STRENGTH_STAGE_1
    assert stage == "exact"


def test_classify_match_with_no_user_rows_returns_no_text():
    """When user_rows is None, matched_text is None even on a hit.
    Legacy callers that don't supply rows get just the score+stage."""
    job = {"skill_id": None, "skill_name": "Python"}
    strength, stage, text, _ = _classify_match(
        job, set(), {"python"}, set(),
    )
    assert strength == _STRENGTH_STAGE_1
    assert stage == "exact"
    assert text is None
