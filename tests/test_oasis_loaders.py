"""Unit tests for matching v2 step 1 -- OaSIS + SCT occupation-title loaders.

Tests exercise the factored row-parsers (`_parse_title_synonym_rows`,
`_parse_lead_statement_rows`) with inline CSV-row dicts. No disk reads,
no network, no DB. The `nodb` marker keeps the conftest TRUNCATE off.

Things these tests pin:
  - NOC codes accepted in both bare ("21232") and sub-occupation
    ("21232.00") form, normalized to bare 5-digit
  - Multi-language and multi-source rows deduplicate on (noc, title)
  - Heuristic column-name lookup handles OaSIS / SCT variants
  - Malformed rows skipped silently, never crash the import
"""
from __future__ import annotations

import os

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from skillbridge.ingest.reference import (
    _first,
    _parse_lead_statement_rows,
    _parse_title_synonym_rows,
)

pytestmark = pytest.mark.nodb


# ---------------------------------------------------------------------------
# _first -- column-name fallback helper
# ---------------------------------------------------------------------------
def test_first_returns_first_non_empty_value():
    row = {"noc_code": "21232", "Title": "Software Developer"}
    assert _first(row, ("noc_code", "NOC Code")) == "21232"
    assert _first(row, ("NOC Code", "noc_code")) == "21232"


def test_first_skips_empty_strings():
    row = {"noc_code": "", "NOC Code": "21232"}
    assert _first(row, ("noc_code", "NOC Code")) == "21232"


def test_first_returns_empty_when_no_keys_match():
    row = {"title": "Software Developer"}
    assert _first(row, ("noc_code", "NOC Code", "Code")) == ""


def test_first_handles_non_string_values():
    """Some CSV libraries return numeric values for purely-numeric columns;
    _first must convert to string and strip."""
    row = {"noc_code": 21232}
    assert _first(row, ("noc_code",)) == "21232"


# ---------------------------------------------------------------------------
# _parse_title_synonym_rows
# ---------------------------------------------------------------------------
def test_title_synonym_parses_minimal_csv():
    rows = [
        {"noc_code": "21232", "title": "Software Developer"},
        {"noc_code": "21232", "title": "Application Programmer"},
        {"noc_code": "72200", "title": "Electrician"},
    ]
    out = _parse_title_synonym_rows(rows, lang="en", source="oasis_example")
    assert ("21232", "Software Developer", "en", "oasis_example") in out
    assert ("21232", "Application Programmer", "en", "oasis_example") in out
    assert ("72200", "Electrician", "en", "oasis_example") in out
    assert len(out) == 3


def test_title_synonym_normalizes_sub_occupation_codes():
    """OaSIS data uses sub-occupation suffixes like 12100.00 -- normalize
    to the 5-digit NOC 2021 code so we match reference.occupation."""
    rows = [
        {"noc_code": "12100.00", "title": "Executive assistants"},
        {"noc_code": "12100.01", "title": "Executive assistant - legal"},
    ]
    out = _parse_title_synonym_rows(rows, lang="en", source="oasis_example")
    assert all(noc == "12100" for noc, _, _, _ in out)
    titles = {title for _, title, _, _ in out}
    assert titles == {"Executive assistants", "Executive assistant - legal"}


def test_title_synonym_handles_alternative_column_names():
    """OaSIS English uses 'noc_code'/'title'; SCT French may use
    'Code CNP 2021'/'Titre alternatif'. Heuristic lookup picks the
    right one."""
    rows = [
        {"Code CNP 2021": "21232", "Titre alternatif": "Programmeur"},
    ]
    out = _parse_title_synonym_rows(rows, lang="fr", source="sct_alternative")
    assert out == [("21232", "Programmeur", "fr", "sct_alternative")]


def test_title_synonym_deduplicates_within_one_pass():
    """Two rows with the same (noc, title) collapse to one tuple."""
    rows = [
        {"noc_code": "21232", "title": "Software Developer"},
        {"noc_code": "21232", "title": "Software Developer"},   # exact dup
    ]
    out = _parse_title_synonym_rows(rows, lang="en", source="oasis_example")
    assert len(out) == 1


@pytest.mark.parametrize("bad_noc", ["", "abc", "21", "21232X", "212322"])
def test_title_synonym_skips_invalid_noc_codes(bad_noc):
    rows = [{"noc_code": bad_noc, "title": "Some Title"}]
    out = _parse_title_synonym_rows(rows, lang="en", source="oasis_example")
    assert out == []


def test_title_synonym_skips_rows_missing_title_or_noc():
    rows = [
        {"noc_code": "21232"},                  # no title
        {"title": "Software Developer"},         # no noc
        {"noc_code": "", "title": "Empty"},      # empty noc
        {"noc_code": "21232", "title": ""},      # empty title
    ]
    out = _parse_title_synonym_rows(rows, lang="en", source="oasis_example")
    assert out == []


def test_title_synonym_strips_whitespace():
    rows = [
        {"noc_code": "  21232  ", "title": "  Software Developer  "},
    ]
    out = _parse_title_synonym_rows(rows, lang="en", source="oasis_example")
    assert out == [("21232", "Software Developer", "en", "oasis_example")]


def test_title_synonym_survives_non_dict_rows():
    """csv.DictReader produces dicts, but a defensive loader shouldn't
    crash if something else slips through (e.g. a string from a header
    row that was accidentally included)."""
    rows = [
        {"noc_code": "21232", "title": "Software Developer"},
        "not-a-dict",
        None,
        {"noc_code": "72200", "title": "Electrician"},
    ]
    out = _parse_title_synonym_rows(rows, lang="en", source="oasis_example")
    assert len(out) == 2


def test_title_synonym_empty_input_returns_empty():
    assert _parse_title_synonym_rows([], lang="en", source="oasis_example") == []


# ---------------------------------------------------------------------------
# _parse_lead_statement_rows
# ---------------------------------------------------------------------------
def test_lead_statement_parses_minimal_csv():
    rows = [
        {"noc_code": "21232", "lead_statement": "Software developers analyse..."},
        {"noc_code": "72200", "Lead Statement": "Electricians install..."},
    ]
    out = _parse_lead_statement_rows(rows)
    assert ("21232", "Software developers analyse...") in out
    assert ("72200", "Electricians install...") in out


def test_lead_statement_normalizes_sub_occupation_codes():
    rows = [{"noc_code": "12100.00", "lead_statement": "EA work..."}]
    out = _parse_lead_statement_rows(rows)
    assert out == [("12100", "EA work...")]


def test_lead_statement_skips_invalid_or_missing():
    rows = [
        {"noc_code": "21232"},               # no statement
        {"lead_statement": "Orphaned"},      # no noc
        {"noc_code": "abc", "lead_statement": "Bad noc"},
        {"noc_code": "21232", "lead_statement": "Real one"},
    ]
    out = _parse_lead_statement_rows(rows)
    assert out == [("21232", "Real one")]


# ---------------------------------------------------------------------------
# Step 1 schema invariants (verified at SQL layer via run_pipeline --schema;
# this test pins the Python-side constants the loaders depend on).
# ---------------------------------------------------------------------------
def test_oasis_version_constant_is_defined():
    """The constant must exist so loaders can stamp reference.occupation
    rows with the import version, supporting later re-syncs."""
    from config import OASIS_VERSION
    assert OASIS_VERSION
    assert isinstance(OASIS_VERSION, str)
