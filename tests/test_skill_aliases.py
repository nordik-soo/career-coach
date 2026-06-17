"""Unit tests for skillbridge.match.aliases.

These verify the canonicalization layer that Sprint 5 step 2 added.
The tests do NOT touch the database -- they cover only string handling
inside the alias module, so they run fast and stay green even when the
SCCC dataset shifts.
"""
from __future__ import annotations

import pytest

from skillbridge.match.aliases import SKILL_ALIASES, canonicalize_skill, _key

# Pure string-handling tests -- no DB needed. Opt out of the conftest's
# autouse TRUNCATE so these run even when Postgres is unreachable.
pytestmark = pytest.mark.nodb


# ---------------------------------------------------------------------------
# _key: punctuation / case / whitespace normalization
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("Class G Licence", "class g licence"),
    ("Class-G Licence", "class g licence"),
    ("class  g  licence ", "class g licence"),
    ("Driver's License", "drivers license"),
    ("Driver’s License", "drivers license"),  # curly apostrophe also stripped
    ("WHMIS 2015", "whmis 2015"),
    ("", ""),
    ("   ", ""),
])
def test_key_normalizes_punctuation_case_and_whitespace(raw, expected):
    assert _key(raw) == expected


# ---------------------------------------------------------------------------
# canonicalize_skill: alias map collapses variants to the same canonical
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("variants, canonical", [
    # Driver / licence -- all the Class G spellings collapse to one form
    (
        ["Class G licence", "Class G license", "G licence", "G license",
         "Class-G Licence", "valid g licence", "Ontario G License"],
        "class g license",
    ),
    # Customer service buckets
    (
        ["Customer Service", "client service", "customer support",
         "Client Services", "customer care"],
        "customer service",
    ),
    # PSW abbreviation -> personal support worker
    (
        ["PSW", "P.S.W.", "p s w"],
        "personal support worker",
    ),
    # CPR variants
    (
        ["CPR", "cardiopulmonary resuscitation", "CPR certified",
         "CPR Certification", "CPR-C"],
        "cpr",
    ),
    # WHMIS variants
    (
        ["WHMIS", "whmis 2015", "WHMIS Certified", "WHMIS Training"],
        "whmis",
    ),
    # Truck and coach variants
    (
        ["Truck & Coach", "truck and coach", "Truck and Coach Technician"],
        "truck and coach",
    ),
])
def test_aliases_collapse_to_same_canonical(variants, canonical):
    canonicals = {canonicalize_skill(v) for v in variants}
    assert canonicals == {canonical}, (
        f"variants {variants} produced {canonicals}, expected {{{canonical!r}}}"
    )


def test_canonicalize_passes_unknown_phrases_through_untouched():
    """No alias hit -> return _key form (no surprise mapping)."""
    assert canonicalize_skill("Python") == "python"
    assert canonicalize_skill("React.js") == "react js"
    assert canonicalize_skill("welding & fabrication") == "welding fabrication"


def test_canonicalize_empty_or_none():
    assert canonicalize_skill("") == ""
    assert canonicalize_skill(None) == ""


def test_alias_map_keys_are_already_in_key_form():
    """Every alias map key should pass _key() unchanged.

    If a contributor adds an entry like 'Class G Licence' (mixed case),
    canonicalize_skill('class g licence') wouldn't find it because the
    lookup key is normalized. This test catches that class of bug.
    """
    offenders = [k for k in SKILL_ALIASES if _key(k) != k]
    assert not offenders, f"alias keys not in normalized form: {offenders}"


def test_alias_map_values_are_canonical_strings():
    """Every canonical value should also normalize to itself.

    Otherwise canonicalize_skill on the canonical form would NOT return
    the same string a variant returns -- breaking the equality check
    callers depend on.
    """
    offenders = [
        (k, v) for k, v in SKILL_ALIASES.items() if _key(v) != v
    ]
    assert not offenders, f"canonical values not in normalized form: {offenders}"
