"""Grounding-boundary typographic normalization tests.

Pins the contract that BOTH grounding boundaries -- resume extract and
chat extractor -- share a single `_normalize_for_grounding` helper so a
PDF whose text contains smart quotes (U+2019), en/em dashes (U+2013/14),
or non-breaking spaces is still considered grounded against an LLM
evidence string that emitted the ASCII equivalents (or vice versa).

The contract explicitly does NOT loosen to fuzzy / paraphrase match:
dropping an apostrophe-s ("Class G driver license" missing the
possessive) still fails grounding. Typographic equivalence only.
"""
from __future__ import annotations

import os

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from skillbridge.chat.extractor import _is_grounded as chat_is_grounded
from skillbridge.resume.extract import (
    _is_grounded as resume_is_grounded,
    _normalize_for_grounding,
)

pytestmark = pytest.mark.nodb


# ============================================================================
# _normalize_for_grounding -- character-class folding
# ============================================================================
@pytest.mark.parametrize("variant", [
    "'",         # U+0027 ASCII
    "‘",   # left single quotation mark
    "’",   # right single quotation mark (the meeting_01 PDF case)
    "‚",   # single low-9
    "‛",   # single high-reversed-9
    "ʼ",   # modifier letter apostrophe
    "ʻ",   # modifier letter turned comma
])
def test_normalize_folds_single_quote_class_to_ascii(variant):
    """Every single-quote-class codepoint must fold to ASCII U+0027."""
    out = _normalize_for_grounding(f"driver{variant}s license")
    assert out == "driver's license"


@pytest.mark.parametrize("variant", [
    '"',         # U+0022 ASCII
    "“",   # left double quotation mark
    "”",   # right double quotation mark
    "„",   # double low-9
    "‟",   # double high-reversed-9
])
def test_normalize_folds_double_quote_class_to_ascii(variant):
    out = _normalize_for_grounding(f"the {variant}fast{variant} way")
    assert out == 'the "fast" way'


@pytest.mark.parametrize("variant", [
    "-",         # U+002D ASCII hyphen-minus
    "‐",   # hyphen
    "‑",   # non-breaking hyphen
    "‒",   # figure dash
    "–",   # en dash
    "—",   # em dash
    "―",   # horizontal bar
    "−",   # minus sign
])
def test_normalize_folds_dash_class_to_hyphen(variant):
    out = _normalize_for_grounding(f"2019{variant}2021")
    assert out == "2019-2021"


@pytest.mark.parametrize("space", [
    " ",         # ASCII space
    " ",   # non-breaking space
    " ",   # figure space
    " ",   # thin space
    " ",   # hair space
    " ",   # narrow no-break space
])
def test_normalize_folds_space_class_to_ascii_space(space):
    out = _normalize_for_grounding(f"Class{space}G")
    assert out == "Class G"


def test_normalize_collapses_whitespace_runs():
    """After folding, consecutive whitespace collapses to one space."""
    out = _normalize_for_grounding("a    \tb")
    assert out == "a b"


def test_normalize_trims_leading_trailing_whitespace():
    assert _normalize_for_grounding("   foo   ") == "foo"


def test_normalize_handles_non_string():
    """Defensive: a non-string input returns the empty string."""
    assert _normalize_for_grounding(None) == ""           # type: ignore[arg-type]
    assert _normalize_for_grounding(42) == ""             # type: ignore[arg-type]


def test_normalize_does_not_remove_punctuation():
    """Typographic folding only. Apostrophes / hyphens / commas are
    preserved -- so paraphrases that drop them still fail grounding."""
    out = _normalize_for_grounding("Class G driver's license")
    # apostrophe survives
    assert "'" in out
    # not stripped to "Class G driver license"
    assert out != "class g driver license"


# ============================================================================
# resume extract -- the meeting_01 reproduction
# ============================================================================
def _load_meeting_01_pdf_text() -> str:
    """The shipped PDF fixture is the actual repro: PDF text contains
    U+2019 at every `driver's` position; the LLM emits ASCII apostrophes
    in its evidence JSON. Pre-fix, the substring check rejected the
    Class G + G2/G certifications because of this encoding mismatch."""
    from skillbridge.resume.parse import parse_resume
    with open("docs/test-resumes/meeting_01_310s_automotive_perfect.pdf", "rb") as f:
        return (parse_resume(f.read(), "meeting_01.pdf").text or "").lower()


def test_pdf_fixture_actually_contains_smart_quote_at_driver_s():
    """Lock the precondition: if the PDF stops carrying U+2019 in some
    future re-render, this test fails LOUDLY so we know the fixture
    drifted (and our regression has nothing to defend against)."""
    text = _load_meeting_01_pdf_text()
    assert "’" in text, (
        "Expected the PDF fixture to contain at least one curly "
        "apostrophe (U+2019). The original meeting_01 bug depended on "
        "that encoding; if the fixture changed, re-verify the repro."
    )


def test_resume_grounding_passes_ascii_evidence_against_pdf_smart_quote_text():
    """The Class G certification: LLM emits ASCII apostrophe evidence,
    PDF text carries the smart quote. After round-27 normalization,
    grounding passes."""
    text = _load_meeting_01_pdf_text()
    ascii_evidence = "Valid Ontario Class G driver's license"
    assert resume_is_grounded(ascii_evidence, text)


def test_resume_grounding_passes_smart_quote_evidence_against_ascii_text():
    """Reverse direction: a resume typed in a plain editor (ASCII
    apostrophes) and an LLM that decided to use smart quotes in its
    evidence string. Same fold should apply."""
    text = "valid ontario class g driver's license with a clean record"
    smart_evidence = "Class G driver’s license"
    assert resume_is_grounded(smart_evidence, text)


def test_resume_grounding_passes_g2_g_with_smart_quote_pdf_text():
    text = _load_meeting_01_pdf_text()
    ascii_evidence = "Valid G2/G driver's license"
    assert resume_is_grounded(ascii_evidence, text)


def test_resume_grounding_passes_whmis_verbatim():
    """A short ASCII fact still grounds normally -- the existing
    >=4-char floor and exact substring contract are unchanged."""
    text = _load_meeting_01_pdf_text()
    assert resume_is_grounded("WHMIS 2015", text)


# ----- negation: apostrophe-stripped paraphrase MUST STILL FAIL -----
def test_resume_grounding_rejects_apostrophe_stripped_paraphrase():
    """The evidence contract is preserved: dropping the possessive
    `'s` is a paraphrase, not a typographic variant. Reject it."""
    text = _load_meeting_01_pdf_text()
    assert not resume_is_grounded("Class G driver license", text)


def test_resume_grounding_rejects_word_dropped_paraphrase():
    text = _load_meeting_01_pdf_text()
    assert not resume_is_grounded("Class G license", text)


def test_resume_grounding_rejects_substituted_word_paraphrase():
    """`Class G drivers permit` swaps `license` for `permit` -- the
    word-substitution paraphrase risks were called out in the design;
    normalization MUST NOT mask them."""
    text = _load_meeting_01_pdf_text()
    assert not resume_is_grounded("Class G driver's permit", text)


def test_resume_grounding_rejects_short_evidence_below_floor():
    """The >=4-char floor is unchanged."""
    text = _load_meeting_01_pdf_text()
    assert not resume_is_grounded("WH", text)
    assert not resume_is_grounded("a's", text)   # normalizes to "a's", 3 chars


# ============================================================================
# Chat extractor -- identical normalization, identical contract
# ============================================================================
def test_chat_grounding_passes_smart_quote_evidence_against_ascii_message():
    msg = "i have my class g driver's license"
    smart_ev = "Class G driver’s license"
    assert chat_is_grounded(smart_ev, msg)


def test_chat_grounding_passes_ascii_evidence_against_smart_quote_message():
    """A user message pasted from a Word doc might contain smart quotes;
    the LLM's evidence will be ASCII. Same direction reversed."""
    msg = "i have my class g driver’s license"
    ascii_ev = "Class G driver's license"
    assert chat_is_grounded(ascii_ev, msg)


def test_chat_grounding_rejects_apostrophe_stripped_paraphrase():
    msg = "i have my class g driver's license"
    assert not chat_is_grounded("Class G driver license", msg)


def test_chat_grounding_handles_dashes_and_spaces():
    """En-dash in message, ASCII hyphen in evidence (or vice versa)."""
    msg = "shifts 2019–2021"            # en-dash
    ascii_ev = "2019-2021"
    assert chat_is_grounded(ascii_ev, msg)


def test_chat_grounding_preserves_short_token_slot_escape():
    """Pre-existing behavior: closed-vocab slots ("ft", "pt", "day")
    pass the >=4-char floor via slot-aware whole-word match. The
    normalization MUST NOT break this."""
    msg = "i want ft"
    assert chat_is_grounded("ft", msg, slot="work_type_preference")
    # Same slot, no whole-word match -> still rejected
    assert not chat_is_grounded("ft", "i have a left foot")


# ============================================================================
# _validate_and_normalize end-to-end -- the round-28 cert-name defect
# ============================================================================
# The full extraction path was deleting every certification and project
# name because the nested-entry pass treated structural `name` fields
# as person-identifier PII. Even with typography normalization passing
# grounding, the LLM's `{"name": "Class G", "evidence": "..."}`
# certification arrived downstream as `{"evidence": ...}` with no name.
# Tests below pin: structural names survive; PII keys NOT declared in
# a group's schema (full_name, email, phone, address, social) still drop.
# ============================================================================
def test_certification_name_survives_normalize():
    """Critical regression: a certification with name + evidence MUST
    keep its name. Pre-fix the name was dropped as
    `certifications_contact_info:name`."""
    from skillbridge.resume.extract import _validate_and_normalize
    text = "Valid Ontario Class G driver's license"
    payload = {
        "skills": [], "work_history": [], "education": [],
        "certifications": [
            {"name": "Class G driver's license",
             "issuer": "MTO",
             "evidence": "Class G driver's license"},
        ],
        "projects": [],
        "languages": [],
    }
    facts, dropped = _validate_and_normalize(payload, text)
    assert len(facts["certifications"]) == 1
    cert = facts["certifications"][0]
    assert cert["name"] == "Class G driver's license"
    assert cert["issuer"] == "MTO"
    # MUST NOT be dropped as contact info
    assert not any("contact_info:name" in d for d in dropped)


def test_pdf_fixture_certifications_have_names_after_normalize():
    """End-to-end on the actual PDF fixture: a payload shaped like a
    competent LLM's output produces certification entries with
    `name` populated AND grounded against the PDF's smart-quote text."""
    from skillbridge.resume.extract import _validate_and_normalize
    from skillbridge.resume.parse import parse_resume
    with open("docs/test-resumes/meeting_01_310s_automotive_perfect.pdf", "rb") as f:
        text = (parse_resume(f.read(), "x.pdf").text or "")
    # The LLM's evidence uses ASCII apostrophes; the PDF carries U+2019.
    # Round-27 typography fix + round-28 cert-name fix together must
    # let BOTH certifications survive with their names intact.
    payload = {
        "skills": [], "work_history": [], "education": [],
        "certifications": [
            {"name": "Class G driver's license",
             "evidence": "Valid Ontario Class G driver's license"},
            {"name": "G2/G driver's license",
             "evidence": "Valid G2/G driver's license"},
            {"name": "310S Automotive Service Technician License",
             "evidence": "Valid Ontario 310S Automotive Service Technician License"},
            {"name": "WHMIS 2015", "evidence": "WHMIS 2015"},
        ],
        "projects": [],
        "languages": [],
    }
    facts, dropped = _validate_and_normalize(payload, text)
    cert_names = [c.get("name") for c in facts["certifications"]]
    assert "Class G driver's license" in cert_names, (
        f"Class G dropped; this is the headline meeting_01 bug. "
        f"got cert_names={cert_names}, dropped={dropped}"
    )
    assert "G2/G driver's license" in cert_names
    assert "310S Automotive Service Technician License" in cert_names
    assert "WHMIS 2015" in cert_names
    # All four certifications survive grounding + nested-name pass
    assert len(facts["certifications"]) == 4


def test_project_name_survives_normalize():
    """Same defect in projects -- `name` is a structural schema field
    (line 233 in extract.py)."""
    from skillbridge.resume.extract import _validate_and_normalize
    text = "Built a dashboard refresh project for the team."
    payload = {
        "skills": [], "work_history": [], "education": [],
        "certifications": [],
        "projects": [
            {"name": "dashboard refresh",
             "summary": "new charts",
             "evidence": "dashboard refresh project"},
        ],
        "languages": [],
    }
    facts, _ = _validate_and_normalize(payload, text)
    assert len(facts["projects"]) == 1
    assert facts["projects"][0]["name"] == "dashboard refresh"


# ---- defense: PII keys NOT declared in the group's schema still drop ----
def test_certification_full_name_still_dropped_as_pii():
    """A nested `full_name` is NEVER schema. The schema-override rule
    only protects structural keys; non-schema PII keys remain dropped."""
    from skillbridge.resume.extract import _validate_and_normalize
    text = "Valid Ontario 310S Automotive Service Technician License"
    payload = {
        "skills": [], "work_history": [], "education": [],
        "certifications": [
            {"name": "310S license",
             "full_name": "John Doe",
             "email": "j@example.ca",
             "phone": "555-1234",
             "evidence": "310S Automotive Service Technician License"},
        ],
        "projects": [],
        "languages": [],
    }
    facts, dropped = _validate_and_normalize(payload, text)
    cert = facts["certifications"][0]
    # Structural name kept
    assert cert["name"] == "310S license"
    # PII NOT kept
    assert "full_name" not in cert
    assert "email" not in cert
    assert "phone" not in cert
    # And the drops are recorded
    assert "certifications_contact_info:full_name" in dropped
    assert "certifications_contact_info:email" in dropped
    assert "certifications_contact_info:phone" in dropped


def test_work_history_name_still_dropped_because_not_in_schema():
    """work_history's schema is (title, employer, summary) -- it does
    NOT declare `name`. So a `name` field there IS PII (the person)
    and continues to drop."""
    from skillbridge.resume.extract import _validate_and_normalize
    text = "Engineer at Acme from 2020 to 2022"
    payload = {
        "skills": [], "work_history": [
            {"title": "Engineer", "employer": "Acme",
             "name": "John Doe",
             "evidence": "Engineer at Acme", "start_year": 2020, "end_year": 2022},
        ],
        "education": [], "certifications": [], "projects": [],
        "languages": [],
    }
    facts, dropped = _validate_and_normalize(payload, text)
    work = facts["work_history"][0]
    assert work["title"] == "Engineer"
    assert work["employer"] == "Acme"
    assert "name" not in work
    assert "work_history_contact_info:name" in dropped


def test_top_level_name_key_still_dropped_as_pii():
    """The top-level pass at line 263 is unchanged: a payload-level
    `name`, `email`, etc. is the person's contact info."""
    from skillbridge.resume.extract import _validate_and_normalize
    text = "Some content."
    payload = {
        "skills": [], "work_history": [], "education": [],
        "certifications": [], "projects": [], "languages": [],
        # Top-level contact keys -- the original threat model
        "name": "John Doe", "email": "j@example.ca", "phone": "555-1234",
    }
    _, dropped = _validate_and_normalize(payload, text)
    assert "top_contact_info:name" in dropped
    assert "top_contact_info:email" in dropped
    assert "top_contact_info:phone" in dropped


# ============================================================================
# Both boundaries share the same helper
# ============================================================================
def test_both_extractors_import_the_same_normalizer():
    """Round-27 contract: the chat extractor imports the resume
    extractor's normalizer to keep the two grounding boundaries from
    drifting. A test that loaded them separately would mask drift."""
    from skillbridge.chat import extractor as chat_extractor_module
    from skillbridge.resume import extract as resume_extract_module
    # Both modules use _normalize_for_grounding from resume.extract.
    # The chat extractor's _is_grounded does a local import; pin the
    # source-of-truth identity by calling it on a known case and
    # checking the result matches the resume helper directly.
    assert chat_extractor_module._is_grounded(
        "Class G driver’s license",
        "class g driver's license",
    ) is True
    assert resume_extract_module._is_grounded(
        "Class G driver’s license",
        "class g driver's license",
    ) is True
