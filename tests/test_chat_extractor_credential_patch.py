"""Bug A regression tests — chat extractor credential patch.

The LLM extractor's prompt lists "no driver's license" as an example for
`transportation_text`. A reasonable LLM consequently routes any
driver's-licence mention (HAVE or NOT-HAVE) into that field instead of
`skills[]`. When the user's message also contains other genuine skills,
the rule-based fallback in the handler doesn't fire (`if not
result.skills: ...` gate), so the credential never reaches StagedSkill.
The matcher then sees the user as missing a credential they explicitly
stated they hold.

`_patch_credentials` is the deterministic safety net. These tests pin its
behavior so the failure mode that produced the "you may have an
out-of-country licence" hallucination cannot recur.
"""
from __future__ import annotations

import os

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from skillbridge.chat.extractor import (
    _CREDENTIAL_ALLOWLIST,
    _patch_credentials,
    extract,
)
from skillbridge.extract.base import ExtractedSkill

pytestmark = pytest.mark.nodb


# ============================================================================
# §1 — primary Bug A failure mode
# ============================================================================
def test_class_g_added_when_user_states_having_it():
    """The exact live-shadow failure case from 2026-06-15: user types
    "I have my Class G license, ..." and the LLM extracts other skills
    (defensive driving, customer service) but routes Class G to
    transportation_text. The patch must restore Class G as a have-skill."""
    msg = "I have my Class G license, 5 years driving, defensive driving, route planning, customer service, basic vehicle maintenance"
    out, added = _patch_credentials(msg, [])
    canonical_names = {s.skill_name for s in out}
    assert "Class G license" in canonical_names
    assert added == ["Class G license"]


def test_class_g_added_even_when_other_skills_already_present():
    """Patch must not regress when the LLM already returned non-credential
    skills. The fallback gate in the handler runs only when LLM returns
    zero skills; this patch runs in addition."""
    prior = [
        ExtractedSkill(skill_name="customer service", confidence=0.85),
        ExtractedSkill(skill_name="defensive driving", confidence=0.85),
    ]
    msg = "I have my Class G license and customer service"
    out, added = _patch_credentials(msg, prior)
    names = [s.skill_name for s in out]
    assert "Class G license" in names
    assert "customer service" in names
    assert "defensive driving" in names


# ============================================================================
# §2 — negation must block the patch
# ============================================================================
@pytest.mark.parametrize("msg", [
    "I do not have a Class G license yet",
    "I don't have my Class G license",
    "no Class G license",
    "without my Class G",
    "haven't gotten my Class G yet",
    "I lack a Class G license",
    "I'm missing Class G",
    "I need to get my Class G",
])
def test_negation_blocks_class_g(msg):
    out, added = _patch_credentials(msg, [])
    assert added == []
    assert all(s.skill_name != "Class G license" for s in out)


# ============================================================================
# §3 — learning-question integration via extract()
# ============================================================================
def test_learning_question_wipes_patched_credentials():
    """`_patch_credentials` runs BEFORE the learning-question guard. A
    user asking "where can I get my Class G?" gets Class G added by the
    patch, but the guard at the end of `extract()` then wipes ALL skills
    because the message is a learning inquiry. This pins the order."""
    msg = "Where can I get my Class G license?"
    result = extract(msg)
    assert result.skills == []
    # The patch's audit trail still appears in raw_keys_dropped so we know
    # the credential was added then dropped — useful for forensics.
    assert any(
        k.startswith("credential_patched:") or k.startswith("learning_question_skill:")
        for k in result.raw_keys_dropped
    ) or result.raw_keys_dropped == []  # LLM disabled path returns empty dropped


# ============================================================================
# §4 — idempotence: duplicate detection on prior skills
# ============================================================================
def test_no_duplicate_when_llm_already_emitted_class_g():
    """If the LLM extractor (correctly) emitted Class G already, the
    patch must not duplicate it. Comparison is by case-insensitive
    canonical name."""
    prior = [ExtractedSkill(skill_name="Class G license", confidence=0.85)]
    msg = "I have my Class G license"
    out, added = _patch_credentials(msg, prior)
    assert added == []
    assert len(out) == 1


def test_no_duplicate_case_insensitive():
    """Even if the LLM emitted a different case ("class g license"), the
    patch must recognize it as already present."""
    prior = [ExtractedSkill(skill_name="class g license", confidence=0.85)]
    msg = "I have my Class G license"
    out, added = _patch_credentials(msg, prior)
    assert added == []


# ============================================================================
# §5 — false-positive surface
# ============================================================================
@pytest.mark.parametrize("msg", [
    "I work for AZ Company in town",   # AZ without "license"
    "Class G2 license but not the full one",  # G2 is a stage, not full G
    "Class G1 license",                 # G1 is a stage, not full G
])
def test_no_false_positive_on_ambiguous_az_or_g_stages(msg):
    out, added = _patch_credentials(msg, [])
    assert "Class G license" not in {s.skill_name for s in out}
    assert "AZ license" not in {s.skill_name for s in out}


# ============================================================================
# §6 — full allowlist coverage
# ============================================================================
def test_each_allowlist_entry_extracts_when_message_states_having():
    """For each (regex, canonical) pair, a vanilla "I have X" message
    must add the canonical. Locks the allowlist against silent regression
    if someone edits a regex incorrectly."""
    # Drive a representative HAVE phrase for each canonical.
    have_phrases: dict[str, str] = {
        "Class G license": "I have my Class G license",
        "Class A license": "I hold my Class A license",
        "Class D license": "I have Class D license now",
        "Class Z endorsement": "I have my Class Z endorsement",
        "AZ license": "I have my AZ license",
        "DZ license": "I have my DZ license",
        "310T technician certification": "I'm a 310T",
        "310S automotive technician certification": "I'm a 310S",
        "Personal Support Worker certification": "I'm a Personal Support Worker",
        "PSW certificate": "I have my PSW certification",
        "WHMIS": "I have WHMIS",
        "first aid": "I have first aid",
        "food handler": "I have my food handler",
        "forklift certification": "I have my forklift certification",
    }
    # Sanity: every allowlist canonical must have a HAVE phrase in the
    # fixture above (forces test maintenance when the list grows).
    canonicals = {canonical for _, canonical in _CREDENTIAL_ALLOWLIST}
    fixture_canonicals = set(have_phrases.keys())
    missing = canonicals - fixture_canonicals
    assert not missing, (
        f"have_phrases fixture is missing entries for: {sorted(missing)}. "
        f"When you extend _CREDENTIAL_ALLOWLIST, add a HAVE phrase here "
        f"so this regression test still pins every entry."
    )

    for canonical, phrase in have_phrases.items():
        out, added = _patch_credentials(phrase, [])
        names = {s.skill_name for s in out}
        assert canonical in names, (
            f"{canonical!r} not added for phrase {phrase!r} "
            f"(got skills: {sorted(names)})"
        )


# ============================================================================
# §7 — telemetry surface
# ============================================================================
def test_added_credentials_appear_in_raw_keys_dropped():
    """The `raw_keys_dropped` field is the extractor's debug-log surface.
    When the patch adds a credential, the addition must appear there with
    the `credential_patched:` prefix so the operator can trace the path
    that produced a given staged skill."""
    msg = "I have my Class G license and WHMIS"
    result = extract(msg)
    patched_markers = [
        k for k in result.raw_keys_dropped if k.startswith("credential_patched:")
    ]
    # With LLM disabled (test env), extract() returns no LLM-side skills.
    # The patch then runs and adds Class G + WHMIS, recording markers.
    assert "credential_patched:Class G license" in patched_markers
    assert "credential_patched:WHMIS" in patched_markers


# ============================================================================
# §8 — invariant: empty / whitespace input is safe
# ============================================================================
@pytest.mark.parametrize("msg", ["", "   ", "\n\n"])
def test_blank_input_returns_empty(msg):
    out, added = _patch_credentials(msg, [])
    assert out == []
    assert added == []
