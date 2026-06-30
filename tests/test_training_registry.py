"""Tests for the training registry loader + lookup + freshness logic.

Three concerns:

  1. The shipped data/training_registry.yaml LOADS without error.
     This is the smoke test the lead engineer specifically asked for:
     the v1 seed (all entries pending) must be structurally valid.

  2. URL-suppression safety net: when verified_at is null OR > 6 months
     old, surface_url() returns None. The shipped YAML, with every
     non-referral entry pending, must produce zero surfaced URLs.

  3. Each hard-validation rule fires on bad input. The loader fails
     LOUD on structural problems rather than silently shipping a
     corrupt registry.

No DB, no LLM, no chat. Pure file + Python.
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from skillbridge.training.models import (
    DEFAULT_FRESHNESS_DAYS,
    Gap,
    Resource,
    normalize_gap_name,
)
from skillbridge.training.registry import (
    RegistryValidationError,
    TrainingRegistry,
)

pytestmark = pytest.mark.nodb


# =========================================================================
# 1. The shipped YAML loads + safety net works end-to-end
# =========================================================================
def test_shipped_yaml_loads_without_error():
    """The v1 seed YAML at data/training_registry.yaml is structurally
    valid. This catches schema regressions before they ship -- if a
    contributor adds an entry that violates the schema, this test
    fails at PR time."""
    registry = TrainingRegistry.from_yaml()
    assert registry.version == 1
    assert len(registry.gaps) >= 13, (
        f"Seed YAML should have at least 13 priority gaps; got "
        f"{len(registry.gaps)}"
    )


def test_shipped_yaml_has_310T_gap():
    """The flagship gap from Michael's CV scenario."""
    registry = TrainingRegistry.from_yaml()
    gap = registry.lookup("310T technician certification")
    assert gap is not None
    assert any(r.type == "credential_pathway" for r in gap.resources)


def test_shipped_yaml_safety_net_only_authorized_urls_surface():
    """SAFETY-NET REGRESSION: every URL that surfaces from the shipped
    YAML must come from a deliberately-authorized verified entry. The
    runtime contract is that pending URLs are SUPPRESSED; the allowlist
    below records the verifications the lead engineer has explicitly
    signed off on.

    Adding a new entry requires (a) setting verified_at + verified_by
    in the YAML and (b) appending the (canonical_name, url) pair here.
    Both steps are deliberate audit-trail claims that the URL has been
    checked and points where the registry summary says it does.

    Historical note: this test originally asserted ZERO surfaced URLs
    while the v1 seed shipped fully pending. After CP4 shadow Round 4
    (2026-06-15) the lead engineer authorized verification of the
    DriveTest Class G entry so the live accounting shadow could
    confirm has_verified_training=True flowed correctly through CP4."""
    registry = TrainingRegistry.from_yaml()
    today = date.today()

    AUTHORIZED_SURFACED: set[tuple[str, str]] = {
        ("310S automotive technician certification", "https://www.saultcollege.ca/programs/apprenticeship/automotive-service-technician"),
        ("310S automotive technician certification", "https://www.skilledtradesontario.ca/trade-information/automotive-service-technician/"),
        ("310T technician certification", "https://www.saultcollege.ca/programs/apprenticeship/truck-and-coach-technician"),
        ("310T technician certification", "https://www.skilledtradesontario.ca/trade-information/truck-and-coach-technician/"),
        ("Class G driver's license", "https://drivetest.ca/"),
        ("Class G driver's license", "https://www.ontario.ca/page/get-g-drivers-licence-new-drivers"),
        ("CompTIA IT support certification", "https://grow.google/certificates/it-support/"),
        ("CompTIA IT support certification", "https://www.comptia.org/certifications/a"),
        ("Microsoft Excel", "https://learn.microsoft.com/en-us/training/browse/?products=office"),
        ("Microsoft Excel", "https://www.coursera.org/specializations/excel"),
        ("Microsoft Office", "https://learn.microsoft.com/en-us/training/browse/?products=office"),
        ("Ontario nursing registration", "https://www.cno.org/"),
        ("Ontario security guard licence", "https://www.ontario.ca/page/get-security-guard-or-private-investigator-licence"),
        ("Personal Support Worker certification", "https://www.ontariocolleges.ca/en/programs?q=personal%20support%20worker"),
        ("QuickBooks and basic accounting", "https://www.saultcollege.ca/programs/business/business"),
        ("Registered Early Childhood Educator registration", "https://www.college-ece.ca/"),
        ("Registered Early Childhood Educator registration", "https://www.saultcollege.ca/programs/community-services/early-childhood-education"),
        ("Smart Serve certification", "https://smartserve.ca/"),
        ("WHMIS", "https://www.ccohs.ca/products/courses/whmis_globally/"),
        ("Working at Heights training", "https://www.ontario.ca/page/training-working-heights"),
        ("commercial driver's license", "https://www.ontario.ca/page/get-truck-drivers-licence"),
        ("customer service", "https://www.coursera.org/learn/customer-service"),
        ("digital literacy", "https://learn.microsoft.com/en-us/training/browse/?products=office"),
        ("first aid and CPR", "https://sja.ca/"),
        ("food handler certification", "https://www.algomapublichealth.com/"),
        ("food handler certification", "https://www.traincan.com/"),
        ("forklift certification", "https://www.ccohs.ca/products/courses/forklifts/"),
        ("medical terminology", "https://www.coursera.org/learn/clinical-terminology"),
        ("payroll compliance training", "https://payroll.ca/"),
        ("records management", "https://www.saultcollege.ca/programs/business/business"),
        ("vulnerable sector check", "https://saultpolice.ca/"),
        ("written communication", "https://www.coursera.org/learn/business-writing"),
        ("worker health and safety awareness", "https://www.ontario.ca/page/worker-health-and-safety-awareness-four-steps"),
    }

    surfaced: set[tuple[str, str]] = set()
    for gap in registry.gaps:
        for resource in gap.resources:
            url = resource.surface_url(today)
            if url is not None:
                surfaced.add((gap.canonical_name, url))

    unauthorized = surfaced - AUTHORIZED_SURFACED
    missing = AUTHORIZED_SURFACED - surfaced

    assert not unauthorized, (
        f"Unauthorized URL is surfacing from the shipped YAML: "
        f"{sorted(unauthorized)}. If you've intentionally verified a "
        f"new entry, add it to AUTHORIZED_SURFACED in this test -- "
        f"DO NOT silence the assertion without reviewing what changed."
    )
    assert not missing, (
        f"Expected verified URL no longer surfaces: {sorted(missing)}. "
        f"Either a verified_at was reverted, the URL was edited, or "
        f"the 180-day freshness window elapsed. Update intentionally."
    )


def test_shipped_yaml_referral_only_entries_remain_visible():
    """Companion to the safety-net test: referral_only resources (SCCC
    entries, which have no URL by design) should still come through as
    Resource objects, just without a URL. The responder uses these as
    the fallback when no surfaced URL is available."""
    registry = TrainingRegistry.from_yaml()
    today = date.today()

    referral_only_count = 0
    for gap in registry.gaps:
        for resource in gap.resources:
            if resource.type == "referral_only":
                referral_only_count += 1
                # surface_url returns None for referral_only by design
                assert resource.surface_url(today) is None
                assert resource.url is None
                assert resource.provider.strip()
                assert resource.summary.strip()

    # Seed has multiple SCCC referrals across gaps; expect at least one
    assert referral_only_count >= 1, (
        "Seed YAML has no referral_only resources. The architecture "
        "depends on referral_only as the safe fallback when verified "
        "URLs aren't available."
    )


# =========================================================================
# 2. Freshness logic on Resource
# =========================================================================
def _resource(
    *,
    type_: str = "credential_pathway",
    url: str | None = "https://example.com/cert",
    verified_at: date | None = date(2026, 6, 1),
    verified_by: str | None = "test",
) -> Resource:
    return Resource(
        provider="Test Provider",
        type=type_,
        url=url,
        summary="A test resource",
        verified_at=verified_at,
        verified_by=verified_by,
    )


def test_resource_is_pending_when_verified_at_is_none():
    r = _resource(verified_at=None, verified_by=None)
    assert r.is_pending
    assert r.surface_url(date(2026, 6, 4)) is None


def test_resource_is_fresh_within_window():
    today = date(2026, 6, 4)
    r = _resource(verified_at=today)
    assert r.is_fresh(today)
    assert r.surface_url(today) == "https://example.com/cert"


def test_resource_at_exact_freshness_boundary_is_fresh():
    """A resource verified exactly 180 days ago is still fresh
    (boundary check; off-by-one would be confusing for reviewers)."""
    today = date(2026, 6, 4)
    boundary = today - timedelta(days=DEFAULT_FRESHNESS_DAYS)
    r = _resource(verified_at=boundary)
    assert r.is_fresh(today)
    assert r.surface_url(today) == "https://example.com/cert"


def test_resource_just_past_freshness_is_stale():
    today = date(2026, 6, 4)
    just_past = today - timedelta(days=DEFAULT_FRESHNESS_DAYS + 1)
    r = _resource(verified_at=just_past)
    assert not r.is_fresh(today)
    assert r.surface_url(today) is None


def test_resource_referral_only_never_surfaces_url():
    """Even if (somehow) a referral_only resource had verified_at set,
    surface_url returns None. The type IS the contract."""
    r = _resource(type_="referral_only", url=None, verified_at=date.today())
    assert r.surface_url(date.today()) is None


def test_resource_fresh_resource_with_future_verified_at_is_handled():
    """Edge case: a contributor sets verified_at to tomorrow by
    accident. Treat as fresh (date wrapping), not stale."""
    today = date(2026, 6, 4)
    tomorrow = today + timedelta(days=1)
    r = _resource(verified_at=tomorrow)
    # We use abs check: 0 <= age_days <= max means fresh
    # tomorrow's age = -1 -> not in range -> not fresh
    # This is intentional; future-dated entries are "pending" until the date.
    assert not r.is_fresh(today)


# =========================================================================
# 3. Lookup -- canonical_name + aliases + normalization
# =========================================================================
def _minimal_registry(gap_canonical: str, aliases: list[str]) -> TrainingRegistry:
    """Build a small registry with one gap for lookup tests."""
    return TrainingRegistry.from_dict({
        "version": 1,
        "registry_verified_at": None,
        "gaps": [{
            "canonical_name": gap_canonical,
            "aliases": aliases,
            "category": "credential",
            "description": "Test gap.",
            "resources": [{
                "provider": "Test Provider",
                "type": "referral_only",
                "url": None,
                "summary": "Test resource.",
                "verified_at": None,
                "verified_by": None,
            }],
        }],
    })


def test_lookup_matches_canonical_name():
    r = _minimal_registry("310T technician certification", ["310T"])
    assert r.lookup("310T technician certification") is not None


def test_lookup_matches_alias():
    r = _minimal_registry("310T technician certification", ["310T", "310T cert"])
    assert r.lookup("310T") is not None
    assert r.lookup("310T cert") is not None


def test_lookup_is_case_insensitive():
    r = _minimal_registry("310T technician certification", ["310T"])
    assert r.lookup("310t TECHNICIAN certification") is not None


def test_lookup_normalizes_whitespace():
    r = _minimal_registry("310T technician certification", ["310T"])
    assert r.lookup("  310T   technician    certification  ") is not None


def test_lookup_strips_trailing_punctuation():
    r = _minimal_registry("310T technician certification", ["310T"])
    assert r.lookup("310T technician certification.") is not None
    assert r.lookup("310T?") is not None


def test_lookup_unknown_returns_none():
    r = _minimal_registry("310T technician certification", ["310T"])
    assert r.lookup("welding") is None
    assert r.lookup("") is None
    assert r.lookup("   ") is None


# ===========================================================================
# Cold-session gap discovery from message text
# ===========================================================================
# Without this, training questions from users who haven't seen matches
# yet (no last_presented_credential_gaps) silently get an empty
# TRAINING block and the LLM improvises providers it knows. With it,
# direct questions like "how do I get my Class G?" surface the
# registry's Class G resources even on a cold session.
def test_find_gaps_in_message_matches_310T():
    registry = TrainingRegistry.from_yaml()
    gaps = registry.find_gaps_in_message("how do I get my 310T?")
    assert len(gaps) >= 1
    assert any(g.canonical_name == "310T technician certification" for g in gaps)


def test_find_gaps_in_message_matches_class_g():
    registry = TrainingRegistry.from_yaml()
    gaps = registry.find_gaps_in_message(
        "how can I get my Class G driver's licence?",
    )
    assert any(g.canonical_name == "Class G driver's license" for g in gaps)


# Slice (2026-06-08): graduated-licence shorthand. The live test of
# 2026-06-08 surfaced "how can get this license G2/G." which didn't match
# any Class G alias, so the message-scan returned [], the router fell
# through to Rule 3, and the responder asked the user to re-list skills
# they'd already provided. Adding G1/G2/G2/G aliases routes these
# phrasings back to the same canonical Class G gap.
@pytest.mark.parametrize("phrase", [
    "how can get this license G2/G.",
    "I need my G2",
    "What's the path for G2/G licence?",
    "G1 then G2 then G",
    "Where do I take the G1?",
    "I'm working on my G2/G driver's license",
])
def test_find_gaps_in_message_matches_class_g_via_graduated_aliases(phrase):
    """Every G1 / G2 / G2/G shorthand routes back to the canonical
    Class G entry. Stops the router falling through to Rule 3 for
    Ontario candidates who naturally say "G2" instead of "Class G"."""
    registry = TrainingRegistry.from_yaml()
    gaps = registry.find_gaps_in_message(phrase)
    canonical_names = [g.canonical_name for g in gaps]
    assert "Class G driver's license" in canonical_names, (
        f"Phrase {phrase!r} did not resolve to Class G; got {canonical_names}"
    )


@pytest.mark.parametrize("alias", [
    "G2", "G2/G", "G2/G licence", "G2/G license",
    "G2/G driver's license", "G2/G driver's licence",
    "G2 licence", "G2 license",
    "G1", "G1 licence", "G1 license",
])
def test_lookup_class_g_via_graduated_aliases(alias):
    """Direct lookup (not message-scan) -- every new alias resolves."""
    registry = TrainingRegistry.from_yaml()
    hit = registry.lookup(alias)
    assert hit is not None, f"alias {alias!r} did not resolve"
    assert hit.canonical_name == "Class G driver's license"


def test_find_gaps_in_message_matches_whmis():
    registry = TrainingRegistry.from_yaml()
    gaps = registry.find_gaps_in_message("where can I take WHMIS?")
    assert any(g.canonical_name == "WHMIS" for g in gaps)


def test_find_gaps_in_message_matches_forklift():
    registry = TrainingRegistry.from_yaml()
    gaps = registry.find_gaps_in_message("forklift certificate please")
    assert any(g.canonical_name == "forklift certification" for g in gaps)


def test_find_gaps_in_message_matches_first_aid():
    registry = TrainingRegistry.from_yaml()
    gaps = registry.find_gaps_in_message("first aid course?")
    assert any(g.canonical_name == "first aid and CPR" for g in gaps)


def test_find_gaps_in_message_matches_cpr():
    registry = TrainingRegistry.from_yaml()
    gaps = registry.find_gaps_in_message("how do I get my CPR?")
    assert any(g.canonical_name == "first aid and CPR" for g in gaps)


def test_find_gaps_in_message_matches_excel():
    """User said Excel was a 'good' example in the design review."""
    registry = TrainingRegistry.from_yaml()
    gaps = registry.find_gaps_in_message("any Excel course you recommend?")
    assert any(g.canonical_name == "Microsoft Excel" for g in gaps)


def test_find_gaps_in_message_matches_food_handler():
    registry = TrainingRegistry.from_yaml()
    gaps = registry.find_gaps_in_message("food handler training")
    assert any(g.canonical_name == "food handler certification" for g in gaps)


# --- Negative cases: must NOT false-positive on common words ---
def test_find_gaps_in_message_does_not_match_bare_g():
    """The 'G' single-letter alias is risky; user mentioning the
    letter G casually must not trigger Class G."""
    registry = TrainingRegistry.from_yaml()
    gaps = registry.find_gaps_in_message("I gave him a G for a grade")
    assert not any(g.canonical_name == "Class G driver's license" for g in gaps)


def test_find_gaps_in_message_does_not_match_bare_word():
    """Microsoft Office's 'Word' alias is blocklisted to avoid
    matching every casual mention of the word 'word'."""
    registry = TrainingRegistry.from_yaml()
    gaps = registry.find_gaps_in_message("I gave my word to him")
    assert not any(g.canonical_name == "Microsoft Office" for g in gaps)


def test_find_gaps_in_message_does_not_match_bare_office():
    """Microsoft Office's 'Office' would false-positive on 'office'
    in everyday context."""
    registry = TrainingRegistry.from_yaml()
    gaps = registry.find_gaps_in_message("I'll be in the office tomorrow")
    assert not any(g.canonical_name == "Microsoft Office" for g in gaps)


def test_find_gaps_in_message_does_not_match_bare_service():
    """'service' alone must not match Customer Service. The full
    phrase 'customer service' does match (tested separately)."""
    registry = TrainingRegistry.from_yaml()
    gaps = registry.find_gaps_in_message("the service was good")
    assert not any(g.canonical_name == "customer service" for g in gaps)


def test_find_gaps_in_message_does_match_customer_service_full_phrase():
    """Companion to the negative case: full 'customer service' IS
    a valid match because the canonical name is multi-word."""
    registry = TrainingRegistry.from_yaml()
    gaps = registry.find_gaps_in_message(
        "any tips for improving my customer service skills?",
    )
    assert any(g.canonical_name == "customer service" for g in gaps)


def test_find_gaps_in_message_does_not_match_bare_safety():
    """'safety' is blocklisted to avoid noise on casual mentions
    (e.g. 'workplace safety is important')."""
    registry = TrainingRegistry.from_yaml()
    gaps = registry.find_gaps_in_message("workplace safety is important")
    # No registry gap has "safety" as a standalone alias, but check anyway
    # that nothing inappropriate matches
    safety_matches = [
        g for g in gaps if "safety" in g.canonical_name.lower()
    ]
    # WHMIS or first-aid don't have "safety" as canonical/alias either
    # so this returns no matches -- the test pins that absence.
    assert len(safety_matches) == 0


def test_find_gaps_in_message_does_not_match_AZ_alone():
    """Two-char abbreviation must not false-positive in random text
    ('AZ' alias for Class A is in the blocklist)."""
    registry = TrainingRegistry.from_yaml()
    gaps = registry.find_gaps_in_message("Arizona is hot in summer")
    az_matches = [g for g in gaps if "commercial" in g.canonical_name.lower()]
    assert len(az_matches) == 0


# --- Empty / edge cases ---
def test_find_gaps_in_message_empty_message_returns_empty():
    registry = TrainingRegistry.from_yaml()
    assert registry.find_gaps_in_message("") == []
    assert registry.find_gaps_in_message(None) == []  # defensive


def test_find_gaps_in_message_dedupes_when_multiple_aliases_hit():
    """A message mentioning both canonical and alias of the same gap
    returns the gap only ONCE."""
    registry = TrainingRegistry.from_yaml()
    gaps = registry.find_gaps_in_message(
        "I need 310T and a truck and coach technician certificate",
    )
    matching = [g for g in gaps if g.canonical_name == "310T technician certification"]
    assert len(matching) == 1


def test_find_gaps_in_message_caps_at_max_results():
    registry = TrainingRegistry.from_yaml()
    # A message that mentions many gaps; max_results caps the return
    msg = "tell me about 310T and Class G and WHMIS and forklift and CPR and Excel"
    gaps = registry.find_gaps_in_message(msg, max_results=3)
    assert len(gaps) <= 3


def test_surface_resources_logs_unknown_gap(caplog):
    """The unknown-gap log line is the telemetry hook for growing the
    registry from real usage. INFO level."""
    import logging
    r = _minimal_registry("310T technician certification", ["310T"])
    caplog.set_level(logging.INFO, logger="skillbridge.training.registry")
    resources = r.surface_resources("welding", today=date.today())
    assert resources == []
    assert any(
        "unknown_gap" in record.message and "welding" in record.message
        for record in caplog.records
    )


def test_surface_resources_returns_known_gap_resources():
    r = _minimal_registry("310T technician certification", ["310T"])
    resources = r.surface_resources("310T", today=date.today())
    assert len(resources) == 1
    assert resources[0].provider == "Test Provider"


def test_surface_resources_respects_limit():
    """Multiple resources returned, capped by `limit`."""
    raw = {
        "version": 1,
        "gaps": [{
            "canonical_name": "g",
            "aliases": ["g"],
            "category": "skill",
            "description": "d",
            "resources": [
                {"provider": f"p{i}", "type": "referral_only",
                 "url": None, "summary": "s",
                 "verified_at": None, "verified_by": None}
                for i in range(5)
            ],
        }],
    }
    r = TrainingRegistry.from_dict(raw)
    assert len(r.surface_resources("g", today=date.today(), limit=2)) == 2


# =========================================================================
# 4. normalize_gap_name helper
# =========================================================================
@pytest.mark.parametrize("a,b", [
    ("310T", "310t"),
    ("310T", " 310T "),
    ("310T", "310T."),
    ("310T technician  certification", "310T technician certification"),
    ("Class G driver's license", "class g driver's license"),
])
def test_normalize_gap_name_collapses_variants(a, b):
    assert normalize_gap_name(a) == normalize_gap_name(b)


# =========================================================================
# 5. Hard validation rules -- each one fires on bad input
# =========================================================================
_VALID_GAP = {
    "canonical_name": "test gap",
    "aliases": ["test"],
    "category": "skill",
    "description": "test desc",
    "resources": [{
        "provider": "p",
        "type": "referral_only",
        "url": None,
        "summary": "s",
        "verified_at": None,
        "verified_by": None,
    }],
}


def _valid_registry_dict(overrides: dict | None = None) -> dict:
    base = {
        "version": 1,
        "registry_verified_at": None,
        "gaps": [{**_VALID_GAP}],
    }
    if overrides:
        base.update(overrides)
    return base


def test_validation_top_level_must_be_dict():
    with pytest.raises(RegistryValidationError, match="top-level"):
        TrainingRegistry.from_dict([])  # type: ignore[arg-type]


def test_validation_version_must_be_1():
    with pytest.raises(RegistryValidationError, match="version"):
        TrainingRegistry.from_dict(_valid_registry_dict({"version": 2}))


def test_validation_missing_version():
    with pytest.raises(RegistryValidationError, match="version"):
        TrainingRegistry.from_dict({"gaps": [_VALID_GAP]})


def test_validation_gaps_must_be_nonempty_list():
    with pytest.raises(RegistryValidationError, match="gaps"):
        TrainingRegistry.from_dict(_valid_registry_dict({"gaps": []}))
    with pytest.raises(RegistryValidationError, match="gaps"):
        TrainingRegistry.from_dict(_valid_registry_dict({"gaps": "not a list"}))


def test_validation_duplicate_canonical_name_rejected():
    with pytest.raises(RegistryValidationError, match="duplicates"):
        TrainingRegistry.from_dict(_valid_registry_dict({
            "gaps": [_VALID_GAP, _VALID_GAP],
        }))


def test_validation_gap_missing_canonical_name():
    bad = {**_VALID_GAP, "canonical_name": ""}
    with pytest.raises(RegistryValidationError, match="canonical_name"):
        TrainingRegistry.from_dict(_valid_registry_dict({"gaps": [bad]}))


def test_validation_gap_missing_aliases():
    bad = {**_VALID_GAP, "aliases": []}
    with pytest.raises(RegistryValidationError, match="aliases"):
        TrainingRegistry.from_dict(_valid_registry_dict({"gaps": [bad]}))


def test_validation_gap_invalid_category():
    bad = {**_VALID_GAP, "category": "fictional"}
    with pytest.raises(RegistryValidationError, match="category"):
        TrainingRegistry.from_dict(_valid_registry_dict({"gaps": [bad]}))


def test_validation_resources_must_be_nonempty():
    bad = {**_VALID_GAP, "resources": []}
    with pytest.raises(RegistryValidationError, match="resources"):
        TrainingRegistry.from_dict(_valid_registry_dict({"gaps": [bad]}))


def test_validation_resource_invalid_type():
    bad_gap = {
        **_VALID_GAP,
        "resources": [{
            "provider": "p", "type": "made_up_type",
            "url": None, "summary": "s",
            "verified_at": None, "verified_by": None,
        }],
    }
    with pytest.raises(RegistryValidationError, match="type"):
        TrainingRegistry.from_dict(_valid_registry_dict({"gaps": [bad_gap]}))


def test_validation_non_referral_must_have_url():
    bad_gap = {
        **_VALID_GAP,
        "resources": [{
            "provider": "p", "type": "credential_pathway",
            "url": None, "summary": "s",
            "verified_at": None, "verified_by": None,
        }],
    }
    with pytest.raises(RegistryValidationError, match="non-empty url"):
        TrainingRegistry.from_dict(_valid_registry_dict({"gaps": [bad_gap]}))


def test_validation_url_must_be_https():
    bad_gap = {
        **_VALID_GAP,
        "resources": [{
            "provider": "p", "type": "credential_pathway",
            "url": "http://example.com",
            "summary": "s", "verified_at": None, "verified_by": None,
        }],
    }
    with pytest.raises(RegistryValidationError, match="https"):
        TrainingRegistry.from_dict(_valid_registry_dict({"gaps": [bad_gap]}))


def test_validation_referral_only_must_have_null_url():
    bad_gap = {
        **_VALID_GAP,
        "resources": [{
            "provider": "p", "type": "referral_only",
            "url": "https://example.com",
            "summary": "s", "verified_at": None, "verified_by": None,
        }],
    }
    with pytest.raises(RegistryValidationError, match="referral_only"):
        TrainingRegistry.from_dict(_valid_registry_dict({"gaps": [bad_gap]}))


def test_validation_verified_at_can_be_null():
    """The whole pending-verification mechanism depends on null
    being a legal verified_at value."""
    # Valid registry with null verified_at -- should load cleanly
    registry = TrainingRegistry.from_dict(_valid_registry_dict())
    assert registry.gaps[0].resources[0].verified_at is None


def test_validation_verified_at_invalid_string_rejected():
    bad_gap = {
        **_VALID_GAP,
        "resources": [{
            "provider": "p", "type": "referral_only",
            "url": None, "summary": "s",
            "verified_at": "not-a-date", "verified_by": None,
        }],
    }
    with pytest.raises(RegistryValidationError, match="verified_at"):
        TrainingRegistry.from_dict(_valid_registry_dict({"gaps": [bad_gap]}))


def test_validation_verified_at_accepts_yyyy_mm_dd_string():
    """YAML round-trip: dates may come in as strings if quoted."""
    good_gap = {
        **_VALID_GAP,
        "resources": [{
            "provider": "p", "type": "referral_only",
            "url": None, "summary": "s",
            "verified_at": "2026-06-04", "verified_by": "test",
        }],
    }
    registry = TrainingRegistry.from_dict(_valid_registry_dict({"gaps": [good_gap]}))
    assert registry.gaps[0].resources[0].verified_at == date(2026, 6, 4)


def test_validation_verified_at_set_requires_verified_by():
    """Audit-trail invariant (post-loader-review fix): a date in
    verified_at MUST be paired with a non-null verified_by. Otherwise
    `surface_url` would expose the URL while leaving no record of who
    verified it -- exactly the accountability gap this rule prevents."""
    bad_gap = {
        **_VALID_GAP,
        "resources": [{
            "provider": "p", "type": "credential_pathway",
            "url": "https://example.com/cert",
            "summary": "s",
            "verified_at": "2026-06-04",
            "verified_by": None,           # <-- missing audit trail
        }],
    }
    with pytest.raises(RegistryValidationError, match="verified_by"):
        TrainingRegistry.from_dict(_valid_registry_dict({"gaps": [bad_gap]}))


def test_validation_both_verified_fields_null_allowed_as_pending():
    """The pending case: both verified_at and verified_by are null.
    No URL surfaces; no audit trail required because there's nothing
    to vouch for yet."""
    pending_gap = {
        **_VALID_GAP,
        "resources": [{
            "provider": "p", "type": "credential_pathway",
            "url": "https://example.com/cert",
            "summary": "s",
            "verified_at": None,
            "verified_by": None,
        }],
    }
    registry = TrainingRegistry.from_dict(_valid_registry_dict({"gaps": [pending_gap]}))
    resource = registry.gaps[0].resources[0]
    assert resource.is_pending
    assert resource.surface_url(date.today()) is None    # safety net still works


def test_validation_both_verified_fields_set_allowed_as_verified():
    """The verified case: both fields set, URL surfaces normally."""
    good_gap = {
        **_VALID_GAP,
        "resources": [{
            "provider": "p", "type": "credential_pathway",
            "url": "https://example.com/cert",
            "summary": "s",
            "verified_at": "2026-06-04",
            "verified_by": "lead-engineer",
        }],
    }
    registry = TrainingRegistry.from_dict(_valid_registry_dict({"gaps": [good_gap]}))
    resource = registry.gaps[0].resources[0]
    assert not resource.is_pending
    assert resource.verified_at == date(2026, 6, 4)
    assert resource.verified_by == "lead-engineer"


def test_validation_verified_at_accepts_python_date():
    """YAML can also deserialize dates to Python date objects directly
    when the value is unquoted (e.g. `verified_at: 2026-06-04`)."""
    good_gap = {
        **_VALID_GAP,
        "resources": [{
            "provider": "p", "type": "referral_only",
            "url": None, "summary": "s",
            "verified_at": date(2026, 6, 4), "verified_by": "test",
        }],
    }
    registry = TrainingRegistry.from_dict(_valid_registry_dict({"gaps": [good_gap]}))
    assert registry.gaps[0].resources[0].verified_at == date(2026, 6, 4)


# =========================================================================
# 6. Immutability
# =========================================================================
def test_registry_is_frozen():
    r = TrainingRegistry.from_dict(_valid_registry_dict())
    with pytest.raises(Exception):
        r.version = 2  # type: ignore[misc]


def test_gap_is_frozen():
    r = TrainingRegistry.from_dict(_valid_registry_dict())
    with pytest.raises(Exception):
        r.gaps[0].canonical_name = "other"  # type: ignore[misc]


def test_resource_is_frozen():
    r = TrainingRegistry.from_dict(_valid_registry_dict())
    with pytest.raises(Exception):
        r.gaps[0].resources[0].provider = "other"  # type: ignore[misc]


def test_resources_is_tuple_not_list():
    """Resources stored as tuple so they can't be mutated even via
    list.append on a shared reference."""
    r = TrainingRegistry.from_dict(_valid_registry_dict())
    assert isinstance(r.gaps[0].resources, tuple)
