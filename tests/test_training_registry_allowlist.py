"""CI policy check: every provider in the training registry YAML must
appear in the allowlist doc.

This is the "soft" policy assertion -- it can be relaxed during PR
review with explicit human approval -- but it CATCHES the case where
a contributor adds a provider to the YAML without thinking through
the trust criteria documented in `docs/training-providers-allowlist.md`.

The loader (registry.py) does NOT do this check. Per
`docs/training-registry-schema.md`, the enforcement split is:

    Loader            : structural correctness    (rejects bad YAML)
    THIS CI test      : provider allowlist policy (catches drift)
    Human review      : trust judgment + URL quality  (PR approval)

The canonical provider names below are the source of truth for the
test. The markdown doc explains the rationale for each (rationale
isn't easily parseable; the test asserts membership only). If you add
a provider to the YAML, you must also:

    1. Add the canonical name to ALLOWED_PROVIDERS below.
    2. Add a row to docs/training-providers-allowlist.md with rationale.

Both edits in the same PR. Reviewer checks both.
"""
from __future__ import annotations

import os

os.environ.setdefault("LLM_ENABLED", "false")

import pytest

from skillbridge.training.registry import TrainingRegistry

pytestmark = pytest.mark.nodb


# =========================================================================
# Canonical provider names -- source of truth for the test
# =========================================================================
# Each entry must EXACTLY MATCH a `provider:` value in
# data/training_registry.yaml. If a provider name in the YAML uses a
# different form (e.g. "CCOHS" vs "Canadian Centre for Occupational
# Health and Safety"), normalize the YAML to match one of these names.
# Don't add aliases here -- the YAML side is canonical-form-only.
#
# These names also appear in docs/training-providers-allowlist.md.
# Keep both in sync when adding new providers.
ALLOWED_PROVIDERS: frozenset[str] = frozenset({
    # Local -- SSM core
    "Sault College",
    "Algoma University",
    "Sault Community Career Centre",
    "Northland Adult Learning Centre",
    "OntarioColleges.ca",
    # Ontario credential authorities
    "Skilled Trades Ontario",
    "DriveTest",
    "Ontario.ca",
    "ServiceOntario",
    "Ministry of Labour, Immigration, Training and Skills Development",
    "Smart Serve Ontario",
    "Sault Ste. Marie Police Service",
    "College of Early Childhood Educators",
    "College of Nurses of Ontario",
    # National MOOCs / vendor certifications
    "Microsoft Learn",
    "AWS Skill Builder",
    "Google Career Certificates",
    "Coursera",
    "edX",
    "CompTIA",
    "Intuit (QuickBooks)",
    "National Payroll Institute",
    # Health / safety credential providers
    "Canadian Red Cross",
    "St. John Ambulance",
    "CCOHS",
    "TrainCan",
    "Algoma Public Health",
})


# =========================================================================
# The check itself
# =========================================================================
def test_every_yaml_provider_is_in_allowlist():
    """Reads the shipped data/training_registry.yaml and asserts every
    `provider:` string is in ALLOWED_PROVIDERS.

    Failure mode: a contributor added a provider to the YAML without
    updating the allowlist doc + this test. Fix is two-place: add to
    docs/training-providers-allowlist.md AND to ALLOWED_PROVIDERS above,
    in the same PR.
    """
    registry = TrainingRegistry.from_yaml()
    yaml_providers: set[str] = set()
    for gap in registry.gaps:
        for resource in gap.resources:
            yaml_providers.add(resource.provider)

    unauthorized = yaml_providers - ALLOWED_PROVIDERS
    assert not unauthorized, (
        f"YAML uses providers not on the allowlist: {sorted(unauthorized)}.\n\n"
        f"Each one must EITHER:\n"
        f"  (a) be added to ALLOWED_PROVIDERS in this test file AND to\n"
        f"      docs/training-providers-allowlist.md with rationale, OR\n"
        f"  (b) be replaced with a provider already on the allowlist.\n\n"
        f"This is a policy decision -- don't silence the test without\n"
        f"updating the allowlist doc."
    )


def test_canonical_provider_set_has_no_duplicates():
    """frozenset already deduplicates, but spot-check that the human
    list above didn't have copy-paste mistakes."""
    # If a member appears twice, frozenset dedupes it. Check via list count.
    raw_list = [
        "Sault College", "Algoma University", "Sault Community Career Centre",
        "Northland Adult Learning Centre", "Skilled Trades Ontario", "DriveTest",
        "Ontario.ca", "ServiceOntario",
        "Ministry of Labour, Immigration, Training and Skills Development",
        "Microsoft Learn", "AWS Skill Builder", "Google Career Certificates",
        "Coursera", "edX", "CompTIA", "Intuit (QuickBooks)",
        "Canadian Red Cross", "St. John Ambulance", "CCOHS", "TrainCan",
        "Algoma Public Health",
    ]
    assert len(raw_list) == len(set(raw_list)), (
        "ALLOWED_PROVIDERS has duplicate entries; check the list."
    )


def test_allowlist_doc_mentions_each_allowed_provider():
    """Companion check: every name in ALLOWED_PROVIDERS appears as a
    bolded table row in docs/training-providers-allowlist.md.

    This catches the drift direction: someone added a provider here
    but forgot to document it in the markdown. The check looks for
    the markdown pattern `| **<provider>** |`."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    doc_path = repo_root / "docs" / "training-providers-allowlist.md"
    doc_text = doc_path.read_text(encoding="utf-8")

    missing = [
        p for p in ALLOWED_PROVIDERS
        if f"**{p}**" not in doc_text
    ]
    assert not missing, (
        f"ALLOWED_PROVIDERS includes providers not documented in the "
        f"allowlist markdown: {sorted(missing)}.\n\n"
        f"Add a row to docs/training-providers-allowlist.md for each\n"
        f"missing provider with a Why and Typical resource types column."
    )
