"""Training registry data model.

Two frozen dataclasses: `Resource` (one training option from one provider)
and `Gap` (the canonical skill/credential the user is missing, plus the
list of resources that address it). Both are immutable so the registry
can be shared safely across request threads after a single startup load.

The freshness logic lives on `Resource`. The runtime safety contract:

  - `verified_at is None`        -> URL SUPPRESSED at runtime (pending state)
  - `verified_at < today - 180d` -> URL SUPPRESSED at runtime (stale state)
  - `type == "referral_only"`    -> URL is null by definition; provider
                                    name is the recommendation

A suppressed URL means the Resource object is still returned to the
responder, but `surface_url(today)` returns None. The responder narrates
"contact the provider directly" or falls through to a SCCC referral
entry from the same gap's resource list.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Literal


# Mirrors docs/training-registry-schema.md §"type taxonomy"
ResourceType = Literal[
    "credential_pathway",  # authoritative regulator/issuer
    "local_training",      # course/program at a local provider
    "apprenticeship",      # workplace + classroom hybrid
    "online_course",       # MOOC or vendor cert page
    "referral_only",       # no URL; counsellor referral
]


Category = Literal[
    "credential",       # licence/certificate issued by an authority
    "skill",            # general competency (Excel, customer service)
    "license",          # driver's licence and similar
    "safety_training",  # WHMIS, first aid, food handler
]


# Freshness window. Mirrors the policy in
# docs/training-providers-allowlist.md §"Re-verification convention".
DEFAULT_FRESHNESS_DAYS = 180


@dataclass(frozen=True)
class Resource:
    """One training option from one provider for one gap."""

    provider: str
    type: str             # ResourceType -- runtime check, not Literal-enforced here
    url: str | None       # None for referral_only or pending/stale
    summary: str
    verified_at: date | None      # None = pending verification
    verified_by: str | None       # None when verified_at is None

    @property
    def is_pending(self) -> bool:
        """True when this resource has never been human-verified.
        Pending resources NEVER surface a URL at runtime."""
        return self.verified_at is None

    def is_fresh(self, today: date, *, max_age_days: int = DEFAULT_FRESHNESS_DAYS) -> bool:
        """True when verified_at is set AND within the freshness window.
        Pending (None) and stale (> max_age_days) resources are NOT fresh."""
        if self.verified_at is None:
            return False
        age_days = (today - self.verified_at).days
        return 0 <= age_days <= max_age_days

    def surface_url(
        self, today: date, *, max_age_days: int = DEFAULT_FRESHNESS_DAYS,
    ) -> str | None:
        """Return the URL only if it's safe to show the user.

        Returns None when:
          - type is referral_only (no URL by design)
          - verified_at is None (pending review)
          - verified_at is older than max_age_days (stale; link rot likely)

        Otherwise returns the URL. The responder must use this method
        rather than the raw `.url` attribute -- handing the raw URL to
        chat would bypass the freshness safety net.
        """
        if self.type == "referral_only":
            return None
        if not self.is_fresh(today, max_age_days=max_age_days):
            return None
        return self.url


@dataclass(frozen=True)
class Gap:
    """A canonical skill or credential the user is missing.

    Indexed for lookup by `canonical_name` plus any number of `aliases`
    (the match engine emits various phrasings for the same gap; the
    aliases let the registry match against all of them deterministically).
    """

    canonical_name: str
    aliases: tuple[str, ...]
    category: str         # Category -- runtime check, not Literal-enforced
    description: str
    resources: tuple[Resource, ...]

    def matches(self, query: str) -> bool:
        """Case-insensitive, whitespace-normalized match against
        canonical_name OR any alias."""
        if not query:
            return False
        normalized = normalize_gap_name(query)
        if normalize_gap_name(self.canonical_name) == normalized:
            return True
        return any(normalize_gap_name(a) == normalized for a in self.aliases)


def normalize_gap_name(name: str) -> str:
    """Normalize a gap name for lookup.

    Lowercase, strip leading/trailing whitespace, collapse internal
    whitespace, strip trailing punctuation. Used on BOTH sides of every
    lookup so 'truck and coach technician certificate' and
    'Truck and Coach Technician Certificate.' both match the same gap.
    """
    s = name.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".,!?;:")
    return s.strip()
