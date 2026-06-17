"""Training registry loader.

Reads the curated YAML at `data/training_registry.yaml` and exposes a
read-only, in-memory `TrainingRegistry` singleton. The loader does
structural validation only (every hard rule from
docs/training-registry-schema.md); provider-name policy enforcement
lives in a separate CI test, not at load time. See that doc for the
"loader vs CI vs human review" enforcement split.

Lookup contract:
  - `lookup(query)` returns the first matching Gap (canonical or alias)
  - `surface_resources(query, today)` returns the resources for that
    gap with URL freshness applied (pending/stale URLs are SUPPRESSED;
    the Resource objects still come through with url=None so the
    responder can narrate the provider name + generic guidance)
  - Unknown gaps return an empty list AND log at INFO so the registry
    can grow from real usage (see the unknown_gap log line for the
    grep pattern)

No DB calls. Pure file + Python. Safe to call from any request thread
after the initial `from_yaml()` build because all dataclasses are
frozen.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from skillbridge.training.models import (
    DEFAULT_FRESHNESS_DAYS,
    Gap,
    Resource,
    normalize_gap_name,
)

log = logging.getLogger(__name__)


# Default location: repo-root/data/training_registry.yaml
# (this file is skillbridge/training/registry.py -> parents[2] = repo root)
_DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "training_registry.yaml"
)


# Schema enums -- mirror docs/training-registry-schema.md
_VALID_RESOURCE_TYPES: frozenset[str] = frozenset({
    "credential_pathway",
    "local_training",
    "apprenticeship",
    "online_course",
    "referral_only",
})

_VALID_CATEGORIES: frozenset[str] = frozenset({
    "credential",
    "skill",
    "license",
    "safety_training",
})


# Aliases (post-normalization) that SHOULD NOT participate in
# message-scan matching. These are common English words or
# single-letter abbreviations that would false-positive on casual
# chat input ("I gave my word", "office hours", "the service was
# good", "I want to G it"). The alias still appears in the YAML so
# direct `registry.lookup("Word")` still works; it just doesn't get
# scanned for in free-text messages.
#
# All entries are pre-normalized (lowercase, single-spaced, no
# trailing punctuation) for direct set membership checks.
_ALIASES_NOT_FOR_MESSAGE_SCAN: frozenset[str] = frozenset({
    # Single letters and 2-char abbreviations that are too short to
    # disambiguate in chat. Their multi-word forms still match
    # ("Class G", "AZ license").
    "g", "a", "d", "z", "az", "dz",
    # Common English words that overlap with brand/credential aliases.
    # Microsoft Office aliases include "Word", "Outlook", "Office" --
    # all too noisy as bare matches. Use "Microsoft Word" /
    # "MS Office" / "Office 365" to disambiguate.
    "word", "outlook", "office",
    # Workplace nouns that are too generic to safely match without
    # context.
    "service", "safety", "care",
})


class RegistryValidationError(ValueError):
    """Raised when the YAML fails one of the schema's hard rules.

    The loader fails LOUD: server startup aborts with a clear message
    rather than silently shipping a corrupt registry. A registry with
    one bad entry would otherwise produce mis-cited URLs in chat, which
    is exactly the failure mode the whole thing exists to prevent.
    """


@dataclass(frozen=True)
class TrainingRegistry:
    """Read-only in-memory registry. Loaded once at server startup.

    All fields are immutable (frozen dataclass; tuples instead of lists)
    so request threads can share a single instance safely after the
    initial build.
    """
    version: int
    registry_verified_at: date | None
    gaps: tuple[Gap, ...]

    # ----- factory: load from disk -----
    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> TrainingRegistry:
        """Load + validate the registry YAML.

        Raises RegistryValidationError on any structural problem.
        Path defaults to `data/training_registry.yaml` at the repo root.
        """
        resolved_path = Path(path) if path is not None else _DEFAULT_REGISTRY_PATH
        with open(resolved_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        log.info(
            "training_registry loading from=%s",
            str(resolved_path),
        )
        return cls._validate_and_build(raw, source=str(resolved_path))

    # ----- factory: build from in-memory dict (tests) -----
    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainingRegistry:
        return cls._validate_and_build(raw, source="<dict>")

    # ----- message-based gap discovery -----
    def find_gaps_in_message(
        self, message: str, *, max_results: int = 5,
    ) -> list[Gap]:
        """Scan the user's message for any canonical_name or alias
        from the registry. Returns matched Gap objects in first-found
        order, deduped, capped at max_results.

        Used by the recommender to discover training intent from
        cold-session messages like "how do I get my Class G?" where
        no prior match turn has populated
        `staged.last_presented_credential_gaps`. Without this method,
        the only way to surface registry data was via the post-match
        carry-forward path -- which left direct training questions
        with an empty TRAINING block.

        Matching rules:
          - Word-boundary regex match (case-insensitive). "G" alone in
            "give me a G" must NOT match the "G license" alias.
          - Aliases in `_ALIASES_NOT_FOR_MESSAGE_SCAN` are SKIPPED for
            message scanning (still allowed for direct lookup via
            `lookup()`). This is the safety net against common English
            words ("Word", "Office", "service") creeping into matches.
          - Canonical names always participate (they're typically
            multi-word and unambiguous, e.g. "customer service").
          - First match per gap wins; alias order within a gap doesn't matter.
        """
        if not message:
            return []
        normalized = " " + normalize_gap_name(message) + " "
        found: list[Gap] = []
        seen: set[str] = set()
        for gap in self.gaps:
            if gap.canonical_name in seen:
                continue
            candidates: list[str] = [gap.canonical_name, *gap.aliases]
            for candidate in candidates:
                normalized_candidate = normalize_gap_name(candidate)
                if not normalized_candidate:
                    continue
                if normalized_candidate in _ALIASES_NOT_FOR_MESSAGE_SCAN:
                    continue
                # Word-boundary phrase match. Pad the message with
                # spaces so single-token candidates at the start/end
                # are bounded.
                if f" {normalized_candidate} " in normalized:
                    seen.add(gap.canonical_name)
                    found.append(gap)
                    if len(found) >= max_results:
                        return found
                    break
        return found

    # ----- lookup -----
    def lookup(self, query: str) -> Gap | None:
        """Find a gap whose canonical_name or any alias matches the
        normalized query. First-match-wins.

        Returns None for unknown gaps; caller is expected to log via
        `register_unknown_gap` (or use `surface_resources` which logs
        automatically)."""
        if not query:
            return None
        for gap in self.gaps:
            if gap.matches(query):
                return gap
        return None

    def surface_resources(
        self,
        query: str,
        *,
        today: date,
        limit: int = 3,
        max_age_days: int = DEFAULT_FRESHNESS_DAYS,
    ) -> list[Resource]:
        """Return up to `limit` Resource objects for the matched gap.

        Resources are returned AS-IS (the `.url` attribute is unchanged
        on the dataclass). URL suppression happens at the CALL SITE
        when callers use `Resource.surface_url(today)` -- that method
        returns None for pending, stale, and referral_only resources.
        The handler's `_registry_training_for_gap` uses `surface_url`,
        which is why pending URLs never reach the chat prompt.

        Keeping the suppression on the method instead of the lookup
        means the safety net travels with the Resource: any new
        consumer that calls `.url` directly will see the candidate
        URL, but the documented contract (and the only consumer in
        production) goes through `surface_url`. Tests assert that
        contract.

        Unknown gaps log an INFO line (`unknown_gap=...`) and return [].
        That's the telemetry hook for growing the registry from real
        chat traffic.
        """
        gap = self.lookup(query)
        if gap is None:
            log.info("training_registry unknown_gap=%r", query)
            return []
        return list(gap.resources[:limit])

    # ============================================================
    # Internal: validation + build
    # ============================================================
    @classmethod
    def _validate_and_build(cls, raw: Any, *, source: str) -> TrainingRegistry:
        if not isinstance(raw, dict):
            raise RegistryValidationError(
                f"{source}: top-level YAML must be a dict, got "
                f"{type(raw).__name__}"
            )

        # ---- version ----
        version = raw.get("version")
        if version != 1:
            raise RegistryValidationError(
                f"{source}: 'version' must be 1, got {version!r}"
            )

        # ---- registry_verified_at (optional, may be null) ----
        rva_raw = raw.get("registry_verified_at")
        rva = _parse_optional_date(rva_raw, where=f"{source}: registry_verified_at")

        # ---- gaps ----
        gaps_raw = raw.get("gaps")
        if not isinstance(gaps_raw, list) or not gaps_raw:
            raise RegistryValidationError(
                f"{source}: 'gaps' must be a non-empty list"
            )

        seen_canonical: set[str] = set()
        gaps: list[Gap] = []
        for i, gap_raw in enumerate(gaps_raw):
            gap = _build_gap(gap_raw, source=source, idx=i)
            normalized = normalize_gap_name(gap.canonical_name)
            if normalized in seen_canonical:
                raise RegistryValidationError(
                    f"{source}: gap {i}: canonical_name "
                    f"{gap.canonical_name!r} duplicates an earlier entry"
                )
            seen_canonical.add(normalized)
            gaps.append(gap)

        return cls(
            version=version,
            registry_verified_at=rva,
            gaps=tuple(gaps),
        )


# ============================================================
# Internal builders + parsers
# ============================================================
def _build_gap(raw: Any, *, source: str, idx: int) -> Gap:
    where = f"{source}: gap {idx}"
    if not isinstance(raw, dict):
        raise RegistryValidationError(f"{where}: must be a dict")

    canonical_name = raw.get("canonical_name")
    if not isinstance(canonical_name, str) or not canonical_name.strip():
        raise RegistryValidationError(f"{where}: missing or empty canonical_name")
    canonical_name = canonical_name.strip()

    aliases = raw.get("aliases")
    if not isinstance(aliases, list) or not aliases:
        raise RegistryValidationError(
            f"{where} ({canonical_name!r}): aliases must be a non-empty list"
        )
    cleaned_aliases: list[str] = []
    for j, a in enumerate(aliases):
        if not isinstance(a, str) or not a.strip():
            raise RegistryValidationError(
                f"{where} ({canonical_name!r}): alias {j} must be a non-empty string"
            )
        cleaned_aliases.append(a.strip())

    category = raw.get("category")
    if category not in _VALID_CATEGORIES:
        raise RegistryValidationError(
            f"{where} ({canonical_name!r}): category {category!r} "
            f"not in {sorted(_VALID_CATEGORIES)}"
        )

    description = raw.get("description")
    if not isinstance(description, str) or not description.strip():
        raise RegistryValidationError(
            f"{where} ({canonical_name!r}): missing or empty description"
        )

    resources_raw = raw.get("resources")
    if not isinstance(resources_raw, list) or not resources_raw:
        raise RegistryValidationError(
            f"{where} ({canonical_name!r}): resources must be a non-empty list"
        )

    resources: list[Resource] = []
    for k, r_raw in enumerate(resources_raw):
        resource = _build_resource(
            r_raw, source=source, gap_idx=idx, gap_name=canonical_name, res_idx=k,
        )
        resources.append(resource)

    return Gap(
        canonical_name=canonical_name,
        aliases=tuple(cleaned_aliases),
        category=category,
        description=description.strip(),
        resources=tuple(resources),
    )


def _build_resource(
    raw: Any, *, source: str, gap_idx: int, gap_name: str, res_idx: int,
) -> Resource:
    where = f"{source}: gap {gap_idx} ({gap_name!r}) resource {res_idx}"
    if not isinstance(raw, dict):
        raise RegistryValidationError(f"{where}: must be a dict")

    provider = raw.get("provider")
    if not isinstance(provider, str) or not provider.strip():
        raise RegistryValidationError(f"{where}: missing or empty provider")
    provider = provider.strip()

    type_ = raw.get("type")
    if type_ not in _VALID_RESOURCE_TYPES:
        raise RegistryValidationError(
            f"{where} ({provider!r}): type {type_!r} not in "
            f"{sorted(_VALID_RESOURCE_TYPES)}"
        )

    url = raw.get("url")
    if type_ == "referral_only":
        if url is not None:
            raise RegistryValidationError(
                f"{where} ({provider!r}): referral_only must have url=null, "
                f"got {url!r}"
            )
    else:
        if not isinstance(url, str) or not url.strip():
            raise RegistryValidationError(
                f"{where} ({provider!r}): type={type_!r} requires a "
                f"non-empty url string"
            )
        if not url.startswith("https://"):
            raise RegistryValidationError(
                f"{where} ({provider!r}): url must start with 'https://', "
                f"got {url!r}"
            )

    summary = raw.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise RegistryValidationError(f"{where} ({provider!r}): missing or empty summary")

    verified_at = _parse_optional_date(
        raw.get("verified_at"), where=f"{where} ({provider!r}): verified_at",
    )

    verified_by_raw = raw.get("verified_by")
    if verified_by_raw is None:
        verified_by = None
    elif isinstance(verified_by_raw, str) and verified_by_raw.strip():
        verified_by = verified_by_raw.strip()
    else:
        raise RegistryValidationError(
            f"{where} ({provider!r}): verified_by must be a non-empty "
            f"string or null"
        )

    # Audit-trail invariant: when verified_at is set, verified_by must
    # name who did the verification. Otherwise a contributor could set
    # verified_at without naming themselves, the URL would surface in
    # chat (surface_url only checks verified_at), and we'd have no
    # record of who vouched. Both fields move together or neither does.
    if verified_at is not None and verified_by is None:
        raise RegistryValidationError(
            f"{where} ({provider!r}): verified_by must be a non-empty "
            f"string when verified_at is set (got verified_at="
            f"{verified_at}, verified_by=null). Every verified URL "
            f"requires an audit trail."
        )

    return Resource(
        provider=provider,
        type=type_,
        url=url.strip() if isinstance(url, str) else None,
        summary=summary.strip(),
        verified_at=verified_at,
        verified_by=verified_by,
    )


def _parse_optional_date(value: Any, *, where: str) -> date | None:
    """YAML deserializes 'YYYY-MM-DD' to either a Python date or a str
    depending on quoting. Handle both. Null = pending verification."""
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError as e:
            raise RegistryValidationError(
                f"{where}: invalid date {value!r}: {e}"
            ) from None
    raise RegistryValidationError(
        f"{where}: must be YYYY-MM-DD date string or null, got "
        f"{type(value).__name__}"
    )


# ============================================================
# Module-level singleton (lazy)
# ============================================================
_registry: TrainingRegistry | None = None


def get_registry() -> TrainingRegistry:
    """Return the lazily-loaded process-wide registry singleton.

    Called by the handler integration. Tests should NOT call this --
    use `TrainingRegistry.from_yaml(path)` or `from_dict(...)` directly
    so each test gets a fresh instance.
    """
    global _registry
    if _registry is None:
        _registry = TrainingRegistry.from_yaml()
    return _registry


def reset_registry() -> None:
    """Drop the singleton. Used by tests that need to force a reload."""
    global _registry
    _registry = None
