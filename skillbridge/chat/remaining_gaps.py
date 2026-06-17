"""Remaining-gaps detection module (R-2 of the remaining-gaps iteration).

Deterministic detection of the seven user-turn categories that drive the
post-match follow-up reasoning layer:

  1. Pending confirmation consumption  -- yes/no/unrelated to a prior
     `kind="confirm"` question the system asked
  2. Retraction against accumulated state -- any explicit negation
     pointing at a credential already in `accumulated_credentials`
     (initiates a `kind="confirm"` with `pending_action="remove"`)
  3. Standard negation for non-accumulated entities -- `kind=None`, falls
     through to existing planner/router
  4. Uncertainty ("think I got X", "pretty sure I have X") -- forces a
     `kind="confirm"` add-confirmation
  5. Explicit completion ("I have X", "I passed X") --
     `kind="subtract"` with mode=`claimed`
  6. Explicit hypothetical ("if I had X", "assume I have X") --
     `kind="subtract"` with mode=`hypothetical`
  7. Generic remaining ("what else?", "anything else?") --
     `kind="subtract"` with no claims when a snapshot exists, otherwise
     `kind="bootstrap"`

Identity contract (docs/remaining-gaps-design.md §4.0 / §4.2 / §4.3):

  Every canonical the detector emits MUST be a value pulled VERBATIM
  from a snapshot.lead_job.credential_gaps[*].canonical slot. The
  detector never invents a fresh canonical from a registry resolution
  and compares against the snapshot. This keeps a snapshot captured in
  Mode B (canonical = normalized display) usable in a later Mode-A turn.

Purity contract:

  The detector is a pure function. It accepts pending state as a value
  copy and MUST NOT mutate any StagedProfile field. The handler owns
  clearing pending state BEFORE the detection call. See
  docs/remaining-gaps-design.md §2 "Clearing ownership."

Registry degradation:

  `registry` is typed `TrainingRegistry | None`. Mode C (registry=None)
  is handled by the deterministic token-fallback resolver in §4.3 with
  no behavior change other than skipping the Mode A registry-assisted
  lookup. Same algorithm for everything else.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


# ============================================================================
# Discriminated-union result shape
# ============================================================================
@dataclass(frozen=True)
class CredentialClaim:
    """A single credential claim emitted in `current_turn_claims`.
    Frozen so downstream code (handler integration, transcript tests)
    cannot mutate the resolved canonical or mode after the detector
    returns. The handler converts to its persisted-dict form on the way
    into `staged.last_assumed_completed_credentials` -- the dict
    representation is the storage shape, this dataclass is the wire
    shape between detector and handler."""
    canonical: str
    mode: str            # "claimed" | "hypothetical"

    def to_dict(self) -> dict[str, str]:
        return {"canonical": self.canonical, "mode": self.mode}


@dataclass(frozen=True)
class RemainingGapsIntent:
    """Discriminated union of detection outcomes. Inspect `.kind` first.

    Shapes:
      kind="subtract"  -> current_turn_claims is a tuple of
                          CredentialClaim entries (frozen, so downstream
                          callers cannot mutate the resolved canonical
                          or mode). Empty tuple is valid (generic
                          "what else?" with snapshot existing -- handler
                          subtracts from accumulated state).
      kind="retract"   -> retract_canonical is the snapshot canonical
                          to remove from accumulated state.
      kind="confirm"   -> confirmation_target_canonical may be None
                          (ambiguous "got it" disambiguation, or a
                          credential verb fired but no unique snapshot
                          match) or a snapshot canonical;
                          confirmation_target_display is the user-facing
                          string. pending_action is "add" or "remove".
      kind="bootstrap" -> the user asked a remaining-gap-shaped question
                          but no snapshot exists. No entity payload.
    """
    kind: str
    current_turn_claims: tuple[CredentialClaim, ...] = ()
    retract_canonical: str | None = None
    confirmation_target_canonical: str | None = None
    confirmation_target_display: str = ""
    pending_action: str | None = None


# ============================================================================
# Generic-credential stop-words for the §4.3 token-fallback resolver
# ============================================================================
# Removing these from both sides before subset comparison so the matcher
# never resolves "the license" against a snapshot entry (which would let
# every credential mention claim every snapshot credential). The list is
# intentionally short -- false negatives are recovered by clarification;
# false positives silently corrupt accumulated state.
_GENERIC_CREDENTIAL_TOKENS: frozenset[str] = frozenset({
    # Generic credential vocabulary. Round-13 NOTE: this set MUST be a
    # superset of `_CREDENTIAL_SHAPED_TOKENS` below so the round-13
    # gate `_is_credential_shaped(entity) AND not _tokens(entity)`
    # reads naturally as "entity is made of nothing but credential
    # vocabulary." If a credential-shaped token isn't generic, a phrase
    # like "I have my cert" would still keep `cert` in the token set
    # and the gate would incorrectly fall through to None.
    "license", "licence", "licenses", "licences",
    "certification", "certifications", "certificate", "certificates",
    "cert",
    "permit", "permits",
    "card", "cards",
    "credential", "credentials",
    "ticket", "tickets",
    "endorsement", "endorsements",
    "course", "training", "class",
    # English filler / first-person markers
    "the", "a", "an", "of", "and", "or",
    "my", "your", "his", "her",
    "i", "ive", "im", "youre",
    # Verbs that appear in the patterns themselves (we want the entity
    # tokens, not the verb)
    "have", "had", "got", "get", "getting",
    "earned", "earn", "earning",
    "finished", "finish", "finishing",
    "passed", "pass", "passing",
    "completed", "complete", "completing",
    "missing", "without", "lacking",
    "now", "already", "currently",
    # Other generic glue
    "for", "to", "from", "with", "in", "on",
    "do", "does", "did", "done",
    "it", "that", "this", "those", "these",
    "any", "some", "other", "another", "more",
    "else", "next", "first", "second", "third",
    "what", "which", "who", "whose",
    "still", "yet", "also",
    # Bare-anaphor markers (resolver handles them separately)
    "one",
})


# ============================================================================
# Regex pattern fragments
# ============================================================================
# Each "category" pattern captures the entity-reference substring as
# group 1, which the resolver then tokenises and looks up against the
# snapshot entries. Patterns are evaluated against the lower-cased
# message; the regex layer is the per-token VERB shape, not the
# entity identity.

# Helper -- match the credential entity that follows a verb. Capture
# extends across `and` / `,` conjunctions so multi-credential claims
# ("I have 310S and Class G", "I have both 310S, G2, and Smart Serve")
# are reachable by a single regex; the caller splits the captured
# string on conjunctions and resolves each part independently.
# Terminators: end-of-clause punctuation, sentence boundary, a small
# set of transition words, and -- critically -- the start of a new
# verb-marker clause like " i think " (so the completion capture
# doesn't swallow a subsequent uncertainty sub-clause).
_ENTITY_TAIL = (
    r"(?:my|the|a|an|that|this|it|one|both)?\s*"
    r"([a-z0-9'/\- ,]{1,160}?)"
    r"(?:[.!?]|$"
    r"|\s+(?:but|so|for|while|right|now|too|also|already)\b"
    r"|\s+i\s+(?:think|believe|guess|don'?t|haven'?t|won'?t|do\s+not|have\s+not|never)"
    r"|\s+(?:probably|maybe|might\s+have)\s+i?\b"
    r"|\s+(?:if|once|after|assuming?|suppose|when|imagine|say|let'?s\s+say)\s+i\b"
    r")"
)

# Completion / hypothetical / negation verb stems. The trailing
# `(?:e?d|ed|ing|s)?` suffix optionally swallows past-tense / progressive
# / 3rd-person forms so e.g. "finished" matches the "finish" stem.
_COMPLETION_VERB = r"(?:have|had|got|gotten|earn(?:ed|ing|s)?|finish(?:ed|ing|es)?|pass(?:ed|ing|es)?|complete(?:d|s|ing)?|do(?:ne|ing|es)?|did|took|taken|take|achieve(?:d|s|ing)?)"
_NEGATION_TRIGGER = r"(?:don'?t|do\s+not|haven'?t|have\s+not|never|won'?t|will\s+not|hasn'?t|has\s+not)"

# Definite-completion verbs
_CLAIM_PAST = re.compile(
    rf"\bi\s+{_COMPLETION_VERB}\s+{_ENTITY_TAIL}",
    re.IGNORECASE,
)
_CLAIM_PRESENT = re.compile(
    rf"\bi\s+(?:now\s+have|already\s+have|currently\s+have|am\s+certified\s+in)\s+{_ENTITY_TAIL}",
    re.IGNORECASE,
)
# Hypothetical -- "if/once/after/assume/suppose I have/had X"
_HYPOTHETICAL = re.compile(
    rf"\b(?:if|once|after|assume|assuming|suppose|imagine|say|let'?s\s+say|when)\s+i\s+(?:have|had|got|get|earn|finish|pass|complete|did|do|take)\s+{_ENTITY_TAIL}",
    re.IGNORECASE,
)
# Uncertainty markers -- "think/believe/guess I have/got X" /
# "(probably|maybe|might have) [I] (have|got|finished) X" /
# "pretty sure I X". The "i" between trigger and verb is optional so
# "maybe I have 310S" matches.
_UNCERTAINTY = re.compile(
    rf"\b(?:"
    rf"i\s+(?:think|believe|guess)\s+i\s+(?:have|got|finish(?:ed)?|complete(?:d)?)"
    rf"|"
    rf"(?:probably|maybe|might\s+have)\s+(?:i\s+)?(?:have|got|finish(?:ed)?)"
    rf"|"
    rf"(?:i'?m\s+)?pretty\s+sure\s+i\s+(?:have|got|finish(?:ed)?|complete(?:d)?)"
    rf")\s+{_ENTITY_TAIL}",
    re.IGNORECASE,
)
# Negation -- "I don't have X", "I haven't got my X", "I never finished my X"
_NEGATION_VERB = re.compile(
    rf"\bi\s+{_NEGATION_TRIGGER}\s+{_COMPLETION_VERB}[\s,]*{_ENTITY_TAIL}",
    re.IGNORECASE,
)
# Negation without explicit "I" -- "actually I don't have X" already
# matches above, but "missing X" / "without X" / "lacking X" don't.
_NEGATION_MISSING = re.compile(
    rf"\b(?:missing|without|lacking|no)\s+{_ENTITY_TAIL}",
    re.IGNORECASE,
)
# Generic remaining-gap requests
_GENERIC_REMAINING = re.compile(
    r"\b(?:what(?:'?s)?\s+(?:else|next|left|other|missing|remaining)|anything\s+else|any\s+other|else\s+do\s+i\s+need|what(?:'?s)?\s+after)\b",
    re.IGNORECASE,
)
# AR-2 cross-detector exclusion. `_GENERIC_REMAINING` will match
# phrases like "what other jobs?" / "any other roles?" / "anything
# else I could do?" -- which are DIFFERENT-ROLE discovery (adjacency
# territory), NOT same-role gap discovery. Step 7 below applies this
# exclusion BEFORE returning an R-3 intent so the adjacency detector
# in chat/adjacent_intent.py can see them.
_DIFFERENT_ROLE_REQUEST = re.compile(
    r"\b(?:what|any)\s+other\s+(?:jobs?|roles?|positions?|openings?|postings?|work|kinds?\s+of\s+work)\b"
    r"|\bother\s+(?:jobs?|roles?|positions?|openings?|postings?)\b"
    r"|\bdifferent\s+(?:jobs?|roles?|positions?)\b"
    r"|\bshow\s+me\s+other\s+(?:jobs?|roles?|positions?)\b"
    r"|\banything\s+else\s+i\s+c(?:ould|an)\s+do\b"
    r"|\broles?\s+like\s+(?:this|that)(?:\s+one)?\b",
    re.IGNORECASE,
)
# Counter-exclusion for the cross-detector lock: a same-role anchor
# ("for this job", "with this role", "about this position") overrides
# the different-role exclusion. "anything else I could do FOR THIS
# JOB?" matches `_DIFFERENT_ROLE_REQUEST` but is unambiguously R-3
# territory because the user explicitly anchors to the current role.
_SAME_ROLE_ANCHOR = re.compile(
    r"\b(?:for|with|about|in|on|toward(?:s)?)\s+"
    r"(?:this|that|the\s+current)\s+"
    r"(?:job|role|position|one|opening|posting)\b",
    re.IGNORECASE,
)
# Counter-counter-exclusion: when the same-role anchor lives INSIDE a
# current-role decline ("I am not interested in this role, what other
# jobs?"; "this role isn't for me, show me other postings"), the user
# is explicitly pivoting AWAY from the current role -- step 7 must
# yield to adjacency. Mirror of `_CURRENT_ROLE_DECLINE_PATTERN` in
# chat/adjacent_intent.py; kept in sync because both detectors apply
# the same compound-semantics rule.
_CURRENT_ROLE_DECLINE_PATTERN = re.compile(
    # Object-style.
    r"\bi(?:\s+(?:don'?t|do not)|\s+am not|'?m not)\s+"
    r"(?:want|like|need|care\s+for|interested\s+in|see\s+myself\s+in)\s+"
    r"(?:any\s+part\s+of\s+)?(?:this|that|the\s+current)\s+"
    r"(?:job|role|position|one|opening|posting)\b"
    r"|"
    # Person-subject form ("I'm not a fit for this role" / "I am not
    # a good fit for the current position").
    r"\bi(?:\s+am|'?m)\s+not\s+a\s+(?:good\s+)?fit\s+for\s+"
    r"(?:this|that|the\s+current)\s+"
    r"(?:job|role|position|one|opening|posting)\b"
    r"|"
    # Role-subject form.
    r"\b(?:this|that|the\s+current)\s+"
    r"(?:job|role|position|one|opening|posting)\s+"
    r"(?:is\s+not|isn'?t|ain'?t)\s+"
    r"(?:for\s+me|a\s+(?:good\s+)?fit|right(?:\s+for\s+me)?|what\s+i\s+want)\b",
    re.IGNORECASE,
)

# Smart-apostrophe normalization for the AR-2 cross-detector patterns.
# Step 7 below applies this before running the three AR-2 patterns so
# "I'm" (smart) and "I'm" (straight) behave identically.
_AR2_APOSTROPHE_NORMALIZER = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
})

# Pending consumption -- affirmative / negative replies. Anchored to the
# *start* of the message because a yes/no buried mid-sentence isn't
# answering the question.
_AFFIRMATIVE = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|yup\b|sure|absolutely|definitely|"
    r"correct|right|that'?s\s+right|i\s+do|i\s+have|i\s+got|"
    r"that'?s\s+correct|that'?s\s+true)\b",
    re.IGNORECASE,
)
_NEGATIVE = re.compile(
    r"^\s*(?:no|nope|nah|not\s+yet|still\s+working|still\s+studying|"
    r"not\s+really|not\s+(?:done|finished)|i\s+don'?t|i\s+haven'?t|"
    r"not\s+quite)\b",
    re.IGNORECASE,
)

# Bare-anaphor markers (resolved against last_discussed_canonical or
# snapshot[0] depending on §2 resolver lifecycle)
_ANAPHOR_TOKENS = re.compile(
    r"\b(?:it|that\s+one|this\s+one|the\s+(?:licence|license|certificate|cert|credential|one))\b",
    re.IGNORECASE,
)


# ============================================================================
# Token helpers (§4.3 deterministic fallback resolver)
# ============================================================================
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _tokens(s: str) -> frozenset[str]:
    """Lowercase, normalise non-alphanumeric runs to spaces, strip,
    split on whitespace, drop generic stop-words. Returns a frozenset
    so callers can use ⊆ directly."""
    if not isinstance(s, str) or not s:
        return frozenset()
    normalized = _NON_ALNUM.sub(" ", s.lower()).strip()
    if not normalized:
        return frozenset()
    return frozenset(normalized.split()) - _GENERIC_CREDENTIAL_TOKENS


def _snapshot_credential_gaps(snapshot: dict | None) -> list[dict[str, str]]:
    """Defensive accessor for `snapshot.lead_job.credential_gaps`. Returns
    [] when snapshot is None or malformed -- callers downstream branch
    on emptiness, not on a missing intermediate."""
    if not isinstance(snapshot, dict):
        return []
    lead = snapshot.get("lead_job")
    if not isinstance(lead, dict):
        return []
    gaps = lead.get("credential_gaps")
    if not isinstance(gaps, list):
        return []
    return [g for g in gaps if isinstance(g, dict)
            and isinstance(g.get("display"), str)
            and isinstance(g.get("canonical"), str)]


def _match_user_ref_by_tokens(
    user_substring: str,
    gaps: list[dict[str, str]],
) -> str | None:
    """§4.3 deterministic token-fallback resolver. Returns the SNAPSHOT'S
    stored canonical when EXACTLY ONE snapshot entry's display tokens
    are a superset of the user reference's non-generic tokens. Zero or
    multiple matches return None (route to clarification at the
    higher-level detector)."""
    user_tokens = _tokens(user_substring)
    if not user_tokens:
        return None
    candidates: list[str] = []
    for g in gaps:
        snap_tokens = _tokens(g["display"])
        if not snap_tokens:
            continue
        if user_tokens.issubset(snap_tokens):
            candidates.append(g["canonical"])
    if len(candidates) == 1:
        return candidates[0]
    return None


def _resolve_user_ref_to_snapshot_canonical(
    user_substring: str,
    snapshot: dict | None,
    registry,
) -> str | None:
    """Map a user-typed credential reference to a snapshot entry.
    Returns the SNAPSHOT'S stored canonical (never a freshly-resolved
    value). §4.2 contract: registry-assisted exact-match first, then
    cross-mode bridge (re-lookup snapshot displays through registry to
    catch Mode-B-snapshot + Mode-A-detection divergence), then
    deterministic token fallback.

    Returns None when no unique snapshot entry matches."""
    gaps = _snapshot_credential_gaps(snapshot)
    if not gaps:
        return None

    # (a) Registry-assisted exact match (Mode A only).
    if registry is not None:
        try:
            hit = registry.lookup(user_substring.strip())
        except Exception:    # pragma: no cover -- defensive
            hit = None
        if hit is not None:
            target = getattr(hit, "canonical_name", None)
            if target:
                # Direct: snapshot canonical equals registry resolution.
                for g in gaps:
                    if g["canonical"] == target:
                        return g["canonical"]
                # Cross-mode bridge: snapshot was captured in Mode B
                # (canonical = normalized display). Re-lookup each
                # snapshot display through the registry; same resolution
                # -> same identity. Returns the SNAPSHOT'S stored
                # canonical, not the registry's.
                for g in gaps:
                    try:
                        snap_hit = registry.lookup(g["display"])
                    except Exception:    # pragma: no cover
                        snap_hit = None
                    if snap_hit is not None and \
                            getattr(snap_hit, "canonical_name", None) == target:
                        return g["canonical"]

    # (b) Deterministic token fallback. Always runs when (a) didn't
    # find a unique candidate.
    return _match_user_ref_by_tokens(user_substring, gaps)


def _resolve_credential_anaphor(
    message: str,
    snapshot: dict | None,
    last_discussed_canonical: str | None,
) -> str | None:
    """Map 'it' / 'that one' / 'the licence' to a snapshot credential
    canonical. Returns None when no anaphor pattern fires or no
    candidate is available.

    Recency rule: if last_discussed_canonical is in the snapshot's gap
    list, prefer it. Otherwise fall back to the snapshot's first
    credential gap (post-match implicit antecedent)."""
    if not _ANAPHOR_TOKENS.search(message or ""):
        return None
    gaps = _snapshot_credential_gaps(snapshot)
    if not gaps:
        return None
    if last_discussed_canonical:
        for g in gaps:
            if g["canonical"] == last_discussed_canonical:
                return last_discussed_canonical
    return gaps[0]["canonical"]


# ============================================================================
# Credential-shape signal (round-12 review boundary)
# ============================================================================
# Words that STRONGLY suggest the user's verb-pattern entity is a
# credential reference. When a verb fires but no unique snapshot entry
# resolves, we ask for clarification ONLY when either (a) the user's
# token set matches multiple snapshot entries (ambiguous reference) or
# (b) the user's substring contains one of these credential-shaped
# words. Ordinary completion verbs over experience / skills / college /
# interview ("I have five years of automotive experience") MUST NOT
# trigger a clarification because nothing in the input signals a
# credential is being discussed.
_CREDENTIAL_SHAPED_TOKENS: frozenset[str] = frozenset({
    "license", "licence", "licenses", "licences",
    "certification", "certifications", "certificate", "certificates",
    "permit", "permits",
    "cert", "credential", "credentials",
    "ticket", "tickets",        # "DZ ticket", "AZ ticket" idioms
    "endorsement", "endorsements",
})


def _is_credential_shaped(entity: str) -> bool:
    """True when the entity text contains generic credential vocabulary.
    Used by the verb-fired-no-resolve confirm path to keep ordinary
    skill / experience / availability / education / interview statements
    out of the clarification flow."""
    if not isinstance(entity, str) or not entity:
        return False
    normalized = _NON_ALNUM.sub(" ", entity.lower()).strip()
    if not normalized:
        return False
    return bool(set(normalized.split()) & _CREDENTIAL_SHAPED_TOKENS)


def _is_ambiguous_against_snapshot(
    user_substring: str,
    gaps: list[dict[str, str]],
) -> bool:
    """True when the substring's non-generic tokens are a subset of
    MULTIPLE snapshot entry displays. Different from 'no match' (zero
    candidates): an ambiguous reference is one the user could mean
    legitimately but we can't tell which without asking."""
    user_tokens = _tokens(user_substring)
    if not user_tokens:
        return False
    candidate_count = 0
    for g in gaps:
        snap_tokens = _tokens(g["display"])
        if not snap_tokens:
            continue
        if user_tokens.issubset(snap_tokens):
            candidate_count += 1
            if candidate_count > 1:
                return True
    return False


# ============================================================================
# Per-category detection helpers
# ============================================================================
_CONJUNCTION_SPLIT = re.compile(r"\s+and\s+|\s*,\s+", re.IGNORECASE)


def _split_entity_conjunction(text: str) -> list[str]:
    """Split a captured entity-tail substring on 'and' / ',' conjunctions.
    Handles multi-credential claims like 'I have 310S and Class G' AND
    list forms like 'I have 310S, Class G, and Smart Serve'. The resolver
    tokenises each part independently, so per-part pre-noun markers
    ('my', 'the') just become generic stop-words and drop out."""
    if not isinstance(text, str):
        return []
    parts = _CONJUNCTION_SPLIT.split(text.strip())
    return [p for p in (s.strip() for s in parts) if p]


def _scan_entity_pattern(
    message: str,
    pattern: re.Pattern,
    snapshot: dict | None,
    registry,
    *,
    last_discussed_canonical: str | None = None,
) -> tuple[list[str], bool, bool]:
    """Run a verb-anchored entity pattern against the full message and
    return (resolved canonicals, did_any_verb_fire, needs_clarification).

    `needs_clarification` is True when at least one captured entity
    didn't resolve to a unique snapshot canonical AND that entity is
    either ambiguous against the snapshot (subset of multiple displays)
    OR credential-shaped (contains license/licence/certificate/permit/...).
    The round-12 boundary: an ordinary `I have five years of automotive
    experience` resolves to NOTHING, IS NOT ambiguous (zero snapshot
    matches), IS NOT credential-shaped -- so the detector falls
    through to None instead of emitting an inappropriate confirm.
    """
    out: list[str] = []
    seen: set[str] = set()
    verb_fired = False
    needs_clarification = False
    gaps = _snapshot_credential_gaps(snapshot)
    for m in pattern.finditer(message):
        verb_fired = True
        entity = (m.group(1) or "").strip()
        if not entity:
            continue
        # Multi-credential split: "310S and Class G" -> ["310S", "Class G"]
        for part in _split_entity_conjunction(entity):
            # Anaphor short-circuit: "it" / "that one" / "the licence"
            # alone resolves to last_discussed or snapshot[0].
            canonical = _resolve_credential_anaphor(
                part, snapshot, last_discussed_canonical,
            ) or _resolve_user_ref_to_snapshot_canonical(
                part, snapshot, registry,
            )
            if canonical:
                if canonical not in seen:
                    seen.add(canonical)
                    out.append(canonical)
            else:
                # Did NOT resolve. Round-13 boundary: confirm ONLY when
                # the reference is either
                #   (a) ambiguous against the snapshot -- subset of
                #       MULTIPLE snapshot displays ("G2" vs G2-driver
                #       and G2-paramedic), or
                #   (b) made of nothing but generic credential vocabulary
                #       ("the licence", "my permit", "my cert") --
                #       distinguishing tokens, if any, would have to
                #       carry a non-credential semantic meaning, in
                #       which case the user is talking about something
                #       else ("work permit" -> immigration; "parking
                #       ticket" -> not a credential).
                #
                # Phrases with distinguishing tokens that match zero
                # snapshot entries ("work permit", "support ticket",
                # "college certificate") return None and route through
                # the standard planner.
                is_generic_only = (
                    _is_credential_shaped(part)
                    and not _tokens(part)
                )
                if (
                    _is_ambiguous_against_snapshot(part, gaps)
                    or is_generic_only
                ):
                    needs_clarification = True
    return out, verb_fired, needs_clarification


# ============================================================================
# Main detector
# ============================================================================
def detect_remaining_gaps_intent(
    message: str,
    snapshot: dict | None,
    registry,
    *,
    accumulated_credentials: list[dict[str, Any]],
    pending_confirmation: dict[str, Any] | None,
    last_discussed_canonical: str | None,
) -> RemainingGapsIntent | None:
    """Returns the detection result. None = no remaining-gaps pattern
    matched; the handler routes through normal planner/router/engine.

    The detector is PURE: it does NOT mutate any of its arguments
    (`accumulated_credentials`, `pending_confirmation`,
    `last_discussed_canonical` are read-only). The handler owns
    StagedProfile writes -- see docs/remaining-gaps-design.md §2.

    Detection ordering (first-match wins):
      1. Pending consumption (yes / no / unrelated against
         pending_confirmation)
      2. Retraction against accumulated_credentials (any explicit
         negation pattern targeting an already-assumed entity)
      3. Standard negation for non-accumulated entities (returns None)
      4. Uncertainty markers (kind="confirm" with pending_action="add")
      5. Explicit completion (kind="subtract" mode="claimed")
      6. Explicit hypothetical (kind="subtract" mode="hypothetical")
      7. Generic remaining (kind="subtract" with empty claims when
         snapshot exists, kind="bootstrap" otherwise)
    """
    if not isinstance(message, str) or not message.strip():
        return None
    msg = message.strip()
    accumulated_canonicals = {
        a["canonical"] for a in (accumulated_credentials or [])
        if isinstance(a, dict) and isinstance(a.get("canonical"), str)
    }
    snapshot_canonicals = {
        g["canonical"] for g in _snapshot_credential_gaps(snapshot)
    }

    # ----- STEP 1: pending consumption -----
    # The handler cleared `staged.pending_credential_confirmation` BEFORE
    # the detector call (see docs/remaining-gaps-design.md §2 clearing
    # ownership). The dict passed in here is the SAVED COPY -- a value,
    # not a live reference. The detector must NOT mutate it.
    #
    # IDENTITY GUARD (round-11 finding 3): the pending canonical MUST
    # exist in the current snapshot. If it doesn't (a forged cookie
    # path, or a snapshot transition that escaped the clearing rules),
    # treat pending as malformed and fall through to fresh detection.
    # Without this, "yes" against a stale pending {canonical: outside}
    # would emit a subtract claim for a value not anchored to any
    # current snapshot entry -- violating the §4.0 invariant.
    if isinstance(pending_confirmation, dict):
        pending_canonical = pending_confirmation.get("canonical")
        pending_action = pending_confirmation.get("action")
        if (
            isinstance(pending_canonical, str)
            and pending_canonical
            and pending_action in {"add", "remove"}
            and pending_canonical in snapshot_canonicals
        ):
            # Anchored affirmative -> execute the pending action.
            if _AFFIRMATIVE.match(msg):
                if pending_action == "add":
                    return RemainingGapsIntent(
                        kind="subtract",
                        current_turn_claims=(
                            CredentialClaim(
                                canonical=pending_canonical,
                                mode="claimed",
                            ),
                        ),
                    )
                # action == "remove"
                return RemainingGapsIntent(
                    kind="retract",
                    retract_canonical=pending_canonical,
                )
            # Anchored negative -> kind=None (handler already cleared
            # pending; nothing more to do).
            if _NEGATIVE.match(msg):
                return None
            # Anything else: pending falls through; the user has moved
            # on. The handler's save-and-clear took the pending field
            # off StagedProfile before this call, so re-running
            # detection from scratch on the new message is safe.

    # ----- STEP 2: retraction against accumulated state -----
    # Any explicit negation pattern (verb form OR missing/without/lacking
    # form) targeting a credential that's currently in
    # accumulated_credentials triggers a `kind="confirm"` with
    # pending_action="remove". The "actually" hedge is NOT required.
    if accumulated_canonicals:
        for pat in (_NEGATION_VERB, _NEGATION_MISSING):
            negated, _, _ = _scan_entity_pattern(
                msg, pat, snapshot, registry,
                last_discussed_canonical=last_discussed_canonical,
            )
            for canonical in negated:
                if canonical in accumulated_canonicals:
                    display = _display_for_canonical(snapshot, canonical) or canonical
                    return RemainingGapsIntent(
                        kind="confirm",
                        confirmation_target_canonical=canonical,
                        confirmation_target_display=display,
                        pending_action="remove",
                    )

    # ----- STEP 3: standard negation for non-accumulated entities -----
    # Any negation pattern that fires for an entity NOT in accumulated
    # is a normal "I don't have X" -- the standard planner handles it.
    # We must NOT proceed to steps 4-7 in that case (a negation followed
    # by a completion-shaped clause should not silently subtract the
    # other entity); return kind=None and let normal routing run.
    has_explicit_negation = bool(
        _NEGATION_VERB.search(msg) or _NEGATION_MISSING.search(msg)
    )
    if has_explicit_negation:
        return None

    # ----- STEP 4-6: completion / hypothetical / uncertainty collection -----
    # All three verb-pattern scanners run unconditionally; we resolve
    # the precedence among them after collection. This shape handles
    # the round-11 finding 1 cases:
    #   - "I have 310S and Class G"            -> two completions (per-pattern split)
    #   - "I have my 310S and I think I have my G2"
    #                                          -> completion(310S) + uncertain(G2)
    #                                          -> definite claim wins for 310S,
    #                                             G2 uncertainty dropped
    #   - "I think I have my 310S"             -> completion AND uncertain for 310S
    #                                          -> uncertain wins for same canonical
    #   - "if I had 310S"                      -> completion AND hypothetical for 310S
    #                                          -> hypothetical wins for same canonical
    completed: list[str] = []
    completed_needs_clarif = False
    for pat in (_CLAIM_PRESENT, _CLAIM_PAST):
        canonicals, _fired, needs = _scan_entity_pattern(
            msg, pat, snapshot, registry,
            last_discussed_canonical=last_discussed_canonical,
        )
        completed_needs_clarif = completed_needs_clarif or needs
        for c in canonicals:
            if c not in completed:
                completed.append(c)

    hypothetical, _hypothetical_fired, hypothetical_needs_clarif = (
        _scan_entity_pattern(
            msg, _HYPOTHETICAL, snapshot, registry,
            last_discussed_canonical=last_discussed_canonical,
        )
    )

    uncertain, _uncertain_fired, uncertain_needs_clarif = (
        _scan_entity_pattern(
            msg, _UNCERTAINTY, snapshot, registry,
            last_discussed_canonical=last_discussed_canonical,
        )
    )

    # Per-canonical precedence:
    #   hypothetical wins over completion for the SAME canonical
    #     ("I have X" sub-match inside "if I have X" is hypothetical)
    #   uncertain wins over completion for the SAME canonical
    #     ("I have X" sub-match inside "I think I have X" is uncertain)
    # `completed_only` is the set of canonicals with a TRULY-definite
    # claim -- no hypothetical or uncertain overlap on the same entity.
    uncertain_set = set(uncertain)
    hypothetical_set = set(hypothetical)
    completed_only = [
        c for c in completed
        if c not in hypothetical_set and c not in uncertain_set
    ]

    # Round-11 finding 1 contract: a definite claim in the message wins
    # over any pure uncertainty about a DIFFERENT entity. The user is
    # clearly committing to at least one credential; uncertainty about
    # others is recoverable on the next turn and shouldn't block the
    # definite signal.
    if completed_only or hypothetical:
        claims_list: list[CredentialClaim] = []
        for c in completed_only:
            claims_list.append(CredentialClaim(canonical=c, mode="claimed"))
        for c in hypothetical:
            if c not in {x.canonical for x in claims_list}:
                claims_list.append(
                    CredentialClaim(canonical=c, mode="hypothetical"),
                )
        return RemainingGapsIntent(
            kind="subtract",
            current_turn_claims=tuple(claims_list),
        )

    # No definite signal -- a pure uncertainty (no overlapping definite
    # claim for the same canonical anywhere) emits confirm. This is the
    # "I think I have X" single-entity case.
    if uncertain:
        canonical = uncertain[0]
        display = _display_for_canonical(snapshot, canonical) or canonical
        return RemainingGapsIntent(
            kind="confirm",
            confirmation_target_canonical=canonical,
            confirmation_target_display=display,
            pending_action="add",
        )

    # Verb fired but no canonical resolved -- emit confirm ONLY when
    # the unresolved reference is credential-shaped or ambiguous (i.e.,
    # `_scan_entity_pattern` flagged needs_clarification). Ordinary
    # completion sentences over experience / skills / availability /
    # education / interview ("I have five years of automotive
    # experience") flow through to None and route via the standard
    # planner. Round-12 boundary.
    if (
        (completed_needs_clarif
         or hypothetical_needs_clarif
         or uncertain_needs_clarif)
        and snapshot_canonicals
    ):
        return RemainingGapsIntent(
            kind="confirm",
            confirmation_target_canonical=None,
            confirmation_target_display="",
            pending_action="add",
        )

    # ----- STEP 7: generic remaining -----
    if _GENERIC_REMAINING.search(msg):
        # AR-2 lock: explicit different-role discovery ("what other
        # jobs?", "any other roles?", "anything else I could do?")
        # belongs to the adjacency detector, NOT to R-3's
        # bootstrap/subtract path. Returning None here lets the
        # downstream adjacency hook (AR-6) see the message.
        #
        # COUNTER-EXCLUSION: a same-role anchor ("for this job",
        # "with this role") keeps R-3 in charge -- UNLESS that anchor
        # appears inside a current-role decline ("I am not interested
        # in this role, what other jobs?"), in which case the user
        # is pivoting away from the current role and R-3 yields.
        #
        # Smart-apostrophe fold so "I'm" (mobile-autocorrect U+2019)
        # behaves identically to "I'm" (straight apostrophe) for the
        # AR-2 patterns. Doesn't affect any other R-3 pattern.
        ar2_msg = msg.translate(_AR2_APOSTROPHE_NORMALIZER)
        if _DIFFERENT_ROLE_REQUEST.search(ar2_msg):
            anchored = _SAME_ROLE_ANCHOR.search(ar2_msg)
            pivoting_away = _CURRENT_ROLE_DECLINE_PATTERN.search(ar2_msg)
            if not anchored or pivoting_away:
                return None
        if snapshot is None or not _snapshot_credential_gaps(snapshot):
            return RemainingGapsIntent(kind="bootstrap")
        return RemainingGapsIntent(
            kind="subtract",
            current_turn_claims=(),
        )

    return None


# ============================================================================
# Internal helper -- display string for a snapshot canonical
# ============================================================================
def _display_for_canonical(snapshot: dict | None, canonical: str) -> str | None:
    for g in _snapshot_credential_gaps(snapshot):
        if g["canonical"] == canonical:
            return g["display"]
    return None
