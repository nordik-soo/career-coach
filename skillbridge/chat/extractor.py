"""Evidence-bound profile extraction.

The LLM is asked to extract structured profile data from a single user
message, AND for each value it must return the verbatim phrase from the
user message that grounds it. We then validate that the phrase actually
appears in the message (case-insensitive substring). Anything ungrounded
is dropped before it touches StagedProfile.

This is the single defense between an enthusiastic Haiku and our database.
Hard rules:
  - Skills the user did not mention -> dropped.
  - Fields with no grounding phrase -> dropped.
  - Fields whose grounding phrase does not occur in the message -> dropped.
  - "Decline" signals from the user populate declined_slots so we never
    re-ask. We do this from a tiny rule-based pass — not the LLM — to
    keep declines deterministic.

The extractor never speaks; that's the responder's job. It only fills
slots from evidence.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from skillbridge.chat.prompts import EVIDENCE_BOUND_EXTRACTOR_PROMPT
from skillbridge.extract.base import ExtractedSkill, resolve_many
from skillbridge.llm import call_json, is_enabled

log = logging.getLogger(__name__)


# Slots we accept from the LLM. Must match StagedProfile.merge_fields().
_EXTRACTABLE_SLOTS: frozenset[str] = frozenset({
    "preferred_location", "target_role_text", "education_text",
    "experience_text", "skills_text", "work_type_preference",
    "language_preferences",
    "salary_expectation_text", "shift_preference",
    "transportation_text", "availability_text",
})

_WORK_TYPE_VALUES: frozenset[str] = frozenset({
    "full-time", "full time", "part-time", "part time",
    "flexible", "remote", "casual", "contract", "seasonal",
    "unknown",
})

_SHIFT_VALUES: frozenset[str] = frozenset({
    "day", "days", "day shift", "evening", "evenings", "evening shift",
    "night", "nights", "night shift", "weekends", "weekend",
    "any", "flexible", "rotating",
})


# Decline / refuse-to-answer signals — keyed by slot. Pure regex; no LLM.
# These cause us to mark the slot as declined and never ask again.
_DECLINE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "salary_expectation_text": (
        re.compile(r"\b(prefer not (to )?say|don'?t want to (say|share)|rather not say)\b.*\b(salary|wage|pay|money)\b", re.I),
        re.compile(r"\b(salary|wage|pay)\b.*\b(prefer not|don'?t want|skip|whatever|doesn'?t matter)\b", re.I),
        re.compile(r"\bwhatever (the )?pay\b", re.I),
    ),
    "shift_preference": (
        re.compile(r"\b(any shift|any time|whenever|flexible on shift)\b", re.I),
    ),
    "transportation_text": (
        re.compile(r"\b(don'?t want to say|prefer not to say)\b.*\b(car|transport|drive)\b", re.I),
    ),
    "availability_text": (
        re.compile(r"\b(not sure|don'?t know|prefer not to say)\b.*\b(when|available|start)\b", re.I),
    ),
}

# Generic "I don't want to answer this" detector. Used for slots the
# assistant has just explicitly asked about.
_BLANKET_DECLINE = re.compile(
    r"\b(skip( that)?|next question|don'?t want to (say|answer)|rather not|prefer not to (say|answer)|pass)\b",
    re.I,
)

# Learning / training / aspiration patterns. When the user's message matches
# any of these, ALL extracted skills are dropped — mentioning a skill in
# the context of wanting to learn it is the opposite of claiming to have
# it, and the matcher would invert the truth otherwise.
#
# Also catches "Show me training for X" / "Where can I learn X" — the
# kind of follow-up questions the assistant's own NextSkillHint surfaces
# after presenting matches. The frontend renders those as click-through
# prompts, so the user's "click" lands as an apparent skill claim.
_LEARNING_QUESTION_PATTERNS = (
    re.compile(r"\b(where|how|what)\b.*\b(can|do|should)\b.*\b(i|to)\b.*\b(learn|study|train|get|build|find)\b", re.I),
    re.compile(r"\bshow me (training|courses?|classes?|programs?)\b", re.I),
    re.compile(r"\b(training|courses?|classes?|programs?)\b.*\b(for|on|in|about)\b", re.I),
    re.compile(r"\bi (want|need|would like) to (learn|build|develop|get|earn|acquire)\b", re.I),
    re.compile(r"\bwhere (can|do) i (learn|study|train|find training|find courses?)\b", re.I),
    re.compile(r"\bhow (can|do) i (get|earn|build|develop|learn)\b", re.I),
    re.compile(r"\bcan you (recommend|suggest) (a |any )?(course|training|program|class)\b", re.I),
    # "What courses teach X?" / "Which programs cover X?" — the verb
    # ("teach", "cover", "include") follows the noun.
    re.compile(r"\b(what|which)\b.*\b(courses?|classes?|programs?|training)\b.*\b(teach|cover|include|offer)\b", re.I),
    # "Tell me about training/courses/programs"
    re.compile(r"\btell me about (training|courses?|classes?|programs?|certifications?)\b", re.I),
    # "Any courses for X?" / "Got training for X?"
    re.compile(r"\b(any|got|have) (training|courses?|classes?|programs?)\b", re.I),
)


def _is_learning_question(message: str) -> bool:
    """True when the message is the user asking about training/learning a skill.

    This guards against extracting "driving" as a skill from
    "Where can I learn driving locally?" — the verbatim word is there
    but the user is explicitly stating they DON'T have it yet. Lets the
    handler drop extracted skills before they touch staged.skills.
    """
    return any(p.search(message) for p in _LEARNING_QUESTION_PATTERNS)


# --------------------------------------------------------------------------
# Bug A — credential patch (2026-06-15)
# --------------------------------------------------------------------------
#
# Locked allowlist of credential phrases the LLM extractor commonly routes
# into transportation_text or omits entirely. Each entry is
# (regex, canonical_name). The regex uses word boundaries (case-insensitive)
# and `\bclass\s*g\b` correctly does NOT match "Class G2" / "Class G1"
# because no word boundary exists between `g` and `2` / `1`. The canonical
# names are chosen to canonicalize (via skillbridge.match.alignment) to the
# same form the JD-side extractor produces on its end — so the matcher
# scores them as a match without further changes.
#
# Scope guard: this list is intentionally small. Every entry is a
# multi-token phrase, a numeric trade code, or a four-letter regulator
# acronym — all of which are unambiguous in normal English. Adding entries
# requires the same sign-off as expanding the registry: a real local
# credential, no false-positive surface, sign-off from the lead engineer.
_CREDENTIAL_ALLOWLIST: tuple[tuple[re.Pattern[str], str], ...] = (
    # Driver's licences (Ontario)
    (re.compile(r"\bclass\s*g\b", re.I), "Class G license"),
    (re.compile(r"\bclass\s*a\b", re.I), "Class A license"),
    (re.compile(r"\bclass\s*d\b", re.I), "Class D license"),
    (re.compile(r"\bclass\s*z\b", re.I), "Class Z endorsement"),
    (re.compile(r"\baz\s+licen[cs]e\b", re.I), "AZ license"),
    (re.compile(r"\bdz\s+licen[cs]e\b", re.I), "DZ license"),
    # Ontario trade certificates of qualification.
    (re.compile(r"\b310\s*t\b", re.I), "310T technician certification"),
    (re.compile(r"\b310\s*s\b", re.I), "310S automotive technician certification"),
    # Care / safety tickets.
    (re.compile(r"\bpersonal\s+support\s+worker\b", re.I), "Personal Support Worker certification"),
    (re.compile(r"\bpsw\s+(certificate|cert|certification)\b", re.I), "PSW certificate"),
    (re.compile(r"\bwhmis\b", re.I), "WHMIS"),
    (re.compile(r"\bfirst\s+aid\b", re.I), "first aid"),
    (re.compile(r"\bfood\s+handler\b", re.I), "food handler"),
    (re.compile(r"\bforklift\s+(certification|certificate|licen[cs]e|ticket)\b", re.I), "forklift certification"),
)


# Negation cues that appear BEFORE a credential phrase when the user is
# stating they DON'T hold it. Window is the 25 characters immediately
# preceding the match — enough to catch "I don't have my Class G" and
# "no Class G license" without dragging in earlier-sentence content.
_CREDENTIAL_NEGATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(no|without|missing|lack|need)\b", re.I),
    re.compile(r"\b(don'?t|do\s+not|doesn'?t|haven'?t|not)\s+(have|hold|got|gotten)\b", re.I),
    re.compile(r"\bnot\s+(yet|currently)\b", re.I),
)


def _credential_in_have_context(message: str, match: re.Match[str]) -> bool:
    """True when the credential phrase appears in a HAVE context (no
    negation cue in the 25 chars immediately before the match)."""
    pre_start = max(0, match.start() - 25)
    pre_window = message[pre_start:match.start()]
    for pattern in _CREDENTIAL_NEGATION_PATTERNS:
        if pattern.search(pre_window):
            return False
    return True


def _patch_credentials(
    message: str, skills: list[ExtractedSkill],
) -> tuple[list[ExtractedSkill], list[str]]:
    """Scan the user's message for credential phrases from
    `_CREDENTIAL_ALLOWLIST` and add any HAVE-context credential that the
    LLM didn't already extract.

    Returns (possibly-extended skills list, list of canonical names added).
    The second element is for telemetry — `raw_keys_dropped` is the only
    debug surface the extractor exposes, so additions are logged there
    with the `credential_patched:` prefix.

    Idempotent: callers may invoke this on already-patched results
    without producing duplicates (the existing-name check is by
    case-insensitive canonical-name comparison).
    """
    if not message:
        return skills, []

    existing_norm: set[str] = set()
    for s in skills:
        if isinstance(s.skill_name, str) and s.skill_name.strip():
            existing_norm.add(s.skill_name.strip().lower())

    added_names: list[str] = []
    new_skills: list[ExtractedSkill] = list(skills)

    for pattern, canonical in _CREDENTIAL_ALLOWLIST:
        if canonical.lower() in existing_norm:
            continue
        m = pattern.search(message)
        if m is None:
            continue
        if not _credential_in_have_context(message, m):
            continue
        new_skills.append(ExtractedSkill(
            skill_name=canonical,
            raw_phrase=m.group(0),
            confidence=0.85,
        ))
        existing_norm.add(canonical.lower())
        added_names.append(canonical)

    if added_names:
        new_skills = resolve_many(new_skills)

    return new_skills, added_names


@dataclass
class ExtractionResult:
    fields: dict[str, Any]
    skills: list[ExtractedSkill]
    declined: list[str]
    off_topic: bool   # extractor returned no fields AND no skills AND no decline
    raw_keys_dropped: list[str]  # for logging only; never user-facing


# =========================================================================
# Public entry point
# =========================================================================
def extract(message: str, *, asked_slots: list[str] | None = None) -> ExtractionResult:
    """Run evidence-bound extraction on one user message.

    asked_slots is the list of slots the assistant most recently asked about.
    If the user's reply matches a blanket-decline pattern, those slots get
    added to declined_slots even without slot-specific decline phrasing.

    LEARNING-QUESTION GUARD: when the user is asking about LEARNING /
    TRAINING / COURSES (e.g. "Where can I learn driving?"), drop any
    extracted skills before they touch the staged profile. The user
    mentioning a skill name in a learning context is the opposite of
    claiming to have it — adding it to skills would invert the matcher.
    """
    asked = list(asked_slots or [])

    fields, skills, dropped = _llm_extract(message)

    # CREDENTIAL PATCH (Bug A fix, 2026-06-15): deterministic safety net.
    # The LLM extractor's prompt lists "no driver's license" as an example
    # for transportation_text, which leads it to route Class G / commercial-
    # licence / trade-credential mentions into that slot instead of skills[].
    # When the user has other genuine skills in the same message, the
    # rule-based fallback in the handler doesn't fire, and the credential
    # never reaches StagedSkill. This patch scans the message for a
    # locked credential allowlist and adds any HAVE-context credential the
    # LLM missed. The learning-question guard below still wins.
    skills, patched_names = _patch_credentials(message, skills)
    if patched_names:
        dropped.extend(f"credential_patched:{n}" for n in patched_names)

    declined = _detect_declines(message, asked, fields)

    # Strip any slot we treat as declined from the field payload — declines
    # win over re-extraction.
    for slot in declined:
        fields.pop(slot, None)

    # Backend-side guard against the LLM extracting skills from learning /
    # training inquiries. The prompt asks Haiku to return empty for these,
    # but this regex is a deterministic safety net so a single Haiku
    # slip-up doesn't corrupt the user's profile.
    if _is_learning_question(message) and skills:
        dropped.extend(f"learning_question_skill:{s.skill_name}" for s in skills)
        skills = []

    off_topic = (not fields and not skills and not declined)

    return ExtractionResult(
        fields=fields,
        skills=skills,
        declined=declined,
        off_topic=off_topic,
        raw_keys_dropped=dropped,
    )


# =========================================================================
# LLM call + evidence validation
# =========================================================================
def _llm_extract(message: str) -> tuple[dict[str, Any], list[ExtractedSkill], list[str]]:
    """Call Haiku for grounded JSON, then drop anything ungrounded.

    LLM-disabled fallback: empty result. The responder will then ask a
    follow-up question (and skills get a second chance via the rule-based
    extractor in the handler).
    """
    dropped: list[str] = []
    if not is_enabled():
        return {}, [], dropped

    payload = call_json(EVIDENCE_BOUND_EXTRACTOR_PROMPT, message, max_tokens=600)
    if not isinstance(payload, dict):
        return {}, [], dropped

    msg_lower = message.lower()

    fields: dict[str, Any] = {}
    raw_fields = payload.get("fields")
    if isinstance(raw_fields, dict):
        for slot, entry in raw_fields.items():
            if slot not in _EXTRACTABLE_SLOTS:
                dropped.append(f"unknown_slot:{slot}")
                continue
            if not isinstance(entry, dict):
                dropped.append(f"bad_shape:{slot}")
                continue
            value = entry.get("value")
            evidence = entry.get("evidence")
            if not value or not evidence:
                dropped.append(f"missing_value_or_evidence:{slot}")
                continue
            # Evidence must be a verbatim substring of the user's message.
            if not _is_grounded(str(evidence), msg_lower, slot=slot):
                dropped.append(f"ungrounded:{slot}")
                continue

            normalized = _normalize_field_value(slot, value)
            if normalized is None:
                dropped.append(f"invalid_value:{slot}")
                continue
            fields[slot] = normalized

    skills: list[ExtractedSkill] = []
    raw_skills = payload.get("skills")
    if isinstance(raw_skills, list):
        for s in raw_skills:
            if not isinstance(s, dict):
                continue
            name = (s.get("name") or "").strip()
            evidence = s.get("evidence")
            if not name or not evidence:
                dropped.append(f"skill_missing_evidence:{name or '?'}")
                continue
            if not _is_grounded(str(evidence), msg_lower, skill_name=name):
                dropped.append(f"skill_ungrounded:{name}")
                continue
            try:
                conf = float(s.get("confidence", 0.7))
            except (TypeError, ValueError):
                conf = 0.7
            skills.append(ExtractedSkill(
                skill_name=name,
                raw_phrase=str(evidence),
                confidence=max(0.0, min(1.0, conf)),
            ))
    skills = resolve_many(skills)

    return fields, skills, dropped


# AR-8d: short technical-skill tokens that the >=4-char grounding
# floor would otherwise drop. Locked allowlist (per reviewer):
#   SQL, API, Git, C#, C++, JS, TS
# Deliberately excluded:
#   R       -- ordinary-word false positives ("R" as letter, "R"
#              as in "R rated", "Mister R", etc.)
#   Go      -- "go to", "let's go", "Go Lions" -- too ambiguous.
#   AI      -- "AI" in many non-skill contexts ("I'd like to ai...").
#   UI      -- "you ai" boundary collisions, "United ..." prefixes.
#   CD      -- "CD player", "ID/CD", "co-CD", etc.
# JavaScript and TypeScript (full forms) are >=4 chars and pass the
# standard substring check directly; no allowlist entry needed.
_APPROVED_SHORT_TECHNICAL_TOKENS: frozenset[str] = frozenset({
    "sql", "api", "git", "c#", "c++", "js", "ts",
})


def _is_approved_short_technical_token(
    name: str, evidence: str, msg_norm: str,
) -> bool:
    """AR-8d: controlled escape for short technical-skill tokens
    that the >=4-char grounding floor would otherwise drop.

    Validity rules (all required):
      1. `name` and `evidence` normalize (strip + lowercase) to the
         SAME approved token. The LLM can't canonicalize "JS" ->
         "JavaScript" on the name side while only quoting "JS" as
         evidence -- both sides must agree on which token the user
         actually wrote.
      2. The token occurs in the (already typographically-
         normalized) `msg_norm` with non-word-character boundaries
         on both sides. Standard `\\b` doesn't anchor C++/C#
         reliably (the `+` and `#` are non-word chars), so the
         check uses explicit `(?<!\\w)` / `(?!\\w)` lookarounds.
         `\\w` covers letters (incl. Unicode), digits, AND
         underscore -- so `API_v2`, `SQL_mode`, `C++_library`,
         `C#_service`, etc. are all correctly rejected, as are
         tokens adjacent to Unicode letters.

    Returns True iff the LLM's claim is grounded in the message.
    """
    if not isinstance(name, str) or not isinstance(evidence, str):
        return False
    norm_name = name.strip().lower()
    norm_evidence = evidence.strip().lower()
    if norm_name not in _APPROVED_SHORT_TECHNICAL_TOKENS:
        return False
    if norm_evidence != norm_name:
        return False
    pattern = (
        r"(?<!\w)"
        + re.escape(norm_name)
        + r"(?!\w)"
    )
    return re.search(pattern, msg_norm, re.I) is not None


def _is_grounded(
    evidence: str,
    message_lower: str,
    *,
    slot: str | None = None,
    skill_name: str | None = None,
) -> bool:
    """Evidence must appear (case-insensitive) in the user's message
    AFTER typographic normalization.

    We require >=4 chars to avoid trivial overlaps (e.g. "I", "the").
    A short ground phrase is almost always a hallucination shortcut.

    The normalization (see `resume.extract._normalize_for_grounding`)
    is shared with the resume extractor so a smart-quote in the user's
    chat input vs an ASCII apostrophe in the LLM's evidence (or vice
    versa) doesn't reject otherwise-valid grounding. Closed-vocabulary
    slot escapes ("day", "ft", "pt") run on the normalized message too.

    AR-8d: controlled allowlist escape for short technical-skill
    tokens (SQL, API, Git, C#, C++, JS, TS). Fires only on skill-
    grounding calls (slot is None and skill_name is provided), and
    only when name + evidence normalize to the SAME approved token
    that actually occurs in the message with `\\w`-class word
    boundaries on both sides. See `_is_approved_short_technical_token`.
    """
    # Local import to avoid a top-of-file dependency that would also
    # pull `extract`'s module-level config; the symbol is stable.
    from skillbridge.resume.extract import _normalize_for_grounding

    ev = _normalize_for_grounding(evidence.lower())
    msg_norm = _normalize_for_grounding(message_lower)
    if len(ev) < 4:
        # Closed-vocabulary replies are often short but meaningful:
        # "day", "ft", "pt". Allow them only for the slots where a
        # one-word answer is expected, and only as a whole word.
        if slot in {"shift_preference", "work_type_preference"} and re.search(
            rf"\b{re.escape(ev)}\b", msg_norm
        ):
            return True
        # AR-8d: controlled short technical-token escape. Only runs
        # on skill-grounding calls (slot is None AND skill_name is
        # provided). Slot escapes above remain authoritative for
        # closed-vocabulary slots.
        if (
            slot is None
            and skill_name is not None
            and _is_approved_short_technical_token(
                skill_name, evidence, msg_norm,
            )
        ):
            return True
        return False
    return ev in msg_norm


def _normalize_field_value(slot: str, value: Any) -> Any:
    """Light validation for closed-vocabulary fields. Open text passes through."""
    if slot == "language_preferences":
        if isinstance(value, list):
            cleaned = [str(v).strip() for v in value if str(v).strip()]
            return cleaned or None
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return None
    if slot == "work_type_preference":
        v = str(value).strip().lower()
        if v in {"ft", "fulltime"}:
            return "full-time"
        if v in {"pt", "parttime"}:
            return "part-time"
        if v in _WORK_TYPE_VALUES:
            return v.replace(" ", "-") if "-" not in v and " " in v else v
        # Allow unknown free-text; caller already required user evidence.
        return v or None
    if slot == "shift_preference":
        v = str(value).strip().lower()
        if v == "day":
            return "days"
        if v == "evening":
            return "evenings"
        if v == "night":
            return "nights"
        if v == "weekend":
            return "weekends"
        if v in _SHIFT_VALUES or v:
            return v
        return None
    if isinstance(value, str):
        v = value.strip()
        return v or None
    return value


# =========================================================================
# Decline detection (deterministic; no LLM)
# =========================================================================
def _detect_declines(message: str, asked_slots: list[str],
                     extracted_fields: dict[str, Any]) -> list[str]:
    """Return the slots the user has explicitly declined to answer.

    Two signals:
      1. Slot-specific patterns (e.g., "rather not say my salary").
      2. Blanket decline ("skip that", "next question") applied to whichever
         slots were most recently asked AND not just filled in this turn.
    """
    declined: list[str] = []

    for slot, patterns in _DECLINE_PATTERNS.items():
        for pat in patterns:
            if pat.search(message):
                declined.append(slot)
                break

    if _BLANKET_DECLINE.search(message):
        for slot in asked_slots:
            if slot in extracted_fields:
                continue
            if slot in declined:
                continue
            declined.append(slot)

    return declined
