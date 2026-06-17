"""Resume facts extractor.

Input: extracted resume text (from skillbridge.resume.parse).
Output: structured `resume_facts_json` per docs/resume-design.md §2.

Hard rules (per the design doc):

  - Evidence-bound. Every leaf fact has a verbatim phrase from the resume
    text (case-insensitive substring, >=4 chars). Anything ungrounded is
    dropped before it appears in the output.
  - No contact info. Name / phone / email / address / social links are
    never extracted. The prompt forbids them and the backend strips them
    if the LLM returns them anyway (defense in depth).
  - Confidence floor (MIN_EXTRACTION_CONFIDENCE = 0.60) is enforced at
    the DERIVE layer, not here. Below-floor skills stay in
    resume_facts_json for transparency; they just don't feed matching.
  - Stable fact_ids are assigned backend-side, not by the LLM. The LLM
    has no business inventing IDs.
  - summary_signals is computed by backend math (year arithmetic +
    keyword classification), not by the LLM.

This module never speaks. The responder narrates the parsed facts during
the RESUME_REVIEW turn.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from skillbridge.llm import call_json, is_enabled
from skillbridge.versions import EXTRACTOR_VERSION_RESUME

log = logging.getLogger(__name__)


# Top-level keys we expect under "facts" in the LLM output.
_FACT_GROUPS = ("skills", "work_history", "education", "certifications", "projects")

# Defense in depth: even if the LLM returns these fields against the prompt's
# explicit forbidance, the backend drops them. They are PII and never used
# for matching.
_CONTACT_KEYS: frozenset[str] = frozenset({
    "name", "first_name", "last_name", "full_name",
    "email", "phone", "phone_number", "address", "city", "postal_code",
    "linkedin", "github", "website", "twitter", "social",
})


@dataclass
class ResumeExtractionResult:
    """Output of extract_resume_facts().

    `facts` is always a valid resume_facts_json shape (possibly with
    empty lists). `parse_warning` carries the reason when extraction
    didn't produce useful facts:
      - "text_too_short"   → resume text was less than 50 chars
      - "llm_disabled"     → LLM_ENABLED=false; no extraction attempted
      - "llm_empty"        → LLM returned non-dict / null
      - "empty_extraction" → LLM ran but every fact was dropped
    """
    facts: dict[str, Any]
    raw_keys_dropped: list[str]
    parse_warning: str | None = None


# =========================================================================
# Public entrypoint
# =========================================================================
def extract_resume_facts(resume_text: str) -> ResumeExtractionResult:
    """Run evidence-bound extraction on resume text.

    Always returns a ResumeExtractionResult. Never raises on malformed
    LLM output — anything weird becomes a dropped reason in
    raw_keys_dropped + a parse_warning on the result.
    """
    if not resume_text or len(resume_text.strip()) < 50:
        return ResumeExtractionResult(
            facts=_empty_facts(),
            raw_keys_dropped=[],
            parse_warning="text_too_short",
        )

    if not is_enabled():
        return ResumeExtractionResult(
            facts=_empty_facts(),
            raw_keys_dropped=[],
            parse_warning="llm_disabled",
        )

    # Resume extraction is the highest-output-token call we make: a
    # well-formatted resume produces structured JSON for many skills,
    # work entries, education, and certifications, each with a verbatim
    # evidence string. 2000 tokens hit truncation on a typical newcomer
    # resume (3KB+ text); 4000 covers everything we've seen without
    # being wasteful. If a resume still truncates here, the response
    # arrives as malformed JSON and call_json returns None, which we
    # already handle as empty extraction below.
    payload = call_json(RESUME_EXTRACTOR_PROMPT, resume_text, max_tokens=4000)
    if not isinstance(payload, dict):
        log.warning("resume extractor: LLM returned non-dict")
        return ResumeExtractionResult(
            facts=_empty_facts(),
            raw_keys_dropped=["llm_returned_non_dict"],
            parse_warning="llm_empty",
        )

    facts, dropped = _validate_and_normalize(payload, resume_text)
    facts["summary_signals"] = _derive_summary_signals(facts)

    parse_warning = None
    if not any(facts[group] for group in _FACT_GROUPS) and not facts["languages"]:
        parse_warning = "empty_extraction"

    return ResumeExtractionResult(
        facts=facts,
        raw_keys_dropped=dropped,
        parse_warning=parse_warning,
    )


# =========================================================================
# Empty / skeleton shape
# =========================================================================
def _empty_facts() -> dict[str, Any]:
    return {
        "version": 1,
        "extracted_at": _now_iso(),
        "extractor_version": EXTRACTOR_VERSION_RESUME,
        "skills": [],
        "work_history": [],
        "education": [],
        "certifications": [],
        "projects": [],
        "languages": [],
        "summary_signals": {},
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# =========================================================================
# Evidence validation + normalization
# =========================================================================
def _validate_and_normalize(
    payload: dict[str, Any], resume_text: str,
) -> tuple[dict[str, Any], list[str]]:
    """Walk the LLM payload, validate every fact's evidence, strip contact
    info, assign fact_ids. Returns (clean facts dict, dropped reasons).
    """
    facts = _empty_facts()
    dropped: list[str] = []
    text_lower = resume_text.lower()
    next_id = {group: 1 for group in _FACT_GROUPS}

    # ---- skills ----
    for raw in payload.get("skills") or []:
        if not isinstance(raw, dict):
            dropped.append("skill_bad_shape")
            continue
        name = (raw.get("name") or "").strip()
        evidence = raw.get("evidence")
        if not name or not evidence:
            dropped.append(f"skill_missing_field:{name or '?'}")
            continue
        if not _is_grounded(str(evidence), text_lower):
            dropped.append(f"skill_ungrounded:{name}")
            continue
        try:
            conf = float(raw.get("confidence", 0.8))
        except (TypeError, ValueError):
            conf = 0.8
        conf = max(0.0, min(1.0, conf))
        facts["skills"].append({
            "fact_id": f"f_skill_{next_id['skills']:03d}",
            "name": name,
            "evidence": str(evidence).strip(),
            "confidence": conf,
            "source": "resume",
        })
        next_id["skills"] += 1

    # ---- work_history ----
    for raw in payload.get("work_history") or []:
        entry = _normalize_entry(
            raw, text_lower, group="work_history",
            string_keys=("title", "employer", "summary"),
            int_keys=("start_year", "end_year"),
            bool_keys=("is_current",),
            dropped=dropped,
        )
        if entry is not None:
            entry["fact_id"] = f"f_work_{next_id['work_history']:03d}"
            facts["work_history"].append(entry)
            next_id["work_history"] += 1

    # ---- education ----
    for raw in payload.get("education") or []:
        entry = _normalize_entry(
            raw, text_lower, group="education",
            string_keys=("credential", "institution", "country"),
            int_keys=("year",),
            bool_keys=(),
            dropped=dropped,
        )
        if entry is not None:
            entry["fact_id"] = f"f_edu_{next_id['education']:03d}"
            facts["education"].append(entry)
            next_id["education"] += 1

    # ---- certifications ----
    for raw in payload.get("certifications") or []:
        entry = _normalize_entry(
            raw, text_lower, group="certifications",
            string_keys=("name", "issuer"),
            int_keys=("year",),
            bool_keys=(),
            dropped=dropped,
        )
        if entry is not None:
            entry["fact_id"] = f"f_cert_{next_id['certifications']:03d}"
            facts["certifications"].append(entry)
            next_id["certifications"] += 1

    # ---- projects ----
    for raw in payload.get("projects") or []:
        entry = _normalize_entry(
            raw, text_lower, group="projects",
            string_keys=("name", "summary"),
            int_keys=(),
            bool_keys=(),
            dropped=dropped,
        )
        if entry is not None:
            entry["fact_id"] = f"f_proj_{next_id['projects']:03d}"
            facts["projects"].append(entry)
            next_id["projects"] += 1

    # ---- languages ----
    # Languages are stored as strings in the v1 schema, but they are still
    # factual user claims. Require the language name to appear in the resume
    # text so the LLM cannot casually add "English" to every profile.
    langs_raw = payload.get("languages") or []
    if isinstance(langs_raw, list):
        seen: set[str] = set()
        for lang in langs_raw:
            if not isinstance(lang, str):
                continue
            clean = lang.strip()
            if not clean:
                continue
            if not _is_grounded(clean, text_lower):
                dropped.append(f"language_ungrounded:{clean}")
                continue
            if clean.lower() not in seen:
                facts["languages"].append(clean)
                seen.add(clean.lower())

    # ---- defensive: drop any top-level contact-info keys ----
    for k in payload.keys():
        if k.lower() in _CONTACT_KEYS:
            dropped.append(f"top_contact_info:{k}")

    return facts, dropped


def _normalize_entry(
    raw: Any,
    text_lower: str,
    *,
    group: str,
    string_keys: tuple[str, ...],
    int_keys: tuple[str, ...],
    bool_keys: tuple[str, ...],
    dropped: list[str],
) -> dict[str, Any] | None:
    """Validate a single fact dict (work / edu / cert / project).

    Returns the cleaned entry (without fact_id; caller assigns) or None
    if the entry has no valid evidence.
    """
    if not isinstance(raw, dict):
        dropped.append(f"{group}_bad_shape")
        return None

    evidence = raw.get("evidence")
    if not evidence or not isinstance(evidence, str):
        dropped.append(f"{group}_missing_evidence")
        return None
    if not _is_grounded(evidence, text_lower):
        dropped.append(f"{group}_ungrounded")
        return None

    entry: dict[str, Any] = {
        "evidence": evidence.strip(),
        "source": "resume",
    }

    # Schema-declared field names override the contact deny-list. Some
    # groups legitimately use a key whose name collides with PII
    # vocabulary -- certifications and projects both name themselves
    # with a `name` field, which the top-level pass at line 263 ALREADY
    # filters away when the LLM emits it as a person identifier. Inside
    # a nested entry the same string refers to a credential/project
    # title, not a person, so a blanket deny-list drop here strips
    # every certification name and breaks downstream matching that
    # reads `cert["name"]`.
    #
    # Round-28 fix: only consult `_CONTACT_KEYS` for keys that AREN'T
    # in this group's declared schema. PII keys the schema doesn't
    # mention (`full_name`, `email`, `phone`, `address`, etc.) still
    # get caught.
    schema_keys: frozenset[str] = frozenset(string_keys + int_keys + bool_keys)
    for k, v in raw.items():
        kl = k.lower()
        if k in ("evidence", "fact_id", "source"):
            continue
        if k not in schema_keys and kl in _CONTACT_KEYS:
            dropped.append(f"{group}_contact_info:{kl}")
            continue

        if k in string_keys and isinstance(v, str):
            cleaned = v.strip()
            if cleaned:
                entry[k] = cleaned
        elif k in int_keys and v is not None:
            try:
                entry[k] = int(v)
            except (TypeError, ValueError):
                pass
        elif k in bool_keys:
            entry[k] = bool(v)

    return entry


# =========================================================================
# Typographic normalization for grounding
# =========================================================================
# PDF extraction often produces curly/smart quotes (U+2018, U+2019, U+201C,
# U+201D), en/em dashes (U+2013, U+2014), and non-breaking spaces (U+00A0)
# in resume text. LLM JSON output almost always normalizes those to ASCII
# (`'`, `"`, `-`, ` `). Pre-fix, the literal-substring `_is_grounded` check
# rejected semantically identical evidence as ungrounded -- the documented
# "Class G driver's license dropped from meeting_01" bug.
#
# Both grounding boundaries (resume/extract.py and chat/extractor.py) MUST
# apply the SAME normalization so a resume fact survives validation
# regardless of which side started out with the smart quote. Keep this in
# one place so they can't drift apart.
#
# Apostrophe-STRIPPED paraphrases still fail: this only maps typographic
# variants to a canonical class, it does NOT remove punctuation. The
# evidence contract ("verbatim phrase from the resume") is preserved --
# we are correcting an encoding mismatch, not loosening to fuzzy match.
_GROUNDING_TRANSLATION = str.maketrans({
    # Single-quote class -> ASCII apostrophe (U+0027)
    "‘": "'",   # left single quotation mark
    "’": "'",   # right single quotation mark (the meeting_01 case)
    "‚": "'",   # single low-9 quotation mark
    "‛": "'",   # single high-reversed-9 quotation mark
    "ʼ": "'",   # modifier letter apostrophe
    "ʻ": "'",   # modifier letter turned comma
    # Double-quote class -> ASCII quotation mark (U+0022)
    "“": '"',   # left double quotation mark
    "”": '"',   # right double quotation mark
    "„": '"',   # double low-9 quotation mark
    "‟": '"',   # double high-reversed-9 quotation mark
    # Dash class -> ASCII hyphen-minus (U+002D)
    "‐": "-",   # hyphen
    "‑": "-",   # non-breaking hyphen
    "‒": "-",   # figure dash
    "–": "-",   # en dash
    "—": "-",   # em dash
    "―": "-",   # horizontal bar
    "−": "-",   # minus sign
    # Space class -> ASCII space (U+0020). Whitespace runs are then
    # collapsed below so multi-space / mixed-space artifacts also align.
    " ": " ",   # non-breaking space
    " ": " ",   # ogham space mark
    " ": " ",   # en quad
    " ": " ",   # em quad
    " ": " ",   # en space
    " ": " ",   # em space
    " ": " ",   # three-per-em space
    " ": " ",   # four-per-em space
    " ": " ",   # six-per-em space
    " ": " ",   # figure space
    " ": " ",   # punctuation space
    " ": " ",   # thin space
    " ": " ",   # hair space
    " ": " ",   # narrow no-break space
    " ": " ",   # medium mathematical space
    "　": " ",   # ideographic space
})
_WHITESPACE_RUN = re.compile(r"\s+")


def _normalize_for_grounding(s: str) -> str:
    """Normalize typographic variants so `_is_grounded` substring checks
    survive PDF→LLM apostrophe / dash / spacing mismatches.

    Single source of truth: applied to BOTH sides (evidence and text)
    in both resume/extract.py and chat/extractor.py. The transforms are
    type-preserving (typographic variants map to ASCII class
    representatives, NEVER strip punctuation), so the evidence contract
    is preserved -- a paraphrase that drops words still fails grounding.
    """
    if not isinstance(s, str):
        return ""
    return _WHITESPACE_RUN.sub(" ", s.translate(_GROUNDING_TRANSLATION)).strip()


def _is_grounded(evidence: str, text_lower: str) -> bool:
    """Evidence must be a >=4 char case-insensitive substring of the
    resume AFTER typographic normalization. See `_normalize_for_grounding`
    for the transform list."""
    ev = _normalize_for_grounding(evidence.lower())
    if len(ev) < 4:
        return False
    return ev in _normalize_for_grounding(text_lower)


# =========================================================================
# Backend-computed summary_signals
# =========================================================================
#
# Keyword sets stay in sync with intake_priority.ROLE_CATEGORIES (informally;
# we duplicate here so the resume extractor doesn't have to import from
# the chat package — keeps the dependency arrow one-way).
_PRIMARY_FIELD_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("software",   ("software", "developer", "engineer", "programmer",
                    "frontend", "backend", "fullstack", "qa", "devops", "it support")),
    ("warehouse",  ("warehouse", "forklift", "shipping", "receiving",
                    "inventory", "picker", "packer", "logistics")),
    ("healthcare", ("nurse", "psw", "personal support", "medical",
                    "healthcare", "clinical", "rpn")),
    ("retail",     ("retail", "cashier", "customer service", "sales associate",
                    "barista", "server")),
    ("admin",      ("administrative", "receptionist", "secretary", "office",
                    "clerk", "data entry", "bookkeeper", "accountant")),
    ("trades",     ("electrician", "plumber", "welder", "carpenter",
                    "mechanic", "hvac", "construction", "millwright")),
)


def _derive_summary_signals(facts: dict[str, Any]) -> dict[str, Any]:
    """Backend math, not LLM judgment.

    - total_years_estimate: sum of (end_year - start_year) ranges in
      work_history, with overlaps merged (so simultaneous roles don't
      double-count).
    - primary_field: keyword frequency over titles + skills.
    - evidence_for_years: the work entries that contributed to the count.
    """
    signals: dict[str, Any] = {}

    # ---- total years (de-overlapping ranges) ----
    ranges: list[tuple[int, int]] = []
    contributing: list[str] = []
    current_year = datetime.now(timezone.utc).year
    for w in facts.get("work_history") or []:
        sy = w.get("start_year")
        ey = w.get("end_year")
        if w.get("is_current") and isinstance(sy, int) and not isinstance(ey, int):
            ey = current_year
        if isinstance(sy, int) and isinstance(ey, int) and ey >= sy:
            ranges.append((sy, ey))
            if w.get("evidence"):
                contributing.append(w["evidence"])

    if ranges:
        merged = _merge_ranges(ranges)
        total_years = sum(end - start for start, end in merged)
        signals["total_years_estimate"] = total_years
        if contributing:
            signals["evidence_for_years"] = " + ".join(contributing)

    # ---- primary field by keyword frequency ----
    titles = " ".join(
        (w.get("title") or "") for w in (facts.get("work_history") or [])
    )
    skill_names = " ".join(
        (s.get("name") or "") for s in (facts.get("skills") or [])
    )
    haystack = (titles + " " + skill_names).lower()

    if haystack.strip():
        scores: dict[str, int] = {}
        for field_name, keywords in _PRIMARY_FIELD_KEYWORDS:
            scores[field_name] = sum(1 for kw in keywords if kw in haystack)
        best_field, best_score = max(scores.items(), key=lambda kv: kv[1])
        if best_score > 0:
            signals["primary_field"] = best_field

    return signals


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping integer year ranges. Avoids double-counting when
    someone held two roles simultaneously."""
    if not ranges:
        return []
    sorted_r = sorted(ranges)
    merged = [sorted_r[0]]
    for start, end in sorted_r[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


# =========================================================================
# Prompt
# =========================================================================
RESUME_EXTRACTOR_PROMPT = """You extract structured profile data from a newcomer's resume for SkillBridge SSM (a Sault Ste. Marie job-matching service).

You MUST output valid JSON with this exact shape (omit any field you cannot fill from the resume text):
{
  "skills":         [{"name": "...", "evidence": "verbatim phrase from resume", "confidence": 0.0-1.0}],
  "work_history":   [{"title": "...", "employer": "...", "start_year": YYYY, "end_year": YYYY, "is_current": false, "summary": "...", "evidence": "verbatim"}],
  "education":      [{"credential": "...", "institution": "...", "year": YYYY, "country": "...", "evidence": "verbatim"}],
  "certifications": [{"name": "...", "issuer": "...", "year": YYYY, "evidence": "verbatim"}],
  "projects":       [{"name": "...", "summary": "...", "evidence": "verbatim"}],
  "languages":      ["English", "French", ...]
}

EVIDENCE RULES — read carefully:
- Every fact (skills, work_history, education, certifications, projects) MUST include "evidence": a verbatim phrase from the resume text (>=4 chars). Anything ungrounded is dropped.
- Do not paraphrase the evidence. Copy directly from the resume.
- If you cannot find verbatim evidence, OMIT that fact. Do not infer.

DO NOT EXTRACT (these are dropped silently — PII risk, not used for matching):
- Name, first_name, last_name, full_name
- Email, phone, address, city, postal_code
- LinkedIn, GitHub, website, social media links

Skills should be:
- Short concrete noun phrases ("React", "forklift operation", "Microsoft Excel", "patient care")
- Not personality traits ("hard worker", "team player")
- Not aspirations ("want to learn nursing")
- Confidence: 0.9 if explicitly listed in a Skills section; 0.7 if mentioned in passing in a job description; 0.5 if heavily inferred

Work history:
- title: job title only ("Software Engineer", not "Software Engineer at Acme")
- employer: company/organization name
- start_year / end_year: 4-digit integers. If still current, set is_current=true and omit end_year.
- summary: one short sentence describing what they did

Education:
- credential: degree/diploma name ("BSc Computer Science", "PSW Certificate")
- year: graduation/completion year (4-digit integer)

Languages: just the language name (English, French, Punjabi, Tagalog, etc.). No proficiency levels.

If the resume is sparse or hard to parse, return whatever you can ground. An empty JSON ({"skills":[],"work_history":[],...}) is acceptable.

No commentary, no markdown, JSON only.
"""
