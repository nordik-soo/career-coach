"""Derive flat StagedProfile slots from resume_facts_json.

`resume_facts_json` (produced by skillbridge.resume.extract) is the
canonical structured parse. The flat slots — `skills_text`,
`experience_text`, `education_text`, and the StagedSkill list — are a
*view* of that data, shaped for the existing match engine and chat
extractor.

Pure functions. Idempotent. Re-runs cleanly on every correction. The
single source of truth is the facts JSON; flat slots are always derived,
never edited directly.

The suppression layer (StagedProfile.suppressed_fact_ids) is applied
*before* derivation via derive_with_suppressions() so the flat slots
already reflect the user's removals when the matcher reads them.
"""
from __future__ import annotations

from typing import Any

from config import MIN_EXTRACTION_CONFIDENCE
from skillbridge.extract.base import resolve_skill
from skillbridge.match.aliases import canonicalize_skill


# Max skills surfaced to the flat skills_text slot. More than this is just
# noise for matching and clutters the responder's context.
MAX_DERIVED_SKILLS: int = 12

# Max work_history entries that contribute to the flat experience_text slot.
MAX_DERIVED_WORK_ENTRIES: int = 3


def derive_staged_slots(facts: dict[str, Any]) -> dict[str, Any]:
    """Map resume_facts_json into flat profile slots.

    Returns a dict with keys:
      - `skills_text`     (str | None)
      - `experience_text` (str | None)
      - `education_text`  (str | None)
      - `skills`          (list[dict] in StagedSkill shape:
                           skill_name, raw_phrase, confidence, source)

    The caller (chat handler) merges these into StagedProfile alongside
    any chat-extracted facts. Suppressed fact_ids should be filtered
    *before* calling this (see derive_with_suppressions).
    """
    return {
        "skills_text": _derive_skills_text(facts),
        "experience_text": _derive_experience_text(facts),
        "education_text": _derive_education_text(facts),
        "skills": _derive_skills_list(facts),
    }


def derive_with_suppressions(
    facts: dict[str, Any],
    suppressed_fact_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Derive flat slots after dropping facts the user has suppressed.

    `resume_facts_json` is immutable parser output. When the user has
    suppressed facts during the RESUME_REVIEW turn ("that's not my job",
    "remove that"), we derive the flat slots from the *effective* view:
    facts minus suppressions. The original entries remain in
    resume_facts_json for audit; they just don't drive matching anymore.
    """
    if not suppressed_fact_ids:
        return derive_staged_slots(facts)

    suppressed = set(suppressed_fact_ids)
    effective: dict[str, Any] = {
        "skills": [
            s for s in (facts.get("skills") or [])
            if s.get("fact_id") not in suppressed
        ],
        "work_history": [
            w for w in (facts.get("work_history") or [])
            if w.get("fact_id") not in suppressed
        ],
        "education": [
            e for e in (facts.get("education") or [])
            if e.get("fact_id") not in suppressed
        ],
        "certifications": [
            c for c in (facts.get("certifications") or [])
            if c.get("fact_id") not in suppressed
        ],
        "projects": [
            p for p in (facts.get("projects") or [])
            if p.get("fact_id") not in suppressed
        ],
        "languages": facts.get("languages") or [],
        "summary_signals": facts.get("summary_signals") or {},
    }
    return derive_staged_slots(effective)


# =========================================================================
# Per-section derivation
# =========================================================================
def _derive_skills_text(facts: dict[str, Any]) -> str | None:
    """Comma-joined names of the top-confidence skills.

    Filters by MIN_EXTRACTION_CONFIDENCE (0.60 by default). Below-floor
    skills stay in the JSON for transparency but don't feed matching —
    keeps the matcher from working with weak signals.
    """
    skills = facts.get("skills") or []
    if not skills:
        return None

    eligible = [
        s for s in skills
        if isinstance(s, dict) and (s.get("confidence") or 0) >= MIN_EXTRACTION_CONFIDENCE
    ]
    # Stable order: highest confidence first, then alphabetical by name.
    eligible.sort(key=lambda s: (-(s.get("confidence") or 0), s.get("name", "")))
    names = [s["name"] for s in eligible[:MAX_DERIVED_SKILLS] if s.get("name")]
    return ", ".join(names) if names else None


def _derive_skills_list(facts: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert resume skills AND certifications into StagedSkill-shaped
    dicts.

    Skills:
      - Above-floor confidence only (MIN_EXTRACTION_CONFIDENCE).
      - source="resume".
      - Confidence carried through from the LLM's per-skill estimate.

    Certifications (round-29 cert -> matcher promotion):
      - Promoted into the same StagedSkill list AFTER skills, with
        deduplication by canonical lookup-form so a certification
        also listed as a skill doesn't double-count.
      - source="resume". `resume_cert` would fall through to "chat" at
        the persistence boundary in chat/handler.py:3137 (which only
        accepts {"chat","resume","form","manual_update"}).
      - Fixed confidence=0.9 -- a credential in the LICENSES section
        is a definite claim, not a per-skill heuristic. The LLM's
        confidence field applies to the loose Skills bucket.
      - Education credentials (degrees / apprenticeships) are NOT
        promoted here; they need separate matching semantics and are
        deferred to a follow-up slice.

    Each entry tags `source: "resume"` so downstream code can tell
    resume-derived facts from chat-derived ones.
    """
    out: list[dict[str, Any]] = []
    seen_canonical: set[str] = set()

    # ---- 1. Skills ----
    # 2026-07-01: resolve each skill against reference.skill with
    # allow_fuzzy=False. Populates skill_id when the resume-extracted
    # name is an EXACT / alias match to an OaSIS competency; leaves
    # None otherwise. Fuzzy is intentionally OFF because resume vocab
    # ('accounts payable management') fuzzes to unrelated OaSIS entries
    # ('Time Management' at threshold 0.75) and corrupts Layer A/C.
    # See resolve_skill docstring for rationale.
    #
    # 2026-07-02 (Path B): preserve the ORIGINAL skill_name even when
    # the resolver returns a canonical form. Rationale: staged.merge_skills
    # dedupes by skill_name.lower() -- so if we swapped "microsoft excel",
    # "microsoft word", "quickbooks online", "adobe acrobat" all to the
    # canonical "Digital Literacy", they'd collapse into ONE staged
    # skill and we'd lose 3-5 name-based retrieval hits against SSM
    # postings that literally say "microsoft excel" etc. By keeping the
    # raw name AND populating skill_id, each tool gets:
    #   - name-based retrieval hit ("microsoft excel" matches job postings
    #     that mention "microsoft excel")
    #   - id-based retrieval hit (F.01.c.02 matches job postings that
    #     require Digital Literacy per reference.noc_skill)
    # user_ids stays deduped (it's a set); user_names gains diversity.
    # Layer C candidate pool widens accordingly.
    skills = facts.get("skills") or []
    for s in skills:
        if not isinstance(s, dict):
            continue
        if (s.get("confidence") or 0) < MIN_EXTRACTION_CONFIDENCE:
            continue
        name = s.get("name")
        if not name:
            continue
        sid, _canonical = resolve_skill(name, allow_fuzzy=False)
        out.append({
            "skill_name": name,  # Path B: preserve raw name for retrieval diversity
            "skill_id": sid,
            "raw_phrase": s.get("evidence"),
            "confidence": float(s.get("confidence") or 0.7),
            "source": "resume",
        })
        seen_canonical.add(canonicalize_skill(name))

    # ---- 2. Certifications promoted as skills ----
    certifications = facts.get("certifications") or []
    for c in certifications:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        if not name:
            continue
        key = canonicalize_skill(name)
        if key and key in seen_canonical:
            # Dedupe: the LLM already emitted this credential under
            # skills too. Keep the existing skills-pass entry (whose
            # confidence is whatever the LLM thought) and skip.
            continue
        seen_canonical.add(key)
        # Same exact-only resolution for certifications. Most will
        # remain skill_id=None (certifications are concrete credentials,
        # not OaSIS competencies), which is correct.
        # Path B: preserve raw certification name (same rationale as
        # the skills block above).
        sid, _canonical = resolve_skill(name, allow_fuzzy=False)
        out.append({
            "skill_name": name,
            "skill_id": sid,
            "raw_phrase": c.get("evidence"),
            "confidence": 0.9,
            "source": "resume",
        })
    return out


def _derive_experience_text(facts: dict[str, Any]) -> str | None:
    """Top N work history entries as a short multi-line string.

    Format per entry:  "Title at Employer (years): summary"
    Missing pieces are silently dropped so the line stays readable.
    """
    work = facts.get("work_history") or []
    if not work:
        return None

    lines: list[str] = []
    for w in work[:MAX_DERIVED_WORK_ENTRIES]:
        if not isinstance(w, dict):
            continue
        title = (w.get("title") or "").strip()
        employer = (w.get("employer") or "").strip()
        summary = (w.get("summary") or "").strip()
        years = _format_years(w)

        if not title and not employer:
            # Entry has neither role nor employer; not worth surfacing.
            continue

        # "Title at Employer" — fall back gracefully if one is missing.
        if title and employer:
            head = f"{title} at {employer}"
        elif title:
            head = title
        else:
            head = employer

        if years:
            head += f" ({years})"
        if summary:
            head += f": {summary}"
        lines.append(head)

    return "\n".join(lines) if lines else None


def _derive_education_text(facts: dict[str, Any]) -> str | None:
    """First (assumed highest) education entry as a comma-joined sentence."""
    edu = facts.get("education") or []
    if not edu:
        return None
    first = edu[0] if isinstance(edu[0], dict) else None
    if not first:
        return None
    parts: list[str] = []
    if first.get("credential"):
        parts.append(str(first["credential"]))
    if first.get("institution"):
        parts.append(str(first["institution"]))
    if first.get("year"):
        parts.append(str(first["year"]))
    return ", ".join(parts) if parts else None


# =========================================================================
# Helpers
# =========================================================================
def _format_years(w: dict[str, Any]) -> str:
    sy = w.get("start_year")
    ey = w.get("end_year")
    if w.get("is_current") and isinstance(sy, int):
        return f"{sy}-present"
    if isinstance(sy, int) and isinstance(ey, int):
        return f"{sy}-{ey}"
    if isinstance(sy, int):
        return str(sy)
    return ""


# =========================================================================
# Compact facts — for cookie-mode session storage
# =========================================================================
# A signed cookie has a ~4 KB hard limit. The full resume_facts_json
# (with verbatim evidence strings + descriptions) often exceeds that for
# even a modest resume. The compact form strips:
#   - evidence strings on every fact
#   - long summary / description fields
# It keeps the structural data the system needs:
#   - fact_id (so suppression keeps working across turns)
#   - name / title / employer / credential / year (so the responder can
#     surface a confirmation summary)
#   - confidence / source flags (so derivation rules still apply)
#
# Trade-off: the responder cannot quote the exact resume phrasing in
# cookie mode. It can still reference the fact ("your 4 years at Acme")
# but not the verbatim quote ("you wrote 'Built customer dashboards'").
# Redis-mode sessions keep the full facts JSON so verbatim grounding
# stays available there.
def compact_facts(facts: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a slimmed copy of resume_facts_json suitable for cookie storage.

    Returns None if input is None. Returns an empty-but-valid shape if
    input is empty.
    """
    if facts is None:
        return None
    out: dict[str, Any] = {
        "version": facts.get("version", 1),
        "extracted_at": facts.get("extracted_at"),
        "extractor_version": facts.get("extractor_version"),
        "skills": [],
        "work_history": [],
        "education": [],
        "certifications": [],
        "projects": [],
        "languages": list(facts.get("languages") or []),
        "summary_signals": dict(facts.get("summary_signals") or {}),
        # Marker so anyone reading these compact facts later knows the
        # evidence fields were intentionally stripped (not lost).
        "_compact": True,
    }

    # summary_signals carries one long evidence string — drop that too.
    out["summary_signals"].pop("evidence_for_years", None)

    for s in facts.get("skills") or []:
        if not isinstance(s, dict):
            continue
        out["skills"].append({
            "fact_id": s.get("fact_id"),
            "name": s.get("name"),
            "confidence": s.get("confidence"),
            "source": s.get("source"),
        })

    for w in facts.get("work_history") or []:
        if not isinstance(w, dict):
            continue
        out["work_history"].append({
            "fact_id": w.get("fact_id"),
            "title": w.get("title"),
            "employer": w.get("employer"),
            "start_year": w.get("start_year"),
            "end_year": w.get("end_year"),
            "is_current": w.get("is_current"),
            "source": w.get("source"),
        })

    for e in facts.get("education") or []:
        if not isinstance(e, dict):
            continue
        out["education"].append({
            "fact_id": e.get("fact_id"),
            "credential": e.get("credential"),
            "institution": e.get("institution"),
            "year": e.get("year"),
            "source": e.get("source"),
        })

    for c in facts.get("certifications") or []:
        if not isinstance(c, dict):
            continue
        out["certifications"].append({
            "fact_id": c.get("fact_id"),
            "name": c.get("name"),
            "issuer": c.get("issuer"),
            "year": c.get("year"),
            "source": c.get("source"),
        })

    for p in facts.get("projects") or []:
        if not isinstance(p, dict):
            continue
        out["projects"].append({
            "fact_id": p.get("fact_id"),
            "name": p.get("name"),
            "source": p.get("source"),
        })

    return out
