"""Skill alias normalization -- Sprint 5 step 2.

A small, curated, evidence-driven map of skill-phrase variants to their
canonical form. Used by the matching engine to collapse spelling and
punctuation differences before comparing a user's skill phrase to a
job's skill phrase.

Scope discipline:
  - Seed ONLY from observed failures in fixtures or real chat sessions.
  - Do NOT model a general skills ontology here. If a variant has not
    actually caused a missed match in the current SCCC dataset, leave
    it out. Aliases earn their way in.
  - When in doubt, prefer fuzzy substring/token overlap (the existing
    `_skills_overlap` in engine.py) over adding more entries.

Categories below mirror the buckets approved for this sprint:
  - Driver / licence variants
  - Trades (truck and coach, diesel)
  - Customer service / admin
  - Cash / payment
  - Healthcare basics (PSW postings)
  - Safety tickets (WHMIS / First Aid / CPR)
"""
from __future__ import annotations

import re


def _key(name: str) -> str:
    """Build the lookup key for an arbitrary skill phrase.

    Lowercases, strips apostrophes, replaces every other non-alphanumeric
    with a single space, then collapses runs of whitespace. This is the
    SAME normalization used to build map keys and to canonicalize lookups,
    so equivalent phrases (case, punctuation, hyphenation) always collide.
    """
    s = name.lower().strip().replace("'", "").replace("’", "")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# Map keys are lookup-form (lowercased, apostrophe-stripped, non-alphanum
# collapsed to single space). Values are the canonical lookup-form that
# multiple variants collapse to. Both ALWAYS pass through _key() before
# being compared, so two phrases that canonicalize to the same value will
# match exactly without depending on the fuzzy overlap fallback.
SKILL_ALIASES: dict[str, str] = {
    # ----- Driver / licence -----
    # Canonical lookup-form: "class g license"
    "class g licence": "class g license",
    "g licence": "class g license",
    "g license": "class g license",
    "class g": "class g license",
    "valid g licence": "class g license",
    "valid g license": "class g license",
    "ontario g licence": "class g license",
    "ontario g license": "class g license",
    "full g license": "class g license",
    "full g licence": "class g license",
    # v1.2.0 LLM extracts these specific phrases on the truck/coach JD;
    # added via Sprint 5 slice 4e (evidence-driven, not speculation).
    "class g drivers license": "class g license",
    "class g drivers licence": "class g license",
    "class g driver license": "class g license",
    "class g driver licence": "class g license",

    # Canonical lookup-form: "drivers license"
    # (apostrophe stripped by _key, so "driver's license" -> "drivers license")
    "drivers licence": "drivers license",
    "driver licence": "drivers license",
    "driver license": "drivers license",
    "valid drivers license": "drivers license",
    "valid drivers licence": "drivers license",

    # NOTE on G2: Ontario's graduated licensing has G1 -> G2 -> G as
    # distinct levels. G2 is NOT Class G; it is an intermediate licence
    # with restrictions. Folding G2 into "class g license" would let a
    # candidate with only G2 silently pass a job's mandatory Class G
    # requirement -- the exact failure round-29 reviewer flagged.
    #
    # Aliases for the FULL "Class G" forms (with explicit "Class G" or
    # "full G" wording) are kept above; aliases for G2 alone are
    # deliberately absent so a G2-only credential doesn't claim Class G.
    #
    # "G2/G" is the resume convention for "graduated through G2 to G"
    # but is ambiguous (sometimes used to mean "currently G2 working
    # toward G"). The conservative default: G2/G gets its OWN canonical
    # ("g2 g license") and DOES NOT match a Class G requirement. A
    # candidate who actually has Class G typically lists it separately
    # (which the perfect_310s resume does: it carries both "Class G
    # driver's license" AND "G2/G driver's license" -- the first is
    # what matches).
    "g2 g": "g2 g license",
    "g2 g license": "g2 g license",
    "g2 g licence": "g2 g license",
    "g2 g drivers license": "g2 g license",
    "g2 g drivers licence": "g2 g license",
    "g2 g driver license": "g2 g license",
    "g2 g driver licence": "g2 g license",
    "valid g2 g license": "g2 g license",
    "valid g2 g licence": "g2 g license",
    "valid g2 g drivers license": "g2 g license",
    "valid g2 g drivers licence": "g2 g license",
    "valid g2 g driver license": "g2 g license",
    "valid g2 g driver licence": "g2 g license",
    # `valid` prefix forms for explicit Class G fold to class g license.
    "valid ontario class g drivers license": "class g license",
    "valid ontario class g drivers licence": "class g license",
    "valid ontario class g driver license": "class g license",
    "valid ontario class g driver licence": "class g license",

    # ----- 310S Automotive Service Technician (Ontario apprenticeship) -----
    # Canonical lookup-form: "310s automotive technician".
    # Round-29 cert->matcher promotion: meeting_01_310s_automotive_perfect
    # has "Valid Ontario 310S Automotive Service Technician License" in
    # its LICENSES section. Without these aliases the promoted StagedSkill
    # canonicalizes to "310s automotive service technician license" or
    # similar variants -- distinct from how the Honda job's job_skill row
    # typically names the credential. Folding every observed form into
    # one canonical lets the matcher match.
    "310s": "310s automotive technician",
    "310 s": "310s automotive technician",
    "310s license": "310s automotive technician",
    "310s licence": "310s automotive technician",
    "310s certification": "310s automotive technician",
    "310s cert": "310s automotive technician",
    "310s automotive": "310s automotive technician",
    "310s automotive technician license": "310s automotive technician",
    "310s automotive technician licence": "310s automotive technician",
    "310s automotive technician certification": "310s automotive technician",
    "310s automotive service technician": "310s automotive technician",
    "310s automotive service technician license": "310s automotive technician",
    "310s automotive service technician licence": "310s automotive technician",
    "valid 310s license": "310s automotive technician",
    "valid 310s licence": "310s automotive technician",
    "ontario 310s license": "310s automotive technician",
    "ontario 310s licence": "310s automotive technician",
    "ontario 310s automotive service technician license": "310s automotive technician",
    "automotive service technician 310s": "310s automotive technician",
    "automotive technician 310s": "310s automotive technician",
    # `valid` / `valid ontario` prefixes -- preserved by the LLM from
    # the LICENSES section verbatim.
    "valid ontario 310s license": "310s automotive technician",
    "valid ontario 310s licence": "310s automotive technician",
    "valid ontario 310s automotive service technician license": "310s automotive technician",
    "valid ontario 310s automotive service technician licence": "310s automotive technician",
    "valid 310s automotive technician license": "310s automotive technician",
    "valid 310s automotive technician licence": "310s automotive technician",
    "valid 310s automotive service technician license": "310s automotive technician",
    "valid 310s automotive service technician licence": "310s automotive technician",

    # ----- 310T Truck and Coach Technician (Ontario apprenticeship) -----
    # Canonical lookup-form: "310t truck and coach technician".
    # Round-33 R-2: credential-strict matching means 310T phrasings
    # only match via the alias map (no fuzzy fallback). The truck/coach
    # job posting frequently lists the credential as the bare code OR
    # as "310T certificate of qualification" OR "310T license"; every
    # observed form must fold to the same canonical or the match
    # silently fails.
    "310t": "310t truck and coach technician",
    "310 t": "310t truck and coach technician",
    "310t license": "310t truck and coach technician",
    "310t licence": "310t truck and coach technician",
    "310t certification": "310t truck and coach technician",
    "310t cert": "310t truck and coach technician",
    "310t certificate": "310t truck and coach technician",
    "310t certificate of qualification": "310t truck and coach technician",
    "310t coq": "310t truck and coach technician",
    "310t truck and coach": "310t truck and coach technician",
    "310t truck and coach license": "310t truck and coach technician",
    "310t truck and coach licence": "310t truck and coach technician",
    "310t truck and coach certification": "310t truck and coach technician",
    "310t technician": "310t truck and coach technician",
    "310t technician license": "310t truck and coach technician",
    "310t technician licence": "310t truck and coach technician",
    "310t technician certification": "310t truck and coach technician",
    "truck and coach 310t": "310t truck and coach technician",
    "truck and coach technician 310t": "310t truck and coach technician",
    "valid 310t license": "310t truck and coach technician",
    "valid 310t licence": "310t truck and coach technician",
    "valid ontario 310t license": "310t truck and coach technician",
    "valid ontario 310t licence": "310t truck and coach technician",
    "valid 310t certificate of qualification": "310t truck and coach technician",

    # ----- Trades -----
    # Canonical: "truck and coach"
    "truck coach": "truck and coach",
    "truck coach technician": "truck and coach",
    "truck and coach technician": "truck and coach",

    # Canonical: "diesel engine repair"
    "diesel repair": "diesel engine repair",
    "diesel mechanics": "diesel engine repair",
    "diesel mechanic": "diesel engine repair",
    "diesel engines": "diesel engine repair",

    # Canonical: "heavy equipment"
    "heavy machinery": "heavy equipment",
    "heavy equipment operation": "heavy equipment",
    "heavy equipment operator": "heavy equipment",

    # ----- Customer service / admin -----
    # Canonical: "customer service"
    "client service": "customer service",
    "client services": "customer service",
    "customer support": "customer service",
    "customer care": "customer service",

    # ----- Cash / payment -----
    # Canonical: "cash handling"
    "handling cash": "cash handling",
    "cashiering": "cash handling",
    "cash transactions": "cash handling",

    # Canonical: "payment processing"
    "processing payments": "payment processing",
    "payments processing": "payment processing",

    # ----- Healthcare basics -----
    # Canonical: "personal support worker"
    "psw": "personal support worker",
    "p s w": "personal support worker",

    # ----- Safety tickets -----
    # Canonical: "first aid"
    "standard first aid": "first aid",
    "first aid certification": "first aid",
    "first aid certified": "first aid",
    # Round-29: resume LICENSES sections commonly bundle these
    # ("Standard First Aid and CPR") -- the matcher treats them as
    # first aid since job-skill rows referencing "first aid" usually
    # accept the combined ticket.
    "standard first aid and cpr": "first aid",
    "first aid and cpr": "first aid",
    "first aid cpr": "first aid",

    # Canonical: "cpr"
    "cardiopulmonary resuscitation": "cpr",
    "cpr certification": "cpr",
    "cpr certified": "cpr",
    "cpr c": "cpr",

    # Canonical: "whmis"
    "whmis certification": "whmis",
    "whmis certified": "whmis",
    "whmis 2015": "whmis",
    "whmis training": "whmis",
}


def canonicalize_skill(name: str | None) -> str:
    """Return the canonical lookup-form for a skill phrase.

    Two phrases that should be treated as the same skill (case, punctuation,
    or known alias) return the same string. Phrases with no known alias
    fall through to their _key() form (lowercased, normalized whitespace).
    """
    if not name:
        return ""
    k = _key(name)
    return SKILL_ALIASES.get(k, k)
