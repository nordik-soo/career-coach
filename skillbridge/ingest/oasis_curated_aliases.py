"""SkillBridge SSM curated OaSIS aliases (locked 2026-07-01).

This module is SkillBridge policy, NOT raw OaSIS taxonomy data. It
answers the question: "when we see this resume phrase, are we willing
to treat it as this OaSIS competency?"

Aliases live in Python (not in the generated data/oasis_skills.csv)
because:
  1. The CSV is ETL output regenerated from official OaSIS source
     files by scripts/build_oasis_loader_csvs.py. Hand-edits to the
     CSV would be silently overwritten on the next taxonomy refresh.
  2. `data/*.csv` is gitignored, so CSV edits would not be tracked.
  3. Aliases are POLICY, and policy belongs in reviewable, versioned
     code -- diff review + tests can enforce the discipline in a way
     free-form CSVs cannot.

The load_oasis_skills() loader in skillbridge/ingest/reference.py
overlays these onto the CSV-loaded rows at DB-write time. Curated
aliases WIN over CSV content when both exist for the same skill_id.

------------------------------------------------------------
CURATION POLICY (category-1 only)
------------------------------------------------------------
An alias is admissible ONLY when the resume phrase deterministically
maps to the OaSIS competency without judgment inference. Two safe
kinds:

  (a) Named-tool -> tool-capability. Presence of a tool on a resume
      proves capability with THAT TOOL, not the abstract competency
      the tool may serve. Microsoft Word proves Digital Literacy
      (a tool competency) -- it does NOT prove Writing (a skill
      the person might be bad at even with Word installed).

  (b) Synonym / morphological variant -> same competency.
      "coordination" -> Coordinating. "negotiation" -> Negotiating.
      "written communication" -> Writing (this IS a stated writing
      claim, unlike "microsoft word").

------------------------------------------------------------
EXPLICITLY PROTECTED (must stay empty)
------------------------------------------------------------
  F.02.a.03  Decision Making    -- LLM judgment territory
  F.02.b.01  Evaluation          -- LLM judgment territory
  F.04.b.05  Time Management     -- safety pin against the observed
                                   'accounts payable management' ->
                                   'Time Management' false positive
                                   (at SKILL_FUZZY_THRESHOLD=0.75).
                                   Even exact-alias resolution here
                                   would corrupt Layer A/C.

------------------------------------------------------------
EXPLICITLY REJECTED CANDIDATES (do NOT re-add without policy review)
------------------------------------------------------------
  "quality assurance"  under Quality Control Testing -- QA is a
      broader process/audit discipline, not testing. Different roles
      even inside the same department.

  "influencing" (bare) under Persuading -- too broad; reads as
      "shaped strategy" or "advised leadership" which is not the
      same as convincing someone to change their behavior. The
      more specific "influencing others" is retained.

  "google docs" under Digital Literacy -- writing-adjacent tool;
      creates the same Word/Writing ambiguity the Word decision
      resolves. `google workspace` covers the general Google-tools
      claim.

Any future addition to CURATED_ALIASES must have hiring-manager
judgment behind it and should be reviewed with the same discipline.
"""
from __future__ import annotations


# ----------------------------------------------------------------------
# The alias table. Keys are reference.skill.skill_id (OaSIS taxonomy).
# Values are tuples of resume phrasings the resolver should treat
# as EXACTLY the keyed OaSIS competency at
# skillbridge/extract/base.py::resolve_skill(..., allow_fuzzy=False).
#
# All lowercase (the resolver compares lowercased). Multi-word phrases
# use single spaces. No wildcards, regex, or partial matches -- only
# exact string equality after case-folding.
# ----------------------------------------------------------------------
CURATED_ALIASES: dict[str, tuple[str, ...]] = {
    # ----- Oral / listening -------------------------------------------
    "F.01.a.01": (  # Oral Communication: Active Listening
        "active listening",
    ),
    "F.01.a.03": (  # Oral Communication: Oral Expression
        "verbal communication",
        "public speaking",
    ),

    # ----- Written communication --------------------------------------
    "F.01.b.02": (  # Writing (STATED writing claims only, NOT tools)
        "written communication",
        "business writing",
        "report writing",
    ),

    # ----- Numeracy ---------------------------------------------------
    "F.01.c.01": (  # Numeracy
        "basic math",
        "workplace math",
    ),

    # ----- Digital Literacy (the tool-heavy row) ----------------------
    # NOTE: Word, Excel, QuickBooks, Acrobat all map HERE (they prove
    # digital-tool use), NOT to Writing. Word does not by itself
    # demonstrate writing quality; a resume that also says "drafted
    # reports" gives Writing evidence via other aliases below.
    "F.01.c.02": (  # Digital Literacy
        "computer literacy",
        "microsoft excel", "ms excel",
        "microsoft word",   # NOT under Writing (see docstring case)
        "microsoft office", "ms office",
        "microsoft outlook",
        "quickbooks", "quickbooks online", "quickbooks desktop",
        "adobe acrobat",
        "google workspace",  # "google docs" was dropped -- see docstring
    ),

    # ----- Analytical / problem solving -------------------------------
    "F.02.a.01": (  # Critical Thinking
        "analytical thinking",
        "analytical skills",
        "critical analysis",
    ),
    "F.02.b.05": (  # Problem Solving
        "problem-solving",
        "solving problems",
    ),

    # ----- Trades-adjacent (equipment) --------------------------------
    "F.03.a.02": (  # Preventative Maintenance
        "preventive maintenance",  # spelling variant only
    ),
    "F.03.a.06": (  # Troubleshooting
        "technical troubleshooting",
        "debugging",
    ),
    "F.03.a.07": (  # Repairing
        "equipment repair",
        "machine repair",
    ),
    "F.03.a.08": (  # Quality Control Testing
        "quality control",
        "qc testing",
        # "quality assurance" DROPPED at sign-off -- see docstring
    ),
    "F.03.a.10": (  # Digital Systems Production
        "software development",
        "programming",
        "coding",
    ),

    # ----- Management ("management" as a stated skill) ----------------
    "F.04.a.01": (  # Management of Financial Resources
        "budget management",
        "financial management",
        "budgeting",
    ),
    "F.04.a.02": (  # Management of Material Resources
        "inventory management",
        "materials management",
    ),
    "F.04.a.03": (  # Management of Personnel Resources
        "people management",
        "staff management",
        "team management",
    ),

    # ----- Interpersonal ----------------------------------------------
    "F.05.a.01": (  # Coordinating
        "coordination",
    ),
    "F.05.a.02": (  # Instructing
        "teaching",
        "training others",
    ),
    "F.05.a.03": (  # Negotiating
        "negotiation",
    ),
    "F.05.a.04": (  # Persuading
        "persuasion",
        "influencing others",  # NOT bare "influencing" -- see docstring
    ),

    # ----------------------------------------------------------------------
    # DELIBERATELY ABSENT (must remain absent):
    #   F.02.a.03 Decision Making   -- LLM judgment territory
    #   F.02.b.01 Evaluation         -- LLM judgment territory
    #   F.04.b.05 Time Management    -- safety pin (AP/AR false positive)
    #   F.01.a.02 Oral Comprehension -- overlaps with Active Listening
    #   F.01.b.01 Reading Comprehension -- no unambiguous synonym
    #   F.02.a.02 Learning and Teaching Strategies -- judgment territory
    #   F.02.b.03 Systems Analysis   -- too specific, no clean synonym
    #   F.03.a.01 Equipment and Tool Selection -- needs trades context
    #   F.03.a.03 Setting Up          -- too vague standalone
    #   F.03.a.04 Operation and Control -- too vague standalone
    #   F.03.a.05 Operation Monitoring -- very specific, no synonym
    #   F.03.a.09 Product Design     -- too specific
    #   F.04.c.01 Monitoring          -- too generic; cross-context bleed
    #   F.05.a.05 Social Perceptiveness -- subjective
    # ----------------------------------------------------------------------
}


def curated_aliases_for(skill_id: str) -> tuple[str, ...]:
    """Return the curated alias tuple for a skill_id, or () if none.

    Read-only view suitable for downstream consumers that want to
    merge with CSV-provided aliases. Callers must NOT mutate the
    returned tuple (they're immutable by type; this is a
    belt-and-braces reminder).
    """
    return CURATED_ALIASES.get(skill_id, ())
