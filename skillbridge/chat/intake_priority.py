"""Intake slot prioritisation — role-aware, conversational.

The chat is meant to feel like a natural conversation, not a form. So:
- default max_n=1 (one focused question per turn — never a checklist)
- slot priority is role-category-aware: a software developer is asked
  about stack/projects before shifts; a warehouse role is asked about
  shifts early because shift work is the norm
- preferred_location sits near the tail: this is an SSM-only product,
  so asking "where in SSM" feels generic — only ask if the user brings
  location up themselves
- each slot carries role-aware prompt hints with concrete examples
  ("React, Python, APIs, databases" vs "forklift, inventory, shipping")

Role categories are classified from target_role_text via keyword match.
'other' is the safe default for anything unclassified — its priority
list resembles a generic admin/retail blend.
"""
from __future__ import annotations

from skillbridge.session.staging import StagedProfile


# ============================================================================
# Role classification
# ============================================================================
ROLE_CATEGORIES: dict[str, tuple[str, ...]] = {
    "software": (
        "software", "developer", "programmer", "engineer", "coder",
        "frontend", "front-end", "front end",
        "backend", "back-end", "back end",
        "fullstack", "full-stack", "full stack",
        "web dev", "mobile dev", "data scientist", "data engineer",
        "devops", "qa engineer", "tester", "it support",
        "sysadmin", "system admin", "network admin",
        "ml engineer", "machine learning",
    ),
    "warehouse": (
        "warehouse", "forklift", "shipping", "receiving", "inventory",
        "picker", "packer", "logistics", "distribution", "loader",
    ),
    "healthcare": (
        "nurse", "psw", "personal support", "caregiver", "patient",
        "medical", "healthcare", "clinical", "rn ", " rn", "lpn", "pca",
        "rpn", "dental", "pharmacy", "lab tech",
    ),
    "retail": (
        "retail", "cashier", "customer service", "sales associate",
        "barista", "server", "store clerk", "store associate",
    ),
    "admin": (
        "administrative", "receptionist", "secretary", "office assistant",
        "office manager", "clerk", "data entry", "executive assistant",
        "bookkeeper", "accountant", "accounting",
    ),
    "trades": (
        "electrician", "plumber", "welder", "carpenter", "mechanic",
        "hvac", "construction", "labourer", "millwright", "machinist",
        "apprentice",
    ),
}


def classify_role(target_role_text: str | None) -> str:
    """Return one of: software, warehouse, healthcare, retail, admin, trades, other."""
    if not target_role_text:
        return "other"
    t = target_role_text.lower()
    for category, keywords in ROLE_CATEGORIES.items():
        if any(kw in t for kw in keywords):
            return category
    return "other"


# ============================================================================
# Per-category slot priority
# ============================================================================
# Slots earlier in the tuple are asked first. Slots NOT in a category's
# list are NEVER proactively asked — they become "passive" slots, captured
# only if the user volunteers them. This keeps the chat tight: 3-4 focused
# questions max, then matches.
#
# What stays in the list (primary slots):
#   - target_role_text is implied (we wouldn't be here without it).
#   - skills_text / experience_text — always primary; the matching signal.
#   - work_type_preference — gates full-time vs part-time jobs.
#   - shift_preference — primary only for warehouse / healthcare / retail
#     (shift work is the norm there); excluded for software / admin
#     because asking a software dev about shifts feels off.
#   - education_text — primary where credentials matter (healthcare,
#     trades); secondary for software.
#
# What is INTENTIONALLY EXCLUDED (passive slots, captured if mentioned
# but never asked):
#   - preferred_location: product is SSM-only by design, so "where in
#     SSM" feels generic. If the user names a specific area, the
#     extractor still picks it up.
#   - transportation_text, availability_text, salary_expectation_text,
#     language_preferences: nice-to-have, not load-bearing for matching.
#     Polluting the question stream with them costs more than it adds.
PRIORITY_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "software": (
        "experience_text",       # what kind of dev — web/backend/mobile/etc
        "skills_text",           # specific stack — React, Python, APIs
        "work_type_preference",
        "education_text",
    ),
    "warehouse": (
        "skills_text",
        "experience_text",
        "shift_preference",
        "work_type_preference",
    ),
    "healthcare": (
        "education_text",        # certifications matter most here
        "experience_text",
        "skills_text",
        "shift_preference",
    ),
    "retail": (
        "experience_text",
        "skills_text",
        "shift_preference",
        "work_type_preference",
    ),
    "admin": (
        "experience_text",
        "skills_text",
        "work_type_preference",
        "education_text",
    ),
    "trades": (
        "skills_text",           # tickets / certs
        "experience_text",
        "education_text",        # apprenticeship level
        "work_type_preference",
    ),
    "other": (
        "experience_text",
        "skills_text",
        "work_type_preference",
    ),
}

# Back-compat alias — exported as PRIORITY_ORDER on the previous API.
PRIORITY_ORDER: tuple[str, ...] = PRIORITY_BY_CATEGORY["other"]


# ============================================================================
# Match readiness
# ============================================================================
MATCH_READY_SLOTS: frozenset[str] = frozenset({
    "target_role_text",
    "skills_text",
    "work_type_preference",
    "experience_text",
    "shift_preference",
    "preferred_location",
})

MIN_SLOTS_FOR_MATCH: int = 3
READY_SLOTS_FOR_STRONG: int = 5


# ============================================================================
# Role-aware prompt hints
# ============================================================================
# PROMPT_HINTS[slot][category] -> "natural phrasing with concrete examples".
# Always include an "other" fallback. The responder weaves the hint into a
# single conversational question — no bullets, no checklist.
PROMPT_HINTS: dict[str, dict[str, str]] = {
    "target_role_text": {
        "other": "what kind of work you're looking for",
    },
    "skills_text": {
        "software": (
            "what you've worked with so far — for example JavaScript, "
            "React, Python, SQL, APIs, WordPress, Java, .NET, or projects "
            "from school or work"
        ),
        "warehouse": (
            "what kind of warehouse work you've done — for example "
            "forklift, inventory, shipping/receiving, picking/packing, "
            "or team lead"
        ),
        "healthcare": (
            "what kind of healthcare work you've done — for example "
            "patient care, personal support, administration, or "
            "clinical roles"
        ),
        "retail": (
            "what tools or systems you've used — for example POS, "
            "cash handling, inventory, or specific products you sold"
        ),
        "admin": (
            "what office software and tasks you've used — for example "
            "Microsoft Word, Excel, scheduling, bookkeeping, QuickBooks"
        ),
        "trades": (
            "what tools, tickets, or certifications you have — for "
            "example welding ticket, electrical apprenticeship, or "
            "specific hand/power tools"
        ),
        "other": "the skills or tools you've used at work",
    },
    "experience_text": {
        "software": (
            "what kind of development you've done — web, backend, "
            "frontend, mobile, data, IT support, or something else"
        ),
        "warehouse": (
            "how long you've worked in warehouses and what roles "
            "you've held"
        ),
        "healthcare": (
            "what roles you've worked in healthcare and roughly how long"
        ),
        "retail": (
            "what retail or customer-service roles you've had and "
            "roughly how long"
        ),
        "admin": (
            "what office or admin roles you've had and roughly how long"
        ),
        "trades": (
            "what trades work you've done and at what level — "
            "apprentice, journeyman, or other"
        ),
        "other": "your previous jobs and roughly how long",
    },
    "education_text": {
        "software": (
            "any education, bootcamp, diploma, degree, or projects "
            "you'd like me to consider"
        ),
        "healthcare": (
            "any certifications, diplomas, or training you have — "
            "for example PSW, RN, RPN, first aid"
        ),
        "trades": (
            "any apprenticeships, tickets, or trade school you've done"
        ),
        "other": "your education or training",
    },
    "work_type_preference": {
        "software": (
            "whether you want full-time, part-time, contract, or "
            "remote/hybrid"
        ),
        "other": (
            "whether you're looking for full-time, part-time, or "
            "flexible work"
        ),
    },
    "shift_preference": {
        "warehouse": (
            "what shifts work for you — days, evenings, nights, "
            "or weekends"
        ),
        "healthcare": (
            "what shifts work for you — days, evenings, nights, "
            "or weekends"
        ),
        "retail": "what shifts you can do — days, evenings, or weekends",
        "other": "any shift preferences",
    },
    "preferred_location": {
        "other": (
            "if there's a particular area of Sault Ste. Marie or "
            "Algoma you'd prefer (or any travel limit)"
        ),
    },
    "transportation_text": {
        "warehouse": "how you get around — own vehicle, transit, or other",
        "trades": (
            "how you get around — own vehicle is sometimes needed "
            "for jobsites"
        ),
        "other": "how you get around — own car, bus, etc.",
    },
    "availability_text": {
        "other": "when you can start",
    },
    "salary_expectation_text": {
        "other": "any pay expectations (totally optional)",
    },
    "language_preferences": {
        "other": "any languages you speak comfortably at work",
    },
}


def prompt_hint(slot: str, target_role_text: str | None) -> str:
    """Role-aware natural-language phrasing for a slot.

    Returns the role-specific hint when available, else the 'other'
    fallback. Always returns a non-empty string.
    """
    category = classify_role(target_role_text)
    by_role = PROMPT_HINTS.get(slot, {})
    return by_role.get(category) or by_role.get("other") or slot


# ============================================================================
# Slot state checks
# ============================================================================
def _has(staged: StagedProfile, slot: str) -> bool:
    if slot == "skills_text":
        return bool(staged.skills_text) or bool(staged.skills)
    if slot == "language_preferences":
        return bool(staged.language_preferences)
    val = getattr(staged, slot, None)
    return bool(val)


def filled_match_slots(staged: StagedProfile) -> int:
    return sum(1 for s in MATCH_READY_SLOTS if _has(staged, s))


def completeness_band(staged: StagedProfile) -> str:
    """Return one of: 'low', 'minimum', 'ready'.

    target_role_text is required for any band above 'low'.
    """
    if not _has(staged, "target_role_text"):
        return "low"
    filled = filled_match_slots(staged)
    if filled >= READY_SLOTS_FOR_STRONG:
        return "ready"
    if filled >= MIN_SLOTS_FOR_MATCH:
        return "minimum"
    return "low"


def pick_slots_to_ask(staged: StagedProfile, *, max_n: int = 1) -> list[str]:
    """Pick up to `max_n` slots to ask about next, in role-priority order.

    Skips slots that are already filled or have been explicitly declined.
    Does NOT deprioritise slots we've asked before — if a slot is still
    primary-priority and the user hasn't either filled it or declined
    it, we ask it again. Repetition is bounded by:
      - small priority list (3-4 primary slots per role)
      - explicit decline detection ("skip that" → declined_slots)
      - state machine jumps to PRESENT_MATCHES once all primary slots
        are exhausted

    Without this, low-priority slots like education_text would jump
    ahead of a still-unfilled critical slot like experience_text just
    because experience_text was asked once. That made the chat feel
    erratic ("why is it asking education when I haven't said what kind
    of work I've done?").
    """
    declined = set(staged.declined_slots)

    # If we do not know what kind of work the user wants yet, ask that
    # first. The role-specific priority lists assume target_role_text has
    # already been captured; without this guard a greeting like "hi" would
    # incorrectly ask about previous jobs before asking the basic goal.
    if not _has(staged, "target_role_text") and "target_role_text" not in declined:
        return ["target_role_text"][:max_n]

    priority = PRIORITY_BY_CATEGORY.get(
        classify_role(staged.target_role_text),
        PRIORITY_BY_CATEGORY["other"],
    )

    out: list[str] = []
    for slot in priority:
        if _has(staged, slot):
            continue
        if slot in declined:
            continue
        out.append(slot)
        if len(out) >= max_n:
            break
    return out
