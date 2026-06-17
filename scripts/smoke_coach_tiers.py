"""AR-9.feat.coach-tiers CP2 step 6 — live smoke driver.

Three scenarios exercise the present_tiered_matches outcome end-to-end:

  scenario A — accounting clerk (Strong + maybe Stretch)
  scenario B — warehouse/forklift  (Strong + maybe Sideways)
  scenario C — generic office       (Sideways-only or no surface)

For each scenario the driver threads a fresh session through 2-3 turns,
captures the final reply, and asserts:
  - final_move == "present_tiered_matches"
  - recommended_jobs == [] (UI stays prose-only)
  - reply contains at least one of the tier headings
  - reply contains at least one https:// URL (URL grounding intact)
  - reply contains NO internal tokens or scaffolding leakage

Prints per-turn move + truncated reply, then a final pass/fail tally.
"""
from __future__ import annotations

import json
import re
import sys
import urllib.request
from dataclasses import dataclass

BASE = "http://127.0.0.1:8000"

TIER_HEADINGS = (
    "**Apply today",
    "**Worth a try",
    "**Sideways move",
)

INTERNAL_TOKEN_PATTERNS = (
    r"\bfinal_move\b",
    r"\bArbiterAction\b",
    r"\bpresent_tiered_matches\b",
    r"\bTieredEvidence\b",
    r"\b(?:strong|stretch|adjacent)_records\b",
    r"<\s*system\s*reminder",
)


@dataclass
class Scenario:
    label: str
    target: str
    skills_text: str
    nudge: str = "Show me jobs that fit."
    expect_present_tiered: bool = True


SCENARIOS = [
    Scenario(
        label="A — accounting clerk",
        target="accounting clerk",
        skills_text=(
            "I use QuickBooks daily, handle accounts payable and "
            "receivable, run payroll, and I'm strong in Excel and "
            "bookkeeping. Five years in a small business."
        ),
    ),
    Scenario(
        label="B — warehouse / forklift",
        target="warehouse worker",
        skills_text=(
            "I have forklift operation, WHMIS certification, "
            "shipping and receiving, and inventory management. "
            "Eight years on the floor."
        ),
    ),
    Scenario(
        label="C — generic office (light)",
        target="office assistant",
        skills_text="I'm good with computers and customer service.",
        # generic profiles often need no surface at all — accept either
        expect_present_tiered=False,
    ),
]


def post(message: str, session_id: str | None) -> dict:
    body = {"message": message}
    if session_id:
        body["session_id"] = session_id
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        BASE + "/v1/chat/messages",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        envelope = json.loads(resp.read().decode("utf-8"))
    return envelope.get("data", envelope)


def show_turn(label: str, user_msg: str, data: dict) -> None:
    move = data.get("final_move", "?")
    reply = data.get("reply", "")
    shown = reply if len(reply) <= 320 else reply[:320] + "..."
    print(f"\n--- {label} ---")
    print(f"USER: {user_msg}")
    print(f"MOVE: {move}")
    print(f"BOT : {shown}")


def run_scenario(s: Scenario) -> tuple[bool, list[str]]:
    print(f"\n========== {s.label} ==========")
    issues: list[str] = []

    d = post(f"I'm looking for work as a {s.target}", None)
    sid = d.get("session_id")
    show_turn("turn 1 (target)", f"...{s.target}", d)

    d = post(f"My skills: {s.skills_text}", sid)
    show_turn("turn 2 (skills)", s.skills_text, d)

    final = d
    if final.get("final_move") not in ("present_matches", "present_tiered_matches"):
        d = post(s.nudge, sid)
        show_turn("turn 3 (nudge)", s.nudge, d)
        final = d

    move = final.get("final_move", "")
    reply = final.get("reply", "")
    recommended = final.get("recommended_jobs") or []

    if s.expect_present_tiered:
        if move != "present_tiered_matches":
            issues.append(f"expected final_move=present_tiered_matches, got {move!r}")
        if recommended:
            issues.append(
                f"recommended_jobs must be [] for tiered move, got {len(recommended)} cards"
            )
        if not any(h in reply for h in TIER_HEADINGS):
            issues.append("reply missing all three tier headings")
        if "https://" not in reply:
            issues.append("reply missing https:// URL grounding")
    else:
        # Scenario C may produce no surface, sideways-only, or full tiered —
        # only require that IF a tiered surface fires, the contract holds.
        if move == "present_tiered_matches":
            if recommended:
                issues.append(
                    f"recommended_jobs must be [] for tiered move, got {len(recommended)}"
                )
            if not any(h in reply for h in TIER_HEADINGS):
                issues.append("tiered move but reply missing tier headings")

    for pat in INTERNAL_TOKEN_PATTERNS:
        if re.search(pat, reply, flags=re.IGNORECASE):
            issues.append(f"reply leaked internal token /{pat}/")

    if issues:
        print(f"\n[FAIL] {s.label}")
        for i in issues:
            print(f"    - {i}")
        return False, issues
    print(f"\n[PASS] {s.label}")
    return True, []


def main() -> int:
    print(f"smoke driver hitting {BASE}/v1/chat/messages")
    results = [run_scenario(s) for s in SCENARIOS]
    passes = sum(1 for ok, _ in results if ok)
    total = len(results)
    print("\n" + "=" * 60)
    print(f"SMOKE RESULT: {passes}/{total} scenarios passed")
    print("=" * 60)
    return 0 if passes == total else 1


if __name__ == "__main__":
    sys.exit(main())
