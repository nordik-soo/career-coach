"""Per-scenario raw dump for CP2 step-6 live verification.

Drives all three tier paths plus the no-soft-offer regression. Prints
the full reply, the final_move, and recommended_jobs count. No
assertions — visual inspection.
"""
from __future__ import annotations

import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8000"


def post(message, session_id):
    body = {"message": message}
    if session_id:
        body["session_id"] = session_id
    req = urllib.request.Request(
        BASE + "/v1/chat/messages",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))["data"]


def dump(label, data):
    print(f"\n>>> {label}")
    print(f"    final_move = {data.get('final_move')!r}")
    rj = data.get("recommended_jobs") or []
    print(f"    recommended_jobs = {len(rj)}")
    reply = data.get("reply", "")
    print(f"    --- reply ({len(reply)} chars) ---")
    print(reply)
    print("    --- end reply ---")


def run(label, target, skills, *, extra_turns=()):
    print(f"\n========== {label} ==========")
    d = post(f"I'm looking for work as a {target}", None)
    sid = d["session_id"]
    dump("turn 1 (target)", d)
    d = post(f"My skills: {skills}", sid)
    dump("turn 2 (skills)", d)
    if d.get("final_move") not in ("present_matches", "present_tiered_matches"):
        d = post("Show me jobs that fit.", sid)
        dump("turn 3 (nudge)", d)
    for n, msg in enumerate(extra_turns, start=len(extra_turns) + 3):
        d = post(msg, sid)
        dump(f"turn {n} (extra)", d)


def main():
    # Apply-today aim: cover all of Diamond J Farms' required skills.
    run("APPLY TODAY aim — accounting clerk full cover",
        "accounting clerk",
        "bookkeeping systems, journal entry posting, account reconciliation, "
        "invoice processing, accounts payable management, accounts receivable, "
        "QuickBooks, Excel, payroll, bank reconciliation")

    # Worth a try regression — same as original scenario A.
    run("WORTH A TRY regression — accounting clerk partial",
        "accounting clerk",
        "I use QuickBooks daily, handle accounts payable and receivable, "
        "run payroll, and I'm strong in Excel and bookkeeping. Five years "
        "in a small business.")

    # Sideways aim attempt 1 — admin/research skills, off-DB target.
    run("SIDEWAYS aim #1 — admin/research skills, off-DB target",
        "marine biologist",
        "organizational skills, written communication, attention to detail, "
        "Microsoft Excel, project management, problem-solving, "
        "data analysis, report writing")

    # Sideways aim attempt 2 — narrow specialist anchor on case management.
    # Three skills (clears ADJACENT_MIN_USER_SKILLS=3), narrow enough not
    # to score Stretch on any single job, anchor strength on the most
    # widely-shared social-services skill (case management = 4 jobs).
    run("SIDEWAYS aim #2 — narrow social-services anchor",
        "park ranger",
        "case management, written communication, organizational skills")

    # Sideways aim attempt 3 — admin Office anchor, target outside DB.
    run("SIDEWAYS aim #3 — Microsoft Office anchor, off-DB target",
        "barber",
        "Microsoft Office proficiency, Microsoft Excel, "
        "customer service, attention to detail")

    return 0


if __name__ == "__main__":
    sys.exit(main())
