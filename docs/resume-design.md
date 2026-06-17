# Resume Upload — Design

Status: approved · Target: Sprint 1 (backend MVP, provable via curl)

The chat is the relationship; the resume is denser input. Users drop a PDF in
the chat the way they would in ChatGPT — no separate workflow, no profile
form. Everything the system says is grounded in either the parsed resume or
the chat conversation. Haiku narrates; the backend owns the facts.

---

## 1. User flow (MVP slice)

```
user opens chat
    │
    │  drops resume.pdf into the input  (or drags it onto the panel)
    ▼
backend
    │  • text extraction (PDF/DOCX/TXT → string)
    │  • evidence-bound Haiku → resume_facts_json
    │  • derive flat StagedProfile slots
    ▼
state = RESUME_REVIEW
    │
    │  assistant: "I read your resume. I see [skills, jobs, education].
    │             Anything to add, fix, or remove?"
    ▼
user replies (corrections are layered, never destructive — see §4)
    ▼
state machine resumes normal flow → PRESENT_MATCHES with grounded explanations
```

Provable end-to-end via `curl` before any frontend exists.

---

## 2. `resume_facts_json` schema

Canonical structured parse. Every fact carries verbatim evidence from the
resume text and a `source` field. Missing fields are **omitted, not
null-filled** — absence is information the system uses.

```json
{
  "version": 1,
  "extracted_at": "2026-05-22T18:00:00Z",
  "extractor_version": "resume-haiku-v1",

  "skills": [
    {
      "fact_id": "f_skill_001",
      "name": "React",
      "evidence": "Built customer dashboards with React and Redux",
      "confidence": 0.92,
      "source": "resume"
    }
  ],

  "work_history": [
    {
      "fact_id": "f_work_001",
      "title": "Software Engineer",
      "employer": "Acme Inc.",
      "start_year": 2020,
      "end_year": 2024,
      "is_current": false,
      "summary": "Built customer-facing dashboards and APIs.",
      "evidence": "Software Engineer at Acme Inc., 2020-2024",
      "source": "resume"
    }
  ],

  "education": [
    {
      "fact_id": "f_edu_001",
      "credential": "BSc Computer Science",
      "institution": "University of X",
      "year": 2018,
      "country": null,
      "evidence": "BSc Computer Science, University of X, 2018",
      "source": "resume"
    }
  ],

  "certifications": [
    {
      "fact_id": "f_cert_001",
      "name": "AWS Solutions Architect",
      "issuer": "Amazon Web Services",
      "year": 2022,
      "evidence": "AWS Certified Solutions Architect (2022)",
      "source": "resume"
    }
  ],

  "projects": [
    {
      "fact_id": "f_proj_001",
      "name": "Portfolio site",
      "summary": "...",
      "evidence": "...",
      "source": "resume"
    }
  ],

  "languages": ["English", "Punjabi"],

  "summary_signals": {
    "total_years_estimate": 6,
    "primary_field": "software",
    "evidence_for_years": "Software Engineer at Acme 2020-2024 + Junior Dev 2018-2020"
  }
}
```

### Hard rules

- Every fact (except scalars in `languages` and `summary_signals`) has:
  - `fact_id`: stable string, used to reference the fact from suppression / correction layers.
  - `evidence`: substring of the original resume text, ≥4 chars, validated by backend.
  - `source`: `"resume"` for parser output. Chat-added facts later use `"chat"`.
- `confidence` only on skills (other fields are present-or-absent — no scoring).
- `summary_signals` is **derived by backend math**, not extracted by the LLM. Re-derived on every correction.
- **No contact information stored.** Name, phone, email, address, links — none of it is extracted into the JSON or persisted anywhere. The extractor's prompt forbids it; if Haiku returns those fields anyway, the backend drops them.
- Confidence floor `MIN_EXTRACTION_CONFIDENCE = 0.60` (same as job-skill extraction).
  - Skills below the floor are kept in `resume_facts_json` for transparency.
  - They are **not** derived into `StagedProfile.skills` (so they don't count toward matching).

### Flat StagedProfile slots are a derived view

```
skills_text       ← top N skills.name (deduped, filtered by confidence floor)
experience_text   ← work_history[0..2] title/employer/summary joined
education_text    ← education[0].credential + institution + year
```

`resume_facts_json` is the single source of truth. The flat slots are
re-derived on every correction. Don't write to the flat slots directly when
a resume is present.

---

## 3. Storage policy

| Session mode | Storage | Stored content | TTL | Cleared on |
|---|---|---|---|---|
| **Anonymous, cookie session** | Signed cookie blob | **Compact** `resume_facts_json` only — raw text + evidence strings are stripped (4 KB cookie cap + privacy). See `resume.compact_facts()`: keeps `fact_id` / `name` / `title` / `employer` / dates / `source` so suppression and the RESUME_REVIEW summary keep working across turns; drops verbatim evidence so the responder can't quote the resume word-for-word. | 30 min sliding | Browser clear, expiry, explicit clear |
| **Anonymous, Redis session** | Redis at `sb:session:<sid>` | Full `resume_text` + full `resume_facts_json` (with evidence) | 30 min sliding | TTL, explicit delete |
| **Authenticated (post-consent)** | `profile.user_profile` columns | `resume_text`, `resume_filename`, `resume_parsed_at`, full `resume_facts_json` | Until DELETE /v1/profiles/me | Delete cascade scrubs all four columns |

### Invariants

- **The binary file (PDF/DOCX) is never persisted.** It is extracted to text
  in memory and discarded as soon as the parse returns.
- Raw `resume_text` persists only post-consent OR in Redis-mode anonymous
  sessions. Cookie-mode anonymous sessions never hold raw text.
- `resume_facts_json` is what the responder reads at runtime.
- Delete cascade (existing pattern in `routes/profiles.py`) extends to scrub
  all four resume columns. The delete-cascade test asserts the columns are
  NULL after delete.
- Consent flush: when the user signs in via `/v1/consent`, the anonymous
  session's `resume_text` (if present in Redis) and `resume_facts_json`
  flush to Postgres in the same transaction as the rest of the staged data.

### Deployment guidance

- **Dev**: cookie mode is fine. Resume upload works; raw text isn't retained
  across refreshes, but the parsed facts are.
- **Production**: use Redis. Document this clearly. Cookie mode is not a
  long-term answer for resume sessions.

---

## 4. Correction & suppression layers

Parser output is the **record of what we read**. User corrections layer on
top, never overwrite. This preserves audit (so we can show the user what we
originally parsed) while respecting their authority over their own profile.

### Three layers, evaluated in order

```
resume_facts_json   ← parser record (immutable after parse)
       +
chat-sourced facts  ← user added during conversation
                      stored in StagedProfile.* with source="chat"
       −
suppressed_fact_ids ← user said "that's not mine" / "remove that"
                      stored in StagedProfile.suppressed_fact_ids: list[str]
```

The **effective profile** the matcher consumes:

```python
effective_skills(staged) =
    (resume_facts.skills − suppressed) ∪ chat_facts.skills

effective_work_history(staged) =
    resume_facts.work_history − suppressed

effective_preferences(staged) =
    staged.work_type_preference, shift_preference, etc.
    (always from chat — preferences aren't in resumes)
```

### Correction patterns

| User says | What the system does | Layer touched |
|---|---|---|
| "I also know Docker" | Append `{name: "Docker", source: "chat"}` to `StagedProfile.skills`. | chat additions |
| "I prefer day shift" | Set `staged.shift_preference = "days"`. | preferences |
| "I was actually a Senior Engineer" | Add chat-source work_history correction (suppress original + add corrected fact, OR add a `correction_note` — see §4.1). | corrections |
| "I never worked there" / "Remove that job" | Add `fact_id` to `staged.suppressed_fact_ids`. Original stays in `resume_facts_json` for audit. | suppression |
| "skip team-lead roles" | Add `"team_lead"` to `declined_slots` as a preference filter, no change to facts. | preferences |

### §4.1: How corrections work in v1 (simple form)

For MVP, corrections are modeled as **suppress + add**:

- "I was Senior, not Junior" →
  1. Suppress the parser's "Junior Engineer at Acme" fact (`fact_id` to `suppressed_fact_ids`).
  2. Add a chat-sourced "Senior Engineer at Acme" fact to `StagedProfile.work_history` (or equivalent chat layer).
- Both facts remain visible to the audit / RESUME_REVIEW summary, but only the
  un-suppressed one feeds matching.

Per-field overrides (e.g., correcting just the title without touching the
employer or dates) can come later as a richer correction model. v1 keeps the
data model flat.

---

## 5. Responder grounding

The responder user_block grows to four grounded sources:

```
USER_MESSAGE: ...
NEXT_ACTION:  ...
ROLE_CATEGORY: software | warehouse | healthcare | retail | admin | trades | other

RESUME_FACTS:   (effective view — facts ∪ chat-additions − suppressions)
                Present only when the user uploaded a resume.
CHAT_FACTS:     (StagedProfile slot values from chat-only extraction)
RESULTS:        (match-engine output — only when show_matches=True)
TRAINING:       (training recommender output)
NEXT_SKILL:     (only when show_matches=True)
```

**Grounding rule (one line):**

> Every factual claim — about the user, a job, training, or the local market —
> must trace to RESUME_FACTS, CHAT_FACTS, RESULTS, or TRAINING. Coaching is
> welcome. Justifying coaching with market / employer claims requires a
> grounded fact.

### Examples

- ✅ *"Your forklift cert and 5 years at Acme line up with the Cygnus posting's top requirements."* (grounded in RESUME_FACTS + RESULTS)
- ✅ *"You may want to make your team-lead experience more visible."* (coaching, no fact claim)
- ❌ *"Sault Ste. Marie warehouses pay $25/hr."* (invented stat)
- ❌ *"Your Bangladesh BSc is equivalent to a Canadian Bachelor's."* (credential equivalence — out of scope)
- ❌ *"You may qualify for RCIP under this NOC."* (immigration advice — out of scope)

### Scope boundaries (responder prompt rules)

These are explicit in the responder prompt and enforced by `_policy_ok`:

- No credential equivalence claims.
- No immigration, legal, medical, or financial advice. Redirect to SCCC / YMCA newcomer services.
- No labour-market statistics not in stored data.
- No dollar amounts (we have no salary data; user reads URLs).
- No mentions of Job Bank, Statistics Canada, "national average", or other federal feeds (SSM-only product).

---

## 6. State machine: `RESUME_REVIEW`

### New state

```python
STATE_RESUME_REVIEW = "resume_review"   # awaiting user confirmation of parsed facts
```

### New action

```python
ACTION_PRESENT_RESUME_FACTS = "PRESENT_RESUME_FACTS"
```

### Transitions

```
any state + resume uploaded
    → state = RESUME_REVIEW
    → action = PRESENT_RESUME_FACTS
    → show_matches = False
    → responder narrates effective resume view + asks for corrections

RESUME_REVIEW + user message (no new upload)
    → re-run chat extractor on the message
    → apply corrections to suppression / chat-facts layers
    → re-derive effective profile + flat slots
    → recompute completeness band
    → flow into standard state machine (PRESENT_MATCHES if ready, etc.)

RESUME_REVIEW + new resume uploaded
    → replace resume_facts_json (new parse wins)
    → preserve declined_slots (preferences) but DROP suppressed_fact_ids
      (they referred to fact_ids from the old parse)
    → loop back to RESUME_REVIEW
```

---

## 7. Implementation skeleton

```
skillbridge/
  resume/
    __init__.py
    parse.py           # PDF/DOCX/TXT → text. pdfplumber + python-docx.
                       # Returns ("", reason) on scanned-PDF / unreadable.
    extract.py         # Evidence-bound Haiku → resume_facts_json.
                       # Same evidence-validation pattern as chat extractor.
                       # Strips contact info, drops sub-confidence skills.
    derive.py          # Effective profile → flat StagedProfile slots.
                       # Pure function. Idempotent on every correction.
  chat/
    intake_state.py    # + STATE_RESUME_REVIEW, ACTION_PRESENT_RESUME_FACTS.
    responder.py       # + RESUME_FACTS block (effective view) in user_block.
                       # + scope-boundary policy in _policy_ok.
    prompts.py         # + RESUME_REVIEW prompt fragment.
    handler.py         # + resume upload entry point.
                       # + corrections wiring (suppression, chat additions).
  session/
    staging.py         # + resume_facts_json, suppressed_fact_ids fields.
                       # + effective_* helper methods.
  routes/
    chat.py            # multipart/form-data support. File goes to
                       # skillbridge.resume.parse → extract → merge.
sql/
  schema.sql           # ALTER profile.user_profile ADD COLUMN:
                       #   resume_text TEXT,
                       #   resume_filename TEXT,
                       #   resume_parsed_at TIMESTAMPTZ,
                       #   resume_facts_json JSONB
                       # Extend DELETE /me UPDATE to scrub these four.
tests/
  test_resume_parse.py       # Empty PDF, scanned PDF, oversize, mixed lang.
  test_resume_extract.py     # Ungrounded drop, contact-info strip,
                             # confidence-floor enforcement.
  test_resume_review.py      # State transition, additive corrections,
                             # suppression behaviour.
  test_delete_cascade.py     # (extend) — resume columns scrubbed on delete.
```

---

## 8. Failure modes

| Scenario | Behaviour |
|---|---|
| Scanned PDF (no extractable text) | API returns success with `parse_warning: "no_text"`. Responder says: *"I couldn't read text from that file. Could you paste your resume text into the chat instead?"* |
| File > 5 MB | API returns 413. Frontend shows inline error before send. |
| LLM extraction returns empty | Responder: *"I had trouble parsing the resume. Tell me a bit about your background and we'll go from there."* Fall back to normal chat intake. |
| Foreign-format resume (sparse fields) | Show whatever extracted. RESUME_REVIEW prompt asks user to fill gaps. |
| Multipart but missing file field | API 400. |
| File present, message text empty | OK. The confirmation turn fires; the user doesn't need to type a message to upload. |
| Re-upload (resume already in session) | Replace `resume_facts_json` (new parse wins). Drop `suppressed_fact_ids` (they pointed at old IDs). Preserve `declined_slots` (preferences). |
| Contact info present in resume | Extractor's prompt forbids returning name/phone/email/address. Backend strips them if returned anyway. Never persisted. |

---

## 9. Explicitly out of scope for MVP

- OCR for scanned PDFs (tell user to paste text instead).
- Multi-file uploads.
- Cover letter / resume rewrite generation.
- Apply-from-platform (user navigates to SCCC source URL).
- Resume re-editing UI (corrections happen via chat).
- Embedding-based JD similarity. v1 stays with skill overlap + title match.
- Per-field correction model (e.g., changing only a job title without
  touching dates). v1 uses suppress + add. Richer correction model can come
  later if real users hit the limit.
- Resume version history (latest parse wins; old parse is replaced).
- Multi-resume per profile (one resume per session).

---

## 10. Versions to bump on first ship

| Constant | New value | Reason |
|---|---|---|
| `EXTRACTOR_VERSION_LLM` | `"llm-haiku-extractor-v1.2.0"` | Resume-aware extractor; same evidence-binding rules, new content type. |
| `CHAT_PROMPT_VERSION` | `"chat-prompt-v1.2.0"` | Responder prompt gains RESUME_FACTS block + scope boundaries. |
| `ENGINE_VERSION_JOB_MATCH` | unchanged | Match engine is untouched; resume facts populate existing inputs. |
| New | `EXTRACTOR_VERSION_RESUME = "resume-haiku-v1"` | Recorded in `resume_facts_json.extractor_version` so future re-parses can be detected. |

---

## 11. Open questions resolved (for the record)

| Question | Decision |
|---|---|
| Cookie session store for resumes? | Yes for dev, no raw text. Production should use Redis. |
| Suppression model? | Layered: parser record is immutable; suppressions are a separate `fact_id` list; chat additions live alongside with `source="chat"`. |
| Confidence floor for resume skills? | `MIN_EXTRACTION_CONFIDENCE = 0.60` same as job extraction. Below-floor skills stay in `resume_facts_json` but don't feed matching. |
| Contact info? | Never extracted, never stored. Extractor strips. |
| Per-field corrections? | Out of scope for v1. Suppress + add covers it. |
