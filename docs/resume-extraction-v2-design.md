# Resume Extraction v2 — Design Direction

Status: draft · Target: next sprint after matching v2 stabilizes

The matching engine is now production-grade: deterministic core + bounded
LLM, evidence-bound, with stage-labelled match strength. But the data
substrate it consumes — parsed resume facts — is still an LLM-only
pipeline. Live test 3 (electrical journeyman CV) demonstrated the
failure mode cleanly: a CV with unambiguous WORK HISTORY and
CERTIFICATIONS sections parsed to `work=0, edu=0, skills=24`, and the
chat then had to ask the user to re-describe work history that was
literally on the resume.

This document scopes the next sprint: **demote Haiku from primary
extractor to constrained normalizer**, the same architectural move
that just made matching v2 honest.

---

## 1. Where the resume extractor is today

```
File upload ──→ parse_resume()          ──→ resume_text (PDF/DOCX → plain text)
                                              │
                                              ▼
                extract_resume_facts()   ──→ Haiku-only extraction
                                              │   evidence-bound prompt;
                                              │   returns full facts JSON
                                              ▼
                derive_*()                ──→ flat slots (skills_text,
                                              experience_text, etc.)
```

The middle step is the weak link. Haiku reads the whole resume and
returns a structured JSON. When it works, it works beautifully — when
it misses a section (or returns malformed JSON, which it has, per the
existing fallback path), the downstream pipeline gets an empty list
where there should be data, and the matcher / chat have no way to
know they're operating on incomplete input.

**What live test 3 demonstrated:**

- Input: `cv_03_electrical_journeyman_strong.pdf` — has clear sections
  (WORK HISTORY, CERTIFICATIONS, EDUCATION, SKILLS), each with
  structured content (job title + employer + date range; cert name +
  issuer).
- Haiku output: `skills=24 work=0 edu=0`.
- Chat behavior: asked the user to walk through work history, because
  the intake state machine correctly observed `experience_text` was
  unset. The chat couldn't tell the difference between "user has no
  work history" and "we failed to parse it from the resume."

That last point is the most important. The current pipeline has **no
confidence signal between extraction and chat**. A failed parse is
indistinguishable from a thin resume.

---

## 2. Deterministic section parser

The first layer of the new pipeline. Reads `resume_text` and produces
a typed section map:

```python
@dataclass
class ResumeSections:
    work_history: SectionText | None     # raw text + char span + confidence
    certifications: SectionText | None
    education: SectionText | None
    skills: SectionText | None
    summary: SectionText | None
    unsectioned_text: str                # everything not matched to a section
    detection_confidence: float          # 0.0-1.0, how confident we are
                                          # that this resume has structure
```

**Detection heuristics** (layered, each contributing a confidence
delta — the parser uses whatever combination works for each resume):

1. **Bold + ALL CAPS header lines** (via `pdfplumber` font-weight
   signal where available). Strongest signal.
2. **All-caps single-line headers** matching common keywords
   (`WORK HISTORY|EXPERIENCE|EMPLOYMENT`,
   `CERTIFICATIONS|CERTIFICATES|LICENSES`,
   `EDUCATION|ACADEMIC BACKGROUND`,
   `SKILLS|TECHNICAL SKILLS|COMPETENCIES`,
   `SUMMARY|PROFILE|OBJECTIVE`). Edit-distance ≤ 2 to tolerate typos.
3. **Title-cased headers** on their own line (`Work History`,
   `Certifications`, etc.) — weaker than all-caps but still a signal.
4. **Horizontal rules / underscores / `===`** dividers below a
   candidate header — boosts confidence.
5. **Blank-line block boundaries** — used to terminate a section when
   no explicit next header is detected.

**Fallback path:** when `detection_confidence < 0.3`, the resume has
no detectable structure (or is heavily designed / single-column
narrative). In that case, the rule extractors run against
`unsectioned_text` as a whole — they're already designed to be
section-agnostic when needed (see §3).

**What this layer does NOT do:**

- Does not interpret the content of any section. That's §3.
- Does not call the LLM. Pure heuristic / regex / `pdfplumber`
  metadata.
- Does not throw on unrecognized layouts. Worst case: low
  detection_confidence + everything in unsectioned_text. Pipeline
  continues with degraded confidence.

---

## 3. Rule extractors per section

Each section gets a dedicated rule extractor. Each extractor produces
a **list of structured entries with a confidence score per entry** and
preserves the verbatim source span so evidence-bound validation stays
intact downstream.

### 3.1 Work history extractor

Scans for repeated `(job_title, employer, date_range)` blocks. Common
patterns:

```
Electrical Journeyman
North Shore Industrial Electrical
2020 to present

Construction Electrician
Algoma Commercial Contractors
2016 to 2020
```

Detection rules:

- **Date range first**: regex for `YYYY[-/]YYYY|present`,
  `Mon YYYY[-/]Mon YYYY|present`, `Mon YYYY – present`, etc. A
  detected date range anchors a work entry.
- **Title + employer scan**: walks backward and forward from each
  date range to find the title and employer lines. Heuristic:
  title is usually the first non-empty line in the block; employer
  is between the title and the date range, often on its own line.
- **Bullet stripping**: lines starting with `•`, `-`, `*` are
  description bullets, not titles/employers.
- **Job-counter consistency**: when multiple date ranges are found,
  each anchors one work entry; the parser should produce N work
  entries from N date ranges.

### 3.2 Certifications extractor

Each non-empty line in the CERTIFICATIONS section becomes one
certification entry. Verbatim preservation is the rule —
certifications are domain-specific and short; the parser must NOT
attempt to summarize them.

```
309A Construction and Maintenance Electrician Certificate of Qualification
Valid Ontario driver's licence
WHMIS 2015
Working at Heights
First Aid and CPR
Lockout/tagout training
```

→ 6 certification entries, each preserving the source line verbatim.

### 3.3 Education extractor

Patterns:
- `<credential>, <institution>, <year>` (comma-separated)
- `<credential>\n<institution>\n<year>` (multi-line block)
- Lines containing common degree tokens: `BSc|MSc|BA|MA|PhD|Diploma|
  Certificate|Bachelor|Master|Apprenticeship`

Each detected entry: `(credential, institution, year, evidence_span)`.

### 3.4 Skills extractor

The skills section is the easiest: usually a comma-separated or
bullet-listed flat list. Rules:

- Split on `,`, `;`, `•`, `-`, `\n` (any of them — whichever produces
  more than one non-empty token).
- Strip surrounding whitespace, leading bullets, trailing punctuation.
- Preserve case (matching engine canonicalizes downstream).
- One entry per cleaned token. No further interpretation.

### 3.5 What runs when sections aren't detected

When `detection_confidence < 0.3`, all four extractors run against
`unsectioned_text`. The work-history and education extractors are
robust to this — they anchor on date ranges and credential tokens
respectively, neither of which depends on section context. The
certifications extractor is more brittle (a line-by-line preservation
rule needs a known section); in unsectioned mode, it runs a
**conservative regex pass** for known certification patterns
(`Class [A-Z] licence`, `\d{3}[A-Z]`, `WHMIS`, `First Aid`, `CPR`,
etc.) instead.

The skills extractor in unsectioned mode looks for "Skills:" /
"Technical Skills:" inline prefixes or returns nothing (cleaner than
guessing).

---

## 4. LLM cleanup boundary

Haiku's role contracts dramatically. **It is a normalizer, not an
extractor.** Specifically:

### 4.1 Allowed LLM operations

| Input | Allowed LLM op | Output |
|---|---|---|
| Date strings like `"Jan 2020 - present"`, `"01/2020-12/2023"` | Normalize to ISO-ish | `start: "2020-01"`, `end: "present"` |
| Certification text like `"309A Construction and Maintenance Electrician Certificate of Qualification"` | Label kind | `kind: "certification"`, `issuer: "Ontario"` if confidently inferrable from text |
| Comma-separated skill string `"Python, SQL, PyTorch, AWS"` | Split into items | 4 separate skill entries |
| Verbatim job title `"Lead Mechanical Engineer (Acting)"` | None — preserve as-is | same string |

### 4.2 Forbidden LLM operations

| Forbidden | Why |
|---|---|
| Inferring missing facts | Hallucination risk — if the resume doesn't say it, we can't claim it |
| Cross-section reordering | The deterministic order is the source of truth |
| Dropping deterministic-parser output | If §3 found 6 certifications, Haiku keeps all 6 |
| Summarizing or rewriting verbatim text | Loses evidence-bound traceability |
| Deduplicating entries | Two entries may be intentionally distinct (e.g. two "Electrician" roles at different employers) |

### 4.3 Enforcement mechanism

Each normalization call is **scoped to a single field** (one date
range, one cert line, one skill string). Haiku receives:

```
INPUT: <single field value>
OPERATION: normalize_date | label_certification | split_skill_list

Return ONLY the normalized result. No commentary. No additional
fields. If the input is empty or unrecognizable, return it unchanged.
```

The caller then validates the output against the input — if Haiku
returned a date that contains characters not in the input string (a
hallucinated year, say), the caller logs a warning and uses the raw
input verbatim. Defense in depth.

---

## 5. Completeness guard + handler integration

The new pipeline produces, alongside the facts JSON, an **extraction
warnings list**:

```python
@dataclass
class ExtractionWarnings:
    work_history_detected_but_unparsed: bool
    certifications_detected_but_unparsed: bool
    education_detected_but_unparsed: bool
    sections_undetected: bool        # detection_confidence < 0.3
    parser_messages: list[str]       # human-readable specifics
```

These warnings flow into the chat handler. The handler decides what
to ask the user based on them:

| Warning | Handler behavior |
|---|---|
| `work_history_detected_but_unparsed=True` | Acknowledge: "I can see work history in your resume, but I couldn't read the job dates cleanly. Can you confirm your most recent role and how long you were there?" |
| `certifications_detected_but_unparsed=True` | "I see you have certifications listed — would you mind walking me through the most relevant ones?" |
| `sections_undetected=True` | Don't promise resume coverage; ask for the user's most relevant facts in plain language |
| All warnings false but a field is still empty (`work_history=[]`) | Current behavior: ask as if no resume provides that field |

This closes the chat UX gap: today the bot can't tell the difference
between "user didn't include work history" and "we failed to parse
it." After this slice, those two cases produce different prompts.

---

## 6. What we don't build yet

| Item | Reason for deferral |
|---|---|
| **OCR for scanned PDFs** | Already handled by the `no_text` parse warning at the `parse.py` layer. Adding OCR is a separate, expensive sprint with its own privacy considerations |
| **DOCX style parsing** (bold-headers via `python-docx`) | `pdfplumber` covers PDFs; DOCX falls back to plain text. The section parser's heuristics (all-caps, keyword matching) cover most cases. Add only if real chats show DOCX as a routine failure mode |
| **Multilingual extraction** (French resumes via OaSIS lexicon) | Defer until bilingual responder support is greenlit. Same scope decision as matching v2 §2 |
| **Semantic deduplication of skills** | Today's alias map + matching v2 canonicalization is enough; semantic dedup at extraction time would re-introduce the LLM-as-primary risk |
| **Section ordering inference** | If the resume puts EDUCATION before WORK HISTORY, the section parser detects both — no further inference needed. We don't need to "understand" the order |
| **Cover-letter parsing** | Out of scope; resume only |

---

## 7. Recommended pickup order

Each piece is independently shippable. Sign-off between each, same
shape as matching v2.

1. **Section parser** (`skillbridge/resume/sections.py`). ~2-3 days.
   Heuristic detection with confidence scoring. Validation: a unit
   test per synthetic CV in `docs/test-resumes/` asserts the expected
   sections are found with confidence ≥ 0.6.

2. **Rule extractors per section** (`skillbridge/resume/rules/`).
   ~2-3 days. One module per section. Each produces structured
   entries with verbatim evidence spans. Validation: unit tests
   against synthetic CV section text; integration test against the
   5 PDFs in the test pack confirms expected work_history /
   certifications / education / skills lists per the README.

3. **LLM cleanup with bounded interface** (`skillbridge/resume/normalize.py`).
   ~1 day. Three Haiku calls (date, certification, skill-split) with
   strict single-field input/output. Includes the output-validates-
   against-input safety check. Falls back to verbatim raw input on
   any anomaly.

4. **Completeness guard + handler integration**. ~1 day. Adds
   `ExtractionWarnings` to the resume_upload response payload.
   Handler reads the warnings and routes intake questions
   accordingly. Validation: live chat test against
   `cv_03_electrical_journeyman_strong.pdf` produces the
   "I can see work history in your resume..." prompt instead of
   the current generic ask.

5. **Replace `extract_resume_facts()` with the new pipeline**.
   ~1 day. Keep the existing function signature so handler.py
   doesn't change. Internal implementation now: parse → sections →
   rules → normalize → derive (existing). The OLD LLM-only path
   stays available behind an env flag for one release cycle
   (`RESUME_EXTRACTOR=v1|v2`, default v2) for rollback safety.

6. **5-CV fixture suite + extraction regression tests**. ~2 days.
   Each synthetic CV becomes an integration fixture. Each fixture
   asserts the documented expected output from the README's table.

Total: ~9-11 days. Stop-and-review checkpoints after each numbered
item. Don't bundle.

---

## 8. Open questions

- **Confidence threshold for "section detected"?** Recommend 0.6 to
  count as detected, 0.3-0.6 as ambiguous (rule extractors still run
  but warnings get flagged), <0.3 as undetected (fall back to
  unsectioned mode + warnings flagged).
- **Should LLM normalization fail-open or fail-closed?** Recommend
  fail-open: if Haiku is unreachable / returns junk, use the
  deterministic output verbatim and log a warning. The whole point
  of v2 is that LLM isn't load-bearing.
- **Do we re-run extraction on existing parsed resumes?** No — only
  on new uploads. Existing `resume_facts_json` rows stay as-is until
  the user re-uploads. (Same pattern as the matching v1.2 → v1.3
  extractor version migration.)
- **Test pack maintenance.** The 5 synthetic CVs are the contract.
  When real chats surface new failure modes, add a 6th, 7th, etc. —
  same alias-map evidence-driven discipline. Don't add fixtures
  speculatively.

---

## Decision required

Sign off on the **direction**: deterministic section parser + rule
extractors + LLM normalization with hard boundary + completeness guard.
Not the architecture. Each numbered step above is a separate sprint
slice with its own scope, sign-off, and tests.

The biggest calls in this doc:

1. **Demote LLM from primary extractor to bounded normalizer.** Same
   architectural move that fixed matching. Answer: yes — live test 3
   demonstrated the failure mode (work=0 on a CV with clear work
   history). The risk of NOT doing this is that every downstream v2
   contract (match strength, semantic stage, NOC normalization) sits
   on top of randomly-empty inputs.

2. **Use the synthetic CV pack as the fixture suite.** Answer: yes —
   the test pack already documents expected output per CV; v2 ships
   when all 5 pass.

3. **Don't OCR scanned PDFs in this sprint.** Answer: defer. Existing
   `no_text` warning is sufficient; OCR is its own infrastructure
   commitment.
