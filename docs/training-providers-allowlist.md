# Training Provider Allowlist

Status: policy doc · 2026-06-04 · prerequisite for the training registry

This document defines which training providers SkillBridge SSM is willing to
link to. It exists so future contributors don't have to re-derive the policy
when adding new entries to the training registry, and so reviewers can
quickly assess whether a proposed PR meets the trust bar.

## Why an allowlist instead of "anything goes"

The chat's grounding rule is: **every URL in the assistant's reply must come
from RESULTS or TRAINING**. The recommender feeds the TRAINING block from the
curated registry. If a provider's URL is in the registry, the chat may cite
it; if it's not, the chat can only refer to the provider generically (e.g.
"Sault Community Career Centre can help"). This makes the registry an
allowlist by construction — adding a provider means adopting a recommendation
we vouch for.

Bad URLs cost the user real time and trust. The allowlist exists to keep that
bar high.

## Trust criteria

A provider qualifies for the registry when **all four** hold:

1. **Reputation**: the organization is the authoritative source for the
   credential, the local college/university, an official vendor (Microsoft,
   AWS, etc.), or a recognized national MOOC (Coursera, edX, Google Career
   Certificates).
2. **URL stability**: the linked page is at a structured, durable URL — not a
   marketing landing page that's expected to disappear in a quarterly site
   refresh.
3. **Breadth**: the resource teaches a broadly applicable skill or credential,
   not a niche one-off course.
4. **Reviewable**: the URL was last verified within the past 6 months. After
   6 months, the responder falls back to a generic referral until the entry
   is re-verified.

If any one of these is uncertain, **default to NO**. False-positive registry
entries cost the user real time when they click a broken link or a misleading
course. The cost of being too strict is just "the registry stays small a
bit longer."

## Allowed providers (the YES list)

### Local — SSM core

| Provider | Why | Typical resource types |
|---|---|---|
| **Sault College** | The local public college; primary trades + applied programs in SSM | `apprenticeship`, `local_training`, `online_course` (continuing ed) |
| **Algoma University** | The local university; degrees, certificates, and continuing ed (business/IT/data) | `local_training` |
| **Sault Community Career Centre** | Local employment + training advisory (sometimes referred to as SCCC). THE referral target when no specific URL is appropriate | `referral_only` (no direct URLs) |
| **Northland Adult Learning Centre** | Newcomer-focused ESL + essential skills (computer literacy, math, basic communication) | `local_training` |
| **OntarioColleges.ca** | Official Ontario public-college program search; useful when a current local program URL is unavailable or changes by intake | `local_training` |

### Ontario credential authorities

| Provider | Why | Typical resource types |
|---|---|---|
| **Skilled Trades Ontario** | The provincial regulator for compulsory and voluntary trades (310T, electrician, plumber, etc.); authoritative on credential pathways | `credential_pathway` |
| **DriveTest** | Official Ontario driver testing (Class G, A, D, Z, M) | `credential_pathway` |
| **Ontario.ca** | Government of Ontario information portal (driver licensing, business licensing, regulated professions). Use `Ontario.ca` for information pages; transactional `ServiceOntario` URLs are a separate entry when needed | `credential_pathway` |
| **ServiceOntario** | Transactional services (renewals, fee payments, document requests). Use only when the URL is a service-transaction page, not informational | `credential_pathway` |
| **Ministry of Labour, Immigration, Training and Skills Development** | Apprenticeship registration and trade compliance | `credential_pathway` |
| **Smart Serve Ontario** | Ontario's recognized responsible alcohol-service certification provider for hospitality and event roles | `credential_pathway` |
| **Sault Ste. Marie Police Service** | Local police-service source for police record checks / vulnerable sector checks in the SSM catchment | `credential_pathway` |
| **College of Early Childhood Educators** | Ontario regulator for Registered Early Childhood Educators (RECE) | `credential_pathway` |
| **College of Nurses of Ontario** | Ontario regulator for RN/RPN/NP registration and nursing practice requirements | `credential_pathway` |

### National MOOCs / vendor certifications

| Provider | Why | Typical resource types |
|---|---|---|
| **Microsoft Learn** | Free, official Microsoft cert paths (MS Office, Azure, Power BI, etc.) | `online_course` |
| **AWS Skill Builder** | Free, official AWS cert paths (Cloud Practitioner, Developer Associate) | `online_course` |
| **Google Career Certificates** (via Coursera) | Broad entry-level certs (IT Support, Data Analytics, UX, Project Management) | `online_course` |
| **Coursera** | When the URL is an official course/specialization page from a recognized institution (Stanford, Google, Meta, U Toronto, etc.). NOT random user-submitted content | `online_course` |
| **edX** | Same standard as Coursera | `online_course` |
| **CompTIA** | Vendor-neutral IT certs (A+, Network+, Security+) | `credential_pathway`, `online_course` |
| **Intuit (QuickBooks)** | Vendor-official accounting training | `online_course` |
| **National Payroll Institute** | Canadian payroll education/certification body for payroll compliance and administration pathways | `credential_pathway` |

### Health/safety credential providers

| Provider | Why | Typical resource types |
|---|---|---|
| **Canadian Red Cross** | First Aid, CPR, mental health certifications | `online_course`, `local_training` |
| **St. John Ambulance** | First Aid, CPR | `online_course`, `local_training` |
| **CCOHS** | Canadian Centre for Occupational Health and Safety. WHMIS, workplace safety | `online_course` |
| **TrainCan** | Food Handler / Food Safe | `online_course` |
| **Algoma Public Health** | Local Algoma-region public health unit. Authoritative for food-handler certification and other regulated public-health training in this catchment | `local_training` |

## Not allowed (the NO list)

These are the failure modes we explicitly reject — examples are illustrative,
not exhaustive:

| Source type | Why no | Concrete example of what NOT to link |
|---|---|---|
| Random blogs | Unstable, no editorial oversight | A WordPress blog post "Top 10 Ways to Get Your 310T" |
| SEO course-review aggregators | Affiliate marketing, incentive to mislead | "Best forklift certification 2026 — read our review!" |
| Unofficial YouTube playlists or channels | Quality varies; "lessons" disappear without notice | "Truck mechanic full course — 47 videos" on a personal channel |
| Personal sites of instructors / coaches | Stability + accountability gap | `myforkliftcoach.com/get-certified/` |
| Pirated course material | Obvious | n/a |
| LLM-generated guides on Medium / dev.to | Often inaccurate, no editorial review | "How I got my 310T in 90 days" personal essay |
| For-profit "lead-gen" sites that exist to capture phone numbers | The user gets a sales call instead of training | n/a |
| News articles about training | Even from reputable outlets, articles age and aren't navigable as resources | Globe and Mail piece on trades shortages |
| Outdated Wikipedia stubs | Better as a reference than as an action link | Wikipedia article on WHMIS |

## Special case: provincial / national government sites

These pass the trust criteria but warrant a note:

- **ontario.ca** pages are stable in their slug structure but redirect when
  programs are renamed. Verify the URL still resolves at the start of each
  6-month re-verification window.
- **canada.ca** pages: for SSM-only product scope, prefer Ontario-level
  information. Federal sites may be cited for things only the federal
  government handles (e.g. WES credential recognition referral, which is
  itself in `referral_only` territory).

## Special case: SCCC and "referral_only"

Sault Community Career Centre is in the registry but typically as
`type: referral_only` — i.e. **no specific URL** — when the corresponding
training is either:

- Not deliverable online (e.g. on-the-job apprenticeship guidance)
- Region-specific in ways that need a human counsellor to map
- Or simply unknown to us

This is the safety valve. When the recommender finds a gap but has no
authoritative URL in the registry, it points the user to SCCC counsellor
intake instead of inventing a URL.

## Re-verification convention

Each registry entry carries `verified_at` and `verified_by` fields. The
recommender applies this rule at runtime:

| `verified_at` age | Behavior |
|---|---|
| ≤ 6 months | Surface the URL normally |
| > 6 months | Surface the provider name + generic guidance, but **suppress the URL** ("contact SCCC or check the provider's site directly") |
| Missing / null | Treat as "pending verification" — same as expired |

This means a stale registry degrades gracefully rather than serving link rot.

## Adding a new provider

When a contributor proposes a new provider for the registry:

1. Verify it meets all four trust criteria above
2. Confirm the URL is at a stable structured path (not a homepage with no
   sub-pages, not a marketing landing page)
3. Add `verified_by` (the contributor's name or PR URL) and `verified_at`
   (today's date)
4. Note in the PR description which gap(s) the provider serves and what
   resource type(s) apply
5. Get review sign-off from the lead engineer before merge

## Out of scope for v1

These would expand the allowlist meaningfully but aren't priority right now:

- Industry-specific union training (e.g. Canadian Welding Bureau training)
- Provincial sector councils
- Employer-specific corporate training (e.g. Tesla, Uber)
- Bootcamps with selective admissions (e.g. Lighthouse Labs, BrainStation)

These can be added later through the same PR review pathway. Defer until the
v1 registry has shipped and we have telemetry on real gap requests.
