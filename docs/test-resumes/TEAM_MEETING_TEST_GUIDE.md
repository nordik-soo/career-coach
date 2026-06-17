# Team Meeting Match Demo

These synthetic resumes test three visibly different matching outcomes.
They are not real people and must not be used for hiring decisions.

## Live SCCC Anchor

- Job: 310S Licensed Automotive Technician
- Employer: Great Lakes Honda
- Location: Sault Ste. Marie
- Posted: June 3, 2026
- Closing date shown in posting: June 12, 2026
- URL: https://saultcareercentre.ca/job/310s-licensed-automotive-technician/
- Verified active: June 8, 2026

Do not use this posting after June 12 without checking it again. The current
database snapshot was stale during preparation, so run the SCCC ingestion
pipeline before the meeting and confirm the job appears as active.

## Demo 1: Strong Match

File: `meeting_01_310s_automotive_strong.pdf`

After resume review, say:

> I want a 310S Licensed Automotive Technician role.

Expected:

- Great Lakes Honda job appears.
- Strong or good band, subject to the engine's current scoring thresholds.
- Matched evidence includes the 310S licence, Class G licence, preventive
  maintenance, brake service, safety inspections, diagnostics, repairs,
  problem solving, customer service, and fast-paced shop work.
- The system must not claim either required licence is missing.

## Demo 2: Weak / Stretch Match

File: `meeting_02_310s_automotive_weak.pdf`

After resume review, say:

> I want a 310S Licensed Automotive Technician role.

Expected:

- The same Great Lakes Honda job appears as stretch or major-gap guidance.
- The system plainly identifies the missing 310S qualification.
- It may recognize preventive-maintenance support, basic brake work, hand
  tools, safety practices, and the G2 licence.
- It must not describe the candidate as already licensed.

## Demo 3: No Current Match

File: `meeting_03_airline_pilot_no_match.pdf`

After resume review, say:

> I want a Commercial Airline Pilot role.

Expected:

- No unrelated automotive, office, trades, or social-service job should be
  promoted as a good or strong match.
- The response should honestly report no current local match or ask whether
  the candidate wants to explore a related aviation role.
- It must not reinterpret aviation skills as automotive qualifications.

## Pre-Meeting Checklist

1. Run the SCCC ingestion pipeline.
2. Confirm the Great Lakes Honda page does not say the listing has expired.
3. Confirm the DB row has `is_active=true`, the exact posting URL, and a
   recent `last_seen_at`.
4. Restart the API so the embedding model can warm before the presentation.
5. Use a fresh browser session for each resume.
6. Capture the API log line containing `final_move`, `results`, and `band`.
