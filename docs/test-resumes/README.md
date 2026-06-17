# Synthetic SCCC Resume Test Pack

These resumes are synthetic candidates for production-level SkillBridge testing.
They are not real people and should not be used for hiring decisions.

Use them to test the chat upload flow:

1. Open the SkillBridge chat UI.
2. Upload one PDF.
3. Confirm the parsed resume facts.
4. If the assistant asks, provide a target role matching the scenario below.
5. Verify the top recommendations, match band, missing skills, and explanation.

## Expected Behaviors

| File | Primary target | Expected behavior |
| --- | --- | --- |
| `cv_01_operations_client_services_strong.pdf` | Senior Manager of Operations & Client Services | Should find a strong or good operations/client-service management match if the Cygnus posting is active. It should cite leadership, QuickBooks/bookkeeping, Microsoft 365/SharePoint, workflow improvement, and client service. |
| `cv_02_truck_coach_credential_gap.pdf` | Truck and Coach Technician | Should surface the truck/coach technician posting as a stretch match. It should recognize welding, diesel repair, truck maintenance, emergency repair, and diagnostic tools, but name missing Class G licence and/or 310T certification if the posting requires them. |
| `cv_03_electrical_journeyman_strong.pdf` | Electrical Journeyman | Should produce a strong or good match for the electrical journeyman posting if active. It should cite 309A Certificate of Qualification, valid driver's licence, industrial electrical troubleshooting, schematics, switchgear, motors, and preventive maintenance. |
| `cv_04_front_desk_customer_service_good.pdf` | Front Desk Agent / customer service | Should produce a good or strong match for front desk/customer-service hotel postings. It should cite guest check-in, reservations, payment processing, property-management systems, phone communication, and customer service. |
| `cv_05_software_developer_negative_control.pdf` | Software developer / data analyst | Negative control. It should not force unrelated trades, hotel, or social-service jobs into good/strong bands. It may return no current SSM software/data roles, or only honest stretch/related matches if the live dataset contains something adjacent. |

## What To Watch

- The assistant should never say the candidate has a missing credential.
- If a semantic match fires, it should say "related to" or "overlaps with", not "you have".
- Apply URLs should be real SCCC URLs from the backend, not invented.
- Missing licence/certification gaps should be named plainly.
- The top-level chat should stay readable: only the most important matched/missing skills, not a long checklist.

## Source Anchors

The pack is based on SCCC postings and local SkillBridge fixtures seen during development:

- Senior Manager of Operations & Client Services, Cygnus Inc.
- Truck and Coach Technician, Garden River First Nation.
- Electrical Journeyman, Viacore.
- Front Desk Agent / customer-service hotel scenario.
- Software/data profile as a negative control.

Because SCCC postings can expire, a resume may move from "strong expected" to "no current match" when the source data changes. That is not automatically a bug; check whether the target posting is still in `core.v_current_job`.
