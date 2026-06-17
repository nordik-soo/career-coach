# OaSIS + SCT reference-data download

Matching v2 step 1 (see [matching-v2-design.md](matching-v2-design.md))
imports the **occupation/title lexicon** from two public Government of
Canada datasets:

- **OaSIS** (Occupational and Skills Information System) — published by ESDC
- **SCT** (Skills and Competencies Taxonomy) — companion dataset

Both are on Canada's [Open Government Portal](https://open.canada.ca/) under
the **Open Government Licence — Canada** (research / academic / commercial
use allowed; attribution required).

The loaders are non-fatal: missing CSVs log a warning and the rest of the
pipeline continues unaffected.

---

## What we use, what we don't

We import the **occupation-title lexicon only** in Step 1. See §2 of the
matching-v2 design doc for why — OaSIS's skill descriptors are O*NET-style
broad competencies and don't bridge the granular tech / trades vocabulary
that drives SkillBridge's day-to-day matching.

| File | What we use it for |
|---|---|
| OaSIS example-titles (EN) | Primary title-synonym list per NOC (English) |
| OaSIS example-titles (FR) | Same, French |
| OaSIS lead-statement (EN) | Augments `reference.occupation.lead_statement_en` |
| OaSIS lead-statement (FR) | Augments `reference.occupation.lead_statement_fr` |
| SCT alternative-titles | Extra synonym layer (bilingual EN/FR in one file) |

We deliberately do **not** import: OaSIS skills layer, abilities,
personal attributes, knowledge, work activities, work context, wage data,
outlook data, or regulated-occupation cross-references. Those are either
out-of-scope (wage/outlook) or already covered by other tables
(`core.regulated_occupation`) or deferred (skill descriptors — see Step 3
of the design doc's pickup order).

---

## Manual download (one-time per OaSIS release)

OaSIS publishes versioned releases roughly yearly. The 2025 v1.0 release
is current. Download the five files below into `./data/` (relative to the
`skillbridge-api/` directory):

```
data/
├── example-titles_oasis_2025_v1.0.csv                                        (EN, OaSIS)
├── exemples-dappellation-demploi_sipec_2025_v1.0.csv                         (FR, OaSIS)
├── lead-statement_oasis_2025_v1.0.csv                                        (EN, OaSIS)
├── enonce-principal_sipec_2025_v1.0.csv                                      (FR, OaSIS)
└── alternatives-titles-skills-and-competencies-taxonomy-2023-version-1.0-en-fr.csv   (bilingual, SCT)
```

**Direct download URLs:**

- OaSIS 2025 dataset page:
  https://open.canada.ca/data/en/dataset/10ce43bd-fb58-4969-806b-4bffebc87bec
  - Example titles EN: https://open.canada.ca/data/dataset/10ce43bd-fb58-4969-806b-4bffebc87bec/resource/3484364c-75a4-4e22-ae9a-d12ea72569d4/download/example-titles_oasis_2025_v1.0.csv
  - Example titles FR: https://open.canada.ca/data/dataset/10ce43bd-fb58-4969-806b-4bffebc87bec/resource/c66e5c70-6b73-4de2-88eb-b1ec4e887452/download/exemples-dappellation-demploi_sipec_2025_v1.0.csv
  - Lead statement EN: https://open.canada.ca/data/dataset/10ce43bd-fb58-4969-806b-4bffebc87bec/resource/d2a00c9b-2b7d-458f-a41f-10011d8164c7/download/lead-statement_oasis_2025_v1.0.csv
  - Lead statement FR: https://open.canada.ca/data/dataset/10ce43bd-fb58-4969-806b-4bffebc87bec/resource/6909fd40-e543-486f-a45d-b39061153de0/download/enonce-principal_sipec_2025_v1.0.csv
- SCT 2025 dataset page:
  https://open.canada.ca/data/en/dataset/618d2756-8c37-4f99-b184-8b3c1ef1b0f5
  - Alternative titles (bilingual): https://open.canada.ca/data/dataset/618d2756-8c37-4f99-b184-8b3c1ef1b0f5/resource/7e735c23-ff32-4550-8736-3a2c9aa959f5/download/alternatives-titles-skills-and-competencies-taxonomy-2023-version-1.0-en-fr.csv

If the filenames or URLs change in a future release, override via env vars
(see `config.py` — `OASIS_EXAMPLE_TITLES_EN_CSV`, etc.) so the loader
finds the new locations without code changes.

---

## Run the import

After dropping the CSVs into `./data/`:

```bash
python run_pipeline.py --reference
```

This runs every reference loader in `skillbridge/ingest/reference.py`,
including the two new ones (`load_oasis_occupation_titles`,
`load_oasis_lead_statements`). Idempotent — safe to re-run.

---

## Verify

```sql
-- How many occupations got OaSIS data?
SELECT COUNT(*) AS occupations_with_oasis
  FROM reference.occupation
 WHERE oasis_version IS NOT NULL;

-- Title-synonym distribution
SELECT lang, source, COUNT(*) AS n
  FROM reference.occupation_title_synonym
 GROUP BY lang, source
 ORDER BY lang, source;

-- Spot-check one NOC code
SELECT title, lang, source
  FROM reference.occupation_title_synonym
 WHERE noc_code = '21232'
 ORDER BY lang, source, title;
```

Expected after a successful import of OaSIS 2025 + SCT 2023:
- ~900 occupations with `oasis_version` populated
- ~3-5 example titles per NOC × 2 languages = ~6,000-9,000 synonyms from `oasis_example`
- ~1-3 alternative titles per NOC × 2 languages = additional ~2,000-5,000 from `sct_alternative`

---

## What changed in the schema

See the `MATCHING v2 STEP 1` block at the end of `sql/schema.sql`. New
columns on `reference.occupation`: `title_fr`, `lead_statement_en`,
`lead_statement_fr`, `oasis_version`. New table:
`reference.occupation_title_synonym`. All migrations idempotent.

---

## When the next OaSIS release ships

1. Download the new CSVs (filenames will include the version, e.g.
   `..._oasis_2026_v1.0.csv`)
2. Either replace the files in `./data/` (keeping the old paths via env
   vars) OR override the env vars in `.env` to point at the new files
3. Bump `OASIS_VERSION` in `.env` (e.g. `OASIS_VERSION=2026_v1.0`)
4. Re-run `python run_pipeline.py --reference`

The loader is idempotent — existing synonyms with the same
`(noc_code, title, lang, source)` key are left alone; new ones are added.
If a synonym was removed in the new release, the old row stays in the DB
(the loader doesn't currently prune). Document the dropped synonym in
release notes if it ever matters for matching.
