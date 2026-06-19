"""One-off ETL: transform official OaSIS 2025 source CSVs into the two
CSVs that skillbridge.ingest.reference's loaders expect.

Inputs (manual download from open.canada.ca dataset
10ce43bd-fb58-4969-806b-4bffebc87bec):
  data/guide_oasis_2025_v4.0.csv      -- taxonomy guide (Code, Structure
                                          Type, Name, Description)
  data/skills_oasis_2025_v1.1.csv     -- 900 occupation profiles x 33
                                          skill columns, semicolon-delimited

Outputs:
  data/oasis_skills.csv          -- 33 skill descriptors for
                                    reference.skill (skill_id, skill_name,
                                    aliases, category, description)
  data/noc_skill_mapping.csv     -- ~24,797 NOC-skill mappings for
                                    reference.noc_skill (noc_code,
                                    skill_id, importance, level)

After running this script:
  python run_pipeline.py --reference

Expected DB state:
  reference.skill          ~ 33 rows
  reference.noc_skill      ~ 24,797 rows
"""
from __future__ import annotations

import csv
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GUIDE_PATH = DATA_DIR / "guide_oasis_2025_v4.0.csv"
SKILLS_PATH = DATA_DIR / "skills_oasis_2025_v1.1.csv"
OUT_SKILLS = DATA_DIR / "oasis_skills.csv"
OUT_NOC_SKILL = DATA_DIR / "noc_skill_mapping.csv"


def load_guide() -> dict[str, tuple[str, str]]:
    """Read the OaSIS taxonomy guide and return {Name -> (Code,
    Description)} for every Descriptor row. Categories,
    Sub-Categories, and Similarity Groups are NOT skill descriptors
    and are skipped."""
    guide: dict[str, tuple[str, str]] = {}
    with open(GUIDE_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)  # header
        for row in reader:
            if len(row) < 4:
                continue
            code, structure_type, name, description = (
                row[0].strip(), row[1].strip(),
                row[2].strip(), row[3].strip(),
            )
            if structure_type != "Descriptor":
                continue
            if not code or not name:
                continue
            guide[name] = (code, description)
    return guide


def write_oasis_skills(guide: dict[str, tuple[str, str]],
                      skill_column_names: list[str]) -> int:
    """Write data/oasis_skills.csv with one row per skill column the
    skills file actually uses. Filters the guide's 244 descriptors
    down to the 33 the OaSIS skill matrix references."""
    n = 0
    with open(OUT_SKILLS, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["skill_id", "skill_name", "aliases", "category", "description"]
        )
        for name in skill_column_names:
            if name not in guide:
                print(f"  WARN  skill column not in guide: {name!r}")
                continue
            code, description = guide[name]
            writer.writerow([code, name, "", "Skills", description])
            n += 1
    return n


def write_noc_skill_mapping(skill_column_names: list[str],
                            guide: dict[str, tuple[str, str]]) -> int:
    """Walk every row of the OaSIS skills matrix, emit one NOC-skill
    row per non-zero cell.

    NOC code is the 'Code OaSIS' column with the .00 sub-profile
    suffix stripped (14200.00 -> 14200). This matches the schema's
    VARCHAR(5) PRIMARY KEY on reference.occupation.noc_code.

    Importance is the float value in the cell (0.0-5.0 per OaSIS).
    Cells with value <= 0 are skipped -- those represent
    not-applicable, not unrated. Level defaults to 0.0 because the
    OaSIS 2025 v1.1 matrix provides a single rating per cell, not
    separate importance + level (the loader's `level` column is
    nullable in shape but the loader writes whatever we pass; 0.0
    is the honest placeholder).
    """
    skill_codes_by_col = {
        col: guide[col][0] for col in skill_column_names if col in guide
    }
    n_written = 0
    n_zero_skipped = 0
    n_parse_skipped = 0
    with open(OUT_NOC_SKILL, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["noc_code", "skill_id", "importance", "level"])
        with open(SKILLS_PATH, "r", encoding="utf-8-sig", newline="") as in_f:
            reader = csv.reader(in_f, delimiter=";")
            headers = next(reader)
            skill_cols_in_file = headers[2:]
            for row in reader:
                if len(row) < 3:
                    continue
                noc_raw = row[0].strip()
                if "." in noc_raw:
                    noc_code = noc_raw.split(".", 1)[0]
                else:
                    noc_code = noc_raw
                if not noc_code or len(noc_code) != 5 or not noc_code.isdigit():
                    continue
                for col_index, col_name in enumerate(skill_cols_in_file, start=2):
                    if col_name not in skill_codes_by_col:
                        continue
                    if col_index >= len(row):
                        continue
                    raw = row[col_index].strip()
                    if not raw:
                        n_zero_skipped += 1
                        continue
                    try:
                        importance = float(raw)
                    except ValueError:
                        n_parse_skipped += 1
                        continue
                    if importance <= 0.0:
                        n_zero_skipped += 1
                        continue
                    writer.writerow([
                        noc_code,
                        skill_codes_by_col[col_name],
                        f"{importance:.1f}",
                        "0.0",
                    ])
                    n_written += 1
    print(f"  rows_written:    {n_written}")
    print(f"  cells_skipped_zero: {n_zero_skipped}")
    print(f"  cells_skipped_parse_error: {n_parse_skipped}")
    return n_written


def main() -> None:
    print("== OaSIS ETL: source CSVs -> loader CSVs ==")
    print(f"Guide:  {GUIDE_PATH}")
    print(f"Skills: {SKILLS_PATH}")
    print()

    print("Loading taxonomy guide...")
    guide = load_guide()
    print(f"  Descriptor entries: {len(guide)}")
    print()

    with open(SKILLS_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        headers = next(reader)
    skill_column_names = list(headers[2:])
    print(f"Skill columns in OaSIS matrix: {len(skill_column_names)}")
    print()

    print(f"Writing {OUT_SKILLS.name}...")
    n_skills = write_oasis_skills(guide, skill_column_names)
    print(f"  rows: {n_skills}")
    print()

    print(f"Writing {OUT_NOC_SKILL.name}...")
    n_mappings = write_noc_skill_mapping(skill_column_names, guide)
    print()

    print("== Done ==")
    print(f"  {OUT_SKILLS.name}:      {n_skills} rows")
    print(f"  {OUT_NOC_SKILL.name}: {n_mappings} rows")
    print()
    print("Next: python run_pipeline.py --reference")


if __name__ == "__main__":
    main()
