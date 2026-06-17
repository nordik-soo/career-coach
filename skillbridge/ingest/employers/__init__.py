"""SSM employer career-page connectors.

Each connector fetches one local employer's public career page, parses
postings, and upserts into core.job_posting under that employer's
approved source name.

Two reference parsers (sault_area_hospital, city_of_ssm_hr) have working
selectors as a starting point — they will likely need tuning against the
live page.

Nine stub parsers are scaffolded with the right URL env vars and
ConnectorResult plumbing, but the actual HTML parsing must be filled in
against the real page once selectors are confirmed.
"""
from skillbridge.ingest.employers._base import EmployerConnector
from skillbridge.ingest.employers.adsab import ADSABConnector
from skillbridge.ingest.employers.algoma_steel import AlgomaSteelConnector
from skillbridge.ingest.employers.algoma_u_careers import AlgomaUCareersConnector
from skillbridge.ingest.employers.cas_algoma import CASAlgomaConnector
from skillbridge.ingest.employers.city_of_ssm import CityOfSSMHRConnector
from skillbridge.ingest.employers.group_health_centre import GroupHealthCentreConnector
from skillbridge.ingest.employers.puc import PUCConnector
from skillbridge.ingest.employers.sault_area_hospital import SaultAreaHospitalConnector
from skillbridge.ingest.employers.sault_college_careers import SaultCollegeCareersConnector
from skillbridge.ingest.employers.school_board import SchoolBoardConnector
from skillbridge.ingest.employers.ymca_ssm import YMCASSMConnector

ALL_EMPLOYER_CONNECTORS = [
    # Reference parsers (selectors are starting points — verify before prod)
    SaultAreaHospitalConnector,
    CityOfSSMHRConnector,
    # Stubs awaiting real selectors against live pages
    AlgomaSteelConnector,
    SaultCollegeCareersConnector,
    AlgomaUCareersConnector,
    PUCConnector,
    GroupHealthCentreConnector,
    YMCASSMConnector,
    CASAlgomaConnector,
    ADSABConnector,
    SchoolBoardConnector,
]

__all__ = ["EmployerConnector", "ALL_EMPLOYER_CONNECTORS"]
