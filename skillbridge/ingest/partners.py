"""Partner job-source connectors.

SSM-only product (BREAKING.md). No region filter at the framework level:
sources are local by construction. SCCC is implemented first using its
public WordPress REST API; other partner sources stay feed-only stubs
until their public endpoint or agreed feed is known.

SCCC integration history:
  v0 — JS-rendered HTML scraping (rejected, partner ToS risk)
  v1 — WP Job Manager AJAX `/jm-ajax/get_listings/` + HTML fragment parse
  v2 — WordPress REST API `/wp-json/wp/v2/job-listings` (this file)

The v2 path is structured JSON, paginated via standard headers, and
needs no per-job follow-up fetches.
"""
from __future__ import annotations

import csv
import html
import io
import logging
import re
from typing import Iterable

import httpx
from bs4 import BeautifulSoup

from config import JOB_SOURCES
from skillbridge.ingest.base import (
    JobConnector,
    NormalizedJob,
    parse_date_loose,
    parse_float_loose,
)

log = logging.getLogger(__name__)


def _config(name: str):
    for s in JOB_SOURCES:
        if s.name == name:
            return s
    return None


# ---------------------------------------------------- 1. SCCC job board
class SCCCConnector(JobConnector):
    """Sault Community Career Centre job board.

    Reads from SCCC's public WordPress REST API endpoint for job listings.
    Pagination via `?per_page=100&page=N` and `X-WP-TotalPages` header.

    Partner posture: the endpoint is public by WordPress default. We
    identify ourselves explicitly via User-Agent, poll daily, and will
    stop or adjust on partner request. See BREAKING.md.
    """

    source_name = "sccc"
    default_url = "https://saultcareercentre.ca/wp-json/wp/v2/job-listings"

    def fetch(self) -> Iterable[NormalizedJob]:
        cfg = _config("sccc")
        if cfg is None or not cfg.enabled:
            log.info("SCCC disabled - skipping")
            return
        url = cfg.url if cfg.url and not cfg.url.startswith("PLACEHOLDER") else self.default_url
        yield from _fetch_sccc_wp_rest(url)


# ----------------------------------------- 2. Welcome to SSM Careers (stub)
class WelcomeSSMConnector(JobConnector):
    """Welcome to SSM Careers.

    Activates when SSM EDC provides a CSV/JSON feed URL.
    """

    source_name = "welcome_ssm"

    def fetch(self) -> Iterable[NormalizedJob]:
        cfg = _config("welcome_ssm")
        if cfg is None or not cfg.enabled:
            log.info("Welcome to SSM disabled - skipping")
            return
        if not cfg.url or cfg.url.startswith("PLACEHOLDER"):
            log.warning("WELCOME_SSM_FEED_URL is placeholder - set it in .env")
            return
        yield from _fetch_csv_or_json(cfg.url, "", source="welcome_ssm")


# --------------------------------------- 3. City of Sault Ste. Marie (stub)
class CitySSMConnector(JobConnector):
    """City of Sault Ste. Marie careers feed placeholder.

    A future sanctioned feed can live here. The public HR career page is
    handled by the employer-connector pattern.
    """

    source_name = "city_ssm"

    def fetch(self) -> Iterable[NormalizedJob]:
        cfg = _config("city_ssm")
        if cfg is None or not cfg.enabled:
            log.info("City of SSM (feed) disabled - skipping")
            return
        if not cfg.url or cfg.url.startswith("PLACEHOLDER"):
            log.warning(
                "CITY_SSM_CAREERS_URL is placeholder - set it for a sanctioned feed; "
                "otherwise the employer-connector page parser handles this source."
            )
            return
        log.info("City of SSM connector is a stub - add feed parser when feed exists")
        return
        yield  # pragma: no cover


# ============================================================================
# SCCC WP REST API client
# ============================================================================
SCCC_USER_AGENT = "SkillBridge-SSM/0.1 (research; nordik.org)"
SCCC_PER_PAGE = 100
SCCC_REQUEST_TIMEOUT = 30          # PR 9A baseline; retries land in 9B
SCCC_MAX_PAGES = 50                # hard ceiling against runaway pagination


def _fetch_sccc_wp_rest(url: str) -> Iterable[NormalizedJob]:
    """Yield NormalizedJob items from the SCCC WP REST endpoint.

    Walks pages until either the response is empty, the X-WP-TotalPages
    header says we're done, or WordPress returns HTTP 400 (its signal for
    "page beyond range"). On any other failure, logs and stops — no
    exceptions cross the boundary so one bad source can't poison the run.
    """
    headers = {
        "User-Agent": SCCC_USER_AGENT,
        "Accept": "application/json",
    }
    yielded = 0
    with httpx.Client(timeout=SCCC_REQUEST_TIMEOUT, headers=headers, follow_redirects=True) as client:
        for page in range(1, SCCC_MAX_PAGES + 1):
            try:
                resp = client.get(url, params={"per_page": SCCC_PER_PAGE, "page": page})
            except httpx.RequestError as e:
                log.error("SCCC request failed on page %d: %s", page, e)
                return

            if resp.status_code == 400:
                # WordPress's "no more pages" signal.
                break

            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                log.error("SCCC HTTP %d on page %d: %s", resp.status_code, page, e)
                return

            try:
                payload = resp.json()
            except ValueError:
                log.error("SCCC page %d returned non-JSON", page)
                return

            if not isinstance(payload, list) or len(payload) == 0:
                break

            for item in payload:
                job = _normalize_sccc_wp_rest_item(item)
                if job:
                    yielded += 1
                    yield job

            total_pages_raw = resp.headers.get("X-WP-TotalPages", "1")
            try:
                total_pages = int(total_pages_raw)
            except ValueError:
                total_pages = 1
            if page >= total_pages:
                break

    log.info("SCCC WP REST: yielded %d normalized job(s)", yielded)


def _normalize_sccc_wp_rest_item(item: dict) -> NormalizedJob | None:
    """Map one WP REST job_listing object to NormalizedJob.

    Required fields: id, title.rendered. Anything else is best-effort.
    Jobs whose location can't be confirmed as SSM are dropped — SCCC's
    board occasionally lists regional postings.
    """
    wp_id = item.get("id")
    if not wp_id:
        return None

    title = _clean_wp_html(_get(item, "title", "rendered"))
    if not title:
        return None

    meta = item.get("meta") or {}
    employer = _str_or_none(meta.get("_company_name"))
    location_raw = _str_or_none(meta.get("_job_location")) or ""
    link = _str_or_none(item.get("link")) or ""

    if not _is_sccc_ssm_location(location_raw, link):
        return None

    # Strip "(Remote)" suffix from the location display while preserving
    # the remote flag.
    location = re.sub(r"\s*\(Remote\)\s*", "", location_raw, flags=re.I).strip() or None
    remote_flag = (
        bool(meta.get("_remote_position"))
        or "remote" in location_raw.lower()
    )

    description = _clean_wp_html(_get(item, "content", "rendered"))
    salary_text = _str_or_none(meta.get("_job_salary"))

    posted_date = parse_date_loose(_iso_date_part(item.get("date")))

    # Closing date: WP Job Manager stores _job_expires (sometimes empty).
    closing_date = parse_date_loose(meta.get("_job_expires"))

    return NormalizedJob(
        source="sccc",
        source_job_id=str(wp_id),
        title=title,
        employer=employer,
        location=location,
        region_code=None,
        description=description or None,
        url=link or None,
        posted_date=posted_date,
        closing_date=closing_date,
        salary_text=salary_text,
        salary_low=parse_float_loose(meta.get("_job_salary_low")),
        salary_high=parse_float_loose(meta.get("_job_salary_high")),
        employment_type=None,  # job-types taxonomy; deferred to 9B
        remote_flag=remote_flag,
        noc_code=None,
        raw_payload={
            "source": "sccc_wp_rest_v2",
            "wp_id": wp_id,
            "slug": item.get("slug"),
            "date": item.get("date"),
            "modified": item.get("modified"),
            "link": link,
            "application": _str_or_none(meta.get("_application")),
            "location_raw": location_raw,
        },
    )


# ============================================================================
# Helpers
# ============================================================================
def _get(obj: dict, *path: str) -> str:
    """Safely walk a nested dict, returning '' on any miss."""
    cur: object = obj
    for key in path:
        if not isinstance(cur, dict):
            return ""
        cur = cur.get(key)
        if cur is None:
            return ""
    return cur if isinstance(cur, str) else ""


def _str_or_none(value) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _iso_date_part(value: str | None) -> str | None:
    """WP returns 2026-05-22T14:30:00 — take the date prefix only."""
    if not value:
        return None
    return value.split("T", 1)[0]


_SHORTCODE_RE = re.compile(r"\[/?[a-zA-Z0-9_\-]+(?:\s[^\]]*)?\]")


def _clean_wp_html(raw: str | None) -> str:
    """Strip HTML + Elementor shortcodes from a WordPress rendered field."""
    if not raw:
        return ""
    # Drop shortcodes first — `[gallery]`, `[/elementor-template]`, etc.
    no_codes = _SHORTCODE_RE.sub("", raw)
    soup = BeautifulSoup(no_codes, "lxml")
    text = soup.get_text("\n", strip=True)
    text = html.unescape(text)
    # Collapse multiple blank lines.
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _is_sccc_ssm_location(location: str | None, url: str) -> bool:
    """Keep SCCC ingestion scoped to the Sault Ste. Marie + Algoma area.

    SCCC's mandate covers the full Algoma District, so a posting in
    Wawa or Chapleau is legitimately local to a newcomer using
    SkillBridge SSM. Communities are configured via LOCAL_CITIES in
    .env so the product scope can be tuned without code changes.

    Empty locations are excluded rather than assumed local — the API
    exposes _job_location explicitly, so missing data is a real
    signal, not just a parsing miss.

    Note: the URL is included in the haystack so SCCC slugs containing
    a city name ("sault-ste-marie", "chapleau", etc.) keep matching
    even when the metadata field is unusual.
    """
    from config import LOCAL_CITIES

    if not (location or "").strip():
        return False
    haystack = f"{location or ''} {url}".lower()
    return any(city in haystack for city in LOCAL_CITIES)


# ============================================================================
# Internal CSV/JSON helpers (used by other partner stubs)
# ============================================================================
def _fetch_csv_or_json(url: str, api_key: str, *, source: str) -> Iterable[NormalizedJob]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        with httpx.Client(timeout=60, headers=headers, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "").lower()
            if "json" in ct:
                payload = resp.json()
                rows = payload if isinstance(payload, list) else (
                    payload.get("data") or payload.get("jobs") or []
                )
                for row in rows:
                    job = _row_to_job(row, source=source)
                    if job:
                        yield job
            else:
                reader = csv.DictReader(io.StringIO(resp.text))
                for row in reader:
                    job = _row_to_job(row, source=source)
                    if job:
                        yield job
    except Exception as e:
        log.error("%s fetch failed: %s", source, e)


def _row_to_job(row: dict, *, source: str) -> NormalizedJob | None:
    title = (row.get("title") or row.get("Job Title") or "").strip()
    source_job_id = (
        row.get("source_job_id") or row.get("id") or row.get("job_id")
        or row.get("Job ID") or row.get("posting_id") or ""
    )
    source_job_id = str(source_job_id).strip()
    if not title or not source_job_id:
        return None
    return NormalizedJob(
        source=source,
        source_job_id=source_job_id,
        title=title,
        employer=(row.get("employer") or row.get("Employer") or "").strip() or None,
        location=(row.get("location") or row.get("Location") or "").strip() or None,
        region_code=row.get("region_code") or None,
        description=(row.get("description") or row.get("Job Description") or "").strip() or None,
        url=row.get("url") or row.get("URL") or None,
        posted_date=parse_date_loose(row.get("posted_date") or row.get("Date Posted")),
        closing_date=parse_date_loose(row.get("closing_date") or row.get("Closing Date")),
        salary_text=(row.get("salary_text") or row.get("Salary") or row.get("Wage") or "").strip() or None,
        salary_low=parse_float_loose(row.get("salary_low") or row.get("Min Wage")),
        salary_high=parse_float_loose(row.get("salary_high") or row.get("Max Wage")),
        employment_type=(row.get("employment_type") or row.get("Employment Type") or "").strip() or None,
        remote_flag=None,
        noc_code=(row.get("noc_code") or row.get("NOC Code") or "").strip() or None,
        raw_payload=dict(row),
    )


ALL_CONNECTORS: list[type[JobConnector]] = [
    SCCCConnector,
    WelcomeSSMConnector,
    CitySSMConnector,
]
