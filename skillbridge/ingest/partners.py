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
from skillbridge.match.region import normalize_declared_job_location
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


# ---------------------------------------------------- 2. AWIC jobs (aggregator)
class AWICJobsConnector(JobConnector):
    """AWIC local job aggregator (GeoJSON feed).

    Reads from AWIC's public WeDataTools REST endpoint that powers their
    own portal Jobs Map. Response is a single GeoJSON FeatureCollection;
    no pagination.

    Local-aggregator provenance rule (see BREAKING.md and the
    core.approved_job_source description for awic_jobs):

        AWIC is treated as a LOCAL AGGREGATOR. Some postings'
        properties.url values point to third-party sources including
        Job Bank. SkillBridge ingests AWIC's curated METADATA layer
        only, not the third-party sources behind the apply URLs. Every
        posting is stamped with source="awic_jobs" so the no-federal-
        source rule enforced by tests/test_source_purity.py is
        satisfied by AWIC's role as local curator, not by inspecting
        the third-party URL.

    SSM-only scope: AWIC serves the whole Algoma District. We filter
    to Sault Ste. Marie using a coordinate bounding box on each
    feature's geometry.coordinates. Postings without coordinates are
    dropped rather than assumed local.

    NOC 2021: AWIC often carries the code in properties.nocs_2021.
    When it does (exactly 5 digits, numeric), we pass it through
    verbatim and the downstream NOC backfill step is a no-op for
    that row. When the code is missing/invalid, noc_code is left None
    and the existing backfill (resolve_title_to_noc_with_score) runs.

    Partner posture: the endpoint is public and unauthenticated by
    AWIC's own portal design. We identify ourselves explicitly via
    User-Agent and will stop or adjust on partner request.
    """

    source_name = "awic_jobs"
    default_url = "https://awic.ca/wp-json/wedatatools/v1/get-jobs-geojson"

    def fetch(self) -> Iterable[NormalizedJob]:
        cfg = _config("awic_jobs")
        if cfg is None or not cfg.enabled:
            log.info("AWIC jobs disabled - skipping")
            return
        url = cfg.url if cfg.url and not cfg.url.startswith("PLACEHOLDER") else self.default_url
        yield from _fetch_awic_geojson(url)


# ----------------------------------------- 3. Welcome to SSM Careers (stub)
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

    # Step 1A (2026-07-15) description axis: bypass `_get` because it
    # silently normalizes non-string terminals to "" — that would hide
    # parse_error signal for shape-defective WP payloads. Walk the path
    # directly and let `_classify_description_field` distinguish
    # missing (None / empty) from parse_error (non-string / oversized).
    _wp_content = item.get("content")
    if isinstance(_wp_content, dict):
        _raw_rendered = _wp_content.get("rendered")
    else:
        _raw_rendered = None

    _description_parse_failed = False
    if _raw_rendered is not None and not isinstance(_raw_rendered, str):
        # WP `content.rendered` should always be a string; a dict/list
        # is a shape defect that must surface as parse_error, not be
        # silently coerced.
        description = None
        _description_parse_failed = True
    else:
        try:
            description = _clean_wp_html(_raw_rendered)
        except Exception:  # noqa: BLE001
            description = None
            _description_parse_failed = True

    salary_text = _str_or_none(meta.get("_job_salary"))

    posted_date = parse_date_loose(_iso_date_part(item.get("date")))

    # Closing date: WP Job Manager stores _job_expires (sometimes empty).
    closing_date = parse_date_loose(meta.get("_job_expires"))

    # Step 1A (2026-07-15) description axis. SCCC captures full WP-
    # rendered content when parsing succeeds. Above try/except sets
    # _description_parse_failed=True when _clean_wp_html raises.
    # _classify_description_field also detects non-string / oversized
    # payloads that slipped through.
    if _description_parse_failed:
        description_full = None
        description_evidence_status = "parse_error"
    else:
        description_full, description_status = _classify_description_field(
            description
        )
        if description_status == "ok":
            description_evidence_status = "full_source"
        elif description_status == "parse_error":
            description_full = None
            description_evidence_status = "parse_error"
        else:  # "missing"
            description_full = None
            description_evidence_status = "missing"

    # Step 1A location axis. Source-declared string comes from
    # `location_raw` (meta._job_location). `_classify_declared_
    # location_field` distinguishes missing from invalid (present-but-
    # non-string). Non-string locations MUST produce invalid, not
    # silently coerced to a string.
    source_location_text, loc_field_status = _classify_declared_location_field(
        location_raw
    )
    if loc_field_status == "invalid":
        normalized_loc = None
        remote_from_canon = False
        location_resolution_status = "invalid"
        location_provenance = "source_declared"  # source tried, we couldn't use
    else:
        (normalized_loc, remote_from_canon) = normalize_declared_job_location(
            source_location_text
        )
        if normalized_loc is not None:
            location_resolution_status = "resolved"
            location_provenance = "source_declared"
        else:
            location_resolution_status = "missing"
            location_provenance = "none" if source_location_text is None else "source_declared"

    return NormalizedJob(
        source="sccc",
        source_job_id=str(wp_id),
        title=title,
        employer=employer,
        # Legacy `location` and `description` derived by upsert_job.
        region_code=None,
        url=link or None,
        posted_date=posted_date,
        closing_date=closing_date,
        salary_text=salary_text,
        salary_low=parse_float_loose(meta.get("_job_salary_low")),
        salary_high=parse_float_loose(meta.get("_job_salary_high")),
        employment_type=None,  # job-types taxonomy; deferred to 9B
        remote_flag=remote_flag or remote_from_canon,
        noc_code=None,
        # Step 1A description axis.
        description_full=description_full,
        description_excerpt=None,
        description_evidence_status=description_evidence_status,
        # Step 1A location axis.
        source_location_text=source_location_text,
        source_coordinates=None,  # SCCC doesn't provide coordinates
        normalized_job_location=normalized_loc,
        location_resolution_status=location_resolution_status,
        location_provenance=location_provenance,
        raw_payload={
            "source": "sccc_wp_rest_v2",
            "wp_id": wp_id,
            "slug": item.get("slug"),
            "date": item.get("date"),
            "modified": item.get("modified"),
            "link": link,
            "application": _str_or_none(meta.get("_application")),
            "location_raw": location_raw,
            # Step 1A (2026-07-16): full WP item stored so Step 1A
            # backfill can re-normalize from raw with the same logic
            # as live ingest. Historical SCCC rows lack this key and
            # take the partial location-only backfill path (see
            # pipeline.step1a_backfill).
            "item": item,
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


# ============================================================================
# AWIC jobs GeoJSON client
# ============================================================================
AWIC_JOBS_USER_AGENT = "SkillBridge-SSM/0.1 (research; nordik.org)"
AWIC_JOBS_REQUEST_TIMEOUT = 30

# NOC 2021 code validity — exactly 5 digits, all numeric.
_NOC_2021_PATTERN = re.compile(r"^\d{5}$")


def _is_valid_noc_2021_code(code) -> bool:
    """True iff `code` is a string of exactly 5 numeric digits (NOC 2021).

    Anything else -- None, empty, wrong length, non-numeric -- returns
    False. Caller uses False as the signal to leave noc_code=None on
    the NormalizedJob and let the downstream backfill run.
    """
    return isinstance(code, str) and bool(_NOC_2021_PATTERN.match(code))


def _validate_geojson_coordinates(coords) -> tuple[list | None, str]:
    """Step 1A (2026-07-15) coordinate validation per spec §2d.

    Contract: GeoJSON coordinate values MUST be JSON arrays (Python
    lists after deserialization). Tuples are rejected as invalid —
    the GeoJSON spec doesn't produce them and accepting them was a
    review-flagged latitude.

    Returns (validated_list_or_none, status). Status is one of:
      "valid"   — coords passed all checks; array preserved as-is
                  with any 3-D altitude value intact.
      "missing" — coords parameter was None.
      "invalid" — coords present but malformed (non-list, length < 2,
                  non-numeric, longitude out of [-180, 180], or
                  latitude out of [-90, 90]).

    Extras beyond index 1 (altitude, etc.) are preserved verbatim
    but ignored for validation — a 3-D GeoJSON point is valid.
    """
    if coords is None:
        return None, "missing"
    if not isinstance(coords, list):
        # Tuples are rejected — GeoJSON spec says JSON array only.
        return None, "invalid"
    if len(coords) < 2:
        return None, "invalid"

    lon, lat = coords[0], coords[1]
    # Reject booleans (bool is a subclass of int in Python).
    if isinstance(lon, bool) or isinstance(lat, bool):
        return None, "invalid"
    if not isinstance(lon, (int, float)) or not isinstance(lat, (int, float)):
        return None, "invalid"
    if not (-180.0 <= float(lon) <= 180.0):
        return None, "invalid"
    if not (-90.0 <= float(lat) <= 90.0):
        return None, "invalid"

    # Preserve the full array (list-of-numbers, altitude included).
    return list(coords), "valid"


# Description-parse sanity guard: reject payloads that would be a
# data-format leak into the DB (e.g., a JSON blob embedded in the
# description column, or an accidental full-page HTML dump).
_DESCRIPTION_MAX_BYTES: int = 1 * 1024 * 1024  # 1 MB


def _classify_description_field(value) -> tuple[str | None, str]:
    """Step 1A (2026-07-15) description classification per spec §2g.

    Given a raw description-shaped input from a connector, returns
    (normalized_text_or_none, status).

    Rules (provenance-driven, not character-count):
      - value is None                            → (None, "missing")
      - value is not a string                    → (None, "parse_error")
      - value.strip() is empty                   → (None, "missing")
      - len(value_bytes) > _DESCRIPTION_MAX_BYTES → (None, "parse_error")
      - otherwise                                → (stripped_text, "ok")

    The "ok" status is a transient label — callers decide whether the
    ok'd text populates description_full (full_source) or
    description_excerpt (excerpt_only) based on their source's
    provenance. Missing and parse_error are terminal labels.
    """
    if value is None:
        return None, "missing"
    if not isinstance(value, str):
        return None, "parse_error"
    try:
        # Byte-length check catches oversized payloads. Encode with
        # errors=replace so a partially-corrupted string can still be
        # measured — the point is size, not correctness.
        if len(value.encode("utf-8", errors="replace")) > _DESCRIPTION_MAX_BYTES:
            return None, "parse_error"
    except Exception:  # noqa: BLE001
        return None, "parse_error"
    stripped = value.strip()
    if not stripped:
        return None, "missing"
    return stripped, "ok"


def _classify_declared_location_field(value) -> tuple[str | None, str]:
    """Step 1A (2026-07-15) source-declared location classification.

    Returns (source_location_text_or_none, status).

    Rules:
      - value is None       → (None, "missing")
      - value is not a str  → (None, "invalid")   ← distinct from missing
      - value.strip() empty → (None, "missing")
      - otherwise           → (stripped_text, "ok")

    The invalid case (present-but-non-string) MUST NOT be silently
    converted to missing. It's a data-shape defect the caller should
    persist as location_resolution_status="invalid" with
    location_provenance="source_declared" — same reasoning as
    malformed geometry: the source tried, we couldn't use it.
    """
    if value is None:
        return None, "missing"
    if not isinstance(value, str):
        return None, "invalid"
    stripped = value.strip()
    if not stripped:
        return None, "missing"
    return stripped, "ok"


def _is_in_ssm_bbox(coords) -> bool:
    """True iff [lng, lat] falls within the configured SSM bounding box.

    Coordinates are WGS84 (matches AWIC's GeoJSON CRS
    urn:ogc:def:crs:OGC:1.3:CRS84). Missing / malformed / non-numeric
    coordinates return False -- the caller treats that as the
    'no_coords' drop reason.
    """
    from config import (
        SSM_BBOX_LAT_MAX,
        SSM_BBOX_LAT_MIN,
        SSM_BBOX_LNG_MAX,
        SSM_BBOX_LNG_MIN,
    )
    if not isinstance(coords, (list, tuple)) or len(coords) < 2:
        return False
    lng, lat = coords[0], coords[1]
    if isinstance(lng, bool) or isinstance(lat, bool):  # bool is int subclass
        return False
    if not isinstance(lng, (int, float)) or not isinstance(lat, (int, float)):
        return False
    return (
        SSM_BBOX_LAT_MIN <= lat <= SSM_BBOX_LAT_MAX
        and SSM_BBOX_LNG_MIN <= lng <= SSM_BBOX_LNG_MAX
    )


def _fetch_awic_geojson(url: str) -> Iterable[NormalizedJob]:
    """Yield NormalizedJob items from AWIC's public GeoJSON endpoint.

    Single request; AWIC returns the full FeatureCollection in one
    response (~150KB / ~160 jobs at capture time). No pagination.

    On any transport / HTTP / JSON error, logs and returns without
    yielding -- no exceptions cross the boundary so one bad source
    can't poison an orchestrator run.

    Emits per-run observability counters at INFO on completion:
      fetched                -- total features seen in the response
      yielded                -- features passed through to the caller
                                (Step 1A: coordinate-based drops
                                removed; only shape-defect drops)
      dropped_malformed      -- feature was missing post_id / title
      with_noc_provided      -- yielded feature had a valid nocs_2021
      needing_noc_backfill   -- yielded feature had no valid nocs_2021
                                (downstream resolve_title_to_noc runs)
      dropped_no_coords      -- LEGACY (always 0 post-Step-1A).
                                Backward-compat for pipeline snapshot.
      dropped_outside_ssm    -- LEGACY (always 0 post-Step-1A).
                                Backward-compat for pipeline snapshot.

    Under Step 1A (2026-07-15) the connector no longer uses
    coordinates or the SSM bounding box as an ingestion gate — the
    live SSM market is defined by v_current_job's location-resolution
    clauses, not by ingest-time geometry filtering. AWIC rows with
    unresolved / missing / invalid location stay outside v_current_job
    but persist in core.job_posting for Step 1B detail-page fetching
    to resolve later.
    """
    headers = {
        "User-Agent": AWIC_JOBS_USER_AGENT,
        "Accept": "application/json",
    }
    fetched = 0
    dropped_no_coords = 0
    dropped_outside_ssm = 0
    dropped_malformed = 0
    yielded = 0
    with_noc_provided = 0
    needing_noc_backfill = 0

    try:
        with httpx.Client(
            timeout=AWIC_JOBS_REQUEST_TIMEOUT, headers=headers,
            follow_redirects=True,
        ) as client:
            resp = client.get(url)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                log.error("AWIC jobs HTTP %d: %s", resp.status_code, e)
                return
            try:
                payload = resp.json()
            except ValueError:
                log.error("AWIC jobs returned non-JSON")
                return
    except httpx.RequestError as e:
        log.error("AWIC jobs request failed: %s", e)
        return

    # AWIC wraps the FeatureCollection in a top-level "data" key:
    #   {"data": {"type": "FeatureCollection", "features": [...]}}
    # Fall back to a bare FeatureCollection shape defensively.
    data = payload.get("data", payload) if isinstance(payload, dict) else {}
    features = data.get("features") if isinstance(data, dict) else None
    if not isinstance(features, list):
        log.error("AWIC jobs: features is not a list")
        return

    # Step 1A (2026-07-15): coordinate-based drop paths removed. Only
    # genuine shape defects (missing post_id / title) drop features
    # now. Coordinate state becomes NormalizedJob metadata (see
    # `_normalize_awic_geojson_feature`).
    for ft in features:
        fetched += 1
        job, drop_reason, noc_provided = _normalize_awic_geojson_feature(ft)
        if drop_reason == "malformed":
            dropped_malformed += 1
            continue
        if job is None:
            # Defensive; normalizer should always pair job=None with a
            # drop_reason. If we hit this branch, log and move on.
            dropped_malformed += 1
            continue
        if noc_provided:
            with_noc_provided += 1
        else:
            needing_noc_backfill += 1
        yielded += 1
        yield job

    # Post-Step-1A telemetry: fetched-vs-yielded is the honest signal.
    # Coordinate counters retained for backward-compat but always 0 —
    # the pipeline snapshot consumer can be updated in a follow-on PR.
    log.info(
        "AWIC jobs: fetched=%d yielded=%d dropped_malformed=%d "
        "with_noc_provided=%d needing_noc_backfill=%d "
        "dropped_no_coords=%d dropped_outside_ssm=%d",
        fetched, yielded, dropped_malformed,
        with_noc_provided, needing_noc_backfill,
        dropped_no_coords, dropped_outside_ssm,
    )


def _normalize_awic_geojson_feature(
    feature,
) -> tuple[NormalizedJob | None, str | None, bool]:
    """Map one AWIC GeoJSON feature to NormalizedJob.

    Step 1A (2026-07-15): coordinate-based ingestion gates removed.
    The pre-Step-1A logic dropped features on missing / malformed /
    outside-bbox coordinates, using geometry that Step 1A explicitly
    labels untrustworthy for job location. Under Step 1A those cases
    become NormalizedJob attributes, not drops:

      Valid coordinates, anywhere → ingest,
        location_resolution_status="unresolved", provenance="geometry"
      Missing coordinates         → ingest,
        location_resolution_status="missing",    provenance="none"
      Malformed coordinates       → ingest,
        location_resolution_status="invalid",    provenance="geometry"

    All AWIC rows stay OUTSIDE `v_current_job` because none carries a
    verified source-declared location — the view filter requires
    `normalized_job_location = 'Sault Ste. Marie'` AND status =
    'resolved'. Persistence continues so Step 1B's detail-page fetch
    can populate real locations later without re-scraping.

    Only genuine data-shape defects drop the feature: non-dict input,
    non-dict properties, missing post_id, or missing/blank title.

    Returns a 3-tuple: (job_or_None, drop_reason_or_None, noc_provided).

      drop_reason values (mutually exclusive):
        "malformed" — feature shape or required identifiers missing
        None        — feature was accepted; `job` is populated

      noc_provided is True iff the feature carried a valid 5-digit
      NOC 2021 code that we passed through. False means job.noc_code is
      None and the downstream backfill will resolve it from the title.
    """
    if not isinstance(feature, dict):
        return None, "malformed", False

    props = feature.get("properties") or {}
    if not isinstance(props, dict):
        return None, "malformed", False

    post_id = props.get("post_id")
    title = props.get("job_title")
    if not post_id or not title:
        return None, "malformed", False
    title_str = str(title).strip()
    if not title_str:
        return None, "malformed", False

    # Step 1A (2026-07-15) coordinate classification. Three states
    # matter and MUST NOT collapse:
    #
    #   1. Geometry key absent OR value is None
    #      → source made no attempt to supply geometry
    #      → location_resolution_status = "missing"
    #      → location_provenance          = "none"
    #
    #   2. Geometry present but not a dict (e.g., "bad-shape" string)
    #      → source ATTEMPTED to supply geometry but shape is broken
    #      → location_resolution_status = "invalid"
    #      → location_provenance          = "geometry"
    #
    #   3. Geometry is a dict → look at coordinates key. Validator
    #      distinguishes valid coords (unresolved/geometry) from
    #      invalid coords (invalid/geometry) from missing coords
    #      (missing/none).
    #
    # Correction 2026-07-16: pre-fix code collapsed case 2 to case 1
    # (`feature.get("geometry") or {}` swallowed non-dict into empty
    # dict, then treated absent coordinates as missing). That erased
    # the distinction between "source provided nothing" and "source
    # provided garbage" — critical for the historical backfill so we
    # can distinguish absent data from corrupted data.
    geometry_raw = feature.get("geometry")
    if geometry_raw is None:
        # Case 1: no attempt to supply geometry.
        coords = None
        _geom_shape_invalid = False
    elif not isinstance(geometry_raw, dict):
        # Case 2: geometry supplied but not a dict — the shape itself
        # is broken. Skip coordinate extraction; validator's "missing"
        # would misclassify this as absent.
        coords = None
        _geom_shape_invalid = True
    else:
        # Case 3: dict-shaped; validator classifies coordinates.
        coords = geometry_raw.get("coordinates")
        _geom_shape_invalid = False

    # NOC 2021: first valid 5-digit code; else None + backfill runs.
    noc_provided = False
    noc_code: str | None = None
    nocs_2021 = props.get("nocs_2021") or []
    if isinstance(nocs_2021, list):
        for candidate in nocs_2021:
            if _is_valid_noc_2021_code(candidate):
                noc_code = candidate
                noc_provided = True
                break

    # AWIC's "type" field is short-form (FT / PT / CO). Pass through
    # unchanged; the general employment_type enum in downstream code
    # accepts these tokens.
    employment_type = _str_or_none(props.get("type"))

    # Step 1A (2026-07-15) description axis. AWIC v1 GeoJSON does NOT
    # carry the full JD body — only an excerpt. Step 1B will add
    # detail-page fetching to recover the full description; until then
    # every AWIC row is excerpt_only, missing, or parse_error. Uses
    # _classify_description_field so a non-string / oversized excerpt
    # produces parse_error rather than being silently stringified.
    excerpt_text, excerpt_status = _classify_description_field(
        props.get("excerpt")
    )
    if excerpt_status == "ok":
        description_excerpt = excerpt_text
        description_evidence_status = "excerpt_only"
    elif excerpt_status == "parse_error":
        description_excerpt = None
        description_evidence_status = "parse_error"
    else:  # "missing"
        description_excerpt = None
        description_evidence_status = "missing"

    # Step 1A location axis. AWIC v1 has NO source-declared location
    # field and geometry is not authoritative for job location
    # (verified: Community Support Worker in Wawa has coordinates
    # pointing at downtown SSM). We preserve geometry as evidence
    # metadata but never treat it as a job location.
    if _geom_shape_invalid:
        # Case 2 from coordinate classification block above: geometry
        # supplied but not dict-shaped. Source attempted → provenance
        # is "geometry"; the attempt failed → status is "invalid".
        validated_coords = None
        location_resolution_status = "invalid"
        location_provenance = "geometry"
    else:
        validated_coords, coord_status = _validate_geojson_coordinates(coords)
        if coord_status == "valid":
            location_resolution_status = "unresolved"  # geometry alone can't resolve
            location_provenance = "geometry"
        elif coord_status == "invalid":
            location_resolution_status = "invalid"
            location_provenance = "geometry"  # tried but failed
        else:  # coord_status == "missing"
            location_resolution_status = "missing"
            location_provenance = "none"

    job = NormalizedJob(
        source="awic_jobs",
        source_job_id=str(post_id),
        title=title_str,
        employer=_str_or_none(props.get("employer")),
        # Legacy `location` and `description` are DERIVED by upsert_job
        # from the new-axis fields when the connector emits an explicit
        # evidence status (see base.py `upsert_job` transitional
        # persistence). Do NOT set them here.
        region_code=None,
        url=_str_or_none(props.get("url")),
        # AWIC's GeoJSON does not include posted / closing dates in v1.
        posted_date=None,
        closing_date=None,
        salary_text=None,
        salary_low=None,
        salary_high=None,
        employment_type=employment_type,
        remote_flag=None,
        noc_code=noc_code,
        # Step 1A description axis.
        description_full=None,          # never captured from AWIC v1 GeoJSON
        description_excerpt=description_excerpt,
        description_evidence_status=description_evidence_status,
        # Step 1A location axis. NEVER hardcode "Sault Ste. Marie".
        source_location_text=None,       # AWIC v1 has no such property
        source_coordinates=validated_coords,
        normalized_job_location=None,    # geometry alone can't resolve
        location_resolution_status=location_resolution_status,
        location_provenance=location_provenance,
        raw_payload={
            "source": "awic_geojson_v1",
            "feature": feature,  # full GeoJSON feature for audit / replay
        },
    )
    return job, None, noc_provided


def _is_sccc_ssm_location(location: str | None, url: str) -> bool:
    """Keep SCCC ingestion scoped to the Sault Ste. Marie + Algoma area.

    SCCC's mandate covers the full Algoma District, so a posting in
    Wawa or Chapleau is legitimately local to a newcomer using
    SkillBridge SSM. Communities are configured via
    SCCC_INGEST_LOCALITIES in .env (renamed 2026-07-16 from
    LOCAL_CITIES) so the ingest scope can be tuned without code
    changes. This is an INGESTION allowlist only — the matching engine
    ignores it; SSM-only v_current_job is the market boundary.

    Empty locations are excluded rather than assumed local — the API
    exposes _job_location explicitly, so missing data is a real
    signal, not just a parsing miss.

    Note: the URL is included in the haystack so SCCC slugs containing
    a city name ("sault-ste-marie", "chapleau", etc.) keep matching
    even when the metadata field is unusual.
    """
    from config import SCCC_INGEST_LOCALITIES

    if not (location or "").strip():
        return False
    haystack = f"{location or ''} {url}".lower()
    return any(city in haystack for city in SCCC_INGEST_LOCALITIES)


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

    # Step 1A (2026-07-15) description axis. Classifier tolerates
    # non-string / oversized inputs (produces parse_error rather
    # than crashing on .strip()).
    raw_desc = row.get("description")
    if raw_desc is None:
        raw_desc = row.get("Job Description")
    description_full, desc_status = _classify_description_field(raw_desc)
    if desc_status == "ok":
        description_evidence_status = "full_source"
    elif desc_status == "parse_error":
        description_full = None
        description_evidence_status = "parse_error"
    else:  # "missing"
        description_full = None
        description_evidence_status = "missing"

    # Step 1A location axis. Classifier distinguishes present-but-
    # non-string (invalid) from missing.
    raw_loc = row.get("location")
    if raw_loc is None:
        raw_loc = row.get("Location")
    source_location_text, loc_field_status = _classify_declared_location_field(
        raw_loc
    )
    if loc_field_status == "invalid":
        normalized_loc = None
        remote_flag = False
        location_resolution_status = "invalid"
        location_provenance = "source_declared"
    else:
        (normalized_loc, remote_flag) = normalize_declared_job_location(
            source_location_text
        )
        if normalized_loc is not None:
            location_resolution_status = "resolved"
            location_provenance = "source_declared"
        else:
            location_resolution_status = "missing"
            location_provenance = "none" if source_location_text is None else "source_declared"

    return NormalizedJob(
        source=source,
        source_job_id=source_job_id,
        title=title,
        employer=(row.get("employer") or row.get("Employer") or "").strip() or None,
        # Legacy `location` / `description` derived by upsert_job.
        region_code=row.get("region_code") or None,
        url=row.get("url") or row.get("URL") or None,
        posted_date=parse_date_loose(row.get("posted_date") or row.get("Date Posted")),
        closing_date=parse_date_loose(row.get("closing_date") or row.get("Closing Date")),
        salary_text=(row.get("salary_text") or row.get("Salary") or row.get("Wage") or "").strip() or None,
        salary_low=parse_float_loose(row.get("salary_low") or row.get("Min Wage")),
        salary_high=parse_float_loose(row.get("salary_high") or row.get("Max Wage")),
        employment_type=(row.get("employment_type") or row.get("Employment Type") or "").strip() or None,
        remote_flag=remote_flag or None,
        noc_code=(row.get("noc_code") or row.get("NOC Code") or "").strip() or None,
        description_full=description_full,
        description_excerpt=None,
        description_evidence_status=description_evidence_status,
        source_location_text=source_location_text,
        source_coordinates=None,
        normalized_job_location=normalized_loc,
        location_resolution_status=location_resolution_status,
        location_provenance=location_provenance,
        raw_payload=dict(row),
    )


ALL_CONNECTORS: list[type[JobConnector]] = [
    SCCCConnector,
    AWICJobsConnector,
    WelcomeSSMConnector,
    CitySSMConnector,
]
