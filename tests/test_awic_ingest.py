"""Unit tests for the AWIC jobs connector (Step 2 of AWIC v1).

DB-free, no live network, no LLM. Tests exercise:
  - _is_valid_noc_2021_code           (pure predicate)
  - _is_in_ssm_bbox                   (pure predicate using config bbox)
  - _normalize_awic_geojson_feature   (pure function; 4 real+synthetic
                                       fixture features exercise every
                                       code path exactly once)
  - _fetch_awic_geojson               (mocked httpx client; verifies the
                                       response-shape handling and the
                                       counter log emission)
  - AWICJobsConnector.fetch()         (gate + delegate; disabled-cfg
                                       and PLACEHOLDER-URL paths)

Fixture: tests/fixtures/awic_jobs_geojson_sample.json (created in
Step 1). Contains 4 features labelled by properties._test_case:
  real_ssm_valid_noc     -- happy path (post_id 8301829)
  real_outside_ssm       -- dropped by bbox filter (post_id 8827928)
  real_ssm_invalid_noc   -- kept, noc_provided=False (post_id 8705234)
  synth_missing_coords   -- dropped by coords filter (post_id 99999901)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
import pytest

from skillbridge.ingest.partners import (
    AWIC_JOBS_USER_AGENT,
    AWICJobsConnector,
    _fetch_awic_geojson,
    _is_in_ssm_bbox,
    _is_valid_noc_2021_code,
    _normalize_awic_geojson_feature,
)


pytestmark = pytest.mark.nodb


FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "awic_jobs_geojson_sample.json"
)


@pytest.fixture(scope="module")
def fixture_payload() -> dict:
    """Load the curated GeoJSON fixture once per module."""
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _feature_by_case(payload: dict, case: str) -> dict:
    """Pull the fixture feature matching properties._test_case."""
    for ft in payload["data"]["features"]:
        if ft["properties"].get("_test_case") == case:
            return ft
    raise KeyError(f"fixture case {case!r} not found")


# ---------------------------------------------------------------- predicates


class TestIsValidNoc2021Code:
    @pytest.mark.parametrize("code", ["31203", "13110", "14200", "00000"])
    def test_five_digit_numeric_strings_are_valid(self, code):
        assert _is_valid_noc_2021_code(code) is True

    @pytest.mark.parametrize("bad", [
        None,
        "",
        "1234",           # 4 digits
        "123456",         # 6 digits
        "3120X",          # non-numeric char
        " 31203",         # leading whitespace
        "31203 ",         # trailing whitespace
        "31 03",          # embedded space
        31203,            # not a string
        ["31203"],        # not a string
    ])
    def test_everything_else_is_invalid(self, bad):
        assert _is_valid_noc_2021_code(bad) is False


class TestIsInSsmBbox:
    """SSM_BBOX_* defaults are 46.4..46.6 lat, -84.5..-84.2 lng
    (config.py)."""

    def test_sault_downtown_coords_are_in(self):
        # Real SSM feature coords from the fixture.
        assert _is_in_ssm_bbox([-84.319367, 46.5483163]) is True

    def test_chapleau_area_coords_are_out(self):
        # Real outside-SSM feature coords from the fixture (~Chapleau).
        assert _is_in_ssm_bbox([-84.784403, 47.994282]) is False

    @pytest.mark.parametrize("bad", [
        None,
        [],
        [123],            # short
        [None, None],
        ["a", "b"],
        [True, False],    # bool is int subclass in Python; we exclude
        "not a list",
        {"lng": -84, "lat": 46},  # dict, not list
    ])
    def test_malformed_coords_are_out(self, bad):
        assert _is_in_ssm_bbox(bad) is False

    def test_corners_are_inclusive(self):
        # Default bbox corners; inclusive by design.
        assert _is_in_ssm_bbox([-84.5, 46.4]) is True
        assert _is_in_ssm_bbox([-84.2, 46.6]) is True

    def test_just_outside_corners(self):
        assert _is_in_ssm_bbox([-84.51, 46.4]) is False
        assert _is_in_ssm_bbox([-84.5, 46.39]) is False


# ---------------------------------------------------------------- normalize


class TestNormalizeFeature:
    """Each fixture case exercises exactly one code path of the
    normalizer. Adding coverage here means adding a case to the
    fixture, not just parameterizing existing features."""

    def test_real_ssm_valid_noc_yields_populated_job(self, fixture_payload):
        """Step 1A (2026-07-15) contract change: AWIC never hardcodes
        `location = "Sault Ste. Marie"`. Legacy `location` is None
        because normalization from geometry alone is not honest;
        `normalized_job_location` is None and status is 'unresolved'
        (geometry present but not authoritative). The posting stays
        outside the live SSM market until Step 1B detail-page fetch
        supplies a source-declared location."""
        ft = _feature_by_case(fixture_payload, "real_ssm_valid_noc")
        job, drop_reason, noc_provided = _normalize_awic_geojson_feature(ft)

        assert drop_reason is None
        assert noc_provided is True
        assert job is not None
        assert job.source == "awic_jobs"          # locked contract
        assert job.source_job_id == "8301829"     # str(post_id)
        assert job.noc_code == "31203"            # passed through
        # Step 1A: NO hardcoded fallback location.
        assert job.location is None
        assert job.source_location_text is None
        assert job.normalized_job_location is None
        assert job.location_resolution_status == "unresolved"
        assert job.location_provenance == "geometry"
        assert job.source_coordinates is not None  # geometry preserved
        assert job.title  # non-empty
        assert job.employer == "Sault Area Hospital Foundation"
        assert job.url and job.url.startswith("https://")
        # Description: excerpt-only under Step 1A until 1B detail fetch.
        assert job.description_full is None
        assert job.description_excerpt  # excerpt populated
        assert job.description_evidence_status == "excerpt_only"
        assert job.raw_payload["source"] == "awic_geojson_v1"
        assert job.raw_payload["feature"] is ft   # full audit copy

    def test_real_outside_ssm_ingested_as_unresolved(self, fixture_payload):
        """Step 1A (2026-07-15): outside-SSM-bbox is no longer a drop
        reason. The pre-Step-1A logic used geometry as an ingestion
        gate; Step 1A says geometry is not authoritative for job
        location (verified: Wawa job has SSM coordinates). Under Step
        1A, outside-bbox features are ingested with
        location_resolution_status='unresolved' and stay outside
        v_current_job because the view requires resolved SSM."""
        ft = _feature_by_case(fixture_payload, "real_outside_ssm")
        job, drop_reason, noc_provided = _normalize_awic_geojson_feature(ft)
        assert job is not None
        assert drop_reason is None
        # Coordinates preserved as evidence, but never treated as
        # authoritative for job location.
        assert job.source_coordinates is not None
        assert job.normalized_job_location is None
        assert job.location_resolution_status == "unresolved"
        assert job.location_provenance == "geometry"

    def test_real_ssm_invalid_noc_kept_but_no_noc(self, fixture_payload):
        """Missing/invalid nocs_2021 => NormalizedJob with noc_code=None,
        noc_provided=False. Downstream backfill will resolve it from
        the title. This is the SCCC-equivalent path."""
        ft = _feature_by_case(fixture_payload, "real_ssm_invalid_noc")
        job, drop_reason, noc_provided = _normalize_awic_geojson_feature(ft)

        assert drop_reason is None
        assert job is not None
        assert job.noc_code is None
        assert noc_provided is False
        # Step 1A: NO hardcoded fallback location.
        assert job.source == "awic_jobs"
        assert job.location is None
        assert job.source_location_text is None
        assert job.normalized_job_location is None
        assert job.location_resolution_status == "unresolved"
        assert job.location_provenance == "geometry"

    def test_synth_missing_coords_ingested_as_missing(self, fixture_payload):
        """Step 1A (2026-07-15): missing coordinates is no longer a
        drop reason. Feature is ingested with
        location_resolution_status='missing' and provenance='none'.
        Row stays outside v_current_job (unresolved location)."""
        ft = _feature_by_case(fixture_payload, "synth_missing_coords")
        job, drop_reason, noc_provided = _normalize_awic_geojson_feature(ft)
        assert job is not None
        assert drop_reason is None
        assert job.source_coordinates is None
        assert job.location_resolution_status == "missing"
        assert job.location_provenance == "none"

    def test_malformed_feature_dropped(self):
        """Step 1A (2026-07-15): only genuine data-shape defects drop.
        Non-dict feature and missing post_id/title. Missing geometry
        is now an ingest with location_resolution_status='missing'."""
        # Not a dict
        j, r, n = _normalize_awic_geojson_feature("not a feature")
        assert j is None and r == "malformed" and n is False
        # Missing geometry entirely: pre-Step-1A dropped as no_coords;
        # Step 1A ingests with title/id present. When BOTH properties
        # AND geometry are absent, the missing-title is what drops.
        j, r, n = _normalize_awic_geojson_feature({"properties": {}})
        assert j is None and r == "malformed"
        # Coords present + in-bbox, but missing post_id/title
        good_geom = {"type": "Point", "coordinates": [-84.32, 46.54]}
        j, r, n = _normalize_awic_geojson_feature({
            "geometry": good_geom, "properties": {"post_id": 1},
        })
        assert j is None and r == "malformed" and n is False

    def test_missing_geometry_with_valid_title_ingests(self):
        """Step 1A regression: feature without geometry but with valid
        post_id + title now INGESTS with location_resolution_status
        ='missing', not dropped."""
        j, r, n = _normalize_awic_geojson_feature({
            "properties": {"post_id": 999, "job_title": "Test Role"},
        })
        assert j is not None
        assert r is None
        assert j.source_coordinates is None
        assert j.location_resolution_status == "missing"
        assert j.location_provenance == "none"

    def test_malformed_geometry_ingests_as_invalid(self):
        """Step 1A: malformed coordinates (out-of-range latitude) is
        now an ingest, not a drop. Row stays outside v_current_job."""
        j, r, n = _normalize_awic_geojson_feature({
            "geometry": {"type": "Point", "coordinates": [-84.32, 999.0]},
            "properties": {"post_id": 1, "job_title": "Test Role"},
        })
        assert j is not None
        assert r is None
        assert j.source_coordinates is None  # invalid rejected
        assert j.location_resolution_status == "invalid"
        assert j.location_provenance == "geometry"

    def test_tuple_coordinates_rejected(self):
        """Step 1A §2d: GeoJSON contract requires JSON array (list).
        Tuples are rejected as invalid — spec correction 2026-07-16."""
        from skillbridge.ingest.partners import _validate_geojson_coordinates
        coords_list, status = _validate_geojson_coordinates(
            (-84.32, 46.54)  # tuple, not list
        )
        assert coords_list is None
        assert status == "invalid"

    def test_three_dimensional_coordinates_preserved(self):
        """Step 1A §2d: 3-D GeoJSON points ([lon, lat, altitude]) are
        valid — extras preserved, ignored for validation."""
        from skillbridge.ingest.partners import _validate_geojson_coordinates
        coords_list, status = _validate_geojson_coordinates(
            [-84.32, 46.54, 150.0]
        )
        assert status == "valid"
        assert coords_list == [-84.32, 46.54, 150.0]

    def test_non_string_excerpt_produces_parse_error(self):
        """Step 1A §2g: non-string description input must produce
        parse_error, not be silently stringified."""
        j, r, n = _normalize_awic_geojson_feature({
            "properties": {
                "post_id": 1,
                "job_title": "Test",
                "excerpt": {"nested": "dict"},  # non-string
            },
        })
        assert j is not None
        assert j.description_excerpt is None
        assert j.description_evidence_status == "parse_error"

    def test_non_dict_geometry_ingests_as_invalid_not_missing(self):
        """Step 1A correction 2026-07-16: geometry supplied but not a
        dict (e.g., a string like 'bad-shape') was previously
        collapsed to missing/none. The honest classification is
        invalid/geometry — source attempted to supply geometry, we
        couldn't use its shape. Distinguishing this from truly-
        missing geometry is load-bearing for the upcoming historical
        backfill, which needs to tell absent data apart from
        corrupted data."""
        j, r, n = _normalize_awic_geojson_feature({
            "geometry": "bad-shape",  # non-dict — attempted but broken
            "properties": {"post_id": 1, "job_title": "Test Role"},
        })
        assert j is not None
        assert r is None
        assert j.source_coordinates is None
        assert j.location_resolution_status == "invalid"
        assert j.location_provenance == "geometry"

    def test_geometry_none_ingests_as_missing(self):
        """Anti-regression alongside the above: geometry key explicitly
        set to None is genuinely missing, not invalid."""
        j, r, n = _normalize_awic_geojson_feature({
            "geometry": None,
            "properties": {"post_id": 1, "job_title": "Test Role"},
        })
        assert j is not None
        assert j.source_coordinates is None
        assert j.location_resolution_status == "missing"
        assert j.location_provenance == "none"

    def test_geometry_number_ingests_as_invalid(self):
        """Non-dict types other than string also produce invalid."""
        j, r, n = _normalize_awic_geojson_feature({
            "geometry": 42,  # numeric — clearly not a geometry object
            "properties": {"post_id": 1, "job_title": "Test Role"},
        })
        assert j is not None
        assert j.source_coordinates is None
        assert j.location_resolution_status == "invalid"
        assert j.location_provenance == "geometry"


class TestJobBankUrlProvenance:
    """Load-bearing policy test (matches BREAKING.md).

    A synthetic AWIC feature whose properties.url points at a
    jobbank.gc.ca URL must still be stamped source='awic_jobs'. The
    federal-source rule is preserved by source IDENTITY (who curated),
    not by URL inspection (where to apply)."""

    def test_awic_posting_with_jobbank_apply_url_stays_awic_jobs(self):
        feature = {
            "geometry": {"type": "Point",
                         "coordinates": [-84.319367, 46.5483163]},
            "properties": {
                "post_id": 42424242,
                "job_title": "Warehouse Associate",
                "employer": "Some Local Employer",
                "url": "https://www.jobbank.gc.ca/jobsearch/jobposting/12345",
                "nocs_2021": ["75110"],
                "excerpt": "Warehouse work in SSM area.",
            },
        }
        job, drop_reason, noc_provided = _normalize_awic_geojson_feature(feature)
        assert drop_reason is None
        assert job is not None
        # THE key assertion: source is AWIC's curation identity, not the
        # federal apply-URL.
        assert job.source == "awic_jobs"
        # URL is retained as the apply-URL (provenance / navigation only).
        assert job.url == (
            "https://www.jobbank.gc.ca/jobsearch/jobposting/12345"
        )


# ---------------------------------------------------------------- fetch


class _MockTransport(httpx.BaseTransport):
    """Minimal httpx transport that returns a canned JSON payload for
    any request. Used to keep the fetch test hermetic."""

    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self.payload = payload
        self.calls: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        content = (
            json.dumps(self.payload).encode() if self.payload is not None
            else b"not-json"
        )
        return httpx.Response(
            self.status_code,
            content=content,
            headers={"content-type": "application/json"},
            request=request,
        )


class TestFetchGeoJson:
    """End-to-end fetch: patch httpx.Client so the code path exercises
    real response handling + counter emission without hitting the
    network."""

    def test_fetch_yields_expected_jobs_and_logs_counters(
        self, monkeypatch, caplog, fixture_payload,
    ):
        """Feed the whole fixture through _fetch_awic_geojson. The
        fixture has 4 features -> expect 2 yielded (1 valid-NOC +
        1 invalid-NOC), 1 dropped for outside SSM, 1 dropped for
        no-coords."""
        transport = _MockTransport(200, fixture_payload)

        # Swap httpx.Client so the connector's `with httpx.Client(...)`
        # gets our mock transport under the hood.
        original_client = httpx.Client
        def _client_with_mock(*args, **kwargs):
            kwargs["transport"] = transport
            return original_client(*args, **kwargs)
        monkeypatch.setattr(
            "skillbridge.ingest.partners.httpx.Client", _client_with_mock,
        )

        with caplog.at_level(
            logging.INFO, logger="skillbridge.ingest.partners"
        ):
            jobs = list(_fetch_awic_geojson("https://example.invalid/"))

        # Step 1A (2026-07-15): pre-Step-1A logic dropped 2 features
        # (1 outside-bbox + 1 missing-coords) and yielded 2. Post-Step-
        # 1A all 4 features ingest — coordinate-based drops removed
        # per spec §2f. Rows stay outside v_current_job via the view's
        # SSM eligibility clauses; the connector is now honest about
        # what it saw.
        assert len(jobs) == 4
        assert all(j.source == "awic_jobs" for j in jobs)
        yielded_ids = {j.source_job_id for j in jobs}
        # Includes the outside-bbox and missing-coords rows now.
        assert "8301829" in yielded_ids
        assert "8705234" in yielded_ids

        # Verify User-Agent was sent on the request.
        assert len(transport.calls) == 1
        assert transport.calls[0].headers.get("user-agent") == AWIC_JOBS_USER_AGENT

        # Counter emission: post-Step-1A shape.
        counter_records = [
            r for r in caplog.records
            if r.levelno == logging.INFO
            and "AWIC jobs: fetched=" in r.getMessage()
        ]
        assert len(counter_records) == 1
        msg = counter_records[0].getMessage()
        assert "fetched=4" in msg
        assert "yielded=4" in msg
        assert "dropped_malformed=0" in msg
        # Legacy backward-compat counters (always 0 under Step 1A).
        assert "dropped_no_coords=0" in msg
        assert "dropped_outside_ssm=0" in msg

    def test_non_200_response_returns_no_jobs(self, monkeypatch, caplog):
        transport = _MockTransport(500, {})
        original_client = httpx.Client
        def _client_with_mock(*args, **kwargs):
            kwargs["transport"] = transport
            return original_client(*args, **kwargs)
        monkeypatch.setattr(
            "skillbridge.ingest.partners.httpx.Client", _client_with_mock,
        )
        with caplog.at_level(
            logging.ERROR, logger="skillbridge.ingest.partners"
        ):
            jobs = list(_fetch_awic_geojson("https://example.invalid/"))
        assert jobs == []
        assert any("AWIC jobs HTTP 500" in r.getMessage() for r in caplog.records)

    def test_non_json_body_returns_no_jobs(self, monkeypatch, caplog):
        transport = _MockTransport(200, None)  # sends "not-json" bytes
        original_client = httpx.Client
        def _client_with_mock(*args, **kwargs):
            kwargs["transport"] = transport
            return original_client(*args, **kwargs)
        monkeypatch.setattr(
            "skillbridge.ingest.partners.httpx.Client", _client_with_mock,
        )
        with caplog.at_level(
            logging.ERROR, logger="skillbridge.ingest.partners"
        ):
            jobs = list(_fetch_awic_geojson("https://example.invalid/"))
        assert jobs == []
        assert any(
            "returned non-JSON" in r.getMessage() for r in caplog.records
        )

    def test_malformed_features_list_returns_no_jobs(
        self, monkeypatch, caplog,
    ):
        # payload has data.features as a dict, not a list
        bad = {"data": {"type": "FeatureCollection",
                        "features": {"oops": "not a list"}}}
        transport = _MockTransport(200, bad)
        original_client = httpx.Client
        def _client_with_mock(*args, **kwargs):
            kwargs["transport"] = transport
            return original_client(*args, **kwargs)
        monkeypatch.setattr(
            "skillbridge.ingest.partners.httpx.Client", _client_with_mock,
        )
        with caplog.at_level(
            logging.ERROR, logger="skillbridge.ingest.partners"
        ):
            jobs = list(_fetch_awic_geojson("https://example.invalid/"))
        assert jobs == []
        assert any(
            "features is not a list" in r.getMessage()
            for r in caplog.records
        )


# ---------------------------------------------------------------- connector class


class TestConnectorClass:
    """AWICJobsConnector.fetch() gates on config; delegates to
    _fetch_awic_geojson otherwise. Verified by monkeypatching the
    delegate."""

    def test_disabled_source_skips_fetch(self, monkeypatch, caplog):
        # Force cfg.enabled = False by returning a fake config.
        class _FakeCfg:
            enabled = False
            url = ""
        monkeypatch.setattr(
            "skillbridge.ingest.partners._config",
            lambda name: _FakeCfg() if name == "awic_jobs" else None,
        )
        called = {"n": 0}
        monkeypatch.setattr(
            "skillbridge.ingest.partners._fetch_awic_geojson",
            lambda url: (called.__setitem__("n", called["n"] + 1) or iter([])),
        )
        with caplog.at_level(
            logging.INFO, logger="skillbridge.ingest.partners"
        ):
            jobs = list(AWICJobsConnector().fetch())
        assert jobs == []
        assert called["n"] == 0
        assert any(
            "AWIC jobs disabled" in r.getMessage() for r in caplog.records
        )

    def test_missing_cfg_skips_fetch(self, monkeypatch):
        """If the config lookup returns None (source not registered),
        the connector must skip rather than crash."""
        monkeypatch.setattr(
            "skillbridge.ingest.partners._config", lambda name: None,
        )
        called = {"n": 0}
        monkeypatch.setattr(
            "skillbridge.ingest.partners._fetch_awic_geojson",
            lambda url: (called.__setitem__("n", called["n"] + 1) or iter([])),
        )
        jobs = list(AWICJobsConnector().fetch())
        assert jobs == []
        assert called["n"] == 0

    def test_placeholder_url_falls_back_to_default(self, monkeypatch):
        """Match SCCC's contract: PLACEHOLDER_* env values must fall
        back to the connector's hardcoded default_url."""
        class _FakeCfg:
            enabled = True
            url = "PLACEHOLDER_AWIC_JOBS_FEED_URL"
        monkeypatch.setattr(
            "skillbridge.ingest.partners._config",
            lambda name: _FakeCfg() if name == "awic_jobs" else None,
        )
        seen_url = {"value": None}
        def _spy(url):
            seen_url["value"] = url
            return iter([])
        monkeypatch.setattr(
            "skillbridge.ingest.partners._fetch_awic_geojson", _spy,
        )
        _ = list(AWICJobsConnector().fetch())
        assert seen_url["value"] == AWICJobsConnector.default_url

    def test_real_url_from_env_used_when_set(self, monkeypatch):
        class _FakeCfg:
            enabled = True
            url = "https://real.example.com/wp-json/wedatatools/v1/get-jobs-geojson"
        monkeypatch.setattr(
            "skillbridge.ingest.partners._config",
            lambda name: _FakeCfg() if name == "awic_jobs" else None,
        )
        seen_url = {"value": None}
        def _spy(url):
            seen_url["value"] = url
            return iter([])
        monkeypatch.setattr(
            "skillbridge.ingest.partners._fetch_awic_geojson", _spy,
        )
        _ = list(AWICJobsConnector().fetch())
        assert seen_url["value"] == _FakeCfg.url

    def test_source_name_is_awic_jobs(self):
        """source_name is what goes into core.job_posting.source. It
        MUST be 'awic_jobs', NOT 'awic' (which is the reports connector)."""
        assert AWICJobsConnector.source_name == "awic_jobs"


# ---------------------------------------------------------------- orchestrator wiring


class TestOrchestratorIntegration:
    """AWICJobsConnector wired into step_ingest_jobs end-to-end.

    DB-free: write_raw_job, upsert_job, sweep_missing_jobs are stubbed
    at the point where the orchestrator imports them. HTTP-free: the
    fetch client uses _MockTransport serving the fixture.

    Step 1A (2026-07-15) contract: the connector no longer applies
    an SSM-bbox ingestion filter. All four fixture features reach
    the persistence boundary; their location-resolution status
    (unresolved / missing / invalid) is set at normalize time and
    v_current_job's SSM-eligibility clauses determine which reach
    the live matcher.

    This is the load-bearing wiring test for Step 3: it fails if
    AWICJobsConnector is dropped from ALL_CONNECTORS, or if the
    orchestrator ever stops calling write_raw_job / upsert_job /
    sweep_missing_jobs for a partner connector.
    """

    def test_step_ingest_jobs_upserts_all_fixture_features(
        self, monkeypatch, fixture_payload,
    ):
        # -- HTTP boundary: fixture served via MockTransport.
        transport = _MockTransport(200, fixture_payload)
        original_client = httpx.Client

        def _client_with_mock(*args, **kwargs):
            kwargs["transport"] = transport
            return original_client(*args, **kwargs)

        monkeypatch.setattr(
            "skillbridge.ingest.partners.httpx.Client", _client_with_mock,
        )

        # -- Force the AWIC config to enabled=True so the test is
        #    hermetic against .env pollution (e.g. someone setting
        #    AWIC_JOBS_ENABLED=false locally). Uses an obviously fake
        #    URL; the MockTransport intercepts regardless of URL value.
        class _FakeAwicCfg:
            enabled = True
            url = "https://mock.invalid/awic-geojson"

        monkeypatch.setattr(
            "skillbridge.ingest.partners._config",
            lambda name: _FakeAwicCfg() if name == "awic_jobs" else None,
        )

        # -- Scope ALL_PARTNER_CONNECTORS to only AWIC. The other three
        #    partner connectors are irrelevant to what Step 3 wires up.
        from skillbridge.ingest.partners import AWICJobsConnector as _AWIC
        monkeypatch.setattr(
            "skillbridge.pipeline.orchestrator.ALL_PARTNER_CONNECTORS",
            [_AWIC],
        )

        # -- DB boundary: record every write / upsert / sweep call.
        raw_calls: list = []
        upsert_calls: list = []
        sweep_calls: list = []

        monkeypatch.setattr(
            "skillbridge.pipeline.orchestrator.write_raw_job",
            lambda job: raw_calls.append(job),
        )
        # upsert_job returns truthy iff a row was inserted/updated. Our
        # stub always returns True so the orchestrator counts every
        # yielded job as upserted -- matching what a healthy DB does
        # for a fresh row.
        monkeypatch.setattr(
            "skillbridge.pipeline.orchestrator.upsert_job",
            lambda job: (upsert_calls.append(job), True)[1],
        )
        # sweep_missing_jobs returns the number of rows deactivated.
        # Return 0 (nothing to deactivate) so the orchestrator's
        # per-source count reflects "clean upsert-only" behavior.
        monkeypatch.setattr(
            "skillbridge.pipeline.orchestrator.sweep_missing_jobs",
            lambda source, seen_ids: (
                sweep_calls.append((source, seen_ids)), 0,
            )[1],
        )

        from skillbridge.pipeline.orchestrator import step_ingest_jobs
        result = step_ingest_jobs()

        # Step 1A (2026-07-15): all 4 features ingest post-Step-1A.
        # Coordinate-based drop paths (no_coords, outside_ssm)
        # removed per spec §2f — geometry is not authoritative for
        # job location. Rows stay outside v_current_job via the
        # view's SSM eligibility clauses; ingestion is honest about
        # what it saw.
        assert len(raw_calls) == 4, (
            f"expected write_raw_job called 4 times (all features "
            f"ingested under Step 1A); got {len(raw_calls)}"
        )
        assert len(upsert_calls) == 4, (
            f"expected upsert_job called 4 times; got {len(upsert_calls)}"
        )

        # Every job carries source='awic_jobs'.
        assert all(j.source == "awic_jobs" for j in raw_calls)
        assert all(j.source == "awic_jobs" for j in upsert_calls)

        # All 4 IDs reach the boundary now (was 2 pre-Step-1A).
        upserted_ids = {j.source_job_id for j in upsert_calls}
        assert "8301829" in upserted_ids
        assert "8705234" in upserted_ids
        assert "8827928" in upserted_ids   # outside SSM bbox — now ingested
        assert "99999901" in upserted_ids  # missing coordinates — now ingested

        raw_ids = {j.source_job_id for j in raw_calls}
        assert raw_ids == upserted_ids

        # sweep_missing_jobs called once with all 4 seen IDs.
        assert len(sweep_calls) == 1
        source_arg, seen_ids_arg = sweep_calls[0]
        assert source_arg == "awic_jobs"
        assert seen_ids_arg == upserted_ids

        # -- Orchestrator returns per-source counts keyed by source_name.
        assert "awic_jobs" in result
        # Step 1A: all 4 features ingest (was 2 pre-Step-1A).
        assert result["awic_jobs"]["upserted"] == 4
        assert result["awic_jobs"]["deactivated"] == 0
