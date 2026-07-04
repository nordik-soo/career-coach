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
        ft = _feature_by_case(fixture_payload, "real_ssm_valid_noc")
        job, drop_reason, noc_provided = _normalize_awic_geojson_feature(ft)

        assert drop_reason is None
        assert noc_provided is True
        assert job is not None
        assert job.source == "awic_jobs"          # locked contract
        assert job.source_job_id == "8301829"     # str(post_id)
        assert job.noc_code == "31203"            # passed through
        assert job.location == "Sault Ste. Marie"  # canonical post-filter
        assert job.title  # non-empty
        assert job.employer == "Sault Area Hospital Foundation"
        assert job.url and job.url.startswith("https://")
        assert job.description  # excerpt populated
        assert job.raw_payload["source"] == "awic_geojson_v1"
        assert job.raw_payload["feature"] is ft   # full audit copy

    def test_real_outside_ssm_dropped_by_bbox(self, fixture_payload):
        ft = _feature_by_case(fixture_payload, "real_outside_ssm")
        job, drop_reason, noc_provided = _normalize_awic_geojson_feature(ft)
        assert job is None
        assert drop_reason == "outside_ssm"
        assert noc_provided is False

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
        # Still SSM-canonical + source-stamped correctly.
        assert job.source == "awic_jobs"
        assert job.location == "Sault Ste. Marie"

    def test_synth_missing_coords_dropped_by_coords(self, fixture_payload):
        ft = _feature_by_case(fixture_payload, "synth_missing_coords")
        job, drop_reason, noc_provided = _normalize_awic_geojson_feature(ft)
        assert job is None
        assert drop_reason == "no_coords"
        assert noc_provided is False

    def test_malformed_feature_dropped(self):
        """Not from the fixture (fixture only has valid GeoJSON shapes).
        These synthetic cases pin the defensive drop-reason wiring."""
        # Not a dict
        j, r, n = _normalize_awic_geojson_feature("not a feature")
        assert j is None and r == "malformed" and n is False
        # Missing geometry entirely => no_coords (geometry key defaults to {})
        j, r, n = _normalize_awic_geojson_feature({"properties": {}})
        assert j is None and r == "no_coords"
        # Coords present + in-bbox, but missing post_id/title
        good_geom = {"type": "Point", "coordinates": [-84.32, 46.54]}
        j, r, n = _normalize_awic_geojson_feature({
            "geometry": good_geom, "properties": {"post_id": 1},
        })
        assert j is None and r == "malformed" and n is False
        j, r, n = _normalize_awic_geojson_feature({
            "geometry": good_geom, "properties": {"job_title": "x"},
        })
        assert j is None and r == "malformed" and n is False


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

        # 2 yielded (real_ssm_valid_noc + real_ssm_invalid_noc).
        assert len(jobs) == 2
        assert all(j.source == "awic_jobs" for j in jobs)
        yielded_ids = {j.source_job_id for j in jobs}
        assert yielded_ids == {"8301829", "8705234"}

        # Verify User-Agent was sent on the request.
        assert len(transport.calls) == 1
        assert transport.calls[0].headers.get("user-agent") == AWIC_JOBS_USER_AGENT

        # Counter emission: last INFO record from partners logger.
        counter_records = [
            r for r in caplog.records
            if r.levelno == logging.INFO
            and "AWIC jobs: fetched=" in r.getMessage()
        ]
        assert len(counter_records) == 1
        msg = counter_records[0].getMessage()
        assert "fetched=4" in msg
        assert "after_ssm_filter=2" in msg
        assert "dropped_outside_ssm=1" in msg
        assert "dropped_no_coords=1" in msg
        assert "with_noc_provided=1" in msg
        assert "needing_noc_backfill=1" in msg

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
