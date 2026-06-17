"""Route guard — durable invariant that every route is either public,
profile-authenticated, or admin-authenticated.

This is the third durable invariant test (alongside consent-boundary and
delete-cascade). If anyone adds a new private route in the future and
forgets the auth dependency, this test fails first.

The test enumerates every route registered on the FastAPI app and asserts:
- Public routes are reachable without auth.
- Profile routes return 401 without a valid bearer token.
- Admin routes return 401 without a valid admin token.
- Profile tokens cannot access admin routes.
- Admin tokens are NOT accepted as profile tokens.
"""
from __future__ import annotations

import os

import api as api_module


# Routes that intentionally do not require authentication. Add new routes
# here only when explicitly designed as public.
PUBLIC_ENDPOINTS: set[tuple[str, str]] = {
    ("GET",  "/"),
    ("GET",  "/health/live"),
    ("GET",  "/health/ready"),
    ("GET",  "/v1/meta/version"),
    ("GET",  "/v1/jobs"),
    ("GET",  "/v1/jobs/{job_id}"),
    ("GET",  "/v1/training-resources"),
    ("GET",  "/v1/training-resources/{resource_id}"),
    ("POST", "/v1/chat/messages"),
    ("POST", "/v1/consent"),
    # FastAPI auto-generated docs (acceptable as public for MVP).
    ("GET",  "/docs"),
    ("GET",  "/redoc"),
    ("GET",  "/docs/oauth2-redirect"),
    ("GET",  "/openapi.json"),
}

# Routes that must be reached with an admin bearer token.
ADMIN_ENDPOINTS: set[tuple[str, str]] = {
    ("GET",  "/v1/admin/data-status"),
    ("POST", "/v1/admin/pipeline/refresh"),
}

# Path-param placeholders to substitute when probing routes.
PARAM_PLACEHOLDER = "00000000-0000-0000-0000-000000000000"


def _enumerate_routes() -> list[tuple[str, str]]:
    """Return list of (method, path) for every API route on the app."""
    out: list[tuple[str, str]] = []
    for route in api_module.app.routes:
        path = getattr(route, "path", None)
        if path is None:
            continue
        methods = (getattr(route, "methods", None) or set()) - {"HEAD", "OPTIONS"}
        for m in methods:
            out.append((m, path))
    return out


def _substitute(path: str) -> str:
    """Replace {profile_id}, {job_id}, etc. with a placeholder UUID."""
    p = path
    for name in ("{profile_id}", "{job_id}", "{match_id}", "{resource_id}"):
        p = p.replace(name, PARAM_PLACEHOLDER)
    return p


# ====================================================================
# Test 1: Every private route returns 401 without auth.
# ====================================================================
def test_all_private_routes_require_auth(client):
    violations: list[tuple[str, str, int]] = []
    for method, path in _enumerate_routes():
        key = (method, path)
        if key in PUBLIC_ENDPOINTS:
            continue
        test_path = _substitute(path)
        resp = client.request(method, test_path)
        if resp.status_code != 401:
            violations.append((method, path, resp.status_code))
    assert not violations, (
        "Private routes that did NOT return 401 without auth:\n"
        + "\n".join(f"  {m} {p} -> {s}" for m, p, s in violations)
        + "\n\nIf the route is intentionally public, add it to "
          "PUBLIC_ENDPOINTS. Otherwise add a require_profile/require_admin "
          "dependency."
    )


# ====================================================================
# Test 2: Public routes are reachable without auth.
# ====================================================================
def test_public_routes_do_not_require_auth(client):
    """GETs should reach a non-401 status. POSTs may 422 (missing body)
    or 200; either is acceptable — the point is auth doesn't gate them."""
    for method, path in PUBLIC_ENDPOINTS:
        if method != "GET":
            continue
        resp = client.request(method, _substitute(path))
        assert resp.status_code != 401, (
            f"Public {method} {path} returned 401 — make sure it has no "
            f"require_profile/require_admin dependency."
        )


# ====================================================================
# Test 3: Profile tokens cannot access admin endpoints.
# ====================================================================
def test_profile_token_cannot_access_admin(client):
    r1 = client.post("/v1/chat/messages", json={"message": "I have customer service skills."})
    sid = r1.json()["data"]["session_id"]
    r2 = client.post(
        "/v1/consent",
        json={"session_id": sid, "consent_purposes": ["profile_storage"]},
    )
    profile_token = r2.json()["data"]["session_token"]

    for method, path in ADMIN_ENDPOINTS:
        resp = client.request(
            method, _substitute(path),
            headers={"Authorization": f"Bearer {profile_token}"},
        )
        assert resp.status_code == 401, (
            f"Profile token leaked admin access on {method} {path} -> {resp.status_code}"
        )


# ====================================================================
# Test 4: Admin tokens cannot access profile endpoints.
# ====================================================================
def test_admin_token_cannot_access_profile_routes(client):
    admin_key = os.environ["ADMIN_API_KEY"]
    headers = {"Authorization": f"Bearer admin:{admin_key}"}
    private_profile_paths = [
        ("GET",    "/v1/profiles/me"),
        ("GET",    "/v1/profiles/me/skills"),
        ("GET",    "/v1/profiles/me/job-matches"),
        ("GET",    "/v1/profiles/me/training-recommendations"),
    ]
    for method, path in private_profile_paths:
        resp = client.request(method, path, headers=headers)
        assert resp.status_code == 401, (
            f"Admin token leaked profile access on {method} {path} -> {resp.status_code}"
        )


# ====================================================================
# Test 5: Valid admin token reaches admin endpoints.
# ====================================================================
def test_admin_token_reaches_admin_endpoints(client):
    admin_key = os.environ["ADMIN_API_KEY"]
    resp = client.get(
        "/v1/admin/data-status",
        headers={"Authorization": f"Bearer admin:{admin_key}"},
    )
    assert resp.status_code == 200, (
        f"Valid admin token rejected on /v1/admin/data-status -> {resp.status_code}"
    )


# ====================================================================
# Test 6: Valid profile token reaches /me routes.
# ====================================================================
def test_profile_token_reaches_me_routes(client):
    r1 = client.post(
        "/v1/chat/messages",
        json={"message": "I have customer service and cash handling skills."},
    )
    sid = r1.json()["data"]["session_id"]
    r2 = client.post(
        "/v1/consent",
        json={"session_id": sid, "consent_purposes": ["profile_storage"]},
    )
    token = r2.json()["data"]["session_token"]
    pid = r2.json()["data"]["profile_id"]

    r3 = client.get("/v1/profiles/me", headers={"Authorization": f"Bearer {token}"})
    assert r3.status_code == 200
    assert r3.json()["data"]["profile_id"] == pid

    r4 = client.get("/v1/profiles/me/skills", headers={"Authorization": f"Bearer {token}"})
    assert r4.status_code == 200


# ====================================================================
# Test 7: POST /v1/profiles is gone.
# ====================================================================
def test_post_profiles_endpoint_is_removed(client):
    """The chat → consent flow is the only path to a profile. The
    legacy POST /v1/profiles endpoint should no longer exist."""
    paths = {p for _, p in _enumerate_routes()}
    assert "/v1/profiles" not in paths, (
        "POST /v1/profiles must be removed (see PR 3 decisions)."
    )


# ====================================================================
# Test 8: Bad bearer formats are uniformly 401, never 500.
# ====================================================================
def test_malformed_bearer_returns_401(client):
    for bad in ["Bearer", "Bearer  ", "Token abc", "Basic abc", "Bearer admin:", "Bearer admin:wrong"]:
        resp = client.get("/v1/profiles/me", headers={"Authorization": bad})
        assert resp.status_code == 401, (
            f"Malformed Authorization '{bad}' returned {resp.status_code} (expected 401)"
        )
