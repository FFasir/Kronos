from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.service_layer.api_admin import _valid_tokens, router


def make_client() -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return TestClient(app)


def test_admin_endpoints_require_login(monkeypatch):
    monkeypatch.setenv("KRONOS_ADMIN_USERNAME", "test-admin")
    monkeypatch.setenv("KRONOS_ADMIN_PASSWORD", "test-password")
    _valid_tokens.clear()
    client = make_client()

    response = client.get("/api/v1/admin/config")

    assert response.status_code == 401


def test_admin_login_and_logout_lifecycle(monkeypatch):
    monkeypatch.setenv("KRONOS_ADMIN_USERNAME", "test-admin")
    monkeypatch.setenv("KRONOS_ADMIN_PASSWORD", "test-password")
    _valid_tokens.clear()
    client = make_client()

    rejected = client.post(
        "/api/v1/admin/login",
        json={"username": "test-admin", "password": "wrong-password"},
    )
    assert rejected.status_code == 200
    assert rejected.json()["success"] is False

    accepted = client.post(
        "/api/v1/admin/login",
        json={"username": "test-admin", "password": "test-password"},
    )
    assert accepted.status_code == 200
    token = accepted.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    authorized = client.get("/api/v1/admin/config", headers=headers)
    assert authorized.status_code == 200

    logged_out = client.post("/api/v1/admin/logout", headers=headers)
    assert logged_out.status_code == 200

    expired = client.get("/api/v1/admin/config", headers=headers)
    assert expired.status_code == 401
