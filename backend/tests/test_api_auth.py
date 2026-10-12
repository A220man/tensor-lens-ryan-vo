"""Tests for authentication and role-based authorization."""
import pytest
from fastapi import HTTPException
from app.core.config import Settings
from app.core.auth import create_demo_token


def test_unauthenticated_request_rejected(client):
    """Verify that requests without authorization header return 401 Unauthorized."""
    resp = client.get("/api/models")
    assert resp.status_code == 401
    assert "Missing authorization credentials" in resp.json()["detail"]


def test_demo_token_generation_and_userinfo(client):
    """Verify demo token generation and authentication via /api/auth/userinfo."""
    token_resp = client.post("/api/auth/demo-token", json={"role": "analyst"})
    assert token_resp.status_code == 200
    token_data = token_resp.json()
    assert "access_token" in token_data
    assert token_data["role"] == "analyst"

    auth_headers = {"Authorization": f"Bearer {token_data['access_token']}"}
    user_resp = client.get("/api/auth/userinfo", headers=auth_headers)
    assert user_resp.status_code == 200
    user_data = user_resp.json()
    assert user_data["roles"] == ["analyst"]
    assert user_data["is_demo"] is True


def test_role_enforcement_viewer_forbidden_from_creating_model(client, viewer_headers):
    """Verify that Viewer role receives 403 Forbidden on analyst/admin mutation endpoints."""
    payload = {
        "name": "Unauthorized Model",
        "family": "llama",
        "parameter_count_billion": 7.0,
        "hidden_dim": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "vocab_size": 32000,
        "max_seq_length": 2048,
        "intermediate_dim": 11008,
    }
    resp = client.post("/api/models", json=payload, headers=viewer_headers)
    assert resp.status_code == 403
    assert "requires 'analyst' permissions" in resp.json()["detail"]


def test_admin_role_can_access_audit_logs(client, admin_headers, viewer_headers):
    """Verify that Admin can access audit logs while Viewer receives 403 Forbidden."""
    admin_resp = client.get("/api/audit-logs", headers=admin_headers)
    assert admin_resp.status_code == 200
    assert isinstance(admin_resp.json(), list)

    viewer_resp = client.get("/api/audit-logs", headers=viewer_headers)
    assert viewer_resp.status_code == 403
    assert "requires 'admin' permissions" in viewer_resp.json()["detail"]


def test_demo_mode_refusal_in_production(monkeypatch):
    """Verify that demo token generation raises 403 Forbidden when ENVIRONMENT=production."""
    from app.core import config
    monkeypatch.setattr(config.settings, "ENVIRONMENT", "production")

    with pytest.raises(HTTPException) as exc_info:
        create_demo_token(role="analyst")
    assert exc_info.value.status_code == 403
    assert "refused in production environment" in exc_info.value.detail
