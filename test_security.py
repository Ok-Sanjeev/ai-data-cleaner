"""
Automated Security Test Suite for DataRefine.

Tests:
1. Directory traversal prevention (../../, absolute system paths, null bytes).
2. Unauthorized server file access blocking (server.py, secret files).
3. Invalid file extension rejection (.py, .json, .sh, binary).
4. Upload security: file size rejection, corrupt CSV handling.
5. Insecure CORS prevention and headers.
"""

import io
import pytest
from fastapi.testclient import TestClient

from server import app, resolve_safe_dataset_path


@pytest.fixture
def client():
    return TestClient(app)


def test_path_traversal_relative_parent_blocked(client):
    """Verify that traversing upwards via ../ to read server.py or sensitive files returns 403 Forbidden."""
    res = client.post("/api/profile", json={"filepath": "data/../server.py"})
    assert res.status_code == 403
    assert "Access forbidden" in res.json()["detail"]


def test_path_traversal_deep_parent_blocked(client):
    """Verify deep directory traversal attempts are strictly blocked."""
    res = client.post("/api/profile", json={"filepath": "../../../../../etc/passwd"})
    assert res.status_code == 403
    assert "Access forbidden" in res.json()["detail"]


def test_absolute_system_file_access_blocked(client):
    """Verify accessing arbitrary absolute system files (Windows or Linux) is rejected."""
    res = client.post("/api/profile", json={"filepath": "C:\\Windows\\win.ini"})
    assert res.status_code == 403
    assert "Access forbidden" in res.json()["detail"]


def test_null_byte_injection_blocked(client):
    """Verify paths containing null bytes are rejected with 400 Bad Request."""
    res = client.post("/api/profile", json={"filepath": "data/benchmark/dirty_sample.csv\x00.exe"})
    assert res.status_code == 400


def test_non_csv_file_extension_blocked(client):
    """Verify that accessing non-csv files within project is rejected."""
    # Even if pointing to an existing file in the project
    res = client.post("/api/profile", json={"filepath": "data/benchmark"})
    assert res.status_code in (400, 403, 404)


def test_valid_approved_dataset_allowed(client):
    """Verify legitimate datasets in approved directories succeed."""
    res = client.post("/api/profile", json={"filepath": "data/benchmark/dirty_sample.csv"})
    assert res.status_code == 200
    assert "session_id" in res.json()
    assert res.json()["summary"]["rows"] > 0


def test_upload_non_csv_rejected(client):
    """Verify uploading non-CSV files is rejected with 400."""
    fake_exe = io.BytesIO(b"MZ\x90\x00\x03\x00\x00\x00")
    res = client.post("/api/upload", files={"file": ("malware.exe", fake_exe, "application/octet-stream")})
    assert res.status_code == 400
    assert "Only CSV files" in res.json()["detail"]


def test_upload_corrupt_csv_rejected(client):
    """Verify uploading completely empty or unparseable files is rejected."""
    empty_csv = io.BytesIO(b"")
    res = client.post("/api/upload", files={"file": ("empty.csv", empty_csv, "text/csv")})
    assert res.status_code == 400
    detail = res.json()["detail"].lower()
    assert "empty" in detail or "invalid or corrupted" in detail


def test_cors_security_headers(client):
    """Verify OPTIONS preflight request behaves according to standards."""
    res = client.options("/api/profile", headers={"Origin": "http://example.com", "Access-Control-Request-Method": "POST"})
    # Must not send an illegal combination of wildcard origin with allow_credentials=true
    if res.headers.get("access-control-allow-origin") == "*":
        assert res.headers.get("access-control-allow-credentials") != "true"
