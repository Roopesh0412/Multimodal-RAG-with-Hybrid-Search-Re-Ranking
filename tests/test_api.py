"""
Integration tests for FastAPI endpoints using TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "version" in data


def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "documents_indexed" in data
    assert "total_chunks" in data
    assert "embedding_model" in data


def test_query_validation(client):
    # Empty question should fail validation (min_length=2)
    response = client.post("/api/v1/query", json={"question": ""})
    assert response.status_code == 422


def test_upload_invalid_extension(client):
    files = {"file": ("document.txt", b"plain text content", "text/plain")}
    response = client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]
