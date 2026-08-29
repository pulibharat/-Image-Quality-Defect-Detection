import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _png_bytes(size=200, color=(120, 140, 160)) -> bytes:
    rng = np.random.default_rng(0)
    arr = np.full((size, size, 3), color, dtype=np.uint8)
    arr = arr + rng.integers(-15, 15, arr.shape, dtype=np.int16).clip(-255, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("ok", "degraded")
    assert "model_loaded" in body


def test_analyze_valid_image_returns_structured_result(client):
    files = {"file": ("test.png", _png_bytes(), "image/png")}
    resp = client.post("/api/analyze", files=files)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert "quality_score" in body
    assert 0 <= body["quality_score"] <= 100
    assert body["quality_label"] in ("ACCEPTABLE", "DEGRADED", "DEFECTIVE")
    assert isinstance(body["issues"], list)
    for issue in body["issues"]:
        assert issue["type"] in (
            "blur", "underexposure", "overexposure", "noise", "corruption", "defect",
        )
        assert issue["severity"] in ("low", "medium", "high")
        assert 0 <= issue["confidence"] <= 1
    assert "features" in body and "sharpness_lap_var" in body["features"]
    assert body["image_url"].startswith("/api/analyses/")


def test_analyze_rejects_non_image_file(client):
    files = {"file": ("not_an_image.txt", b"this is definitely not an image", "text/plain")}
    resp = client.post("/api/analyze", files=files)
    assert resp.status_code == 400
    assert "detail" in resp.json()


def test_analyze_rejects_empty_file(client):
    files = {"file": ("empty.png", b"", "image/png")}
    resp = client.post("/api/analyze", files=files)
    assert resp.status_code == 400


def test_history_list_and_detail_roundtrip(client):
    files = {"file": ("history_test.png", _png_bytes(color=(30, 30, 30)), "image/png")}
    created = client.post("/api/analyze", files=files).json()

    listed = client.get("/api/analyses?limit=5")
    assert listed.status_code == 200
    body = listed.json()
    assert body["total"] >= 1
    assert any(item["id"] == created["id"] for item in body["items"])

    detail = client.get(f"/api/analyses/{created['id']}")
    assert detail.status_code == 200
    assert detail.json()["id"] == created["id"]

    image_resp = client.get(f"/api/analyses/{created['id']}/image")
    assert image_resp.status_code == 200


def test_get_nonexistent_analysis_returns_404(client):
    resp = client.get("/api/analyses/does-not-exist")
    assert resp.status_code == 404
