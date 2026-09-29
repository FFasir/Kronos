from fastapi.testclient import TestClient

from src.service_layer.web_server import app


def test_stock_detail_reads_local_archive_without_mock_data():
    client = TestClient(app)
    response = client.get("/api/v1/index/000001?days=10")
    assert response.status_code == 200
    body = response.json()
    assert body["type"] == "stock"
    assert len(body["kline"]) == 10


def test_missing_prediction_returns_explicit_unavailable_status():
    client = TestClient(app)
    response = client.get("/api/v1/prediction/latest?symbol=not-a-symbol")
    assert response.status_code in {404, 503}


def test_a_share_catalog_supports_server_side_pagination():
    client = TestClient(app)
    first = client.get("/api/v1/index/list?type=stock&limit=2&offset=0")
    second = client.get("/api/v1/index/list?type=stock&limit=2&offset=2")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["total"] > 5000
    assert first.json()["indices"][0]["code"] != second.json()["indices"][0]["code"]


def test_stock_detail_default_covers_about_five_years():
    client = TestClient(app)
    response = client.get("/api/v1/index/000001")

    assert response.status_code == 200
    assert len(response.json()["kline"]) == 1250
