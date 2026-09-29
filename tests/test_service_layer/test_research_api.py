from fastapi.testclient import TestClient

import time

from src.service_layer.api_admin import _valid_tokens
from src.service_layer.web_server import app


def test_research_summary_uses_frozen_artifacts_and_selects_base():
    _valid_tokens["research-test"] = time.time() + 60
    response = TestClient(app).get(
        "/api/v1/admin/research/summary",
        headers={"Authorization": "Bearer research-test"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["production_model"]["key"] == "kronos_base"
    assert len(body["comparison"]) == 4
    assert len(body["selected_holdout_equity"]) == 176

    selected = next(item for item in body["comparison"] if item["selected"])
    assert selected["holdout"]["net_compounded_return"] > 0.25
    assert selected["holdout"]["total_relative_excess_return"] > 0
    assert selected["holdout"]["max_drawdown"] > -0.05
