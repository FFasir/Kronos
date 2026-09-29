import pandas as pd
from fastapi.testclient import TestClient

from src.service_layer import api_signal
from src.service_layer import api_prediction
from src.service_layer.models import SignalResult
from src.service_layer.prediction_service import PredictionService
from src.service_layer.web_server import app


def test_latest_signal_prefers_persisted_batch(monkeypatch):
    persisted = SignalResult(
        symbol="000852",
        name="中证1000",
        date="2026-07-15",
        signal_type="BUY",
        pred_return=0.012,
        pred_close=7412.5,
        confidence=0.72,
        position_pct=1.0,
        reason="persisted",
    )
    monkeypatch.setattr(api_signal, "_try_get_real_signals", lambda: [persisted])

    def fail_live_inference():
        raise AssertionError("live inference must not run when persisted signals exist")

    monkeypatch.setattr(api_signal, "_try_model_signals", fail_live_inference)
    response = TestClient(app).get("/api/v1/signal/latest")

    assert response.status_code == 200
    assert response.json()["date"] == "2026-07-15"
    assert response.json()["signals"][0]["position_pct"] == 1.0


def test_prediction_service_uses_active_model_factory(monkeypatch):
    frame = pd.DataFrame(
        {
            "open": [100.0] * 250,
            "high": [101.0] * 250,
            "low": [99.0] * 250,
            "close": [100.0] * 250,
            "volume": [1_000_000.0] * 250,
            "amount": [100_000_000.0] * 250,
        },
        index=pd.bdate_range("2025-07-31", periods=250),
    )
    calls = {"factory": 0}

    def fake_create_model(config):
        calls["factory"] += 1
        return object(), "kronos"

    class FakePredictor:
        def __init__(self, model, config):
            pass

        def predict_batch(self, frames):
            assert "000852" in frames
            return [{
                "symbol": "000852",
                "pred_date": "2026-07-22",
                "as_of_date": "2026-07-15",
                "pred_close_t5": 101.0,
                "pred_return": 0.01,
                "last_close": 100.0,
                "confidence": 0.7,
            }]

    class FakeStorage:
        def __init__(self, config):
            pass

        def load_raw_data(self, code):
            if code != "000852":
                raise FileNotFoundError(code)
            return frame

    monkeypatch.setattr("src.model_layer.model_factory.create_model", fake_create_model)
    monkeypatch.setattr("src.model_layer.kronos_predictor.KronosPredictor", FakePredictor)
    monkeypatch.setattr("src.data_layer.storage.StorageLayer", FakeStorage)

    predictions = PredictionService().get_latest_predictions("000852")

    assert calls["factory"] == 1
    assert predictions is not None
    assert predictions[0]["name"] == "中证1000"


def test_unconfigured_stock_does_not_load_live_model(monkeypatch):
    def fail_create_model(config):
        raise AssertionError("an arbitrary stock must use persisted batch predictions")

    monkeypatch.setattr("src.model_layer.model_factory.create_model", fail_create_model)

    assert PredictionService().get_latest_predictions("000001") is None


def test_latest_prediction_for_symbol_queries_only_one_row(monkeypatch):
    class FakeStorage:
        def __init__(self, config):
            pass

        def execute_sql(self, query, params=()):
            assert "LIMIT 1" in query
            assert params == ("000852",)
            return [[
                "000852", "2026-07-22", 7995.03, 0.0227, 7817.83,
                0.987, 0.6, 0.9, 250, 7.27, "中证1000",
            ]]

    monkeypatch.setattr("src.data_layer.storage.StorageLayer", FakeStorage)

    result = api_prediction._try_get_real_predictions("000852")

    assert result is not None
    assert len(result) == 1
