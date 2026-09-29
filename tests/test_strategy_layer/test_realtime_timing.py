import pandas as pd

from src.strategy_layer import realtime_timing
from src.strategy_layer.realtime_timing import build_component_signals


def test_current_component_loader_requires_complete_official_snapshot(tmp_path, monkeypatch):
    research = tmp_path / "data" / "raw" / "research"
    research.mkdir(parents=True)
    pd.DataFrame({
        "code": [f"{value:06d}" for value in range(1, 1001)],
        "name": [f"stock-{value}" for value in range(1, 1001)],
        "snapshot_date": "2026-07-15",
    }).to_csv(research / "csi1000_csindex_20260715.csv", index=False)
    monkeypatch.setattr(realtime_timing, "PROJECT_ROOT", tmp_path)

    result = realtime_timing.load_current_csi1000_components()

    assert len(result) == 1000
    assert result["code"].nunique() == 1000


def test_component_signal_state_machine_sells_after_five_batches():
    predictions = pd.DataFrame([
        {
            "code": "000001", "date": "2026-07-15", "pred_close": 10.2,
            "pred_return": 0.02, "confidence": 0.7,
        },
        {
            "code": "000002", "date": "2026-07-15", "pred_close": 20.2,
            "pred_return": 0.01, "confidence": 0.6,
        },
    ])
    prior = pd.DataFrame([
        {"code": "000001", "date": "2026-07-08", "signal_type": "BUY", "position_pct": 1.0},
        {"code": "000001", "date": "2026-07-14", "signal_type": "HOLD", "position_pct": 1.0},
    ])

    signals = build_component_signals(
        predictions,
        prior,
        batch_dates=["2026-07-09", "2026-07-10", "2026-07-11", "2026-07-14", "2026-07-15"],
        buy_threshold=0.005,
        hold_days=5,
    )

    by_code = {item["symbol"]: item for item in signals}
    assert by_code["000001"]["signal_type"] == "SELL"
    assert by_code["000002"]["signal_type"] == "BUY"
    assert by_code["000002"]["position_pct"] == 1.0


def test_missing_prediction_materializes_explicit_cash_signal():
    predictions = pd.DataFrame([
        {
            "code": "000003", "date": "2026-07-15", "pred_close": None,
            "pred_return": None, "confidence": None,
        }
    ])

    signal = build_component_signals(
        predictions,
        pd.DataFrame(),
        batch_dates=["2026-07-15"],
        buy_threshold=0.005,
        hold_days=5,
    )[0]

    assert signal["signal_type"] == "CASH"
    assert signal["confidence"] == 0.0
    assert "预测缺失" in signal["reason"]
