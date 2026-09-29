from src.service_layer.api_backtest import _load_frozen_index_predictions


def test_frozen_index_predictions_are_base_model_and_execution_symbol():
    frame = _load_frozen_index_predictions("2025-02-01", "2025-02-28")
    assert not frame.empty
    assert set(frame["symbol"]) == {"512100"}
    assert frame["date"].min().strftime("%Y-%m-%d") >= "2025-02-01"
    assert frame["pred_return"].notna().all()
