import numpy as np
import pandas as pd

from src.model_layer.kronos_predictor import KronosPredictor


class FakeConfig:
    seed = 100

    def get(self, *keys, default=None):
        values = {
            ("model", "training", "seq_len"): 250,
            ("model", "training", "pred_len"): 5,
            ("model", "inference", "temperature"): 0.6,
            ("model", "inference", "top_p"): 0.9,
            ("model", "inference", "sample_count"): 3,
            ("model", "inference", "batch_size"): 2,
        }
        return values.get(tuple(keys), default)


class FakeNativeModel:
    is_native_kronos = True
    model_version = "test"

    def __init__(self):
        self.batch_sizes = []
        self.received_frames = []

    def predict_dataframes_with_uncertainty(self, frames, pred_len, T, top_p, sample_count):
        self.batch_sizes.append(len(frames))
        self.received_frames.extend(frame.copy() for frame in frames)
        forecasts = []
        forecast_stds = []
        for frame in frames:
            close = float(frame["close"].iloc[-1]) * 1.01
            forecasts.append(pd.DataFrame({"close": [close] * pred_len}))
            forecast_stds.append(pd.DataFrame({"close": [close * 0.01] * pred_len}))
        return forecasts, forecast_stds


def make_frame(offset: float) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", periods=260)
    close = np.linspace(10 + offset, 20 + offset, len(index))
    return pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.full(len(index), 1_000_000.0),
        },
        index=index,
    )


def test_native_predictions_are_vectorized_and_keep_order():
    model = FakeNativeModel()
    predictor = KronosPredictor(model, FakeConfig())
    frames = {"000001": make_frame(0), "000002": make_frame(1), "000003": make_frame(2)}

    results = predictor.predict_batch(frames)

    assert model.batch_sizes == [2, 1]
    assert [result["symbol"] for result in results] == list(frames)
    assert all(abs(result["pred_return"] - 0.01) < 1e-9 for result in results)
    assert all(result["temperature"] == 0.6 for result in results)
    assert all(result["top_p"] == 0.9 for result in results)
    assert all(0.5 < result["confidence"] < 1.0 for result in results)
    assert all(abs(result["score"] - 6.0) < 1e-9 for result in results)


def test_native_inference_passes_amount_and_backfills_missing_values():
    model = FakeNativeModel()
    predictor = KronosPredictor(model, FakeConfig())
    frames = {
        "with_amount": make_frame(0).assign(amount=123456.0),
        "without_amount": make_frame(1),
    }

    predictor.predict_batch(frames, lookback=5, sample_count=2, batch_size=2)

    assert model.received_frames[0]["amount"].eq(123456.0).all()
    assert model.received_frames[1]["amount"].equals(
        model.received_frames[1]["close"] * model.received_frames[1]["volume"]
    )
