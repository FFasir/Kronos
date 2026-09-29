"""Live predictions backed by the model selected in the admin registry."""

from __future__ import annotations

from typing import Optional

from src.common.config import load_config
from src.common.logger import get_logger


logger = get_logger(__name__)

_service: Optional["PredictionService"] = None


class PredictionService:
    """Lazily load the active Kronos model for development-time inference."""

    def __init__(self):
        self.config = None
        self.predictor = None
        self.model_path: Optional[str] = None

    def _ensure_loaded(self) -> None:
        if self.predictor is not None:
            return

        from src.model_layer.kronos_predictor import KronosPredictor
        from src.model_layer.model_factory import create_model

        self.config = load_config()
        model, self.model_path = create_model(self.config)
        self.predictor = KronosPredictor(model, self.config)
        logger.info("PredictionService loaded active model path=%s", self.model_path)

    def get_latest_predictions(self, symbol: Optional[str] = None) -> Optional[list[dict]]:
        """Generate predictions from real local OHLCV using the active model.

        Any symbol backed by local OHLCV data (the 12 configured market assets,
        plus individual stocks served from the daily_hfq archive) can be
        predicted.  A scale-consistency guard in StorageLayer.load_raw_data
        keeps adjusted historical prices from being polluted by unadjusted
        daily snapshots, so per-stock predictions stay on a sane price scale.
        """
        config = self.config or load_config()
        if symbol:
            # Map to the configured asset or resolve arbitrary stock codes.
            frames, names = {}, {}
            try:
                from src.data_layer.storage import StorageLayer
                storage = StorageLayer(config)
                # 非配置资产(个股)：若本地后复权覆盖层滞后于最新交易日，按日刷新一次。
                # 配置指数/ETF 有自己的行情更新任务，不走个股覆盖层。
                configured_codes = {
                    str(item["code"]) for item in config.get_all_symbols()
                }
                if symbol not in configured_codes:
                    try:
                        from src.data_layer.stock_hfq_live import ensure_current
                        ensure_current(symbol, config=config)
                    except Exception:
                        pass
                frame = storage.load_raw_data(symbol)
                frames[symbol] = frame
                row = storage.execute_sql(
                    "SELECT name FROM stock_info WHERE code = ? LIMIT 1", (symbol,)
                )
                names[symbol] = str(row[0][0]) if row else symbol
            except Exception as exc:
                logger.info("skip live prediction %s: %s", symbol, exc)
                return None
            self._ensure_loaded()
        else:
            configured = config.get_all_symbols()
            if not configured:
                return None
            self._ensure_loaded()
            if self.predictor is None or self.config is None:
                return None
            from src.data_layer.storage import StorageLayer

            storage = StorageLayer(self.config)
            frames = {}
            names = {}
            for item in configured:
                code = str(item["code"])
                try:
                    frame = storage.load_raw_data(code)
                except (FileNotFoundError, ValueError) as exc:
                    logger.info("Skipping %s live prediction: %s", code, exc)
                    continue
                frames[code] = frame
                names[code] = str(item.get("name") or code)
            if not frames:
                return None

        predictions = self.predictor.predict_batch(frames)
        successful = []
        for prediction in predictions:
            if prediction.get("error") or not prediction.get("pred_date"):
                continue
            prediction["name"] = names.get(prediction["symbol"], prediction["symbol"])
            successful.append(prediction)
        return successful or None

    def get_latest_signals(self) -> Optional[list[dict]]:
        """Generate paper-strategy signals from active-model predictions."""
        predictions = self.get_latest_predictions()
        if not predictions or self.config is None:
            return None

        from src.strategy_layer.signal_generator import SignalGenerator

        signal_date = max(
            str(item.get("as_of_date") or item.get("pred_date"))
            for item in predictions
        )
        signals = SignalGenerator(self.config).generate(predictions, signal_date)
        names = {item["symbol"]: item.get("name") for item in predictions}
        for signal in signals:
            signal["name"] = names.get(signal["symbol"], signal["symbol"])
        return signals or None


def get_prediction_service() -> PredictionService:
    global _service
    if _service is None:
        _service = PredictionService()
    return _service


def reset_prediction_service() -> None:
    """Drop active-model references after an administrator switches models."""
    global _service
    _service = None
