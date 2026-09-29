"""
Kronos 指数/ETF 择时策略系统 — 模型预测推理

使用微调后的模型（Path A Kronos 或 Path B Fallback）进行价格预测推理。
"""

import json
import numpy as np
import pandas as pd
import torch
from statistics import NormalDist
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from src.common.logger import get_logger
from src.common.utils import set_seed, format_date

logger = get_logger(__name__)


class KronosPredictor:
    """
    统一预测推理器。

    支持 Path A (Kronos) 和 Path B (FallbackModel) 两种模型后端，
    对外暴露相同的预测接口。
    """

    def __init__(
        self,
        model,
        config,
        device: torch.device = None,
    ):
        """
        Args:
            model: KronosWrapper 或 FallbackModel 实例
            config: 系统配置
            device: 计算设备
        """
        self.model = model
        self.config = config

        from src.common.utils import detect_device
        self.device = device or detect_device(config.get("model", "device", default="auto"))

        # 推理参数默认值
        self.seq_len = config.get("model", "training", "seq_len", default=60)
        self.pred_len = config.get("model", "training", "pred_len", default=5)
        self.default_T = config.get("model", "inference", "temperature", default=0.6)
        self.default_top_p = config.get("model", "inference", "top_p", default=0.9)
        self.default_sample_count = config.get("model", "inference", "sample_count", default=1)
        self.default_batch_size = config.get("model", "inference", "batch_size", default=32)

        set_seed(config.seed)

    def predict_single(
        self,
        df: pd.DataFrame,
        lookback: int = None,
        T: float = None,
        top_p: float = None,
        sample_count: int = None,
    ) -> Dict[str, Any]:
        """
        单标的单次预测。

        Args:
            df: 含 OHLCV 列的 DataFrame (索引为日期，按时间升序)
            lookback: 回溯窗口天数 (None = 使用配置默认值)
            T: temperature (None = 默认)
            top_p: nucleus sampling top_p

        Returns:
            {
                "pred_close_t5": float,       # 预测未来第5日收盘价
                "pred_return": float,         # 预测5日收益率
                "last_close": float,          # 最新收盘价
                "pred_date": str,             # 被预测的日期
                "model_version": str,
            }
        """
        lookback = lookback or self.seq_len
        T = T or self.default_T
        top_p = top_p or self.default_top_p
        sample_count = sample_count or self.default_sample_count

        ohlcv_cols = ["open", "high", "low", "close", "volume"]

        # 确保有足够的回溯数据
        if len(df) < lookback:
            logger.warning(f"数据不足: 需要 {lookback} 天, 实际 {len(df)} 天")
            return self._empty_prediction()

        # 取最近 lookback 天的 OHLCV
        # 注意：fallback 模型在原始价格（未归一化）上训练，其 predict() 直接
        # 输出绝对收盘价，因此推理端必须喂入原始 OHLCV，与训练/prediction_service
        # 的口径保持一致。此前的 min-max 归一化会把输入缩放到 [0,1]，而输出仍被
        # 当作绝对价处理，导致预测无意义（见修复清单 #3）。
        recent = df[ohlcv_cols].iloc[-lookback:].values.astype(np.float32)

        # 推理
        try:
            if getattr(self.model, "is_native_kronos", False):
                native_frame = self._prepare_native_frame(df, lookback)
                forecasts, forecast_stds = self.model.predict_dataframes_with_uncertainty(
                    [native_frame], pred_len=self.pred_len, T=T,
                    top_p=top_p, sample_count=sample_count,
                )
                forecast = forecasts[0]
                pred_close = float(forecast["close"].iloc[-1])
                pred_std = float(forecast_stds[0]["close"].iloc[-1])
            else:
                X = torch.from_numpy(recent).unsqueeze(0).to(self.device)
                pred_close = self.model.predict(X, horizon=self.pred_len)
                pred_std = None
        except Exception as e:
            logger.error(f"推理失败: {e}")
            return self._empty_prediction()

        last_close = float(df["close"].iloc[-1])
        pred_return = (pred_close / last_close - 1.0) if last_close > 0 else 0.0
        confidence = self._estimate_prob_up(pred_close, last_close, pred_std)

        # 被预测的日期: 最后日期 + pred_len 个交易日
        last_date = pd.Timestamp(df.index[-1])
        pred_date = last_date + pd.offsets.BDay(self.pred_len)

        return {
            "pred_close_t5": pred_close,
            "pred_return": pred_return,
            "last_close": last_close,
            "pred_date": format_date(pred_date),
            "as_of_date": format_date(last_date),
            "confidence": confidence,
            "prob_up": confidence,
            "score": self._return_score(pred_return),
            "prediction_std": pred_std,
            "temperature": T,
            "top_p": top_p,
            "lookback_window": lookback,
        }

    def predict_batch(
        self,
        symbol_dfs: Dict[str, pd.DataFrame],
        lookback: int = None,
        T: float = None,
        top_p: float = None,
        sample_count: int = None,
        batch_size: int = None,
    ) -> List[Dict[str, Any]]:
        """
        批量预测多个标的。

        Args:
            symbol_dfs: {symbol: DataFrame} 映射
            lookback: 回溯窗口
            T: temperature
            top_p: top_p

        Returns:
            [{symbol, ...预测字段...}, ...]
        """
        lookback = lookback or self.seq_len
        T = T or self.default_T
        top_p = top_p or self.default_top_p
        sample_count = sample_count or self.default_sample_count
        batch_size = batch_size or self.default_batch_size

        if getattr(self.model, "is_native_kronos", False):
            return self._predict_native_batch(
                symbol_dfs,
                lookback=lookback,
                T=T,
                top_p=top_p,
                sample_count=sample_count,
                batch_size=batch_size,
            )

        results = []
        for symbol, df in symbol_dfs.items():
            try:
                pred = self.predict_single(
                    df, lookback=lookback, T=T, top_p=top_p, sample_count=sample_count
                )
                pred["symbol"] = symbol
                results.append(pred)
                logger.debug(f"  {symbol}: pred_return={pred['pred_return']:.4%}")
            except Exception as e:
                logger.error(f"预测 {symbol} 失败: {e}")
                results.append({
                    "symbol": symbol,
                    "pred_close_t5": 0.0,
                    "pred_return": 0.0,
                    "last_close": 0.0,
                    "pred_date": None,
                    "confidence": 0.0,
                    "error": str(e),
                })

        return results

    def _predict_native_batch(
        self,
        symbol_dfs: Dict[str, pd.DataFrame],
        lookback: int,
        T: float,
        top_p: float,
        sample_count: int,
        batch_size: int,
    ) -> List[Dict[str, Any]]:
        """Use the official vectorized predictor while preserving input order."""
        pending = []
        results: dict[str, Dict[str, Any]] = {}
        for symbol, df in symbol_dfs.items():
            if len(df) < lookback:
                result = self._empty_prediction()
                result["symbol"] = symbol
                results[symbol] = result
                continue
            try:
                frame = self._prepare_native_frame(df, lookback)
                if frame.isna().any().any() or (frame <= 0).any().any():
                    raise ValueError("OHLCVA contains missing or non-positive values")
                pending.append((symbol, df, frame))
            except Exception as exc:
                result = self._empty_prediction()
                result.update({"symbol": symbol, "error": str(exc)})
                results[symbol] = result

        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            try:
                forecasts, forecast_stds = self.model.predict_dataframes_with_uncertainty(
                    [item[2] for item in batch],
                    pred_len=self.pred_len,
                    T=T,
                    top_p=top_p,
                    sample_count=sample_count,
                )
                for (symbol, df, _), forecast, forecast_std in zip(
                    batch, forecasts, forecast_stds
                ):
                    pred_close = float(forecast["close"].iloc[-1])
                    if not np.isfinite(pred_close) or pred_close <= 0:
                        raise ValueError(f"invalid predicted close for {symbol}: {pred_close}")
                    last_close = float(df["close"].iloc[-1])
                    last_date = pd.Timestamp(df.index[-1])
                    pred_return = pred_close / last_close - 1
                    pred_std = float(forecast_std["close"].iloc[-1])
                    confidence = self._estimate_prob_up(pred_close, last_close, pred_std)
                    results[symbol] = {
                        "symbol": symbol,
                        "pred_close_t5": pred_close,
                        "pred_return": pred_return,
                        "last_close": last_close,
                        "pred_date": format_date(last_date + pd.offsets.BDay(self.pred_len)),
                        "as_of_date": format_date(last_date),
                        "confidence": confidence,
                        "prob_up": confidence,
                        "score": self._return_score(pred_return),
                        "prediction_std": pred_std,
                        "temperature": T,
                        "top_p": top_p,
                        "lookback_window": lookback,
                    }
            except Exception as exc:
                logger.warning("native batch inference failed; isolating %s symbols: %s", len(batch), exc)
                for symbol, df, _ in batch:
                    result = self.predict_single(
                        df,
                        lookback=lookback,
                        T=T,
                        top_p=top_p,
                        sample_count=sample_count,
                    )
                    result["symbol"] = symbol
                    results[symbol] = result
            logger.info("native inference progress: %s/%s", min(start + batch_size, len(pending)), len(pending))

        return [results[symbol] for symbol in symbol_dfs]

    def predict_from_storage(
        self,
        symbols: Optional[List[str]] = None,
        lookback: int = None,
        T: float = None,
        top_p: float = None,
        sample_count: int = None,
        batch_size: int = None,
    ) -> List[Dict[str, Any]]:
        """
        从 HDF5 存储加载数据并批量预测。

        Args:
            symbols: 标的代码列表 (None = 全部)
            lookback: 回溯窗口
            T: temperature
            top_p: top_p

        Returns:
            预测结果列表
        """
        from src.data_layer.storage import StorageLayer

        storage = StorageLayer(self.config)

        if symbols is None:
            symbols = storage.list_stock_symbols()

        symbol_dfs = {}
        for sym in symbols:
            try:
                df = storage.load_raw_data(sym)
                if df is not None and len(df) > 0:
                    symbol_dfs[sym] = df
            except Exception as e:
                logger.warning(f"加载 {sym} 数据失败: {e}")

        return self.predict_batch(
            symbol_dfs,
            lookback=lookback,
            T=T,
            top_p=top_p,
            sample_count=sample_count,
            batch_size=batch_size,
        )

    def save_predictions(
        self,
        predictions: List[Dict[str, Any]],
        output_dir: str = None,
    ) -> str:
        """
        保存预测结果到 JSON 文件。

        Args:
            predictions: 预测结果列表
            output_dir: 输出目录 (None = 使用配置)

        Returns:
            保存的文件路径
        """
        if output_dir is None:
            output_dir = self.config.get("data", "outputs_dir", default="data/outputs")
            output_dir = Path(output_dir) / "predictions" / datetime.now().strftime("%Y-%m-%d")
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        output = {
            "generated_at": datetime.now().isoformat(),
            "model_type": getattr(self.model, "model_version", "unknown"),
            "config": {
                "lookback": self.seq_len,
                "pred_len": self.pred_len,
                "T": self.default_T,
                "top_p": self.default_top_p,
                "sample_count": self.default_sample_count,
                "batch_size": self.default_batch_size,
            },
            "predictions": predictions,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info(f"预测结果已保存: {filepath}")
        self._persist_predictions(predictions)
        return str(filepath)

    def _persist_predictions(self, predictions: List[Dict[str, Any]]) -> None:
        """Persist successful predictions for selection, signals, and the UI."""
        from src.data_layer.storage import StorageLayer

        rows = []
        for item in predictions:
            if item.get("error") or not item.get("pred_date") or not item.get("symbol"):
                continue
            rows.append((
                item["symbol"],
                item.get("as_of_date") or datetime.now().strftime("%Y-%m-%d"),
                item["pred_date"],
                float(item["pred_close_t5"]),
                float(item.get("pred_return", 0)),
                float(item.get("last_close", 0)),
                float(item.get("prediction_std", 0)),
                float(item.get("score", self._return_score(float(item.get("pred_return", 0))))),
                float(item.get("confidence", 0)),
                float(item.get("temperature", self.default_T)),
                float(item.get("top_p", self.default_top_p)),
                int(item.get("lookback_window", self.seq_len)),
                str(getattr(self.model, "model_version", "unknown")),
            ))
        if not rows:
            logger.warning("no successful predictions to persist")
            return
        StorageLayer(self.config).execute_many(
            """INSERT INTO model_prediction
               (code, date, pred_date, pred_close, pred_return, last_close,
                prediction_std, score, confidence,
                temperature, top_p, lookback, model_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(code, date, pred_date) DO UPDATE SET
                 pred_close=excluded.pred_close, pred_return=excluded.pred_return,
                 last_close=excluded.last_close,
                 prediction_std=excluded.prediction_std,
                 score=excluded.score,
                 confidence=excluded.confidence, temperature=excluded.temperature,
                 top_p=excluded.top_p, lookback=excluded.lookback,
                 model_version=excluded.model_version""",
            rows,
        )
        logger.info("persisted %s predictions", len(rows))

    # ─── 辅助方法 ───

    @staticmethod
    def _prepare_native_frame(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """Build the six-feature frame used by Kronos training and inference."""
        columns = ["open", "high", "low", "close", "volume"]
        frame = df[columns].iloc[-lookback:].copy()
        if "amount" in df:
            frame["amount"] = pd.to_numeric(df["amount"].iloc[-lookback:], errors="coerce")
        else:
            frame["amount"] = np.nan
        fallback_amount = frame["close"] * frame["volume"]
        frame["amount"] = frame["amount"].where(frame["amount"] > 0, fallback_amount)
        return frame

    @staticmethod
    def _return_score(pred_return: float) -> float:
        """Map expected return to the requirement's bounded 0-10 score."""
        return float(np.clip(5.0 + pred_return * 100.0, 0.0, 10.0))

    @staticmethod
    def _estimate_prob_up(
        pred_close: float,
        last_close: float,
        pred_std: Optional[float],
    ) -> float:
        """
        Estimate P(future close > current close) from sample dispersion.

        Backends without sampling uncertainty use a centered logistic fallback.
        """
        if pred_std is not None and np.isfinite(pred_std) and pred_std > 1e-8:
            z_score = (pred_close - last_close) / pred_std
            return float(np.clip(NormalDist().cdf(z_score), 0.001, 0.999))
        pred_return = pred_close / last_close - 1 if last_close > 0 else 0.0
        return float(1.0 / (1.0 + np.exp(-pred_return / 0.02)))

    def _empty_prediction(self) -> Dict[str, Any]:
        """返回空预测结果"""
        return {
            "pred_close_t5": 0.0,
            "pred_return": 0.0,
            "last_close": 0.0,
            "pred_date": None,
            "confidence": 0.0,
            "prob_up": 0.0,
            "score": 0.0,
            "error": "insufficient_data",
        }
