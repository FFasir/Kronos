"""Background jobs for A-share data, preprocessing, and strategy materialization."""

from __future__ import annotations

import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from src.common.config import Config, PROJECT_ROOT, load_config
from src.common.logger import get_logger
from src.data_layer.a_share_downloader import AShareDailyDownloader
from src.data_layer.downloader import DataDownloader
from src.data_layer.storage import StorageLayer
from src.strategy_layer.signal_generator import SignalGenerator
from src.strategy_layer.stock_selector import StockSelector


logger = get_logger(__name__)


class TaskScheduler:
    """Own the scheduled daily workflow and user-triggered background jobs."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or load_config()
        timezone = self.config.get("scheduler", "timezone", default="Asia/Shanghai")
        self.scheduler = BackgroundScheduler(timezone=timezone)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kronos-jobs")
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        update_time = self.config.get("scheduler", "daily_update_time", default="16:30")
        hour, minute = (int(part) for part in update_time.split(":"))
        self.scheduler.add_job(
            lambda: self.submit("daily_pipeline"),
            CronTrigger(
                day_of_week="mon-fri",
                hour=hour,
                minute=minute,
                timezone=self.scheduler.timezone,
            ),
            id="a_share_daily_pipeline",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        if self.config.get("model", "rolling", "enabled", default=False):
            monthly_time = self.config.get(
                "scheduler", "monthly_training_time", default="02:00"
            )
            monthly_hour, monthly_minute = (int(part) for part in monthly_time.split(":"))
            self.scheduler.add_job(
                lambda: self.submit("rolling_training"),
                CronTrigger(
                    day=int(self.config.get("scheduler", "monthly_training_day", default=1)),
                    hour=monthly_hour,
                    minute=monthly_minute,
                    timezone=self.scheduler.timezone,
                ),
                id="kronos_monthly_rolling_training",
                replace_existing=True,
                coalesce=True,
                max_instances=1,
            )
        self.scheduler.start()
        self._started = True
        logger.info("A-share scheduler started: weekdays at %s", update_time)

    def shutdown(self) -> None:
        if self._started:
            self.scheduler.shutdown(wait=False)
            self._started = False
        self.executor.shutdown(wait=False, cancel_futures=True)

    def submit(self, name: str) -> dict:
        handlers: dict[str, Callable[[], dict]] = {
            "data_update": self.update_data,
            "timing_data_update": self.update_timing_data,
            "preprocess": self.preprocess,
            "timing_signal": self.generate_timing_signal,
            "prediction": self.predict,
            "rolling_training": self.rolling_training,
            "selection_signal": self.materialize_selection_and_signals,
            "stock_hfq_update": self.update_stock_hfq,
            "daily_pipeline": self.daily_pipeline,
        }
        if name not in handlers:
            raise ValueError(f"unknown job: {name}")
        with self._lock:
            if any(job["name"] == name and job["status"] in {"accepted", "running"} for job in self._jobs.values()):
                raise RuntimeError(f"job is already running: {name}")
            job_id = uuid.uuid4().hex
            record = {
                "job_id": job_id,
                "name": name,
                "status": "accepted",
                "progress": 0.0,
                "submitted_at": datetime.now().isoformat(timespec="seconds"),
                "result": None,
                "error": None,
            }
            self._jobs[job_id] = record
        self.executor.submit(self._run, job_id, handlers[name])
        return dict(record)

    def status(self, job_id: str) -> Optional[dict]:
        with self._lock:
            record = self._jobs.get(job_id)
            return dict(record) if record else None

    def list_jobs(self) -> list[dict]:
        with self._lock:
            return [dict(item) for item in reversed(list(self._jobs.values())[-50:])]

    def _run(self, job_id: str, handler: Callable[[], dict]) -> None:
        self._update(job_id, status="running", progress=0.05, started_at=datetime.now().isoformat(timespec="seconds"))
        try:
            result = handler()
            self._update(
                job_id,
                status="completed",
                progress=1.0,
                result=result,
                completed_at=datetime.now().isoformat(timespec="seconds"),
            )
        except Exception as exc:
            logger.exception("background job %s failed", job_id)
            self._update(
                job_id,
                status="failed",
                error=str(exc),
                completed_at=datetime.now().isoformat(timespec="seconds"),
            )

    def _update(self, job_id: str, **values) -> None:
        with self._lock:
            self._jobs[job_id].update(values)

    def update_stock_hfq(self) -> dict:
        """Refresh per-stock hfq overlay for recently predicted/selected symbols."""
        from datetime import timedelta

        from src.data_layer.stock_hfq_live import refresh_symbols

        storage = StorageLayer(self.config)
        configured_codes = sorted({str(item["code"]) for item in self.config.get_all_symbols()})
        since = (date.today() - timedelta(days=30)).isoformat()
        placeholders = ",".join("?" for _ in configured_codes) or "''"
        rows = storage.execute_sql(
            f"""SELECT code FROM model_prediction WHERE date >= ? AND code NOT IN ({placeholders})
                UNION SELECT code FROM price_prediction WHERE substr(created_at,1,10) >= ? AND code NOT IN ({placeholders})
                UNION SELECT code FROM stock_selection WHERE date >= ? AND code NOT IN ({placeholders})
                ORDER BY code""",
            tuple([since] + configured_codes + [since] + configured_codes + [since] + configured_codes),
        )
        codes = [str(row["code"]) for row in rows]
        if not codes:
            return {"total": 0, "refreshed": 0, "message": "无近期被预测/入选的个股，跳过"}
        result = refresh_symbols(codes, config=self.config, force=False,
                                 request_interval=1.0)
        return result

    def update_data(self) -> dict:
        result = AShareDailyDownloader(self.config).download()
        if result["status"] != "completed":
            raise RuntimeError(result.get("error", "A-share data update failed"))
        return result

    def update_timing_data(self) -> dict:
        """Refresh the CSI1000 signal source and its 512100 execution asset."""
        symbols = [
            str(self.config.get("strategy", "paper_symbol", default="000852")),
            str(self.config.get("strategy", "execution_symbol", default="512100")),
        ]
        result = DataDownloader(self.config).download(
            mode="full",
            symbols=symbols,
            start=str(self.config.get("data", "download_start", default="2010-01-01")),
            end=date.today().strftime("%Y-%m-%d"),
        )
        if result["failed_count"]:
            raise RuntimeError(f"timing market data update failed: {result['failed']}")
        return result

    def preprocess(self) -> dict:
        archive = PROJECT_ROOT / self.config.get(
            "data", "a_share_archive", default="A股数据/daily.zip"
        )
        output = PROJECT_ROOT / self.config.get(
            "data", "component_processed_path", default="data/processed/components.h5"
        )
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "import_stock_archive.py"),
            "--archive", str(archive),
            "--output", str(output),
        ]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=int(self.config.get("scheduler", "preprocess_timeout_seconds", default=7200)),
        )
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout)[-2000:])
        return {"output": str(output), "message": completed.stdout[-1000:]}

    def materialize_selection_and_signals(self) -> dict:
        from src.service_layer.api_prediction import _try_get_real_predictions
        from src.service_layer.api_selection import _with_liquidity

        predictions = _try_get_real_predictions()
        if not predictions:
            raise RuntimeError("no persisted prediction batch; run model prediction first")
        candidates = _with_liquidity(predictions)
        # select() 返回 {"selected": [...], "stats": ...}; 遍历前必须解包，
        # 否则 for item in selected 会把 dict 键当字符串处理 → TypeError。
        result = StockSelector(self.config).select(candidates)
        selected = result["selected"]
        signal_date = self._latest_market_date() or date.today().isoformat()
        # 信号只针对已选中的股票生成，而不是对全量候选（几万行垃圾信号）。
        signals = SignalGenerator(self.config).generate(selected, signal_date)

        storage = StorageLayer(self.config)
        storage.execute_write("DELETE FROM stock_selection WHERE date = ?", (signal_date,))
        storage.execute_many(
            """INSERT INTO stock_selection (date, code, prob_up, score, rank, selected)
               VALUES (?, ?, ?, ?, ?, 1)""",
            [
                (
                    signal_date,
                    item["symbol"],
                    item.get("prob_up"),
                    item.get("score"),
                    item.get("rank"),
                )
                for item in selected
            ],
        )
        storage.execute_many(
            """INSERT INTO trading_signal
               (code, date, signal_type, pred_return, pred_close, confidence, position_pct, reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(code, date) DO UPDATE SET
                 signal_type=excluded.signal_type, pred_return=excluded.pred_return,
                 pred_close=excluded.pred_close, confidence=excluded.confidence,
                 position_pct=excluded.position_pct, reason=excluded.reason""",
            [
                (
                    item["symbol"], signal_date, item["signal_type"], item.get("pred_return"),
                    item.get("pred_close"), item.get("confidence"), item.get("position_pct", 0),
                    item.get("reason"),
                )
                for item in signals
            ],
        )
        return {"date": signal_date, "selected": len(selected), "signals": len(signals)}

    def generate_timing_signal(self) -> dict:
        """Persist the index signal and all current CSI 1000 stock signals."""
        symbols = self.config.get("strategy", "timing_symbols", default=None)
        if not symbols:
            symbols = [self.config.get("strategy", "paper_symbol", default="000852")]
        symbols = [str(symbol) for symbol in symbols]

        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "predict.py"),
            "--symbols", ",".join(symbols),
            "--summary-only",
        ]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=int(self.config.get(
                "scheduler", "prediction_timeout_seconds", default=7200
            )),
        )
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout)[-4000:])

        storage = StorageLayer(self.config)
        placeholders = ",".join("?" for _ in symbols)
        latest_prediction = storage.execute_sql(
            f"SELECT MAX(date) FROM model_prediction WHERE code IN ({placeholders})",
            tuple(symbols),
        )
        signal_date = (
            str(latest_prediction[0][0])
            if latest_prediction and latest_prediction[0][0]
            else date.today().isoformat()
        )
        rows = storage.execute_sql(
            f"""SELECT p.code, p.pred_date, p.pred_close, p.pred_return,
                       p.last_close, p.confidence, p.temperature, p.top_p,
                       p.lookback, p.score, s.name
                FROM model_prediction p
                LEFT JOIN stock_info s ON s.code = p.code
                WHERE p.date = ? AND p.code IN ({placeholders})
                ORDER BY p.code""",
            (signal_date, *symbols),
        )
        predictions = [
            {
                "symbol": str(row[0]),
                "name": str(row[10] or row[0]),
                "pred_date": str(row[1]),
                "pred_close_t5": float(row[2]),
                "pred_return": float(row[3]),
                "last_close": float(row[4]),
                "confidence": float(row[5]),
                "prob_up": float(row[5]),
                "temperature": float(row[6]),
                "top_p": float(row[7]),
                "lookback_window": int(row[8]),
                "score": float(row[9]),
            }
            for row in rows
        ]
        if not predictions:
            raise RuntimeError(f"no successful timing predictions for {signal_date}")

        positions = self._load_timing_positions(storage, symbols, signal_date)
        signals = SignalGenerator(self.config).generate(
            predictions, signal_date, existing_positions=positions
        )
        execution_symbol = str(self.config.get(
            "strategy", "execution_symbol", default="512100"
        ))
        for signal in signals:
            signal["reason"] = (
                f"{signal.get('reason', '')}; 信号收盘后生成，下一交易日开盘交易"
                f"{execution_symbol}，第5个后续交易日收盘退出"
            )
        SignalGenerator(self.config)._save_signals_to_db(pd.DataFrame(signals))
        from src.strategy_layer.realtime_timing import materialize_component_timing_signals

        component_result = materialize_component_timing_signals(
            self.config, storage, refresh=True
        )
        return {
            "date": signal_date,
            "model_predictions": len(predictions),
            "signals": signals,
            "execution_symbol": execution_symbol,
            "components": component_result,
            "message": completed.stdout[-1000:],
        }

    def _load_timing_positions(
        self,
        storage: StorageLayer,
        symbols: list[str],
        signal_date: str,
    ) -> dict:
        """Reconstruct five-day paper positions from prior persisted signals."""
        from src.strategy_layer.signal_generator import PositionState

        positions = {}
        for symbol in symbols:
            latest = storage.execute_sql(
                """SELECT date, signal_type, position_pct
                   FROM trading_signal
                   WHERE code = ? AND date < ?
                   ORDER BY date DESC LIMIT 1""",
                (symbol, signal_date),
            )
            if not latest or str(latest[0][1]) not in {"BUY", "HOLD"}:
                continue
            entries = storage.execute_sql(
                """SELECT date FROM trading_signal
                   WHERE code = ? AND date < ? AND signal_type = 'BUY'
                   ORDER BY date DESC LIMIT 1""",
                (symbol, signal_date),
            )
            if not entries:
                continue
            entry_date = str(entries[0][0])
            terminal = storage.execute_sql(
                """SELECT 1 FROM trading_signal
                   WHERE code = ? AND date > ? AND date < ?
                     AND signal_type IN ('SELL', 'CASH') LIMIT 1""",
                (symbol, entry_date, signal_date),
            )
            if terminal:
                continue
            market = storage.load_raw_data(symbol)
            trading_days_held = int(
                (
                    (market.index > pd.Timestamp(entry_date))
                    & (market.index <= pd.Timestamp(signal_date))
                ).sum()
            )
            positions[symbol] = {
                "entry_date": entry_date,
                "hold_day_count": trading_days_held,
                "state": PositionState.LONG,
                "position_pct": float(latest[0][2] or 1.0),
            }
        return positions

    def predict(self) -> dict:
        storage = StorageLayer(self.config)
        storage.init_database()
        eligible = storage.execute_sql(
            """SELECT COUNT(*) AS count
               FROM model_prediction p
               WHERE p.actual_return IS NULL
                 AND EXISTS (
                   SELECT 1 FROM stock_daily d
                   WHERE d.code = p.code AND d.date = p.pred_date
                 )"""
        )
        realized = int(eligible[0]["count"]) if eligible else 0
        if realized:
            storage.execute_write(
                """UPDATE model_prediction
                   SET actual_close = (
                         SELECT d.close FROM stock_daily d
                         WHERE d.code = model_prediction.code
                           AND d.date = model_prediction.pred_date
                       ),
                       actual_return = (
                         SELECT d.close FROM stock_daily d
                         WHERE d.code = model_prediction.code
                           AND d.date = model_prediction.pred_date
                       ) / NULLIF(
                         COALESCE(
                           last_close,
                           pred_close / NULLIF(1 + pred_return, 0)
                         ), 0
                       ) - 1
                   WHERE actual_return IS NULL
                     AND EXISTS (
                       SELECT 1 FROM stock_daily d
                       WHERE d.code = model_prediction.code
                         AND d.date = model_prediction.pred_date
                     )"""
            )
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "predict.py"),
            "--summary-only",
        ]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=int(self.config.get(
                "scheduler", "prediction_timeout_seconds", default=7200
            )),
        )
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout)[-4000:])
        return {"realized_predictions": realized, "message": completed.stdout[-2000:]}

    def rolling_training(self) -> dict:
        rolling = self.config.get("model", "rolling", default={}) or {}
        target_month = str(rolling.get("target_month") or datetime.now().strftime("%Y-%m"))
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_monthly_rolling_finetune.py"),
            "--start-month", target_month,
            "--end-month", target_month,
            "--source", str(PROJECT_ROOT / rolling.get(
                "source_dataset", "data/official_processed_datasets_all_a_share"
            )),
            "--dataset-root", str(PROJECT_ROOT / rolling.get(
                "dataset_root", "data/rolling_kronos"
            )),
            "--model-root", str(PROJECT_ROOT / rolling.get(
                "model_root", "models/rolling_kronos"
            )),
            "--base-tokenizer", str(PROJECT_ROOT / self.config.get(
                "model", "finetuned_tokenizer_path", default="models/kronos_tokenizer"
            )),
            "--base-predictor", str(PROJECT_ROOT / self.config.get(
                "model", "finetuned_predictor_path", default="models/kronos_base"
            )),
            "--lookback", str(rolling.get("lookback", 65)),
            "--horizon", str(rolling.get("horizon", 20)),
            "--validation-days", str(rolling.get("validation_days", 22)),
            "--epochs", str(rolling.get("epochs", 10)),
            "--train-samples", str(rolling.get("train_samples", 60000)),
            "--val-samples", str(rolling.get("val_samples", 6000)),
            "--batch-size", str(rolling.get("batch_size", 30)),
            "--prediction-samples", str(rolling.get("prediction_samples", 5)),
            "--prediction-batch-size", str(rolling.get("prediction_batch_size", 32)),
            "--device", str(rolling.get("device", "cuda")),
            "--seed", str(self.config.seed),
        ]
        if rolling.get("activate_after_training", False):
            command.append("--activate-model")
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=int(self.config.get(
                "scheduler", "rolling_training_timeout_seconds", default=28800
            )),
        )
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout)[-4000:])
        return {"target_month": target_month, "message": completed.stdout[-3000:]}

    def daily_pipeline(self) -> dict:
        timing_data_result = self.update_timing_data()
        try:
            stock_data_result = self.update_data()
        except Exception as exc:
            logger.exception("A-share snapshot update failed")
            stock_data_result = {"status": "failed", "error": str(exc)}
        prediction_result = self.predict()
        # 全市场每日预测：保证任意个股页面都有当日新鲜预测（约 8 分钟/批）。
        # 由 scheduler.full_a_share_daily 控制，默认开启。
        full_a_share_result = None
        if self.config.get("scheduler", "full_a_share_daily", default=True):
            try:
                full_a_share_result = self.predict_all_a_share()
            except Exception as exc:
                logger.exception("full A-share daily prediction failed")
                full_a_share_result = {"status": "failed", "error": str(exc)}
        # 回填「目标日已到期」批次的 actual_return/actual_close，
        # 保证个股"预测 vs 实际 档案"随时间自动出现到期对照（幂等，可重复跑）。
        actual_fill_result = None
        try:
            actual_fill_result = self.fill_expired_actual_returns()
        except Exception as exc:
            logger.exception("expired actual-return fill failed")
            actual_fill_result = {"status": "failed", "error": str(exc)}
        strategy_result = self.generate_timing_signal()
        return {
            "timing_data": timing_data_result,
            "stock_data": stock_data_result,
            "prediction": prediction_result,
            "full_a_share": full_a_share_result,
            "actual_fill": actual_fill_result,
            "strategy": strategy_result,
        }

    def predict_all_a_share(self) -> dict:
        """全 A 股截面预测（--all-a-share --from-storage），每日收盘后刷新任意个股的预测。"""
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "predict.py"),
            "--all-a-share", "--from-storage",
        ]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=int(self.config.get(
                "scheduler", "prediction_timeout_seconds", default=7200
            )),
        )
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout)[-4000:])
        return {"status": "completed", "message": completed.stdout[-1000:]}

    def fill_expired_actual_returns(self) -> dict:
        """回填已到期批次的 actual_return/actual_close（幂等；目标日未到期自动跳过）。"""
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "backfill_actual_returns.py"),
        ]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=int(self.config.get(
                "scheduler", "actual_fill_timeout_seconds", default=7200
            )),
        )
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout)[-4000:])
        return {"status": "completed", "message": completed.stdout[-1000:]}

    def _latest_market_date(self) -> Optional[str]:
        storage = StorageLayer(self.config)
        rows = storage.execute_sql("SELECT MAX(date) AS latest FROM stock_daily")
        return str(rows[0]["latest"]) if rows and rows[0]["latest"] else None


_instance: Optional[TaskScheduler] = None
_instance_lock = threading.Lock()


def get_task_scheduler() -> TaskScheduler:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = TaskScheduler()
        return _instance
