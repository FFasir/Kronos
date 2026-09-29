"""Point-in-time realtime timing signals for current CSI 1000 constituents."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import PROJECT_ROOT
from src.common.logger import get_logger


logger = get_logger(__name__)
MIN_COMPONENTS = 900


def refresh_current_csi1000_components(output_dir: Path | None = None) -> Path:
    """Download the official current CSI 1000 constituent snapshot."""
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("AKShare is required to refresh CSI 1000 constituents") from exc

    raw = ak.index_stock_cons_csindex(symbol="000852")
    if raw is None or raw.empty:
        raise RuntimeError("CSI Index returned no CSI 1000 constituents")
    code_column = next(
        (column for column in ["成分券代码", "股票代码", "品种代码", "con_code", "code"] if column in raw),
        None,
    )
    if code_column is None:
        raise RuntimeError(f"unexpected CSI constituent columns: {list(raw.columns)}")
    name_column = next(
        (column for column in ["成分券名称", "股票名称", "品种名称", "name"] if column in raw),
        None,
    )
    snapshot = pd.DataFrame({
        "code": raw[code_column].astype(str).str.strip().str.split(".").str[0].str.zfill(6),
        "name": raw[name_column].astype(str).str.strip() if name_column else "",
    })
    snapshot = snapshot[snapshot["code"].str.fullmatch(r"\d{6}")].drop_duplicates("code")
    if len(snapshot) < MIN_COMPONENTS:
        raise RuntimeError(
            f"official CSI 1000 snapshot is incomplete: {len(snapshot)} < {MIN_COMPONENTS}"
        )
    source_date = None
    if "日期" in raw:
        parsed = pd.to_datetime(raw["日期"], errors="coerce").dropna()
        if not parsed.empty:
            source_date = parsed.max().strftime("%Y-%m-%d")
    snapshot["snapshot_date"] = source_date or date.today().isoformat()
    snapshot = snapshot.sort_values("code").reset_index(drop=True)

    output_dir = output_dir or PROJECT_ROOT / "data" / "raw" / "research"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = snapshot["snapshot_date"].iloc[0].replace("-", "")
    output = output_dir / f"csi1000_csindex_{suffix}.csv"
    snapshot.to_csv(output, index=False, encoding="utf-8-sig")
    logger.info("refreshed official CSI 1000 snapshot: %s members=%s", output, len(snapshot))
    return output


def load_current_csi1000_components(refresh: bool = False) -> pd.DataFrame:
    """Load a validated current snapshot, refreshing it when requested or absent."""
    research_dir = PROJECT_ROOT / "data" / "raw" / "research"
    if refresh:
        try:
            refresh_current_csi1000_components(research_dir)
        except Exception as exc:
            logger.warning("unable to refresh CSI 1000 snapshot, using local copy: %s", exc)

    candidates = sorted(
        research_dir.glob("csi1000_csindex_*.csv"),
        key=lambda path: (path.name, path.stat().st_mtime),
        reverse=True,
    )
    if not candidates:
        path = refresh_current_csi1000_components(research_dir)
        candidates = [path]

    for path in candidates:
        frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
        code_column = next(
            (column for column in ["code", "成分券代码", "股票代码", "con_code"] if column in frame),
            None,
        )
        if code_column is None:
            continue
        result = pd.DataFrame({
            "code": frame[code_column].astype(str).str.strip().str.split(".").str[0].str.zfill(6),
            "name": frame["name"].fillna("").astype(str) if "name" in frame else "",
            "snapshot_date": frame["snapshot_date"].astype(str) if "snapshot_date" in frame else "",
        })
        result = result[result["code"].str.fullmatch(r"\d{6}")].drop_duplicates("code")
        if len(result) >= MIN_COMPONENTS:
            return result.sort_values("code").reset_index(drop=True)
    raise RuntimeError(
        "no complete official CSI 1000 snapshot is available; refresh requires at least "
        f"{MIN_COMPONENTS} members"
    )


def materialize_component_timing_signals(config, storage, refresh: bool = False) -> dict:
    """Create one realtime timing signal for every current CSI 1000 member."""
    members = load_current_csi1000_components(refresh=refresh)
    member_codes = set(members["code"])
    recent_rows = storage.execute_sql(
        """SELECT code, date, pred_date, pred_close, pred_return, last_close,
                  confidence, model_version
           FROM model_prediction
           WHERE date >= date((SELECT MAX(date) FROM model_prediction), '-14 day')
           ORDER BY date DESC, code"""
    )
    predictions = pd.DataFrame([dict(row) for row in recent_rows])
    if predictions.empty:
        raise RuntimeError("no persisted model prediction batch is available")
    predictions["code"] = predictions["code"].astype(str).str.zfill(6)
    predictions = predictions[predictions["code"].isin(member_codes)].copy()
    coverage = predictions.groupby("date")["code"].nunique().sort_index()
    eligible_dates = coverage[coverage >= MIN_COMPONENTS].index
    if len(eligible_dates) == 0:
        best = int(coverage.max()) if not coverage.empty else 0
        raise RuntimeError(
            f"latest prediction batches cover at most {best} current CSI 1000 members; "
            f"at least {MIN_COMPONENTS} are required"
        )
    signal_date = str(eligible_dates[-1])
    predicted_batch = predictions[
        predictions["date"] == signal_date
    ].drop_duplicates("code", keep="last")
    batch = members[["code"]].merge(predicted_batch, on="code", how="left")
    batch["date"] = signal_date

    prior_rows = storage.execute_sql(
        """SELECT code, date, signal_type, position_pct
           FROM trading_signal WHERE date < ? ORDER BY code, date""",
        (signal_date,),
    )
    prior = pd.DataFrame([dict(row) for row in prior_rows])
    batch_dates = [str(value) for value in eligible_dates if str(value) <= signal_date]
    signals = build_component_signals(
        batch,
        prior,
        batch_dates=batch_dates,
        buy_threshold=float(config.get("strategy", "buy_threshold", default=0.005)),
        hold_days=int(config.get("strategy", "hold_days", default=5)),
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
                item["symbol"], signal_date, item["signal_type"], item["pred_return"],
                item["pred_close"], item["confidence"], item["position_pct"], item["reason"],
            )
            for item in signals
        ],
    )
    counts = pd.Series([item["signal_type"] for item in signals]).value_counts().to_dict()
    model_versions = sorted({str(value) for value in batch["model_version"].dropna().unique()})
    return {
        "date": signal_date,
        "snapshot_date": str(members["snapshot_date"].max()),
        "members": len(members),
        "prediction_coverage": len(predicted_batch),
        "signals": len(signals),
        "signal_counts": counts,
        "model_versions": model_versions,
    }


def build_component_signals(
    predictions: pd.DataFrame,
    prior_signals: pd.DataFrame,
    batch_dates: list[str],
    buy_threshold: float,
    hold_days: int,
) -> list[dict[str, Any]]:
    """Pure signal-state transition used by production and unit tests."""
    latest_prior: dict[str, dict] = {}
    last_buy: dict[str, str] = {}
    if not prior_signals.empty:
        for row in prior_signals.to_dict("records"):
            code = str(row["code"]).zfill(6)
            latest_prior[code] = row
            if str(row["signal_type"]) == "BUY":
                last_buy[code] = str(row["date"])

    signals = []
    for row in predictions.sort_values("code").to_dict("records"):
        code = str(row["code"]).zfill(6)
        prediction_available = pd.notna(row.get("pred_return"))
        pred_return = float(row.get("pred_return")) if prediction_available else 0.0
        latest = latest_prior.get(code)
        active = latest is not None and str(latest.get("signal_type")) in {"BUY", "HOLD"}
        if active:
            entry_date = last_buy.get(code, str(latest["date"]))
            held = sum(entry_date < value <= str(row["date"]) for value in batch_dates)
            if held >= hold_days:
                signal_type = "SELL"
                reason = f"逐股固定持有到期 ({held}/{hold_days}个交易日)"
            else:
                signal_type = "HOLD"
                reason = f"逐股持仓中 ({held}/{hold_days}个交易日)"
        elif not prediction_available:
            signal_type = "CASH"
            reason = "当日逐股Kronos预测缺失，禁止交易"
        elif pred_return > buy_threshold:
            signal_type = "BUY"
            reason = f"逐股五日预测收益 {pred_return:.2%} > {buy_threshold:.2%}"
        else:
            signal_type = "CASH"
            reason = f"逐股五日预测收益 {pred_return:.2%} <= {buy_threshold:.2%}"
        signals.append({
            "symbol": code,
            "signal_type": signal_type,
            "pred_return": pred_return,
            "pred_close": float(row.get("pred_close")) if pd.notna(row.get("pred_close")) else 0.0,
            "confidence": float(row.get("confidence")) if pd.notna(row.get("confidence")) else 0.0,
            "position_pct": 0.0,
            "reason": reason + "; 信号日收盘后生成，下一交易日开盘执行该股票",
        })

    longs = [item for item in signals if item["signal_type"] in {"BUY", "HOLD"}]
    equal_weight = 1.0 / len(longs) if longs else 0.0
    for item in longs:
        item["position_pct"] = equal_weight
    return signals
