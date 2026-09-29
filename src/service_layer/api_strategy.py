"""
Kronos — 策略观点 API

端点:
  GET /api/v1/strategy/opinion — 解释"为什么当前是这个仓位、什么条件下会变成买入"

数据来源:
  - trading_signal 表中配置择时标的的最新信号（含决策原因）；
  - 若信号未生成，则按 model_prediction 最新预测按规则推导并明确标注。
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter

from src.common.logger import get_logger
from src.service_layer.models import StrategyOpinion, StrategyOpinionTarget

logger = get_logger(__name__)

router = APIRouter(prefix="/strategy", tags=["策略观点"])


def _signal_from_prediction(pred_return: float, buy_threshold: float) -> tuple[str, str, float]:
    """按论文规则由预测收益率推导信号。"""
    if pred_return > buy_threshold:
        return "BUY", f"模型预测未来5日收益率 {pred_return:.2%} > 阈值 {buy_threshold:.2%}，触发满仓买入", 1.0
    return "CASH", f"模型预测未来5日收益率 {pred_return:.2%} ≤ 阈值 {buy_threshold:.2%}，保持空仓观望", 0.0


@router.get("/opinion", response_model=StrategyOpinion)
async def get_strategy_opinion():
    """
    返回当前策略观点：

    - 每个择时标的的最新信号、预测收益率、建议仓位与决策原因；
    - 触发条件说明（什么情况下会从空仓变为买入）；
    - latest_data_date 用于前端判断信号是否过期。
    """
    from src.common.config import load_config
    from src.data_layer.storage import StorageLayer

    config = load_config()
    storage = StorageLayer(config)

    symbols = [str(s) for s in config.get("strategy", "timing_symbols", default=None) or []]
    if not symbols:
        symbols = [str(config.get("strategy", "paper_symbol", default="000852"))]
    symbols = [s.zfill(6) for s in symbols]

    buy_threshold = float(config.get("strategy", "buy_threshold", default=0.005))
    hold_days = int(config.get("strategy", "hold_days", default=5))
    execution_symbol = str(config.get("strategy", "execution_symbol", default="512100"))
    risk_enabled = bool(config.get("strategy", "risk_controls_enabled", default=False))

    # ── 1. trading_signal 最新信号 ──
    placeholders = ",".join("?" for _ in symbols)
    rows = storage.execute_sql(
        f"""SELECT code, date, signal_type, pred_return, pred_close,
                   confidence, position_pct, reason
            FROM trading_signal
            WHERE code IN ({placeholders})
            ORDER BY date DESC, code""",
        tuple(symbols),
    )
    latest: dict[str, tuple] = {}
    for r in rows:
        code = str(r[0]).zfill(6)
        if code not in latest:
            latest[code] = tuple(r)

    # ── 2. 无信号时按最新模型预测推导 ──
    for sym in symbols:
        if sym in latest:
            continue
        pred_rows = storage.execute_sql(
            """SELECT code, date, pred_return, confidence
               FROM model_prediction
               WHERE code = ? ORDER BY date DESC LIMIT 1""",
            (sym,),
        )
        if pred_rows and pred_rows[0]:
            r = pred_rows[0]
            pred_return = float(r[2]) if r[2] is not None else 0.0
            confidence = float(r[3]) if r[3] is not None else 0.5
            signal_type, reason, position = _signal_from_prediction(pred_return, buy_threshold)
            latest[sym] = (sym, str(r[1]), signal_type, pred_return, None,
                           confidence, position, reason + "（由最新预测按规则推导，未持久化）")

    # ── 3. 最新数据日期（用于过期判断）──
    data_rows = storage.execute_sql("SELECT MAX(date) FROM model_prediction")
    latest_data_date = str(data_rows[0][0]) if data_rows and data_rows[0][0] else None

    # ── 名称解析：优先配置中的指数/ETF 名称（000852 同时是股票代码，勿混淆）──
    names: dict[str, str] = {}
    try:
        for item in config.get_all_symbols():
            names[str(item.get("code", "")).zfill(6)] = str(item.get("name", item.get("code", "")))
    except Exception:
        pass
    unresolved = [s for s in latest.keys() if s not in names]
    if unresolved:
        try:
            from src.service_layer.api_signal import _symbol_names
            names.update(_symbol_names(storage, unresolved))
        except Exception:
            pass

    targets = []
    for sym in symbols:
        r = latest.get(sym)
        if r is None:
            continue
        targets.append(StrategyOpinionTarget(
            symbol=sym,
            name=names.get(sym, sym),
            signal_type=str(r[2]),
            pred_return=float(r[3]) if r[3] is not None else None,
            position_pct=float(r[6] or 0.0),
            reason=str(r[7]) if r[7] else None,
            date=str(r[1]),
        ))

    rule = (
        f"每个交易日收盘后，Kronos 模型对标的未来 {hold_days} 个交易日收益率做出预测；"
        f"若预测收益率 > {buy_threshold:.1%} 则满仓买入，固定持有 {hold_days} 个交易日后卖出；"
        f"否则保持空仓观望。信号在收盘后生成，下一交易日开盘执行（标的 {execution_symbol}）。"
    )
    risk_rules = (
        [
            f"止损 {float(config.get('strategy', 'stop_loss', default=-0.08)):.0%}（相对买入价）",
            f"止盈 {float(config.get('strategy', 'take_profit', default=0.15)):.0%}（相对买入价）",
            f"同时持仓 ≤ {int(config.get('strategy', 'max_positions', default=5))} 只",
        ]
        if risk_enabled
        else ["论文复现口径：不叠加止损/止盈（risk_controls_enabled=false）"]
    )

    logger.info(f"GET /strategy/opinion → {len(targets)} 个标的")
    return StrategyOpinion(
        strategy_name="Kronos 指数择时（五日动量，论文 §3.2 口径）",
        rule=rule,
        buy_threshold=buy_threshold,
        hold_days=hold_days,
        execution_symbol=execution_symbol,
        risk_controls_enabled=risk_enabled,
        risk_rules=risk_rules,
        latest_data_date=latest_data_date,
        targets=targets,
        generated_at=datetime.now().isoformat(timespec="seconds"),
    )
