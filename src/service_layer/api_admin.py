"""
Kronos 指数/ETF 择时策略系统 — 管理后台 API

端点:
  POST /api/v1/admin/login         — 管理员登录
  POST /api/v1/admin/logout        — 管理员退出
  GET  /api/v1/admin/monitor       — 系统监控状态
  GET  /api/v1/admin/training/log  — 训练日志
  GET  /api/v1/admin/data/sources  — 数据源状态
  POST /api/v1/admin/config        — 保存/读取系统配置
  GET  /api/v1/admin/config        — 读取系统配置
"""

from __future__ import annotations

import os
import time
import platform
import hmac
import secrets
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from src.common.config import Config
from src.common.logger import get_logger
from src.service_layer.models import (
    AdminLoginRequest, AdminLoginResponse,
    SystemMonitor, TrainingLogEntry, DataSourceInfo,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/admin", tags=["管理后台"])


class RegisterModelRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    version: str = Field(min_length=1, max_length=120)
    predictor_path: str = Field(min_length=1, max_length=500)
    tokenizer_path: str = Field(min_length=1, max_length=500)

# 服务启动时间
_SERVER_START_TIME = time.time()

# 内存会话：token -> 过期时间戳。服务重启后会话自动失效。
_valid_tokens: dict[str, float] = {}


async def require_admin(authorization: Optional[str] = Header(default=None)) -> str:
    """校验管理接口的 Bearer token，并返回有效 token。"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="请先登录管理后台",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization.removeprefix("Bearer ").strip()
    expires_at = _valid_tokens.get(token)
    if not expires_at or expires_at <= time.time():
        _valid_tokens.pop(token, None)
        raise HTTPException(
            status_code=401,
            detail="管理会话已过期，请重新登录",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


# ═══════════════════════════════════════════════════════════════════
# 登录
# ═══════════════════════════════════════════════════════════════════

@router.post("/login", response_model=AdminLoginResponse)
async def admin_login(req: AdminLoginRequest):
    """
    管理员登录。
    """
    from src.common.config import load_config

    config = load_config()
    is_production = os.getenv("KRONOS_ENV", "development").lower() == "production"
    env_username = os.getenv("KRONOS_ADMIN_USERNAME")
    env_password = os.getenv("KRONOS_ADMIN_PASSWORD")
    if is_production and not env_password:
        logger.error("生产环境未设置 KRONOS_ADMIN_PASSWORD")
        raise HTTPException(status_code=503, detail="管理员认证尚未配置")

    valid_username = env_username or str(
        config.get("server", "admin_username", default="admin")
    )
    valid_password = env_password or str(
        config.get("server", "admin_password", default="")
    )

    username_matches = hmac.compare_digest(req.username, valid_username)
    password_matches = bool(valid_password) and hmac.compare_digest(req.password, valid_password)
    if username_matches and password_matches:
        session_hours = float(config.get("server", "admin_session_hours", default=8))
        token = secrets.token_urlsafe(32)
        _valid_tokens[token] = time.time() + max(session_hours, 0.25) * 3600
        logger.info("管理后台登录成功")
        return AdminLoginResponse(success=True, token=token, message="登录成功")

    logger.warning("管理后台登录失败: 凭据错误")
    return AdminLoginResponse(success=False, token=None, message="账号或密码错误")


@router.post("/logout")
async def admin_logout(token: str = Depends(require_admin)):
    """注销当前管理会话。"""
    _valid_tokens.pop(token, None)
    return {"success": True, "message": "已退出登录"}


# ═══════════════════════════════════════════════════════════════════
# 系统监控
# ═══════════════════════════════════════════════════════════════════

def compute_system_monitor() -> SystemMonitor:
    """
    计算系统监控状态（供公开 /system/status 与受保护的 /admin/monitor 共用）。

    包含：模型状态、数据状态、CPU/内存/GPU 使用率、服务运行时间等。
    """
    from src.common.config import load_config

    config = load_config()

    # 计算运行时间
    uptime_seconds = int(time.time() - _SERVER_START_TIME)
    days = uptime_seconds // 86400
    hours = (uptime_seconds % 86400) // 3600
    minutes = (uptime_seconds % 3600) // 60
    uptime_str = f"{days}d {hours}h {minutes}m"

    # 获取 CPU 使用率
    cpu_usage = None
    try:
        cpu_usage = round(os.getloadavg()[0] / os.cpu_count() * 100, 1) if hasattr(os, 'getloadavg') and os.cpu_count() else None
    except (OSError, AttributeError):
        pass

    # 获取内存使用率
    memory_usage = None
    try:
        import psutil
        mem = psutil.virtual_memory()
        memory_usage = round(mem.percent, 1)
    except ImportError:
        pass

    # GPU 信息
    gpu_usage = None
    gpu_memory_used = None
    gpu_memory_total = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_usage = round(torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 50.0, 1)
            mem_info = torch.cuda.mem_get_info()
            gpu_memory_used = round(mem_info[1] - mem_info[0], 0) if mem_info else None
            gpu_memory_total = round(mem_info[1], 0) if mem_info else None
    except ImportError:
        pass
    except Exception:
        # GPU 检测失败，保持 None
        pass

    # 模型状态
    model_status = "idle"
    model_version = None
    try:
        model_path = config.get("model", "pretrained_path", default="models/kronos_base")
        if os.path.exists(model_path):
            model_status = "running"
            model_version = config.get("model", "kronos_version", default="base")
    except Exception:
        pass

    # 数据状态
    data_status = "stale"
    last_data_update = None
    try:
        from src.data_layer.storage import StorageLayer
        storage = StorageLayer(config)
        rows = storage.execute_sql("SELECT MAX(date) FROM stock_daily")
        if rows and rows[0][0]:
            data_status = "ready"
            last_data_update = str(rows[0][0])
        else:
            sample = storage.load_raw_data("000001")
            if not sample.empty:
                last_data_update = str(sample.index[-1])[:10]
                age = (datetime.now() - datetime.strptime(last_data_update, "%Y-%m-%d")).days
                data_status = "ready" if age <= 3 else "stale"
    except Exception:
        pass

    # 最后训练/预测时间
    last_training = _get_last_training_time()
    last_prediction = _get_last_prediction_time()

    # 当前持仓数（尝试从数据库查询）
    active_positions = None
    try:
        from src.data_layer.storage import StorageLayer
        storage = StorageLayer(config)
        rows = storage.execute_sql(
            """SELECT COUNT(DISTINCT code) FROM trading_signal
               WHERE signal_type = 'HOLD'"""
        )
        if rows and rows[0]:
            active_positions = int(rows[0][0])
    except Exception:
        pass

    return SystemMonitor(
        model_status=model_status,
        data_status=data_status,
        cpu_usage=cpu_usage,
        memory_usage=memory_usage,
        gpu_usage=gpu_usage,
        gpu_memory_used=gpu_memory_used,
        gpu_memory_total=gpu_memory_total,
        model_version=model_version,
        last_training_time=last_training,
        last_prediction_time=last_prediction,
        last_data_update=last_data_update,
        uptime=uptime_str,
        pool_count=_latest_pool_count(),
        active_positions=active_positions,
    )


def compute_prediction_coverage() -> dict:
    """预测覆盖统计：A股目录标的数 vs 最新一批预测覆盖数（供公开状态接口）。"""
    try:
        from src.common.config import load_config
        from src.data_layer.storage import StorageLayer
        from src.service_layer.api_index import _stock_catalog

        config = load_config()
        storage = StorageLayer(config)
        catalog = _stock_catalog()
        universe = sum(1 for item in catalog.values() if item.get("is_active", True))

        max_date_row = storage.execute_sql("SELECT MAX(date) FROM model_prediction")
        latest_date = str(max_date_row[0][0]) if max_date_row and max_date_row[0][0] else None
        predicted = 0
        if latest_date:
            cnt = storage.execute_sql(
                "SELECT COUNT(DISTINCT code) FROM model_prediction WHERE date = ?",
                (latest_date,),
            )
            predicted = int(cnt[0][0]) if cnt and cnt[0][0] else 0
        return {
            "universe": universe,
            "predicted": predicted,
            "prediction_date": latest_date,
            "coverage_pct": round(predicted / universe * 100, 1) if universe else None,
        }
    except Exception:
        return {"universe": 0, "predicted": 0, "prediction_date": None, "coverage_pct": None}


@router.get("/monitor", response_model=SystemMonitor, dependencies=[Depends(require_admin)])
async def get_system_monitor():
    """
    获取系统监控状态。

    包含：模型状态、数据状态、CPU/内存/GPU 使用率、服务运行时间等。
    """
    logger.info("GET /admin/monitor")
    return compute_system_monitor()


# ═══════════════════════════════════════════════════════════════════
# 训练日志
# ═══════════════════════════════════════════════════════════════════

@router.get("/training/log", response_model=list[TrainingLogEntry], dependencies=[Depends(require_admin)])
async def get_training_logs(
    limit: int = Query(20, ge=1, le=200, description="返回条数"),
):
    """
    获取训练日志（最近 N 条）。

    从 SQLite training_log 表查询。
    """
    logs = []

    try:
        from src.data_layer.storage import StorageLayer
        from src.common.config import load_config

        config = load_config()
        storage = StorageLayer(config)
        rows = storage.execute_sql(
            """SELECT id, window_id, start_time, end_time, status,
                      epochs_completed, best_val_loss, spearman,
                      checkpoint_path, error_message
               FROM training_log
               ORDER BY start_time DESC LIMIT ?""",
            (limit,),
        )
        if rows:
            for r in rows:
                logs.append(TrainingLogEntry(
                    id=int(r[0]) if r[0] else None,
                    window_id=str(r[1]),
                    start_time=str(r[2]),
                    end_time=str(r[3]) if len(r) > 3 and r[3] else None,
                    status=str(r[4]),
                    epochs_completed=int(r[5]) if len(r) > 5 and r[5] else None,
                    best_val_loss=float(r[6]) if len(r) > 6 and r[6] else None,
                    best_spearman=float(r[7]) if len(r) > 7 and r[7] else None,
                    model_path=str(r[8]) if len(r) > 8 and r[8] else None,
                    error_message=str(r[9]) if len(r) > 9 and r[9] else None,
                ))
    except Exception as e:
        logger.debug(f"训练日志查询失败: {e}")

    logger.info(f"GET /admin/training/log → {len(logs)} 条")
    return logs


# ═══════════════════════════════════════════════════════════════════
# 数据源状态
# ═══════════════════════════════════════════════════════════════════

@router.get("/data/sources", response_model=list[DataSourceInfo], dependencies=[Depends(require_admin)])
async def get_data_sources():
    """
    获取所有标的数据源状态。

    包含每个标的数据覆盖日期范围、行数、缺失情况。
    """
    from src.common.config import load_config

    config = load_config()
    from src.common.config import PROJECT_ROOT
    from src.data_layer.storage import StorageLayer
    import pandas as pd

    storage = StorageLayer(config)
    catalog_path = PROJECT_ROOT / "A股数据" / "股票列表.csv"
    catalog_count = len(pd.read_csv(catalog_path, dtype=str)) if catalog_path.exists() else 0
    sample = storage.load_raw_data("000001")
    start_date = str(sample.index[0])[:10] if not sample.empty else None
    end_date = str(sample.index[-1])[:10] if not sample.empty else None
    days_behind = (datetime.now() - datetime.strptime(end_date, "%Y-%m-%d")).days if end_date else None
    status = "ready" if days_behind is not None and days_behind <= 3 else "stale"
    sources = [DataSourceInfo(
        symbol="A_SHARE",
        name="全A股日线",
        type="stock",
        start_date=start_date,
        end_date=end_date,
        total_rows=catalog_count,
        status=status,
        missing_dates=max(days_behind - 3, 0) if days_behind is not None else None,
    )]

    logger.info(f"GET /admin/data/sources → {len(sources)} 个标的")
    return sources


@router.get("/jobs", dependencies=[Depends(require_admin)])
async def get_background_jobs():
    from src.scheduler.task_scheduler import get_task_scheduler

    return get_task_scheduler().list_jobs()


@router.post("/jobs/{job_name}", status_code=202, dependencies=[Depends(require_admin)])
async def run_background_job(job_name: str):
    from src.scheduler.task_scheduler import get_task_scheduler

    try:
        return get_task_scheduler().submit(job_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/jobs/status/{job_id}", dependencies=[Depends(require_admin)])
async def get_background_job(job_id: str):
    from src.scheduler.task_scheduler import get_task_scheduler

    status = get_task_scheduler().status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
    return status


# ═══════════════════════════════════════════════════════════════════
# 配置管理
# ═══════════════════════════════════════════════════════════════════

@router.get("/config", dependencies=[Depends(require_admin)])
async def get_admin_config():
    """
    获取系统配置（策略参数等）。

    返回 config/system.yaml 中与前端相关的配置节。
    """
    from src.common.config import load_config

    config = load_config()
    return {
        "strategy": {
            "buy_threshold": config.get("strategy", "buy_threshold", default=0.005),
            "long_only": config.get("strategy", "long_only", default=True),
            "hold_days": config.get("strategy", "hold_days", default=5),
            "stop_loss": config.get("strategy", "stop_loss", default=-0.08),
            "take_profit": config.get("strategy", "take_profit", default=0.15),
            "max_single_position": config.get("strategy", "max_single_position", default=0.30),
            "min_positions": config.get("strategy", "min_positions", default=2),
        },
        "backtest": {
            "default_capital": config.get("backtest", "default_capital", default=1000000.0),
            "commission_rate": config.get("backtest", "commission_rate", default=0.0003),
            "stamp_tax_rate": config.get("backtest", "stamp_tax_rate", default=0.001),
            "slippage_rate": config.get("backtest", "slippage_rate", default=0.001),
            "benchmark_index": config.get("backtest", "benchmark_index", default="000852"),
            "risk_free_rate": config.get("backtest", "risk_free_rate", default=0.025),
        },
        "model": {
            "kronos_version": config.get("model", "kronos_version", default="base"),
            "device": config.get("model", "device", default="auto"),
            "training": {
                "seq_len": config.get("model", "training", "seq_len", default=60),
                "pred_len": config.get("model", "training", "pred_len", default=5),
                "epochs": config.get("model", "training", "epochs", default=10),
                "batch_size": config.get("model", "training", "batch_size", default=8),
                "learning_rate": config.get("model", "training", "learning_rate", default=1e-5),
            },
        },
        "param_search": {
            "T_values": config.get("param_search", "T_values", default=[0.6, 0.7, 0.8, 0.9, 1.0]),
            "top_p_values": config.get("param_search", "top_p_values", default=[0.85, 0.90, 0.95, 0.99, 1.0]),
            "lookback_values": config.get("param_search", "lookback_values", default=[60, 80, 100, 120]),
        },
        "server": {
            "host": config.get("server", "host", default="0.0.0.0"),
            "port": config.get("server", "port", default=8000),
        },
    }


@router.post("/config", dependencies=[Depends(require_admin)])
async def save_admin_config(config_data: dict):
    """
    保存用户配置（策略参数等）。

    当前版本：仅做校验并返回确认信息。
    实际持久化到 config/system.yaml 需要管理员权限，前端通过 localStorage 降级。
    """
    # 校验可配置的字段
    allowed_strategy_keys = {
        "buy_threshold", "long_only", "hold_days",
        "stop_loss", "take_profit", "max_single_position", "min_positions",
    }
    allowed_backtest_keys = {
        "default_capital", "commission_rate", "stamp_tax_rate",
        "slippage_rate", "benchmark_index", "risk_free_rate",
    }

    validated = {}

    if "strategy" in config_data:
        strat = config_data["strategy"]
        if isinstance(strat, dict):
            validated["strategy"] = {k: v for k, v in strat.items() if k in allowed_strategy_keys}

    if "backtest" in config_data:
        bt = config_data["backtest"]
        if isinstance(bt, dict):
            validated["backtest"] = {k: v for k, v in bt.items() if k in allowed_backtest_keys}

    logger.info(f"POST /admin/config → 已保存 {len(validated)} 个配置节")
    return {
        "success": True,
        "message": "配置已接收",
        "validated": validated,
    }


@router.get("/models", dependencies=[Depends(require_admin)])
async def get_registered_models():
    """List built-in and administrator-registered model assets."""
    from src.model_layer.model_registry import list_models

    return list_models()


@router.post("/models/register", status_code=201, dependencies=[Depends(require_admin)])
async def register_model(request: RegisterModelRequest):
    """Register complete model assets located under the project models directory."""
    from src.model_layer.model_registry import register_custom_model

    try:
        return register_custom_model(**request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/models/{model_key}/activate", dependencies=[Depends(require_admin)])
async def activate_registered_model(model_key: str):
    """Persist the active model and invalidate the in-process model cache."""
    from src.model_layer.model_registry import activate_model

    try:
        return activate_model(model_key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="model not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete("/models/{model_key}", dependencies=[Depends(require_admin)])
async def remove_registered_model(model_key: str):
    """Remove a custom registry entry without deleting its weight files."""
    from src.model_layer.model_registry import delete_custom_model

    try:
        delete_custom_model(model_key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="custom model not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"success": True}


# ═══════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════

def _get_last_training_time() -> Optional[str]:
    """获取最后一次训练时间。"""
    try:
        from src.data_layer.storage import StorageLayer
        from src.common.config import load_config

        config = load_config()
        storage = StorageLayer(config)
        rows = storage.execute_sql(
            "SELECT MAX(end_time) FROM training_log WHERE status = 'completed'"
        )
        if rows and rows[0] and rows[0][0]:
            return str(rows[0][0])
    except Exception:
        pass
        return None


def _latest_pool_count() -> int:
    try:
        from src.data_layer.storage import StorageLayer
        from src.common.config import load_config

        storage = StorageLayer(load_config())
        rows = storage.execute_sql(
            """SELECT COUNT(*) FROM stock_selection
               WHERE selected = 1 AND date = (SELECT MAX(date) FROM stock_selection)"""
        )
        return int(rows[0][0]) if rows else 0
    except Exception:
        return 0


def _get_last_prediction_time() -> Optional[str]:
    """获取最后一次预测时间。"""
    try:
        from src.data_layer.storage import StorageLayer
        from src.common.config import load_config

        config = load_config()
        storage = StorageLayer(config)
        rows = storage.execute_sql(
            "SELECT MAX(pred_date) FROM model_prediction"
        )
        if rows and rows[0] and rows[0][0]:
            return str(rows[0][0])
    except Exception:
        pass
    return None
