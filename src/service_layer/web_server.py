"""
Kronos 指数/ETF 择时策略系统 — FastAPI Web 服务入口

启动方式:
  开发: uvicorn src.service_layer.web_server:app --reload --port 8000
  生产: uvicorn src.service_layer.web_server:app --host 0.0.0.0 --port 8000 --workers 2

访问:
  - React SPA 入口: http://localhost:8000                    （→ frontend/dist/index.html）
  - 股民仪表盘:     http://localhost:8000/user/dashboard
  - 管理后台:       http://localhost:8000/admin/monitor
  - 旧版原型看板:   http://localhost:8000/user_portal.html  （保留作为参考）
  - Swagger 文档:   http://localhost:8000/docs
  - ReDoc:          http://localhost:8000/redoc
  - API 基础路径:   http://localhost:8000/api/v1/

SPA fallback: 所有非 /api/* GET 请求返回 frontend/dist/index.html，
由 React Router 处理客户端路由。
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, RedirectResponse

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# React SPA 构建输出目录
_FRONTEND_DIST = _PROJECT_ROOT / "frontend" / "dist"

# 服务启动时间（用于 /health uptime 计算）
import time as _time_module
_SERVER_START_TIME = _time_module.time()

from src.common.config import load_config, PROJECT_ROOT
from src.common.logger import get_logger, setup_logger
from src.common.exceptions import KronosError

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 应用生命周期
# ═══════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期管理：启动时初始化资源，关闭时释放。"""
    # ── 启动 ──
    logger.info("=" * 60)
    logger.info("  Kronos 择时策略系统 — Web 服务启动")
    logger.info("=" * 60)

    # 加载配置
    try:
        config = load_config()
        host = config.get("server", "host", default="0.0.0.0")
        port = config.get("server", "port", default=8000)
        logger.info(f"  配置加载成功: {host}:{port}")
        logger.info(f"  数据源: {config.get('data', 'data_source', default='akshare')}")
        logger.info(f"  模型版本: {config.get('model', 'kronos_version', default='base')}")
        logger.info(f"  随机种子: {config.seed}")
    except Exception as e:
        logger.warning(f"  配置加载失败（使用默认值）: {e}")

    # 检查模型可用性
    try:
        from src.model_layer.model_factory import detect_model_path
        if 'config' in dir():
            model_path = detect_model_path(config)
            logger.info(f"  模型层就绪，检测到模型路径: {model_path}")
    except Exception as e:
        logger.warning(f"  模型层未就绪: {e}")

    # 检查数据存储
    try:
        from src.data_layer.storage import StorageLayer

        db_path = StorageLayer(config).init_database()
        logger.info(f"  SQLite 数据库就绪: {db_path}")
    except Exception:
        pass

    # 静态文件目录
    static_dir = _PROJECT_ROOT / "界面原型"
    js_dir = static_dir / "js"
    if static_dir.exists():
        logger.info(f"  静态文件目录（原型）: {static_dir}")
    if js_dir.exists():
        logger.info(f"  JS 模块目录（原型）: {js_dir}")

    # React SPA 构建输出
    if _FRONTEND_DIST.exists():
        logger.info(f"  React SPA 目录: {_FRONTEND_DIST}")
    else:
        logger.warning(f"  React SPA 未构建（运行 cd frontend && npm run build）")

    task_scheduler = None
    try:
        if config.get("scheduler", "enabled", default=True):
            from src.scheduler.task_scheduler import get_task_scheduler

            task_scheduler = get_task_scheduler()
            task_scheduler.start()
    except Exception as e:
        logger.warning(f"  定时任务启动失败: {e}")

    logger.info(f"  API 文档: http://localhost:8000/docs")
    logger.info(f"  SPA 入口: http://localhost:8000")
    logger.info(f"  原型看板: http://localhost:8000/user_portal.html")
    logger.info(f"  管理后台: http://localhost:8000/admin/login")
    logger.info("=" * 60)

    yield  # ← 服务运行中

    # ── 关闭 ──
    if task_scheduler is not None:
        task_scheduler.shutdown()
    logger.info("Kronos Web 服务关闭")


# ═══════════════════════════════════════════════════════════════════
# FastAPI 应用实例
# ═══════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Kronos 指数/ETF 择时策略系统",
    description="""
基于 Kronos 金融大模型的 A 股择时策略系统 API。

## 功能模块

- **指数/ETF 数据** — 12 个标的的基本信息、K线数据查询
- **预测数据** — Kronos 模型价格预测、预测准确率统计
- **择时信号** — BUY/CASH/HOLD/SELL 信号查询
- **策略回测** — 异步回测任务提交、进度查询、结果展示
- **管理后台** — 系统监控、训练日志、数据源管理

## 设计说明

- API 基础路径: `/api/v1/`
- 所有端点支持 mock 数据降级（当底层数据/模型不可用时）
- 回测为异步执行，提交后通过 task_id 轮询结果
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════════════════════════
# CORS 中间件
# ═══════════════════════════════════════════════════════════════════

_cors_origins = [
    origin.strip()
    for origin in os.getenv("KRONOS_CORS_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=bool(_cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Task-Id", "X-Request-Id"],
)


# ═══════════════════════════════════════════════════════════════════
# 全局异常处理
# ═══════════════════════════════════════════════════════════════════

@app.exception_handler(KronosError)
async def kronos_exception_handler(request: Request, exc: KronosError):
    """处理 Kronos 自定义异常，返回结构化 JSON 错误。"""
    status_code = getattr(exc, "status_code", 500)
    logger.error(f"KronosError [{exc.error_code}]: {exc}")
    return JSONResponse(
        status_code=status_code,
        content={
            "error": True,
            "error_code": exc.error_code,
            "message": str(exc),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """处理未捕获的通用异常。"""
    logger.error(f"未处理异常 [{request.method} {request.url.path}]: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "error_code": "E500",
            "message": "服务器内部错误",
            "detail": str(exc) if app.debug else None,
        },
    )


# ═══════════════════════════════════════════════════════════════════
# 请求日志中间件
# ═══════════════════════════════════════════════════════════════════

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录所有 HTTP 请求的简要日志。"""
    import time
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    logger.debug(f"{request.method} {request.url.path} → {response.status_code} ({elapsed:.3f}s)")
    return response


# ═══════════════════════════════════════════════════════════════════
# 注册 API 路由
# ═══════════════════════════════════════════════════════════════════

from src.service_layer.api_index import router as index_router
from src.service_layer.api_prediction import router as prediction_router
from src.service_layer.api_signal import router as signal_router
from src.service_layer.api_backtest import router as backtest_router
from src.service_layer.api_admin import router as admin_router
from src.service_layer.api_selection import router as selection_router
from src.service_layer.api_research import router as research_router
from src.service_layer.api_quote import router as quote_router
from src.service_layer.api_strategy import router as strategy_router

API_PREFIX = "/api/v1"

app.include_router(index_router, prefix=API_PREFIX)
app.include_router(prediction_router, prefix=API_PREFIX)
app.include_router(signal_router, prefix=API_PREFIX)
app.include_router(backtest_router, prefix=API_PREFIX)
app.include_router(admin_router, prefix=API_PREFIX)
app.include_router(selection_router, prefix=API_PREFIX)
app.include_router(research_router, prefix=API_PREFIX)
app.include_router(quote_router, prefix=API_PREFIX)
app.include_router(strategy_router, prefix=API_PREFIX)

logger.info("API 路由注册完成: 5 个模块")


# ═══════════════════════════════════════════════════════════════════
# 静态文件服务
# ═══════════════════════════════════════════════════════════════════

_STATIC_DIR = _PROJECT_ROOT / "界面原型"
_JS_DIR = _STATIC_DIR / "js"

# 挂载原型 /js/ 目录（保留界面原型参考）
if _JS_DIR.exists():
    app.mount("/js", StaticFiles(directory=str(_JS_DIR)), name="js")
    logger.info(f"静态 JS 挂载（原型）: /js → {_JS_DIR}")

# 挂载 React SPA 构建产物 /assets/ 目录
if _FRONTEND_DIST.exists():
    _ASSETS_DIR = _FRONTEND_DIST / "assets"
    if _ASSETS_DIR.exists():
        app.mount("/assets", StaticFiles(directory=str(_ASSETS_DIR)), name="frontend_assets")
        logger.info(f"静态资源挂载（SPA）: /assets → {_ASSETS_DIR}")


@app.get("/user_portal.html", response_class=HTMLResponse, include_in_schema=False)
async def serve_user_portal():
    """股民策略看板入口（旧版原型，保留作为参考）。"""
    portal_path = _STATIC_DIR / "user_portal.html"
    if portal_path.exists():
        return HTMLResponse(content=portal_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>user_portal.html 不存在</h1>", status_code=404)


@app.get("/admin_portal.html", include_in_schema=False)
async def serve_admin_portal():
    """旧版后台入口统一转到受登录守卫保护的新入口。"""
    return RedirectResponse(url="/admin/login", status_code=307)


@app.get("/", include_in_schema=False)
async def root():
    """根路径 → React SPA 入口。"""
    if _FRONTEND_DIST.exists():
        index_path = _FRONTEND_DIST / "index.html"
        if index_path.exists():
            return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    # 降级：无 SPA 时返回 API 信息
    return {"message": "Kronos 择时策略系统 API", "docs": "/docs", "portal": "/user_portal.html"}


# ═══════════════════════════════════════════════════════════════════
# 健康检查（必须在 SPA fallback 之前注册）
# ═══════════════════════════════════════════════════════════════════

@app.get("/health", include_in_schema=False)
async def health_check():
    """健康检查端点。"""
    from datetime import datetime

    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": int(_time_module.time() - _SERVER_START_TIME),
    }


@app.get("/api/v1/system/status", include_in_schema=False)
async def system_status():
    """免登录的系统状态摘要，供股民端仪表盘展示。

    仅暴露非敏感字段：模型/数据状态、数据更新时间、模型版本、预测覆盖等。
    """
    from src.service_layer.api_admin import compute_system_monitor, compute_prediction_coverage

    m = compute_system_monitor()
    coverage = compute_prediction_coverage()
    return {
        "model_status": m.model_status,
        "data_status": m.data_status,
        "model_version": m.model_version,
        "last_data_update": m.last_data_update,
        "last_prediction_time": m.last_prediction_time,
        "prediction_coverage": coverage,
    }


# ═══════════════════════════════════════════════════════════════════
# SPA 路由 fallback（必须放在所有显式路由之后，最后注册）
# ═══════════════════════════════════════════════════════════════════

@app.get("/{path:path}", include_in_schema=False)
async def spa_fallback(path: str):
    """
    SPA 路由 fallback — 非 /api/* /docs /redoc 的 GET 请求返回 index.html。

    由于 FastAPI 路由优先级机制，已注册的路由（/api/v1/*、/docs、/redoc、
    /health、/user_portal.html 等）会优先匹配，不会进入此 catch-all。
    仅 SPA 前端路由（如 /user/dashboard、/admin/monitor 等）会到达这里。
    """
    # 跳过 API 路径（理论上不会到达，保留作为安全网）
    if path.startswith("api/"):
        return JSONResponse(
            status_code=404,
            content={"error": True, "error_code": "E404", "message": f"API 端点不存在: /{path}"},
        )

    # 返回 SPA 入口 HTML
    if _FRONTEND_DIST.exists():
        index_path = _FRONTEND_DIST / "index.html"
        if index_path.exists():
            return HTMLResponse(content=index_path.read_text(encoding="utf-8"))

    return HTMLResponse(
        content="<h1>前端未构建</h1><p>请运行 <code>cd frontend && npm run build</code></p>",
        status_code=404,
    )


# ═══════════════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════════════

def main():
    """CLI 启动入口。"""
    import uvicorn
    from src.common.config import load_config

    try:
        config = load_config()
        host = config.get("server", "host", default="0.0.0.0")
        port = config.get("server", "port", default=8000)
        reload = config.get("server", "reload", default=False)
    except Exception:
        host, port, reload = "0.0.0.0", 8000, False

    uvicorn.run(
        "src.service_layer.web_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
