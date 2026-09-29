"""
Kronos 指数/ETF 择时策略系统 — 日志模块

统一日志格式: [时间] [级别] [模块:行号] 消息
支持同时输出到控制台和文件。
"""

import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


# 日志格式
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(module)s:%(lineno)d] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "kronos",
    level: str = "INFO",
    log_dir: Optional[str] = None,
    log_file: str = "kronos.log",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console: bool = True,
) -> logging.Logger:
    """
    初始化日志器。

    Args:
        name: 日志器名称
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
        log_dir: 日志目录，None则不写文件
        log_file: 日志文件名
        max_bytes: 单文件最大大小
        backup_count: 保留的备份文件数
        console: 是否输出到控制台

    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()  # 避免重复添加handler

    # 控制台handler
    if console:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(ch)

    # 文件handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, log_file)
        fh = RotatingFileHandler(
            file_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(fh)

    return logger


def get_logger(name: str = "kronos") -> logging.Logger:
    """获取已配置的日志器（如未初始化则返回默认logger）"""
    return logging.getLogger(name)


# 默认模块级logger
_default_logger = logging.getLogger("kronos")
_default_logger.addHandler(logging.NullHandler())
