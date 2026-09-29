#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日志查看器

用法: python scripts/tools/tail_logs.py [--lines 20]

聚合显示 logs/ 目录下所有日志文件的尾部内容
"""
import sys
import io
import argparse
from pathlib import Path
from datetime import datetime

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

_ROOT = Path(__file__).resolve().parent.parent.parent


def tail(path: Path, n: int) -> list:
    """读取文件最后 n 行（UTF-8，容错解码）"""
    try:
        with open(path, "rb") as f:
            data = f.read()
        lines = data.decode("utf-8", errors="replace").splitlines()
        return lines[-n:]
    except Exception as e:
        return [f"<读取失败: {e}>"]


def main():
    parser = argparse.ArgumentParser(description="聚合查看日志尾部")
    parser.add_argument("--lines", type=int, default=20, help="每个文件显示的行数（默认20）")
    args = parser.parse_args()

    log_dir = _ROOT / "logs"
    if not log_dir.exists():
        print(f"日志目录不存在: {log_dir}")
        return 1

    log_files = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not log_files:
        print(f"logs/ 下没有 .log 文件")
        return 0

    print("=" * 70)
    print(f"Kronos 日志查看器 (每文件末 {args.lines} 行)")
    print("=" * 70)

    for path in log_files:
        size_kb = path.stat().st_size / 1024
        mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n─── {path.name} ({size_kb:.1f} KB, 更新于 {mtime}) " + "─" * 20)
        for line in tail(path, args.lines):
            print(f"  {line}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
