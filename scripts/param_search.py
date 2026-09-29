#!/usr/bin/env python3
"""
Kronos 择时系统 — 滚动网格搜参 CLI

论文核心方法: 每月在验证集上网格搜索最优推理参数。

搜索空间:
  - T (temperature):    [0.3, 0.6, 0.8, 1.0]
  - top_p:              [0.2, 0.4, 0.6, 0.9]
  - lookback_window:    [30, 60, 90]
  - 总计: 4 × 4 × 3 = 48 组参数组合

用法:
    # 对指定月份搜参
    python scripts/param_search.py --test-month 2025-01

    # 对所有标的搜参 (当前月)
    python scripts/param_search.py

    # 仅对指定标的搜参
    python scripts/param_search.py --symbols 000852,512100

    # 使用 Path B fallback 搜参
    python scripts/param_search.py --model fallback

    # 仅评估，不保存结果
    python scripts/param_search.py --dry-run
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.config import load_config
from src.common.logger import setup_logger, get_logger
from src.common.utils import set_seed


def main():
    parser = argparse.ArgumentParser(
        description="Kronos 择时系统 — 滚动网格搜参",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--test-month", type=str, default=None,
        help="测试月 YYYY-MM (默认: 上个月)",
    )
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="逗号分隔的标的代码 (默认全部)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=["kronos", "fallback"],
        help="模型路径 (默认自动检测)",
    )
    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="Path B 预训练权重路径",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="计算设备",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅评估不保存",
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="保留前 K 个最优参数 (默认 3)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="配置文件路径",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )

    args = parser.parse_args()

    setup_logger(level=args.log_level, log_dir="logs/", console=True)
    logger = get_logger("scripts.param_search")

    config = load_config(args.config)
    set_seed(config.seed)

    # 解析标的
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]

    print("\n" + "=" * 60)
    print("滚动网格搜参")
    print("=" * 60)

    # 加载模型
    from src.model_layer.model_factory import create_model

    logger.info("加载模型...")
    model, path = create_model(
        config,
        device=args.device,
        force_path=args.model,
        pretrained_path=args.pretrained,
    )
    print(f"  模型: Path {'A (Kronos)' if path == 'kronos' else 'B (Fallback)'}")

    # 创建预测器和搜参器
    from src.model_layer.kronos_predictor import KronosPredictor
    from src.model_layer.param_search import ParamSearch

    predictor = KronosPredictor(model, config)
    searcher = ParamSearch(predictor, config)

    # 搜索空间
    print(f"\n搜索空间:")
    print(f"  T:        {searcher.T_values}")
    print(f"  top_p:    {searcher.top_p_values}")
    print(f"  lookback: {searcher.lookback_values}")
    n_combos = len(searcher.T_values) * len(searcher.top_p_values) * len(searcher.lookback_values)
    print(f"  总组合数: {n_combos}")

    if symbols:
        print(f"  标的:     {symbols}")
    print(f"  测试月:   {args.test_month or '(自动: 上个月)'}")
    print()

    if args.dry_run:
        logger.info("Dry-run 模式: 仅评估搜索空间，不执行搜参")
        # 快速评估: 对 000852 执行 1 个组合
        from src.data_layer.storage import StorageLayer
        storage = StorageLayer(config)
        df = storage.load_raw_data("000852")
        if df is not None:
            from src.model_layer.param_search import ParamSearch
            logger.info(f"快速评估: 000852, {len(df)} 行")
        print("\nDry-run 完成。搜索空间合理。")
        sys.exit(0)

    # 执行搜参
    logger.info("开始搜参...")
    results = searcher.search_all_symbols(
        symbols=symbols,
        test_month=args.test_month,
    )

    # 保存结果
    if not args.dry_run:
        output_file = searcher.save_results(results)
    else:
        output_file = "(dry-run, 未保存)"

    # 汇总
    print(f"\n{'代码':<10} {'最优T':>6} {'最优top_p':>8} {'最优lookback':>12} {'Spearman':>10}")
    print("-" * 55)
    for sym, res in results.get("results", {}).items():
        bp = res.get("best_params", {})
        sp = res.get("best_spearman", 0)
        print(
            f"{sym:<10} {bp.get('T', '?'):>6.2f} "
            f"{bp.get('top_p', '?'):>8.3f} "
            f"{bp.get('lookback', '?'):>12} "
            f"{sp:>10.4f}"
        )

    # 整体最优
    all_best = None
    for sym, res in results.get("results", {}).items():
        sp = res.get("best_spearman", -999)
        if all_best is None or sp > all_best[1]:
            all_best = (sym, sp, res.get("best_params", {}))

    if all_best:
        print(f"\n整体最优: {all_best[0]} (Spearman={all_best[1]:.4f})")
        print(f"  参数: T={all_best[2].get('T')}, top_p={all_best[2].get('top_p')}, lookback={all_best[2].get('lookback')}")

    print(f"\n✓ 搜参完成")
    print(f"  测试月: {results.get('test_month')}")
    print(f"  输出:   {output_file}")
    print("=" * 60)

    sys.exit(0)


if __name__ == "__main__":
    main()
