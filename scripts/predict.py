#!/usr/bin/env python3
"""
Kronos 择时系统 — 批量预测推理 CLI

用法:
    # 使用 Kronos/fallback 模型进行批量预测
    python scripts/predict.py

    # 预测指定标的
    python scripts/predict.py --symbols 000852,512100,000300

    # 使用特定模型路径
    python scripts/predict.py --model fallback

    # 指定推理参数
    python scripts/predict.py --T 0.8 --top-p 0.95 --lookback 60

    # 从 HDF5 加载数据进行预测
    python scripts/predict.py --from-storage
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
        description="Kronos 择时系统 — 批量预测推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="逗号分隔的A股代码 (默认读取 config strategy.prediction_symbols；"
             "未配置时回退 paper_symbol+execution_symbol；绝不默认全A)",
    )
    parser.add_argument(
        "--all-a-share", action="store_true",
        help="显式声明预测全部A股（覆盖 prediction_symbols 默认范围）",
    )
    parser.add_argument(
        "--allow-domain-mismatch", action="store_true",
        help="即使激活模型的目标域不是全A股，也强制运行 --all-a-share（通常不推荐）",
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
        "--T", type=float, default=None,
        help="temperature 参数 (默认使用配置)",
    )
    parser.add_argument(
        "--top-p", type=float, default=None,
        help="top_p 参数",
    )
    parser.add_argument(
        "--lookback", type=int, default=None,
        help="回溯窗口天数",
    )
    parser.add_argument(
        "--samples", type=int, default=None,
        help="每只股票的采样路径数量 (默认使用配置)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="官方 Kronos GPU 批量推理大小 (默认使用配置)",
    )
    parser.add_argument(
        "--from-storage", action="store_true",
        help="从 HDF5 存储加载数据 (推荐)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="计算设备",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="输出目录",
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
    parser.add_argument(
        "--summary-only", action="store_true",
        help="不打印逐只股票结果，适用于自动调度",
    )

    args = parser.parse_args()

    setup_logger(level=args.log_level, log_dir="logs/", console=True)
    logger = get_logger("scripts.predict")

    config = load_config(args.config)
    set_seed(config.seed)

    # 解析标的
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    elif not args.all_a_share:
        # 默认只预测配置显式声明的范围（prediction_symbols），不再静默遍历全 A 股。
        cfg_symbols = config.get("strategy", "prediction_symbols", default=None)
        if cfg_symbols:
            symbols = [str(s) for s in cfg_symbols]
        else:
            symbols = sorted({
                str(config.get("strategy", "paper_symbol", default="000852")),
                str(config.get("strategy", "execution_symbol", default="512100")),
            })

    print("\n" + "=" * 60)
    print("Kronos 批量预测推理")
    print("=" * 60)

    # 目标域校验：--all-a-share 用指数择时模型跑全市场截面预测会产生无意义结果，
    # 且会向 model_prediction 灌入数万行垃圾预测（曾发生于 2026-09-03）。
    if args.all_a_share:
        from src.model_layer.model_registry import active_model_domain
        domain = active_model_domain()
        if domain != "all_a_share":
            print("")
            print("  ⚠ 目标域校验失败：当前激活模型不是全 A 股截面模型。")
            print(f"    激活模型目标域: {domain}")
            print("    全 A 股批量预测要求目标域 = all_a_share（如 new_all_a_share_hfq_v2）。")
            print("    指数择时模型只适用于其对齐的指数/ETF，对个股尺度与截面排序不可信。")
            if not args.allow_domain_mismatch:
                print("    如需强行运行，请加 --allow-domain-mismatch（不建议）。")
                print("=" * 60)
                sys.exit(2)
            print("    --allow-domain-mismatch 已指定，继续（后果自负）。")
        else:
            print(f"    目标域校验通过: {domain}")

    if symbols:
        print(f"  预测范围: {len(symbols)} 个标的 → {', '.join(symbols)}")
    else:
        print("  预测范围: 全部A股 (--all-a-share)")

    # 创建模型
    from src.model_layer.model_factory import create_model

    logger.info("加载模型...")
    model, path = create_model(
        config,
        device=args.device,
        force_path=args.model,
        pretrained_path=args.pretrained,
    )
    print(f"  模型: Path {'A (Kronos)' if path == 'kronos' else 'B (Fallback)'}")

    # 创建预测器
    from src.model_layer.kronos_predictor import KronosPredictor

    predictor = KronosPredictor(model, config)

    if args.T is not None:
        predictor.default_T = args.T
    if args.top_p is not None:
        predictor.default_top_p = args.top_p

    print(f"\n推理参数:")
    print(f"  T (temperature): {predictor.default_T}")
    print(f"  top_p:           {predictor.default_top_p}")
    print(f"  lookback:        {args.lookback or predictor.seq_len} 天")
    print(f"  pred_len:        {predictor.pred_len} 天")
    print()

    # 执行预测
    if args.from_storage:
        logger.info("从 HDF5 存储加载数据并预测...")
        predictions = predictor.predict_from_storage(
            symbols=symbols,
            lookback=args.lookback,
            T=args.T,
            top_p=getattr(args, 'top_p', None),
            sample_count=args.samples,
            batch_size=args.batch_size,
        )
    else:
        # 从 HDF5 加载原始数据
        from src.data_layer.storage import StorageLayer

        storage = StorageLayer(config)

        if symbols is None:
            symbols = storage.list_stock_symbols()

        symbol_dfs = {}
        for sym in symbols:
            try:
                df = storage.load_raw_data(sym)
                if df is not None and len(df) > 0:
                    symbol_dfs[sym] = df
                    if len(symbol_dfs) % 250 == 0:
                        logger.info("行情加载进度: %s/%s", len(symbol_dfs), len(symbols))
            except Exception as e:
                logger.warning(f"  {sym}: 加载失败 - {e}")

        predictions = predictor.predict_batch(
            symbol_dfs,
            lookback=args.lookback,
            T=args.T,
            top_p=args.top_p,
            sample_count=args.samples,
            batch_size=args.batch_size,
        )

    # 保存结果
    output_file = predictor.save_predictions(predictions, output_dir=args.output_dir)

    # 打印结果
    if not args.summary_only:
        print(f"\n{'代码':<10} {'最新收盘':>10} {'预测T+5':>10} {'预测收益率':>10} {'上涨概率':>8}")
        print("-" * 55)
        for p in predictions:
            code = p.get("symbol", "?")
            last = p.get("last_close", 0)
            pred = p.get("pred_close_t5", 0)
            ret = p.get("pred_return", 0)
            conf = p.get("confidence", 0)
            signal = "BUY" if ret > 0.005 else "CASH"
            print(f"{code:<10} {last:>10.2f} {pred:>10.2f} {ret:>9.2%} {conf:>7.0%}  → {signal}")

    successful = [
        item for item in predictions
        if not item.get("error") and item.get("pred_date") and item.get("symbol")
    ]
    failed = len(predictions) - len(successful)
    print(f"\n✓ 预测完成: 请求 {len(predictions)}，成功 {len(successful)}，失败 {failed}")
    print(f"  输出文件: {output_file}")
    print("=" * 60)

    sys.exit(0)


if __name__ == "__main__":
    main()
