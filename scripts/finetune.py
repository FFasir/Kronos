#!/usr/bin/env python3
"""
Kronos 择时系统 — Kronos-base 微调 CLI

论文方法: 在中证1000成分股/指数ETF日线数据上微调 Kronos-base。

用法:
    # 微调 Kronos-base (GPU 推荐)
    python scripts/finetune.py

    # 指定 epochs 和数据范围
    python scripts/finetune.py --epochs 5 --symbols 000852,000300

    # 仅用 ETF 数据微调
    python scripts/finetune.py --symbols 512100,510300,510500

    # 从 checkpoint 恢复
    python scripts/finetune.py --resume models/checkpoints/finetune_epoch_3_0.1234.pt

    # 使用成分股数据 (数据量大，需 GPU)
    python scripts/finetune.py --use-components
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
        description="Kronos 择时系统 — Kronos-base 微调",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="逗号分隔的标的代码 (默认使用配置中的12个标的)",
    )
    parser.add_argument(
        "--use-components", action="store_true",
        help="使用中证1000成分股数据 (数据量大，需 GPU)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="训练 epochs (默认使用配置值: 10)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="batch size (默认使用配置值)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="计算设备 (默认使用配置)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="从 checkpoint 恢复训练",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="models/checkpoints/",
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
    logger = get_logger("scripts.finetune")

    config = load_config(args.config)
    set_seed(config.seed)

    # 解析标的
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]

    print("\n" + "=" * 60)
    print("Kronos-base 微调")
    print("=" * 60)

    # 加载模型
    from src.model_layer.model_factory import create_model

    logger.info("加载模型...")

    if args.resume:
        # 从 checkpoint 恢复
        from src.model_layer.fallback_model import FallbackModel
        import torch
        from src.common.utils import detect_device

        device_obj = detect_device(args.device or config.get("model", "device", default="auto"))
        checkpoint = torch.load(args.resume, map_location=device_obj, weights_only=False)
        logger.info(f"从 checkpoint 恢复: {args.resume}")

        # 先尝试作为 Kronos 加载
        try:
            model, path = create_model(config, device=args.device, force_path="kronos")
            if hasattr(model, 'model'):
                model.model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
            logger.info("Kronos 模型权重已加载")
        except Exception:
            # 降级到 Fallback
            model, path = create_model(config, device=args.device, force_path="fallback")
            if hasattr(model, 'load_state_dict'):
                model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
            logger.info("Fallback 模型权重已加载")
    else:
        model, path = create_model(config, device=args.device)

    print(f"  模型路径: Path {'A (Kronos)' if path == 'kronos' else 'B (Fallback)'}")

    if getattr(model, "is_native_kronos", False):
        logger.error("官方 Kronos 不支持本项目旧的 raw-OHLCV 微调循环。")
        logger.error("请使用官方仓库的 finetune/qlib_data_preprocess.py、train_tokenizer.py 和 train_predictor.py。")
        logger.error("详见 docs/云服务器训练指南.md 的官方微调章节。")
        sys.exit(2)

    # 准备数据
    from src.model_layer.kronos_loader import prepare_finetune_data

    logger.info("准备训练数据...")

    if args.use_components:
        logger.info("使用 data/processed/components.h5 中的股票数据")
        logger.info("若文件不存在，请先运行 scripts/import_stock_archive.py")

    data = prepare_finetune_data(config, symbols=symbols, use_components=args.use_components)

    train_df = data.get("train")
    val_df = data.get("val")

    if train_df is None or len(train_df) == 0:
        logger.error("训练数据为空！请先执行:")
        logger.error("  python scripts/download.py --mode full")
        logger.error("  python scripts/preprocess.py")
        sys.exit(2)

    print(f"\n训练数据:")
    print(f"  标的数:  {len(data['train_symbols'])}")
    print(f"  训练集:  {len(train_df)} 行")
    if val_df is not None:
        print(f"  验证集:  {len(val_df)} 行")
    print(f"  训练范围: {train_df.index.min()} ~ {train_df.index.max()}")

    # 微调
    from src.model_layer.kronos_finetune import KronosFinetune

    finetune = KronosFinetune(model, config, model_type=path)

    if args.epochs is not None:
        finetune.epochs = args.epochs
    if args.batch_size is not None:
        finetune.batch_size = args.batch_size

    print(f"\n微调配置:")
    print(f"  epochs:      {finetune.epochs}")
    print(f"  batch_size:  {finetune.batch_size}")
    print(f"  seq_len:     {finetune.seq_len}")
    print(f"  pred_len:    {finetune.pred_len}")
    print(f"  lr:          {finetune.lr}")
    print(f"  weight_decay:{finetune.weight_decay}")
    print(f"  patience:    {finetune.patience}")
    print()

    result = finetune.run(
        train_df=train_df,
        val_df=val_df,
        checkpoint_dir=args.checkpoint_dir,
    )

    # 输出结果
    print("\n" + "=" * 60)
    print("微调完成!")
    print("=" * 60)
    print(f"  Best Epoch:        {result.get('best_epoch', 'N/A')}")
    print(f"  Best Val Loss:     {result.get('best_val_loss', 'N/A'):.6f}")
    print(f"  Spearman ρ:        {result.get('spearman', 'N/A'):.4f}  (论文: 0.856)")
    print(f"  MAE:               {result.get('mae', 'N/A'):.2f}     (论文: 275.5)")
    print(f"  Direction Accuracy: {result.get('direction_accuracy', 'N/A'):.2%}")
    print(f"  Total Epochs:      {result.get('total_epochs', 'N/A')}")
    print(f"  Duration:          {result.get('duration_seconds', 'N/A')} 秒")
    print(f"  Checkpoint:        {result.get('checkpoint_path', 'N/A')}")
    print("=" * 60)

    # 对标论文
    spearman = result.get("spearman", 0)
    if spearman > 0.70:
        print(f"\n✓ Spearman {spearman:.4f} > 0.70 (可接受范围)")
    elif spearman > 0.50:
        print(f"\n⚠ Spearman {spearman:.4f} 在 0.50-0.70 之间 (需优化)")
    else:
        print(f"\n✗ Spearman {spearman:.4f} < 0.50 (需检查)")

    sys.exit(0 if spearman > 0.40 else 1)


if __name__ == "__main__":
    main()
