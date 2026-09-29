#!/usr/bin/env python3
"""
Kronos 择时系统 — Path B Fallback 模型训练 CLI

用法:
    # 训练 Path B fallback 模型 (CPU, 12个标的)
    python scripts/train.py --model fallback

    # 指定 epochs 和标的
    python scripts/train.py --model fallback --epochs 10 --symbols 000852,512100

    # 从 checkpoint 继续训练
    python scripts/train.py --model fallback --pretrained models/checkpoints/fallback_best.pt

    # 使用 Path A (Kronos) 微调 (需要 GPU)
    python scripts/train.py --model kronos --epochs 5
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
        description="Kronos 择时系统 — 模型训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default="fallback",
        choices=["kronos", "fallback"],
        help="模型类型: kronos (Path A) / fallback (Path B)",
    )
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="逗号分隔的标的代码 (默认全部)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="训练 epochs (默认使用配置值)",
    )
    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="Path B 预训练权重路径 (用于继续训练)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="计算设备 (默认使用配置)",
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
    logger = get_logger("scripts.train")

    config = load_config(args.config)
    set_seed(config.seed)

    # 解析标的
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]

    print("\n" + "=" * 60)
    print(f"模型训练: Path {'A (Kronos)' if args.model == 'kronos' else 'B (Fallback)'}")
    print("=" * 60)

    # 准备训练数据
    from src.model_layer.kronos_loader import prepare_finetune_data

    logger.info("准备训练数据...")
    data = prepare_finetune_data(config, symbols=symbols)

    train_df = data.get("train")
    val_df = data.get("val")

    if train_df is None or len(train_df) == 0:
        logger.error("训练数据为空，请先执行数据下载和预处理")
        sys.exit(2)

    print(f"\n训练数据:")
    print(f"  标的数:  {len(data['train_symbols'])}")
    print(f"  训练集:  {len(train_df)} 行")
    if val_df is not None:
        print(f"  验证集:  {len(val_df)} 行")

    # 创建模型
    from src.model_layer.model_factory import create_model

    logger.info(f"创建模型: {args.model}")

    model, path = create_model(
        config,
        device=args.device,
        force_path=args.model,
        pretrained_path=args.pretrained,
    )

    # 微调/训练
    from src.model_layer.kronos_finetune import KronosFinetune

    finetune = KronosFinetune(model, config, model_type=args.model)

    if args.epochs is not None:
        finetune.epochs = args.epochs

    print(f"\n训练配置:")
    print(f"  epochs:    {finetune.epochs}")
    print(f"  batch_size:{finetune.batch_size}")
    print(f"  seq_len:   {finetune.seq_len}")
    print(f"  pred_len:  {finetune.pred_len}")
    print(f"  lr:        {finetune.lr}")
    print(f"  patience:  {finetune.patience}")
    print(f"  device:    {finetune.device}")
    print()

    result = finetune.run(
        train_df=train_df,
        val_df=val_df,
        checkpoint_dir=args.checkpoint_dir,
    )

    # 输出结果
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"  best_epoch:       {result.get('best_epoch', 'N/A')}")
    print(f"  best_val_loss:    {result.get('best_val_loss', 'N/A'):.6f}")
    print(f"  Spearman:         {result.get('spearman', 'N/A'):.4f}")
    print(f"  MAE:              {result.get('mae', 'N/A'):.2f}")
    print(f"  Direction Accuracy:{result.get('direction_accuracy', 'N/A'):.2%}")
    print(f"  总耗时:            {result.get('duration_seconds', 'N/A')} 秒")
    print(f"  Checkpoint:       {result.get('checkpoint_path', 'N/A')}")
    print("=" * 60)

    # 退出码
    if result.get("spearman", 0) > 0.5 or result.get("best_val_loss", float("inf")) < float("inf"):
        sys.exit(0)
    else:
        logger.warning("训练结果未达预期阈值")
        sys.exit(1)


if __name__ == "__main__":
    main()
