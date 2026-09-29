"""
Path B Fallback 模型训练 (带 flush, 适合后台运行)
用法: python scripts/train_fallback.py
"""
import sys, os, time
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd, numpy as np
from src.common.config import load_config
from src.common.utils import detect_device, set_seed
from src.model_layer.fallback_model import FallbackModel, FallbackTrainer

LOG_FILE = str(_PROJECT_ROOT / "logs" / "train_fallback.log")
os.makedirs(str(_PROJECT_ROOT / "logs"), exist_ok=True)

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

INDICES = ["000852","000300","000905","399006","000688","000016"]
NAMES = {"000852":"中证1000","000300":"沪深300","000905":"中证500","399006":"创业板指","000688":"科创50","000016":"上证50"}

def main():
    config = load_config()
    set_seed(config.seed)

    # ── 1. 加载数据 ──
    log("=" * 50)
    log("步骤 1/3: 加载训练数据")
    raw_dir = Path(config.data.get("raw_dir", "data/raw"))
    train_start = config.data.get("finetune_train_start", "2010-01-01")
    train_end   = config.data.get("finetune_train_end",   "2022-12-31")
    val_start   = config.data.get("finetune_val_start",   "2023-01-01")
    val_end     = config.data.get("finetune_val_end",     "2024-12-31")

    train_frames, val_frames = [], []
    for code in INDICES:
        fp = raw_dir / f"{code}_{NAMES[code]}_daily.csv"
        if not fp.exists(): continue
        df = pd.read_csv(fp)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df[["open","high","low","close","volume"]]
        t = df[(df.index >= train_start) & (df.index <= train_end)]
        v = df[(df.index >= val_start) & (df.index <= val_end)]
        if len(t) > 0: train_frames.append(t)
        if len(v) > 0: val_frames.append(v)
        log(f"  {code} {NAMES[code]}: train={len(t)}, val={len(v)}")

    train_df = pd.concat(train_frames).sort_index()
    val_df = pd.concat(val_frames).sort_index()
    log(f"训练集: {len(train_df)} 行 | 验证集: {len(val_df)} 行")

    # ── 2. 创建模型 ──
    log("=" * 50)
    log("步骤 2/3: 创建模型")
    device = detect_device(config.get("model", "device", default="auto"))
    log(f"设备: {device}")

    fb = config.get("model", "fallback", default={})
    model = FallbackModel(
        d_model=fb.get("d_model", 256), n_layers=fb.get("n_layers", 6),
        n_heads=fb.get("n_heads", 8), d_ff=fb.get("d_ff", 1024),
        dropout=fb.get("dropout", 0.1), max_seq_len=fb.get("max_seq_len", 512),
        input_dim=5, pred_len=config.get("model", "training", "pred_len", default=5),
    )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"FallbackModel 参数: {n_params:,}")

    # ── 3. 训练 ──
    log("=" * 50)
    log("步骤 3/3: 开始训练")
    log(f"epochs=50, seq_len=60, pred_len=5, batch_size=4, early_stop_patience=10")
    t0 = time.time()

    trainer = FallbackTrainer(model, config, device)
    checkpoint_dir = str(_PROJECT_ROOT / "models" / "checkpoints")

    # 重定向 print → 同时写文件和 stdout (FallbackTrainer 内部用 logger)
    result = trainer.train(train_df, val_df, checkpoint_dir)

    elapsed = time.time() - t0
    log("=" * 50)
    log(f"训练完成! 耗时: {elapsed/60:.1f} 分钟")
    log(f"  最优 loss: {result['best_loss']:.6f}")
    log(f"  最优 epoch: {result['best_epoch']}/{result['total_epochs']}")
    log(f"  Checkpoint: {checkpoint_dir}/fallback_best.pt")
    log("=" * 50)

if __name__ == "__main__":
    main()
