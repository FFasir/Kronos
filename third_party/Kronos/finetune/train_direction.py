#!/usr/bin/env python3
"""Train a discriminative 5-day direction head on top of Kronos.

Feature:  context = decode_s1(tokenizer.encode(window))[:, lookback-1, :]
(causal attention -> depends on history bars only).
Label:    1 if close[lookback+horizon-1] > close[lookback-1] else 0.
Loss:     BCEWithLogits on a small MLP head; backbone trains at low LR.

Single-GPU script driven by env vars (mirrors finetune config):
  KRONOS_PROJECT_ROOT, KRONOS_DATASET_PATH, KRONOS_SAVE_PATH,
  KRONOS_EXPERIMENT_ID, KRONOS_EPOCHS, KRONOS_BATCH_SIZE,
  KRONOS_TRAIN_SAMPLES, KRONOS_VAL_SAMPLES, KRONOS_NUM_WORKERS,
  KRONOS_SEED, KRONOS_LOOKBACK, KRONOS_PRED_LEN,
  KRONOS_TOKENIZER_PATH, KRONOS_MODEL_PATH
"""
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from dataset import QlibDataset


class DirectionHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, ctx):
        return self.net(ctx).squeeze(-1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.seed)

    from model import Kronos, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained(str(cfg.pretrained_tokenizer_path))
    tokenizer.eval().to(device)
    model = Kronos.from_pretrained(str(cfg.pretrained_predictor_path))
    model.train().to(device)
    print(f"Model loaded, d_model={model.d_model}, n_layers={model.n_layers}")

    head = DirectionHead(model.d_model).to(device)
    save_dir = Path(cfg.save_path) / cfg.predictor_save_folder_name
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    train_dataset = QlibDataset(data_type="train")
    valid_dataset = QlibDataset(data_type="val")
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                              num_workers=cfg.num_workers, drop_last=False)
    val_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size,
                            num_workers=cfg.num_workers, drop_last=False)
    print(f"[DIR] train={len(train_dataset)} windows/epoch, val={len(valid_dataset)}")

    param_groups = [
        {"params": head.parameters(), "lr": 5e-4},
        {"params": model.parameters(), "lr": 1e-5},
    ]
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[5e-4, 1e-5], steps_per_epoch=len(train_loader),
        epochs=cfg.epochs, pct_start=0.03, div_factor=10)

    loss_fn = nn.BCEWithLogitsLoss()
    best_val_acc = -1.0
    best_epoch = -1
    val_acc_history = []
    close_idx = 3  # window features: open, high, low, close, vol, amt
    lb = cfg.lookback_window

    for epoch in range(cfg.epochs):
        model.train()
        head.train()
        train_dataset.set_epoch_seed(epoch)
        valid_dataset.set_epoch_seed(0)
        t0 = time.time()
        tot_loss = 0.0
        for i, (batch_x, batch_x_stamp) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
            with torch.no_grad():
                tok0, tok1 = tokenizer.encode(batch_x, half=True)
            _, ctx = model.decode_s1(tok0, tok1, batch_x_stamp)
            feat = ctx[:, lb - 1, :]
            logit = head(feat)
            closes = batch_x[..., close_idx]
            y = (closes[:, -1] > closes[:, lb - 1]).float()
            loss = loss_fn(logit, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()
            tot_loss += loss.item() * batch_x.size(0)
            if (i + 1) % 100 == 0:
                print(f"[E{epoch + 1}/{cfg.epochs} S{i + 1}/{len(train_loader)}] loss={loss.item():.4f}", flush=True)

        # validation: BCE + direction accuracy
        model.eval(); head.eval()
        v_loss = 0.0; n = 0; correct = 0
        with torch.no_grad():
            for batch_x, batch_x_stamp in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
                tok0, tok1 = tokenizer.encode(batch_x, half=True)
                _, ctx = model.decode_s1(tok0, tok1, batch_x_stamp)
                logit = head(ctx[:, lb - 1, :])
                closes = batch_x[..., close_idx]
                y = (closes[:, -1] > closes[:, lb - 1]).float()
                v_loss += loss_fn(logit, y).item() * batch_x.size(0)
                pred = (logit > 0).float()
                correct += (pred == y).sum().item()
                n += batch_x.size(0)
        v_acc = correct / max(n, 1)
        val_acc_history.append(v_acc)
        print(f"--- Epoch {epoch + 1}/{cfg.epochs} --- val BCE={v_loss / max(n, 1):.4f} "
              f"val dir-acc={v_acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_epoch = epoch + 1
            torch.save({
                "model": model.state_dict(),
                "head": head.state_dict(),
                "best_val_acc": v_acc,
            }, str(save_dir / "checkpoints" / "best_model.pt"))
            print(f"best model saved (val acc {v_acc:.4f})", flush=True)

    summary = {
        "best_val_acc": best_val_acc, "best_epoch": best_epoch,
        "val_acc_history": val_acc_history,
        "lookback": lb, "horizon": cfg.predict_window,
        "train_pool": len(train_dataset.indices) if hasattr(train_dataset, "indices") else None,
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
