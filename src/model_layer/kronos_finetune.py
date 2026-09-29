"""
Kronos 指数/ETF 择时策略系统 — Kronos 模型微调

按论文方法在成分股/指数数据上微调 Kronos-base：
- 训练集: 2010-01 ~ 2022-12 日线 OHLCV
- 验证集: 2023-01 ~ 2024-12
- 优化器: AdamW, lr=1e-5
- 调度器: CosineAnnealingLR
- 早停: patience=3 (Path A) / patience=5 (配置)
- 监控指标: 验证集 Spearman 相关系数 + MAE
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from src.common.logger import get_logger
from src.common.exceptions import ModelLoadError, ConvergenceError
from src.common.utils import set_seed

logger = get_logger(__name__)


class KronosFinetune:
    """
    Kronos-base 微调器。

    在成分股/ETF 日线数据上使用 Kronos 内置交叉熵 loss 微调模型。
    支持 Path A (Kronos 真实模型) 和降级 Path B (FallbackModel)。
    """

    def __init__(
        self,
        model,
        config,
        device: torch.device = None,
        model_type: str = "kronos",
    ):
        """
        Args:
            model: KronosWrapper 或 FallbackModel 实例
            config: 系统配置
            device: 计算设备
            model_type: "kronos" | "fallback"
        """
        self.model = model
        self.config = config
        self.model_type = model_type

        from src.common.utils import detect_device
        self.device = device or detect_device(config.get("model", "device", default="auto"))

        # 训练超参数
        self.epochs = config.get("model", "training", "epochs", default=10)
        self.batch_size = config.get("model", "training", "batch_size", default=8)
        self.seq_len = config.get("model", "training", "seq_len", default=60)
        self.pred_len = config.get("model", "training", "pred_len", default=5)
        self.lr = config.get("model", "training", "learning_rate", default=1e-5)
        self.weight_decay = config.get("model", "training", "weight_decay", default=1e-5)
        self.grad_clip = config.get("model", "training", "grad_clip", default=1.0)
        self.patience = config.get("model", "training", "early_stop_patience", default=5)

        # 获取模型的实际参数
        if hasattr(model, "model"):
            self._inner_model = model.model
        else:
            self._inner_model = model

    def _prepare_samples(
        self, df, seq_len: int, pred_len: int, max_samples: int = None
    ) -> tuple:
        """
        从 DataFrame 构建训练样本 (numpy array)。

        Args:
            df: 含 OHLCV 列的 DataFrame
            seq_len: 输入窗口
            pred_len: 预测步长
            max_samples: 最大样本数 (None = 不限)

        Returns:
            (X_np, y_np): X (N, seq_len, 5), y (N, pred_len)
        """
        X_list, y_list = [], []
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        # A merged DataFrame contains several stocks.  Samples must be created
        # within each stock series; otherwise a window can cross from one stock
        # directly into the next and corrupt both input and target.
        groups = df.groupby("symbol", sort=False) if "symbol" in df.columns else [(None, df)]
        for _, symbol_df in groups:
            data = symbol_df.sort_index()[ohlcv_cols].values.astype(np.float32)
            total = len(data) - seq_len - pred_len + 1
            if total <= 0:
                continue
            step = max(1, total // max_samples) if max_samples and total > max_samples else 1
            for i in range(0, total, step):
                X_list.append(data[i:i + seq_len])
                # y: 未来 pred_len 天的收盘价
                y_list.append(data[i + seq_len:i + seq_len + pred_len, 3])

        if len(X_list) == 0:
            return np.empty((0, seq_len, 5), dtype=np.float32), np.empty((0, pred_len), dtype=np.float32)

        return np.stack(X_list), np.stack(y_list)

    def _compute_metrics(
        self, preds: np.ndarray, targets: np.ndarray
    ) -> Dict[str, float]:
        """
        计算评估指标。

        Args:
            preds: 预测收盘价 (N,)
            targets: 真实收盘价 (N,)

        Returns:
            {"spearman": float, "mae": float, "direction_accuracy": float}
        """
        from scipy.stats import spearmanr

        # MAE
        mae = float(np.mean(np.abs(preds - targets)))

        # Spearman 相关系数
        if len(preds) > 2:
            spearman_corr, _ = spearmanr(preds, targets)
            spearman_corr = float(spearman_corr) if not np.isnan(spearman_corr) else 0.0
        else:
            spearman_corr = 0.0

        # 方向准确率
        if len(preds) > 1:
            pred_diff = np.diff(preds.flatten() if preds.ndim > 1 else preds)
            true_diff = np.diff(targets.flatten() if targets.ndim > 1 else targets)
            if len(pred_diff) > 0:
                dir_acc = float(np.mean(np.sign(pred_diff) == np.sign(true_diff)))
            else:
                dir_acc = 0.0
        else:
            dir_acc = 0.0

        return {
            "spearman": spearman_corr,
            "mae": mae,
            "direction_accuracy": dir_acc,
        }

    def run(
        self,
        train_df,
        val_df=None,
        checkpoint_dir: str = "models/checkpoints/",
        max_train_samples: int = None,
        max_val_samples: int = None,
    ) -> Dict[str, Any]:
        """
        执行微调。

        Args:
            train_df: 训练集 DataFrame
            val_df: 验证集 DataFrame
            checkpoint_dir: checkpoint 保存目录
            max_train_samples: 训练集最大样本数 (None = 不限)
            max_val_samples: 验证集最大样本数

        Returns:
            {
                "best_val_loss": float,
                "best_epoch": int,
                "spearman": float,
                "mae": float,
                "direction_accuracy": float,
                "checkpoint_path": str,
                "train_losses": [...],
                "val_losses": [...],
                "duration_seconds": float,
            }
        """
        set_seed(self.config.seed)
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"开始微调: model_type={self.model_type}, epochs={self.epochs}")
        start_time = time.time()

        # 准备数据
        X_train, y_train = self._prepare_samples(
            train_df, self.seq_len, self.pred_len, max_train_samples
        )
        logger.info(f"训练集样本: {len(X_train)} (seq_len={self.seq_len}, pred_len={self.pred_len})")

        X_val, y_val = None, None
        if val_df is not None:
            X_val, y_val = self._prepare_samples(
                val_df, self.seq_len, self.pred_len, max_val_samples
            )
            logger.info(f"验证集样本: {len(X_val)}")

        # 判断是否是 FallbackModel (有自己的训练逻辑)
        from src.model_layer.fallback_model import FallbackModel
        if isinstance(self._inner_model, FallbackModel):
            return self._run_fallback_finetune(
                train_df, val_df, checkpoint_dir
            )

        # Kronos 微调逻辑
        return self._run_kronos_finetune(
            X_train, y_train, X_val, y_val, checkpoint_dir, start_time
        )

    def _run_kronos_finetune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        checkpoint_dir: str,
        start_time: float,
    ) -> Dict[str, Any]:
        """Kronos 模型微调循环"""

        # 设置训练模式
        self._inner_model.train()

        # 收集可训练参数
        trainable_params = [p for p in self._inner_model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            # 如果模型参数全部 frozen，尝试解冻最后几层
            logger.warning("没有可训练参数，尝试解冻模型")
            for p in self._inner_model.parameters():
                p.requires_grad = True
            trainable_params = [p for p in self._inner_model.parameters() if p.requires_grad]

        logger.info(f"可训练参数: {sum(p.numel() for p in trainable_params):,}")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )

        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        best_checkpoint = None

        device = self.device

        for epoch in range(1, self.epochs + 1):
            # ── 训练 ──
            self._inner_model.train()
            epoch_loss = 0.0
            n_batches = 0

            perm = torch.randperm(len(X_train))
            for i in range(0, len(X_train), self.batch_size):
                idx = perm[i:i + self.batch_size]
                X_batch = torch.from_numpy(X_train[idx]).to(device)
                y_batch = torch.from_numpy(y_train[idx]).to(device)

                optimizer.zero_grad()

                try:
                    # 尝试 Kronos 训练接口
                    if hasattr(self._inner_model, 'forward'):
                        output = self._inner_model(X_batch)
                        if isinstance(output, dict):
                            loss = output.get("loss")
                            if loss is None:
                                logits = output.get("logits", output.get("last_hidden_state"))
                                loss = nn.functional.mse_loss(
                                    logits[:, -1, :y_batch.shape[-1]], y_batch
                                )
                        else:
                            loss = nn.functional.mse_loss(output, y_batch)
                    else:
                        loss = torch.tensor(0.0, device=device)
                except Exception as e:
                    logger.warning(f"Forward pass 失败: {e}, 跳过此batch")
                    continue

                loss.backward()

                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.grad_clip)

                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(avg_train_loss)
            scheduler.step()

            # ── 验证 ──
            val_metrics = {}
            if X_val is not None and len(X_val) > 0:
                self._inner_model.eval()
                val_loss_total = 0.0
                val_batches = 0
                all_preds = []
                all_targets = []

                with torch.no_grad():
                    for i in range(0, len(X_val), self.batch_size):
                        X_batch = torch.from_numpy(X_val[i:i + self.batch_size]).to(device)
                        y_batch = torch.from_numpy(y_val[i:i + self.batch_size]).to(device)

                        output = self._inner_model(X_batch)
                        if isinstance(output, dict):
                            logits = output.get("logits", output.get("last_hidden_state"))
                            preds = logits[:, -1, :y_batch.shape[-1]]
                        else:
                            preds = output

                        loss = nn.functional.mse_loss(preds, y_batch)
                        val_loss_total += loss.item()
                        val_batches += 1

                        all_preds.append(preds[:, 0].cpu().numpy())  # 第一日预测
                        all_targets.append(y_batch[:, 0].cpu().numpy())  # 第一日真实

                avg_val_loss = val_loss_total / max(val_batches, 1)
                val_losses.append(avg_val_loss)

                # 计算指标
                if all_preds and all_targets:
                    pred_array = np.concatenate(all_preds)
                    target_array = np.concatenate(all_targets)
                    val_metrics = self._compute_metrics(pred_array, target_array)
            else:
                avg_val_loss = avg_train_loss
                val_losses.append(None)

            monitor = avg_val_loss

            # 日志
            lr = scheduler.get_last_lr()[0]
            metric_str = " ".join(
                f"{k}={v:.4f}" for k, v in val_metrics.items()
            ) if val_metrics else ""
            logger.info(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"train_loss={avg_train_loss:.6f} | val_loss={avg_val_loss:.6f} | "
                f"lr={lr:.2e} {metric_str}"
            )

            # ── 早停 & checkpoint ──
            if monitor < best_val_loss:
                best_val_loss = monitor
                best_epoch = epoch
                patience_counter = 0

                ckpt_path = Path(checkpoint_dir) / f"finetune_epoch_{epoch}_{monitor:.4f}.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self._inner_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": monitor,
                    "val_metrics": val_metrics,
                    "timestamp": datetime.now().isoformat(),
                }, ckpt_path)
                best_checkpoint = str(ckpt_path)
                logger.info(f"  ✓ 新最优: {ckpt_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"早停于 epoch {epoch} (patience={self.patience})")
                    break

        duration = time.time() - start_time

        # 最终评估
        final_metrics = {"spearman": 0.0, "mae": 0.0, "direction_accuracy": 0.0}
        if X_val is not None and len(X_val) > 0:
            self._inner_model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for i in range(0, len(X_val), self.batch_size):
                    X_batch = torch.from_numpy(X_val[i:i + self.batch_size]).to(device)
                    y_batch = torch.from_numpy(y_val[i:i + self.batch_size]).to(device)
                    output = self._inner_model(X_batch)
                    if isinstance(output, dict):
                        logits = output.get("logits", output.get("last_hidden_state"))
                        preds = logits[:, -1, :1]
                    else:
                        preds = output[:, :1] if output.dim() > 1 else output
                    all_preds.append(preds.cpu().numpy().flatten())
                    all_targets.append(y_batch[:, 0].cpu().numpy())
            if all_preds and all_targets:
                final_metrics = self._compute_metrics(
                    np.concatenate(all_preds), np.concatenate(all_targets)
                )

        result = {
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "spearman": final_metrics["spearman"],
            "mae": final_metrics["mae"],
            "direction_accuracy": final_metrics["direction_accuracy"],
            "checkpoint_path": best_checkpoint,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "total_epochs": epoch,
            "duration_seconds": round(duration, 1),
        }

        # 保存结果 JSON
        result_path = Path(checkpoint_dir) / "finetune_result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"微调结果已保存: {result_path}")

        logger.info(
            f"微调完成: spearman={final_metrics['spearman']:.4f}, "
            f"mae={final_metrics['mae']:.2f}, "
            f"dir_acc={final_metrics['direction_accuracy']:.2%}, "
            f"耗时={duration:.0f}s"
        )

        return result

    def _run_fallback_finetune(
        self,
        train_df,
        val_df,
        checkpoint_dir: str,
    ) -> Dict[str, Any]:
        """使用 FallbackTrainer 训练 Path B 模型"""
        from src.model_layer.fallback_model import FallbackTrainer

        trainer = FallbackTrainer(self._inner_model, self.config, self.device)
        # 覆盖为 fallback 的 epochs
        trainer.epochs = self.config.get("model", "training", "epochs_fallback", default=50)
        trainer.patience = self.config.get("model", "training", "early_stop_patience_fallback", default=10)

        result = trainer.train(train_df, val_df, checkpoint_dir)

        # 计算最终指标
        from src.model_layer.fallback_model import FallbackModel
        if val_df is not None and isinstance(self._inner_model, FallbackModel):
            X_val, y_val = trainer._prepare_batch(val_df, self.seq_len, self.pred_len)
            if len(X_val) > 0:
                self._inner_model.eval()
                with torch.no_grad():
                    all_preds = []
                    all_targets = []
                    for i in range(0, len(X_val), 32):
                        X_batch = X_val[i:i + 32].to(self.device)
                        y_batch = y_val[i:i + 32]
                        preds = self._inner_model(X_batch)[:, 0].cpu().numpy()
                        all_preds.append(preds)
                        all_targets.append(y_batch[:, 0].numpy())
                    if all_preds:
                        metrics = self._compute_metrics(
                            np.concatenate(all_preds), np.concatenate(all_targets)
                        )
                        result.update(metrics)

        result_path = Path(checkpoint_dir) / "finetune_result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)

        return result
