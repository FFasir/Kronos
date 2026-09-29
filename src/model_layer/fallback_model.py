"""
Kronos 指数/ETF 择时策略系统 — Path B Fallback 模型

当 Kronos-base 不可用时的替代方案。
Decoder-only Transformer，类 GPT 架构，用于日线 OHLCV 序列 → 未来价格预测。

输入: (batch, seq_len, 5) — OHLCV 序列
输出: (batch, pred_len) 或 (batch,) — 未来收盘价预测
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple, List
from datetime import datetime

from src.common.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════
# Transformer 组件
# ═══════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """可学习的位置编码"""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class CausalSelfAttention(nn.Module):
    """因果自注意力（带 mask 防止看到未来）"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("causal_mask", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 构建 causal mask
        if self.causal_mask is None or self.causal_mask.size(-1) < seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            self.register_buffer("causal_mask", mask)
        else:
            mask = self.causal_mask[:seq_len, :seq_len]

        # Scaled dot-product attention
        scale = math.sqrt(self.d_k)
        scores = (q @ k.transpose(-2, -1)) / scale
        scores = scores.masked_fill(mask[:seq_len, :seq_len], float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Decoder-only Transformer 块"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ═══════════════════════════════════════════════
# Fallback 模型主类
# ═══════════════════════════════════════════════

class FallbackModel(nn.Module):
    """
    Path B: 自建 Decoder-only Transformer 用于股价预测。

    架构:
      输入投影:  OHLCV (5维) → d_model
      位置编码:  可学习 Positional Embedding
      Transformer: n_layers × TransformerBlock (causal attention)
      回归头:    最后位置 hidden → pred_len 维输出

    Args:
        d_model:     隐层维度 (default: 256)
        n_layers:    Transformer 层数 (default: 6)
        n_heads:     注意力头数 (default: 8)
        d_ff:        前馈网络隐层维度 (default: 1024)
        dropout:     Dropout 率 (default: 0.1)
        max_seq_len: 最大序列长度 (default: 512)
        input_dim:   输入特征维度 (default: 5, OHLCV)
        pred_len:    预测步长 (default: 5, 未来第5日收盘价)
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        input_dim: int = 5,
        pred_len: int = 5,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        self.pred_len = pred_len
        self.max_seq_len = max_seq_len

        # 输入投影: OHLCV → d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # Transformer 层
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 最终 LayerNorm
        self.ln_f = nn.LayerNorm(d_model)

        # 回归头: 取最后位置的 hidden → 预测未来收盘价
        self.head = nn.Linear(d_model, pred_len)

        # 序列回归头: 预测 20 日收盘价序列
        self.head_sequence = nn.Linear(d_model, 20)

        # 初始化
        self._init_weights()

        # 记录参数数量
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"FallbackModel 初始化完成: 总参数 {total:,}, 可训练 {trainable:,}")

    def _init_weights(self):
        """Xavier/Gaussian 初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ═══════════════════════════════════════════════
    # 窗口归一化（训练/推理共用，无未来泄露）
    # ═══════════════════════════════════════════════

    EPS = 1e-8

    @staticmethod
    def normalize_window(window: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        按「窗口末日收盘价」归一化一个 OHLCV 输入窗口。

        - OHLC 四列除以窗口最后一日的收盘价 → 比值（≈1.0 附近），
          消除指数价格 10^3~10^4 的绝对量级；
        - volume 列除以窗口内成交量均值 → 消除 10^8 量级，避免主导投影。

        归一化因子仅来自窗口内已知数据（末日收盘价、窗口内均量），
        不使用任何未来信息，因此无数据泄露。

        Args:
            window: (seq_len, 5) 原始 OHLCV，列序 [open, high, low, close, volume]

        Returns:
            (normed, last_close):
              normed     — (seq_len, 5) 归一化后的窗口
              last_close  — 该窗口末日收盘价，用于把模型输出的比值还原为绝对价
        """
        window = window.astype(np.float32).copy()
        last_close = float(window[-1, 3])
        denom = last_close if abs(last_close) > FallbackModel.EPS else 1.0

        normed = window.copy()
        normed[:, 0:4] = window[:, 0:4] / denom  # OHLC → 比值
        vol_mean = float(window[:, 4].mean())
        vol_denom = vol_mean if abs(vol_mean) > FallbackModel.EPS else 1.0
        normed[:, 4] = window[:, 4] / vol_denom  # volume → 相对均量

        return normed, last_close

    @staticmethod
    def denormalize_close(ratio, last_close: float):
        """把模型输出的「收盘价/末收盘价」比值还原为绝对收盘价。"""
        return ratio * last_close

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: (batch, seq_len, input_dim) — 已归一化的 OHLCV 序列

        Returns:
            (batch, pred_len) — 预测的未来收盘价「比值」(相对输入窗口末收盘价)
        """
        batch, seq_len, _ = x.shape

        # 输入投影
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # 位置编码
        x = self.pos_encoding(x)

        # Transformer 层
        for layer in self.layers:
            x = layer(x)

        # 最终 norm
        x = self.ln_f(x)

        # 取最后位置的 hidden state
        last_hidden = x[:, -1, :]  # (batch, d_model)

        # 回归预测
        out = self.head(last_hidden)  # (batch, pred_len)

        return out

    # ═══════════════════════════════════════════════
    # 统一预测接口 (与 Kronos 对齐)
    # ═══════════════════════════════════════════════

    @torch.no_grad()
    def predict(self, ohlcv_sequence: torch.Tensor, horizon: int = 5) -> float:
        """
        预测未来第 horizon 日收盘价。

        Args:
            ohlcv_sequence: (seq_len, 5) 或 (1, seq_len, 5) — 历史 OHLCV
            horizon: 预测第几天的收盘价 (1-5)

        Returns:
            float — 预测收盘价
        """
        self.eval()

        # 确保 3D tensor
        if ohlcv_sequence.dim() == 2:
            ohlcv_sequence = ohlcv_sequence.unsqueeze(0)

        # 输入为原始 OHLCV，先按窗口末收盘价归一化（与训练口径一致）
        window = ohlcv_sequence[0].detach().cpu().numpy()
        normed, last_close = self.normalize_window(window)

        device = next(self.parameters()).device
        x = torch.from_numpy(normed).unsqueeze(0).to(device)

        ratios = self.forward(x)  # (1, pred_len) —— 相对末收盘价的比值
        idx = min(horizon - 1, self.pred_len - 1)
        ratio = ratios[0, idx].item()
        return self.denormalize_close(ratio, last_close)  # 还原为绝对收盘价

    @torch.no_grad()
    def predict_sequence(self, ohlcv_sequence: torch.Tensor, horizon: int = 20) -> np.ndarray:
        """
        预测未来 horizon 日收盘价序列。

        Args:
            ohlcv_sequence: (seq_len, 5) 或 (1, seq_len, 5) — 历史 OHLCV
            horizon: 预测天数 (1-20)

        Returns:
            np.ndarray shape (horizon,) — 预测收盘价序列
        """
        self.eval()

        if ohlcv_sequence.dim() == 2:
            ohlcv_sequence = ohlcv_sequence.unsqueeze(0)

        # 输入为原始 OHLCV，按窗口末收盘价归一化（与训练口径一致）
        window = ohlcv_sequence[0].detach().cpu().numpy()
        normed, last_close = self.normalize_window(window)

        device = next(self.parameters()).device
        x = torch.from_numpy(normed).unsqueeze(0).to(device)

        # 使用序列预测头
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        last_hidden = x[:, -1, :]
        seq_ratios = self.head_sequence(last_hidden)  # (1, 20) —— 比值

        ratios = seq_ratios[0, :horizon].cpu().numpy()
        return self.denormalize_close(ratios, last_close)  # 还原为绝对收盘价序列

    # ═══════════════════════════════════════════════
    # 训练辅助
    # ═══════════════════════════════════════════════

    def compute_loss(self, batch: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算 MSE loss。

        Args:
            batch: (batch, seq_len, input_dim)
            targets: (batch, pred_len) — 真实未来收盘价

        Returns:
            scalar loss
        """
        preds = self.forward(batch)
        return F.mse_loss(preds, targets)

    def save(self, path: str):
        """保存模型权重"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # 过滤掉 causal_mask buffer (会在 forward 时自动重建)
        state = self.state_dict()
        state = {k: v for k, v in state.items() if "causal_mask" not in k}

        # 获取 n_heads 和 d_ff (从第一层推断)
        n_heads = self.layers[0].attention.n_heads
        d_ff = self.layers[0].ff[0].out_features

        torch.save({
            "model_state_dict": state,
            "d_model": self.d_model,
            "n_layers": len(self.layers),
            "n_heads": n_heads,
            "d_ff": d_ff,
            "dropout": self.layers[0].ff[1].p if hasattr(self.layers[0].ff[1], 'p') else 0.1,
            "input_dim": self.input_dim,
            "pred_len": self.pred_len,
            "max_seq_len": self.max_seq_len,
            "saved_at": datetime.now().isoformat(),
        }, path)
        logger.info(f"模型已保存: {path}")

    @classmethod
    def load(cls, path: str, device: str = "auto") -> "FallbackModel":
        """
        从 checkpoint 加载模型。

        Args:
            path: checkpoint 文件路径
            device: 目标设备

        Returns:
            FallbackModel 实例
        """
        from src.common.utils import detect_device

        device_obj = detect_device(device)
        checkpoint = torch.load(path, map_location=device_obj, weights_only=False)

        model = cls(
            d_model=checkpoint.get("d_model", 256),
            n_layers=checkpoint.get("n_layers", 6),
            n_heads=checkpoint.get("n_heads", 8),
            d_ff=checkpoint.get("d_ff", 1024),
            dropout=checkpoint.get("dropout", 0.1),
            input_dim=checkpoint.get("input_dim", 5),
            pred_len=checkpoint.get("pred_len", 5),
            max_seq_len=checkpoint.get("max_seq_len", 512),
        )

        # 过滤掉 causal_mask (checkpoint 中可能包含也可能不包含)
        model_state = checkpoint["model_state_dict"]
        model_state = {k: v for k, v in model_state.items() if "causal_mask" not in k}

        model.load_state_dict(model_state)
        model.to(device_obj)
        model.eval()

        logger.info(f"模型已加载: {path} (saved_at={checkpoint.get('saved_at', 'unknown')})")
        return model


# ═══════════════════════════════════════════════
# 训练器
# ═══════════════════════════════════════════════

class FallbackTrainer:
    """
    Path B 模型训练器。

    在 CPU 上使用 12 个指数/ETF 数据训练。
    """

    def __init__(
        self,
        model: FallbackModel,
        config,
        device: torch.device = None,
    ):
        self.model = model
        self.config = config

        from src.common.utils import detect_device
        self.device = device or detect_device(config.get("model", "device", default="auto"))

        self.model.to(self.device)

        # 优化器
        lr = config.get("model", "training", "learning_rate_fallback", default=1e-4)
        weight_decay = config.get("model", "training", "weight_decay", default=1e-5)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # 学习率调度器
        epochs = config.get("model", "training", "epochs_fallback", default=50)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

        self.epochs = epochs
        self.patience = config.get("model", "training", "early_stop_patience_fallback", default=10)
        self.grad_clip = config.get("model", "training", "grad_clip", default=1.0)
        self.seq_len = config.get("model", "training", "seq_len", default=60)
        self.pred_len = config.get("model", "training", "pred_len", default=5)

        self.best_loss = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0

    def _prepare_batch(
        self, df, seq_len: int, pred_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从 DataFrame 构建训练样本。

        每个样本: 前 seq_len 天 OHLCV → 未来 pred_len 天收盘价。

        Args:
            df: 含 open/high/low/close/volume 列的 DataFrame, 索引为日期
            seq_len: 输入序列长度
            pred_len: 预测步长

        Returns:
            (X, y): X shape (n_samples, seq_len, 5), y shape (n_samples, pred_len)
        """
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        data = df[ohlcv_cols].values.astype(np.float32)

        X_list, y_list = [], []
        for i in range(len(data) - seq_len - pred_len + 1):
            window = data[i:i + seq_len]                       # (seq_len, 5) 原始
            future_close = data[i + seq_len:i + seq_len + pred_len, 3]  # (pred_len,) 原始收盘价

            # 按窗口末收盘价归一化：输入窗口 + 目标收盘价用同一 last_close，
            # 目标转为「未来收盘价/末收盘价」比值，与模型输出口径一致。
            normed, last_close = FallbackModel.normalize_window(window)
            denom = last_close if abs(last_close) > FallbackModel.EPS else 1.0

            X_list.append(normed)
            y_list.append(future_close / denom)

        if len(X_list) == 0:
            return (
                torch.empty(0, seq_len, 5),
                torch.empty(0, pred_len),
            )

        X = torch.from_numpy(np.stack(X_list))
        y = torch.from_numpy(np.stack(y_list))
        return X, y

    def train(
        self,
        train_df,
        val_df=None,
        checkpoint_dir: str = "models/checkpoints/",
    ) -> dict:
        """
        训练模型。

        Args:
            train_df: 训练数据 DataFrame
            val_df: 验证数据 DataFrame (可选)
            checkpoint_dir: checkpoint 保存目录

        Returns:
            {"best_loss": float, "best_epoch": int, "train_losses": [...], "val_losses": [...]}
        """
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        X_train, y_train = self._prepare_batch(train_df, self.seq_len, self.pred_len)
        logger.info(f"训练集样本: {len(X_train)} (seq_len={self.seq_len}, pred_len={self.pred_len})")

        X_val, y_val = None, None
        if val_df is not None:
            X_val, y_val = self._prepare_batch(val_df, self.seq_len, self.pred_len)
            logger.info(f"验证集样本: {len(X_val)}")

        batch_size = self.config.get("model", "training", "batch_size_cpu", default=4)

        train_losses = []
        val_losses = []

        for epoch in range(1, self.epochs + 1):
            # ── 训练 ──
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            perm = torch.randperm(len(X_train))
            for i in range(0, len(X_train), batch_size):
                idx = perm[i:i + batch_size]
                X_batch = X_train[idx].to(self.device)
                y_batch = y_train[idx].to(self.device)

                self.optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = F.mse_loss(preds, y_batch)
                loss.backward()

                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches
            train_losses.append(avg_train_loss)
            self.scheduler.step()

            # ── 验证 ──
            if X_val is not None and len(X_val) > 0:
                self.model.eval()
                with torch.no_grad():
                    val_preds = []
                    for i in range(0, len(X_val), batch_size):
                        X_batch = X_val[i:i + batch_size].to(self.device)
                        val_preds.append(self.model(X_batch))
                    val_preds_all = torch.cat(val_preds, dim=0)
                    val_loss = F.mse_loss(val_preds_all, y_val.to(self.device)).item()
                val_losses.append(val_loss)

                monitor = val_loss
                loss_label = f"val_loss={val_loss:.6f}"
            else:
                monitor = avg_train_loss
                loss_label = f"train_loss={avg_train_loss:.6f}"
                val_losses.append(None)

            # 日志
            lr = self.scheduler.get_last_lr()[0]
            logger.info(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"train_loss={avg_train_loss:.6f} | {loss_label} | lr={lr:.2e}"
            )

            # ── 早停 & checkpoint ──
            if monitor < self.best_loss:
                self.best_loss = monitor
                self.best_epoch = epoch
                self.patience_counter = 0
                ckpt_path = str(Path(checkpoint_dir) / "fallback_best.pt")
                self.model.save(ckpt_path)
                logger.info(f"  ✓ 新最优模型: {ckpt_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"早停于 epoch {epoch} (patience={self.patience})")
                    break

        result = {
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "total_epochs": epoch,
        }

        logger.info(f"训练完成: best_loss={self.best_loss:.6f} @ epoch {self.best_epoch}")
        return result
