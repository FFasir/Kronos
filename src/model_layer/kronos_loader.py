"""
Kronos 指数/ETF 择时策略系统 — Kronos 模型加载器

从 HuggingFace Hub 加载 Kronos-base 模型，统一模型接口。
Kronos 是清华大学发布的时序大模型（~300M参数），内置K线分词器。

Path A: 使用 Kronos-base 预训练权重进行价格预测。
"""

import os
import sys
import importlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from src.common.logger import get_logger
from src.common.exceptions import ModelLoadError, OOMError

logger = get_logger(__name__)


class KronosWrapper:
    """
    Kronos-base 模型包装器。

    对外暴露统一接口:
      - predict(ohlcv_sequence, horizon=5) → float
      - predict_sequence(ohlcv_sequence, horizon=20) → np.ndarray

    内部处理 Kronos K线分词器 → token → forward → 解码 → OHLCV 预测值。
    """

    def __init__(
        self,
        model,
        tokenizer=None,
        device: torch.device = None,
        model_version: str = "base",
        native_predictor=None,
    ):
        """
        Args:
            model: Kronos 模型实例 (HuggingFace model 或 Kronos SDK model)
            tokenizer: K线分词器 (可选，Kronos 可能内置)
            device: 计算设备
            model_version: 模型版本标识
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cpu")
        self.model_version = model_version
        self.native_predictor = native_predictor
        self.is_native_kronos = native_predictor is not None

        self.model.to(self.device)
        self.model.eval()

        # 尝试获取模型配置
        self.d_model = getattr(model.config, "d_model", 768) if hasattr(model, "config") else 768
        logger.info(f"KronosWrapper 初始化完成: version={model_version}, d_model={self.d_model}")

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        pred_len: int,
        T: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 1,
    ) -> pd.DataFrame:
        """Run the official predictor with real OHLCV values and timestamps."""
        if not self.is_native_kronos:
            raise ModelLoadError("当前后端不是官方 Kronos Predictor")

        frame = df.copy()
        required = ["open", "high", "low", "close"]
        if not all(column in frame.columns for column in required):
            raise ValueError(f"缺少 OHLC 列: {required}")
        if not isinstance(frame.index, pd.DatetimeIndex):
            frame.index = pd.date_range("2020-01-01", periods=len(frame), freq="B")
        frame.index = pd.to_datetime(frame.index)
        future = pd.bdate_range(frame.index[-1] + pd.offsets.BDay(1), periods=pred_len)
        return self.native_predictor.predict(
            df=frame,
            x_timestamp=pd.Series(frame.index),
            y_timestamp=pd.Series(future),
            pred_len=pred_len,
            T=T,
            top_p=top_p,
            sample_count=sample_count,
            verbose=False,
        )

    def predict_dataframes(
        self,
        frames: list[pd.DataFrame],
        pred_len: int,
        T: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 1,
    ) -> list[pd.DataFrame]:
        """Run native Kronos inference for an equal-length batch."""
        if not self.is_native_kronos:
            raise ModelLoadError("当前后端不是官方 Kronos Predictor")
        prepared = []
        history_timestamps = []
        future_timestamps = []
        for frame in frames:
            current = frame.copy()
            if not isinstance(current.index, pd.DatetimeIndex):
                current.index = pd.to_datetime(current.index)
            current.index = pd.to_datetime(current.index)
            prepared.append(current)
            history_timestamps.append(pd.Series(current.index))
            future_timestamps.append(pd.Series(pd.bdate_range(
                current.index[-1] + pd.offsets.BDay(1), periods=pred_len
            )))
        return self.native_predictor.predict_batch(
            prepared,
            history_timestamps,
            future_timestamps,
            pred_len=pred_len,
            T=T,
            top_k=0,
            top_p=top_p,
            sample_count=sample_count,
            verbose=False,
        )

    def predict_dataframes_with_uncertainty(
        self,
        frames: list[pd.DataFrame],
        pred_len: int,
        T: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 5,
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        """Return forecast means and sampling standard deviations."""
        if not self.is_native_kronos:
            raise ModelLoadError("当前后端不是官方 Kronos Predictor")
        if sample_count < 2:
            forecasts = self.predict_dataframes(
                frames, pred_len, T=T, top_p=top_p, sample_count=sample_count
            )
            return forecasts, [forecast * 0.0 for forecast in forecasts]

        prepared = []
        history_timestamps = []
        future_timestamps = []
        for frame in frames:
            current = frame.copy()
            if not isinstance(current.index, pd.DatetimeIndex):
                current.index = pd.to_datetime(current.index)
            current.index = pd.to_datetime(current.index)
            prepared.append(current)
            history_timestamps.append(pd.Series(current.index))
            future_timestamps.append(pd.Series(pd.bdate_range(
                current.index[-1] + pd.offsets.BDay(1), periods=pred_len
            )))
        return self.native_predictor.predict_batch(
            prepared,
            history_timestamps,
            future_timestamps,
            pred_len=pred_len,
            T=T,
            top_k=0,
            top_p=top_p,
            sample_count=sample_count,
            verbose=False,
            return_std=True,
        )

    @torch.no_grad()
    def predict(self, ohlcv_sequence: torch.Tensor, horizon: int = 5) -> float:
        """
        预测未来第 horizon 日收盘价。

        Args:
            ohlcv_sequence: (seq_len, 5) — 历史 OHLCV 序列
            horizon: 预测第几天的收盘价 (1-20)

        Returns:
            float — 预测收盘价
        """
        self.model.eval()

        if self.is_native_kronos:
            values = ohlcv_sequence.detach().float().cpu().numpy()
            if values.ndim == 3:
                values = values[0]
            frame = pd.DataFrame(values, columns=["open", "high", "low", "close", "volume"])
            predicted = self.predict_dataframe(frame, pred_len=horizon)
            return float(predicted["close"].iloc[-1])

        # 确保 3D
        if ohlcv_sequence.dim() == 2:
            ohlcv_sequence = ohlcv_sequence.unsqueeze(0)  # (1, seq_len, 5)

        ohlcv_sequence = ohlcv_sequence.to(self.device)

        # 尝试 Kronos 标准推理接口
        try:
            # 如果模型有 kronos 专用 forward 方法
            if hasattr(self.model, 'generate'):
                output = self.model.generate(
                    ohlcv_sequence,
                    max_new_tokens=horizon,
                    temperature=1.0,
                    top_p=0.95,
                )
                # 从输出中提取收盘价
                if isinstance(output, dict):
                    closes = output.get("close", output.get("pred_close"))
                    return float(closes[0, min(horizon - 1, len(closes[0]) - 1)].item())
                elif isinstance(output, torch.Tensor):
                    return float(output[0, min(horizon - 1, output.shape[-1] - 1)].item())
            else:
                # 回退：直接 forward
                output = self.model(ohlcv_sequence)
                if isinstance(output, dict):
                    logits = output.get("logits", output.get("last_hidden_state"))
                    if logits is not None:
                        return float(logits[0, -1, 0].item())
                elif isinstance(output, torch.Tensor):
                    return float(output[0, min(horizon - 1, output.shape[-1] - 1)].item())
        except Exception as e:
            logger.warning(f"Kronos predict 异常: {e}, 使用零值fallback")
            return 0.0

        return 0.0

    @torch.no_grad()
    def predict_sequence(self, ohlcv_sequence: torch.Tensor, horizon: int = 20) -> np.ndarray:
        """
        预测未来 horizon 日收盘价序列。

        Args:
            ohlcv_sequence: (seq_len, 5) — 历史 OHLCV 序列
            horizon: 预测天数 (1-20)

        Returns:
            np.ndarray shape (horizon,) — 预测收盘价序列
        """
        self.model.eval()

        if self.is_native_kronos:
            values = ohlcv_sequence.detach().float().cpu().numpy()
            if values.ndim == 3:
                values = values[0]
            frame = pd.DataFrame(values, columns=["open", "high", "low", "close", "volume"])
            predicted = self.predict_dataframe(frame, pred_len=horizon)
            return predicted["close"].to_numpy(dtype=np.float32)

        if ohlcv_sequence.dim() == 2:
            ohlcv_sequence = ohlcv_sequence.unsqueeze(0)

        ohlcv_sequence = ohlcv_sequence.to(self.device)

        try:
            if hasattr(self.model, 'generate'):
                output = self.model.generate(
                    ohlcv_sequence,
                    max_new_tokens=horizon,
                    temperature=1.0,
                    top_p=0.95,
                )
                if isinstance(output, dict):
                    closes = output.get("close", output.get("pred_close"))
                    if closes is not None:
                        return closes[0, :horizon].cpu().numpy()
                elif isinstance(output, torch.Tensor):
                    return output[0, :horizon].cpu().numpy()
            else:
                output = self.model(ohlcv_sequence)
                if isinstance(output, dict):
                    logits = output.get("logits", output.get("last_hidden_state"))
                    if logits is not None:
                        return logits[0, -1, :horizon].cpu().numpy()
                elif isinstance(output, torch.Tensor):
                    return output[0, :horizon].cpu().numpy()
        except Exception as e:
            logger.warning(f"Kronos predict_sequence 异常: {e}")

        return np.zeros(horizon, dtype=np.float32)


# ═══════════════════════════════════════════════
# 加载函数
# ═══════════════════════════════════════════════

def _check_huggingface_availability() -> bool:
    """检查 HuggingFace hub 是否可用"""
    try:
        import huggingface_hub
        return True
    except ImportError:
        logger.warning("huggingface_hub 未安装")
        return False


def _check_model_cached(hf_model_id: str) -> bool:
    """检查模型是否已在 HuggingFace 缓存中"""
    try:
        from huggingface_hub import scan_cache_dir
        caches = scan_cache_dir()
        for repo in caches.repos:
            if hf_model_id.replace("/", "--") in repo.repo_id or hf_model_id in repo.repo_id:
                logger.info(f"模型已在缓存: {repo.repo_id}")
                return True
    except Exception:
        pass

    # 检查默认缓存路径
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir_name = f"models--{hf_model_id.replace('/', '--')}"
    model_cache = cache_dir / model_dir_name
    if model_cache.exists():
        logger.info(f"模型缓存目录存在: {model_cache}")
        return True

    return False


def _check_kronos_sdk() -> bool:
    """检查 kronos 专用 SDK 是否安装"""
    try:
        import kronos
        return True
    except ImportError:
        return False


def load_kronos_from_hf(
    hf_model_id: str = "NeoQuasar/Kronos-base",
    tokenizer_id: Optional[str] = None,
    official_repo_path: Optional[str] = None,
    device: str = "auto",
    use_mirror: bool = True,
    trust_remote_code: bool = True,
    **kwargs,
) -> KronosWrapper:
    """
    从 HuggingFace Hub 加载 Kronos 模型。

    Args:
        hf_model_id: HuggingFace 模型 ID 或本地完整模型目录
        tokenizer_id: HuggingFace tokenizer ID 或本地 tokenizer 目录；None 时复用模型来源
        official_repo_path: 官方 Kronos 仓库源码路径（包含 model/ 包）
        device: "auto" | "cuda" | "cpu"
        use_mirror: 是否使用国内镜像 (hf-mirror.com)
        trust_remote_code: 是否信任远程代码

    Returns:
        KronosWrapper 实例

    Raises:
        ModelLoadError: 加载失败
    """
    from src.common.utils import detect_device
    device_obj = detect_device(device)

    # 设置镜像
    if use_mirror:
        if "HF_ENDPOINT" not in os.environ:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            logger.info("使用 HuggingFace 镜像: hf-mirror.com")

    if not _check_huggingface_availability():
        raise ModelLoadError(
            "huggingface_hub 未安装。请执行: pip install huggingface_hub transformers"
        )

    logger.info(f"正在加载官方 Kronos 模型: {hf_model_id}")
    repo_path = Path(official_repo_path or os.environ.get("KRONOS_OFFICIAL_REPO", ""))
    if not repo_path.is_dir() or not (repo_path / "model" / "__init__.py").is_file():
        raise ModelLoadError(
            "未找到官方 Kronos 源码。请将 https://github.com/shiyu-coder/Kronos "
            "放到 third_party/Kronos，或设置 KRONOS_OFFICIAL_REPO。"
        )

    try:
        repo_text = str(repo_path.resolve())
        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)
        importlib.invalidate_caches()
        from model import Kronos, KronosPredictor, KronosTokenizer

        tokenizer_source = tokenizer_id or hf_model_id
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_source)
        model = Kronos.from_pretrained(hf_model_id)
        predictor = KronosPredictor(model, tokenizer, device=str(device_obj), max_context=512)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"官方 Kronos 模型加载成功: {total_params:,} 参数")
        return KronosWrapper(
            model,
            tokenizer,
            device_obj,
            model_version=kwargs.get("model_version", "base"),
            native_predictor=predictor,
        )
    except Exception as e:
        raise ModelLoadError(f"Kronos 模型加载失败: {e}") from e


def _validate_model(model, device: torch.device):
    """Dummy forward pass 验证模型输出形状正确"""
    try:
        seq_len, input_dim = 60, 5
        dummy_input = torch.randn(1, seq_len, input_dim).to(device)

        with torch.no_grad():
            output = model(dummy_input)

        if isinstance(output, dict):
            logger.info(f"Dummy forward pass 通过: keys={list(output.keys())}")
            # 检查是否有 NaN
            for key, val in output.items():
                if isinstance(val, torch.Tensor) and torch.isnan(val).any():
                    logger.warning(f"  ⚠ 输出 '{key}' 包含 NaN!")
        elif isinstance(output, torch.Tensor):
            has_nan = torch.isnan(output).any().item()
            logger.info(
                f"Dummy forward pass 通过: shape={output.shape}, "
                f"has_nan={has_nan}"
            )
        else:
            logger.info(f"Dummy forward pass 通过: type={type(output)}")

    except Exception as e:
        logger.warning(f"Dummy forward pass 异常: {e}")


# ═══════════════════════════════════════════════
# Kronos Finetune 数据准备
# ═══════════════════════════════════════════════

def prepare_finetune_data(
    config,
    symbols: Optional[List[str]] = None,
    use_components: bool = False,
) -> dict:
    """
    为 Kronos 微调准备训练/验证数据。

    论文方法: 使用中证1000成分股 2010-2022 日线OHLCV 作为训练集，
    2023-2024 作为验证集。

    Args:
        config: 系统配置
        symbols: 标的列表 (None = 使用配置中的12个指数/ETF，或组件 HDF5 全部标的)
        use_components: 从 data/processed/components.h5 读取股票数据

    Returns:
        {
            "train": pd.DataFrame,  # 训练集合并数据
            "val": pd.DataFrame,    # 验证集合并数据
            "train_symbols": [...], # 训练集包含的标的
        }
    """
    import pandas as pd
    from src.data_layer.storage import StorageLayer

    storage = StorageLayer(config)
    h5_path = storage.h5_path
    if use_components:
        h5_path = Path(config.data.get("component_processed_path", "data/processed/components.h5"))
        if not h5_path.is_absolute():
            from src.common.config import PROJECT_ROOT
            h5_path = PROJECT_ROOT / h5_path

    if not h5_path.exists():
        logger.warning(f"HDF5 文件不存在: {h5_path}, 返回空数据")
        return {"train": None, "val": None, "train_symbols": []}

    if symbols is None:
        if use_components:
            with pd.HDFStore(h5_path, mode="r") as store:
                symbol_list = [key.rsplit("/", 1)[-1] for key in store.keys() if key.startswith("/raw/")]
        else:
            symbol_objs = config.get_all_symbols()
            symbol_list = [s["code"] for s in symbol_objs]
    else:
        symbol_list = symbols

    train_start = config.get("data", "finetune_train_start", default="2010-01-01")
    train_end = config.get("data", "finetune_train_end", default="2022-12-31")
    val_start = config.get("data", "finetune_val_start", default="2023-01-01")
    val_end = config.get("data", "finetune_val_end", default="2024-12-31")

    train_frames = []
    val_frames = []
    loaded_symbols = []

    for sym in symbol_list:
        try:
            if use_components:
                with pd.HDFStore(h5_path, mode="r") as store:
                    key = f"/raw/{sym}"
                    df = store[key] if key in store else None
            else:
                df = storage.load_raw_data(sym)
            if df is None or len(df) == 0:
                continue

            df_train = df[(df.index >= train_start) & (df.index <= train_end)]
            df_val = df[(df.index >= val_start) & (df.index <= val_end)]

            if len(df_train) > 0:
                df_train = df_train.copy()
                df_train["symbol"] = sym
                train_frames.append(df_train)

            if len(df_val) > 0:
                df_val = df_val.copy()
                df_val["symbol"] = sym
                val_frames.append(df_val)

            loaded_symbols.append(sym)
            logger.info(f"  {sym}: train={len(df_train)}, val={len(df_val)}")

        except Exception as e:
            logger.warning(f"加载 {sym} 数据失败: {e}")

    train_df = pd.concat(train_frames) if train_frames else None
    val_df = pd.concat(val_frames) if val_frames else None

    logger.info(
        f"微调数据准备完成: {len(loaded_symbols)} 标的, "
        f"train={len(train_df) if train_df is not None else 0} 行, "
        f"val={len(val_df) if val_df is not None else 0} 行"
    )

    return {
        "train": train_df,
        "val": val_df,
        "train_symbols": loaded_symbols,
    }
