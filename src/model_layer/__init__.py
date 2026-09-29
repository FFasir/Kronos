"""Kronos 指数/ETF 择时策略系统 — 模型层

模块:
  - kronos_loader:    从 HuggingFace 加载 Kronos-base 模型 (Path A)
  - kronos_finetune:  微调 Kronos-base 或 FallbackModel
  - kronos_predictor: 统一预测推理接口
  - param_search:     月度滚动网格搜参 (论文核心)
  - fallback_model:   Path B — 自建 Decoder-only Transformer
  - model_factory:    自动检测并选择 Path A / Path B

使用:
  from src.model_layer.model_factory import create_model
  model, path = create_model(config)
  pred = model.predict(ohlcv_seq, horizon=5)
"""

from src.model_layer.fallback_model import FallbackModel, FallbackTrainer
from src.model_layer.model_factory import create_model, detect_model_path, get_model, get_model_path

__all__ = [
    "FallbackModel",
    "FallbackTrainer",
    "create_model",
    "detect_model_path",
    "get_model",
    "get_model_path",
]
