"""
Kronos 指数/ETF 择时策略系统 — 模型工厂

自动检测可用的模型后端并创建统一接口的模型实例。

检测逻辑:
  1. 尝试从 HuggingFace 加载 Kronos-base → Path A
  2. 检查本地缓存是否已有 Kronos 模型 → Path A
  3. 全部失败 → Path B (FallbackModel)

上层调用方只需:
  model = create_model(config)
  pred = model.predict(ohlcv_sequence, horizon=5)
"""

import os
import torch
from pathlib import Path
from typing import Optional, Tuple

from src.common.logger import get_logger
from src.common.exceptions import ModelLoadError

logger = get_logger(__name__)


# 全局模型缓存 (单例模式)
_model_instance = None
_model_path = None  # "kronos" | "fallback"


def _has_model_weights(path: Path) -> bool:
    """Reject partial Hugging Face downloads that contain only config files."""
    if not path.is_dir():
        return False
    min_weight_size = 10 * 1024 * 1024
    return any(
        file.stat().st_size >= min_weight_size
        for pattern in ("*.safetensors", "pytorch_model*.bin", "*.ckpt")
        for file in path.glob(pattern)
    )


def _has_tokenizer_assets(path: Path) -> bool:
    """A tokenizer config alone is a common sign of an interrupted download."""
    if not (path / "config.json").is_file():
        return False
    min_asset_size = 1024 * 1024
    patterns = ("*.safetensors", "*.bin", "*.model", "*.pt", "tokenizer.json", "vocab.json")
    return any(
        file.stat().st_size >= min_asset_size
        for pattern in patterns
        for file in path.glob(pattern)
    )


def _existing_local_path(value: str, require_weights: bool) -> Optional[str]:
    path = Path(value)
    if not path.is_absolute():
        from src.common.config import PROJECT_ROOT
        path = PROJECT_ROOT / path
    if path.is_dir() and (not require_weights or _has_model_weights(path)):
        return str(path)
    return None


def detect_model_path(config) -> str:
    """
    检测可用的模型路径。

    Returns:
        "kronos" 或 "fallback"
    """
    # 1. 检查是否强制使用 fallback
    force_fallback = os.environ.get("KRONOS_FORCE_FALLBACK", "").lower() in ("1", "true", "yes")
    if force_fallback:
        logger.info("环境变量 KRONOS_FORCE_FALLBACK=1, 使用 Path B fallback")
        return "fallback"

    # 2. 检查 HuggingFace 网络可达性
    hf_available = _check_hf_reachable(config)

    # 3. 检查模型本地缓存
    model_cached = _check_local_cache(config)

    if hf_available or model_cached:
        logger.info(f"模型检测: HF可达={hf_available}, 本地缓存={model_cached} → Path A (Kronos)")
        return "kronos"

    logger.info("Kronos-base 不可用 → Path B (FallbackModel)")
    return "fallback"


def create_model(
    config,
    device: str = None,
    force_path: str = None,
    pretrained_path: str = None,
) -> Tuple[object, str]:
    """
    创建模型实例。

    Args:
        config: 系统配置
        device: "auto" | "cuda" | "cpu" (None = 使用配置)
        force_path: 强制使用 "kronos" 或 "fallback" (None = 自动检测)
        pretrained_path: Path B 预训练权重路径 (可选)

    Returns:
        (model_instance, path_label) — path_label 为 "kronos" 或 "fallback"

    Raises:
        ModelLoadError: 所有路径都失败
    """
    global _model_instance, _model_path

    from src.common.utils import detect_device

    if device is None:
        device = config.get("model", "device", default="auto")
    device_obj = detect_device(device)

    # 确定路径
    if force_path:
        path = force_path
    else:
        path = detect_model_path(config)

    logger.info(f"创建模型: path={path}, device={device_obj}")

    if path == "kronos":
        try:
            model, _model_path = _create_kronos(config, device_obj)
            _model_instance = model
            return model, _model_path
        except Exception as e:
            logger.warning(f"Path A (Kronos) 创建失败: {e}, 降级为 Path B")
            if force_path == "kronos":
                raise ModelLoadError(f"Path A (Kronos) 创建失败: {e}") from e
            # 降级到 Path B
            path = "fallback"

    if path == "fallback":
        try:
            model, _model_path = _create_fallback(config, device_obj, pretrained_path)
            _model_instance = model
            return model, _model_path
        except Exception as e:
            raise ModelLoadError(f"Path B (Fallback) 创建失败: {e}")

    raise ModelLoadError("无法创建任何模型实例")


def _create_kronos(config, device_obj: torch.device):
    """创建 Kronos 模型实例"""
    from src.model_layer.kronos_loader import load_kronos_from_hf

    hf_id = config.get("model", "kronos_hf_id", default="NeoQuasar/Kronos-base")
    tokenizer_id = config.get(
        "model", "kronos_tokenizer_hf_id", default="NeoQuasar/Kronos-Tokenizer-base"
    )
    use_finetuned = config.get("model", "use_finetuned", default=False)
    model_value = config.get("model", "pretrained_path", default="models/kronos_base")
    tokenizer_value = config.get("model", "tokenizer_path", default="models/kronos_tokenizer")
    model_version = "base"
    if use_finetuned:
        finetuned_model = config.get("model", "finetuned_predictor_path", default="")
        finetuned_tokenizer = config.get("model", "finetuned_tokenizer_path", default="")
        finetuned_tokenizer_path = Path(finetuned_tokenizer)
        if not finetuned_tokenizer_path.is_absolute():
            from src.common.config import PROJECT_ROOT
            finetuned_tokenizer_path = PROJECT_ROOT / finetuned_tokenizer_path
        if (
            _existing_local_path(finetuned_model, require_weights=True)
            and _has_tokenizer_assets(finetuned_tokenizer_path)
        ):
            model_value = finetuned_model
            tokenizer_value = finetuned_tokenizer
            model_version = config.get(
                "model", "finetuned_model_version", default="finetuned"
            )
        else:
            logger.warning("Configured fine-tuned Kronos assets are incomplete; using base weights")

    local_model = _existing_local_path(model_value, require_weights=True)
    tokenizer_path = Path(tokenizer_value)
    if not tokenizer_path.is_absolute():
        from src.common.config import PROJECT_ROOT
        tokenizer_path = PROJECT_ROOT / tokenizer_path
    local_tokenizer = str(tokenizer_path) if _has_tokenizer_assets(tokenizer_path) else None
    repo_path = _existing_local_path(
        config.get("model", "official_repo_path", default="third_party/Kronos"), require_weights=False
    )

    kronos = load_kronos_from_hf(
        hf_model_id=local_model or hf_id,
        tokenizer_id=local_tokenizer or tokenizer_id,
        official_repo_path=repo_path,
        device=str(device_obj),
        use_mirror=True,
        trust_remote_code=True,
        model_version=model_version,
    )

    logger.info(f"Path A: Kronos {model_version} 加载成功 (device={device_obj})")
    return kronos, "kronos"


def _create_fallback(config, device_obj: torch.device, pretrained_path: str = None):
    """创建 Fallback 模型实例"""
    from src.model_layer.fallback_model import FallbackModel

    fb_config = config.get("model", "fallback", default={})

    # 自动检测训练好的 checkpoint
    if not pretrained_path:
        default_ckpt = Path("models/checkpoints/fallback_best.pt")
        if default_ckpt.exists():
            pretrained_path = str(default_ckpt)

    if pretrained_path and Path(pretrained_path).exists():
        # 加载已有权重
        model = FallbackModel.load(pretrained_path, device=str(device_obj))
        logger.info(f"Path B: 从 {pretrained_path} 加载预训练权重")
    else:
        # 使用配置参数创建新模型
        model = FallbackModel(
            d_model=fb_config.get("d_model", 256),
            n_layers=fb_config.get("n_layers", 6),
            n_heads=fb_config.get("n_heads", 8),
            d_ff=fb_config.get("d_ff", 1024),
            dropout=fb_config.get("dropout", 0.1),
            max_seq_len=fb_config.get("max_seq_len", 512),
            input_dim=5,  # OHLCV
            pred_len=config.get("model", "training", "pred_len", default=5),
        )
        model.to(device_obj)
        logger.info("Path B: 创建新 FallbackModel (随机初始化)")

    model.eval()
    return model, "fallback"


# ═══════════════════════════════════════════════
# 检测辅助函数
# ═══════════════════════════════════════════════

def _check_hf_reachable(config) -> bool:
    """检查 HuggingFace 连接是否可达"""
    try:
        # 尝试使用镜像
        import urllib.request
        import urllib.error

        # 先检查 hf-mirror.com
        try:
            req = urllib.request.Request("https://hf-mirror.com", method="HEAD")
            urllib.request.urlopen(req, timeout=5)
            if "HF_ENDPOINT" not in os.environ:
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            logger.info("HuggingFace 镜像 (hf-mirror.com) 可达")
            return True
        except Exception:
            pass

        # 检查官方站点
        try:
            req = urllib.request.Request("https://huggingface.co", method="HEAD")
            urllib.request.urlopen(req, timeout=10)
            logger.info("HuggingFace 官方站点可达")
            return True
        except Exception:
            pass

        return False
    except Exception:
        return False


def _check_local_cache(config) -> bool:
    """检查本地是否已有 Kronos 模型缓存"""
    from src.model_layer.kronos_loader import _check_model_cached

    if config.get("model", "use_finetuned", default=False):
        finetuned_model = _existing_local_path(
            config.get("model", "finetuned_predictor_path", default=""), require_weights=True
        )
        finetuned_tokenizer_value = config.get("model", "finetuned_tokenizer_path", default="")
        finetuned_tokenizer = Path(finetuned_tokenizer_value)
        if not finetuned_tokenizer.is_absolute():
            from src.common.config import PROJECT_ROOT
            finetuned_tokenizer = PROJECT_ROOT / finetuned_tokenizer
        if finetuned_model and _has_tokenizer_assets(finetuned_tokenizer):
            logger.info(f"检测到完整本地微调 Kronos 权重: {finetuned_model}")
            return True

    local_model = _existing_local_path(
        config.get("model", "pretrained_path", default="models/kronos_base"), require_weights=True
    )
    if local_model:
        logger.info(f"检测到完整本地 Kronos 权重: {local_model}")
        return True

    hf_id = config.get("model", "kronos_hf_id", default="NeoQuasar/Kronos-base")
    return _check_model_cached(hf_id)


def get_model():
    """获取全局模型实例（如已创建）"""
    global _model_instance
    return _model_instance


def get_model_path() -> Optional[str]:
    """获取当前模型路径"""
    global _model_path
    return _model_path


def clear_model_cache() -> None:
    """Release the cached model so the next prediction loads the active registry entry."""
    global _model_instance, _model_path
    _model_instance = None
    _model_path = None
