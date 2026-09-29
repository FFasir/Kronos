"""Validated registry and activation state for production Kronos models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CUSTOM_REGISTRY = PROJECT_ROOT / "config" / "custom_models.json"
ACTIVE_MODEL = PROJECT_ROOT / "config" / "active_model.json"

BUILTIN_MODELS: tuple[dict[str, Any], ...] = (
    {
        "key": "kronos_base",
        "name": "Kronos Base",
        "version": "base-selected-20260716",
        "predictor_path": "models/kronos_base",
        "tokenizer_path": "models/kronos_tokenizer",
        "kind": "builtin",
        "research_status": "selected",
    },
    {
        "key": "legacy_all_a_share",
        "name": "旧全 A 微调",
        "version": "all-a-share-20260715",
        "predictor_path": "models/official_finetune/all_a_share/all_a_share_predictor/checkpoints/best_model",
        "tokenizer_path": "models/official_finetune/all_a_share/all_a_share_tokenizer/checkpoints/best_model",
        "kind": "builtin",
        "research_status": "not_selected",
    },
    {
        "key": "new_all_a_share_hfq_v2",
        "name": "新全 A / HFQ v2",
        "version": "all-a-share-hfq-v2-20260716",
        "predictor_path": "models/official_finetune/all_a_share_hfq_v2_20260715/predictor/checkpoints/best_model",
        "tokenizer_path": "models/official_finetune/all_a_share_hfq_v2_20260715/tokenizer/checkpoints/best_model",
        "kind": "builtin",
        "research_status": "not_selected",
    },
    {
        "key": "legacy_csi1000",
        "name": "旧 CSI1000 微调",
        "version": "legacy-csi1000-20260715",
        "predictor_path": "models/official_finetune/csi1000_predictor/checkpoints/best_model",
        "tokenizer_path": "models/official_finetune/csi1000_tokenizer/checkpoints/best_model",
        "kind": "builtin",
        "research_status": "not_selected",
    },
)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_custom_models() -> list[dict[str, Any]]:
    if not CUSTOM_REGISTRY.is_file():
        return []
    data = json.loads(CUSTOM_REGISTRY.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def _resolve_model_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (PROJECT_ROOT / path).resolve()
    models_root = (PROJECT_ROOT / "models").resolve()
    if not resolved.is_relative_to(models_root):
        raise ValueError("model paths must stay inside the project models directory")
    return resolved


def _model_is_ready(model: dict[str, Any]) -> tuple[bool, str | None]:
    try:
        predictor = _resolve_model_path(str(model["predictor_path"]))
        tokenizer = _resolve_model_path(str(model["tokenizer_path"]))
    except (KeyError, ValueError) as exc:
        return False, str(exc)

    predictor_weights = [*predictor.glob("*.safetensors"), *predictor.glob("pytorch_model*.bin")]
    tokenizer_assets = [
        *tokenizer.glob("*.safetensors"),
        *tokenizer.glob("*.bin"),
        *tokenizer.glob("*.model"),
        *tokenizer.glob("*.pt"),
    ]
    if not any(path.is_file() and path.stat().st_size >= 10 * 1024 * 1024 for path in predictor_weights):
        return False, "predictor weights are missing or incomplete"
    if not (tokenizer / "config.json").is_file():
        return False, "tokenizer config.json is missing"
    if not any(path.is_file() and path.stat().st_size >= 1024 * 1024 for path in tokenizer_assets):
        return False, "tokenizer assets are missing or incomplete"
    return True, None


def _active_key() -> str:
    if ACTIVE_MODEL.is_file():
        try:
            return str(json.loads(ACTIVE_MODEL.read_text(encoding="utf-8")).get("key", "kronos_base"))
        except json.JSONDecodeError:
            pass
    return "kronos_base"


def active_model_domain() -> str:
    """Classify the active model's training/target domain from its registry metadata.

    Returns:
        "all_a_share"  — cross-sectional A-share model (suitable for --all-a-share)
        "index_timing" — single-index/ETF timing model (000300/510300, CSI1000 ...)
        "base"         — Kronos base (no domain-specific fine-tune)
        "unknown"      — unclassifiable custom model

    The classifier is deliberately conservative: a custom model that is not
    recognizably an all-A-share model is NOT treated as one, so a bulk
    all-market run can never silently happen under the wrong model.
    """
    raw = {}
    if ACTIVE_MODEL.is_file():
        try:
            raw = json.loads(ACTIVE_MODEL.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raw = {}

    key = str(raw.get("key", "kronos_base"))
    model_cfg = raw.get("model", {}) if isinstance(raw.get("model"), dict) else {}
    version = str(model_cfg.get("finetuned_model_version") or raw.get("version") or "")

    custom_names = {
        item.get("key"): item.get("name", "")
        for item in _read_custom_models()
        if isinstance(item, dict)
    }
    name = custom_names.get(key, "")
    text = " ".join((key, version, name)).lower()

    all_a_share_hits = (
        "all_a_share", "all-a-share", "all a share", "全a", "全 a", "全a股", "全a-share",
    )
    if any(hit in text for hit in all_a_share_hits):
        return "all_a_share"

    timing_hits = (
        "timing", "idx300", "idx500", "idx1000", "csi1000", "csi300", "csi500",
        "择时", "指数",
    )
    if any(hit in text for hit in timing_hits):
        return "index_timing"

    if key == "kronos_base" or not model_cfg.get("use_finetuned"):
        return "base"
    return "unknown"


def list_models() -> dict[str, Any]:
    active = _active_key()
    models = [dict(model) for model in BUILTIN_MODELS] + _read_custom_models()
    for model in models:
        ready, reason = _model_is_ready(model)
        model["ready"] = ready
        model["validation_error"] = reason
        model["active"] = model["key"] == active
    return {"active_key": active, "models": models, "restart_required": False}


def register_custom_model(name: str, version: str, predictor_path: str, tokenizer_path: str) -> dict[str, Any]:
    normalized_predictor = _resolve_model_path(predictor_path).relative_to(PROJECT_ROOT).as_posix()
    normalized_tokenizer = _resolve_model_path(tokenizer_path).relative_to(PROJECT_ROOT).as_posix()
    digest = hashlib.sha256(f"{version}\0{normalized_predictor}\0{normalized_tokenizer}".encode()).hexdigest()[:12]
    model = {
        "key": f"custom-{digest}",
        "name": name.strip(),
        "version": version.strip(),
        "predictor_path": normalized_predictor,
        "tokenizer_path": normalized_tokenizer,
        "kind": "custom",
        "research_status": "unverified",
    }
    if not model["name"] or not model["version"]:
        raise ValueError("name and version are required")
    ready, reason = _model_is_ready(model)
    if not ready:
        raise ValueError(reason or "model assets are incomplete")
    custom = [item for item in _read_custom_models() if item.get("key") != model["key"]]
    custom.append(model)
    _write_json_atomic(CUSTOM_REGISTRY, custom)
    return {**model, "ready": True, "validation_error": None, "active": False}


def activate_model(key: str) -> dict[str, Any]:
    models = [dict(model) for model in BUILTIN_MODELS] + _read_custom_models()
    model = next((item for item in models if item.get("key") == key), None)
    if model is None:
        raise KeyError(key)
    ready, reason = _model_is_ready(model)
    if not ready:
        raise ValueError(reason or "model assets are incomplete")

    if key == "kronos_base":
        model_config = {
            "use_finetuned": False,
            "pretrained_path": model["predictor_path"],
            "tokenizer_path": model["tokenizer_path"],
        }
    else:
        model_config = {
            "use_finetuned": True,
            "finetuned_predictor_path": model["predictor_path"],
            "finetuned_tokenizer_path": model["tokenizer_path"],
            "finetuned_model_version": model["version"],
        }
    _write_json_atomic(
        ACTIVE_MODEL,
        {"key": model["key"], "version": model["version"], "model": model_config},
    )

    from src.common.config import reset_config
    from src.model_layer.model_factory import clear_model_cache
    from src.service_layer.prediction_service import reset_prediction_service

    reset_config()
    clear_model_cache()
    reset_prediction_service()
    return {"success": True, "active_key": key, "model": model, "message": "模型已切换，后续预测任务将加载新权重"}


def delete_custom_model(key: str) -> None:
    if _active_key() == key:
        raise ValueError("the active model cannot be removed")
    custom = _read_custom_models()
    updated = [item for item in custom if item.get("key") != key]
    if len(updated) == len(custom):
        raise KeyError(key)
    _write_json_atomic(CUSTOM_REGISTRY, updated)
