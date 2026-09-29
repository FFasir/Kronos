import json

import pytest

from src.model_layer import model_registry


def _make_assets(root, predictor="models/custom/predictor", tokenizer="models/custom/tokenizer"):
    predictor_dir = root / predictor
    tokenizer_dir = root / tokenizer
    predictor_dir.mkdir(parents=True)
    tokenizer_dir.mkdir(parents=True)
    (predictor_dir / "model.safetensors").write_bytes(b"p" * (10 * 1024 * 1024))
    (tokenizer_dir / "config.json").write_text("{}", encoding="utf-8")
    (tokenizer_dir / "model.safetensors").write_bytes(b"t" * (1024 * 1024))
    return predictor, tokenizer


def test_custom_model_registration_rejects_paths_outside_models(tmp_path, monkeypatch):
    monkeypatch.setattr(model_registry, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(model_registry, "CUSTOM_REGISTRY", tmp_path / "config/custom_models.json")

    with pytest.raises(ValueError, match="inside"):
        model_registry.register_custom_model("bad", "v1", "../outside", "models/tokenizer")


def test_custom_model_registration_and_activation_are_persistent(tmp_path, monkeypatch):
    predictor, tokenizer = _make_assets(tmp_path)
    monkeypatch.setattr(model_registry, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(model_registry, "CUSTOM_REGISTRY", tmp_path / "config/custom_models.json")
    monkeypatch.setattr(model_registry, "ACTIVE_MODEL", tmp_path / "config/active_model.json")
    monkeypatch.setattr("src.model_layer.model_factory.clear_model_cache", lambda: None)

    registered = model_registry.register_custom_model("Custom", "v1", predictor, tokenizer)
    result = model_registry.activate_model(registered["key"])

    assert result["success"] is True
    active = json.loads(model_registry.ACTIVE_MODEL.read_text(encoding="utf-8"))
    assert active["key"] == registered["key"]
    assert active["model"]["use_finetuned"] is True
