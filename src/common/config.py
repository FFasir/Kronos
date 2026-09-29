"""
Kronos 指数/ETF 择时策略系统 — 配置管理模块

从 YAML 配置文件加载系统配置，提供统一配置访问接口。
使用单例模式确保全局配置一致性。
"""

import os
import yaml
from pathlib import Path
from typing import Any, Optional


# 项目根目录（2级向上，从 src/common/ 到项目根）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "system.yaml"


class Config:
    """系统配置单例"""

    _instance: Optional["Config"] = None
    _data: dict = {}

    def __new__(cls, config_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load(config_path)
        return cls._instance

    def _load(self, config_path: Optional[str] = None):
        path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")

        with open(path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f)

        active_model_path = _PROJECT_ROOT / "config" / "active_model.json"
        if active_model_path.is_file():
            import json

            override = json.loads(active_model_path.read_text(encoding="utf-8"))
            model_override = override.get("model", {})
            if isinstance(model_override, dict):
                self._data.setdefault("model", {}).update(model_override)

        # 路径标准化：相对路径 → 绝对路径（基于项目根目录）
        self._resolve_paths()

    def _resolve_paths(self):
        """将配置中的相对路径转为绝对路径"""
        path_keys = [
            ("data", "raw_dir"),
            ("data", "processed_dir"),
            ("data", "outputs_dir"),
            ("database", "sqlite_path"),
            ("logging", "dir"),
        ]
        for section, key in path_keys:
            if section in self._data and key in self._data[section]:
                val = self._data[section][key]
                if not os.path.isabs(val):
                    self._data[section][key] = str(_PROJECT_ROOT / val)

    # ---- 便捷访问 ----

    @property
    def server(self) -> dict:
        return self._data.get("server", {})

    @property
    def data(self) -> dict:
        return self._data.get("data", {})

    @property
    def model(self) -> dict:
        return self._data.get("model", {})

    @property
    def strategy(self) -> dict:
        return self._data.get("strategy", {})

    @property
    def backtest(self) -> dict:
        return self._data.get("backtest", {})

    @property
    def database(self) -> dict:
        return self._data.get("database", {})

    @property
    def logging_config(self) -> dict:
        return self._data.get("logging", {})

    @property
    def seed(self) -> int:
        return self._data.get("reproducibility", {}).get("random_seed", 42)

    def get(self, *keys, default=None) -> Any:
        """深层取值: config.get('model', 'training', 'epochs')"""
        val = self._data
        for key in keys:
            if isinstance(val, dict):
                val = val.get(key)
            else:
                return default
        return val if val is not None else default

    def get_index_list(self) -> list[dict]:
        """获取指数列表"""
        return self._data.get("data", {}).get("index_list", [])

    def get_etf_list(self) -> list[dict]:
        """获取ETF列表"""
        return self._data.get("data", {}).get("etf_list", [])

    def get_all_symbols(self) -> list[dict]:
        """获取所有标的（指数+ETF）"""
        return self.get_index_list() + self.get_etf_list()

    @property
    def raw_data(self) -> dict:
        return self._data


# 重置单例（测试用）
def reset_config():
    Config._instance = None


# 便捷函数
def load_config(config_path: Optional[str] = None) -> Config:
    """加载配置（单例）"""
    return Config(config_path)


PROJECT_ROOT = _PROJECT_ROOT
