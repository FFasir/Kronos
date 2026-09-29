"""
Kronos 指数/ETF 择时策略系统 — 自定义异常

基于设计文档 (附D) 定义的错误码体系。
"""


class KronosError(Exception):
    """所有系统异常的基类"""

    def __init__(self, message: str, error_code: str = "E000"):
        self.error_code = error_code
        super().__init__(f"[{error_code}] {message}")


# ---- 数据层异常 ----

class DataNotFoundError(KronosError):
    """数据未找到（E001）"""

    def __init__(self, message: str = "数据未找到"):
        super().__init__(message, "E001")


class DataLeakError(KronosError):
    """数据泄露检测失败（E002）：训练集日期 >= 验证集日期"""

    def __init__(self, train_max_date: str = "", val_min_date: str = ""):
        msg = f"数据泄露: 训练集最大日期({train_max_date}) >= 验证集最小日期({val_min_date})"
        super().__init__(msg, "E002")


# ---- 模型层异常 ----

class ModelLoadError(KronosError):
    """模型加载失败（E003）"""

    def __init__(self, message: str = "模型加载失败"):
        super().__init__(message, "E003")


class OOMError(KronosError):
    """显存/内存不足（E004）"""

    def __init__(self, message: str = "显存/内存不足"):
        super().__init__(message, "E004")


class CheckpointCorruptError(KronosError):
    """Checkpoint文件损坏（E005）"""

    def __init__(self, path: str = ""):
        super().__init__(f"Checkpoint文件损坏: {path}", "E005")


class ValidationError(KronosError):
    """数据/参数验证失败（E006）"""

    def __init__(self, message: str = "验证失败"):
        super().__init__(message, "E006")


class ConvergenceError(KronosError):
    """模型不收敛（E007）"""

    def __init__(self, message: str = "模型训练不收敛"):
        super().__init__(message, "E007")


class PredictionShapeError(KronosError):
    """预测输出形状异常（E008）"""

    def __init__(self, expected: str = "", actual: str = ""):
        super().__init__(f"预测输出形状异常: 期望{expected}, 实际{actual}", "E008")


# ---- 策略层异常 ----

class BacktestError(KronosError):
    """回测引擎异常"""

    def __init__(self, message: str = "回测执行失败"):
        super().__init__(message, "E100")


class SignalGenerationError(KronosError):
    """信号生成异常"""

    def __init__(self, message: str = "信号生成失败"):
        super().__init__(message, "E101")


# ---- 服务层异常 ----

class APIError(KronosError):
    """API层异常"""

    def __init__(self, message: str = "API请求失败", status_code: int = 500):
        self.status_code = status_code
        super().__init__(message, "E200")
