# 测试结构

测试目录与后端分层对应：

- `test_data_layer/`：行情、A 股市场规则和存储迁移。
- `test_model_layer/`：Kronos 数据集、训练目标、推理和概率校准。
- `test_strategy_layer/`：选股和信号逻辑。
- `test_service_layer/`：接口契约、鉴权和调度。

统一从项目根目录执行：

```powershell
.\venv\Scripts\python.exe -m pytest -q tests
```

不要在项目根目录新增 `test_phase*.py` 或手工打印结果的阶段脚本；验收逻辑应是可重复运行并带断言的 pytest 用例。
