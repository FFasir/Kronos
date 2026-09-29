# Kronos 择时系统 - 常用工具

这个目录包含日常开发/运维的便捷工具脚本。

## 可用工具

### 1. 快速预测 (`quick_predict.py`)
获取最新市场数据的预测信号（用于验证模型是否正常工作）

```bash
python scripts/tools/quick_predict.py
# 输出：6 个标的的预测收益率、信号类型
```

### 2. 回测验证 (`verify_backtest.py`)
快速跑一个短窗口回测，验证无未来泄露 + 模型可用性

```bash
python scripts/tools/verify_backtest.py [--days 30]
# 输出：绩效指标、交易笔数、是否有异常
```

### 3. 模型健康检查 (`check_model.py`)
检查模型 checkpoint 是否存在、可加载、预测是否合理

```bash
python scripts/tools/check_model.py
# 输出：模型路径、参数量、单次预测结果、是否正常
```

### 4. 数据库状态 (`db_status.py`)
查看 SQLite 数据库各表的记录数、最新日期

```bash
python scripts/tools/db_status.py
# 输出：各表行数、最新预测/交易日期
```

### 5. 日志查看器 (`tail_logs.py`)
聚合查看训练、回测、API 日志尾部

```bash
python scripts/tools/tail_logs.py [--lines 20]
# 输出：多个日志文件的最后 N 行
```

## 使用场景

- **每日上线前验证**：`check_model.py` + `quick_predict.py`
- **代码改动后冒烟**：`verify_backtest.py`
- **排查问题**：`tail_logs.py` + `db_status.py`
