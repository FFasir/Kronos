# 命令行脚本索引

`scripts/` 只保存可直接执行的稳定入口；可复用业务逻辑应放在 `src/`。为兼容训练记录和服务器命令，入口保持扁平文件名。

## 数据

- `download.py`：指数、ETF 或指定证券日线下载。
- `import_stock_archive.py`：导入本地 A 股 ZIP 归档。
- `preprocess.py`：项目通用数据预处理。
- `build_qlib_data.py`：从后复权 A 股 ZIP 构建 Qlib 二进制数据。
- `build_monthly_kronos_dataset.py`：生成时间隔离的月度滚动训练集。
- `download_research_data.py`：下载基准、ETF 和成分历史研究数据。

## 模型

- `download_kronos.py`、`load_kronos.py`：准备和检查 Kronos 权重。
- `run_official_finetune.py`：运行 Tokenizer 和 Predictor 两阶段微调。
- `run_monthly_rolling_finetune.py`：运行月度滚动续训、校准和预测。
- `calibrate_monthly_probabilities.py`：拟合时间隔离概率校准器。
- `predict.py`、`predict_monthly_kronos.py`：执行基础或月度模型推理。
- `train.py`、`train_fallback.py`、`finetune.py`：轻量 fallback 模型教学路径。

## 评估与运维

- `evaluate_kronos_checkpoints.py`、`evaluate_monthly_predictions.py`：独立模型评估。
- `run_etf_baselines.py`、`run_paper_walk_forward.py`：基线和滚动回测。
- `finalize_all_a_share_report.py`：整理全 A 股训练结果。
- `init_db.py`、`verify_deployment.py`：初始化数据库和部署验收。
- `param_search.py`：策略参数搜索。

临时脚本不要放入本目录；一次性检查应写成 `tests/` 下的 pytest 用例，生成的发布包写入 `tmp/`。
