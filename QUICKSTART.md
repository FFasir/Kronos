# Kronos A股分析策略系统 — 快速启动指南

> 本文面向**第一次拿到这份代码的人**，按顺序做完即可把系统跑起来。
> 代码仓库只包含源码，不包含行情数据和模型产物，这两样需要按第 3 步补齐。

---

## 0. 前置要求

| 依赖 | 版本要求 | 检查命令 |
|---|---|---|
| Python | ≥ 3.10 | `python --version` |
| Node.js | ≥ 18 | `node --version` |
| Git | 任意 | `git --version` |

---

## 1. 克隆代码

```bash
git clone https://github.com/FFasir/Kronos.git
cd Kronos
```

---

## 2. 安装依赖

### 后端（Python）

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
# python -m venv venv
# source venv/bin/activate

pip install -r requirements.txt
```

> 说明：`requirements.txt` 里包含模型微调用的 `torch`、`pyqlib`、`comet_ml` 等重型依赖。
> 如果你**只跑前后端 + 行情数据**、暂时不训练模型，可以先只装 Web 相关依赖：
> `pip install fastapi "uvicorn[standard]" pydantic pyyaml pandas numpy tables h5py akshare sqlalchemy apscheduler httpx pytest`

### 前端（Node）

```bash
cd frontend
npm install
```

---

## 3. 准备数据（代码里没有，必须自己补）

行情数据和模型预测**不在仓库里**，补齐方式如下：

### 3.1 行情数据（二选一）

**方式 A — 联网下载（推荐）**

```bash
# 全 A 股收盘快照 → 生成股票列表 + 当日行情
python scripts/download_real_snapshot.py

# 指数日线 + 下载 6 只 ETF（需要网络）
python scripts/import_local_data.py --etfs
```

**方式 B — 找团队要归档**

向项目成员索取以下归档，放到 `data/` 目录（与代码同级，**不进入 git**）：

- `daily_hfq.zip` — 全 A 股后复权日线（约 500MB）
- `指数_日_kline.zip` — 指数日线
- `A股数据/股票列表.csv` — 股票目录

然后执行：

```bash
python scripts/import_local_data.py
```

### 3.2 模型预测

选股池、交易信号、回测、预测 K 线这些功能依赖 Kronos 模型产出，**在没有模型预测之前这些页面是空的**。模型预测的产出落点：

- 数据库表：`model_prediction` / `trading_signal` / `price_prediction`
- 或文件：`data/outputs/base_csi1000_etf_*/test_predictions.csv`

模型代码在 `third_party/Kronos/`。权重为公开资源，执行 `python scripts/download_kronos.py`
从 Hugging Face 拉取 `NeoQuasar/Kronos-base` 与对应 tokenizer 到 `models/`（该目录不入库）。

> 只想**先预览界面效果**、不依赖真实模型，可以注入 mock 数据：
> ```bash
> python scripts/seed_mock_data.py
> ```

---

## 4. 启动服务

### 4.1 后端（终端 A）

```bash
python -m uvicorn src.service_layer.web_server:app --host 127.0.0.1 --port 8000 --reload
```

- API 文档：http://127.0.0.1:8000/docs

### 4.2 前端（终端 B）

```bash
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

- 页面地址：http://127.0.0.1:5173

---

## 5. 验证

```bash
# 后端测试
python -m pytest -q -s tests

# 前端类型检查 + 构建
cd frontend
npm run build
```

---

## 6. 常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| 仪表盘"股票池数量"= 0 | 没有模型预测，`stock_selection` 表为空 | 见 3.2，等模型产出或跑 mock |
| 选股池/信号/回测页面空 | 同上 | 同上 |
| 回测报"未找到冻结滚动预测文件" | 缺 `data/outputs/base_csi1000_etf_*/test_predictions.csv` | 按 3.2 跑一次模型预测 |
| akshare 下载超时 | 本机开了代理（如 Clash）拦截了连接 | 脚本内已设 `NO_PROXY="*"`，若仍失败检查系统代理 |
| 前端连不上后端 | 端口或 CORS 不对 | 确认后端在 8000 端口，前端在 5173 |

---

## 7. 目录结构速览

```text
src/            后端分层代码（common/data/model/strategy/service/scheduler）
frontend/       React + Vite 前端
scripts/        命令行入口（下载、导入、训练、评估）
config/         运行配置（system.yaml）
deploy/         生产部署脚本
tests/          pytest 测试
third_party/    Kronos 模型源码（带本项目补丁）
docs/           需求、验收、训练记录
data/           行情与预测产物（运行生成，不进 git）
models/         模型权重（不进 git）
```
