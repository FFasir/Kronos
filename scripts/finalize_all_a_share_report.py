#!/usr/bin/env python3
"""Write an immutable completion report after the all-A-share training pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def best_epoch(log_path: Path) -> tuple[int | None, float | None]:
    pattern = re.compile(r"--- Epoch (\d+)/\d+ Summary ---\s+Validation Loss: ([\d.]+)")
    matches = [(int(epoch), float(loss)) for epoch, loss in pattern.findall(log_path.read_text(encoding="utf-8", errors="replace"))]
    return min(matches, key=lambda item: item[1]) if matches else (None, None)


def main() -> None:
    model_root = ROOT / "models" / "official_finetune" / "all_a_share"
    tokenizer_dir = model_root / "all_a_share_tokenizer"
    predictor_dir = model_root / "all_a_share_predictor"
    tokenizer_summary = json.loads((tokenizer_dir / "summary.json").read_text(encoding="utf-8"))
    predictor_summary = json.loads((predictor_dir / "summary.json").read_text(encoding="utf-8"))
    tokenizer_epoch, tokenizer_loss = best_epoch(ROOT / "logs" / "all_a_share_tokenizer.log")
    predictor_epoch, predictor_loss = best_epoch(ROOT / "logs" / "all_a_share_predictor.log")
    manifest = json.loads((ROOT / "data" / "processed" / "qlib" / "all_a_share_20260715" / "manifest.json").read_text(encoding="utf-8"))
    tokenizer_weight = tokenizer_dir / "checkpoints" / "best_model" / "model.safetensors"
    predictor_weight = predictor_dir / "checkpoints" / "best_model" / "model.safetensors"

    report = f"""# EXP-20260715-02: 全市场可用 A 股 Kronos-base 微调结果

> 状态：完成
>
> 范围：使用 `daily.zip` 中全部可用个股微调官方 Kronos，不是历史中证1000成分股复现。

## 数据

| 项目 | 值 |
| --- | ---: |
| 原始 ZIP 成员 | {manifest['archive_members']} |
| 有效股票 | {len(manifest['symbols'])} |
| 无效剔除 | {manifest['skipped']['invalid']} |
| 历史不足剔除 | {manifest['skipped']['insufficient_history']} |
| 日历 | {manifest['calendar_start']} 至 {manifest['calendar_end']} |
| 训练窗口 | 8,025,165（每 epoch 抽样 60,000） |
| 验证窗口 | 1,047,034（每 epoch 抽样 6,000） |

## 微调结果

| 阶段 | 最佳 epoch | 日志最佳 validation loss | summary 最佳 validation loss | 权重 SHA256 |
| --- | ---: | ---: | ---: | --- |
| Tokenizer | {tokenizer_epoch} | {tokenizer_loss} | {tokenizer_summary['final_result']['best_val_loss']} | `{sha256(tokenizer_weight)}` |
| Predictor | {predictor_epoch} | {predictor_loss} | {predictor_summary['final_result']['best_val_loss']} | `{sha256(predictor_weight)}` |

最佳权重目录：

```text
{tokenizer_dir.relative_to(ROOT)}/checkpoints/best_model
{predictor_dir.relative_to(ROOT)}/checkpoints/best_model
```

## 结论与限制

该训练验证了全市场可用 A 股 K 线数据对官方 Kronos 的两阶段适配流程。它不使用当前中证1000快照筛选，因而不具备“当前成员回填”的成分股偏差；但原始 ZIP 是否覆盖全部历史退市股尚未独立验证，不能将该实验称为无偏全市场收益研究。

后续择时应使用中证1000生成信号，并以 `512100` ETF 的真实 OHLCV 执行样本外回测。未完成 ETF 回测前，不对策略收益作结论。
"""
    output = ROOT / "docs" / "训练记录" / "20260715_all_a_share_kronos_base_结果.md"
    output.write_text(report, encoding="utf-8")
    print(f"final report written: {output}")


if __name__ == "__main__":
    main()
