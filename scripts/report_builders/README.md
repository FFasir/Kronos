# scripts/report_builders/ —— 报告/图复现脚本

从 `tmp/_scripts/` 提升到正式位置的交付物构建器（脚本内为绝对路径，移动目录不影响运行）：

| 脚本 | 产物 |
|---|---|
| `make_cicc_report_figs.py` | `docs/训练记录/figs_cicc/` 的 c1–c4 与 d_monthly_params 图 |
| `make_report_figs_2026.py` | `figs_cicc/c5_skill_vs_naive.png`（模型 vs 朴素基准）、`c6_caliber_flip.png`（口径 × 区间翻转） |
| `make_curve_fig_2025_2026.py` | `figs_cicc/c7_curve_2025_2026.png`（2025–2026 连续净值 + 逐月收益；内置「文本框包围盒 vs 曲线」几何重叠自检，运行时会打印 PASS/FAIL） |
| `build_cicc_style_html.py` | `docs/训练记录/20260908_Kronos择时_中证风格报告.html` —— **由报告 md 直接生成，保证 md 与 html 永不脱节** |
| `build_report_pdf.py` | `docs/训练记录/20260908_Kronos择时_报告.pdf` —— 用 playwright/chromium 从上述 HTML 导出 |
| `build_cicc_report.py` | `docs/训练记录/20260908_Kronos择时_通俗报告.html`（通俗版，自包含 HTML） |
| `build_report.py` | `docs/训练记录/_实验过程/20260907_滚动微调对比报告.html`（早期实验版本） |
| `make_report_figs.py` | `docs/训练记录/_实验过程/figs_20260907/fig*.png`（早期实验图） |

## 推荐生成顺序（改了报告 md 之后）

```bash
python scripts/report_builders/make_cicc_report_figs.py        # 基础图（c1-c4, d）
python scripts/report_builders/make_report_figs_2026.py        # c5, c6
python scripts/report_builders/make_curve_fig_2025_2026.py     # c7
python scripts/report_builders/build_cicc_style_html.py        # md -> 中证风格 HTML
python scripts/report_builders/build_report_pdf.py             # HTML -> PDF
python scripts/report_builders/build_cicc_report.py            # 通俗版 HTML（内容独立维护）
```

> 通俗版（`build_cicc_report.py`）的正文写在脚本内的 f-string 里，需要手动同步；
> 中证风格版与 PDF 则完全由 `docs/训练记录/20260908_Kronos择时_报告.md` 派生。
> 图脚本统一使用 `Microsoft YaHei` 渲染中文（本机可用），不会再出现中文方框。
