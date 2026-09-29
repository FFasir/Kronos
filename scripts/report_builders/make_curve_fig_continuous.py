# -*- coding: utf-8 -*-
"""生成 2025-01~2026-08 连续 20 个月测试曲线图（c7，无缺口版）。

读取本地归档的连续回测产物：
  data/server_outputs_20260908/base_spearman_2025_2026_20m/
  data/server_outputs_20260908/base_fixed_2025win_2025_2026_20m/
输出：docs/训练记录/figs_cicc/c7_curve_2025_2026.png
内置「文本框包围盒 vs 曲线」几何重叠自检（打印 PASS/FAIL）。
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parents[2]
SRV = BASE / "data" / "server_outputs_20260908"
FIG = BASE / "docs" / "训练记录" / "figs_cicc"

SPEARMAN = "base_spearman_2025_2026_20m"
FIXED = "base_fixed_2025win_2025_2026_20m"


def load(sub):
    d = SRV / sub
    eq = pd.read_csv(d / "daily_equity.csv", parse_dates=["date"])
    sm = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    return eq, sm


s, sm_s = load(SPEARMAN)
f, sm_f = load(FIXED)

fig = plt.figure(figsize=(13.8, 8.8))
gs = fig.add_gridspec(2, 1, height_ratios=[2.05, 1], hspace=0.30)
ax = fig.add_subplot(gs[0])

# 持仓期底纹
ax.fill_between(s["date"], 0.90, 1.30, where=s["in_market"].to_numpy(),
                color="#f2c9c6", alpha=0.45, step="post", linewidth=0)

ax.plot(s["date"], s["equity"], color="#c00000", lw=2.2,
        label="Kronos base + Spearman 选参（原定稿方案）")
ax.plot(f["date"], f["equity"], color="#2e7d32", lw=1.7,
        label="固定一套参数（不选参，F2）")
ax.plot(s["date"], s["benchmark_equity"], color="#1f4e79", lw=1.8, ls="--",
        label="买入持有（中证1000）")

# 年份分界
ax.axvline(pd.Timestamp("2026-01-01"), color="#888", ls=":", lw=1.4)
ax.text(pd.Timestamp("2025-07-01"), 0.972, "测试区间①：2025-01~12", fontsize=9.5,
        color="#1f4e79", ha="center", va="center")
ax.text(pd.Timestamp("2026-05-05"), 0.972, "测试区间②：2026-01~08（样本外）", fontsize=9.5,
        color="#1f4e79", ha="center", va="center")

# 2026-07 事件
ax.annotate("2026-07 三连亏\n−6.8% / −4.5% / −3.8%",
            xy=(pd.Timestamp("2026-07-16"), 1.16), xytext=(pd.Timestamp("2026-01-10"), 0.972),
            fontsize=8.8, color="#c00000", ha="center", va="center",
            arrowprops=dict(arrowstyle="->", color="#c00000", lw=1.1,
                            connectionstyle="arc3,rad=-0.15"))

ax.set_ylim(0.94, 1.62)
ax.set_ylabel("累计净值（起点 = 1.0）", fontsize=11)
ax.set_title("中证1000 · Kronos 择时：2025-01 ~ 2026-08 连续 20 个月测试曲线（净收益，含成本）",
             fontsize=13)
ax.grid(alpha=0.3)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

for series, col, color, name in [
    (s, "equity", "#c00000", "Spearman选参"),
    (f, "equity", "#2e7d32", "固定参数"),
    (s, "benchmark_equity", "#1f4e79", "买入持有"),
]:
    v = series[col].iloc[-1]
    ax.text(pd.Timestamp("2026-09-08"), v, f"{v:.3f}  ({v-1:+.1%})", fontsize=9,
            color=color, va="center")

ax.legend(loc="upper left", ncol=1, fontsize=9.5, framealpha=0.95)
ax.text(0.5, -0.115, "说明：单一连续回测（2025-01~2026-08，无缺口）；每月以前一月为验证集独立选参；浅红底纹 = 原定稿方案持仓期",
        transform=ax.transAxes, ha="center", fontsize=8.5, color="#666")

# ===== 下：逐月收益 =====
ax2 = fig.add_subplot(gs[1])
g = s.copy()
g["m"] = g["date"].dt.to_period("M")
rows, prev_e, prev_b = [], 1.0, 1.0
for m, sub in g.groupby("m"):
    e, b = sub["equity"].iloc[-1], sub["benchmark_equity"].iloc[-1]
    rows.append((str(m), e / prev_e - 1, b / prev_b - 1))
    prev_e, prev_b = e, b
ms = pd.DataFrame(rows, columns=["month", "strat", "bench"])

x = np.arange(len(ms))
w = 0.4
ax2.bar(x - w / 2, ms["strat"] * 100, w, label="策略（Spearman 选参）", color="#c00000")
ax2.bar(x + w / 2, ms["bench"] * 100, w, label="买入持有", color="#1f4e79", alpha=0.85)
ax2.axhline(0, color="#333", lw=0.9)
n2025 = int((ms["month"] < "2026").sum())
ax2.axvline(n2025 - 0.5, color="#888", ls=":", lw=1.4)
ax2.text((n2025 - 1) / 2, 14.0, "2025", ha="center", fontsize=10, color="#1f4e79")
ax2.text((n2025 + len(ms) - 1) / 2, 14.0, "2026", ha="center", fontsize=10, color="#1f4e79")
ax2.set_xticks(x)
ax2.set_xticklabels([m[2:] for m in ms["month"]], fontsize=8)
ax2.set_ylabel("月收益（%）", fontsize=10)
ax2.set_title("逐月收益对比（柱顶数字为各柱月收益）", fontsize=11)
ax2.legend(fontsize=9, loc="lower left")
ax2.grid(axis="y", alpha=0.3)
ax2.set_ylim(-26, 18)
for i, (sv, bv) in enumerate(zip(ms["strat"], ms["bench"])):
    for off_x, v, color in [(-w / 2, sv, "#c00000"), (w / 2, bv, "#1f4e79")]:
        y = v * 100
        # 小柱（|y| 很小）也强制离开零轴，避免标签压在坐标轴上
        if y >= 0:
            y_lab, va = max(y + 0.7, 1.5), "bottom"
        else:
            y_lab, va = min(y - 0.7, -1.5), "top"
        ax2.text(i + off_x, y_lab, f"{y:+.1f}", ha="center", va=va,
                 fontsize=7.2, color=color)

fig.subplots_adjust(top=0.93, bottom=0.075, left=0.065, right=0.935)

# ===== 几何校验 =====
fig.canvas.draw()
inv = ax.transData.inverted()

def bbox_data(obj):
    bb = obj.get_window_extent(fig.canvas.get_renderer())
    (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    t0 = pd.Timestamp(mdates.num2date(x0)).tz_localize(None)
    t1 = pd.Timestamp(mdates.num2date(x1)).tz_localize(None)
    return t0, float(y0), t1, float(y1)

curves = [("Spearman", s, "equity"), ("Fixed", f, "equity"), ("BuyHold", s, "benchmark_equity")]

def check(name, obj, pad=0.004):
    x0, y0, x1, y1 = bbox_data(obj)
    bad = []
    for cn, df, col in curves:
        m = (df["date"] >= x0) & (df["date"] <= x1)
        if not m.any():
            continue
        v = df.loc[m, col]
        if not (v.max() < y0 - pad or v.min() > y1 + pad):
            bad.append(f"{cn}({v.min():.3f}~{v.max():.3f})")
    print(f"  [{'PASS' if not bad else 'FAIL ' + ','.join(bad)}] {name:<26} "
          f"x[{x0:%Y-%m-%d}~{x1:%Y-%m-%d}] y[{y0:.3f}~{y1:.3f}]")

print("=== 文本/曲线几何校验 ===")
for t in ax.texts:
    check(t.get_text().split("\n")[0][:22], t)
if ax.get_legend() is not None:
    check("图例", ax.get_legend().get_frame())

fig.savefig(FIG / "c7_curve_2025_2026.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("wrote c7_curve_2025_2026.png")
print()
print("连续 20 个月：Spearman选参 %.2f%% | 固定参数 %.2f%% | 买入持有 %.2f%%" % (
    (s["equity"].iloc[-1] - 1) * 100, (f["equity"].iloc[-1] - 1) * 100,
    (s["benchmark_equity"].iloc[-1] - 1) * 100))
print("补齐区间 2025-10~12 表现：")
print(ms[ms["month"].str.startswith(("2025-10", "2025-11", "2025-12"))].to_string(index=False))
