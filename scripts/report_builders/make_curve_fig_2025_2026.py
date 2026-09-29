# -*- coding: utf-8 -*-
"""生成 2025-2026 连续测试曲线图（c7）：累计净值 + 逐月收益对比。"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parents[2]
SRV = BASE / "data" / "server_outputs_20260908"
FIG = BASE / "docs" / "训练记录" / "figs_cicc"

SEG = {
    "s": ("base_walkforward_spearman_2025_9m", "base_walkforward_spearman_2026_8m"),
    "f2": ("base_fixed_2025win_2025_9m", "base_fixed_2025win_2026_8m"),
}

def load(sub):
    eq = pd.read_csv(SRV / sub / "daily_equity.csv", parse_dates=["date"])
    sm = json.loads((SRV / sub / "summary.json").read_text(encoding="utf-8"))
    return eq, sm

def chain(dirs):
    """把两段权益曲线按累计收益拼接（段间有 2025-10~12 的数据缺口）。"""
    e1, s1 = load(dirs[0])
    e2, s2 = load(dirs[1])
    m_e = e1["equity"].iloc[-1]
    m_b = e1["benchmark_equity"].iloc[-1]
    p1 = e1.copy()
    p2 = e2.copy()
    p2["equity"] = p2["equity"] * m_e
    p2["benchmark_equity"] = p2["benchmark_equity"] * m_b
    return p1, p2, s1, s2

s1, s2, sm1, sm2 = chain(SEG["s"])
f1, f2, fm1, fm2 = chain(SEG["f2"])

# ---------------- 图 ----------------
fig = plt.figure(figsize=(13.5, 8.6))
gs = fig.add_gridspec(2, 1, height_ratios=[2.05, 1], hspace=0.28)

# ===== 上：累计净值 =====
ax = fig.add_subplot(gs[0])

# 在市区间底纹（原定稿方案）
for seg in (s1, s2):
    inm = seg["in_market"].to_numpy()
    ax.fill_between(seg["date"], 0.90, 1.42, where=inm, color="#f2c9c6", alpha=0.45,
                    step="post", linewidth=0, label=None)

ax.plot(pd.concat([s1, s2])["date"], pd.concat([s1, s2])["equity"],
        color="#c00000", lw=2.2, label="Kronos base + Spearman 选参（原定稿方案）")
ax.plot(pd.concat([f1, f2])["date"], pd.concat([f1, f2])["equity"],
        color="#2e7d32", lw=1.7, ls="-", alpha=0.9, label="固定一套参数（不选参，F2）")
ax.plot(pd.concat([s1, s2])["date"], pd.concat([s1, s2])["benchmark_equity"],
        color="#1f4e79", lw=1.8, ls="--", label="买入持有（中证1000）")

# 区间分界与缺口
ax.axvline(pd.Timestamp("2025-09-30"), color="#888", ls=":", lw=1.4)
ax.axvspan(pd.Timestamp("2025-09-30"), pd.Timestamp("2026-01-05"), color="#dddddd", alpha=0.55)
ax.text(pd.Timestamp("2025-11-17"), 1.115, "缺口\n2025-10~12\n未测试", ha="center", va="center",
        fontsize=9, color="#555")

# 段落标注（放在曲线下方的空白带；配合下方几何校验确证无重叠）
ax.text(pd.Timestamp("2025-06-15"), 0.972, "测试区间①：2025-01~09\n（原始定稿区间）",
        fontsize=9.5, color="#1f4e79", ha="center", va="center")
ax.text(pd.Timestamp("2026-05-05"), 0.972, "测试区间②：2026-01~08\n（样本外新数据）",
        fontsize=9.5, color="#1f4e79", ha="center", va="center")

# 关键事件标注
ax.annotate("2026-07 三连亏\n−6.8% / −4.5% / −3.8%",
            xy=(pd.Timestamp("2026-07-16"), 1.155), xytext=(pd.Timestamp("2026-01-10"), 0.972),
            fontsize=8.8, color="#c00000", ha="center", va="center",
            arrowprops=dict(arrowstyle="->", color="#c00000", lw=1.1,
                            connectionstyle="arc3,rad=-0.15"))

ax.set_ylim(0.94, 1.56)
ax.set_ylabel("累计净值（起点 = 1.0）", fontsize=11)
ax.set_title("中证1000 · Kronos 择时：2025-2026 连续测试曲线（净收益，含成本）", fontsize=13)
ax.grid(alpha=0.3)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax.get_xticklabels(), rotation=0, fontsize=9)

# 末值标注
for series, xoff, color, name in [
    (pd.concat([s1, s2]), 0, "#c00000", "Spearman 选参"),
    (pd.concat([f1, f2]), 0, "#2e7d32", "固定参数"),
    (pd.concat([s1, s2]), 0, "#1f4e79", "买入持有"),
]:
    col = "equity" if name != "买入持有" else "benchmark_equity"
    v = series[col].iloc[-1]
    ax.text(pd.Timestamp("2026-09-08"), v, f"{v:.3f}  ({v-1:+.1%})", fontsize=9,
            color=color, va="center")

leg = ax.legend(loc="upper center", ncol=3, fontsize=9.5, framealpha=0.95)
ax.text(0.5, -0.115, "说明：两段区间各自独立选参（验证月=测试月前一月）；浅红底纹 = 原定稿方案持仓期",
        transform=ax.transAxes, ha="center", fontsize=8.5, color="#666")

# ===== 下：逐月收益 =====
ax2 = fig.add_subplot(gs[1])

def monthly(raw1, raw2):
    """逐月收益：必须用未缩放的原始分段（每段权益各自从 1.0 起），否则首月会把累计值当收益。"""
    rows = []
    for seg in (raw1, raw2):
        g = seg.copy()
        g["m"] = g["date"].dt.to_period("M")
        prev_e, prev_b = 1.0, 1.0
        for m, sub in g.groupby("m"):
            e, b = sub["equity"].iloc[-1], sub["benchmark_equity"].iloc[-1]
            rows.append((str(m), e / prev_e - 1, b / prev_b - 1))
            prev_e, prev_b = e, b
    return pd.DataFrame(rows, columns=["month", "strat", "bench"])

raw_s1, _ = load(SEG["s"][0])
raw_s2, _ = load(SEG["s"][1])
ms = monthly(raw_s1, raw_s2)
x = np.arange(len(ms))
w = 0.4
ax2.bar(x - w / 2, ms["strat"] * 100, w, label="策略（Spearman 选参）", color="#c00000")
ax2.bar(x + w / 2, ms["bench"] * 100, w, label="买入持有", color="#1f4e79", alpha=0.85)
ax2.axhline(0, color="#333", lw=0.9)
ax2.axvline(8.5, color="#888", ls=":", lw=1.4)
ax2.text(4.0, 13.4, "2025", ha="center", fontsize=10, color="#1f4e79")
ax2.text(12.5, 13.4, "2026", ha="center", fontsize=10, color="#1f4e79")
ax2.set_xticks(x)
ax2.set_xticklabels([m[2:] for m in ms["month"]], fontsize=8.5)
ax2.set_ylabel("月收益（%）", fontsize=10)
ax2.set_title("逐月收益对比（柱顶数字为各柱月收益；横轴按交易日顺序排列，点线处为 2025-10~12 缺口）",
              fontsize=11)
ax2.legend(fontsize=9, loc="lower left")
ax2.grid(axis="y", alpha=0.3)
ax2.set_ylim(-25, 17)
for i, (s, b) in enumerate(zip(ms["strat"], ms["bench"])):
    ax2.text(i - w / 2, s * 100 + (0.7 if s >= 0 else -1.6), f"{s*100:+.1f}",
             ha="center", va="bottom" if s >= 0 else "top", fontsize=7.4, color="#c00000")
    ax2.text(i + w / 2, b * 100 + (0.7 if b >= 0 else -1.6), f"{b*100:+.1f}",
             ha="center", va="bottom" if b >= 0 else "top", fontsize=7.4, color="#1f4e79")

fig.subplots_adjust(top=0.93, bottom=0.075, left=0.065, right=0.935)

# ===== 几何校验：所有文本框包围盒 vs 曲线路径，确保零重叠 =====
fig.canvas.draw()
inv = ax.transData.inverted()

def bbox_data(txt):
    bb = txt.get_window_extent(fig.canvas.get_renderer())
    (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    # 日期轴的反变换给出 matplotlib 日期数值（float），需转回 Timestamp
    t0 = pd.Timestamp(mdates.num2date(x0)).tz_localize(None)
    t1 = pd.Timestamp(mdates.num2date(x1)).tz_localize(None)
    return t0, float(y0), t1, float(y1)

curves = {
    "Spearman": pd.concat([s1, s2]),
    "Fixed": pd.concat([f1, f2]),
    "BuyHold": pd.concat([s1, s2]),
}
colmap = {"Spearman": "equity", "Fixed": "equity", "BuyHold": "benchmark_equity"}

def check(name, txt, pad=0.004):
    x0, y0, x1, y1 = bbox_data(txt)
    problems = []
    for cn, df in curves.items():
        m = (df["date"] >= x0) & (df["date"] <= x1)
        if not m.any():
            continue
        vals = df.loc[m, colmap[cn]]
        lo, hi = vals.min(), vals.max()
        if not (hi < y0 - pad or lo > y1 + pad):
            problems.append(f"{cn}({lo:.3f}~{hi:.3f})")
    status = "PASS" if not problems else "FAIL 与曲线相交: " + ", ".join(problems)
    print(f"  [{status}] {name:<22} bbox x[{x0:%Y-%m-%d}~{x1:%Y-%m-%d}] y[{y0:.3f}~{y1:.3f}]")

print("=== 文本/曲线几何校验 ===")
for t in ax.texts:
    check(t.get_text().split("\n")[0][:20], t)
leg = ax.get_legend()
if leg is not None:
    check("图例", leg.get_frame())

fig.savefig(FIG / "c7_curve_2025_2026.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("wrote c7_curve_2025_2026.png")
print()
print("17 个月累计（已按段拼接，起点=1.0）：Spearman选参 %.2f%% | 固定参数 %.2f%% | 买入持有 %.2f%%" % (
    (pd.concat([s1, s2])["equity"].iloc[-1] - 1) * 100,
    (pd.concat([f1, f2])["equity"].iloc[-1] - 1) * 100,
    (pd.concat([s1, s2])["benchmark_equity"].iloc[-1] - 1) * 100))
print(ms.to_string(index=False))
