# -*- coding: utf-8 -*-
"""生成种子稳健性图（c8）：各口径 4 个种子的收益分布 vs 买入持有。

布局要点：统计量以「顶部文字框」形式给出，避免与均值线/参考线交叠；
自检：所有文字包围盒与散点、参考线、均值线求交，打印 PASS/FAIL。
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parents[2]
SRV = BASE / "data" / "server_outputs_20260908"
FIG = BASE / "docs" / "训练记录" / "figs_cicc"
SEEDS = [100, 200, 300, 400]


def collect(fixed):
    net, ex, diri = [], [], []
    for s in SEEDS:
        if fixed:
            d = "base_fixed_2025win_2025_2026_20m" if s == 100 else f"base_fixed_2025win_2025_2026_20m_seed{s}"
        else:
            d = "base_spearman_2025_2026_20m" if s == 100 else f"base_spearman_2025_2026_20m_seed{s}"
        j = json.loads((SRV / d / "summary.json").read_text(encoding="utf-8"))
        net.append(j["net_compounded_return"] * 100)
        ex.append(j["total_relative_excess_return"] * 100)
        diri.append(j["directional_accuracy"] * 100)
    return np.array(net), np.array(ex), np.array(diri)


S = collect(False)
F = collect(True)
bh = json.loads((SRV / "base_spearman_2025_2026_20m" / "summary.json").read_text(encoding="utf-8"))[
    "benchmark_buy_and_hold_return"] * 100

fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.0))
GROUPS = [("原定稿方案\n（每月滚动选参）", "原定稿方案", S, "#c00000"),
          ("固定一套参数\n（完全不选参）", "固定参数", F, "#2e7d32")]

# ================= 左：净收益分布 =================
ax = axes[0]
ax.set_ylim(-9, 60)
ax.set_xlim(-0.75, 1.55)
for i, (xtick, short, (net, ex, diri), color) in enumerate(GROUPS):
    jit = np.linspace(-0.085, 0.085, len(net))
    ax.scatter(np.full(len(net), i) + jit, net, s=100, color=color, zorder=4,
               edgecolor="white", linewidth=1.3)
    for v, jx in zip(net, jit):
        ax.text(i + jx, v + 1.4, f"{v:.1f}", ha="center", va="bottom", fontsize=8.6, color=color, zorder=5)
    m, sd = net.mean(), net.std(ddof=1)
    ax.plot([i - 0.31, i + 0.31], [m, m], color=color, lw=2.8, zorder=3)
    # 顶部统计框
    ax.text(i, 57.5, f"均值 {m:.1f}%   σ {sd:.1f}pp\n跑赢大盘 {(ex>0).sum()}/{len(ex)} 个种子",
            ha="center", va="top", fontsize=9.2, color=color, zorder=6,
            bbox=dict(boxstyle="round,pad=0.34", fc="white", ec=color, lw=1.0, alpha=0.95))
    # 跨度（画在零轴以下的空白带）
    ax.annotate("", xy=(i - 0.44, net.min()), xytext=(i - 0.44, net.max()),
                arrowprops=dict(arrowstyle="<->", color="#999", lw=1.2))
    ax.text(i, -6.4, f"跨度 {net.max()-net.min():.0f}pp", ha="center", va="center",
            fontsize=8.6, color="#666")

ax.axhline(bh, color="#1f4e79", ls="--", lw=1.8, zorder=2)
ax.text(-0.70, bh - 1.4, f"买入持有 {bh:+.1f}%", fontsize=9.5, color="#1f4e79", ha="left", va="top")
ax.set_xticks([0, 1])
ax.set_xticklabels([g[0] for g in GROUPS], fontsize=10)
ax.set_ylabel("20 个月净收益（%）", fontsize=11)
ax.set_title("只换随机种子（samples=10），其余完全相同", fontsize=11.5)
ax.grid(axis="y", alpha=0.3)

# ================= 右：方向准确率 =================
ax2 = axes[1]
ax2.set_ylim(42.5, 58)
ax2.set_xlim(-0.75, 1.55)
for i, (xtick, short, (net, ex, diri), color) in enumerate(GROUPS):
    jit = np.linspace(-0.085, 0.085, len(diri))
    ax2.scatter(np.full(len(diri), i) + jit, diri, s=100, color=color, zorder=4,
                edgecolor="white", linewidth=1.3)
    # 右panel 较窄（46~52%），逐点数值标签易与 50% 参考线/均值线交叠，
    # 故不给单点标签，数值由顶部统计框（均值+范围）与报告表格给出。
    m = diri.mean()
    ax2.plot([i - 0.31, i + 0.31], [m, m], color=color, lw=2.8, zorder=3)
    ax2.text(i, 55.6, f"均值 {m:.1f}%\n范围 {diri.min():.1f}~{diri.max():.1f}%",
             ha="center", va="top", fontsize=9.2, color=color, zorder=6,
             bbox=dict(boxstyle="round,pad=0.34", fc="white", ec=color, lw=1.0, alpha=0.95))
ax2.axhline(50, color="#1f4e79", ls="--", lw=1.8, zorder=2)
ax2.text(-0.70, 50.35, "随机水平 = 50%", fontsize=9.5, color="#1f4e79", ha="left")
ax2.set_xticks([0, 1])
ax2.set_xticklabels([g[1] for g in GROUPS], fontsize=10)
ax2.set_ylabel("方向准确率（%）", fontsize=11)
ax2.set_title("8 次运行全部落在 46~52% 之间", fontsize=11.5)
ax2.grid(axis="y", alpha=0.3)

fig.suptitle("种子稳健性检验：单一种子的收益数字是采样噪声，不是策略能力", fontsize=13.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])

# ================= 自检：文字包围盒 vs 散点/线 =================
fig.canvas.draw()
ren = fig.canvas.get_renderer()
problems = []
for a in (ax, ax2):
    ylim = a.get_ylim()
    lines_y = [ln.get_ydata()[0] for ln in a.get_lines() if len(ln.get_ydata()) > 1]
    pts = [(a.transData.transform((x, y))) for x, y in
           [(c.get_offsets()[0][0], c.get_offsets()[0][1]) for c in a.collections] ]
    for t in a.texts:
        label = (t.get_text().splitlines() or [""])[0][:16]
        if not label:
            continue  # 空白注释（如跨度箭头）
        bb = t.get_window_extent(ren)
        # 与水平线（虚线/均值线）求交
        for ly in lines_y:
            ypx = a.transData.transform((0, ly))[1]
            if bb.y0 - 1 <= ypx <= bb.y1 + 1:
                problems.append(f"{a.get_title()[:12]} 文本『{label}』与 y={ly:.2f} 的水平线相交")
        # 与散点求交
        for c in a.collections:
            for x, y in c.get_offsets():
                px, py = a.transData.transform((x, y))
                if bb.x0 - 2 <= px <= bb.x1 + 2 and bb.y0 - 2 <= py <= bb.y1 + 2:
                    problems.append(f"{a.get_title()[:12]} 文本『{label}』压住散点")
print("=== 自检 ===")
if problems:
    for p in sorted(set(problems)):
        print("  [FAIL]", p)
else:
    print("  [PASS] 所有文字与参考线/均值线/散点均无交叠")

fig.savefig(FIG / "c8_seed_robustness.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("wrote c8_seed_robustness.png")
print("S:", [f"{v:.2f}" for v in S[0]], "均值 %.2f σ %.2f" % (S[0].mean(), S[0].std(ddof=1)))
print("F:", [f"{v:.2f}" for v in F[0]], "均值 %.2f σ %.2f" % (F[0].mean(), F[0].std(ddof=1)))
