# -*- coding: utf-8 -*-
"""生成报告新增图：(1) 模型 vs 朴素基准预测能力；(2) 口径 × 区间 收益翻转。"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

BASE = Path(__file__).resolve().parents[2]
SRV = BASE / "data" / "server_outputs_20260908"
FIG = BASE / "docs" / "训练记录" / "figs_cicc"
FIG.mkdir(parents=True, exist_ok=True)

# ---------- 数据 ----------
def skill(sub):
    pr = pd.read_csv(SRV / sub / "test_predictions.csv")
    pred, actual, last = pr["pred_close_t5"], pr["actual_close_t5"], pr["last_close"]
    pred_r = pred / last - 1
    act_r = actual / last - 1
    return {
        "r2_model": pearsonr(pred, actual)[0] ** 2,
        "r2_naive": pearsonr(last, actual)[0] ** 2,
        "mae_model": np.mean(np.abs(pred - actual)),
        "mae_naive": np.mean(np.abs(last - actual)),
        "r2_ret": pearsonr(pred_r, act_r)[0] ** 2,
        "dir": np.mean((pred_r > 0) == (act_r > 0)),
        "dir_base": (act_r > 0).mean(),
    }

s25 = skill("base_walkforward_spearman_2025_9m")
s26 = skill("base_walkforward_spearman_2026_8m")

# ---------- 图 c5：模型 vs 朴素基准 ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

ax = axes[0]
labels = ["2025\n(Jan-Sep)", "2026\n(Jan-Aug)"]
x = np.arange(2)
w = 0.36
mv = [s25["r2_model"] * 100, s26["r2_model"] * 100]
nv = [s25["r2_naive"] * 100, s26["r2_naive"] * 100]
b1 = ax.bar(x - w / 2, mv, w, label="Kronos", color="#c0504d")
b2 = ax.bar(x + w / 2, nv, w, label="Naive (predict no change)", color="#4f81bd")
for b, v in zip(b1, mv):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}%", ha="center", fontsize=9)
for b, v in zip(b2, nv):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}%", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("R² of price level (pred vs actual T5)", fontsize=9)
ax.set_title("Price-level fit: model is NOT better\nthan 'price stays flat'", fontsize=10)
ax.legend(fontsize=8, loc="upper center", ncol=1, framealpha=0.9)
ax.set_ylim(0, 135)
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
mv = [s25["mae_model"], s26["mae_model"]]
nv = [s25["mae_naive"], s26["mae_naive"]]
b1 = ax.bar(x - w / 2, mv, w, label="Kronos", color="#c0504d")
b2 = ax.bar(x + w / 2, nv, w, label="Naive", color="#4f81bd")
for b, v in zip(b1, mv):
    ax.text(b.get_x() + b.get_width() / 2, v + 8, f"{v:.0f}", ha="center", fontsize=9)
for b, v in zip(b2, nv):
    ax.text(b.get_x() + b.get_width() / 2, v + 8, f"{v:.0f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("MAE (index points)", fontsize=9)
ax.set_title("Level error: model is WORSE\n(+8.9% in 2025, +29.5% in 2026)", fontsize=10)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

ax = axes[2]
items = ["Return R²\n(2025)", "Return R²\n(2026)", "Direction acc.\n(2025)", "Direction acc.\n(2026)"]
vals = [s25["r2_ret"] * 100, s26["r2_ret"] * 100, s25["dir"] * 100, s26["dir"] * 100]
base_vals = [0, 0, s25["dir_base"] * 100, s26["dir_base"] * 100]
xs = np.arange(4)
b1 = ax.bar(xs - w / 2, vals, w, label="Kronos", color="#c0504d")
b2 = ax.bar(xs + w / 2, base_vals, w, label="Trivial baseline", color="#9bbb59")
for b, v in zip(b1, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=9)
for b, v in zip(b2, base_vals):
    if v > 0:
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=9)
ax.set_xticks(xs)
ax.set_xticklabels(items, fontsize=8)
ax.set_ylabel("percent", fontsize=9)
ax.set_title("Useful skill ≈ 0:\nreturn R²≈0, direction ≤ 'always long'", fontsize=10)
ax.legend(fontsize=8)
ax.set_ylim(0, 92)
ax.grid(axis="y", alpha=0.3)

fig.suptitle("Kronos price-prediction skill vs naive baselines (index 000852, T5 horizon)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(FIG / "c5_skill_vs_naive.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("wrote c5_skill_vs_naive.png")

# ---------- 图 c6：口径 × 区间 ----------
cases = [
    ("S  Spearman sel.", "base_walkforward_spearman_2025_9m", "base_walkforward_spearman_2026_8m"),
    ("N  Net-return sel.", "base_netreturn_2025_9m", "base_netreturn_2026_8m"),
    ("T  Thr.+net-ret sel.", "base_threshold_2025_9m", "base_threshold_2026_8m"),
    ("F1 Fixed paper default", "base_fixed_walkforward_2025_9m", "base_fixed_paper_2026_8m"),
    ("F2 Fixed 2025-best", "base_fixed_2025win_2025_9m", "base_fixed_2025win_2026_8m"),
]
rows = []
for label, d25, d26 in cases:
    a = json.loads((SRV / d25 / "summary.json").read_text(encoding="utf-8"))
    b = json.loads((SRV / d26 / "summary.json").read_text(encoding="utf-8"))
    rows.append((label, a["net_compounded_return"] * 100, b["net_compounded_return"] * 100,
                 (1 + a["net_compounded_return"]) * (1 + b["net_compounded_return"]) * 100 - 100))
df = pd.DataFrame(rows, columns=["caliber", "y2025", "y2026", "combined"])

fig, ax = plt.subplots(figsize=(11, 5.2))
x = np.arange(len(df))
w = 0.27
b1 = ax.bar(x - w, df["y2025"], w, label="2025-01~09 (B&H +30.67%)", color="#4f81bd")
b2 = ax.bar(x, df["y2026"], w, label="2026-01~08 (B&H +0.21%)", color="#c0504d")
b3 = ax.bar(x + w, df["combined"], w, label="combined 17m (B&H +30.95%)", color="#9bbb59")
for bars in (b1, b2, b3):
    for b in bars:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, v + (1.2 if v >= 0 else -3.4),
                f"{v:+.1f}", ha="center", fontsize=8)
ax.axhline(30.67, color="#4f81bd", ls="--", lw=1.2, alpha=0.8)
ax.axhline(0.21, color="#c0504d", ls="--", lw=1.2, alpha=0.8)
ax.axhline(30.95, color="#9bbb59", ls="--", lw=1.2, alpha=0.8)
ax.text(0.05, 31.8, "B&H combined 17m = +30.95%  (all strategy bars below this line)",
        fontsize=8.5, color="#6b8e23")
ax.set_xticks(x)
ax.set_xticklabels(df["caliber"], fontsize=9)
ax.set_ylabel("net return (%)", fontsize=10)
ax.set_title("Parameter-selection caliber x period: ranking FLIPS (rank corr = −0.20)\n"
             "Every caliber ends BELOW buy-and-hold over the combined 17 months",
             fontsize=11)
ax.legend(fontsize=9, loc="upper right")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(-12, 44)
fig.tight_layout()
fig.savefig(FIG / "c6_caliber_flip.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("wrote c6_caliber_flip.png")
print()
print(df.to_string(index=False))
