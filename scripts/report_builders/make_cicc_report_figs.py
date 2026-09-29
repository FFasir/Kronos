#!/usr/bin/env python3
"""Generate CICC-style figures for the chosen scheme (Kronos base + spearman rolling search)
plus a full fine-tuning-method comparison chart."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "outputs"
FIG = ROOT / "report_figs_cicc"
FIG.mkdir(exist_ok=True)

BASE_SP = OUT / "base_walkforward_spearman_2025_9m"


def load_daily(d):
    return pd.read_csv(d / "daily_equity.csv", parse_dates=["date"])


def load_preds(d):
    return pd.read_csv(d / "test_predictions.csv", parse_dates=["origin_date"])


def load_summary(d):
    return json.loads((d / "summary.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- fig_c1: NAV + position
def fig1():
    d = load_daily(BASE_SP)
    dfix = load_daily(OUT / "base_fixed_walkforward_2025_9m")
    fig, (ax, axp) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [4.2, 1.0], "hspace": 0.05})
    ax.plot(d["date"], d["benchmark_equity"], color="#555", ls="--", lw=1.4, label="CSI1000 index (buy & hold)")
    ax.plot(dfix["date"], dfix["equity"], color="#d62728", lw=1.6, label="base fixed default (lookback=400)")
    ax.plot(d["date"], d["equity"], color="#2ca02c", lw=2.2, label="Kronos base + Spearman rolling search")
    ax.axvspan(pd.Timestamp("2025-07-01"), d["date"].max(), color="#fff3cd", alpha=0.5, lw=0)
    ax.axhline(1.339, color="crimson", ls=":", lw=1.3)
    ax.text(d["date"].min(), 1.339 + 0.005, "Paper headline 33.9%", color="crimson", fontsize=9)
    ax.text(pd.Timestamp("2025-07-18"), ax.get_ylim()[0] + 0.02, "July 2025 trend regime", fontsize=9, color="#8a6d00")
    ax.set_ylabel("NAV (net of costs, start = 1.0)")
    ax.set_title("CSI1000 timing: Kronos base + monthly Spearman rolling search  (2025-01-02 ~ 2025-09-30)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    axp.fill_between(d["date"], d["in_market"].astype(float), 0, color="#2ca02c", alpha=0.7)
    axp.set_ylim(0, 1.06)
    axp.set_yticks([0, 1])
    axp.set_yticklabels(["cash", "long"])
    axp.set_ylabel("pos.", fontsize=8)
    axp.xaxis.set_major_locator(mdates.MonthLocator())
    axp.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.align_ylabels()
    fig.savefig(FIG / "c1_nav_base.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig_c2: pred vs actual
def fig2():
    p = load_preds(BASE_SP).sort_values("origin_date")
    sp = spearmanr(p["pred_close_t5"], p["actual_close_t5"]).statistic
    mae = float(np.abs(p["pred_close_t5"] - p["actual_close_t5"]).mean())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(p["origin_date"], p["actual_close_t5"], color="#555", lw=1.6, label="Actual T5 close (index)")
    ax.plot(p["origin_date"], p["pred_close_t5"], color="#2ca02c", lw=1.3, label="Predicted T5 close")
    ax.set_title(f"Predicted vs actual 5-day-ahead close — Spearman={sp:.3f}, MAE={mae:.0f} pts")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    fig.tight_layout()
    fig.savefig(FIG / "c2_pred_actual_base.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig_c3: all-method comparison
METHODS = [
    ("base_fixed", "base fixed default (lookback=400)", "base_fixed_walkforward_2025_9m", "#d62728"),
    ("base_sp", "base + Spearman search (thr=0.5%)", "base_walkforward_spearman_2025_9m", "#2ca02c"),
    ("base_nr_thr", "base + net-return search + thr grid", "base_thr_walkforward_2025_9m", "#82c985"),
    ("roll_nr", "Rolling 12m 2ep + net-return", "rolling_walkforward_netreturn_2025_9m", "#1f77b4"),
    ("roll_sp", "Rolling 12m 2ep + Spearman", "rolling_walkforward_spearman_2025_9m", "#6baed6"),
    ("static_nr", "Static FT 7.5y + net-return", "static_ft_walkforward_netreturn_2025_9m", "#ff7f0e"),
    ("static_sp", "Static FT 7.5y + Spearman", "static_ft_walkforward_spearman_2025_9m", "#fdae6b"),
    ("ep4_nr", "Rolling 4-epoch + net-return", "hfq_rolling12m_EP4_walkforward_netreturn_2025_9m", "#9467bd"),
    ("ep4_sp", "Rolling 4-epoch + Spearman", "hfq_rolling12m_EP4_walkforward_spearman_2025_9m", "#c5b0d5"),
    ("exp_nr", "Rolling expanding + net-return", "hfq_rolling_exp_walkforward_netreturn_2025_9m", "#8c564b"),
    ("exp_sp", "Rolling expanding + Spearman", "hfq_rolling_exp_walkforward_spearman_2025_9m", "#c49c94"),
    ("qfq_nr", "QFQ static + net-return", "qfq_static_ft_walkforward_netreturn_2025_9m", "#17becf"),
    ("qfq_sp", "QFQ static + Spearman", "qfq_static_ft_walkforward_spearman_2025_9m", "#9edae5"),
    ("cs13y_snr", "CSMAR 13y static + net-return", "csmar13y_static_walkforward_netreturn_2025_9m", "#e377c2"),
    ("cs13y_ssp", "CSMAR 13y static + Spearman", "csmar13y_static_walkforward_spearman_2025_9m", "#f7b6d2"),
    ("cs13y_rnr", "CSMAR 13y rolling + net-return", "csmar13y_rolling_walkforward_netreturn_2025_9m", "#bcbd22"),
    ("cs13y_rsp", "CSMAR 13y rolling + Spearman", "csmar13y_rolling_walkforward_spearman_2025_9m", "#dbdb8d"),
]


def fig3():
    labels, vals, colors = [], [], []
    for _, lab, name, col in METHODS:
        d = OUT / name
        if not (d / "summary.json").is_file():
            continue
        m = load_summary(d)
        labels.append(lab)
        vals.append(m["net_compounded_return"])
        colors.append(col)
    bh = load_summary(BASE_SP)["benchmark_buy_and_hold_return"]

    n = len(labels)
    pos = np.arange(n)[::-1]  # labels[0] on top
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.barh(pos, vals, color=colors, height=0.62, zorder=3)
    # reference lines
    ax.axvline(bh, color="#555", ls="--", lw=1.3, zorder=2)
    ax.axvline(0.339, color="crimson", ls=":", lw=1.5, zorder=2)
    # value labels (right of positive bars, left of negative bars)
    for yi, v in zip(pos, vals):
        if v >= 0:
            ax.text(v + 0.006, yi, f"{v:.1%}", va="center", ha="left", fontsize=8)
        else:
            ax.text(v - 0.006, yi, f"{v:.1%}", va="center", ha="right", fontsize=8)
    # reference annotations in the top headroom, at different heights to avoid overlap
    ax.text(bh, n + 0.7, f"Buy & Hold  {bh:.1%}", color="#333", fontsize=9, ha="left")
    ax.text(0.339, n + 0.2, "Paper headline  33.9%", color="crimson", fontsize=9, ha="left")
    ax.set_yticks(pos, labels=labels, fontsize=8.5)
    ax.set_ylim(-0.6, n + 1.0)
    ax.set_xlim(-0.20, 0.46)
    ax.set_xlabel("Net return 2025-01~09 (net of costs)")
    ax.set_title("Fine-tuning method comparison on CSI1000 timing (2025-01-01 ~ 2025-09-30)")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.grid(alpha=0.3, axis="x", zorder=0)
    fig.subplots_adjust(left=0.30, right=0.97, top=0.94, bottom=0.07)
    fig.savefig(FIG / "c3_methods_compare.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- fig_c4: monthly attribution
def fig4():
    d = load_daily(BASE_SP)
    d["month"] = d["date"].dt.to_period("M")
    strat = d.groupby("month").apply(lambda g: g["equity"].iloc[-1] / g["equity"].iloc[0] - 1)
    bm = d.groupby("month").apply(lambda g: g["close"].iloc[-1] / g["close"].iloc[0] - 1)
    idx = np.arange(len(strat))
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.bar(idx - 0.2, strat.values, 0.4, color="#2ca02c", label="Base+Spearman strategy")
    ax.bar(idx + 0.2, bm.values, 0.4, color="#9e9e9e", label="CSI1000 index")
    ax.set_xticks(idx)
    ax.set_xticklabels([str(p) for p in strat.index], fontsize=8)
    ax.set_ylabel("Net return")
    ax.set_title("Monthly attribution (base + Spearman, 2025-01~09)")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "c4_monthly_base.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig_c5: monthly chosen params
def fig5():
    p = OUT / "base_walkforward_spearman_2025_9m" / "monthly_params.csv"
    df = pd.read_csv(p)
    months = df["test_month"].tolist()
    x = np.arange(len(months))
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 6.4), sharex=True)
    panels = [
        ("Temperature T", df["temperature"], (0.3, 1.0), "#1f77b4"),
        ("top_p", df["top_p"], (0.2, 0.9), "#ff7f0e"),
        ("Lookback (days)", df["lookback"], (30, 90), "#2ca02c"),
    ]
    for ax, (name, vals, (lo, hi), col) in zip(axes, panels):
        ax.plot(x, vals, marker="o", ms=6, color=col, lw=1.6)
        for xi, v in zip(x, vals):
            ax.annotate(f"{v:g}", (xi, v), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8.5)
        ax.set_ylabel(name, fontsize=10)
        span = (hi - lo)
        ax.set_ylim(lo - 0.3 * span, hi + 0.5 * span)
        ax.axvline(5.5, color="#ccc", ls="--", lw=1)
        ax.text(5.6, ax.get_ylim()[0] + 0.06 * span, "Jul 2025", fontsize=8, color="#8a6d00")
        ax.grid(alpha=0.3)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(months, fontsize=9)
    axes[0].set_title("Monthly rolling parameter selection (chosen scheme: base + Spearman)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "d_monthly_params.png", dpi=160)
    plt.close(fig)


fig1()
fig2()
fig3()
fig4()
fig5()
print("cicc figures written to", FIG)
for p in sorted(FIG.glob("*.png")):
    print(p.name, p.stat().st_size)
