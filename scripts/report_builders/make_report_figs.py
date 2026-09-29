#!/usr/bin/env python3
"""Generate paper-style comparison figures for the rolling fine-tune report."""
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
FIG = ROOT / "report_figs"
FIG.mkdir(exist_ok=True)

PLOT_STYLE = {
    "ROLL-nr": dict(color="#1f77b4", ls="-", lw=2.2, label="Rolling FT (net-return sel.)"),
    "ROLL-sp": dict(color="#aec7e8", ls="-", lw=1.8, label="Rolling FT (Spearman sel.)"),
    "FT-static-nr": dict(color="#ff7f0e", ls="-", lw=2.0, label="Static FT (net-return sel.)"),
    "FT-static-sp": dict(color="#ffbb78", ls="-", lw=1.6, label="Static FT (Spearman sel.)"),
    "BASE-sp": dict(color="#2ca02c", ls="-", lw=2.2, label="Kronos base (Spearman sel.)"),
}

DIRS = {
    "ROLL-nr": "rolling_walkforward_netreturn_2025_9m",
    "ROLL-sp": "rolling_walkforward_spearman_2025_9m",
    "FT-static-nr": "static_ft_walkforward_netreturn_2025_9m",
    "FT-static-sp": "static_ft_walkforward_spearman_2025_9m",
    "BASE-sp": "base_walkforward_spearman_2025_9m",
}


def load_daily(name):
    return pd.read_csv(OUT / DIRS[name] / "daily_equity.csv", parse_dates=["date"])


def load_preds(name):
    return pd.read_csv(OUT / DIRS[name] / "test_predictions.csv", parse_dates=["origin_date"])


def load_summary(name):
    return json.loads((OUT / DIRS[name] / "summary.json").read_text(encoding="utf-8"))


def annotate_paper(ax):
    ax.axhline(1.339, color="crimson", ls="--", lw=1.3, alpha=0.9)
    ax.text(ax.get_xlim()[0] + 2, 1.339 + 0.006, "Paper headline 33.9%", color="crimson", fontsize=9)


# ----------------------------------------------------------------- fig 1: NAV curves + position strip
def fig1():
    dailies = {k: load_daily(k) for k in DIRS}
    ref = dailies["ROLL-nr"]
    fig, (ax, axp) = plt.subplots(
        2, 1, figsize=(10, 6.6), sharex=True,
        gridspec_kw={"height_ratios": [4.2, 1.0], "hspace": 0.06},
    )
    ax.plot(ref["date"], ref["benchmark_equity"], color="#555555", ls="--", lw=1.4,
            label="CSI1000 index (buy & hold)")
    for k in DIRS:
        ax.plot(dailies[k]["date"], dailies[k]["equity"], **PLOT_STYLE[k])
    ax.axvspan(pd.Timestamp("2025-07-01"), ref["date"].max(), color="#fff3cd", alpha=0.55, lw=0)
    ax.text(pd.Timestamp("2025-07-20"), ax.get_ylim()[0] + 0.02, "July 2025 regime shift",
            fontsize=9, color="#8a6d00")
    annotate_paper(ax)
    ax.set_ylabel("NAV (net of costs, start = 1.0)")
    ax.set_title("Fig.1  Strategy NAV comparison  (2025-01-02 ~ 2025-09-30, CSI1000 timing)")
    ax.legend(loc="upper left", fontsize=8.5, ncol=2, framealpha=0.9)
    ax.grid(alpha=0.3)
    # position strip of ROLL-nr
    pos = dailies["ROLL-nr"]
    axp.fill_between(pos["date"], pos["in_market"].astype(float), 0, color="#1f77b4", alpha=0.75)
    axp.set_ylim(0, 1.06)
    axp.set_yticks([0, 1])
    axp.set_yticklabels(["cash", "long"])
    axp.set_ylabel("pos.", fontsize=8)
    axp.set_title("Position (Rolling FT / net-return sel.)", fontsize=8, loc="left")
    axp.xaxis.set_major_locator(mdates.MonthLocator())
    axp.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in axp.get_xticklabels():
        label.set_rotation(0)
    fig.align_ylabels()
    fig.savefig(FIG / "fig1_nav_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- fig 2: net-return bars + H1/H2 split
def fig2():
    order = ["ROLL-nr", "ROLL-sp", "FT-static-nr", "FT-static-sp", "BASE-sp"]
    nets = [load_summary(k)["net_compounded_return"] for k in order]
    bh = load_summary("BASE-sp")["benchmark_buy_and_hold_return"]
    short = {"ROLL-nr": "Rolling\n(net-ret)", "ROLL-sp": "Rolling\n(Spearman)",
             "FT-static-nr": "Static\n(net-ret)", "FT-static-sp": "Static\n(Spearman)",
             "BASE-sp": "Base\n(Spearman)"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    bars = ax1.bar(range(len(order)), nets, color=[PLOT_STYLE[k]["color"] for k in order], width=0.62)
    ax1.axhline(bh, color="#555555", ls="--", lw=1.2)
    ax1.text(len(order) - 0.5, bh + 0.008, f"Buy&Hold {bh:.1%}", color="#333", fontsize=9, ha="right")
    ax1.axhline(0.339, color="crimson", ls=":", lw=1.4)
    ax1.text(len(order) - 0.5, 0.339 + 0.008, "Paper 33.9%", color="crimson", fontsize=9, ha="right")
    ax1.set_xticks(range(len(order)))
    ax1.set_xticklabels([short[k] for k in order], fontsize=8.5)
    ax1.set_ylabel("Net return 2025-01~09")
    ax1.set_title("(a) Net return by variant", fontsize=10)
    ax1.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    for i, v in enumerate(nets):
        ax1.text(i, v + 0.008, f"{v:.1%}", ha="center", fontsize=8.5)
    ax1.grid(alpha=0.3, axis="y")
    # H1/H2
    h1 = [growth_of(k, "2025-01-01", "2025-06-30") for k in order]
    h2 = [growth_of(k, "2025-07-01", "2025-09-30") for k in order]
    x = np.arange(len(order))
    w = 0.38
    ax2.bar(x - w / 2, h1, w, color="#bccfe8", edgecolor="#5b7fa6", label="H1 (Jan~Jun)")
    ax2.bar(x + w / 2, h2, w, color="#ffd9a8", edgecolor="#c07a1a", label="H2 (Jul~Sep)")
    ax2.axhline(growth_of("BASE-sp", "2025-01-01", "2025-06-30"), color="#555", ls="--", lw=1)
    ax2.axhline(growth_of("BASE-sp", "2025-07-01", "2025-09-30"), color="#2ca02c", ls="--", lw=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels([short[k] for k in order], fontsize=8.5)
    ax2.set_ylabel("Net return")
    ax2.set_title("(b) H1 vs H2 (July-regime test)", fontsize=10)
    ax2.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax2.legend(fontsize=8.5)
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_returns_bars.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def growth_of(name, lo, hi):
    d = load_daily(name)
    sub = d[(d["date"] >= pd.Timestamp(lo)) & (d["date"] <= pd.Timestamp(hi))]
    return float(sub["equity"].iloc[-1] / sub["equity"].iloc[0] - 1)


# ----------------------------------------------------------------- fig 3: predicted vs actual T5 close
def fig3():
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    colors = {"BASE-sp": "#2ca02c", "ROLL-nr": "#1f77b4", "FT-static-nr": "#ff7f0e"}
    order3 = ["BASE-sp", "ROLL-nr", "FT-static-nr"]
    for name, ax in zip(order3, axes):
        p = load_preds(name).sort_values("origin_date")
        ax.plot(p["origin_date"], p["actual_close_t5"], color="#555555", lw=1.6,
                label="Actual T5 close (index)")
        ax.plot(p["origin_date"], p["pred_close_t5"], color=colors[name], lw=1.3, alpha=0.95,
                label="Predicted T5 close")
        sp = spearmanr(p["pred_close_t5"], p["actual_close_t5"]).statistic
        mae = float(np.abs(p["pred_close_t5"] - p["actual_close_t5"]).mean())
        ax.set_title(f"{name}  Spearman={sp:.3f}  MAE={mae:.0f} pts", fontsize=9.5, loc="left")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[0].set_title("Fig.3  Predicted vs actual 5-day-ahead close (per-signal streams, monthly rolling params)",
                      fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_pred_vs_actual.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- fig 4: Spearman & MAE bars (paper chart-18 style)
def fig4():
    order = ["BASE-sp", "FT-static-nr", "ROLL-nr"]
    short = {"BASE-sp": "Base", "FT-static-nr": "Static FT", "ROLL-nr": "Rolling FT"}
    sps, maes = [], []
    for k in order:
        p = load_preds(k)
        sps.append(spearmanr(p["pred_close_t5"], p["actual_close_t5"]).statistic)
        maes.append(float(np.abs(p["pred_close_t5"] - p["actual_close_t5"]).mean()))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.9))
    cols = ["#2ca02c", "#ff7f0e", "#1f77b4"]
    a1.bar(range(3), sps, color=cols, width=0.55)
    a1.set_xticks(range(3)); a1.set_xticklabels([short[k] for k in order])
    a1.set_ylabel("Spearman (price level)")
    a1.set_title("T5 prediction Spearman", fontsize=10)
    for i, v in enumerate(sps):
        a1.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=9)
    a2.bar(range(3), maes, color=cols, width=0.55)
    a2.set_xticks(range(3)); a2.set_xticklabels([short[k] for k in order])
    a2.set_ylabel("MAE (index points)")
    a2.set_title("T5 prediction MAE", fontsize=10)
    for i, v in enumerate(maes):
        a2.text(i, v + 4, f"{v:.0f}", ha="center", fontsize=9)
    for ax in (a1, a2):
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Fig.4  Prediction quality on the timing signal (2025-01~09, level-based)", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG / "fig4_spearman_mae.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


fig1()
fig2()
fig3()
fig4()
print("figures written to", FIG)
for p in sorted(FIG.glob("*.png")):
    print(p.name, p.stat().st_size)
