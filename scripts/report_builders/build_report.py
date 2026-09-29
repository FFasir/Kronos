# -*- coding: utf-8 -*-
"""Build the self-contained HTML report embedding the four figures."""
import base64
from pathlib import Path

DOCS = Path(__file__).resolve().parents[2] / "docs" / "训练记录"
FIGDIR = DOCS / "figs_20260907"
OUT = DOCS / "20260907_滚动微调对比报告.html"


def b64(name):
    return base64.b64encode((FIGDIR / name).read_bytes()).decode("ascii")


fig1 = b64("fig1_nav_curves.png")
fig2 = b64("fig2_returns_bars.png")
fig3 = b64("fig3_pred_vs_actual.png")
fig4 = b64("fig4_spearman_mae.png")

CSS = """
body{font-family:"Microsoft YaHei","PingFang SC",Arial,sans-serif;max-width:1080px;margin:24px auto;padding:0 20px;color:#222;line-height:1.65}
h1{font-size:24px;border-bottom:3px solid #1f77b4;padding-bottom:8px}
h2{font-size:18px;margin-top:28px;color:#1f4e79;border-left:5px solid #1f77b4;padding-left:10px}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13.5px}
th,td{border:1px solid #ccc;padding:6px 9px;text-align:right}
th{background:#eef3fa;text-align:center}
tr:nth-child(even){background:#f8fafc}
img{max-width:100%;border:1px solid #ddd;border-radius:4px;margin:8px 0}
.figcap{font-size:12.5px;color:#555;margin:2px 0 16px}
.note{background:#fff8e6;border:1px solid #e6c85c;padding:10px 14px;font-size:13px;border-radius:4px}
.kpi{color:#1f77b4;font-weight:bold}
"""

body = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>中证1000 Kronos 滚动微调原型实验 —— 对比报告（论文风格图）</title>
<style>{CSS}</style></head><body>

<h1>中证1000 · Kronos 择时：滚动微调原型 vs 静态微调 vs 零样本（论文风格对比）</h1>
<p><b>日期：</b>2026-09-07 ｜ <b>复现对象：</b>CICC《大语言时序模型Kronos的A股择时应用》33.9%（测试区间 2025-01-01 ~ 2025-09-30）<br>
<b>数据/环境：</b>中证1000 时点成分（无幸存者偏差，hfq）；RTX 4090 单卡；成本：佣金 0.03% + 滑点 0.1%（无印花税）</p>

<h2>0. 摘要</h2>
<ul>
<li>滚动微调（每月用近 12 个月数据对 predictor 2-epoch warm-start 重训）<b>全面优于静态单次微调</b>：净收益 +22.4%~+27.1% vs +16.0%~+20.7%；</li>
<li>关键的 <b>2025-07 后（H2）</b>验证：滚动 spearman 线 +7.30% vs 静态 +1.59%（<b>+5.7pp</b>），证实"定期微调提升 7 月后趋势敏感度"的论文论述方向；</li>
<li>但最优滚动口径净 +27.1%（毛 +31.2%），<b>仍未达论文 33.9%</b>；零样本 base（净 +32.0%、毛 +36.9%）最接近且方向准确率 55.1% 全场最高；</li>
<li>方向准确率仍是硬约束：滚动线胜率 83% 但方向准确率仅 47.8%，收益来自少数高质量交易。</li>
</ul>

<h2>1. 实验口径</h2>
<table>
<tr><th>项目</th><th>滚动微调（本原型）</th><th>静态微调（前序）</th><th>零样本 base</th></tr>
<tr><td>模型</td><td>base → 逐月重训 9 个 predictor（tokenizer 固定）</td><td>2015-06~2022 静态两阶段 10ep</td><td>Kronos base（无微调）</td></tr>
<tr><td>训练窗口</td><td>测试月前 12 个月（693→624 只成分）</td><td>2015-06~2022（7.5 年）</td><td>—</td></tr>
<tr><td>推理参数</td><td colspan="3">逐月网格搜参：T∈{{0.3,0.6,0.8,1.0}}×top_p∈{{0.2,0.4,0.6,0.9}}×lookback∈{{30,60,90}}，按验证月 m−1 的 Spearman 或 net_return 择优</td></tr>
<tr><td>信号/执行</td><td colspan="3">pred T5 收益 &gt;0.5% 做多；T−1 收盘信号→T 开盘建仓；固定持有 5 个交易日；满仓/空仓</td></tr>
</table>

<h2>2. 净值对比曲线（对应论文图表21）</h2>
<img src="data:image/png;base64,{fig1}" alt="fig1">
<div class="figcap">图1：2025-01-02~2025-09-30 各方案净值（含成本）。黄色底纹为 2025-07 起的趋势市；红色虚线为论文宣称的 33.9% 终点；底部色条为滚动微调（net_return 选参）的持仓/空仓。滚动两条线（蓝系）在 7 月后明显跑赢静态两条线（橙系），但始终低于 base（绿）与指数（灰虚线）。</div>

<h2>3. 收益对比柱状 + H1/H2 拆解</h2>
<img src="data:image/png;base64,{fig2}" alt="fig2">
<div class="figcap">图2：(a) 9 个月净收益：滚动微调显著高于静态；base 最高且唯一跑赢买入持有（虚线）；(b) H1/H2 拆解：所有方案 H1 都强，H2（7 月后趋势市）只有滚动与 base 跟上——静态微调在 H2 基本失效（+1.6%~+2.7%），这正是论文"定期微调提升趋势敏感度"的核心论点。</div>

<h2>4. 预测 vs 真实 T5 收盘价（对应论文图表20）</h2>
<img src="data:image/png;base64,{fig3}" alt="fig3">
<div class="figcap">图3：逐月滚动选参下的单日预测流（每个信号日一条）与真实 5 日后收盘价的对比。水平上模型都能"贴住"真实序列（水平 Spearman &gt;0.91），但三条线差异很小——再次印证：<b>水平预测准 ≠ 方向预测准</b>，差异主要来自 7 月后的跟随能力。</div>

<h2>5. 预测质量（对应论文图表18 口径，T5 水平）</h2>
<img src="data:image/png;base64,{fig4}" alt="fig4">
<div class="figcap">图4：T5 收盘价预测的 Spearman 与 MAE。base/静态/滚动三者在水平指标上几乎无差别（0.92~0.94），说明微调/滚动提升的不是"水平拟合"，而是交易侧行为（触发与持仓节奏）。</div>

<h2>6. 汇总指标表（9 个月，论文测试区间）</h2>
<table>
<tr><th>方案</th><th>毛收益</th><th>净收益</th><th>胜率</th><th>方向准确率</th><th>交易数</th><th>在市值率</th><th>净超额 vs B&amp;H</th></tr>
<tr><td>Rolling FT (net-return)</td><td>+31.17%</td><td><b>+27.14%</b></td><td>83.3%</td><td>47.8%</td><td>12</td><td>32.8%</td><td>−2.70pp</td></tr>
<tr><td>Rolling FT (Spearman)</td><td>+26.30%</td><td>+22.42%</td><td>75.0%</td><td>44.2%</td><td>12</td><td>32.8%</td><td>−6.31pp</td></tr>
<tr><td>Static FT (net-return)</td><td>+23.85%</td><td>+20.67%</td><td>80.0%</td><td>50.0%</td><td>10</td><td>27.3%</td><td>−7.65pp</td></tr>
<tr><td>Static FT (Spearman)</td><td>+19.35%</td><td>+15.99%</td><td>72.7%</td><td>42.8%</td><td>11</td><td>30.1%</td><td>−11.24pp</td></tr>
<tr><td>Kronos base (Spearman)</td><td>+36.91%</td><td><b>+32.01%</b></td><td>71.4%</td><td><b>55.1%</b></td><td>14</td><td>38.3%</td><td><b>+1.03pp</b></td></tr>
<tr><td>买入持有（000852）</td><td colspan="6">B&amp;H = +30.67%（基准）</td><td>—</td></tr>
<tr><td>论文宣称（图表21）</td><td colspan="6">33.9%（毛/净口径未披露，年化超额 9%）</td><td>—</td></tr>
</table>

<h2>7. 结论与说明</h2>
<ul>
<li><b>机制验证成功</b>：滚动微调让模型带上近期 regime 权重，H2 收益显著高于静态；</li>
<li><b>数值未复现 33.9%</b>：最优滚动净 +27.1%/毛 +31.2%；若论文为毛口径则差距 ~2.7pp，若为净口径则 ~6.8pp（base 净 +32.0% 最接近）；</li>
<li>剩余差距的候选来源：论文未披露的滚动重训超参（频率/epoch/窗口/tokenizer 是否滚动）、训练期 2010–2022（本实验 2015-06 起）、qfq 口径、以及图表 21 无逐线数值、33.9% 归属系正文推断；</li>
<li class="note"><b>诚实声明</b>：滚动评估的验证月 m−1 落在滚动训练窗口内，参数搜索有轻度样本内选择偏差；base 高值依赖逐月滚动搜参；均不应视为论文方法的"确定性复现成功"。</li>
</ul>

<p style="color:#888;font-size:12px;margin-top:28px">数据来源：服务器 /root/kronos-strategy（EXP-RL-2025-01~09、data/outputs/*_2025_9m）。图 1~4 与表 6 全部来自同一批回测输出，口径一致。</p>
</body></html>"""

OUT.write_text(body, encoding="utf-8")
print("written", OUT, OUT.stat().st_size, "bytes")
