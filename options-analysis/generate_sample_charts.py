"""
generate_sample_charts.py

Generates sample charts for portfolio display using synthetic data that
reflects the narrative pattern in insights_report.md (win rates, trend
direction, ticker/strategy mix). Actual P&L figures are not used.

Run:
    python generate_sample_charts.py
Output: charts/ directory (9 PNG files)
"""

import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUT_DIR = "charts"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Synthetic monthly P&L (index = relative units, not real dollars) ──────────
# Pattern from insights_report: ✅/❌ per month, H2 2025 strong, May 2026 best
MONTHS = [
    "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
    "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12",
    "2026-01", "2026-02", "2026-03", "2026-04", "2026-05",
]
# Relative performance (sign matches ✅/❌; magnitude reflects narrative)
MONTHLY_REL = [
    42, -18, 35, -55, 28, 31,   # 2025 H1: April tariff drawdown
    -12, 8, 48, 95, 62, 71,     # 2025 H2: Q4 dominant
    88, -45, 72, 41, 130,       # 2026: MSFT blow-up Feb, May all-time best
]

monthly = pd.DataFrame({"month": MONTHS, "pnl": MONTHLY_REL})
monthly["cum_pnl"] = monthly["pnl"].cumsum()
monthly["label"] = monthly["month"].apply(lambda m: m[-2:] + "'" + m[2:4])

# ── Ticker P&L (relative) ─────────────────────────────────────────────────────
TICKER_PNL = {
    "COST": 210, "NVDA": 175, "MRVL": 88, "AVGO": 72, "IWM": 65,
    "MU": 58, "GOOGL": 52, "SPXW": 48, "CLS": 44, "TSLA": 31,
    "CRWD": -22, "PLTR": -28, "EWY": -19, "NFLX": -68, "MSFT": -95,
}
und_s = pd.Series(TICKER_PNL).sort_values()
top_winner = und_s.index[-1]
top_loser  = und_s.index[0]
top_combo  = pd.concat([und_s.head(4), und_s.tail(10)])

# ── Ticker open trade counts ──────────────────────────────────────────────────
TICKER_FREQ = {
    "NVDA": 38, "COST": 34, "TSLA": 28, "GOOGL": 24, "IWM": 22,
    "AVGO": 20, "MU": 18, "SPXW": 16, "CLS": 14, "MRVL": 10, "META": 9, "PLTR": 8,
}
freq = pd.Series(TICKER_FREQ).sort_values(ascending=False)

# ── Quarterly ─────────────────────────────────────────────────────────────────
QUARTERS = ["2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2"]
QPNL     = [59, -27, 56, 228, 115, 171]
qtr = pd.DataFrame({"quarter": QUARTERS, "pnl": QPNL})
qtr["cum_pnl"] = qtr["pnl"].cumsum()

def bar_colors(vals):
    return ["#e74c3c" if v < 0 else "#00d4a8" for v in vals]


def _fmt(v):
    sign = "+" if v >= 0 else ""
    return f"{sign}{v}"


# ── Chart 1: Monthly P&L + Cumulative ────────────────────────────────────────
fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                     row_heights=[0.6, 0.4], vertical_spacing=0.08,
                     subplot_titles=("Monthly P&L", "Cumulative P&L"))
fig1.add_trace(go.Bar(
    x=monthly["label"], y=monthly["pnl"],
    marker_color=bar_colors(monthly["pnl"]),
    text=[_fmt(v) for v in monthly["pnl"]],
    textposition="outside", textfont=dict(size=10),
    width=0.55, showlegend=False
), row=1, col=1)
fig1.add_trace(go.Scatter(
    x=monthly["label"], y=monthly["cum_pnl"],
    mode="lines+markers",
    line=dict(color="#f39c12", width=3), marker=dict(size=8),
    fill="tozeroy", fillcolor="rgba(243,156,18,0.15)", showlegend=False
), row=2, col=1)
fig1.update_layout(
    title={"text": "Cumulative P&L — Jan 2025 to May 2026<br>"
                   "<span style='font-size:15px;font-weight:normal;'>All Accounts | 78% Monthly Win Rate (14/18 months)</span>"},
    height=600, margin=dict(t=100)
)
fig1.update_yaxes(title_text="P&L (relative)", row=1, col=1)
fig1.update_yaxes(title_text="Cumul. (relative)", row=2, col=1)
fig1.update_xaxes(title_text="Month", row=2, col=1)
fig1.write_image(f"{OUT_DIR}/chart1_monthly_pnl.png")
print("✅ chart1")

# ── Chart 2: Ticker P&L ───────────────────────────────────────────────────────
fig2 = go.Figure(go.Bar(
    x=top_combo.values,
    y=[f"  {t}" for t in top_combo.index],
    orientation="h",
    marker_color=bar_colors(top_combo.values),
    text=[_fmt(v) for v in top_combo.values],
    textposition="auto", textfont=dict(size=11)
))
fig2.update_layout(
    title={"text": f"{top_winner} Leads Gains; {top_loser} Biggest Drag<br>"
                   "<span style='font-size:15px;font-weight:normal;'>Net P&L by Ticker | Top 10 Winners + 4 Losers (relative units)</span>"},
    xaxis=dict(title_text="Net P&L (relative)"),
    height=700, margin=dict(l=100, r=80, t=120, b=60)
)
fig2.write_image(f"{OUT_DIR}/chart2_ticker_pnl.png")
print("✅ chart2")

# ── Chart 3: PUT vs CALL donut ────────────────────────────────────────────────
put_share, call_share = 76, 24
fig3 = go.Figure(go.Pie(
    labels=["PUT", "CALL"], values=[put_share, call_share],
    hole=0.4, marker_colors=["#00d4a8", "#3498db"],
    textinfo="label+percent", textfont_size=14, pull=[0.03, 0.03]
))
fig3.update_layout(
    title={"text": f"PUT Spreads Drive {put_share}% of Gross P&L<br>"
                   "<span style='font-size:15px;font-weight:normal;'>Primary strategy: short put credit spreads</span>"},
    legend=dict(orientation="v", x=1.0)
)
fig3.write_image(f"{OUT_DIR}/chart3_put_vs_call.png")
print("✅ chart3")

# ── Chart 4: Monthly trade count ─────────────────────────────────────────────
TRADE_COUNTS = [14, 11, 16, 8, 13, 15, 10, 12, 17, 22, 19, 20, 23, 9, 18, 15, 21]
tc = pd.DataFrame({"label": monthly["label"], "count": TRADE_COUNTS})
fig4 = go.Figure(go.Bar(
    x=tc["label"], y=tc["count"],
    marker_color="#3498db",
    text=tc["count"], textposition="outside", width=0.6
))
fig4.update_layout(
    title={"text": "New Positions (SELL_OPEN) per Month<br>"
                   "<span style='font-size:15px;font-weight:normal;'>Jan 2025 – May 2026 | All Accounts</span>"},
    xaxis=dict(title_text="Month"),
    yaxis=dict(title_text="# of New Positions")
)
fig4.write_image(f"{OUT_DIR}/chart4_trade_count.png")
print("✅ chart4")

# ── Chart 5: Account P&L ─────────────────────────────────────────────────────
accounts  = ["Account A", "Account B", "Account C", "Account D"]
acct_pnl  = [220, 175, 130, 95]
fig5 = go.Figure(go.Bar(
    x=accounts, y=acct_pnl,
    marker_color=["#3498db", "#9b59b6", "#e67e22", "#1abc9c"],
    text=[_fmt(v) for v in acct_pnl],
    textposition="inside", textfont=dict(size=13, color="white"), width=0.45
))
fig5.update_layout(
    title={"text": "P&L by Account<br>"
                   "<span style='font-size:15px;font-weight:normal;'>Jan 2025 – May 2026 | Relative units</span>"},
    xaxis=dict(title_text="Account"),
    yaxis=dict(title_text="Net P&L (relative)"),
    margin=dict(l=80, r=40, t=120, b=60)
)
fig5.write_image(f"{OUT_DIR}/chart5_account_pnl.png")
print("✅ chart5")

# ── Chart 6: Weekday P&L ─────────────────────────────────────────────────────
wdays   = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
wpnl    = [42, 65, 118, 97, 38]
fig6 = go.Figure(go.Bar(
    x=wdays, y=wpnl,
    marker_color=bar_colors(wpnl),
    text=[_fmt(v) for v in wpnl], textposition="outside", width=0.5
))
fig6.update_layout(
    title={"text": "Wednesday & Thursday Are Best Trading Days<br>"
                   "<span style='font-size:15px;font-weight:normal;'>Net P&L by Day of Week | All Accounts</span>"},
    xaxis=dict(title_text="Day of Week"),
    yaxis=dict(title_text="Net P&L (relative)")
)
fig6.write_image(f"{OUT_DIR}/chart6_weekday_pnl.png")
print("✅ chart6")

# ── Chart 7: Ticker frequency ─────────────────────────────────────────────────
fig7 = go.Figure(go.Bar(
    x=freq.values,
    y=[f"  {t}" for t in freq.index],
    orientation="h",
    marker_color="#9b59b6",
    text=freq.values,
    textposition="inside", textfont=dict(size=12, color="white")
))
fig7.update_layout(
    title={"text": f"{freq.index[0]} Leads with {freq.iloc[0]} Opening Trades<br>"
                   "<span style='font-size:15px;font-weight:normal;'>New Positions (SELL_OPEN) by Ticker | Top 12</span>"},
    xaxis=dict(title_text="# of Trades"),
    height=500, margin=dict(l=90, r=60, t=120, b=60)
)
fig7.write_image(f"{OUT_DIR}/chart7_ticker_frequency.png")
print("✅ chart7")

# ── Chart 8: Quarterly P&L trajectory ────────────────────────────────────────
fig8 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                     row_heights=[0.6, 0.4], vertical_spacing=0.08,
                     subplot_titles=("Quarterly P&L", "Cumulative P&L"))
fig8.add_trace(go.Bar(
    x=qtr["quarter"], y=qtr["pnl"],
    marker_color=bar_colors(qtr["pnl"]),
    text=[_fmt(v) for v in qtr["pnl"]],
    textposition="outside", width=0.5, showlegend=False
), row=1, col=1)
fig8.add_trace(go.Scatter(
    x=qtr["quarter"], y=qtr["cum_pnl"],
    mode="lines+markers",
    line=dict(color="#f39c12", width=3), marker=dict(size=8),
    fill="tozeroy", fillcolor="rgba(243,156,18,0.15)", showlegend=False
), row=2, col=1)
fig8.update_layout(
    title={"text": "Quarterly P&L Trajectory<br>"
                   "<span style='font-size:15px;font-weight:normal;'>Consistent growth Q1 2025 → Q2 2026 | 2026 Q2 best quarter on record</span>"}
)
fig8.update_yaxes(title_text="P&L (relative)", row=1, col=1)
fig8.update_yaxes(title_text="Cumul. (relative)", row=2, col=1)
fig8.write_image(f"{OUT_DIR}/chart8_quarterly_pnl.png")
print("✅ chart8")

# ── Chart 9: Ticker efficiency scatter ───────────────────────────────────────
EFFICIENCY = {
    "COST": (34, 210), "NVDA": (38, 175), "MRVL": (10, 88),
    "AVGO": (20, 72),  "IWM":  (22, 65),  "MU":   (18, 58),
    "GOOGL":(24, 52),  "SPXW": (16, 48),  "CLS":  (14, 44),
    "TSLA": (28, 31),  "PLTR": (8, -28),  "MSFT": (12, -95),
    "NFLX": (10, -68), "CRWD": (7, -22),
}
eff = pd.DataFrame(
    [(t, c, p) for t, (c, p) in EFFICIENCY.items()],
    columns=["ticker", "trade_count", "total_pnl"]
)
eff["pnl_per_trade"] = eff["total_pnl"] / eff["trade_count"]

fig9 = go.Figure(go.Scatter(
    x=eff["trade_count"], y=eff["pnl_per_trade"],
    mode="markers+text",
    text=eff["ticker"], textposition="top center", textfont=dict(size=11),
    marker=dict(
        size=[max(10, min(45, abs(v) / 4)) for v in eff["total_pnl"]],
        color=["#e74c3c" if v < 0 else "#00d4a8" for v in eff["total_pnl"]],
        opacity=0.85, line=dict(width=1, color="white"),
    )
))
fig9.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
fig9.update_layout(
    title={"text": "Ticker Efficiency: P&L per Trade vs Frequency<br>"
                   "<span style='font-size:15px;font-weight:normal;'>Bubble size = |Total P&L| · green = profit · red = loss</span>"},
    xaxis=dict(title_text="Number of Trades (SELL_OPEN)"),
    yaxis=dict(title_text="P&L per Trade (relative)"),
    height=620, margin=dict(l=80, r=60, t=120, b=60)
)
fig9.write_image(f"{OUT_DIR}/chart9_ticker_efficiency.png")
print("✅ chart9")

print(f"\n✅  9 sample charts saved to ./{OUT_DIR}/")
print("   Note: charts use synthetic data reflecting narrative patterns.")
print("   Actual P&L figures maintained in private repository.")
