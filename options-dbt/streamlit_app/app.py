"""
Options Trading Analytics Dashboard
------------------------------------
Streamlit + DuckDB + Plotly
Public demo runs on synthetic sample data (real trade history is private).
Point DUCKDB_PATH at a dbt-built database to use it with real data.
"""

import os
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Options Trading Analytics",
    page_icon="📈",
    layout="wide",
)

# ── DB connection ─────────────────────────────────────────────────────────────

DB_PATH = os.environ.get("DUCKDB_PATH", os.path.join(os.path.dirname(__file__), "sample_options_trading.duckdb"))

@st.cache_resource
def get_conn():
    return duckdb.connect(DB_PATH, read_only=True)

@st.cache_data(ttl=3600)
def query(sql: str) -> pd.DataFrame:
    return get_conn().execute(sql).df()

# ── Load data ─────────────────────────────────────────────────────────────────

monthly   = query("SELECT * FROM main_marts.mart_monthly_pnl ORDER BY month, account_name")
tickers   = query("SELECT * FROM main_marts.mart_ticker_performance ORDER BY total_pnl DESC")
accounts  = query("SELECT * FROM main_marts.mart_account_summary ORDER BY lifetime_pnl DESC")
trades    = query("SELECT * FROM main_intermediate.int_spread_trades WHERE is_open = false ORDER BY close_date DESC")

# ── Sidebar nav ───────────────────────────────────────────────────────────────

st.sidebar.title("📈 Options Analytics")
page = st.sidebar.radio(
    "Navigate",
    ["Portfolio Overview", "Ticker Performance", "Account Summary"],
)

st.sidebar.divider()
st.sidebar.caption("**Stack:** dbt · DuckDB · Streamlit · Plotly")
st.sidebar.caption("**Data:** Synthetic sample data — real trade history is private")
st.sidebar.caption("**Accounts:** 4 · **Tickers:** 48 · **Trades:** 162")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: Portfolio Overview
# ─────────────────────────────────────────────────────────────────────────────

if page == "Portfolio Overview":
    st.title("Portfolio Overview")
    st.caption("Demo with synthetic sample data — figures are illustrative  ·  Jan 2025 – May 2026  ·  4 Accounts")

    # KPI row
    total_pnl   = accounts["lifetime_pnl"].sum()
    total_trades = accounts["total_trades"].sum()
    avg_win_rate = monthly["win_rate_pct"].mean()
    total_fees  = accounts["lifetime_fees"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lifetime P&L",     f"${total_pnl:,.0f}")
    c2.metric("Total Trades",     f"{int(total_trades):,}")
    c3.metric("Monthly Win Rate", f"{avg_win_rate:.1f}%")
    c4.metric("Total Fees Paid",  f"${total_fees:,.0f}")

    st.divider()

    # Monthly P&L + Cumulative combo
    monthly_agg = (
        monthly.groupby("month", as_index=False)
        .agg(net_pnl=("net_pnl", "sum"), cumulative_pnl=("cumulative_pnl", "max"))
        .sort_values("month")
    )

    fig = go.Figure()
    colors = monthly_agg["net_pnl"].apply(lambda v: "#00C851" if v >= 0 else "#FF4444")
    fig.add_bar(
        x=monthly_agg["month"], y=monthly_agg["net_pnl"],
        name="Monthly P&L", marker_color=colors,
        yaxis="y1",
    )
    fig.add_scatter(
        x=monthly_agg["month"], y=monthly_agg["cumulative_pnl"],
        name="Cumulative P&L", mode="lines+markers",
        line=dict(color="#4A90D9", width=2.5),
        yaxis="y2",
    )
    fig.update_layout(
        title="Monthly Net P&L & Cumulative Growth",
        yaxis=dict(title="Monthly P&L ($)"),
        yaxis2=dict(title="Cumulative P&L ($)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Monthly by account (stacked)
    col_left, col_right = st.columns(2)

    with col_left:
        fig2 = px.bar(
            monthly, x="month", y="net_pnl", color="account_name",
            barmode="stack",
            title="Monthly P&L by Account",
            labels={"net_pnl": "Net P&L ($)", "month": "", "account_name": "Account"},
            color_discrete_sequence=["#4A90D9", "#00C851", "#F4B548", "#8F3D56"],
            height=350,
        )
        fig2.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig2, use_container_width=True)

    with col_right:
        # Quarterly summary table
        monthly["quarter"] = pd.to_datetime(monthly["month"]).dt.to_period("Q").astype(str)
        qtly = (
            monthly.groupby("quarter", as_index=False)
            .agg(net_pnl=("net_pnl", "sum"), trade_count=("trade_count", "sum"), win_rate_pct=("win_rate_pct", "mean"))
            .sort_values("quarter")
        )
        qtly["net_pnl"] = qtly["net_pnl"].map("${:,.0f}".format)
        qtly["win_rate_pct"] = qtly["win_rate_pct"].map("{:.1f}%".format)
        qtly.columns = ["Quarter", "Net P&L", "Trades", "Win Rate"]
        st.markdown("**Quarterly Breakdown**")
        st.dataframe(qtly, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: Ticker Performance
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Ticker Performance":
    st.title("Ticker Performance")
    st.caption("Demo with synthetic sample data — figures are illustrative · P&L breakdown by ticker · Jan 2025 – May 2026")

    col_left, col_right = st.columns(2)

    with col_left:
        winners = tickers[tickers["total_pnl"] > 0].head(12)
        fig = px.bar(
            winners, x="total_pnl", y="ticker", orientation="h",
            title="Top Winners",
            labels={"total_pnl": "Net P&L ($)", "ticker": ""},
            color="total_pnl",
            color_continuous_scale=["#90EE90", "#00C851"],
            height=400,
        )
        fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        losers = tickers[tickers["total_pnl"] < 0].tail(8)
        fig = px.bar(
            losers, x="total_pnl", y="ticker", orientation="h",
            title="Biggest Losses",
            labels={"total_pnl": "Net P&L ($)", "ticker": ""},
            color="total_pnl",
            color_continuous_scale=["#FF4444", "#FFB3B3"],
            height=400,
        )
        fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Bubble: win rate vs P&L
    fig3 = px.scatter(
        tickers, x="win_rate_pct", y="total_pnl",
        size="total_trades", color="is_excluded",
        hover_name="ticker",
        title="Win Rate vs Total P&L  (bubble size = trade count)",
        labels={"win_rate_pct": "Win Rate (%)", "total_pnl": "Total P&L ($)", "is_excluded": "Excluded"},
        color_discrete_map={True: "#FF4444", False: "#4A90D9"},
        size_max=50,
        height=420,
    )
    fig3.add_hline(y=0, line_dash="dot", line_color="gray")
    fig3.add_vline(x=50, line_dash="dot", line_color="gray")
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # Excluded tickers callout
    excluded = tickers[tickers["is_excluded"] == True][["ticker", "total_pnl", "total_trades", "win_rate_pct"]]
    st.markdown("**⚠️ Excluded from Rotation**")
    st.caption("MSFT and NFLX removed after large idiosyncratic losses in Q1 2026. IONQ and RGTI removed due to quantum sector volatility.")
    ex_display = excluded.copy()
    ex_display["total_pnl"] = ex_display["total_pnl"].map("${:,.0f}".format)
    ex_display["win_rate_pct"] = ex_display["win_rate_pct"].map("{:.1f}%".format)
    ex_display.columns = ["Ticker", "Total P&L", "Trades", "Win Rate"]
    st.dataframe(ex_display, use_container_width=True, hide_index=True)

    st.divider()

    # Full table with search
    st.markdown("**Full Ticker Table**")
    search = st.text_input("Search ticker", "")
    display = tickers if not search else tickers[tickers["ticker"].str.contains(search.upper())]
    display_fmt = display[[
        "ticker", "option_type", "total_trades", "wins", "losses",
        "win_rate_pct", "total_pnl", "avg_pnl_per_trade", "avg_holding_days"
    ]].copy()
    display_fmt.columns = [
        "Ticker", "Type", "Trades", "Wins", "Losses",
        "Win %", "Total P&L", "Avg P&L/Trade", "Avg Days"
    ]
    st.dataframe(
        display_fmt.style
            .format({"Total P&L": "${:,.0f}", "Avg P&L/Trade": "${:,.0f}", "Win %": "{:.1f}%", "Avg Days": "{:.1f}"}),
        use_container_width=True,
        hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: Account Summary
# ─────────────────────────────────────────────────────────────────────────────

elif page == "Account Summary":
    st.title("Account Summary")
    st.caption("Demo with synthetic sample data — figures are illustrative · Lifetime performance across 4 accounts")

    # Account KPI cards
    cols = st.columns(len(accounts))
    for i, (_, row) in enumerate(accounts.iterrows()):
        cols[i].metric(
            row["account_name"],
            f"${row['lifetime_pnl']:,.0f}",
            f"{row['monthly_win_rate_pct']:.1f}% monthly win rate",
        )

    st.divider()

    # Cumulative P&L by account
    fig = px.line(
        monthly, x="month", y="cumulative_pnl", color="account_name",
        title="Cumulative P&L Growth by Account",
        labels={"cumulative_pnl": "Cumulative P&L ($)", "month": "", "account_name": "Account"},
        color_discrete_sequence=["#4A90D9", "#00C851", "#F4B548", "#8F3D56"],
        height=380,
    )
    fig.update_traces(mode="lines+markers", marker=dict(size=4))
    st.plotly_chart(fig, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        # Grouped monthly P&L
        fig2 = px.bar(
            monthly, x="month", y="net_pnl", color="account_name",
            barmode="group",
            title="Monthly P&L by Account (Grouped)",
            labels={"net_pnl": "Net P&L ($)", "month": "", "account_name": "Account"},
            color_discrete_sequence=["#4A90D9", "#00C851", "#F4B548", "#8F3D56"],
            height=350,
        )
        fig2.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig2, use_container_width=True)

    with col_right:
        # Account summary table
        tbl = accounts[[
            "account_name", "lifetime_pnl", "total_trades",
            "trade_win_rate_pct", "best_single_trade", "worst_single_trade", "unique_tickers_traded"
        ]].copy()
        tbl.columns = ["Account", "Lifetime P&L", "Trades", "Win %", "Best Trade", "Worst Trade", "Tickers"]
        st.markdown("**Account Stats**")
        st.dataframe(
            tbl.style.format({
                "Lifetime P&L": "${:,.0f}",
                "Best Trade": "${:,.0f}",
                "Worst Trade": "${:,.0f}",
                "Win %": "{:.1f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # Trade detail table
    st.markdown("**Recent Trade Detail**")
    # older sample DBs lack account_name on int_spread_trades
    has_account = "account_name" in trades.columns
    account_filter = "All"
    if has_account:
        account_filter = st.selectbox("Filter by account", ["All"] + sorted(trades["account_name"].unique()))
    result_filter = st.selectbox("Filter by result", ["All", "WIN", "LOSS", "BREAKEVEN"])

    filtered = trades.copy()
    if has_account and account_filter != "All":
        filtered = filtered[filtered["account_name"] == account_filter]
    if result_filter != "All":
        filtered = filtered[filtered["trade_result"] == result_filter]

    detail_cols = (["account_name"] if has_account else []) + [
        "ticker", "option_type", "open_date", "close_date",
        "holding_days", "realized_pnl", "close_reason", "trade_result"
    ]
    display = filtered[detail_cols].head(100).copy()
    display.columns = (["Account"] if has_account else []) + [
        "Ticker", "Type", "Open", "Close",
        "Days", "P&L", "Reason", "Result"
    ]
    st.dataframe(
        display.style.format({"P&L": "${:,.0f}"}),
        use_container_width=True,
        hide_index=True,
    )
