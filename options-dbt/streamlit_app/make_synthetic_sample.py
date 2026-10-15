"""
Build a FULLY SYNTHETIC sample_options_trading.duckdb with the exact schema the
Streamlit app reads (current dbt models). Every row is random (fixed seed) —
no real trade, amount, date or account is used.
Usage: python make_synthetic_sample.py OUT.duckdb  (OUT must not exist or will be overwritten table by table)
"""
import sys, hashlib
import numpy as np, pandas as pd, duckdb

OUT = sys.argv[1]
rng = np.random.default_rng(20261015)

ACCOUNTS = {"Account_A": 0.40, "Account_B": 0.30, "Account_C": 0.25, "Account_D": 0.05}
TICKERS = {  # ticker: (approx price, put share, edge)
    "SPY": (560, 0.95, 0.9), "QQQ": (480, 0.95, 0.8), "IWM": (215, 0.95, 0.6), "AAPL": (220, 0.85, 0.7),
    "NVDA": (130, 0.85, 0.8), "GOOGL": (170, 0.85, 0.7), "AMZN": (190, 0.85, 0.6), "META": (560, 0.85, 0.5),
    "AMD": (150, 0.85, 0.4), "JPM": (210, 0.9, 0.6), "TSLA": (240, 0.8, 0.3), "MSFT": (420, 0.9, -1.2),
    "NFLX": (680, 0.9, -1.0),
}
EXCLUDED = {"MSFT", "NFLX"}       # mirrors the app's "removed tickers" section; numbers are random
start, end = pd.Timestamp("2025-01-02"), pd.Timestamp("2026-09-25")
bdays = pd.bdate_range(start, end)

rows = []
for i in range(900):
    acct = rng.choice(list(ACCOUNTS), p=list(ACCOUNTS.values()))
    tk = rng.choice(list(TICKERS))
    px, put_share, edge = TICKERS[tk]
    opt = "PUT" if rng.random() < put_share else "CALL"
    open_d = pd.Timestamp(rng.choice(bdays))
    dte = int(rng.integers(14, 46))
    expiry = open_d + pd.Timedelta(days=dte)
    width = float(rng.choice([5, 10]))
    contracts = float(rng.choice([1, 1, 1, 2]))
    credit_ps = round(width * rng.uniform(0.10, 0.30), 2)
    entry_credit = round(credit_ps * 100 * contracts, 2)
    strike = round(px * (0.92 if opt == "PUT" else 1.08) / width) * width
    fees = round(0.66 * 2 * contracts, 2)
    p_win = min(0.92, max(0.35, 0.72 + 0.08 * edge))
    if rng.random() < p_win:
        if rng.random() < 0.35:   # held to expiry, expired worthless
            close_d, exit_amt, reason = expiry, 0.0, "EXPIRED_WORTHLESS"
        else:
            capture = rng.uniform(0.45, 0.95)
            close_d = open_d + pd.Timedelta(days=int(rng.integers(3, max(4, dte - 2))))
            exit_amt, reason = round(entry_credit * (1 - capture), 2), "CLOSED"
    else:
        loss_mult = min(rng.gamma(2.0, 0.8), width * 100 * contracts / entry_credit - 1)
        close_d = open_d + pd.Timedelta(days=int(rng.integers(2, max(3, dte))))
        exit_amt, reason = round(entry_credit * (1 + loss_mult), 2), "CLOSED"
    rows.append(dict(account=acct, ticker=tk, option_type=opt, open_date=open_d, close_date=close_d,
                     expiry_date=expiry, strike=float(strike), width=width, contracts=contracts,
                     entry_credit=entry_credit, exit_amount=exit_amt, total_fees=fees, close_reason=reason))
t = pd.DataFrame(rows)
# a handful still open at the data cut-off
open_mask = t["open_date"] > end - pd.Timedelta(days=20)
t.loc[open_mask, ["close_date", "exit_amount"]] = [pd.NaT, np.nan]
t.loc[open_mask, "close_reason"] = "OPEN"
t["is_open"] = open_mask
t["realized_pnl"] = np.where(t.is_open, np.nan, (t.entry_credit - t.exit_amount).round(2))
t["realized_pnl_gross"] = (t.realized_pnl + t.total_fees).round(2)
t["holding_days"] = (t.close_date.fillna(end) - t.open_date).dt.days.astype("int64")
t["trade_result"] = np.where(t.is_open, "OPEN", np.where(t.realized_pnl > 0, "WIN", np.where(t.realized_pnl < 0, "LOSS", "BREAKEVEN")))
t["max_risk_proxy"] = t.entry_credit.abs()          # same proxy as the dbt model
t["pnl_pct_of_credit"] = (t.realized_pnl / t.entry_credit * 100).round(1)
t["open_month"] = t.open_date.dt.to_period("M").dt.to_timestamp()
t["open_quarter"] = t.open_date.dt.to_period("Q").dt.to_timestamp()
t["open_year"] = t.open_date.dt.year.astype("int64")
t["spread_id"] = [hashlib.md5(f"synthetic-{i}".encode()).hexdigest() for i in range(len(t))]
t["margin_type"] = np.where(t.account.isin(["Account_A"]), "Margin", "Cash")
t["leg_count"] = 2
t["account_name"] = t.account
trades = t[["spread_id", "account_name", "ticker", "expiry_date", "option_type", "strike", "margin_type", "open_date", "close_date",
            "holding_days", "entry_credit", "exit_amount", "realized_pnl", "total_fees", "realized_pnl_gross", "leg_count",
            "contracts", "close_reason", "is_open", "trade_result", "max_risk_proxy", "pnl_pct_of_credit", "open_month",
            "open_quarter", "open_year"]].copy()

c = t[~t.is_open].copy()
c["month"] = c.close_date.dt.to_period("M").dt.to_timestamp()
c["win"] = (c.realized_pnl > 0).astype(float)
c["loss"] = (c.realized_pnl < 0).astype(float)

m = c.groupby(["month", "account"]).agg(trade_count=("realized_pnl", "size"), wins=("win", "sum"), losses=("loss", "sum"),
                                        net_pnl=("realized_pnl", "sum"), total_fees=("total_fees", "sum")).reset_index()
m = m.sort_values(["account", "month"])
m["year"] = m.month.dt.year.astype("int64"); m["month_num"] = m.month.dt.month.astype("int64")
m["net_pnl"] = m.net_pnl.round(2); m["total_fees"] = m.total_fees.round(2)
m["net_pnl_after_fees"] = m.net_pnl        # amounts are already net of fees
m["win_rate_pct"] = (m.wins / m.trade_count * 100).round(1)
m["cumulative_pnl"] = m.groupby("account").net_pnl.cumsum().round(2)
m["pnl_mom_change"] = m.groupby("account").net_pnl.diff().round(2)
m["is_win_month"] = m.net_pnl > 0
monthly = m.rename(columns={"account": "account_name"})[["month", "year", "month_num", "account_name", "trade_count", "wins",
          "losses", "net_pnl", "total_fees", "net_pnl_after_fees", "win_rate_pct", "cumulative_pnl", "pnl_mom_change", "is_win_month"]]
monthly["trade_count"] = monthly.trade_count.astype("int64")

acc = []
for a, g in c.groupby("account"):
    mm = m[m.account == a]
    acc.append(dict(account_name=a, total_trades=int(len(g)), total_wins=float(g.win.sum()), total_losses=float(g.loss.sum()),
                    trade_win_rate_pct=round(g.win.mean() * 100, 1), monthly_win_rate_pct=round(mm.is_win_month.mean() * 100, 1),
                    lifetime_pnl=round(g.realized_pnl.sum(), 2), lifetime_fees=round(g.total_fees.sum(), 2),
                    lifetime_pnl_net_fees=round(g.realized_pnl.sum(), 2), avg_pnl_per_trade=round(g.realized_pnl.mean(), 2),
                    best_single_trade=round(g.realized_pnl.max(), 2), worst_single_trade=round(g.realized_pnl.min(), 2),
                    best_month_pnl=round(mm.net_pnl.max(), 2), worst_month_pnl=round(mm.net_pnl.min(), 2),
                    unique_tickers_traded=int(g.ticker.nunique()), active_months=int(len(mm)), total_months=int(len(mm)),
                    win_months=float(mm.is_win_month.sum()), first_trade_date=g.open_date.min(), last_trade_date=g.close_date.max()))
accounts = pd.DataFrame(acc)

tp = []
for (tk, ot), g in c.groupby(["ticker", "option_type"]):
    tp.append(dict(ticker=tk, option_type=ot, total_trades=int(len(g)), wins=float(g.win.sum()), losses=float(g.loss.sum()),
                   total_pnl=round(g.realized_pnl.sum(), 2), avg_pnl_per_trade=round(g.realized_pnl.mean(), 2),
                   best_trade=round(g.realized_pnl.max(), 2), worst_trade=round(g.realized_pnl.min(), 2),
                   total_fees=round(g.total_fees.sum(), 2), avg_holding_days=round(g.holding_days.mean(), 1),
                   avg_dte_at_open=round((g.expiry_date - g.open_date).dt.days.mean(), 1),
                   first_trade_date=g.open_date.min(), last_trade_date=g.close_date.max(),
                   accounts_active=int(g.account.nunique()), account_list=", ".join(sorted(g.account.unique())),
                   win_rate_pct=round(g.win.mean() * 100, 1), pnl_per_trade=round(g.realized_pnl.mean(), 2),
                   is_excluded=tk in EXCLUDED))
tickers = pd.DataFrame(tp).sort_values("total_pnl", ascending=False)
tickers.insert(18, "pnl_rank", range(1, len(tickers) + 1))
tickers["pnl_rank"] = tickers.pnl_rank.astype("int64")

con = duckdb.connect(OUT)
for schema in ("main_intermediate", "main_marts"):
    con.execute(f"create schema if not exists {schema}")
con.register("t_", trades); con.execute("create or replace table main_intermediate.int_spread_trades as select * from t_")
con.register("m_", monthly); con.execute("create or replace table main_marts.mart_monthly_pnl as select * from m_")
con.register("a_", accounts); con.execute("create or replace table main_marts.mart_account_summary as select * from a_")
con.register("k_", tickers); con.execute("create or replace table main_marts.mart_ticker_performance as select * from k_")
con.execute("create or replace table main.about_this_data as select 'SYNTHETIC — every row is randomly generated (seed 20261015). Not real trades, amounts or accounts.' as note")
con.close()
print("trades", len(trades), "closed", len(c), "accounts", len(accounts), "tickers", len(tickers), "months", monthly.month.nunique())
print("lifetime pnl (synthetic):", round(c.realized_pnl.sum()), "win rate:", round(c.win.mean() * 100, 1))
