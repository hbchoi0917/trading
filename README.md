# Options Trading Automation

**End-to-end automated pipeline for selling vertical put credit spreads — from screening to execution to position management.**

[![CI](https://github.com/hbchoi0917/trading/actions/workflows/ci.yml/badge.svg)](https://github.com/hbchoi0917/trading/actions/workflows/ci.yml)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B?logo=streamlit&logoColor=white)](https://options-trading-dash.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Tastytrade](https://img.shields.io/badge/Tastytrade-SDK%20v12-purple.svg)](https://tastytrade.com)
[![License](https://img.shields.io/badge/license-Personal%20Use-red.svg)](LICENSE)

---

## What's in this repo

| Folder | Description |
|--------|-------------|
| [`options-screener/`](options-screener/) | **Demo screener** — scans a watchlist for put credit spread entry signals using RSI, Bollinger Bands, IV Rank, IV/HV ratio, and earnings blackout filters. Outputs a signals CSV. |
| [`options-analysis/`](options-analysis/) | **Trade analysis** — pipeline + charts for analyzing personal options trade history (P&L, ticker breakdown, strategy split, efficiency). See its own [README](options-analysis/README.md). |
| [`options-dbt/streamlit_app/`](options-dbt/streamlit_app/) | **Live dashboard** — interactive 3-page Streamlit app (DuckDB + Plotly). [**→ Open Dashboard**](https://options-trading-dash.streamlit.app/) |
| [`deploy/`](deploy/) | **Server setup** — one-command Ubuntu EC2 setup, cron wrappers, log rotation. |

---

## Overview

This pipeline automates the full lifecycle of a put credit spread strategy:

1. **Screen** — identify high-probability entry candidates using 10+ technical and volatility indicators
2. **Execute** — place vertical spread orders on Tastytrade during peak intraday liquidity windows, when bid-ask spreads are tightest (market-open and near-close entries are avoided — inflated IV and wide spreads degrade fill quality)
3. **Monitor** — auto-close positions at profit target or DTE threshold
4. **Alert** — send Telegram / Gmail notifications on every entry, close, and error event
5. **Covered Calls** — separate alert module for manual CC management on Fidelity positions

Designed to run unattended on a cloud server (AWS EC2).

> **Note:** The screener in this repo is a **demo / educational baseline**. The full production system (broker integration, order execution, position monitoring, cron-scheduled deployment) runs in a private repository. See the [customization notes](options-screener/screener.py) at the top of `screener.py` for what's needed to build a complete system.

---

## Screening Logic

Tickers are organized into tiers by liquidity and volatility profile. All signals must pass momentum, mean-reversion, volatility, and earnings blackout filters. RSI and Bollinger Band thresholds shift automatically across four VIX regimes (LOW / NORMAL / ELEVATED / HIGH).

---

## Risk Management

Every order passes through layered, deterministic safeguards before and after entry:

- **Pre-trade buying power check** — required margin is validated against
  available option buying power before any order is submitted; fails open
  to the broker's own enforcement on API errors
- **Per-spread risk cap** — maximum defined risk per position, with
  ticker-class-specific overrides (index vs. equity vs. high-beta names)
- **Per-ticker contract limits** — high-volatility names are capped at
  reduced contract counts regardless of signal strength
- **Monthly drawdown circuit breaker** — all new entries pause automatically
  if month-to-date P&L breaches a configured loss limit
- **Profit target close** — positions are closed automatically once a
  configured percentage of the collected credit has decayed
- **Stop loss** — positions are closed when the loss reaches a multiple of
  the credit collected (the standard premium-selling convention); the stop
  fires immediately and is never gated by the holding window
- **DTE-based forced close** — positions are closed regardless of P&L once
  expiration approaches, avoiding gamma risk in the final weeks
- **Minimum holding window** — profit-taking is gated by a minimum holding
  period to prevent same-day churn; risk-management closes (stop loss, DTE)
  are exempt and always fire
- **Fill confirmation** — close orders are verified against live order status
  before being reported as filled; unfilled orders are re-priced toward the
  ask (price chasing), then cancelled and retried on the next run
- **Position ledger** — every entry and close is recorded to a CSV ledger,
  driving month-to-date statistics in summary notifications
- **Earnings blackout** — entries are skipped around earnings announcements
- **Concentration guard** — simultaneous signals across correlated tickers
  are flagged as a single macro bet, not independent trades

---

## Quick Start

```bash
# 1. Install dependencies
cd options-screener
pip install -r requirements.txt

# 2. Run the screener
python screener.py
# → signals_YYYYMMDD.csv + console output
```

For full automation, additional setup is required:
- **Broker API** — integrate with Tastytrade, IBKR, or similar for order placement
- **Cloud server** — deploy to AWS EC2 (or equivalent) with cron scheduling in US/Eastern timezone
- **Notifications** — add Telegram / Gmail alert integration

---

## Cron Schedule (Production)

| Time ET | Job |
|---------|-----|
| 12:30 PM | Monitor — profit target / DTE checks |
| 3:25 PM | Covered call alert (manual execution in Fidelity) |
| 3:30 PM | Entry — screener + order placement |
| 4:05 PM | Monitor — post-close final scan |
| 4:30 PM | Summary — daily email; weekly email on last trading day |

---

## Design Decision: Rule-Based, Not AI-Driven

With brokerages launching AI-powered trading features — where models trade
on behalf of users and loss accountability remains a legal grey area — this
system deliberately takes the opposite architecture. This is a considered
engineering decision, not a limitation.

| | This system (rule-based) | Quant (statistics-driven) | AI (LLM/ML judgment) |
|--|--|--|--|
| Decision basis | Explicit thresholds | Data-derived formulas | Probabilistic model output |
| Reproducibility | Full | Full (same data → same result) | Not guaranteed |
| Explainability | Line-of-code level | Explicit formulas | Near-impossible |
| Accountability | 100% the owner's | 100% the owner's | Grey area |

**Why deterministic rules:**

1. **The edge is structural, not predictive.** Premium selling profits from
   the volatility risk premium (implied vol persistently exceeding realized
   vol) — closer to selling insurance than forecasting direction. A
   prediction model adds little to this edge.
2. **Sample size is too small for ML.** A handful of trades per month cannot
   train or even validate a model — overfitting is guaranteed.
3. **Auditability is the moat.** Every entry and exit traces to an explicit
   rule. Subtle execution bugs (leg mis-pairing across positions, same-day
   open/close churn) were only findable *because* the logic is deterministic
   — with probabilistic judgment, "bug or model decision?" becomes
   unanswerable.
4. **Responsibility stays clear.** User-authored rules, user's account,
   user's risk caps. Losses are unambiguously the owner's responsibility —
   no broker-AI fiduciary ambiguity.

**The accepted evolution path is quant-style, not AI-style:** as the trade
ledger accumulates history, fixed thresholds can become data-derived
(volatility-rank-based delta adjustment, per-ticker sizing from realized win
rates, ATR-based width selection) — all explicit formulas, all backtestable,
all reproducible. The VIX regime-adjusted thresholds already in place are
the first step in that direction. LLM-based trade judgment is permanently
out of scope for this strategy.

---

## Tech Stack

- **[yfinance](https://pypi.org/project/yfinance/)** — market data and option chains
- **[pandas](https://pandas.pydata.org/)** — data manipulation
- **[tastytrade](https://pypi.org/project/tastytrade/)** ≥ 12.0 — broker SDK (production)
- **[python-dotenv](https://pypi.org/project/python-dotenv/)** — credential management

---

## License

Personal use only — not for commercial use or redistribution. See [LICENSE](LICENSE).

---

## Disclaimer

For personal use and educational purposes only. Not financial advice. Options trading involves substantial risk of loss.
