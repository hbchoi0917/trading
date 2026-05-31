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
2. **Execute** — place vertical spread orders on Tastytrade 1 hour before market close
3. **Monitor** — auto-close positions at profit target or DTE threshold
4. **Alert** — send Telegram / Gmail notifications on every entry, close, and error event
5. **Covered Calls** — separate alert module for manual CC management on Fidelity positions

Designed to run unattended on a cloud server (AWS EC2).

> **Note:** The screener in this repo is a **demo / educational baseline**. The full production system (broker integration, order execution, position monitoring, cron-scheduled deployment) runs in a private repository. See the [customization notes](options-screener/screener.py) at the top of `screener.py` for what's needed to build a complete system.

---

## Screening Logic

Tickers are organized into tiers by liquidity and volatility profile. All signals must pass momentum, mean-reversion, volatility, and earnings blackout filters. RSI and Bollinger Band thresholds shift automatically across four VIX regimes (LOW / NORMAL / ELEVATED / HIGH).

---

## Risk Controls

- Profit target close, DTE-based close, and emergency BTC with retry logic
- Hard cap on concurrent open positions
- Monthly drawdown circuit breaker — pauses entries if MTD P&L hits limit
- Per-ticker exposure cap across all accounts

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
