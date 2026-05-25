# Options Trading Automation

**End-to-end automated pipeline for selling vertical put credit spreads — from screening to execution to position management.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Tastytrade](https://img.shields.io/badge/Tastytrade-SDK%20v12-purple.svg)](https://tastytrade.com)
[![pandas](https://img.shields.io/badge/pandas-2.0+-green.svg)](https://pandas.pydata.org/)

## Overview

This pipeline automates the full lifecycle of a put credit spread strategy:

1. **Screen** — identify high-probability entry candidates using 10+ technical and volatility indicators
2. **Execute** — place vertical spread orders on Tastytrade at 3:30 PM ET (1 hour before close)
3. **Monitor** — auto-close positions at 80% profit target or DTE threshold
4. **Alert** — send Gmail notifications on every entry, close, and error event

Designed to run unattended on a cloud server (AWS EC2) while traveling across time zones.

---

## Architecture

```
options-screener/
├── options_premium_screener.py   # Screening engine → signals_YYYYMMDD.csv
├── auto_trade.py                 # Daily pipeline orchestrator
├── notifications.py              # Unified alert system (Gmail / Telegram)
├── position_tracker.py           # CSV-based position ledger
├── healthcheck.py                # Pre-flight environment check
├── requirements.txt
│
├── broker/
│   ├── client.py                 # Tastytrade session management
│   ├── executor.py               # Order placement + position monitor
│   ├── spread_builder.py         # Option chain → SpreadSpec → NewOrder
│   └── setup_auth.py             # One-time OAuth credential setup
│
└── tests/
    ├── test_auto_trade.py        # Signal parsing unit tests
    ├── test_executor.py          # Order execution unit tests
    └── test_spread_builder.py    # Spread construction unit tests

deploy/
├── setup.sh                      # One-command Ubuntu server setup
├── run_entry.sh                  # Cron wrapper: screener + entry
├── run_monitor.sh                # Cron wrapper: position monitor
└── crontab.template              # Cron schedule (ET timezone)
```

---

## Pipeline Flow

```
[3:30 PM ET — cron]
        │
        ▼
options_premium_screener.py
  ├─ Fetch VIX → determine regime (LOW / NORMAL / ELEVATED / HIGH)
  ├─ Download 1yr OHLCV for each ticker (yfinance)
  ├─ Compute RSI, Bollinger Bands, ATR, MACD, IV Rank, IV/HV
  ├─ Apply VIX-adjusted filters + earnings blackout
  ├─ Score signals 0–100
  └─ Output: signals_YYYYMMDD.csv

        │
        ▼
auto_trade.py entry
  ├─ Load + normalize signals CSV
  ├─ Check monthly drawdown circuit breaker (positions.csv)
  ├─ For each signal (sorted by strength, max 5):
  │     broker/spread_builder.py → find best strike/expiry
  │     broker/executor.py       → place limit order (dry-run or live)
  │     notifications.py         → Gmail alert on entry
  └─ notify_monitor_summary()

[12:30 PM ET + 4:05 PM ET — cron]
        │
        ▼
auto_trade.py monitor
  ├─ Fetch all open positions across accounts
  ├─ Evaluate close triggers:
  │     profit_target  — P&L ≥ 80% of credit collected
  │     dte_expiry     — DTE ≤ 14
  │     emergency      — price ≤ long put strike
  ├─ Place BTC orders for triggered positions
  └─ notifications.py → Gmail alert on each close
```

---

## Screening Logic

### Tier Structure

| Tier | Tickers | Delta Target | Notes |
|------|---------|--------------|-------|
| **TIER1_CORE** | SPX, COST, NVDA, IWM, GOOGL | 0.10–0.18 | SPX: gap-down + RSI trigger |
| **TIER2_WATCH** | MSFT, AAPL, AMZN, META, AVGO, CRWD, PLTR, AMD, MU, TSLA, QQQM, CLS, STX | 0.08–0.13 | ATR% cap ≤ 5.0% |

### Technical Indicators

| Indicator | Purpose | Threshold |
|-----------|---------|-----------|
| RSI (14) | Oversold detection | VIX-adjusted: 28–38 |
| Bollinger Bands (20) | Price extremes | BB position < VIX-adjusted threshold |
| SMA (200) | Long-term trend filter | Price > SMA200 required |
| ATR % (14) | Volatility measurement | > 1.0% required; Tier 2 cap 5.0% |
| Volume Surge | Liquidity confirmation | > 1.2× 50-day avg |
| IV Rank (52-week) | Premium quality — Pass 1 | ≥ 25 |
| IV/HV Ratio | Premium quality — Pass 2 | ≥ 1.0 |
| VIX Regime | Macro classifier | LOW / NORMAL / ELEVATED / HIGH |

### VIX-Adjusted Thresholds

| Regime | VIX | RSI | BB Position | Rationale |
|--------|-----|-----|-------------|-----------|
| LOW | < 15 | 28 | < 0.25 | Premium thin — deep oversold only |
| NORMAL | 15–20 | 35 | < 0.40 | Standard |
| ELEVATED | 20–30 | 38 | < 0.45 | Fat premium — slightly relaxed |
| HIGH | > 30 | 30 | < 0.30 | Tail risk — thresholds tightened |

### Risk Controls

- **Monthly drawdown circuit breaker**: pauses new entries if MTD P&L ≤ −$2,000
- **Max risk per spread**: $1,000 (spread width × 100)
- **High-beta cap**: IONQ, RGTI, MARA limited to 2 contracts
- **Earnings blackout**: ±5/+1 days around earnings
- **Cluster guard**: warns when ≥ 5 tickers trigger simultaneously

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/hbchoi0917/trading.git
cd trading/options-screener
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
nano .env   # fill in TT_SECRET, TT_REFRESH, GMAIL_PASSWORD
```

### 3. Tastytrade OAuth setup (one-time)

```bash
python broker/setup_auth.py
# Choose option 2 → paste Client Secret + Refresh Token → certification? y
```

Get credentials from [developer.tastytrade.com](https://developer.tastytrade.com) → Sandbox → Create Application.

### 4. Health check

```bash
python healthcheck.py
```

All required checks must pass before proceeding.

### 5. Test connection

```bash
python auto_trade.py monitor   # dry-run, no orders submitted
```

---

## Cloud Deployment (AWS EC2)

```bash
# On your EC2 instance (Ubuntu 24.04):
git clone https://github.com/hbchoi0917/trading.git
cd trading
git checkout main
bash deploy/setup.sh

# Fill in credentials:
nano options-screener/.env

# Verify:
cd options-screener && /path/to/.venv/bin/python3 healthcheck.py
```

`setup.sh` handles: Python venv, dependencies, US/Eastern timezone, logrotate, and cron installation automatically.

### Cron Schedule (US/Eastern)

| Time ET | Time KST | Job |
|---------|----------|-----|
| 12:30 PM | 1:30 AM | Monitor — profit target / DTE checks |
| 3:30 PM | 4:30 AM | Entry — screener + order placement |
| 4:05 PM | 5:05 AM | Monitor — post-close final scan |

---

## Usage

```bash
# Run screener only (generates signals_YYYYMMDD.csv)
python options_premium_screener.py

# Place orders from today's signals (dry-run)
python auto_trade.py entry

# Place orders live
python auto_trade.py entry --live

# Monitor + auto-close positions (dry-run)
python auto_trade.py monitor

# Run full pipeline: entry + monitor
python auto_trade.py all
```

---

## Configuration (.env)

```bash
# Tastytrade
TT_PAPER_TRADING=true       # true = sandbox, false = live
TT_DRY_RUN=true             # true = log only, false = submit orders
TT_SECRET=                  # OAuth client secret
TT_REFRESH=                 # OAuth refresh token
TT_ACCOUNT_NUMBERS=         # comma-separated (blank = all)

# Notifications (Gmail — primary; Telegram — optional)
GMAIL_SENDER=
GMAIL_PASSWORD=             # App Password (not your login password)
GMAIL_RECEIVER=
TG_BOT_TOKEN=               # optional
TG_CHAT_ID=                 # optional
```

---

## Testing

```bash
python -m pytest tests/ -v
# 43 tests, all passing — no live broker connection required
```

Tests cover: signal normalization, delta parsing, spread construction (mocked option chain), order execution (mocked account), monthly drawdown circuit breaker.

---

## Go-Live Checklist

- [ ] `python healthcheck.py` — all required checks pass
- [ ] `python auto_trade.py monitor` — connects, scans positions without error
- [ ] `TT_DRY_RUN=false`, `TT_PAPER_TRADING=true` — sandbox orders fill correctly
- [ ] Gmail alerts received for entry / close events
- [ ] Cron fires at 3:30 PM ET, log appears in `logs/entry_YYYY-MM-DD.log`
- [ ] Set `TT_PAPER_TRADING=false` only after sandbox testing is complete

---

## Dependencies

- **tastytrade** ≥ 12.0.0 — broker SDK (OAuth, order placement, position data)
- **pandas** ≥ 2.0.0 — data manipulation
- **yfinance** ≥ 0.2.28 — market data, option chains, earnings calendar
- **python-dotenv** ≥ 1.0.0 — environment variable management
- **pandas-ta** — optional TA library; falls back to built-in pandas RSI/ATR/MACD if unavailable

---

## Disclaimer

This tool is for personal use and educational purposes only. Not financial advice. Options trading involves substantial risk of loss. Past screening results do not guarantee future performance.
