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
4. **Alert** — send Telegram / Gmail notifications on every entry, close, and error event
5. **Covered Calls** — separate alert module for manual CC management on Fidelity positions

Designed to run unattended on a cloud server (AWS EC2) while traveling across time zones.

---

## Architecture

```
options-screener/
├── options_premium_screener.py   # Screening engine → signals_YYYYMMDD.csv
├── covered_call_screener.py      # CC alert module (Fidelity positions, manual execution)
├── auto_trade.py                 # Daily pipeline orchestrator
├── notifications.py              # Unified alert system (Telegram / Gmail)
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
├── run_cc.sh                     # Cron wrapper: covered call screener
├── run_summary.sh                # Cron wrapper: daily/weekly summary email
└── crontab.template              # Cron schedule (ET timezone)
```

---

## Pipeline Flow

```
[3:25 PM ET — cron]
        │
        ▼
covered_call_screener.py
  ├─ Scan VOO, QQQM, EWY, GOOGL, SCHD, DRAM
  ├─ Entry conditions: green day + RSI 58–78 + BB > 0.65 + above SMA200
  ├─ Earnings guard: skip if earnings within 7 days
  └─ Alert: Telegram/Gmail with 3 suggested OTM strikes + bid/ask
     (execute manually in Fidelity)

[3:30 PM ET — cron]
        │
        ▼
options_premium_screener.py
  ├─ Skip if FOMC decision day or VIX < 16
  ├─ Fetch VIX → determine regime (LOW / NORMAL / ELEVATED / HIGH)
  ├─ Download 1yr OHLCV for each ticker (yfinance)
  ├─ Compute RSI, Bollinger Bands, ATR, MACD, IV Rank, IV/HV
  ├─ Apply all entry filters (see Screening Logic below)
  ├─ Score signals 0–100
  └─ Output: signals_YYYYMMDD.csv

        │
        ▼
auto_trade.py entry
  ├─ Load + normalize signals CSV
  ├─ Check monthly drawdown circuit breaker (positions.csv)
  ├─ Check portfolio exposure caps (MSFT/TSLA/SNDK/ASML — cross-account total)
  ├─ Check live position count → available slots = 25 − open_spreads
  ├─ For each signal (ranked by strength, up to min(10, available_slots)):
  │     spread_builder.py → compare $10×1 vs $5×2, pick higher total premium
  │     executor.py       → place limit order (dry-run or live)
  │     notifications.py  → Telegram/Gmail alert on entry
  └─ notify_monitor_summary()

[12:30 PM ET + 4:05 PM ET — cron]
        │
        ▼
auto_trade.py monitor
  ├─ Fetch all open positions across accounts
  ├─ Evaluate close triggers:
  │     profit_target  — P&L ≥ 80% of credit AND BTC debit ≤ $0.60/share
  │     dte_expiry     — DTE ≤ 12
  │     emergency      — price ≤ long put strike (retry at 1.05× mid after 90s)
  ├─ Place BTC orders for triggered positions
  └─ notifications.py → alert on each close

[4:30 PM ET — cron]
        │
        ▼
auto_trade.py summary
  ├─ Read positions.csv → entries/closes today, open positions, MTD P&L
  ├─ Cap warnings: tickers at >80% of portfolio exposure limit
  ├─ Expiry alerts: positions at DTE ≤ 9
  ├─ Send daily summary email (every trading day)
  └─ Send weekly summary email (last trading day only — Friday or Thursday if Friday is holiday)
        └─ Week P&L, MTD P&L, next-week expiries, top winner/loser
```

---

## Screening Logic

### Tier Structure

| Tier | Tickers | Delta Target | Notes |
|------|---------|--------------|-------|
| **TIER1_CORE** | SPX, COST, NVDA, IWM, GOOGL, TSLA, CLS | 0.15–0.22 (COST: 0.15–0.28, TSLA: 0.12–0.17) | SPX: gap-down + RSI trigger; CLS promoted from T2 |
| **TIER2_WATCH** | AAPL, AMZN, META, AVGO, CRWD, AMD, MU, STX, ASML, GS, JPM, DRAM, MRVL | 0.12–0.20 (MRVL: 0.10–0.15) | ATR% cap ≤ 5.0%; DRAM promoted from T3; MRVL added May 2026 |
| **TIER3_WATCH** | PLTR, MSFT, SNDK, EWY | 0.08–0.13 | ATR% cap ≤ 5.0% — higher volatility, conservative delta |

### Entry Filters (all must pass)

| Filter | Condition | Notes |
|--------|-----------|-------|
| VIX floor | VIX ≥ 16 (warning only) | Below this, log warning but proceed on individual IV merit |
| RSI (14) | VIX + tier-adjusted oversold | Tier 1: 33–43 / Tier 2/3: 28–38 depending on regime |
| Bollinger Band | BB position < threshold | Tier 1: 0.35–0.55 / Tier 2/3: 0.25–0.45 depending on regime |
| SMA (200) | Price > SMA200 | Long-term uptrend intact |
| ATR % (14) | > 1.0% (Tier 2/3: also ≤ 5.0%) | Adequate volatility; cap prevents excessive risk |
| Volume surge | > avg50 (Tier 2/3 only) | Tier 1 exempt — big-cap selloffs always have elevated volume |
| IV Rank | ≥ 25 | Premium historically elevated (Pass 1) |
| IV/HV Ratio | ≥ 1.0 | Options priced above realized vol (Pass 2) |
| Earnings blackout | No earnings within 3 days | Wider 7-day buffer for covered calls |
| FOMC day | Warning only | 2 PM announcement priced in by 3:30 PM entry window |

### VIX-Adjusted Thresholds

Tier 1 uses wider RSI/BB thresholds (+5 RSI, +0.10 BB) than Tier 2/3. Tier pre-selection already acts as a quality filter; IV Rank / IV/HV provide the premium backstop. Volume check is omitted for Tier 1.

| Regime | VIX | RSI (Tier 1) | RSI (Tier 2/3) | BB (Tier 1) | BB (Tier 2/3) | Rationale |
|--------|-----|--------------|----------------|-------------|---------------|-----------|
| LOW | < 15 | 33 | 28 | < 0.35 | < 0.25 | Premium thin — still selective |
| NORMAL | 15–20 | 40 | 35 | < 0.50 | < 0.40 | Standard |
| ELEVATED | 20–30 | 43 | 38 | < 0.55 | < 0.45 | Fat premium — slightly relaxed |
| HIGH | > 30 | 35 | 30 | < 0.40 | < 0.30 | Tail risk — tightened |

**Per-ticker RSI overrides (take precedence over tier defaults):** COST = 45, GOOGL = 40

### Spread Construction

| Parameter | Value |
|-----------|-------|
| Width comparison | $10×1 vs $5×2 — whichever yields higher total premium |
| DTE window | 28–45 days (monthly expiry preferred) |
| Min credit — $10-wide | $1.30/share ($130/contract) |
| Min credit — $5-wide | $0.95/share ($95/contract) |
| STO limit price | Between mid and natural credit (mid + ask) / 2, rounded to $0.05 |
| Quad witching expiry | Allowed; target delta scaled ×0.75 (floor 0.08) |

### Risk Controls

| Rule | Value |
|------|-------|
| Max risk per spread | $1,000 (spread width × 100) |
| Max concurrent positions | 25 (dynamic — checked against live account) |
| Max entries per run | 10 (further capped to available slots) |
| Profit target | P&L ≥ 80% of credit collected |
| BTC debit cap | ≤ $0.60/share (profit-target close only) |
| DTE close threshold | ≤ 12 DTE (close regardless of P&L) |
| Emergency BTC | Place at mid → wait 90s → retry at 1.05× if unfilled |
| Monthly drawdown | Pause entries if MTD P&L ≤ −$2,000 |
| Portfolio exposure cap | MSFT ≤ $2,000 / TSLA ≤ $3,000 / SNDK ≤ $2,500 / ASML ≤ $3,000 — cross-account total max risk |
| Conservative cap | MSFT — max 2 contracts per account |
| Cluster guard | Warn when ≥ 5 tickers signal simultaneously |

---

## Covered Call Module

`covered_call_screener.py` scans Fidelity positions daily for covered call entry opportunities. **No orders are placed automatically** — alerts are sent to Telegram/Gmail for manual execution in Fidelity.

**Tickers monitored:** VOO, QQQM, EWY, GOOGL, SCHD, DRAM

**Entry conditions (opposite of put credit spreads):**
- Green day (stock up on the day)
- RSI 58–78 (mildly overbought, not extreme momentum)
- BB position > 0.65 (near upper band)
- Above SMA-200
- No earnings within 7 days

**Alert content:** current price, RSI, BB position, suggested expiry (21–35 DTE), and 3 OTM call strikes (3% / 5% / 7% OTM) with bid/ask and IV.

```bash
# Test alert
python3 covered_call_screener.py --test

# Run manually
python3 covered_call_screener.py
```

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
nano .env   # fill in TT_SECRET, TT_REFRESH, TG_BOT_TOKEN, GMAIL_PASSWORD
```

### 3. Tastytrade OAuth setup (one-time)

```bash
python3 broker/setup_auth.py
# Choose option 2 → paste Client Secret + Refresh Token → certification? y
```

Get credentials from [developer.tastytrade.com](https://developer.tastytrade.com) → Sandbox → Create Application.

### 4. Health check

```bash
python3 healthcheck.py
```

All required checks must pass before proceeding.

### 5. Test connection

```bash
python3 auto_trade.py monitor   # dry-run, no orders submitted
```

---

## Cloud Deployment (AWS EC2)

```bash
# On your EC2 instance (Ubuntu 24.04) — first time:
git clone https://github.com/hbchoi0917/trading.git
cd trading
bash deploy/setup.sh

# Fill in credentials:
nano options-screener/.env

# Verify:
cd options-screener && python3 healthcheck.py
```

**Updating an existing deployment:**

```bash
cd ~/trading
git pull origin main
chmod +x deploy/run_cc.sh   # only needed if run_cc.sh is new
bash deploy/setup.sh        # reinstalls crontab with new CC job
```

`setup.sh` handles: Python venv, dependencies, US/Eastern timezone, logrotate, and cron installation automatically.

### Cron Schedule (US/Eastern)

| Time ET | Time KST | Job |
|---------|----------|-----|
| 12:30 PM | 1:30 AM +1 | Monitor — profit target / DTE checks |
| 3:25 PM | 4:25 AM +1 | Covered call alert — Fidelity (manual execution) |
| 3:30 PM | 4:30 AM +1 | Entry — screener + order placement |
| 4:05 PM | 5:05 AM +1 | Monitor — post-close final scan |
| 4:30 PM | 5:30 AM +1 | Summary — daily email; weekly email on last trading day |

---

## Usage

```bash
# Run screener only (generates signals_YYYYMMDD.csv)
python3 options_premium_screener.py

# Place orders from today's signals (dry-run)
python3 auto_trade.py entry

# Place orders live
python3 auto_trade.py entry --live

# Monitor + auto-close positions (dry-run)
python3 auto_trade.py monitor

# Run full pipeline: entry + monitor
python3 auto_trade.py all

# Send daily/weekly summary email
python3 auto_trade.py summary

# Covered call alert (manual execution in Fidelity)
python3 covered_call_screener.py
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

# Notifications — Telegram (recommended: instant push to phone)
TG_BOT_TOKEN=               # from @BotFather
TG_CHAT_ID=                 # your chat ID

# Notifications — Gmail (fallback)
GMAIL_SENDER=
GMAIL_PASSWORD=             # App Password (not your login password)
GMAIL_RECEIVER=
```

---

## Testing

```bash
python3 -m pytest tests/ -v
# 48 tests, all passing — no live broker connection required
```

Tests cover: signal normalization, delta parsing, spread construction (mocked option chain), order execution (mocked account), monthly drawdown circuit breaker, build_best_spread width comparison.

---

## Go-Live Checklist

- [ ] `python3 healthcheck.py` — all required checks pass
- [ ] `python3 auto_trade.py monitor` — connects, scans positions without error
- [ ] `TT_DRY_RUN=false`, `TT_PAPER_TRADING=true` — sandbox orders fill correctly
- [ ] Telegram or Gmail alert received for entry / close events
- [ ] `python3 covered_call_screener.py --test` — CC alert notification received
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
