# Options Trading Automation

**End-to-end automated pipeline for selling vertical put credit spreads — from screening to execution to position management.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Tastytrade](https://img.shields.io/badge/Tastytrade-SDK%20v12-purple.svg)](https://tastytrade.com)
[![License](https://img.shields.io/badge/license-Personal%20Use-red.svg)](LICENSE)

---

## Overview

This pipeline automates the full lifecycle of a put credit spread strategy:

1. **Screen** — identify high-probability entry candidates using 10+ technical and volatility indicators
2. **Execute** — place vertical spread orders on Tastytrade 1 hour before market close
3. **Monitor** — auto-close positions at profit target or DTE threshold
4. **Alert** — send Telegram / Gmail notifications on every entry, close, and error event
5. **Covered Calls** — separate alert module for manual CC management on Fidelity positions

Designed to run unattended on a cloud server (AWS EC2).

---

## Architecture

```
options-screener/
├── options_premium_screener.py   # Screening engine → signals CSV
├── covered_call_screener.py      # Covered call alert module
├── auto_trade.py                 # Daily pipeline orchestrator
├── notifications.py              # Telegram / Gmail alerts
├── position_tracker.py           # CSV-based position ledger
├── healthcheck.py                # Pre-flight environment check
│
├── broker/
│   ├── client.py                 # Tastytrade session management
│   ├── executor.py               # Order placement + position monitor
│   ├── spread_builder.py         # Option chain → spread selection
│   └── setup_auth.py             # One-time OAuth credential setup
│
└── tests/                        # Unit tests, no live broker required

deploy/
├── setup.sh                      # One-command Ubuntu server setup
├── run_entry.sh / run_monitor.sh # Cron wrappers
├── run_cc.sh / run_summary.sh
└── crontab.template
```

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

## Cron Schedule (US/Eastern)

| Time ET | Job |
|---------|-----|
| 12:30 PM | Monitor — profit target / DTE checks |
| 3:25 PM | Covered call alert (manual execution in Fidelity) |
| 3:30 PM | Entry — screener + order placement |
| 4:05 PM | Monitor — post-close final scan |
| 4:30 PM | Summary — daily email; weekly email on last trading day |

---

## Tech Stack

- **[tastytrade](https://pypi.org/project/tastytrade/)** ≥ 12.0 — broker SDK
- **[pandas](https://pandas.pydata.org/)** — data manipulation
- **[yfinance](https://pypi.org/project/yfinance/)** — market data and option chains
- **[python-dotenv](https://pypi.org/project/python-dotenv/)** — credential management

---

## License

Personal use only — not for commercial use or redistribution. See [LICENSE](LICENSE).

---

## Disclaimer

For personal use and educational purposes only. Not financial advice. Options trading involves substantial risk of loss.
