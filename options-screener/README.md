# Options Screener — Demo

A runnable put credit spread screener that scans a watchlist daily for entry candidates.

> **Demo / educational baseline.** Customization is required before live trading.
> Full automation (order execution, position monitoring, cloud deployment) requires
> additional broker API integration. See notes in [`screener.py`](screener.py).

---

## Filters

All conditions must pass for a signal to be generated:

| Filter | Indicator |
|--------|-----------|
| Trend | Price > SMA-200 |
| Momentum | RSI (14) — threshold adjusts with VIX regime |
| Mean reversion | Bollinger Band position (lower band proximity) |
| Volatility | ATR% > minimum; volume > 50-day average |
| Premium quality | IV Rank ≥ 25; IV/HV ratio ≥ 1.0 |
| Earnings | Skip 3 days before announcement |
| FOMC | Warning on rate-decision days |

**VIX regime adjustment** — RSI and BB thresholds shift across four regimes
(LOW / NORMAL / ELEVATED / HIGH) so entry criteria stay calibrated to market conditions.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python screener.py
```

Outputs `signals_YYYYMMDD.csv` + console table. Each row is a ticker that passed all filters, with RSI, BB position, IV Rank, IV/HV ratio, and a suggested expiry date.

---

## Customization

| Parameter | Location | Notes |
|-----------|----------|-------|
| Watchlist | `WATCHLIST` | Replace example tickers with names you've researched |
| RSI / BB thresholds | `VIX_REGIMES` | Back-test against your own trade history before changing |
| Delta target | `DELTA_TARGET` | Lower = further OTM = higher probability, lower premium |
| DTE window | `DTE_MIN / DTE_MAX` | 21–45 days is typical for monthly spreads |
| IV Rank floor | `IV_RANK_MIN` | Raise to be more selective on premium quality |

---

## For Full Automation

To run this as a production system:

1. **Broker API** — add order placement via Tastytrade, IBKR, or similar
2. **Cloud server** — deploy to AWS EC2 (Ubuntu); schedule via cron in US/Eastern timezone
3. **Notifications** — add Telegram or email alerts on entry / close events
4. **Position monitor** — add logic to auto-close at profit target or DTE threshold

---

## Dependencies

- `yfinance` — market data and option chains
- `pandas` — data manipulation
- `pandas-ta` — optional TA library (falls back to built-in pandas if unavailable)
