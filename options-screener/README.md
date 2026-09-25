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
| Earnings | Skipped within a configurable window around announcements |
| FOMC | Warning on rate-decision days |

**VIX regime adjustment** — RSI and BB thresholds shift across four regimes
(LOW / NORMAL / ELEVATED / HIGH) so entry criteria stay calibrated to market conditions.

---

## Methodology Notes

### Signals

- **Hourly RSI, not daily** — on large down days, daily RSI can read neutral
  (~50) while hourly RSI correctly shows oversold. Daily RSI reflects
  yesterday's close; hourly reflects what the market is doing right now. For
  intraday entry decisions, hourly is the right timeframe (falls back to
  daily if intraday data is unavailable).
- **Real-time intraday patch** — today's OHLCV bar is updated with live
  intraday data before screening, so signals reflect current prices rather
  than yesterday's close.
- **Index screening differs from equities** — for broad-market indices with
  European-style, cash-settled options (no early assignment risk), RSI is
  not used as an entry gate. RSI is a directional-trader tool; for premium
  sellers, IV Rank determines whether there is edge to capture. Index
  entries require only trend confirmation (long-term moving average) plus
  the IV dual-pass filter.
- **IV dual-pass filter** — Pass 1 asks "is IV elevated vs. its own 52-week
  history?" (IV Rank); Pass 2 asks "is the market paying above recent
  realized volatility right now?" (IV/HV ratio). Both must pass; if IV data
  is unavailable, the filter fails open so a data outage never silently
  blocks a valid entry — and a product whose IV data fails persistently is
  paused rather than left trading without its quality gate.
- **Screening conditions are tiered by conviction, not applied uniformly** —
  a small set of names with an established track record use a simplified,
  volatility-and-premium-quality-based entry condition instead of the full
  directional-timing pattern (oversold-dip-in-an-uptrend) applied to newer or
  less-proven names. This trades some directional selectivity for entry
  frequency where the trade-off has already been validated against real
  results, while newer candidates still have to earn that treatment.

### Selection & calibration

- **Data-driven ticker selection, re-weighted by realized return per trade**
  — the tradeable ticker set is periodically re-evaluated against the
  owner's own multi-quarter trading history, ranked by **realized P&L per
  trade** rather than by trade frequency or total P&L alone — both of which
  reward high-volume, thin-margin names over fewer, more efficient ones.
  Names confirmed to be structurally weak performers stay excluded even after
  conditions loosen elsewhere; a single bad month is distinguished from a
  sustained pattern before either confirming an exclusion or admitting a
  new name.
- **Thresholds are calibrated from observed near-misses, not guessed** —
  screening and credit floors are revisited using logged rejection data
  across multiple sessions. A threshold is loosened only when rejected
  candidates cluster just under it; when rejections are spread far below it
  (e.g. a volume filter catching genuinely low-conviction moves), the filter
  is judged to be working as designed and left alone.

### Measuring results

- **Ledger-driven position identity** — the broker only reports net
  quantity per strike, which cannot distinguish overlapping positions that
  happen to share a strike. The system's own ledger, not broker state, is
  the source of truth for what was actually opened; the broker is consulted
  only for live pricing and for anything the ledger has no record of.
- **Realized P&L, not cash flow, is the performance metric** — summary
  notifications report realized P&L for the current month alongside the
  prior month and quarter-to-date. Net cash flow (summing every premium
  received and paid in a period) is deliberately not used: opening a credit
  spread brings cash in before its outcome is known, so a month with many
  new positions can look profitable while carrying unresolved losses. On
  the owner's own history, monthly cash flow and realized P&L diverged
  sharply — in one month even pointing in opposite directions.
- **Return measured per day of capital held, not just per trade** — closes
  are grouped by why they closed (profit target vs. time-based exit vs.
  stop) and compared on realized return per dollar of risk per day held, so
  a faster, smaller win can be correctly weighed against a slower, larger
  one instead of comparing raw dollar totals.

---

## Files

| File | Description |
|------|-------------|
| `screener.py` | Main screening engine — applies all filters, outputs signals CSV |
| `.env.example` | Template for credentials (Tastytrade, Telegram, Gmail) |
| `healthcheck.py` | Pre-flight check for packages, env vars, network, timezone |
| `requirements.txt` | Python dependencies |

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your credentials
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
5. **Position ledger + reconciliation** — record every fill to a durable ledger
   and reconcile it against the broker's actual positions each run, alerting on
   any drift before it corrupts P&L or risk limits
6. **Live-trading safety** — keep dry-run and paper flags defaulted on, and gate
   the first real-money phase behind a minimal-size rollout (one low-risk name,
   one contract, a hard cap on concurrent positions)

See [`deploy/`](../deploy/) for ready-to-use server setup scripts and cron templates.
The complete production system (execution, monitoring, reconciliation, live-safety
gating) runs in a private repository — this repo is the screening baseline.

---

## Dependencies

- `yfinance` — market data and option chains
- `pandas` — data manipulation
- `pandas-ta` — optional TA library (falls back to built-in pandas if unavailable)
