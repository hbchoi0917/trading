# Options Screener — Demo

**English** | [한국어](README.ko.md)

A runnable put credit spread screener that scans a watchlist daily for entry candidates.

> **Demo / educational baseline.** This file shows the structure of the filters; the
> production system's rules and thresholds differ (see Methodology Notes below for how
> the live entry conditions actually work). Customization is required before live trading.
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
| Premium quality | IV Rank ≥ 25 (approximated from realized volatility, see below); IV/HV ratio ≥ 1.0 |
| Earnings | Blackout window before and after announcements (see below) |
| FOMC | Warning on rate-decision days |

**Earnings blackout** — no new entries in a configurable window before *and after* an
earnings announcement, and the chosen expiry is kept clear of the earnings date on both sides.
The earnings lookup fails open but logs why it failed, so a broken data source can't silently
turn the protection off.

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
- **IV dual-pass filter, and its limits** — Pass 1 compares today's implied
  volatility with a 52-week range; Pass 2 compares it with recent realized
  volatility (IV/HV ratio). Two limits are stated plainly. First, the free
  data source has no history of implied volatility, so the 52-week range is
  that of *realized* volatility: the "rank" is an approximation and reads
  higher than a true IV Rank. Second, in practice the IV/HV ratio does most
  of the filtering; after a selloff the rank passes almost everywhere. If IV
  data is unavailable the filter fails open for single names and fails
  closed for the index product, whose only quality gate it is. Replacing the
  proxy with the broker's own IV metrics is planned.
- **Most traded names use a simple entry condition** — the full
  directional-timing pattern (an oversold dip in an uptrend) is applied to
  names still under observation. Names cleared for trading enter on a
  volatility condition plus the premium, earnings, overheat and risk-cap
  gates. This was a deliberate trade of selectivity for entry frequency, and
  it means the live book behaves close to an always-on premium-selling
  position in the tradable list. Whether to restore a trend or selloff gate
  is an open question under review, not a settled design.

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
- **Structural fit, not just track record** — a candidate's typical daily
  price range is compared with the spread width the risk caps allow. When one
  day's move is many times the width, a breached spread runs to near-maximum
  loss before a stop can act, so a strong historical record alone doesn't
  earn a slot; option open interest is checked as well. Names are demoted
  as readily as they are promoted when the automated record doesn't carry
  over from the manual one
- **Thresholds are calibrated from observed near-misses, not guessed** —
  screening and credit floors are revisited using logged rejection data
  across multiple sessions. A threshold is loosened only when rejected
  candidates cluster just under it; when rejections are spread far below it
  (e.g. a volume filter catching genuinely low-conviction moves), the filter
  is judged to be working as designed and left alone. Thresholds that
  interact (entry delta and the credit floor) are checked together against
  live quotes before either moves, and off-hours or non-monotonic quote
  snapshots are discarded rather than used.

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
- **Broker exports are cross-checked, not trusted blindly** — quarterly
  transaction exports can silently omit rows, and legitimate split fills
  look like duplicates. Monthly totals are reconciled against separate
  monthly exports, and the opening and closing legs are matched against
  each other to confirm that nothing was dropped or double-counted.
- **Results are broken down, not just summed** — a read-only report splits
  realized results by close reason, underlying, entry-credit bucket and
  month, measures capital use against the risk budget, and tracks fill
  slippage by close type. It is what showed the losing bucket coinciding
  exactly with thin-credit entries, and the floor turning the book into a
  single name.
- **Cohorts across rule changes** — entries opened after a rule change are
  reported separately from earlier ones, utilization is read against the cap
  in force at the time, and a change is judged only after enough closes
  have accumulated.

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

Outputs `signals_YYYYMMDD.csv` + console table. Each row is a ticker that passed all filters, with RSI, BB position, IV Rank / percentile, IV/HV ratio, a suggested expiry (monthly preferred, kept clear of earnings) and the next earnings date. Rows are sorted by `Signal_Strength`, a 0–100 score used only to rank signals that already passed, and every row carries `Cluster_Risk`, which is set when too many tickers fire on the same day. The log explains each rejected ticker (`no signal — …`) and any earnings lookup that failed.

---

## Customization

| Parameter | Location | Notes |
|-----------|----------|-------|
| Watchlist | `WATCHLIST` | Replace example tickers with names you've researched |
| RSI / BB thresholds | `VIX_REGIMES` | Back-test against your own trade history before changing |
| Delta target | `DELTA_TARGET` | Lower = further OTM = higher probability, lower premium |
| DTE window | `DTE_MIN / DTE_MAX` | 21–45 days is typical for monthly spreads |
| IV Rank floor | `IV_RANK_MIN` | Raise to be more selective on premium quality |
| Earnings blackout | `EARNINGS_BUFFER_BEFORE / EARNINGS_BUFFER_AFTER` | Days before / after an announcement with no new entries |
| Expiry vs. earnings | `EARNINGS_EXPIRY_BUFFER_DAYS` | Expiries this close to earnings (either side) are never chosen |
| Cluster warning | `CLUSTER_WARN_THRESHOLD` | Same-day signal count that triggers the correlated-signals warning |

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
