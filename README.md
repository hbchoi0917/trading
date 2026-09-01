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

> **Project status:** the strategy ran for 18 months as a manual, discretionary
> book and is now automated. Automation is rolling out to real money through a
> deliberate **minimal-size gate** (single lowest-risk name, one contract, a hard
> cap on concurrent positions and on risk concentrated in any single name) that
> validates one full entry → hold → close lifecycle before any scaling — a
> milestone the system has now reached, with multiple unattended cycles
> completed. The tradeable name set and risk budget have since scaled up
> gradually, each expansion checked against real trading history rather than
> assumed. See [Live-Trading Safety](#broker-environments--live-trading-safety).

> **Note:** The screener in this repo is a **demo / educational baseline**. The full production system (broker integration, order execution, position monitoring, cron-scheduled deployment) runs in a private repository. See the [customization notes](options-screener/screener.py) at the top of `screener.py` for what's needed to build a complete system.

---

## Screening Logic

Tickers are organized into tiers by liquidity and volatility profile. All signals must pass momentum, mean-reversion, volatility, and earnings blackout filters. RSI and Bollinger Band thresholds shift automatically across four VIX regimes (LOW / NORMAL / ELEVATED / HIGH).

Three entry paths reflect different levels of established conviction, not a
single uniform bar:

- **Equities / ETFs (standard)** — the full stack of momentum (hourly RSI),
  mean-reversion (Bollinger position), trend, volatility, earnings blackout,
  and the IV dual-pass premium-quality gate.
- **Established, high-conviction names** — a smaller set of names with a
  proven track record against real trading history trade the full
  directional-timing pattern for a simplified, volatility-and-premium-quality
  bar instead: is there enough movement to be worth selling into, and is the
  premium not historically cheap right now. This deliberately trades some
  directional selectivity for entry frequency, on names where that trade-off
  has already been checked against real results — newer candidates still have
  to earn it.
- **Broad-market indices** — European-style, cash-settled options carry no early
  assignment risk, so RSI timing is dropped as an entry gate; entries require
  only long-term trend confirmation plus the IV dual-pass filter. For a premium
  seller on an index, IV Rank (not RSI position) is what determines whether there
  is edge to capture.

### Methodology Notes

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
  blocks a valid entry.
- **Data-driven ticker selection, re-weighted by realized return per
  trade** — the tradeable ticker set is periodically re-evaluated against
  the owner's own multi-quarter trading history, ranked by **realized P&L
  per trade** rather than by trade frequency or total P&L alone — both of
  which reward high-volume, thin-margin names over fewer, more efficient
  ones. Names confirmed to be structurally weak performers (not just
  temporarily out of favor) stay excluded even after conditions loosen
  elsewhere; a single bad month is distinguished from a sustained pattern
  before either confirming an exclusion or admitting a new name.
- **Risk caps scale with the tradeable roster, not just with capital** — as
  more independently-vetted, largely uncorrelated names are added to the
  tradeable set, the total open-risk budget can reasonably increase (a
  book spread across more uncorrelated names carries less concentration
  risk per dollar), while the *share* of that budget any single name can
  claim is reduced in step, so growing the roster doesn't just grow how
  much one name can absorb.

---

## Order Sizing & Execution

Sizing and order placement are as deterministic as screening:

- **Adaptive per-name sizing** — the highest-conviction core name uses a
  closer-to-the-money delta while its trend is intact, and switches to a
  two-layer strike ladder (collecting premium at two different strikes in the
  same expiry) when it pulls back into a downtrend, where elevated IV is exactly
  the premium-selling edge. Established, high-conviction names that trade
  outside a confirmed trend get the same defensive-ladder treatment (a lower,
  more out-of-the-money delta range) rather than their normal calm-market
  target. Other names use conservative, further-OTM targets.
- **Per-instrument-class credit/width floor** — spreads that don't collect
  enough premium relative to their width are skipped as low return-on-risk.
  The minimum is calibrated per instrument class rather than being a single
  fixed number or a full exemption for any class: broad-market index products
  get a lower floor than single-name equities (reflecting their structurally
  thinner but still worthwhile premium and lower assignment risk), but even
  they have a floor now, benchmarked against what comparably low-risk names
  actually collect — a full exemption turned out to mean genuinely unlimited,
  not just "looser."
- **Per-ticker daily entry limits** — most names are capped at a single entry
  per day; the laddering core name is allowed a small, explicit multiple. This
  stops overlapping scheduled runs from stacking unintended duplicates.
- **Peak-liquidity timing** — entries are placed inside the mid-session liquidity
  window when bid-ask spreads are tightest; market-open and near-close windows
  are avoided (inflated IV, widest spreads).
- **Market-calendar aware** — entry and monitor runs skip weekends and US market
  holidays automatically from an exchange calendar (no hardcoded date lists), and
  intraday retry loops respect real early-close (half) days.

---

## Risk Management

Every order passes through layered, deterministic safeguards before and after entry:

- **Pre-trade buying power check** — required margin is validated against
  available option buying power before any order is submitted; fails open
  to the broker's own enforcement on API errors
- **Per-spread risk cap** — maximum defined risk per position, with
  ticker-class-specific overrides (index vs. equity vs. high-beta names)
- **Layered open-risk caps** — beyond the per-position cap, a permanent cap on
  total risk across all open positions AND a separate cap on risk concentrated
  in any single underlying, so no one name can dominate the book. Both apply
  at all times, independent of trading phase, and both scale as the tradeable
  roster grows (see Methodology Notes above).
- **Per-ticker contract limits** — high-volatility names are capped at
  reduced contract counts regardless of signal strength
- **Monthly drawdown circuit breaker** — all new entries pause automatically
  if month-to-date P&L breaches a configured loss limit
- **Profit target close** — positions are closed automatically once a
  configured percentage of the collected credit has decayed
- **Stop loss** — positions are closed when the loss reaches a multiple of
  the credit collected (the standard premium-selling convention); the stop
  fires immediately regardless of moneyness or the holding window — a
  volatility spike can still hurt an otherwise-safe position, so this
  safeguard is never suppressed
- **Moneyness-aware, DTE-based forced close** — a position approaching
  expiration is force-closed only if it's still meaningfully at risk (in or
  very near the money); one that has moved safely out of the money is left
  to ride out its remaining decay instead of being closed at an unnecessary
  loss. A small buffer around the strike (a fraction of a percent of the
  underlying's price) keeps a marginal, easily-reversible breach from being
  treated the same as a real one.
- **Minimum holding window** — profit-taking is gated by a minimum holding
  period to prevent same-day churn; risk-management closes (stop loss, DTE)
  are exempt and always fire
- **Fill confirmation** — close orders are verified against live order status
  before being reported as filled; unfilled orders are re-priced toward the
  ask (price chasing), then cancelled and retried on the next run
- **Execution slippage tracking** — every close records what the live market
  implied it would cost alongside what it actually cost, so execution quality
  is measured from real fills rather than assumed
- **Position ledger** — every entry and close is recorded to a CSV ledger,
  driving month-to-date statistics in summary notifications
- **Crash-safe ledger writes** — each fill is booked immediately and
  idempotently (keyed by order id) under a file lock with atomic replace, so a
  crash mid-run never corrupts or double-books the ledger
- **Ledger ↔ broker reconciliation** — on every monitor run, open ledger rows
  are cross-checked against the broker's actual positions per account (matched by
  strike / expiry / type). Any divergence — the ledger shows open but the broker
  holds none, or the broker holds a position the ledger never recorded — fires an
  alert, and downstream P&L and circuit-breaker figures are flagged unreliable
  until a human resolves it. This reconciliation is contract-symbol aware: some
  index option products settle under more than one distinct contract root
  depending on the specific expiration chosen (e.g. a standard monthly vs. a
  weekly variant), and the check derives the correct root from the expiration
  itself rather than assuming a single fixed one — an earlier fixed assumption
  produced false divergence alerts and blocked pricing/closing for the
  affected expirations.
- **Entry netting guard** — a new position is refused if opening it would
  cancel out an existing position at the broker (brokers net identical option
  contracts together, so an overlapping book can otherwise create a position
  neither the ledger nor the broker can price or close). Shared strikes are
  still allowed when they don't net to zero — a narrow, deliberate exception,
  not a general restriction on overlap
- **Unpriceable-position escalation** — if a position is ever missing a leg at
  the broker, it is never priced from the remaining leg alone (doing so can
  read a small position as a catastrophic loss); the system alerts and skips
  instead of guessing
- **Minimal-size live rollout gate** — an optional phase restricts real-money
  entries to a single lowest-risk name, caps concurrent open positions, and
  forces quantity to one contract, validating a full lifecycle at trivial size
  before any scaling (fail-safe: on a broker API error the open-position count
  reports "at cap" and blocks rather than over-opens)
- **Double-fill prevention** — entry and close retry loops confirm a cancel
  actually succeeded before placing a replacement; a failed cancel usually means
  the order just filled, and blindly replacing it would double the position or
  leave a naked leg
- **Cross-run duplicate-entry guard** — a ticker that already has a working
  order at the broker is skipped on the next scheduled run (with a small,
  explicit allowance for intentional laddering); the broker stays the authority
- **Data-quality guards** — option quotes whose delta drifts too far from target
  (an incomplete streamer snapshot) or that are one-sided (missing bid or ask)
  are rejected before an order is ever built
- **Assignment blind-spot alerts** — the monitor scans all positions, not just
  options, and alerts on equity positions from short-put assignment and on any
  option position the ledger has no record of
- **Connection / API retry** — transient broker and network failures are retried
  so a momentary blip doesn't abort an entire run
- **Fail-safe alerting** — shell wrappers trap pipeline errors into email alerts,
  and an optional dead-man switch pings an external monitor after each successful
  summary, so even a fully-down server surfaces (nothing left to email you)
- **Earnings blackout** — entries are skipped within a configurable window
  around earnings announcements
- **Concentration guard** — simultaneous signals across correlated tickers
  are flagged as a single macro bet, not independent trades

---

## Position Tracking

The system keeps its own ledger of every position it opens, and treats that
ledger — not the broker's position list — as the source of truth for what a
spread *is*.

This is not redundancy. Brokers report **net quantity per strike**, so once
two spreads share a strike (a deliberate laddering pattern), which long
belongs to which short is no longer recoverable from broker data. Anything
derived from that ambiguity — position counts, close orders, P&L — inherits
it. The ledger records each spread as it was actually opened, and the broker
is consulted only for what it alone knows: live marks, assigned shares, and
positions the ledger has no record of.

- **Recorded at the fill, not the quote** — the entry retry loop steps its
  limit down over the session, so the credit finally collected is usually
  below the price quoted when the order was written. The fill is what gets
  booked.
- **Net of costs** — realized P&L deducts commissions and fees. The same
  number feeds the drawdown circuit breaker, and a gross figure would make
  the account look safer than it is, allowing more risk to run than intended.
- **Reconciled every run** — ledger and broker are compared as signed net
  quantity per strike across both legs, and any divergence raises an alert
  that explicitly marks all downstream P&L as unreliable until a human
  resolves it.
- **Fails closed** — a ledger row that cannot be parsed into a well-formed
  spread is skipped rather than guessed at, and an empty ledger closes
  nothing. A wrong leg here would become a wrong live order.
- **Performance compared per day of capital held** — closes are grouped by
  why they closed (profit target vs. time-based exit) and compared on
  realized return per dollar of risk per day held, so a faster, smaller win
  can be correctly weighed against a slower, larger one instead of comparing
  raw dollar totals.

---

## Broker Environments & Live-Trading Safety

Going from backtest to real money is where quiet money-losing bugs live, so the
path to live orders is gated behind explicit, defaulted-safe flags:

- **Two safety flags, both default on** — a dry-run flag (log orders without
  submitting) and a paper-routing flag. Submitting a real order requires
  explicitly disabling dry-run *and* selecting the live endpoint; neither happens
  by accident.
- **Sandbox ≠ paper account** — a broker's certification / API-integration
  sandbox is **not** a persistent paper-trading account. It can reset
  periodically, silently dropping positions, and its fills are deterministic by
  rule rather than realistic — so it can validate wiring but never a multi-week
  strategy. The healthcheck now **hard-fails** if the system is configured to
  point at the sandbox while believing it is paper trading, and the client logs a
  loud warning on every connect in that state, so the misconfiguration cannot go
  unnoticed. *(This guard exists because the mistake was made the hard way.)*
- **Minimal-size rollout gate** — see [Risk Management](#risk-management); the
  first real-money phase is deliberately too small to matter.

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

All jobs run Mon–Fri in US/Eastern; weekends and US market holidays are skipped
automatically via an exchange calendar.

| Time ET | Job |
|---------|-----|
| Pre-market | Healthcheck — credentials / dependencies / network; emails on failure |
| Late morning | Monitor — profit target / DTE checks after the open stabilizes |
| Midday | Entry — screener + order placement (peak liquidity window) |
| Early afternoon | Entry — afternoon re-screen within the liquidity window |
| Mid-afternoon | Monitor — catch profit targets hit during the trading day |
| Late afternoon | Covered call alert (manual execution in Fidelity) |
| After close | Summary — daily email; weekly email on the last trading day; optional dead-man ping |

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
- **[pandas-market-calendars](https://pypi.org/project/pandas-market-calendars/)** — NYSE holiday / early-close calendar (production)
- **[python-dotenv](https://pypi.org/project/python-dotenv/)** — credential management

Analytics side: **dbt** · **DuckDB** · **Streamlit** · **Plotly** (see [`options-dbt/`](options-dbt/)).

---

## Recent Updates

High-level themes from recent iterations (no thresholds, per-ticker parameters,
or account figures — those stay private):

- **Roster-scaled diversification** — the tradeable name set was widened
  substantially, with each addition checked against the owner's own
  multi-quarter trading history and ranked by realized return per trade
  rather than by how often a name traded — a metric that otherwise favors
  high-volume, thin-margin names. Names confirmed as structurally weak
  performers, not just temporarily out of favor, stay excluded. Open-risk
  caps were raised in step, with the per-name share of the total budget
  reduced rather than held flat, so a larger roster translates into real
  diversification rather than just a bigger ceiling for the same
  concentration.
- **Tiered screening by conviction** — a subset of well-established names
  now use a simplified volatility-and-premium-quality entry bar instead of
  the full directional-timing pattern, trading some selectivity for entry
  frequency on names where that trade-off has already been validated —
  newer candidates still earn the full pattern first.
- **Moneyness-aware risk-management closes** — a time-based forced close
  now accounts for how far a position actually is from its strike, not just
  calendar proximity to expiration; a marginal, easily-reversible breach is
  no longer treated the same as a position genuinely at risk. Loss-limit
  closes remain unconditional regardless of moneyness.
- **Per-instrument-class credit floor** — the minimum acceptable premium
  relative to defined risk is no longer a full exemption for any instrument
  class; index/cash-settled products get their own floor, calibrated
  against what comparably low-risk names actually collect.
- **Root-symbol reconciliation fix** — index option products that settle
  under more than one contract root depending on the expiration chosen
  (e.g. standard monthly vs. weekly) previously used a single hardcoded
  root when reconciling the ledger against the broker, producing false
  divergence alerts for the affected expirations; the correct root is now
  derived from the expiration date itself.
- **Ledger-driven position identity** — the broker only reports net quantity
  per strike, which cannot distinguish overlapping positions that happen to
  share a strike. The system's own ledger, not broker state, is now the
  source of truth for what was actually opened; the broker is consulted only
  for live pricing and for anything the ledger has no record of. This
  resolved several defects that traced back to the same root cause —
  mis-sized risk caps, false alerts on a healthy book, and P&L distorted by
  an overlapping position.
- **Unattended lifecycle validation** — the minimal-size gate exists to
  validate a full entry → hold → close cycle with no manual intervention
  before any scaling; the system has now completed that cycle multiple times
  unattended.
- **Layered + concentration risk caps made permanent** — the open-risk caps
  no longer depend on which rollout phase is active; a dedicated cap now also
  bounds risk concentrated in any single underlying, closing a gap where the
  total cap alone did not prevent one name from dominating the book.
- **Entry-time safety guards** — a new position is refused if it would net an
  existing position's leg to zero at the broker (which would otherwise leave
  a position neither system can price or close), and a position missing a leg
  at the broker is escalated rather than priced off the leg that remains.
- **Execution-quality measurement** — every close now records what the market
  implied it would cost alongside what it actually cost, so execution
  slippage is measured from real fills instead of assumed; performance is
  compared per day of capital held, not just per trade, so exit-timing
  trade-offs can be judged on real numbers.
- **Live-trading transition** — moved from wiring/validation toward real money
  behind a minimal-size rollout gate, with explicit defaulted-safe dry-run and
  paper flags and a hard healthcheck failure when misconfigured against a broker
  sandbox that isn't a real paper account.
- **Ledger ↔ broker reconciliation** — every monitor run reconciles the local
  ledger against actual broker positions and alerts on any drift, flagging P&L
  and circuit-breaker numbers as unreliable until resolved.
- **Order-safety hardening** — double-fill prevention (cancel-confirmed reprice),
  cross-run duplicate-entry guard, crash-safe idempotent ledger writes, option
  data-quality guards, connection/API retries, and assignment blind-spot alerts.
- **Execution refinements** — credit/width ratio filter, adaptive per-name delta
  and strike laddering, per-ticker daily entry limits, and market-calendar-aware
  scheduling (weekends, holidays, and early-close days handled automatically).
- **Screening methodology** — hourly RSI with real-time intraday price patching,
  VIX-regime-adjusted thresholds, IV dual-pass premium gate, and index-specific
  screening (trend + IV only, no RSI gate).

---

## License

Personal use only — not for commercial use or redistribution. See [LICENSE](LICENSE).

---

## Disclaimer

For personal use and educational purposes only. Not financial advice. Options trading involves substantial risk of loss.
