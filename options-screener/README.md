# Options Premium Screener

**Automated multi-factor technical analysis pipeline for curated options premium selling candidates**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-2.0+-green.svg)](https://pandas.pydata.org/)
[![yfinance](https://img.shields.io/badge/yfinance-latest-orange.svg)](https://github.com/ranaroussi/yfinance)

## 🎯 Project Overview

Production-ready Python screener analyzing a **curated watchlist of high-conviction tickers** using 10+ technical and volatility indicators to identify high-probability entry candidates for **Vertical Put Spread (Credit Put Spread)** options strategies.

The screener is designed for **premium sellers** — not directional traders. Every filter and threshold is chosen to maximize premium quality (high IV, fat spreads) while minimizing tail risk.

### Key Features
- **IV Rank & IV/HV Ratio**: Dual-pass premium quality filter — only enter when IV is historically elevated *and* options are priced above recent realized volatility
- **Dynamic VIX-adjusted thresholds**: RSI and Bollinger Band entry bars shift automatically based on VIX regime (LOW / NORMAL / ELEVATED / HIGH)
- **Concentration risk guard**: Flags when too many tickers trigger simultaneously — prevents disguised macro bets
- **Earnings blackout**: Automatically skips tickers within ±5/+1 days of earnings
- **Tiered watchlist**: TIER1 (SPX + core names, wider delta) vs TIER2 (watchlist, tighter delta + ATR volatility cap)
- **Smart signal scoring**: 0–100 signal strength algorithm across RSI, BB position, volume surge, ATR
- **Production-grade logging**: Timestamped CSV outputs + rotating log files per run

## 📊 Technical Indicators

| Indicator | Purpose | Threshold / Logic |
|-----------|---------|-------------------|
| **RSI (14)** | Oversold detection | VIX-adjusted: 28–38 (regime-dependent) |
| **Bollinger Bands (20)** | Price extremes | BB position < VIX-adjusted threshold (0.25–0.45) |
| **SMA (200)** | Long-term trend filter | Price > SMA200 required |
| **ATR % (14)** | Volatility measurement | > 1.0% required; Tier 2 cap at 5.0% |
| **Volume Surge** | Liquidity confirmation | > 1.2× 50-day avg volume |
| **IV Rank (52-week)** | Premium quality — Pass 1 | ≥ 25: IV historically elevated |
| **IV/HV Ratio** | Premium quality — Pass 2 | ≥ 1.0: options priced above realized vol |
| **HV (30-day realized)** | Baseline volatility context | Annualized from 30-day log returns |
| **VIX Regime** | Macro environment classifier | LOW / NORMAL / ELEVATED / HIGH |
| **MACD Histogram** | Momentum confirmation | Included in output; not a hard filter |
| **Support Level** | Risk proximity assessment | 20-day low; distance % included in output |

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/hbchoi0917/trading.git
cd trading/options-screener
pip install -r requirements.txt
```

### Run Screener
```bash
python options_premium_screener.py
```

## 📈 Sample Output

```
====================================================================
OPTIONS PREMIUM SCREENER — WINNING STOCKS PRIORITY MODE
====================================================================
[VIX] Current: 18.4 — Regime: NORMAL
[VIX-PARAMS] Regime=NORMAL | RSI threshold=35 | BB threshold=0.40 | SPX RSI threshold=30
VIX Regime                     : NORMAL
RSI Threshold (general, adj.)  : 35 (base: 35)
RSI Threshold (SPX, adj.)      : 30 (base: 30) + gap-down >= -1.0%
BB Threshold (adj.)            : 0.40 (base: 0.40)
IV Rank minimum (Pass 1)       : 25 (below = premium historically cheap)
IV/HV minimum (Pass 2)         : 1.0 (below = options not priced above realized vol)
IV filter policy               : fail-open (None = proceed without filter)
Cluster warn threshold         : 5 simultaneous signals
====================================================================

✓ [TIER1_CORE] NVDA: RSI 31.2 (thr=35) | BB 0.22 (thr=0.40) | ATR% 3.10 |
  IV Rank 58.3 | IV/HV 1.24 | Signal: 78/100 | Expiry: 2026-05-15 (DTE 38)

====================================================================
⚠️  CONCENTRATION RISK: 6 tickers triggered simultaneously ...
    These likely share macro exposure — treat as correlated, not independent trades.
====================================================================
Total signals : 6  (SPX: 0, T1: 2, T2: 4)
```

**Output columns in `signals_YYYYMMDD.csv`**:
| Column | Description |
|--------|-------------|
| `Tier` | TIER1_CORE or TIER2_WATCH |
| `Signal_Strength` | 0–100 composite score |
| `RSI` / `RSI_Threshold_Used` | RSI value + regime-adjusted threshold |
| `BB_Position` / `BB_Threshold_Used` | Bollinger Band position + threshold |
| `ATR_%` | ATR as % of price |
| `VIX` / `VIX_Regime` | VIX level + regime label |
| `IV_Rank` / `IV_Pct` | 52-week IV rank and percentile |
| `HV_30` / `IV_HV_Ratio` | 30-day realized vol + IV/HV ratio |
| `Expiry_Date` / `Expiry_DTE` / `Is_Monthly` | Target expiry info |
| `Earnings_Avoided` / `Earnings_Blackout` | Earnings proximity check |
| `Cluster_Risk` | Concentration risk flag |
| `Position_Mgmt` | Tier-specific rollover/close guidance |

## 🏗️ Architecture

```
┌──────────────────────────┐
│   Curated Ticker Lists      │  ← TIER1_CORE + TIER2_WATCHLIST
└───────────┬──────────────┘
           │
           ▼
┌──────────────────────────┐
│   VIX Fetch + Regime        │  ← ^VIX → LOW/NORMAL/ELEVATED/HIGH
└───────────┬──────────────┘
           │
           ▼
┌──────────────────────────┐
│   yfinance Price Data       │  ← 1y daily OHLCV per ticker
└───────────┬──────────────┘
           │
           ▼
┌──────────────────────────┐
│   Technical Analysis        │  ← RSI, BB, SMA200, ATR, MACD,
│   Engine (pandas_ta)        │    Volume, Support Level
└───────────┬──────────────┘
           │
           ▼
┌──────────────────────────┐
│   Earnings Blackout Check   │  ← yfinance calendar ±5/+1d window
└───────────┬──────────────┘
           │
           ▼
┌──────────────────────────┐
│   VIX-Adjusted Screening    │  ← Dynamic RSI + BB thresholds
│   Logic                     │    per regime
└───────────┬──────────────┘
           │
           ▼
┌──────────────────────────┐
│   IV Rank + IV/HV Engine    │  ← ATM option mid-price →
│                             │    Brenner-Subrahmanyam IV approx
│                             │    + 30-day realized vol
└───────────┬──────────────┘
           │
           ▼
┌──────────────────────────┐
│   Cluster / Concentration   │  ← Flags ≥5 simultaneous signals
│   Guard                     │    as correlated macro exposure
└───────────┬──────────────┘
           │
           ▼
┌──────────────────────────┐
│   Output Layer              │  ← CSV: signals_YYYYMMDD.csv
│                             │     LOG: screener_YYYYMMDD.log
└──────────────────────────┘
```

## 🔍 Screening Logic

### Tier Structure

| Tier | Tickers | Delta Target | Special Rules |
|------|---------|--------------|---------------|
| **TIER1_CORE** | SPX, COST, NVDA, IWM, GOOGL | 0.10–0.18 | SPX: gap-down + RSI trigger |
| **TIER2_WATCH** | MSFT, AAPL, AMZN, META, AVGO, CRWD, PLTR, AMD, MU, TSLA, QQQM, CLS, STX | 0.08–0.13 | ATR% cap ≤ 5.0% |

### VIX-Adjusted Entry Thresholds

| VIX Regime | VIX Range | RSI Threshold | BB Position | SPX RSI | Rationale |
|------------|-----------|--------------|-------------|---------|----------|
| **LOW** | < 15 | 28 | < 0.25 | 25 | Premium thin — only enter on deep oversold |
| **NORMAL** | 15–20 | 35 | < 0.40 | 30 | Standard thresholds |
| **ELEVATED** | 20–30 | 38 | < 0.45 | 33 | Fat premium — slightly relaxed |
| **HIGH** | > 30 | 30 | < 0.30 | 28 | Tail risk high — thresholds tightened |

### IV Dual-Pass Filter (Premium Quality Gate)

```
Pass 1 — IV Rank (52-week context):
  IV Rank = (current IV − 52w low) / (52w high − 52w low) × 100
  Threshold: IV Rank ≥ 25
  Fail: premium historically cheap → signal suppressed
  Answers: "Is IV elevated vs its own history?"

Pass 2 — IV/HV Ratio (current relative value):
  IV/HV = implied vol / 30-day realized vol
  Threshold: IV/HV ≥ 1.0
  Fail: options not priced above recent realized moves → signal suppressed
  Answers: "Is the market paying up for options right now?"

Fail-open policy: if IV data is unavailable (None), signal is NOT suppressed.
  A data outage never silently blocks a valid entry.
```

### Primary Technical Filters (All Must Pass)

```python
1. RSI < rsi_threshold          # VIX-regime adjusted (28–38)
2. Price > SMA(200)             # Bullish long-term trend
3. Volume > AVG_VOL(50)         # Liquidity confirmation
4. BB_Position < bb_threshold   # VIX-regime adjusted (0.25–0.45)
5. ATR% > 1.0                   # Adequate volatility
6. Volume_Surge > 1.2×          # Above-average activity
7. NOT in earnings blackout     # ±5/+1 day window around earnings
8. IV Rank ≥ 25                 # Pass 1: IV historically elevated (fail-open)
9. IV/HV Ratio ≥ 1.0            # Pass 2: options priced above realized vol (fail-open)
```

### Signal Strength Algorithm

```
Score Components (0–100):
  RSI extremity      → 0–30 pts  (RSI < 25: 30, < 30: 25, < 35: 20, else: 10)
  BB position        → 0–30 pts  (pos < 0.15: 30, < 0.25: 25, < 0.35: 20, else: 10)
  Volume surge       → 0–25 pts  (> 2.0×: 25, > 1.5×: 20, > 1.2×: 15, else: 10)
  ATR / Volatility   → 0–15 pts  (> 3.0%: 15, > 2.0%: 12, > 1.5%: 10, else: 5)
```

### Concentration / Cluster Guard

```
Threshold: 5 simultaneous signals
Action   : WARNING only (signals are not auto-suppressed)
Message  : Lists all triggering tickers and advises treating as
           correlated macro exposure — size down or select top 2–3
           highest-conviction names only.
```

### Position Management Rules

| Tier | Condition | Action |
|------|-----------|--------|
| All | DTE ≤ 4 | Routine close/rollover review |
| Tier 2 | DTE ≤ 7 + price < short put | Stage 1: net credit roll → debit roll (≤50% of credit) → close |
| Tier 2 | DTE ≤ 7 + price ≤ long put | Stage 2: emergency close |
| All | Profit ≥ 80% of max credit | Early close |

## 📁 Project Structure

```
trading/
└── options-screener/
    ├── options_premium_screener.py   # Main screening engine
    ├── requirements.txt              # Dependencies
    └── README.md                     # This file
```

## 💡 Strategy & Risk Philosophy

**Target strategy**: Vertical Put Credit Spreads — collect premium from oversold, high-IV stocks in confirmed uptrends.

**Why IV Rank matters**: Selling premium at low IV Rank is the most common retail mistake. Premium sellers profit from IV contraction — entering at low IV means collecting thin premium with no edge.

**Why VIX regime matters**: In HIGH VIX environments, even rich-looking premium carries elevated gap-down and black-swan risk. Thresholds are *tightened* (not relaxed) in HIGH VIX to demand stronger confirmation signals before entry.

**Why cluster guard matters**: If 6 of 13 tickers trigger on the same day, they are almost certainly responding to the same macro event. That is one trade, not six. The guard surfaces this so position sizing reflects actual risk concentration.

## 🎓 Technical Skills Demonstrated

- **Options pricing**: Brenner-Subrahmanyam ATM IV approximation from live option chains
- **Volatility analysis**: IV Rank, IV Percentile, IV/HV ratio, 30-day realized vol
- **Data engineering**: yfinance ELT pipeline — price data, option chains, earnings calendar
- **Statistical analysis**: Multi-factor screening with regime-conditional thresholds
- **Risk management**: Concentration guard, earnings blackout, tier-based position rules
- **Code quality**: Modular functions, fail-open error handling, structured logging
- **Version control**: Git workflow, clean commit history

## 🚀 Future Enhancements

- [x] IV Rank / IV Percentile integration
- [x] IV/HV ratio dual-pass filter
- [x] Dynamic VIX-adjusted entry thresholds
- [x] Earnings blackout window
- [x] Concentration / cluster risk guard
- [ ] Backtesting framework with historical signal performance
- [ ] Email/Slack alerts for new signals
- [ ] SQLite database for signal history tracking
- [ ] Web dashboard (Streamlit)

## 📝 Dependencies

- **pandas**: Data manipulation and analysis
- **yfinance**: Market data + option chain + earnings calendar
- **pandas-ta**: Technical analysis indicators (RSI, BB, ATR, MACD)
- **math / datetime**: IV approximation and DTE calculations

## 📄 License

MIT License

## 👤 Author

**Hanbit Choi**
- LinkedIn: [linkedin.com/in/hanbitchoi0917](https://www.linkedin.com/in/hanbitchoi0917/)
- GitHub: [@hbchoi0917](https://github.com/hbchoi0917)

---

**⚠️ Disclaimer**: This tool is for educational and research purposes only. Not financial advice. Options trading involves substantial risk.
