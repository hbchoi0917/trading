# Options Trading dbt Pipeline

**Transform raw Fidelity transaction exports into spread-level P&L — powered by dbt + DuckDB.**

[![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B?logo=streamlit&logoColor=white)](https://options-trading-dash.streamlit.app/)

---

## What Problem This Solves

Fidelity exports raw **transaction legs** — one row per contract action. There is no native concept of a "spread trade" or realized P&L per trade cycle. A single put credit spread generates at minimum two legs (open + close), and rolls can produce four or more.

This pipeline solves that by:

1. Parsing OCC option symbols (`-COST260402P940` → ticker `COST`, expiry `2026-04-02`, strike `940`, type `PUT`)
2. Grouping legs into complete spread trade cycles using a natural key (account + ticker + expiry + option\_type + strike)
3. Computing realized P&L, win/loss classification, holding days, and DTE at entry per cycle
4. Aggregating into monthly, ticker-level, and account-level mart tables
5. Powering a live Streamlit dashboard directly from the marts

---

## Pipeline Overview

```
Fidelity CSV export
        │
        ▼
┌─────────────────────────────┐
│  STAGING                    │  Parse, clean, type-cast
│  stg_fidelity_transactions  │  One row = one contract leg
└────────────────┬────────────┘
                 │
        ┌────────▼────────┐
        │  INTERMEDIATE   │  Apply business logic
        │  int_option_legs│  Enrich with DTE, notional, leg role
        │  int_spread_    │  Group legs → spread trade cycles
        │  trades         │  Compute realized P&L per cycle
        └────────┬────────┘
                 │
   ┌─────────────┼──────────────┐
   ▼             ▼              ▼
mart_monthly  mart_ticker   mart_account
_pnl          _performance  _summary
   │             │              │
   └─────────────┴──────────────┘
                 │
                 ▼
        Streamlit Dashboard
```

---

## Layer-by-Layer Explanation

### Staging — "원본을 그대로, 깔끔하게"

**Input:** Raw Fidelity CSV
```
run_date   | action                                     | symbol         | amount
2025-03-31 | YOU SOLD OPENING TRANSACTION PUT (PLTR)... | -PLTR250502P82 | 670.30
```

**What staging does:** Column renaming, type casting, and symbol parsing only. No business logic.

```sql
-- Extract ticker, expiry, strike, option type from OCC symbol
regexp_extract(symbol, '-?([A-Z]+)\d{6}[PC]', 1)        as ticker       -- 'PLTR'
strptime('20' || regexp_extract(symbol, '\d{6}', 0), '%Y%m%d') as expiry_date  -- 2025-05-02
cast(regexp_extract(symbol, '[PC]([\d\.]+)', 1) as double)     as strike        -- 82.0
case when action ilike '%OPENING%' then 'OPENING' end          as transaction_type
case when action ilike '%YOU SOLD%' then 'SOLD'   end          as direction
```

**Output:** One row = one contract leg. Faithful to source, no interpretation.

---

### Intermediate — "비즈니스 로직 적용"

**The core problem:** Fidelity gives legs; we need spreads.

```
leg 1: PLTR PUT 82 | SOLD OPENING   | +670.30  ← entry, premium received
leg 2: PLTR PUT 82 | BOUGHT CLOSING | -120.00  ← exit, bought back
              ↓ group by (account + ticker + expiry + option_type + strike)
spread: PLTR PUT 82 | realized_pnl = +550.30 | WIN | held 32 days
```

**Design decision documented in `int_spread_trades.sql`:**
> We group by natural key rather than attempting sequential leg matching. This handles rolls naturally — a CLOSING leg followed by a new OPENING leg on the same key creates two separate cycle rows.

**Enrichments added in `int_option_legs.sql`:**

| Column | Calculation | Why it matters |
|---|---|---|
| `dte_at_trade` | `expiry_date - trade_date` | Entry DTE discipline analysis |
| `notional_value` | `strike × 100 × contracts` | Risk exposure sizing |
| `leg_role` | `SOLD + OPENING → SHORT_OPEN` | Spread structure audit |

**Output:** One row = one complete spread trade cycle with realized P&L, win/loss flag, holding days.

---

### Marts — "질문에 바로 대답할 수 있는 형태"

Intermediate gives trade-level rows. Marts answer business questions directly.

**`mart_monthly_pnl`** — *"How did each account perform each month?"*
```sql
sum(realized_pnl)                                            as net_pnl
sum(net_pnl) over (partition by account_name order by month) as cumulative_pnl
net_pnl - lag(net_pnl) over (partition by account_name order by month) as pnl_mom_change
case when net_pnl > 0 then true else false end               as is_win_month
```

**`mart_ticker_performance`** — *"Which tickers are worth rotating into?"*
```sql
sum(realized_pnl)                          as total_pnl
count(*) filter (trade_result = 'WIN')     as wins
rank() over (order by total_pnl desc)      as pnl_rank
case when ticker in ('MSFT','NFLX','IONQ','RGTI') then true end as is_excluded
```

**`mart_account_summary`** — *"How does each account compare lifetime?"*
- Trade-level win rate vs. month-level win rate (two distinct metrics)
- Best/worst single trade and best/worst month per account
- Joins `int_spread_trades` + `mart_monthly_pnl` to compute both dimensions

**Output:** Streamlit dashboard queries these marts directly via `SELECT * FROM main_marts.mart_*`.

---

## Insights This Enables

Questions that were **unanswerable** from raw Fidelity exports, now trivially queryable:

| Question | Before | After |
|---|---|---|
| What did this spread actually return? | ❌ legs only | ✅ `realized_pnl` per cycle |
| Monthly win rate per account? | ❌ | ✅ `mart_monthly_pnl.is_win_month` |
| Which DTE range is most profitable? | ❌ | ✅ `avg_dte_at_open` in ticker mart |
| COST vs TSLA — who's more efficient per trade? | ❌ | ✅ `pnl_per_trade` + `pnl_rank` |
| Should MSFT be re-entered? | ❌ | ✅ `is_excluded` flag with loss history |

---

## Portfolio Impact

> *"Fidelity's raw export gives transaction legs. This pipeline turns legs into trades, trades into monthly P&L, and monthly P&L into portfolio decisions — with a live dashboard as the delivery layer."*

This project demonstrates the full analytics engineering workflow:

1. **Messy source** → real brokerage exports with unparsed OCC symbols and no trade IDs
2. **Non-trivial transformation** → leg-to-spread matching without explicit foreign keys
3. **Layered modeling** → staging / intermediate / marts with clear separation of concerns
4. **Data quality** → schema tests (unique, not\_null, accepted\_values) + 4 singular SQL assertions
5. **Delivery** → live Streamlit dashboard, Google Sheets export via `export_to_sheets.py`

---

## Project Structure

```
options-dbt/
├── options_dbt/                    ← dbt project root
│   ├── models/
│   │   ├── staging/
│   │   │   ├── stg_fidelity_transactions.sql   Parse OCC symbols, classify legs
│   │   │   ├── _sources.yml
│   │   │   └── _stg_models.yml                 Schema tests
│   │   ├── intermediate/
│   │   │   ├── int_option_legs.sql             Enrich legs (DTE, notional, role)
│   │   │   ├── int_spread_trades.sql           Group legs → spread P&L cycles
│   │   │   └── _int_models.yml                 Schema tests
│   │   └── marts/
│   │       ├── mart_monthly_pnl.sql
│   │       ├── mart_ticker_performance.sql
│   │       ├── mart_account_summary.sql
│   │       └── _mart_models.yml                Schema tests
│   ├── tests/
│   │   ├── assert_win_loss_pnl_sign.sql        WIN → pnl > 0, LOSS → pnl < 0
│   │   ├── assert_close_after_open.sql         close_date ≥ open_date
│   │   ├── assert_dte_non_negative.sql         DTE at trade ≥ 0
│   │   └── assert_win_rate_valid_range.sql     win_rate_pct in [0, 100]
│   ├── seeds/
│   │   └── sample_fidelity_transactions.csv   Synthetic sample data for dbt seed
│   ├── scripts/
│   │   ├── export_to_sheets.py                DuckDB → Google Sheets → Looker Studio
│   │   └── LOOKER_STUDIO_SETUP.md
│   ├── dbt_project.yml
│   └── profiles.yml                           DuckDB connection via env_var
└── streamlit_app/
    ├── app.py                                 3-page interactive dashboard
    ├── sample_options_trading.duckdb          Pre-built marts (162 sample trades)
    └── requirements.txt
```

---

## Quick Start

```bash
# 1. Install dbt-duckdb
pip install dbt-core dbt-duckdb

# 2. Seed sample data and run models
cd options-dbt/options_dbt
dbt seed          # loads sample_fidelity_transactions.csv
dbt run           # builds all models
dbt test          # runs schema + singular tests

# 3. Launch dashboard
cd ../streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

---

## Stack

- **[dbt-core](https://docs.getdbt.com/)** — model orchestration, schema tests, documentation
- **[DuckDB](https://duckdb.org/)** — embedded OLAP engine, no server required
- **[Streamlit](https://streamlit.io/)** — dashboard deployment
- **[Plotly](https://plotly.com/)** — interactive chart rendering
- **[pandas](https://pandas.pydata.org/)** — DataFrame layer between DuckDB and Streamlit
