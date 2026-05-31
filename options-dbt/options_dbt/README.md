# Options Trading dbt Project

Analytics pipeline for a real options trading portfolio — 18 months of live trade data (Jan 2025–May 2026), 4 brokerage accounts, 50 tickers.

Raw data source: Fidelity transaction exports (options legs only).

---

## Stack

| Layer | Tool |
|---|---|
| Transform | dbt-core 1.11 |
| Warehouse | DuckDB (local, zero-infra) |
| Source data | Fidelity CSV export (options transactions) |

---

## Project Structure

```
models/
├── staging/
│   └── stg_fidelity_transactions   # parse OCC symbol → ticker/expiry/strike/option_type
├── intermediate/
│   ├── int_option_legs              # enriched legs: DTE, leg_role, notional
│   └── int_spread_trades            # group legs → realized P&L per spread cycle
└── marts/
    ├── mart_monthly_pnl             # monthly P&L + cumulative, by account
    ├── mart_ticker_performance      # win rate, avg P&L, trade frequency per ticker
    └── mart_account_summary         # lifetime performance per account
```

---

## Key Design Decisions

**Surrogate key with row_number:** Fidelity exports can have identical rows when multiple contracts are traded at the same price on the same day. A `row_number()` partition handles deduplication without dropping valid data.

**Spread matching by natural key:** Legs are grouped by `(account, ticker, expiry, option_type, strike)` rather than sequential pairing. This correctly handles rolled positions — a CLOSING leg followed by a new OPENING leg produces two separate trade cycles.

**DIVIDEND / ASSIGNMENT routing:** `transaction_category` flag on `stg_fidelity_transactions` isolates non-option cash flows before they reach spread matching logic, preventing P&L contamination.

---

## Setup

```bash
pip install dbt-duckdb

# run with sample data (37 anonymized rows, all edge cases covered)
dbt seed --select sample_fidelity_transactions
dbt run
dbt test
```

---

## Data Quality Tests (10 total, all passing)

- `unique` + `not_null` on `transaction_id`
- `not_null` on `trade_date`, `account_name`, `transaction_type`, `amount`
- `accepted_values` on `account_name`, `transaction_type`, `transaction_category`, `option_type`
