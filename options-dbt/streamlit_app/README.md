# Options Trading Analytics Dashboard

Streamlit + DuckDB + Plotly dashboard built on top of a dbt analytics pipeline.

**Data:** The public demo runs on **synthetic sample data** (4 accounts · 48 tickers · 162 trades, Jan 2025 – May 2026). The real 18-month trade history is private; point `DUCKDB_PATH` at a dbt-built database to run the same dashboard on real data. Figures shown in the live demo are illustrative and do not reflect actual performance.

**Stack:**
```
Fidelity CSV exports
    → dbt (DuckDB) staging / intermediate / marts
        → Streamlit + Plotly dashboard
```

## Pages

| Page | Content |
|---|---|
| Portfolio Overview | Lifetime KPIs, monthly P&L bar + cumulative line, stacked by account, quarterly table |
| Ticker Performance | Top winners/losers, win rate vs P&L bubble chart, excluded tickers, full searchable table |
| Account Summary | Per-account KPIs, cumulative growth lines, monthly grouped bars, trade detail |

## Run locally

```bash
pip install -r requirements.txt

# with full data (private)
DUCKDB_PATH=/path/to/options_trading.duckdb streamlit run app.py

# with sample data (public)
DUCKDB_PATH=sample_options_trading.duckdb streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set main file as `options-dbt/streamlit_app/app.py`
4. Add secret: `DUCKDB_PATH = "sample_options_trading.duckdb"`
5. Deploy → get public URL for portfolio
