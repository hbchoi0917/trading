"""
export_to_sheets.py
-------------------
Exports dbt mart tables from DuckDB to Google Sheets for Looker Studio.

Setup:
    1. pip install duckdb gspread google-auth
    2. Create a Google Cloud service account and download credentials JSON
       https://console.cloud.google.com → APIs → Google Sheets API → Credentials
    3. Share the target Google Sheet with the service account email
    4. Set GOOGLE_CREDENTIALS_PATH and SPREADSHEET_ID below (or via env vars)

Usage:
    python scripts/export_to_sheets.py
"""

import os
import duckdb
import gspread
from google.oauth2.service_account import Credentials

# ── Config ────────────────────────────────────────────────────────────────────

DUCKDB_PATH         = os.environ.get("DUCKDB_PATH", "options_trading.duckdb")
CREDENTIALS_PATH    = os.environ.get("GOOGLE_CREDENTIALS_PATH", "google_credentials.json")
SPREADSHEET_ID      = os.environ.get("SPREADSHEET_ID", "YOUR_SPREADSHEET_ID_HERE")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

EXPORTS = {
    "monthly_pnl":        ("main_marts.mart_monthly_pnl",        "mart_monthly_pnl"),
    "ticker_performance": ("main_marts.mart_ticker_performance",  "mart_ticker_performance"),
    "account_summary":    ("main_marts.mart_account_summary",     "mart_account_summary"),
    "spread_trades":      ("main_intermediate.int_spread_trades", "int_spread_trades"),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_sheet_client():
    creds = Credentials.from_service_account_file(CREDENTIALS_PATH, scopes=SCOPES)
    return gspread.authorize(creds)


def df_to_sheet(ws, df):
    """Clear sheet and write DataFrame including headers."""
    ws.clear()
    data = [df.columns.tolist()] + df.astype(str).values.tolist()
    ws.update(data, value_input_option="USER_ENTERED")
    print(f"  → {ws.title}: {len(df)} rows written")


def ensure_worksheet(spreadsheet, title: str):
    """Get or create a worksheet by title."""
    try:
        return spreadsheet.worksheet(title)
    except gspread.WorksheetNotFound:
        return spreadsheet.add_worksheet(title=title, rows=5000, cols=30)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import pandas as pd

    print(f"Connecting to DuckDB: {DUCKDB_PATH}")
    con = duckdb.connect(DUCKDB_PATH)

    print("Authenticating with Google Sheets...")
    gc = get_sheet_client()
    spreadsheet = gc.open_by_key(SPREADSHEET_ID)
    print(f"Opened spreadsheet: {spreadsheet.title}")

    for label, (table, sheet_title) in EXPORTS.items():
        print(f"\nExporting {table}...")
        df = con.execute(f"SELECT * FROM {table}").df()

        # convert date columns to string for Sheets compatibility
        for col in df.select_dtypes(include=["datetime64[ns]", "object"]).columns:
            df[col] = df[col].astype(str).replace("NaT", "")

        ws = ensure_worksheet(spreadsheet, sheet_title)
        df_to_sheet(ws, df)

    con.close()
    print("\nDone. Refresh your Looker Studio report to see updated data.")


if __name__ == "__main__":
    main()
