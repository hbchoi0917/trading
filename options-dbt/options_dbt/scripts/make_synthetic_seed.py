"""Write seeds/sample_fidelity_transactions.csv: a small, fully synthetic Fidelity-style export.

Every row is invented: tickers, dates, strikes, prices and accounts do not come from any
real trade. The scenarios are chosen to exercise the models' edge cases:

- credit put spreads closed for a win, closed for a loss, and expired worthless
- a roll (close and reopen at a later expiry on the same day)
- a split fill (one order filled as two rows on the same day)
- a short put that was assigned
- covered calls (expired, bought back, still open)
- positions still open at the end of the window, and closes whose opening trade is
  before the window (so their realized P&L is the closing leg only)
- half-dollar strikes, a dotted ticker (BRK.B), a description with a comma,
  and both Margin and Cash account types

Run from options-dbt/options_dbt:  python scripts/make_synthetic_seed.py
"""
import csv
import datetime as dt
import pathlib

OUT = pathlib.Path(__file__).resolve().parent.parent / "seeds" / "sample_fidelity_transactions.csv"

NAMES = {
    "AAPL": "APPLE INC", "AMZN": "AMAZON.COM INC", "SPY": "SPDR S&P 500 ETF TRUST",
    "QQQ": "INVESCO QQQ TR UNIT SER 1", "AMD": "ADVANCED MICRO DEVICES INC",
    "JPM": "JPMORGAN CHASE & CO", "KO": "COCA-COLA CO", "XOM": "EXXON MOBIL CORP",
    "BRK.B": "BERKSHIRE HATHAWAY INC, CL B",
}
COMMISSION, FEE = 0.65, 0.04   # per contract
COLUMNS = ["run_date", "account", "action", "symbol", "description", "type", "exchange_quantity",
           "exchange_currency", "currency", "price", "quantity", "exchange_rate", "commission",
           "fees", "accrued_interest", "amount", "settlement_date"]

rows = []


def d(s):
    return dt.date.fromisoformat(s)


def next_business_day(day):
    day += dt.timedelta(days=1)
    while day.weekday() >= 5:
        day += dt.timedelta(days=1)
    return day


def fmt_strike(k):
    return f"{k:g}"


def contract(ticker, kind, expiry, strike):
    """Return (OCC-style symbol, Fidelity-style description) for one option contract."""
    exp = d(expiry)
    k = fmt_strike(strike)
    symbol = f"-{ticker}{exp:%y%m%d}{kind[0]}{k}"
    desc = f"{kind} ({ticker}) {NAMES[ticker]} {exp:%b %d %y}".upper() + f" ${k} (100 SHS)"
    return symbol, desc


def trade(day, account, side, effect, ticker, kind, expiry, strike, price, n=1, acct_type="Margin"):
    """A filled order. side: SOLD / BOUGHT, effect: OPENING / CLOSING."""
    assert d(day) <= d(expiry), "a trade cannot be after expiry"
    symbol, desc = contract(ticker, kind, expiry, strike)
    commission, fees = round(COMMISSION * n, 2), round(FEE * n, 2)
    gross = price * 100 * n
    amount = gross - commission - fees if side == "SOLD" else -(gross + commission + fees)
    rows.append({
        "run_date": day, "account": account,
        "action": f"YOU {side} {effect} TRANSACTION {desc} ({acct_type})",
        "symbol": symbol, "description": desc, "type": acct_type, "exchange_quantity": 0.0,
        "exchange_currency": "", "currency": "USD", "price": price,
        "quantity": float(-n if side == "SOLD" else n), "exchange_rate": 0.0,
        "commission": commission, "fees": fees, "accrued_interest": "",
        "amount": round(amount, 2), "settlement_date": next_business_day(d(day)).isoformat(),
    })


def posting(event, account, ticker, kind, expiry, strike, qty, acct_type="Margin", as_of=None):
    """A broker posting with no cash: EXPIRED or ASSIGNED, booked the next business day."""
    as_of = d(as_of or expiry)
    symbol, desc = contract(ticker, kind, expiry, strike)
    head = (f"EXPIRED {desc.replace(' (100 SHS)', '')} as of {as_of:%b-%d-%Y}"
            if event == "EXPIRED" else f"ASSIGNED as of {as_of:%b-%d-%Y}")
    rows.append({
        "run_date": next_business_day(as_of).isoformat(), "account": account,
        "action": f"{head} {desc} ({acct_type})", "symbol": symbol, "description": desc,
        "type": acct_type, "exchange_quantity": 0.0, "exchange_currency": "", "currency": "USD",
        "price": "", "quantity": float(qty), "exchange_rate": 0.0, "commission": "", "fees": "",
        "accrued_interest": "", "amount": 0.0, "settlement_date": "",
    })


def put_spread(account, ticker, open_day, expiry, short_k, long_k, short_px, long_px, n=1):
    trade(open_day, account, "SOLD", "OPENING", ticker, "PUT", expiry, short_k, short_px, n)
    trade(open_day, account, "BOUGHT", "OPENING", ticker, "PUT", expiry, long_k, long_px, n)


def close_spread(account, ticker, day, expiry, short_k, long_k, short_px, long_px, n=1):
    trade(day, account, "BOUGHT", "CLOSING", ticker, "PUT", expiry, short_k, short_px, n)
    trade(day, account, "SOLD", "CLOSING", ticker, "PUT", expiry, long_k, long_px, n)


# ── scenarios ──────────────────────────────────────────────────────────────────
# closes whose opening trade is before the window
close_spread("Account_A", "JPM", "2025-01-10", "2025-01-17", 240, 235, 0.20, 0.05)

# win: closed early at a fraction of the credit
put_spread("Account_A", "AAPL", "2025-01-06", "2025-02-07", 220, 215, 3.10, 1.60)
close_spread("Account_A", "AAPL", "2025-01-24", "2025-02-07", 220, 215, 0.85, 0.30)

# loss: stopped out after a selloff
put_spread("Account_A", "AMZN", "2025-02-18", "2025-03-21", 210, 200, 4.20, 2.05)
close_spread("Account_A", "AMZN", "2025-03-10", "2025-03-21", 210, 200, 9.40, 5.10)

# expired worthless
put_spread("Account_B", "SPY", "2025-03-03", "2025-04-04", 560, 555, 2.35, 1.42)
posting("EXPIRED", "Account_B", "SPY", "PUT", "2025-04-04", 560, 1)
posting("EXPIRED", "Account_B", "SPY", "PUT", "2025-04-04", 555, -1)

# short put assigned (half-dollar strike)
trade("2025-03-24", "Account_A", "SOLD", "OPENING", "AMD", "PUT", "2025-04-11", 102.5, 2.15)
posting("ASSIGNED", "Account_A", "AMD", "PUT", "2025-04-11", 102.5, 1)

# covered call in a cash account, expired
trade("2025-05-12", "Account_C", "SOLD", "OPENING", "KO", "CALL", "2025-06-20", 72.5, 0.62, acct_type="Cash")
posting("EXPIRED", "Account_C", "KO", "CALL", "2025-06-20", 72.5, 1, acct_type="Cash")

# split fill: the 2-lot short leg filled as two rows on the same day
trade("2025-06-09", "Account_A", "SOLD", "OPENING", "QQQ", "PUT", "2025-07-18", 500, 5.05)
trade("2025-06-09", "Account_A", "SOLD", "OPENING", "QQQ", "PUT", "2025-07-18", 500, 5.05)
trade("2025-06-09", "Account_A", "BOUGHT", "OPENING", "QQQ", "PUT", "2025-07-18", 490, 3.20, n=2)
close_spread("Account_A", "QQQ", "2025-07-01", "2025-07-18", 500, 490, 1.10, 0.55, n=2)

put_spread("Account_D", "JPM", "2025-08-04", "2025-09-05", 270, 265, 1.95, 1.05)
close_spread("Account_D", "JPM", "2025-08-22", "2025-09-05", 270, 265, 0.40, 0.12)

# single short put in a cash account, bought back
trade("2025-09-15", "Account_B", "SOLD", "OPENING", "XOM", "PUT", "2025-10-17", 105, 1.30, acct_type="Cash")
trade("2025-10-03", "Account_B", "BOUGHT", "CLOSING", "XOM", "PUT", "2025-10-17", 105, 0.35, acct_type="Cash")

# dotted ticker and a comma in the description, expired
put_spread("Account_A", "BRK.B", "2025-10-20", "2025-11-21", 480, 470, 3.60, 2.10)
posting("EXPIRED", "Account_A", "BRK.B", "PUT", "2025-11-21", 480, 1)
posting("EXPIRED", "Account_A", "BRK.B", "PUT", "2025-11-21", 470, -1)

# covered call bought back at a loss before an ex-dividend date
trade("2025-11-10", "Account_C", "SOLD", "OPENING", "KO", "CALL", "2025-12-19", 75, 0.48, acct_type="Cash")
trade("2025-12-15", "Account_C", "BOUGHT", "CLOSING", "KO", "CALL", "2025-12-19", 75, 1.25, acct_type="Cash")

# roll: close for a loss and reopen at a later expiry on the same day, then close for a win
put_spread("Account_A", "AAPL", "2025-12-01", "2026-01-02", 255, 250, 2.40, 1.30)
close_spread("Account_A", "AAPL", "2025-12-19", "2026-01-02", 255, 250, 4.80, 2.55)
put_spread("Account_A", "AAPL", "2025-12-19", "2026-01-23", 255, 250, 5.70, 3.40)
close_spread("Account_A", "AAPL", "2026-01-14", "2026-01-23", 255, 250, 1.20, 0.50)

put_spread("Account_D", "SPY", "2026-02-02", "2026-03-06", 640, 630, 4.40, 2.60)
close_spread("Account_D", "SPY", "2026-02-20", "2026-03-06", 640, 630, 1.35, 0.70)

# still open at the end of the window
put_spread("Account_A", "AMZN", "2026-03-16", "2026-04-17", 207.5, 202.5, 2.85, 1.75)

trade("2026-04-06", "Account_B", "SOLD", "OPENING", "QQQ", "PUT", "2026-05-15", 540, 6.20)
posting("EXPIRED", "Account_B", "QQQ", "PUT", "2026-05-15", 540, 1)

# covered call still open
trade("2026-05-11", "Account_C", "SOLD", "OPENING", "KO", "CALL", "2026-06-18", 77.5, 0.55, acct_type="Cash")

# ── write ─────────────────────────────────────────────────────────────────────
rows.sort(key=lambda r: (r["run_date"], r["account"]))
with OUT.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
print(f"wrote {len(rows)} synthetic rows to {OUT.name}")
