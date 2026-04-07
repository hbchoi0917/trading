"""
position_tracker.py
-------------------
CSV-based position ledger for put credit spread positions.
Tracks open/closed positions, monitors daily for action triggers,
and suggests rollover parameters.

Designed for automated trading while traveling (timezone-independent).
Gmail SMTP notifications: alerts fire even when sleeping in a different timezone.

Setup:
    1. Copy .env.example to .env and fill in your Gmail credentials
    2. Add position entries via add_position()
    3. Schedule monitor_positions() via cron (runs daily at US market open)

Usage:
    from position_tracker import add_position, monitor_positions

    add_position(
        ticker='TSLA',
        tier='TIER2_WATCH',
        short_put_strike=200.0,
        expiry_date='2026-05-16',
        entry_credit=1.25,
        contracts=1,
    )

    monitor_positions()
"""

import os
import smtplib
import ssl
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional; set env vars manually if not installed

from options_premium_screener import (
    evaluate_tier2_position,
    SPREAD_WIDTH,
    EARLY_CLOSE_PROFIT_PCT,
    BASE_DTE_ACTION,
    MAX_ROLLOVER_DEBIT_PCT,
    T2_ROLLOVER_DTE,
    T2_EMERGENCY_CLOSE_DTE,
    DTE_MAX,
)

# ============ CONFIG ============
POSITIONS_FILE = 'positions.csv'

# Gmail credentials loaded from .env (NEVER hardcode here)
GMAIL_SENDER   = os.getenv('GMAIL_SENDER', '')
GMAIL_PASSWORD = os.getenv('GMAIL_PASSWORD', '')  # App Password (16 chars)
GMAIL_RECEIVER = os.getenv('GMAIL_RECEIVER', '')

COLUMNS = [
    'position_id',
    'ticker',
    'tier',
    'short_put_strike',
    'long_put_strike',
    'expiry_date',
    'entry_date',
    'entry_credit',
    'contracts',
    'total_credit_usd',
    'status',
    'close_date',
    'close_debit',
    'pnl_usd',
    'close_reason',
    'roll_to_position_id',
    'notes',
]


# ============ EMAIL NOTIFICATION ============
def notify(subject: str, body: str):
    """
    Send Gmail alert via SMTP SSL (port 465).
    Credentials loaded from .env — never committed to GitHub.
    Falls back to print() if credentials not configured.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S CT')
    full_body  = f"[{timestamp}]\n\n{body}"

    if not GMAIL_SENDER or not GMAIL_PASSWORD or not GMAIL_RECEIVER:
        # Fallback: print to console/log if .env not set
        print(f"\n{'='*60}")
        print(f"[ALERT] {timestamp}")
        print(f"Subject: {subject}")
        print(full_body)
        print('='*60)
        return

    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From']    = GMAIL_SENDER
        msg['To']      = GMAIL_RECEIVER
        msg.attach(MIMEText(full_body, 'plain'))

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
            server.login(GMAIL_SENDER, GMAIL_PASSWORD)
            server.sendmail(GMAIL_SENDER, GMAIL_RECEIVER, msg.as_string())

        print(f"[NOTIFY] Email sent: {subject}")

    except Exception as e:
        print(f"[NOTIFY] Email failed: {e}")
        print(f"  Subject: {subject}")
        print(f"  Body: {full_body}")


# ============ HELPERS ============
def _load() -> pd.DataFrame:
    if os.path.exists(POSITIONS_FILE):
        df = pd.read_csv(POSITIONS_FILE, dtype=str)
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = ''
        return df[COLUMNS]
    return pd.DataFrame(columns=COLUMNS)


def _save(df: pd.DataFrame):
    df.to_csv(POSITIONS_FILE, index=False)


def _get_current_price(ticker: str) -> float | None:
    try:
        data = yf.download(ticker, period='2d', interval='1d', progress=False, group_by=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(0)
        data.columns = [c.capitalize() for c in data.columns]
        return float(data['Close'].iloc[-1])
    except Exception:
        return None


def _make_position_id(ticker: str, entry_date: str, short_put_strike: float) -> str:
    return f"{ticker}_{entry_date.replace('-', '')}_{int(short_put_strike)}"


# ============ ADD POSITION ============
def add_position(
    ticker: str,
    tier: str,
    short_put_strike: float,
    expiry_date: str,
    entry_credit: float,
    contracts: int = 1,
    long_put_strike: float = None,
    notes: str = '',
) -> str:
    """
    Log a new put spread position.
    long_put_strike defaults to short_put_strike - SPREAD_WIDTH ($10).
    Returns position_id.
    """
    df = _load()
    if long_put_strike is None:
        long_put_strike = short_put_strike - SPREAD_WIDTH

    entry_date_str = datetime.now().strftime('%Y-%m-%d')
    position_id    = _make_position_id(ticker, entry_date_str, short_put_strike)
    total_credit   = round(entry_credit * contracts * 100, 2)

    new_row = {
        'position_id':          position_id,
        'ticker':               ticker,
        'tier':                 tier,
        'short_put_strike':     str(short_put_strike),
        'long_put_strike':      str(long_put_strike),
        'expiry_date':          expiry_date,
        'entry_date':           entry_date_str,
        'entry_credit':         str(entry_credit),
        'contracts':            str(contracts),
        'total_credit_usd':     str(total_credit),
        'status':               'OPEN',
        'close_date':           '',
        'close_debit':          '',
        'pnl_usd':              '',
        'close_reason':         '',
        'roll_to_position_id':  '',
        'notes':                notes,
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    _save(df)
    print(f"[TRACKER] Position added: {position_id} | Credit: ${entry_credit}/share (${total_credit} total)")
    notify(
        subject=f"[NEW POSITION] {position_id}",
        body=(
            f"New put spread opened:\n"
            f"  Ticker    : {ticker} ({tier})\n"
            f"  Short Put : ${short_put_strike}\n"
            f"  Long Put  : ${long_put_strike}\n"
            f"  Expiry    : {expiry_date}\n"
            f"  Credit    : ${entry_credit}/share (${total_credit} total for {contracts} contract(s))"
        )
    )
    return position_id


# ============ UPDATE / CLOSE POSITION ============
def update_position(
    position_id: str,
    close_debit: float,
    close_reason: str,
    roll_to_position_id: str = '',
    notes: str = '',
):
    """
    Mark a position as CLOSED or ROLLED with realized P&L.
    close_reason: 'PROFIT_TARGET' | 'ROUTINE_REVIEW' | 'ROLLOVER' | 'EMERGENCY_CLOSE' | 'EXPIRED'
    """
    df = _load()
    mask = df['position_id'] == position_id
    if not mask.any():
        print(f"[TRACKER] Position {position_id} not found.")
        return

    idx          = df[mask].index[0]
    entry_credit = float(df.at[idx, 'entry_credit'])
    contracts    = int(df.at[idx, 'contracts'])
    pnl          = round((entry_credit - close_debit) * contracts * 100, 2)
    status       = 'ROLLED' if close_reason == 'ROLLOVER' else 'CLOSED'

    df.at[idx, 'status']              = status
    df.at[idx, 'close_date']          = datetime.now().strftime('%Y-%m-%d')
    df.at[idx, 'close_debit']         = str(close_debit)
    df.at[idx, 'pnl_usd']            = str(pnl)
    df.at[idx, 'close_reason']        = close_reason
    df.at[idx, 'roll_to_position_id'] = roll_to_position_id
    if notes:
        df.at[idx, 'notes'] = notes
    _save(df)

    pnl_label = f"+${pnl}" if pnl >= 0 else f"-${abs(pnl)}"
    print(f"[TRACKER] {position_id} → {status} | P&L: {pnl_label} | Reason: {close_reason}")
    notify(
        subject=f"[{status}] {position_id} — P&L {pnl_label}",
        body=(
            f"Position closed:\n"
            f"  ID          : {position_id}\n"
            f"  Status      : {status}\n"
            f"  Close debit : ${close_debit}/share\n"
            f"  Realized P&L: {pnl_label}\n"
            f"  Reason      : {close_reason}\n"
            f"  Rolled to   : {roll_to_position_id or 'N/A'}"
        )
    )


# ============ ROLLOVER SUGGESTION ============
def suggest_rollover(ticker: str, short_put_strike: float, entry_credit: float) -> dict:
    max_debit     = round(entry_credit * MAX_ROLLOVER_DEBIT_PCT, 2)
    max_debit_usd = round(max_debit * 100, 2)
    today         = date.today()
    roll_start    = today + timedelta(days=DTE_MAX + 1)
    roll_end      = today + timedelta(days=DTE_MAX + 30)

    try:
        t            = yf.Ticker(ticker)
        raw_expiries = t.options
        candidates   = sorted([
            datetime.strptime(e, '%Y-%m-%d').date()
            for e in raw_expiries
            if roll_start <= datetime.strptime(e, '%Y-%m-%d').date() <= roll_end
        ])
    except Exception:
        candidates = []

    target_expiry = candidates[0] if candidates else None
    dte_new       = (target_expiry - today).days if target_expiry else None

    return {
        'ticker':                 ticker,
        'current_short_put':      short_put_strike,
        'current_long_put':       short_put_strike - SPREAD_WIDTH,
        'suggested_expiry':       str(target_expiry) if target_expiry else 'N/A',
        'suggested_dte':          dte_new,
        'priority_1':             f'Net credit roll: lower strike + {target_expiry} expiry',
        'priority_2':             f'Same strike {short_put_strike} + {target_expiry}: max debit ${max_debit}/share (${max_debit_usd}/contract)',
        'max_rollover_debit':     max_debit,
        'max_rollover_debit_usd': max_debit_usd,
        'fallback':               'EMERGENCY_CLOSE if neither viable',
    }


# ============ DAILY MONITOR ============
def monitor_positions():
    """
    Daily position monitor. Schedule via cron at US market open.
    Korea timezone example: 10:30 PM KST = 8:30 AM CT (market open)

    Cron example (server in US/Central):
        30 8 * * 1-5 /usr/bin/python3 /path/to/position_tracker.py
    """
    df = _load()
    open_positions = df[df['status'] == 'OPEN']

    if open_positions.empty:
        print("[MONITOR] No open positions.")
        return

    print(f"\n[MONITOR] {len(open_positions)} open position(s) — {datetime.now().strftime('%Y-%m-%d %H:%M CT')}")

    for _, row in open_positions.iterrows():
        ticker           = row['ticker']
        tier             = row['tier']
        short_put_strike = float(row['short_put_strike'])
        long_put_strike  = float(row['long_put_strike'])
        expiry_date      = datetime.strptime(row['expiry_date'], '%Y-%m-%d').date()
        entry_credit     = float(row['entry_credit'])
        contracts        = int(row['contracts'])
        position_id      = row['position_id']
        dte              = (expiry_date - date.today()).days

        current_price = _get_current_price(ticker)
        if current_price is None:
            print(f"[MONITOR] {position_id}: price fetch failed. Skipping.")
            continue

        print(f"[MONITOR] {position_id}: price=${current_price:.2f} | DTE={dte} | short={short_put_strike} | long={long_put_strike}")

        # ---- 1. Profit target hint (all tiers) ----
        if current_price > short_put_strike * 1.05 and dte <= 14:
            notify(
                subject=f"[{position_id}] 💰 PROFIT TARGET CHECK",
                body=(
                    f"Price ${current_price:.2f} is >5% above short put ${short_put_strike}\n"
                    f"DTE: {dte} | Entry credit: ${entry_credit}/share\n\n"
                    f"Check Tastytrade: if current debit <= "
                    f"${round(entry_credit*(1-EARLY_CLOSE_PROFIT_PCT),2)}/share, "
                    f"close now for 80% profit target."
                )
            )

        # ---- 2. Tier 2 emergency checks ----
        if tier == 'TIER2_WATCH':
            eval_result = evaluate_tier2_position(
                ticker, current_price, short_put_strike, expiry_date, entry_credit
            )
            action = eval_result['action']

            if action == 'EMERGENCY_CLOSE':
                notify(
                    subject=f"🚨 [{position_id}] EMERGENCY CLOSE — DEEP ITM",
                    body=(
                        f"DEEP ITM: price ${current_price:.2f} <= long put ${long_put_strike}\n"
                        f"DTE: {dte} — near maximum loss zone.\n\n"
                        f"ACTION: BTC (Buy to Close) on Tastytrade immediately.\n"
                        f"Entry credit: ${entry_credit}/share\n"
                        f"Max loss if expired: "
                        f"${round((SPREAD_WIDTH - entry_credit) * contracts * 100, 2)}"
                    )
                )

            elif action == 'ROLLOVER':
                roll = suggest_rollover(ticker, short_put_strike, entry_credit)
                notify(
                    subject=f"⚠️ [{position_id}] ROLLOVER TRIGGERED — Short Put ITM",
                    body=(
                        f"SHORT PUT ITM: price ${current_price:.2f} < short put ${short_put_strike}\n"
                        f"DTE: {dte}\n\n"
                        f"ROLLOVER PRIORITY:\n"
                        f"  1번: Net credit roll → lower strike + {roll['suggested_expiry']} "
                        f"(DTE {roll['suggested_dte']}) — receive credit\n"
                        f"  2번: Same strike ({short_put_strike}) + {roll['suggested_expiry']} — "
                        f"max debit ${roll['max_rollover_debit']}/share "
                        f"(${roll['max_rollover_debit_usd']}/contract)\n"
                        f"  Fallback: BTC immediately if neither viable"
                    )
                )

        # ---- 3. Routine review (all tiers) ----
        if dte <= BASE_DTE_ACTION:
            notify(
                subject=f"📋 [{position_id}] ROUTINE REVIEW — DTE {dte}",
                body=(
                    f"DTE {dte} — review close or rollover on Tastytrade.\n\n"
                    f"Price: ${current_price:.2f} | Short put: ${short_put_strike} | Long put: ${long_put_strike}\n"
                    f"Entry credit: ${entry_credit}/share\n"
                    f"If OTM and theta decayed ≥ 80%, close for profit."
                )
            )

    print("[MONITOR] Done.")


# ============ SUMMARY ============
def print_summary():
    df = _load()
    if df.empty:
        print("No positions on record.")
        return
    open_pos   = df[df['status'] == 'OPEN']
    closed_pos = df[df['status'].isin(['CLOSED', 'ROLLED'])]
    total_pnl  = pd.to_numeric(closed_pos['pnl_usd'], errors='coerce').sum()
    print(f"\n{'='*60}")
    print(f"POSITION SUMMARY — {datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'='*60}")
    print(f"Open     : {len(open_pos)}")
    print(f"Closed   : {len(closed_pos)}")
    print(f"Total P&L: ${total_pnl:,.2f}")
    print('='*60)
    if not open_pos.empty:
        print("\nOPEN POSITIONS:")
        print(open_pos[['position_id','tier','short_put_strike','long_put_strike',
                         'expiry_date','entry_credit','contracts','total_credit_usd']].to_string(index=False))
    if not closed_pos.empty:
        print("\nCLOSED/ROLLED:")
        print(closed_pos[['position_id','tier','close_reason','pnl_usd']].to_string(index=False))


if __name__ == '__main__':
    monitor_positions()
    print_summary()
