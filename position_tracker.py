"""
position_tracker.py
-------------------
CSV-based position ledger for put credit spread positions.
Tracks open/closed positions, monitors daily for action triggers,
and suggests rollover parameters.

Designed for automated trading while traveling (timezone-independent).
All times stored in US/Central (Chicago). Notification stub ready for
email/SMS integration.

Usage:
    from position_tracker import add_position, monitor_positions

    # Log a new trade
    add_position(
        ticker='TSLA',
        tier='TIER2_WATCH',
        short_put_strike=200.0,
        long_put_strike=190.0,
        expiry_date='2026-05-16',
        entry_credit=1.25,     # per share = $125/contract
        contracts=1,
    )

    # Run daily monitor (call from cron/scheduler)
    monitor_positions()
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, date
from options_premium_screener import (
    evaluate_tier2_position,
    get_target_expiry,
    SPREAD_WIDTH,
    EARLY_CLOSE_PROFIT_PCT,
    BASE_DTE_ACTION,
    MAX_ROLLOVER_DEBIT_PCT,
    T2_ROLLOVER_DTE,
    T2_EMERGENCY_CLOSE_DTE,
)

# ============ CONFIG ============
POSITIONS_FILE = 'positions.csv'

COLUMNS = [
    'position_id',       # unique ID: TICKER_YYYYMMDD_shortStrike
    'ticker',
    'tier',              # TIER1_CORE | TIER2_WATCH
    'short_put_strike',  # higher strike (credit leg)
    'long_put_strike',   # lower strike (hedge leg) = short - SPREAD_WIDTH
    'expiry_date',       # YYYY-MM-DD
    'entry_date',
    'entry_credit',      # premium received per share (e.g. 1.25 = $125/contract)
    'contracts',
    'total_credit_usd',  # entry_credit * contracts * 100
    'status',            # OPEN | CLOSED | ROLLED
    'close_date',
    'close_debit',       # per share cost to close
    'pnl_usd',           # realized P&L in dollars
    'close_reason',      # PROFIT_TARGET | ROUTINE_REVIEW | ROLLOVER | EMERGENCY_CLOSE | EXPIRED
    'roll_to_position_id',  # if rolled, link to new position
    'notes',
]


# ============ HELPERS ============
def _load() -> pd.DataFrame:
    if os.path.exists(POSITIONS_FILE):
        df = pd.read_csv(POSITIONS_FILE, dtype=str)
        # Ensure all columns exist
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
    date_str = entry_date.replace('-', '')
    return f"{ticker}_{date_str}_{int(short_put_strike)}"


# ============ NOTIFICATION STUB ============
def notify(subject: str, body: str):
    """
    Notification hook. Replace print with email/SMS/Slack as needed.
    Example integrations:
        - smtplib for email
        - Twilio for SMS
        - requests to Slack webhook
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S CT')
    print(f"\n{'='*60}")
    print(f"[ALERT] {timestamp}")
    print(f"Subject: {subject}")
    print(body)
    print('='*60)


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

    entry_date    = datetime.now().strftime('%Y-%m-%d')
    position_id   = _make_position_id(ticker, entry_date, short_put_strike)
    total_credit  = round(entry_credit * contracts * 100, 2)

    new_row = {
        'position_id':       position_id,
        'ticker':            ticker,
        'tier':              tier,
        'short_put_strike':  str(short_put_strike),
        'long_put_strike':   str(long_put_strike),
        'expiry_date':       expiry_date,
        'entry_date':        entry_date,
        'entry_credit':      str(entry_credit),
        'contracts':         str(contracts),
        'total_credit_usd':  str(total_credit),
        'status':            'OPEN',
        'close_date':        '',
        'close_debit':       '',
        'pnl_usd':           '',
        'close_reason':      '',
        'roll_to_position_id': '',
        'notes':             notes,
    }

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    _save(df)
    print(f"[TRACKER] Position added: {position_id} | Credit: ${entry_credit}/share (${total_credit} total)")
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

    idx           = df[mask].index[0]
    entry_credit  = float(df.at[idx, 'entry_credit'])
    contracts     = int(df.at[idx, 'contracts'])
    pnl           = round((entry_credit - close_debit) * contracts * 100, 2)
    status        = 'ROLLED' if close_reason == 'ROLLOVER' else 'CLOSED'

    df.at[idx, 'status']               = status
    df.at[idx, 'close_date']           = datetime.now().strftime('%Y-%m-%d')
    df.at[idx, 'close_debit']          = str(close_debit)
    df.at[idx, 'pnl_usd']             = str(pnl)
    df.at[idx, 'close_reason']         = close_reason
    df.at[idx, 'roll_to_position_id']  = roll_to_position_id
    if notes:
        df.at[idx, 'notes'] = notes
    _save(df)
    print(f"[TRACKER] {position_id} → {status} | P&L: ${pnl} | Reason: {close_reason}")


# ============ ROLLOVER SUGGESTION ============
def suggest_rollover(ticker: str, short_put_strike: float, entry_credit: float):
    """
    Suggest rollover target expiry.
    Priority 1: net credit roll (lower strike, new expiry)
    Priority 2: same strike, new expiry — debit <= entry_credit * MAX_ROLLOVER_DEBIT_PCT

    Returns dict with suggested action details.
    """
    max_debit     = round(entry_credit * MAX_ROLLOVER_DEBIT_PCT, 2)
    max_debit_usd = round(max_debit * 100, 2)

    # Find next expiry beyond current DTE_MAX window
    from options_premium_screener import DTE_MAX
    from datetime import timedelta
    today       = date.today()
    roll_start  = today + timedelta(days=DTE_MAX + 1)
    roll_end    = today + timedelta(days=DTE_MAX + 30)

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
        'ticker':              ticker,
        'current_short_put':   short_put_strike,
        'current_long_put':    short_put_strike - SPREAD_WIDTH,
        'suggested_expiry':    str(target_expiry) if target_expiry else 'N/A',
        'suggested_dte':       dte_new,
        'priority_1':          f'Net credit roll: lower strike + {target_expiry} expiry (receive credit)',
        'priority_2':          f'Same strike ({short_put_strike}) + {target_expiry} expiry: max debit ${max_debit}/share (${max_debit_usd}/contract)',
        'max_rollover_debit':  max_debit,
        'max_rollover_debit_usd': max_debit_usd,
        'fallback':            'EMERGENCY_CLOSE if neither priority viable',
    }


# ============ DAILY MONITOR ============
def monitor_positions():
    """
    Daily position monitor. Run via cron/scheduler.
    Checks all OPEN positions for:
      - 80% profit target (all tiers)
      - DTE <= 4 routine review (all tiers)
      - Tier 2: Stage 1 rollover (DTE<=7, price < short_put)
      - Tier 2: Stage 2 emergency close (DTE<=7, price <= long_put)
    Sends notifications for any action required.
    """
    df = _load()
    open_positions = df[df['status'] == 'OPEN']

    if open_positions.empty:
        print("[MONITOR] No open positions.")
        return

    print(f"\n[MONITOR] Checking {len(open_positions)} open position(s) — {datetime.now().strftime('%Y-%m-%d %H:%M CT')}")

    for _, row in open_positions.iterrows():
        ticker           = row['ticker']
        tier             = row['tier']
        short_put_strike = float(row['short_put_strike'])
        long_put_strike  = float(row['long_put_strike'])
        expiry_date      = datetime.strptime(row['expiry_date'], '%Y-%m-%d').date()
        entry_credit     = float(row['entry_credit'])
        contracts        = int(row['contracts'])
        position_id      = row['position_id']
        today            = date.today()
        dte              = (expiry_date - today).days

        # Fetch current price
        current_price = _get_current_price(ticker)
        if current_price is None:
            print(f"[MONITOR] {position_id}: Could not fetch price. Skipping.")
            continue

        # Fetch current option premium (approximate via mid-price)
        # For full automation, replace with broker API quote
        # Here we estimate time value decay as a proxy
        current_premium_estimate = None  # placeholder for broker API

        print(f"[MONITOR] {position_id}: price=${current_price} | DTE={dte} | "
              f"short_put={short_put_strike} | long_put={long_put_strike}")

        # ---- 1. Profit target check (all tiers) ----
        # Requires live option quote from broker API to calculate precisely
        # Stub: flag for review when DTE is low and price is well above strikes
        if current_price > short_put_strike * 1.05 and dte <= 14:
            notify(
                subject=f"[{position_id}] PROFIT TARGET CHECK",
                body=(
                    f"Position: {position_id}\n"
                    f"Price ${current_price} is >5% above short put ${short_put_strike}\n"
                    f"DTE: {dte} | Entry credit: ${entry_credit}/share\n"
                    f"Check if current debit <= ${round(entry_credit*(1-EARLY_CLOSE_PROFIT_PCT),2)}/share "
                    f"(80% profit target). If so, close now."
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
                        f"Position: {position_id}\n"
                        f"DEEP ITM: price ${current_price} <= long put ${long_put_strike}\n"
                        f"DTE: {dte} — Near maximum loss zone.\n"
                        f"ACTION: BTC (Buy to Close) immediately.\n"
                        f"Entry credit: ${entry_credit}/share | Max loss: "
                        f"${round((SPREAD_WIDTH - entry_credit) * contracts * 100, 2)}"
                    )
                )

            elif action == 'ROLLOVER':
                roll_suggestion = suggest_rollover(ticker, short_put_strike, entry_credit)
                notify(
                    subject=f"⚠️ [{position_id}] ROLLOVER TRIGGERED — Short Put ITM",
                    body=(
                        f"Position: {position_id}\n"
                        f"SHORT PUT ITM: price ${current_price} < short put ${short_put_strike}\n"
                        f"DTE: {dte}\n\n"
                        f"ROLLOVER PRIORITY:\n"
                        f"  1st: Net credit roll to lower strike + {roll_suggestion['suggested_expiry']} "
                        f"(DTE {roll_suggestion['suggested_dte']}) — receive credit\n"
                        f"  2nd: Same strike ({short_put_strike}) + {roll_suggestion['suggested_expiry']} — "
                        f"max debit ${roll_suggestion['max_rollover_debit']}/share "
                        f"(${roll_suggestion['max_rollover_debit_usd']}/contract)\n"
                        f"  Fallback: EMERGENCY_CLOSE if neither viable\n"
                    )
                )

        # ---- 3. Routine review (all tiers, DTE <= BASE_DTE_ACTION) ----
        if dte <= BASE_DTE_ACTION:
            notify(
                subject=f"📋 [{position_id}] ROUTINE REVIEW — DTE {dte}",
                body=(
                    f"Position: {position_id} | Tier: {tier}\n"
                    f"DTE {dte} <= {BASE_DTE_ACTION}: review close or rollover.\n"
                    f"Price: ${current_price} | Short put: ${short_put_strike} | "
                    f"Long put: ${long_put_strike}\n"
                    f"Entry credit: ${entry_credit}/share\n"
                    f"If OTM and theta decayed sufficiently, close for profit."
                )
            )

    print("[MONITOR] Done.")


# ============ SUMMARY ============
def print_summary():
    """Print a summary of all positions."""
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
    print(f"Open positions   : {len(open_pos)}")
    print(f"Closed positions : {len(closed_pos)}")
    print(f"Realized P&L     : ${total_pnl:,.2f}")
    print(f"{'='*60}")
    if not open_pos.empty:
        print("\nOPEN POSITIONS:")
        print(open_pos[['position_id', 'tier', 'short_put_strike', 'long_put_strike',
                         'expiry_date', 'entry_credit', 'contracts', 'total_credit_usd']].to_string(index=False))
    if not closed_pos.empty:
        print("\nCLOSED/ROLLED POSITIONS:")
        print(closed_pos[['position_id', 'tier', 'close_reason', 'pnl_usd']].to_string(index=False))


if __name__ == '__main__':
    monitor_positions()
    print_summary()
