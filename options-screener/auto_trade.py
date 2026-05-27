"""
auto_trade.py — Daily automated trading pipeline.

Runs two phases:
  1. ENTRY  (at market open, ~9:35 ET):
       - Run options_premium_screener.py (or load today's signals CSV)
       - Apply monthly drawdown circuit breaker
       - Place put credit spread orders on Tastytrade

  2. MONITOR (at market open and mid-day):
       - Scan all open positions
       - Auto-close at profit target (80%) or DTE threshold (≤14)
       - Alert on rollover / emergency triggers

Usage:
    # Run screener then place orders
    python auto_trade.py entry [--account ACCT] [--live]

    # Monitor + auto-close existing positions
    python auto_trade.py monitor [--live]

    # Run screener + entry + monitor in sequence
    python auto_trade.py all [--account ACCT] [--live]

Environment:
    TT_PAPER_TRADING    — "true" for paper, "false" for live
    TT_DRY_RUN          — "true" to log without submitting (default: true)
    TT_ACCOUNT_NUMBERS  — comma-separated accounts (blank = all)
"""

import argparse
import asyncio
import csv
import logging
import os
import re
from datetime import date
from decimal import Decimal
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from broker.client import TastyClient
from broker.executor import (
    execute_entries_from_signals,
    monitor_and_close,
    check_monthly_drawdown,
    DRY_RUN_DEFAULT,
    MAX_CONCURRENT_POSITIONS,
    MAX_ENTRIES_PER_RUN,
    PORTFOLIO_EXPOSURE_LIMITS,
)
from notifications import (
    notify,
    notify_circuit_breaker,
    notify_monitor_summary,
    notify_daily_summary,
    notify_weekly_summary,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

POSITIONS_FILE = Path("positions.csv")

# NYSE market holidays — update annually
# Source: https://www.nyse.com/markets/hours-calendars
_NYSE_HOLIDAYS = {
    date(2026, 1, 1),    # New Year's Day
    date(2026, 1, 19),   # MLK Day
    date(2026, 2, 16),   # Presidents Day
    date(2026, 4, 3),    # Good Friday
    date(2026, 5, 25),   # Memorial Day
    date(2026, 7, 3),    # Independence Day (observed)
    date(2026, 9, 7),    # Labor Day
    date(2026, 11, 26),  # Thanksgiving
    date(2026, 12, 25),  # Christmas
    date(2027, 1, 1),    # New Year's Day
    date(2027, 1, 18),   # MLK Day
    date(2027, 2, 15),   # Presidents Day
    date(2027, 3, 26),   # Good Friday
    date(2027, 5, 31),   # Memorial Day
    date(2027, 7, 5),    # Independence Day (observed)
    date(2027, 9, 6),    # Labor Day
    date(2027, 11, 25),  # Thanksgiving
    date(2027, 12, 24),  # Christmas (observed)
}


def is_last_trading_day() -> bool:
    """Return True if today is the last trading day of the current week."""
    today = date.today()
    if today.weekday() >= 5 or today in _NYSE_HOLIDAYS:
        return False
    # Walk forward to find the next trading day
    next_day = today + timedelta(days=1)
    while next_day.weekday() >= 5 or next_day in _NYSE_HOLIDAYS:
        next_day += timedelta(days=1)
    # If next trading day is Monday, today was the last day of this trading week
    return next_day.weekday() == 0


# ── Signal loading & normalization ────────────────────────────────────────────

def _parse_delta_target(raw: str) -> float:
    """
    Parse Delta_Target from screener CSV.

    Screener outputs a range string like "0.10–0.18" or "0.08–0.13".
    We take the midpoint as the actual target delta.
    Falls back to 0.15 if parsing fails.
    """
    try:
        # Handle both en-dash (–) and regular hyphen (-), strip whitespace
        clean = re.sub(r"[–—]", "-", str(raw)).strip()
        parts = [float(p.strip()) for p in clean.split("-") if p.strip()]
        if len(parts) == 2:
            return round((parts[0] + parts[1]) / 2, 4)
        if len(parts) == 1:
            return parts[0]
    except (ValueError, AttributeError):
        pass
    return 0.15


def _normalize_signal(row: dict) -> dict:
    """
    Normalize a raw CSV row from the screener to the keys expected by executor.

    Screener column → executor key:
      Ticker          → ticker         (screener index written as first column)
      Signal_Strength → signal_strength
      Delta_Target    → target_delta   (parsed from "0.10–0.18" range string)
      Tier            → tier           (kept for logging)
      Expiry_Date     → expiry_date
      Expiry_DTE      → expiry_dte
    """
    # Lowercase all keys first for robust matching
    lowered = {k.lower(): v for k, v in row.items()}

    ticker = (
        lowered.get("ticker") or
        lowered.get("symbol") or
        ""
    ).strip().upper()

    try:
        signal_strength = int(float(lowered.get("signal_strength", 0)))
    except (ValueError, TypeError):
        signal_strength = 0

    target_delta = _parse_delta_target(lowered.get("delta_target", "0.15"))

    # Screener only generates put credit spread signals
    spread_type = "put_credit"

    return {
        "ticker":          ticker,
        "signal_strength": signal_strength,
        "target_delta":    target_delta,
        "spread_type":     spread_type,
        "tier":            lowered.get("tier", ""),
        "expiry_date":     lowered.get("expiry_date", ""),
        "expiry_dte":      lowered.get("expiry_dte", ""),
        "vix_regime":      lowered.get("vix_regime", ""),
        "cluster_risk":    lowered.get("cluster_risk", "").lower() == "true",
    }


def load_signals(signals_file: Path) -> list[dict]:
    """
    Load and normalize screener signals CSV.

    Handles the Fidelity-style index column ("Ticker") and converts all
    screener column names to the lowercase keys expected by executor.py.
    Filters out rows with empty/invalid tickers.
    """
    if not signals_file.exists():
        logger.warning(f"Signals file not found: {signals_file}")
        return []

    with open(signals_file, newline="") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    signals = [_normalize_signal(row) for row in raw_rows]
    signals = [s for s in signals if s["ticker"]]   # drop blank-ticker rows

    if signals:
        logger.info(
            f"Loaded {len(signals)} signals from {signals_file.name} | "
            f"tickers: {[s['ticker'] for s in signals]}"
        )
        cluster_risk = any(s["cluster_risk"] for s in signals)
        if cluster_risk:
            logger.warning(
                "CLUSTER RISK flagged by screener — "
                "multiple correlated tickers triggered simultaneously. "
                "Treat as correlated macro exposure; consider reducing quantity."
            )
    else:
        logger.info(f"No valid signals in {signals_file.name}")

    return signals


def get_signals_path() -> Path:
    today = date.today().strftime("%Y%m%d")
    return Path(f"signals_{today}.csv")


# ── Monthly P&L from position tracker ────────────────────────────────────────

def get_monthly_pnl_from_tracker(positions_file: Path = POSITIONS_FILE) -> Decimal:
    """
    Compute current month's realized P&L from positions.csv.

    Reads the position_tracker CSV, filters closed/rolled positions
    whose close_date falls in the current calendar month, and sums pnl_usd.
    Returns Decimal (negative = loss).
    """
    if not positions_file.exists():
        logger.info(f"positions.csv not found at {positions_file} — skipping drawdown check")
        return Decimal("0")

    try:
        import pandas as pd
        df = pd.read_csv(positions_file)
        if df.empty or "close_date" not in df.columns or "pnl_usd" not in df.columns:
            return Decimal("0")

        today = date.today()
        df["close_date"] = pd.to_datetime(df["close_date"], errors="coerce")
        monthly = df[
            (df["status"].isin(["CLOSED", "ROLLED"])) &
            (df["close_date"].dt.year  == today.year) &
            (df["close_date"].dt.month == today.month)
        ]
        pnl = pd.to_numeric(monthly["pnl_usd"], errors="coerce").sum()
        return Decimal(str(round(pnl, 2)))
    except Exception as e:
        logger.warning(f"Could not compute monthly P&L from positions file: {e}")
        return Decimal("0")


# ── Portfolio-level exposure check ───────────────────────────────────────────

def get_portfolio_exposure(ticker: str, positions_file: Path = POSITIONS_FILE) -> Decimal:
    """
    Return total current max risk for `ticker` across ALL open positions (all accounts).

    max_risk per position = (short_put_strike − long_put_strike) × contracts × 100
    Used to enforce PORTFOLIO_EXPOSURE_LIMITS before any new entry is placed.
    """
    if not positions_file.exists():
        return Decimal("0")
    try:
        import pandas as pd
        df = pd.read_csv(positions_file)
        if df.empty or "ticker" not in df.columns:
            return Decimal("0")
        open_pos = df[(df["ticker"] == ticker) & (df["status"] == "OPEN")]
        if open_pos.empty:
            return Decimal("0")
        total = (
            (open_pos["short_put_strike"] - open_pos["long_put_strike"])
            * open_pos["contracts"]
            * 100
        ).sum()
        return Decimal(str(round(float(total), 2)))
    except Exception as e:
        logger.warning(f"Portfolio exposure check failed for {ticker}: {e}")
        return Decimal("0")


# ── Summary data helpers ──────────────────────────────────────────────────────

def compute_daily_summary_data(positions_file: Path = POSITIONS_FILE) -> dict:
    """
    Gather positions.csv data for the daily summary email.

    Returns dict with keys:
      placed_today, closed_today, open_positions, mtd_pnl,
      expiring_soon (DTE ≤ 9), cap_warnings (>80% of limit)
    """
    empty = {
        "placed_today": 0, "closed_today": 0, "open_positions": [],
        "mtd_pnl": Decimal("0"), "expiring_soon": [], "cap_warnings": [],
    }
    if not positions_file.exists():
        return empty
    try:
        import pandas as pd
        df = pd.read_csv(positions_file)
        if df.empty:
            return empty

        today = date.today()

        # Parse dates once
        df["entry_date_d"] = pd.to_datetime(df.get("entry_date"), errors="coerce").dt.date
        df["close_date_d"] = pd.to_datetime(df.get("close_date"), errors="coerce").dt.date
        df["expiry_date_d"] = pd.to_datetime(df.get("expiry_date"), errors="coerce").dt.date

        placed_today = int((df["entry_date_d"] == today).sum())

        is_closed = df["status"].isin(["CLOSED", "ROLLED"])
        closed_today = int((is_closed & (df["close_date_d"] == today)).sum())

        # MTD P&L
        is_this_month = df["close_date_d"].apply(
            lambda d: bool(d and d.year == today.year and d.month == today.month)
        )
        mtd_raw = pd.to_numeric(df.loc[is_closed & is_this_month, "pnl_usd"], errors="coerce").sum()
        mtd_pnl = Decimal(str(round(float(mtd_raw), 2)))

        # Open positions
        open_df = df[df["status"] == "OPEN"].copy()
        open_positions = []
        expiring_soon  = []
        for _, row in open_df.iterrows():
            expiry = row["expiry_date_d"]
            dte    = (expiry - today).days if expiry else None
            pos = {
                "ticker":       str(row["ticker"]),
                "short_strike": float(row["short_put_strike"]),
                "long_strike":  float(row["long_put_strike"]),
                "expiry":       str(expiry) if expiry else "?",
                "dte":          dte,
                "contracts":    int(row["contracts"]),
            }
            open_positions.append(pos)
            if dte is not None and dte <= 9:
                expiring_soon.append(pos)

        # Portfolio cap warnings >80%
        cap_warnings = []
        for ticker, limit in PORTFOLIO_EXPOSURE_LIMITS.items():
            exposure = get_portfolio_exposure(ticker, positions_file)
            if exposure >= limit * Decimal("0.8"):
                pct = int(exposure / limit * 100)
                cap_warnings.append(f"{ticker} ${exposure:.0f}/${limit:.0f} ({pct}%)")

        return {
            "placed_today":   placed_today,
            "closed_today":   closed_today,
            "open_positions": open_positions,
            "mtd_pnl":        mtd_pnl,
            "expiring_soon":  expiring_soon,
            "cap_warnings":   cap_warnings,
        }
    except Exception as e:
        logger.warning(f"compute_daily_summary_data failed: {e}")
        return empty


def compute_weekly_summary_data(positions_file: Path = POSITIONS_FILE) -> dict:
    """
    Gather positions.csv data for the weekly summary email (last trading day only).

    Returns dict with keys:
      week_start, week_pnl, week_placed, week_closed,
      next_week_expiring, top_winner, top_loser
    """
    empty = {
        "week_start": str(date.today()), "week_pnl": Decimal("0"),
        "week_placed": 0, "week_closed": 0,
        "next_week_expiring": [], "top_winner": None, "top_loser": None,
    }
    if not positions_file.exists():
        return empty
    try:
        import pandas as pd
        df = pd.read_csv(positions_file)
        if df.empty:
            return empty

        today      = date.today()
        week_start = today - timedelta(days=today.weekday())   # Monday

        df["entry_date_d"] = pd.to_datetime(df.get("entry_date"), errors="coerce").dt.date
        df["close_date_d"] = pd.to_datetime(df.get("close_date"), errors="coerce").dt.date
        df["expiry_date_d"] = pd.to_datetime(df.get("expiry_date"), errors="coerce").dt.date

        week_placed = int((df["entry_date_d"] >= week_start).sum())

        is_closed  = df["status"].isin(["CLOSED", "ROLLED"])
        is_this_week = df["close_date_d"].apply(lambda d: bool(d and d >= week_start))
        week_closed_df = df[is_closed & is_this_week].copy()
        week_closed = len(week_closed_df)
        week_pnl_raw = pd.to_numeric(week_closed_df["pnl_usd"], errors="coerce").sum()
        week_pnl = Decimal(str(round(float(week_pnl_raw), 2)))

        # Next week expiring open positions
        next_week_start = week_start + timedelta(weeks=1)
        next_week_end   = next_week_start + timedelta(days=4)
        open_df = df[df["status"] == "OPEN"].copy()
        next_week_expiring = []
        for _, row in open_df.iterrows():
            expiry = row["expiry_date_d"]
            if expiry and next_week_start <= expiry <= next_week_end:
                next_week_expiring.append({
                    "ticker": str(row["ticker"]),
                    "expiry": str(expiry),
                    "dte":    (expiry - today).days,
                })

        # Top winner / loser this week by ticker P&L
        top_winner = top_loser = None
        if not week_closed_df.empty:
            week_closed_df["pnl_usd"] = pd.to_numeric(week_closed_df["pnl_usd"], errors="coerce")
            by_ticker = week_closed_df.groupby("ticker")["pnl_usd"].sum()
            if not by_ticker.empty:
                top_winner = {"ticker": by_ticker.idxmax(), "pnl": float(by_ticker.max())}
                top_loser  = {"ticker": by_ticker.idxmin(), "pnl": float(by_ticker.min())}

        return {
            "week_start":          str(week_start),
            "week_pnl":            week_pnl,
            "week_placed":         week_placed,
            "week_closed":         week_closed,
            "next_week_expiring":  next_week_expiring,
            "top_winner":          top_winner,
            "top_loser":           top_loser,
        }
    except Exception as e:
        logger.warning(f"compute_weekly_summary_data failed: {e}")
        return empty


# ── Account selection ─────────────────────────────────────────────────────────

def get_target_accounts(client: TastyClient) -> list[str]:
    env_accounts = os.environ.get("TT_ACCOUNT_NUMBERS", "").strip()
    if env_accounts:
        return [a.strip() for a in env_accounts.split(",") if a.strip()]
    return [a["account_number"] for a in client.list_accounts()]


# ── Phase 1: Entry ────────────────────────────────────────────────────────────

async def run_entry(client: TastyClient, dry_run: bool) -> None:
    accounts = get_target_accounts(client)
    signals_path = get_signals_path()
    signals = load_signals(signals_path)

    if not signals:
        logger.info("No signals today — skipping entry phase")
        return

    # Monthly drawdown circuit breaker (reads positions.csv)
    monthly_pnl = get_monthly_pnl_from_tracker()
    if monthly_pnl != Decimal("0"):
        logger.info(f"Current month realized P&L: ${monthly_pnl:,.2f}")
    if check_monthly_drawdown(monthly_pnl):
        logger.warning("Entry phase aborted — monthly drawdown limit hit.")
        from broker.executor import MONTHLY_DRAWDOWN_LIMIT
        notify_circuit_breaker(float(monthly_pnl), float(MONTHLY_DRAWDOWN_LIMIT))
        return

    # Portfolio-level single-ticker exposure cap (cross-account)
    # Filters signals BEFORE distributing to any account.
    filtered_signals = []
    for sig in signals:
        ticker = sig["ticker"]
        limit  = PORTFOLIO_EXPOSURE_LIMITS.get(ticker)
        if limit is not None:
            current = get_portfolio_exposure(ticker)
            if current >= limit:
                logger.warning(
                    f"PORTFOLIO CAP BLOCKED: {ticker} "
                    f"exposure ${current:.0f} ≥ limit ${limit:.0f} "
                    f"— no new entries in any account"
                )
                notify(
                    f"🚫 Portfolio Cap — {ticker}",
                    f"Current exposure: ${current:.0f}\n"
                    f"Limit: ${limit:.0f}\n"
                    f"New entries blocked across all accounts.",
                )
                continue
            if current > Decimal("0"):
                logger.info(
                    f"PORTFOLIO EXPOSURE: {ticker} "
                    f"${current:.0f} / ${limit:.0f} limit ({current/limit*100:.0f}% used)"
                )
        filtered_signals.append(sig)

    if len(filtered_signals) < len(signals):
        blocked = [s["ticker"] for s in signals if s not in filtered_signals]
        logger.warning(f"Portfolio cap removed signals: {blocked}")
    signals = filtered_signals

    if not signals:
        logger.info("All signals blocked by portfolio exposure caps — skipping entry phase")
        return

    for acct_num in accounts:
        # ── Concurrent position cap ───────────────────────────────────────────
        try:
            positions     = await client.get_positions(acct_num)
            open_spreads  = sum(1 for p in positions if p.quantity_direction == "Short")
        except Exception as e:
            logger.warning(f"[{acct_num}] Position fetch for cap check failed ({e}) — proceeding")
            open_spreads = 0

        available_slots = MAX_CONCURRENT_POSITIONS - open_spreads
        if available_slots <= 0:
            logger.info(
                f"[{acct_num}] Concurrent cap reached: "
                f"{open_spreads}/{MAX_CONCURRENT_POSITIONS} positions open — skipping entry"
            )
            continue

        entries_this_run = min(MAX_ENTRIES_PER_RUN, available_slots)
        logger.info(
            f"=== Entry: account {acct_num} | "
            f"{open_spreads}/{MAX_CONCURRENT_POSITIONS} open | "
            f"{available_slots} slots free | entering up to {entries_this_run} ==="
        )
        results = await execute_entries_from_signals(
            client=client,
            account_number=acct_num,
            signals=signals,
            dry_run=dry_run,
            max_entries=entries_this_run,
        )

        placed  = [r for r in results if r.success]
        skipped = [r for r in results if not r.success]
        logger.info(
            f"Entry summary: {len(placed)} placed, {len(skipped)} skipped"
        )
        for r in skipped:
            logger.info(f"  SKIP {r.symbol}: {r.reject_reason}")
        notify_monitor_summary(
            placed=len(placed),
            skipped=len(skipped),
            closed=0,
            monthly_pnl=float(monthly_pnl),
        )


# ── Phase 2: Monitor / Auto-close ────────────────────────────────────────────

async def run_monitor(client: TastyClient, dry_run: bool) -> None:
    logger.info("=== Monitor: scanning all positions ===")
    results = await monitor_and_close(client, dry_run=dry_run)

    if not results:
        logger.info("Monitor: no close triggers found")
        return

    closed = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    for r in results:
        status = "OK" if r.success else "FAILED"
        logger.info(
            f"  [{status}] {r.account} {r.symbol} "
            f"trigger={r.trigger} order_id={r.order_id}"
        )
    notify_monitor_summary(
        placed=0,
        skipped=len(failed),
        closed=len(closed),
        monthly_pnl=float(get_monthly_pnl_from_tracker()),
    )


# ── Phase 3: Daily / Weekly summary ──────────────────────────────────────────

def run_summary() -> None:
    """
    Send daily summary email, and weekly summary on the last trading day.
    Reads positions.csv only — no broker connection required.
    """
    from broker.executor import MONTHLY_DRAWDOWN_LIMIT

    data = compute_daily_summary_data()
    notify_daily_summary(
        placed=data["placed_today"],
        closed=data["closed_today"],
        open_positions=data["open_positions"],
        mtd_pnl=float(data["mtd_pnl"]),
        drawdown_limit=float(MONTHLY_DRAWDOWN_LIMIT),
        expiring_soon=data["expiring_soon"],
        cap_warnings=data["cap_warnings"],
    )
    logger.info("Daily summary sent.")

    if is_last_trading_day():
        wdata = compute_weekly_summary_data()
        notify_weekly_summary(
            week_start=wdata["week_start"],
            week_pnl=float(wdata["week_pnl"]),
            mtd_pnl=float(data["mtd_pnl"]),
            week_placed=wdata["week_placed"],
            week_closed=wdata["week_closed"],
            next_week_expiring=wdata["next_week_expiring"],
            top_winner=wdata["top_winner"],
            top_loser=wdata["top_loser"],
        )
        logger.info("Weekly summary sent (last trading day).")
    else:
        logger.info("Not the last trading day — weekly summary skipped.")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    dry_run = not args.live

    if dry_run:
        logger.info("Mode: DRY-RUN (orders logged but not submitted — pass --live to execute)")
    else:
        logger.warning("Mode: LIVE — real orders WILL be submitted to Tastytrade!")

    # summary reads only positions.csv — no broker connection needed
    if args.command == "summary":
        run_summary()
        return

    async with TastyClient() as client:
        acct_list = [a["account_number"] for a in client.list_accounts()]
        logger.info(f"Connected accounts: {acct_list}")

        if args.command in ("entry", "all"):
            await run_entry(client, dry_run)

        if args.command in ("monitor", "all"):
            await run_monitor(client, dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated options trading pipeline")
    parser.add_argument(
        "command",
        choices=["entry", "monitor", "all", "summary"],
        help="Phase to run: entry | monitor | all | summary (daily/weekly email)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Submit real orders to Tastytrade (default: dry-run)",
    )
    parser.add_argument(
        "--account",
        metavar="ACCT_NUMBER",
        help="Target a specific account (overrides TT_ACCOUNT_NUMBERS)",
    )
    args = parser.parse_args()

    if args.account:
        os.environ["TT_ACCOUNT_NUMBERS"] = args.account

    asyncio.run(main(args))
