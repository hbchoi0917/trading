"""
auto_trade.py — Daily automated trading pipeline.

Runs two phases:
  1. ENTRY  (at market open, ~9:35 ET):
       - Load today's screener signals from signals_YYYYMMDD.csv
       - Apply risk filters
       - Place put credit spread orders on Tastytrade

  2. MONITOR (at market open and mid-day):
       - Scan all open positions
       - Auto-close at profit target (80%) or DTE threshold (≤14)
       - Alert on rollover / emergency triggers

Usage:
    # Entry phase (run after screener, at market open)
    python auto_trade.py entry [--account ACCT] [--dry-run] [--live]

    # Monitor phase (run at any time during market hours)
    python auto_trade.py monitor [--dry-run] [--live]

    # Both in sequence
    python auto_trade.py all [--account ACCT] [--dry-run] [--live]

Environment:
    TT_USERNAME, TT_PASSWORD  — Tastytrade credentials (see .env.example)
    TT_PAPER_TRADING          — "true" for paper, "false" for live
    TT_DRY_RUN                — "true" to log without submitting
    TT_ACCOUNT_NUMBERS        — comma-separated accounts (blank = all)
"""

import argparse
import asyncio
import csv
import logging
import os
import sys
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
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ── Signal loading ────────────────────────────────────────────────────────────

def load_signals(signals_file: Path) -> list[dict]:
    """Load screener signals CSV. Returns list of signal dicts."""
    if not signals_file.exists():
        logger.warning(f"Signals file not found: {signals_file}")
        return []
    with open(signals_file) as f:
        reader = csv.DictReader(f)
        signals = list(reader)
    logger.info(f"Loaded {len(signals)} signals from {signals_file.name}")
    return signals


def get_signals_path() -> Path:
    today = date.today().strftime("%Y%m%d")
    return Path(f"signals_{today}.csv")


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

    for acct_num in accounts:
        logger.info(f"=== Entry: account {acct_num} ===")

        # Check monthly drawdown circuit breaker
        # (In production, compute from transaction history; here we log a reminder)
        logger.info("TODO: compute monthly P&L and call check_monthly_drawdown()")

        results = await execute_entries_from_signals(
            client=client,
            account_number=acct_num,
            signals=signals,
            dry_run=dry_run,
            max_entries=5,
        )

        placed  = [r for r in results if r.success]
        skipped = [r for r in results if not r.success]
        logger.info(f"Entry summary: {len(placed)} placed, {len(skipped)} skipped")
        for r in skipped:
            logger.info(f"  SKIP {r.symbol}: {r.reject_reason}")


# ── Phase 2: Monitor / Auto-close ────────────────────────────────────────────

async def run_monitor(client: TastyClient, dry_run: bool) -> None:
    logger.info("=== Monitor: scanning all positions ===")
    results = await monitor_and_close(client, dry_run=dry_run)

    if not results:
        logger.info("Monitor: no close triggers found")
        return

    for r in results:
        status = "OK" if r.success else "FAILED"
        logger.info(f"  [{status}] {r.account} {r.symbol} trigger={r.trigger} order={r.order_id}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    dry_run = not args.live

    if dry_run:
        logger.info("Mode: DRY-RUN (set --live to submit real orders)")
    else:
        logger.warning("Mode: LIVE — real orders will be submitted!")

    async with TastyClient() as client:
        logger.info(f"Accounts: {[a['account_number'] for a in client.list_accounts()]}")

        if args.command in ("entry", "all"):
            await run_entry(client, dry_run)

        if args.command in ("monitor", "all"):
            await run_monitor(client, dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated options trading pipeline")
    parser.add_argument(
        "command",
        choices=["entry", "monitor", "all"],
        help="Phase to run",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Submit real orders (default: dry-run)",
    )
    parser.add_argument(
        "--account",
        metavar="ACCT_NUMBER",
        help="Override TT_ACCOUNT_NUMBERS env var",
    )
    args = parser.parse_args()

    if args.account:
        os.environ["TT_ACCOUNT_NUMBERS"] = args.account

    asyncio.run(main(args))
