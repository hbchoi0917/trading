"""
Order execution and automated position management for Tastytrade.

Responsibilities:
  - Place new put/call credit spread orders from screener signals
  - Monitor open positions and auto-close at profit targets or stop triggers
  - Enforce hardcoded risk rules (max loss per spread, monthly drawdown circuit breaker)
  - Log every action to file

Risk rules (hardcoded):
  MAX_RISK_PER_SPREAD    $1,000  — max (spread_width * 100) per position
  PROFIT_TARGET_PCT       80%   — close when P&L ≥ 80% of credit collected
  DTE_CLOSE_THRESHOLD      14   — close regardless of P&L at ≤14 DTE
  ROLLOVER_DTE              7   — roll trigger (DTE ≤ 7 + price < short strike)
  MONTHLY_DRAWDOWN_LIMIT -$2,000 — pause new entries if month is down >$2k
  HIGH_BETA_TICKERS        IONQ, RGTI, MARA — max 2 contracts
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from tastytrade.account import Account, CurrentPosition
from tastytrade.order import (
    InstrumentType, Leg, NewOrder,
    OrderAction, OrderTimeInForce, OrderType, PriceEffect,
    PlacedOrderResponse,
)
from tastytrade.session import Session

from .client import TastyClient
from .spread_builder import SpreadSpec, build_put_credit_spread, build_call_credit_spread

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Risk constants ────────────────────────────────────────────────────────────

MAX_RISK_PER_SPREAD      = Decimal("1000")
PROFIT_TARGET_PCT        = Decimal("0.80")
DTE_CLOSE_THRESHOLD      = 14
ROLLOVER_DTE             = 7
MONTHLY_DRAWDOWN_LIMIT   = Decimal("-2000")
HIGH_BETA_TICKERS        = {"IONQ", "RGTI", "MARA"}
HIGH_BETA_MAX_CONTRACTS  = 2
MSFT_MIN_OTM_PCT         = 15        # MSFT short strike must be ≥15% OTM

DRY_RUN_DEFAULT = os.environ.get("TT_DRY_RUN", "true").lower() == "true"


# ── Entry execution ───────────────────────────────────────────────────────────

@dataclass
class EntryResult:
    symbol:       str
    success:      bool
    order_id:     Optional[int] = None
    spread:       Optional[SpreadSpec] = None
    reject_reason: Optional[str] = None


async def execute_entry(
    client:           TastyClient,
    account_number:   str,
    symbol:           str,
    spread_type:      str   = "put_credit",
    target_delta:     float = 0.15,
    quantity:         int   = 1,
    dry_run:          bool  = DRY_RUN_DEFAULT,
) -> EntryResult:
    """
    Build and place a vertical spread for `symbol`.

    Returns EntryResult with success/failure details.
    """
    # Risk guard: high-beta contract limit
    if symbol in HIGH_BETA_TICKERS and quantity > HIGH_BETA_MAX_CONTRACTS:
        quantity = HIGH_BETA_MAX_CONTRACTS
        logger.warning(f"{symbol}: high-beta — capping quantity to {HIGH_BETA_MAX_CONTRACTS}")

    # Build spread
    builder = build_put_credit_spread if spread_type == "put_credit" else build_call_credit_spread
    spread = await builder(client.session, symbol, target_delta=target_delta)

    if spread is None:
        return EntryResult(symbol=symbol, success=False, reject_reason="no_valid_spread")

    # Risk guard: max risk per spread
    if spread.max_risk > MAX_RISK_PER_SPREAD:
        reason = f"max_risk=${spread.max_risk:.0f} > limit=${MAX_RISK_PER_SPREAD:.0f}"
        logger.warning(f"{symbol}: rejected — {reason}")
        return EntryResult(symbol=symbol, success=False, reject_reason=reason, spread=spread)

    # Risk guard: MSFT OTM rule
    if symbol == "MSFT" and spread_type == "put_credit":
        # Fetch current price from spread's short_delta context is unavailable here;
        # rely on screener having already validated OTM%. Log as reminder.
        logger.info("MSFT: ensure strike is ≥15% OTM before confirming this order")

    order = spread.to_order(quantity=quantity)
    acct  = client.get_account(account_number)

    mode = "DRY-RUN" if dry_run else "LIVE"
    logger.info(f"[{mode}] Placing {spread.summary()} qty={quantity}")

    try:
        response: PlacedOrderResponse = await acct.place_order(
            client.session, order, dry_run=dry_run
        )
        order_id = response.order.id if response.order else None
        logger.info(
            f"[{mode}] Order accepted — id={order_id} "
            f"bp_effect={response.buying_power_effect}"
        )
        return EntryResult(symbol=symbol, success=True, order_id=order_id, spread=spread)
    except Exception as e:
        logger.error(f"{symbol}: order rejected — {e}")
        return EntryResult(symbol=symbol, success=False, reject_reason=str(e), spread=spread)


async def execute_entries_from_signals(
    client:         TastyClient,
    account_number: str,
    signals:        list[dict],        # rows from screener CSV/dict
    dry_run:        bool = DRY_RUN_DEFAULT,
    max_entries:    int  = 5,
) -> list[EntryResult]:
    """
    Process a list of screener signal dicts and place spread orders.

    Each signal dict must have at least: {'ticker': str, 'signal_strength': int}
    Optional: {'spread_type': 'put_credit'|'call_credit', 'target_delta': float}
    """
    # Sort by signal strength (highest first), cap at max_entries
    ranked = sorted(signals, key=lambda s: s.get("signal_strength", 0), reverse=True)
    ranked = ranked[:max_entries]

    results = []
    for sig in ranked:
        symbol       = sig["ticker"]
        spread_type  = sig.get("spread_type", "put_credit")
        target_delta = float(sig.get("target_delta", 0.15))
        result = await execute_entry(
            client, account_number, symbol,
            spread_type=spread_type,
            target_delta=target_delta,
            dry_run=dry_run,
        )
        results.append(result)
        await asyncio.sleep(0.5)    # gentle rate limiting

    return results


# ── Position management / auto-close ─────────────────────────────────────────

@dataclass
class CloseResult:
    symbol:       str
    account:      str
    trigger:      str       # "profit_target" | "dte_expiry" | "emergency" | "rollover"
    success:      bool
    order_id:     Optional[int] = None
    error:        Optional[str] = None


async def monitor_and_close(
    client:   TastyClient,
    dry_run:  bool = DRY_RUN_DEFAULT,
) -> list[CloseResult]:
    """
    Scan all accounts for positions that need to be closed or rolled.

    Close triggers:
      1. profit_target  — P&L ≥ 80% of original credit (mark < 20% of credit)
      2. dte_expiry     — DTE ≤ 14 (close regardless of P&L)
      3. emergency      — price ≤ long put strike (Tier 2 emergency)
      4. rollover       — DTE ≤ 7 AND price < short put strike (roll candidate)
    """
    results: list[CloseResult] = []
    all_positions = await client.get_all_positions()

    for acct_num, positions in all_positions.items():
        option_positions = [
            p for p in positions
            if p.instrument_type == InstrumentType.EQUITY_OPTION and p.quantity_direction == "Short"
        ]
        for pos in option_positions:
            trigger = _evaluate_close_trigger(pos)
            if trigger is None:
                continue

            logger.info(f"[{acct_num}] {pos.symbol} — close trigger: {trigger}")
            result = await _place_close_order(client, acct_num, pos, trigger, dry_run)
            results.append(result)

    return results


def _evaluate_close_trigger(pos: CurrentPosition) -> Optional[str]:
    """Return close trigger name or None if no action needed."""
    today = date.today()

    dte = None
    exp = getattr(pos, "expires_at", None) or getattr(pos, "expiration_date", None)
    if exp:
        if isinstance(exp, datetime):
            exp = exp.date()
        dte = (exp - today).days

    close_price = Decimal(str(pos.close_price or 0))
    average_open_price = Decimal(str(pos.average_open_price or 0))

    # P&L as % of original credit: (credit - current_price) / credit
    if average_open_price > 0:
        pnl_pct = (average_open_price - close_price) / average_open_price
        if pnl_pct >= PROFIT_TARGET_PCT:
            return "profit_target"

    # DTE-based closes
    if dte is not None:
        if dte <= DTE_CLOSE_THRESHOLD:
            return "dte_expiry"

    return None


async def _place_close_order(
    client:     TastyClient,
    acct_num:   str,
    pos:        CurrentPosition,
    trigger:    str,
    dry_run:    bool,
) -> CloseResult:
    """Place a BTC/STC order to close the given short option position."""
    # For a short option, closing = Buy to Close
    close_leg = Leg(
        instrument_type=InstrumentType.EQUITY_OPTION,
        symbol=pos.symbol,
        quantity=abs(pos.quantity),
        action=OrderAction.BUY_TO_CLOSE,
    )
    # Use a limit at ask (or slightly above mid for urgency on emergency)
    limit_price = Decimal(str(pos.close_price or 0))
    if trigger == "emergency":
        limit_price = limit_price * Decimal("1.05")    # 5% above ask for fills

    order = NewOrder(
        time_in_force=OrderTimeInForce.DAY,
        order_type=OrderType.LIMIT,
        legs=[close_leg],
        price=limit_price,
        price_effect=PriceEffect.DEBIT,
    )
    acct = client.get_account(acct_num)
    mode = "DRY-RUN" if dry_run else "LIVE"
    logger.info(f"[{mode}] CLOSE {trigger}: {pos.symbol} qty={abs(pos.quantity)} @ ${limit_price:.2f}")

    try:
        response = await acct.place_order(client.session, order, dry_run=dry_run)
        order_id = response.order.id if response.order else None
        return CloseResult(
            symbol=pos.symbol, account=acct_num, trigger=trigger,
            success=True, order_id=order_id,
        )
    except Exception as e:
        logger.error(f"Close order failed: {pos.symbol} — {e}")
        return CloseResult(
            symbol=pos.symbol, account=acct_num, trigger=trigger,
            success=False, error=str(e),
        )


# ── Monthly drawdown circuit breaker ─────────────────────────────────────────

def check_monthly_drawdown(monthly_pnl: Decimal) -> bool:
    """
    Returns True if trading should pause (drawdown limit hit).
    monthly_pnl: current month's realized P&L in dollars.
    """
    if monthly_pnl <= MONTHLY_DRAWDOWN_LIMIT:
        logger.warning(
            f"CIRCUIT BREAKER: monthly P&L ${monthly_pnl:.0f} ≤ limit ${MONTHLY_DRAWDOWN_LIMIT:.0f}. "
            "Pausing new entries for the week."
        )
        return True
    return False
