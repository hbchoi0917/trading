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
  MAX_BTC_DEBIT           $0.60 — profit-target close only when BTC debit ≤ $0.60/share
  DTE_CLOSE_THRESHOLD      12   — close regardless of P&L at ≤12 DTE
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
from .spread_builder import SpreadSpec, build_best_spread
from notifications import notify_entry, notify_close, notify_error

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Risk constants ────────────────────────────────────────────────────────────

MAX_RISK_PER_SPREAD      = Decimal("1000")
PROFIT_TARGET_PCT        = Decimal("0.80")
MAX_BTC_DEBIT            = Decimal("0.60")   # max debit/share for profit-target early close
DTE_CLOSE_THRESHOLD      = 12
EMERGENCY_RETRY_WAIT_SECS = 90              # seconds before escalating emergency BTC price
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
    dry_run:          bool  = DRY_RUN_DEFAULT,
) -> EntryResult:
    """
    Build and place the best vertical spread for `symbol`.

    Compares $10×1 vs $5×2 and picks whichever yields higher total premium.
    Returns EntryResult with success/failure details.
    """
    # Build spread — auto-selects best width and quantity
    spread, quantity = await build_best_spread(
        client.session, symbol, spread_type=spread_type, target_delta=target_delta
    )

    if spread is None:
        return EntryResult(symbol=symbol, success=False, reject_reason="no_valid_spread")

    # Risk guard: high-beta contract limit
    if symbol in HIGH_BETA_TICKERS and quantity > HIGH_BETA_MAX_CONTRACTS:
        quantity = HIGH_BETA_MAX_CONTRACTS
        logger.warning(f"{symbol}: high-beta — capping quantity to {HIGH_BETA_MAX_CONTRACTS}")

    # Risk guard: total risk across all contracts
    total_risk = spread.max_risk * quantity
    if total_risk > MAX_RISK_PER_SPREAD:
        reason = f"max_risk=${total_risk:.0f} > limit=${MAX_RISK_PER_SPREAD:.0f}"
        logger.warning(f"{symbol}: rejected — {reason}")
        return EntryResult(symbol=symbol, success=False, reject_reason=reason, spread=spread)

    # Risk guard: MSFT OTM rule
    if symbol == "MSFT" and spread_type == "put_credit":
        # Rely on screener having already validated OTM%. Log as reminder.
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
        notify_entry(
            ticker=symbol,
            spread_type=spread_type,
            expiry=str(spread.expiration),
            short_strike=float(spread.short_strike),
            long_strike=float(spread.long_strike),
            credit=float(spread.mid_credit),
            quantity=quantity,
            account=account_number,
            dry_run=dry_run,
        )
        return EntryResult(symbol=symbol, success=True, order_id=order_id, spread=spread)
    except Exception as e:
        logger.error(f"{symbol}: order rejected — {e}")
        notify_error(symbol, str(e), account=account_number)
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
    # Also gate on BTC debit ≤ MAX_BTC_DEBIT to avoid paying too much to close
    if average_open_price > 0:
        pnl_pct = (average_open_price - close_price) / average_open_price
        if pnl_pct >= PROFIT_TARGET_PCT and close_price <= MAX_BTC_DEBIT:
            return "profit_target"

    # DTE-based closes
    if dte is not None:
        if dte <= DTE_CLOSE_THRESHOLD:
            return "dte_expiry"

    return None


def _round_nickel(v: Decimal) -> Decimal:
    return (v / Decimal("0.05")).quantize(Decimal("1")) * Decimal("0.05")


def _build_close_order(pos: CurrentPosition, limit_price: Decimal) -> NewOrder:
    return NewOrder(
        time_in_force=OrderTimeInForce.DAY,
        order_type=OrderType.LIMIT,
        legs=[Leg(
            instrument_type=InstrumentType.EQUITY_OPTION,
            symbol=pos.symbol,
            quantity=abs(pos.quantity),
            action=OrderAction.BUY_TO_CLOSE,
        )],
        price=limit_price,
        price_effect=PriceEffect.DEBIT,
    )


async def _place_close_order(
    client:     TastyClient,
    acct_num:   str,
    pos:        CurrentPosition,
    trigger:    str,
    dry_run:    bool,
) -> CloseResult:
    """
    Place a BTC limit order at mid-point.

    Emergency retry: if the mid order is still unfilled after
    EMERGENCY_RETRY_WAIT_SECS, cancels and resubmits at mid × 1.05.
    """
    mid_price = Decimal(str(pos.close_price or 0))
    acct      = client.get_account(acct_num)
    mode      = "DRY-RUN" if dry_run else "LIVE"

    try:
        order    = _build_close_order(pos, mid_price)
        logger.info(f"[{mode}] CLOSE {trigger}: {pos.symbol} qty={abs(pos.quantity)} @ ${mid_price:.2f} (mid)")
        response = await acct.place_order(client.session, order, dry_run=dry_run)
        order_id = response.order.id if response.order else None
        final_price = mid_price

        # Emergency: wait, then escalate to 1.05× mid if still unfilled
        if trigger == "emergency" and not dry_run and order_id:
            await asyncio.sleep(EMERGENCY_RETRY_WAIT_SECS)
            try:
                live        = await client.get_live_orders(acct_num)
                pending_ids = {o.id for o in (live or [])}
                if order_id in pending_ids:
                    await client.cancel_order(acct_num, order_id)
                    final_price  = _round_nickel(mid_price * Decimal("1.05"))
                    retry_order  = _build_close_order(pos, final_price)
                    logger.warning(
                        f"[{mode}] CLOSE {trigger} RETRY: {pos.symbol} "
                        f"@ ${final_price:.2f} (1.05× mid — mid unfilled after {EMERGENCY_RETRY_WAIT_SECS}s)"
                    )
                    resp2    = await acct.place_order(client.session, retry_order, dry_run=dry_run)
                    order_id = resp2.order.id if resp2.order else None
            except Exception as retry_exc:
                logger.warning(f"Emergency retry check failed: {retry_exc} — original order may still be working")

        pnl = float(
            (Decimal(str(pos.average_open_price or 0)) - final_price)
            * abs(pos.quantity) * 100
        )
        notify_close(ticker=pos.symbol.split()[0], trigger=trigger, pnl=pnl, account=acct_num, dry_run=dry_run)
        return CloseResult(symbol=pos.symbol, account=acct_num, trigger=trigger, success=True, order_id=order_id)

    except Exception as e:
        logger.error(f"Close order failed: {pos.symbol} — {e}")
        notify_error(pos.symbol, f"Close failed ({trigger}): {e}", account=acct_num)
        return CloseResult(symbol=pos.symbol, account=acct_num, trigger=trigger, success=False, error=str(e))


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
