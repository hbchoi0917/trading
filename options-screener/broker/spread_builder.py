"""
Option chain lookup and vertical spread leg construction for Tastytrade.

Supports:
  - Put credit spreads  (sell higher strike, buy lower strike)
  - Call credit spreads (sell lower strike, buy higher strike)

Selects strikes by target delta proximity and DTE window.
"""

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Literal, Optional

from tastytrade.instruments import NestedOptionChain, NestedOptionChainExpiration
from tastytrade.order import (
    InstrumentType, Leg, NewOrder,
    OrderAction, OrderTimeInForce, OrderType, PriceEffect,
)
from tastytrade.session import Session

logger = logging.getLogger(__name__)

SpreadType = Literal["put_credit", "call_credit"]


@dataclass
class SpreadSpec:
    """Fully resolved spread ready to be placed as a NewOrder."""
    underlying:     str
    spread_type:    SpreadType
    short_symbol:   str
    long_symbol:    str
    short_strike:   float
    long_strike:    float
    expiration:     date
    dte:            int
    mid_credit:     Decimal
    max_risk:       Decimal        # |short_strike - long_strike| * 100
    short_delta:    Optional[float] = None

    def to_order(
        self,
        quantity: int = 1,
        limit_credit: Optional[Decimal] = None,
    ) -> NewOrder:
        """
        Build a NewOrder for this spread.

        limit_credit: credit to collect per spread (per contract, in dollars).
                      Defaults to mid_credit rounded to nearest $0.05.
        """
        credit = limit_credit or _round_to_nickel(self.mid_credit)
        if credit <= 0:
            raise ValueError(f"Spread credit must be positive, got {credit}")

        if self.spread_type == "put_credit":
            short_action = OrderAction.SELL_TO_OPEN
            long_action  = OrderAction.BUY_TO_OPEN
        else:
            short_action = OrderAction.SELL_TO_OPEN
            long_action  = OrderAction.BUY_TO_OPEN

        legs = [
            Leg(
                instrument_type=InstrumentType.EQUITY_OPTION,
                symbol=self.short_symbol,
                quantity=quantity,
                action=short_action,
            ),
            Leg(
                instrument_type=InstrumentType.EQUITY_OPTION,
                symbol=self.long_symbol,
                quantity=quantity,
                action=long_action,
            ),
        ]
        return NewOrder(
            time_in_force=OrderTimeInForce.DAY,
            order_type=OrderType.LIMIT,
            legs=legs,
            price=credit,
            price_effect=PriceEffect.CREDIT,
        )

    def summary(self) -> str:
        return (
            f"{self.underlying} {self.spread_type.upper()} "
            f"{self.expiration} "
            f"${self.short_strike:.0f}/{self.long_strike:.0f} "
            f"DTE={self.dte} "
            f"credit=${self.mid_credit:.2f} "
            f"max_risk=${self.max_risk:.0f}"
        )


# ── Main builder ─────────────────────────────────────────────────────────────

async def build_put_credit_spread(
    session:        Session,
    symbol:         str,
    target_delta:   float  = 0.15,
    dte_min:        int    = 21,
    dte_max:        int    = 35,
    spread_width:   float  = 10.0,
) -> Optional[SpreadSpec]:
    """
    Find the best put credit spread for `symbol`.

    Selects the expiration closest to 28 DTE within [dte_min, dte_max],
    then finds the strike whose absolute delta is closest to target_delta.

    Returns None if no suitable expiration/strike is found.
    """
    return await _build_spread(
        session, symbol, "put_credit", target_delta, dte_min, dte_max, spread_width
    )


async def build_call_credit_spread(
    session:        Session,
    symbol:         str,
    target_delta:   float  = 0.15,
    dte_min:        int    = 21,
    dte_max:        int    = 35,
    spread_width:   float  = 10.0,
) -> Optional[SpreadSpec]:
    """
    Find the best call credit spread for `symbol`.
    """
    return await _build_spread(
        session, symbol, "call_credit", target_delta, dte_min, dte_max, spread_width
    )


async def _build_spread(
    session:      Session,
    symbol:       str,
    spread_type:  SpreadType,
    target_delta: float,
    dte_min:      int,
    dte_max:      int,
    spread_width: float,
) -> Optional[SpreadSpec]:
    today = date.today()

    try:
        chains = await NestedOptionChain.get(session, symbol)
    except Exception as e:
        logger.warning(f"{symbol}: option chain fetch failed — {e}")
        return None

    if not chains:
        logger.warning(f"{symbol}: empty option chain")
        return None

    chain = chains[0]

    # Select expiration closest to 28 DTE within window
    target_dte = 28
    best_exp = _pick_expiration(chain.expirations, today, dte_min, dte_max, target_dte)
    if best_exp is None:
        logger.info(f"{symbol}: no expiration in [{dte_min}, {dte_max}] DTE")
        return None

    exp_date = best_exp.expiration_date
    dte      = (exp_date - today).days

    is_put  = spread_type == "put_credit"
    strikes = best_exp.puts if is_put else best_exp.calls
    if not strikes:
        logger.warning(f"{symbol}: no {'put' if is_put else 'call'} strikes for {exp_date}")
        return None

    # Find short leg: strike whose |delta| is closest to target_delta
    short_strike_data = min(
        strikes,
        key=lambda s: abs((s.delta or 0) - target_delta),
    )
    short_strike = float(short_strike_data.strike_price)

    if is_put:
        long_strike = short_strike - spread_width
    else:
        long_strike = short_strike + spread_width

    long_strike_data = _find_strike(strikes, long_strike)
    if long_strike_data is None:
        logger.info(f"{symbol}: long strike ${long_strike:.0f} not found in chain")
        return None

    # Estimate mid credit from bid/ask
    short_mid = _mid(short_strike_data.bid, short_strike_data.ask)
    long_mid  = _mid(long_strike_data.bid, long_strike_data.ask)
    if short_mid is None or long_mid is None:
        logger.info(f"{symbol}: missing bid/ask data")
        return None

    mid_credit = short_mid - long_mid
    if mid_credit <= Decimal("0"):
        logger.info(f"{symbol}: non-positive mid credit {mid_credit}")
        return None

    max_risk = Decimal(str(spread_width)) * 100 - mid_credit * 100

    return SpreadSpec(
        underlying   = symbol,
        spread_type  = spread_type,
        short_symbol = short_strike_data.symbol,
        long_symbol  = long_strike_data.symbol,
        short_strike = short_strike,
        long_strike  = float(long_strike_data.strike_price),
        expiration   = exp_date,
        dte          = dte,
        mid_credit   = mid_credit,
        max_risk     = max_risk,
        short_delta  = short_strike_data.delta,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _pick_expiration(
    expirations: list[NestedOptionChainExpiration],
    today: date,
    dte_min: int,
    dte_max: int,
    target_dte: int,
) -> Optional[NestedOptionChainExpiration]:
    candidates = [
        e for e in expirations
        if dte_min <= (e.expiration_date - today).days <= dte_max
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda e: abs((e.expiration_date - today).days - target_dte))


def _find_strike(strikes, target: float):
    best = min(strikes, key=lambda s: abs(float(s.strike_price) - target))
    if abs(float(best.strike_price) - target) <= 2.5:
        return best
    return None


def _mid(bid, ask) -> Optional[Decimal]:
    if bid is None or ask is None:
        return None
    try:
        return (Decimal(str(bid)) + Decimal(str(ask))) / 2
    except Exception:
        return None


def _round_to_nickel(v: Decimal) -> Decimal:
    """Round to nearest $0.05."""
    return (v / Decimal("0.05")).quantize(Decimal("1")) * Decimal("0.05")
