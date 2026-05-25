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

# Minimum mid credit (per share) for specific tickers.
# COST is exempt from is_red_day in the screener; compensate by requiring
# at least $1.00/share ($100/contract) so green-day entries are only taken
# when premium is meaningfully fat.
TICKER_MIN_CREDIT: dict[str, Decimal] = {
    'COST': Decimal('1.00'),
}

# Minimum mid credit (per share) by spread width — applied globally before
# the ticker override check. Prevents entering thin-premium spreads where
# credit-to-risk ratio doesn't justify the position.
#   $10-wide: $1.30/share = $130/contract (13% of spread width)
#    $5-wide: $0.95/share =  $95/contract (19% of spread width)
SPREAD_WIDTH_MIN_CREDIT: dict[float, Decimal] = {
    10.0: Decimal('1.30'),
     5.0: Decimal('0.95'),
}

# ── Quad witching delta guard ─────────────────────────────────────────────────
# When the selected expiry falls on a quad witching Friday, reduce target_delta
# so the short strike lands further OTM — provides more cushion against the
# exaggerated intraday moves and wide spreads typical on QW day.
_QW_MONTHS     = {3, 6, 9, 12}
QW_DELTA_SCALE = 0.75   # multiply target_delta by this on QW expiry
QW_DELTA_FLOOR = 0.08   # minimum delta after scaling


def _is_quad_witching_expiry(exp_date: date) -> bool:
    """True if exp_date is the 3rd Friday of Mar/Jun/Sep/Dec."""
    if exp_date.month not in _QW_MONTHS or exp_date.weekday() != 4:
        return False
    first = date(exp_date.year, exp_date.month, 1)
    first_fri = first + timedelta(days=(4 - first.weekday()) % 7)
    return exp_date == first_fri + timedelta(weeks=2)


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
    max_credit:     Optional[Decimal] = None   # natural credit = short_ask - long_bid

    def to_order(
        self,
        quantity: int = 1,
        limit_credit: Optional[Decimal] = None,
    ) -> NewOrder:
        """
        Build a NewOrder for this spread.

        limit_credit: override credit (per contract). When omitted, uses the
                      midpoint of mid_credit and max_credit (natural credit),
                      rounded to $0.05 — slightly above mid to improve fill odds.
        """
        if limit_credit is not None:
            credit = limit_credit
        elif self.max_credit is not None:
            credit = _round_to_nickel((self.mid_credit + self.max_credit) / 2)
        else:
            credit = _round_to_nickel(self.mid_credit)
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
    dte_min:        int    = 28,
    dte_max:        int    = 45,
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
    dte_min:        int    = 28,
    dte_max:        int    = 45,
    spread_width:   float  = 10.0,
) -> Optional[SpreadSpec]:
    """
    Find the best call credit spread for `symbol`.
    """
    return await _build_spread(
        session, symbol, "call_credit", target_delta, dte_min, dte_max, spread_width
    )


async def build_best_spread(
    session:      Session,
    symbol:       str,
    spread_type:  str   = "put_credit",
    target_delta: float = 0.15,
    dte_min:      int   = 28,
    dte_max:      int   = 45,
) -> tuple[Optional[SpreadSpec], int]:
    """
    Compare $10 spread × 1 contract vs $5 spread × 2 contracts.
    Return (best_spread, quantity) whichever yields higher total premium.
    Total premium = mid_credit × quantity × 100.
    Both scenarios cap at $1,000 max risk.
    """
    builder = build_put_credit_spread if spread_type == "put_credit" else build_call_credit_spread

    spread_10 = await builder(session, symbol, target_delta, dte_min, dte_max, spread_width=10.0)
    spread_5  = await builder(session, symbol, target_delta, dte_min, dte_max, spread_width=5.0)

    total_10 = spread_10.mid_credit * 100 if spread_10 else Decimal("0")
    total_5  = spread_5.mid_credit  * 200 if spread_5  else Decimal("0")

    logger.info(
        f"{symbol}: $10×1 total=${total_10:.2f}  $5×2 total=${total_5:.2f}"
        f"  → {'$5×2' if spread_5 and total_5 > total_10 else '$10×1'}"
    )

    if spread_5 and total_5 > total_10:
        return spread_5, 2
    if spread_10:
        return spread_10, 1
    return None, 0


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

    # Quad witching expiry: go further OTM to absorb elevated gamma/noise
    if _is_quad_witching_expiry(exp_date):
        orig_delta   = target_delta
        target_delta = max(QW_DELTA_FLOOR, round(target_delta * QW_DELTA_SCALE, 3))
        logger.info(f"{symbol}: QW expiry {exp_date} — delta {orig_delta:.2f}→{target_delta:.3f}")

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

    # Width-based minimum credit check (global quality gate)
    width_min = SPREAD_WIDTH_MIN_CREDIT.get(float(spread_width))
    if width_min is not None and mid_credit < width_min:
        logger.info(
            f"{symbol}: ${spread_width:.0f}-wide mid ${mid_credit:.2f} "
            f"< width minimum ${width_min:.2f} — skipping"
        )
        return None

    # Per-ticker minimum credit guard (e.g. COST bypasses red-day filter,
    # so require fat premium to compensate for green-day entry risk)
    min_credit = TICKER_MIN_CREDIT.get(symbol)
    if min_credit is not None and mid_credit < min_credit:
        logger.info(
            f"{symbol}: mid credit ${mid_credit:.2f} < required minimum ${min_credit:.2f} — skipping"
        )
        return None

    # Natural credit = max receivable (short_ask − long_bid); used to price STO above mid
    try:
        short_ask = Decimal(str(short_strike_data.ask))
        long_bid  = Decimal(str(long_strike_data.bid))
        max_credit: Optional[Decimal] = short_ask - long_bid if short_ask > long_bid else None
    except Exception:
        max_credit = None

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
        max_credit   = max_credit,
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
