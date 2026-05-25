"""
Tests for broker/spread_builder.py.

Mocks NestedOptionChain.get() so no live Tastytrade session is needed.
"""

import sys
import os
import asyncio
import unittest
from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from broker.spread_builder import (
    SpreadSpec,
    _pick_expiration,
    _find_strike,
    _mid,
    _round_to_nickel,
    build_put_credit_spread,
)


def _make_strike(symbol, strike, delta, bid, ask):
    s = MagicMock()
    s.symbol = symbol
    s.strike_price = Decimal(str(strike))
    s.delta = delta
    s.bid = bid
    s.ask = ask
    return s


def _make_expiration(exp_date, puts=None, calls=None):
    e = MagicMock()
    e.expiration_date = exp_date
    e.puts = puts or []
    e.calls = calls or []
    return e


class TestHelpers(unittest.TestCase):

    def test_round_to_nickel(self):
        self.assertEqual(_round_to_nickel(Decimal("1.23")), Decimal("1.25"))
        self.assertEqual(_round_to_nickel(Decimal("1.21")), Decimal("1.20"))
        self.assertEqual(_round_to_nickel(Decimal("0.50")), Decimal("0.50"))

    def test_mid(self):
        self.assertEqual(_mid(1.0, 1.5), Decimal("1.25"))

    def test_mid_none_inputs(self):
        self.assertIsNone(_mid(None, 1.5))
        self.assertIsNone(_mid(1.0, None))

    def test_find_strike_exact(self):
        strikes = [_make_strike("X", 100, -0.15, 1.0, 1.2),
                   _make_strike("Y", 110, -0.20, 1.5, 1.7)]
        result = _find_strike(strikes, 100.0)
        self.assertEqual(float(result.strike_price), 100.0)

    def test_find_strike_too_far_returns_none(self):
        strikes = [_make_strike("X", 100, -0.15, 1.0, 1.2)]
        result = _find_strike(strikes, 120.0)
        self.assertIsNone(result)

    def test_pick_expiration_closest_to_28(self):
        today = date.today()
        exps = [
            _make_expiration(today + timedelta(days=21)),
            _make_expiration(today + timedelta(days=28)),
            _make_expiration(today + timedelta(days=35)),
        ]
        best = _pick_expiration(exps, today, dte_min=21, dte_max=35, target_dte=28)
        self.assertEqual(best.expiration_date, today + timedelta(days=28))

    def test_pick_expiration_none_in_window(self):
        today = date.today()
        exps = [_make_expiration(today + timedelta(days=10))]
        result = _pick_expiration(exps, today, dte_min=21, dte_max=35, target_dte=28)
        self.assertIsNone(result)


class TestBuildPutCreditSpread(unittest.IsolatedAsyncioTestCase):

    def _make_chain(self, today):
        exp_date = today + timedelta(days=28)
        # Tastytrade SDK returns put deltas as positive values when queried
        puts = [
            _make_strike("AAPL_PUT_190", 190, 0.14, 1.20, 1.40),
            _make_strike("AAPL_PUT_180", 180, 0.08, 0.60, 0.80),
        ]
        exp = _make_expiration(exp_date, puts=puts)
        chain = MagicMock()
        chain.expirations = [exp]
        return chain

    async def test_returns_spread_spec(self):
        today = date.today()
        chain = self._make_chain(today)
        session = MagicMock()
        with patch("broker.spread_builder.NestedOptionChain.get", new=AsyncMock(return_value=[chain])):
            result = await build_put_credit_spread(session, "AAPL", target_delta=0.15)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, SpreadSpec)
        self.assertEqual(result.underlying, "AAPL")
        self.assertEqual(result.spread_type, "put_credit")
        self.assertEqual(result.short_strike, 190.0)
        self.assertEqual(result.long_strike, 180.0)
        self.assertGreater(result.mid_credit, Decimal("0"))

    async def test_returns_none_when_no_chain(self):
        session = MagicMock()
        with patch("broker.spread_builder.NestedOptionChain.get", new=AsyncMock(return_value=[])):
            result = await build_put_credit_spread(session, "AAPL")
        self.assertIsNone(result)

    async def test_returns_none_on_fetch_error(self):
        session = MagicMock()
        with patch("broker.spread_builder.NestedOptionChain.get",
                   new=AsyncMock(side_effect=Exception("network error"))):
            result = await build_put_credit_spread(session, "AAPL")
        self.assertIsNone(result)

    async def test_max_risk_calculation(self):
        today = date.today()
        chain = self._make_chain(today)
        session = MagicMock()
        with patch("broker.spread_builder.NestedOptionChain.get", new=AsyncMock(return_value=[chain])):
            result = await build_put_credit_spread(session, "AAPL", spread_width=10.0)
        # max_risk = (10 * 100) - (mid_credit * 100)
        expected_max = Decimal("1000") - result.mid_credit * 100
        self.assertEqual(result.max_risk, expected_max)


class TestSpreadSpecToOrder(unittest.TestCase):

    def _make_spec(self):
        return SpreadSpec(
            underlying="AAPL",
            spread_type="put_credit",
            short_symbol="AAPL_SHORT",
            long_symbol="AAPL_LONG",
            short_strike=190.0,
            long_strike=180.0,
            expiration=date.today() + timedelta(days=28),
            dte=28,
            mid_credit=Decimal("1.30"),
            max_risk=Decimal("870"),
            short_delta=-0.14,
        )

    def test_to_order_has_two_legs(self):
        spec = self._make_spec()
        order = spec.to_order(quantity=1)
        self.assertEqual(len(order.legs), 2)

    def test_to_order_quantity_multiplied(self):
        spec = self._make_spec()
        order = spec.to_order(quantity=3)
        for leg in order.legs:
            self.assertEqual(leg.quantity, 3)

    def test_to_order_credit_rounded_to_nickel(self):
        spec = self._make_spec()
        order = spec.to_order()
        self.assertEqual(order.price % Decimal("0.05"), Decimal("0"))


if __name__ == "__main__":
    unittest.main()
