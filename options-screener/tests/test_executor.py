"""
Tests for broker/executor.py.

All Tastytrade SDK calls are mocked so no live session is needed.
"""

import sys
import os
import unittest
from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from broker.executor import (
    check_monthly_drawdown,
    execute_entry,
    execute_entries_from_signals,
    MONTHLY_DRAWDOWN_LIMIT,
    MAX_RISK_PER_SPREAD,
    HIGH_BETA_TICKERS,
    HIGH_BETA_MAX_CONTRACTS,
)
from broker.spread_builder import SpreadSpec


def _make_spread(max_risk=Decimal("870"), mid_credit=Decimal("1.30")):
    return SpreadSpec(
        underlying="AAPL",
        spread_type="put_credit",
        short_symbol="AAPL_SHORT",
        long_symbol="AAPL_LONG",
        short_strike=190.0,
        long_strike=180.0,
        expiration=date.today() + timedelta(days=28),
        dte=28,
        mid_credit=mid_credit,
        max_risk=max_risk,
        short_delta=-0.14,
    )


def _make_client(spread=None, order_id=42):
    client = MagicMock()
    client.session = MagicMock()

    mock_response = MagicMock()
    mock_response.order.id = order_id
    mock_response.buying_power_effect = MagicMock()

    mock_account = MagicMock()
    mock_account.place_order = AsyncMock(return_value=mock_response)
    client.get_account = MagicMock(return_value=mock_account)

    return client


class TestCheckMonthlyDrawdown(unittest.TestCase):

    def test_below_limit_returns_true(self):
        self.assertTrue(check_monthly_drawdown(Decimal("-2500")))

    def test_at_limit_returns_true(self):
        self.assertTrue(check_monthly_drawdown(MONTHLY_DRAWDOWN_LIMIT))

    def test_above_limit_returns_false(self):
        self.assertFalse(check_monthly_drawdown(Decimal("-1999")))

    def test_positive_pnl_returns_false(self):
        self.assertFalse(check_monthly_drawdown(Decimal("500")))

    def test_zero_pnl_returns_false(self):
        self.assertFalse(check_monthly_drawdown(Decimal("0")))


class TestExecuteEntry(unittest.IsolatedAsyncioTestCase):

    async def test_successful_entry(self):
        spread = _make_spread()
        client = _make_client(spread)
        with patch("broker.executor.build_put_credit_spread", new=AsyncMock(return_value=spread)), \
             patch("notifications.notify_entry"):
            result = await execute_entry(client, "ACCT123", "AAPL", dry_run=True)
        self.assertTrue(result.success)
        self.assertEqual(result.symbol, "AAPL")
        self.assertEqual(result.order_id, 42)

    async def test_no_valid_spread_returns_failure(self):
        client = _make_client()
        with patch("broker.executor.build_put_credit_spread", new=AsyncMock(return_value=None)):
            result = await execute_entry(client, "ACCT123", "AAPL", dry_run=True)
        self.assertFalse(result.success)
        self.assertEqual(result.reject_reason, "no_valid_spread")

    async def test_max_risk_exceeded_returns_failure(self):
        oversized_spread = _make_spread(max_risk=MAX_RISK_PER_SPREAD + Decimal("1"))
        client = _make_client()
        with patch("broker.executor.build_put_credit_spread", new=AsyncMock(return_value=oversized_spread)):
            result = await execute_entry(client, "ACCT123", "AAPL", dry_run=True)
        self.assertFalse(result.success)
        self.assertIn("max_risk", result.reject_reason)

    async def test_high_beta_quantity_capped(self):
        spread = _make_spread()
        client = _make_client(spread)
        ticker = next(iter(HIGH_BETA_TICKERS))
        placed_quantities = []

        async def fake_place_order(session, order, dry_run):
            for leg in order.legs:
                placed_quantities.append(leg.quantity)
            r = MagicMock()
            r.order.id = 1
            r.buying_power_effect = MagicMock()
            return r

        client.get_account.return_value.place_order = fake_place_order

        with patch("broker.executor.build_put_credit_spread", new=AsyncMock(return_value=spread)), \
             patch("notifications.notify_entry"):
            result = await execute_entry(
                client, "ACCT123", ticker, quantity=10, dry_run=True
            )
        self.assertTrue(result.success)
        for q in placed_quantities:
            self.assertLessEqual(q, HIGH_BETA_MAX_CONTRACTS)

    async def test_order_exception_returns_failure(self):
        spread = _make_spread()
        client = _make_client(spread)
        client.get_account.return_value.place_order = AsyncMock(
            side_effect=Exception("insufficient buying power")
        )
        with patch("broker.executor.build_put_credit_spread", new=AsyncMock(return_value=spread)), \
             patch("notifications.notify_error"):
            result = await execute_entry(client, "ACCT123", "AAPL", dry_run=True)
        self.assertFalse(result.success)
        self.assertIn("insufficient buying power", result.reject_reason)


class TestExecuteEntriesFromSignals(unittest.IsolatedAsyncioTestCase):

    def _signals(self):
        return [
            {"ticker": "AAPL", "signal_strength": 90, "spread_type": "put_credit", "target_delta": 0.15},
            {"ticker": "NVDA", "signal_strength": 80, "spread_type": "put_credit", "target_delta": 0.12},
            {"ticker": "MSFT", "signal_strength": 70, "spread_type": "put_credit", "target_delta": 0.15},
        ]

    async def test_sorted_by_signal_strength(self):
        spread = _make_spread()
        client = _make_client(spread)
        placed_order = []

        original_execute = execute_entry

        async def tracking_execute(c, acct, symbol, **kwargs):
            placed_order.append(symbol)
            from broker.executor import EntryResult
            return EntryResult(symbol=symbol, success=True, order_id=1, spread=spread)

        with patch("broker.executor.execute_entry", side_effect=tracking_execute):
            results = await execute_entries_from_signals(
                client, "ACCT123", self._signals(), dry_run=True, max_entries=3
            )
        # First placed should be AAPL (strength 90), then NVDA (80), then MSFT (70)
        self.assertEqual(placed_order, ["AAPL", "NVDA", "MSFT"])

    async def test_max_entries_respected(self):
        spread = _make_spread()
        client = _make_client(spread)
        call_count = []

        async def counting_execute(c, acct, symbol, **kwargs):
            call_count.append(symbol)
            from broker.executor import EntryResult
            return EntryResult(symbol=symbol, success=True, order_id=1, spread=spread)

        with patch("broker.executor.execute_entry", side_effect=counting_execute):
            results = await execute_entries_from_signals(
                client, "ACCT123", self._signals(), dry_run=True, max_entries=2
            )
        self.assertEqual(len(call_count), 2)
        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
