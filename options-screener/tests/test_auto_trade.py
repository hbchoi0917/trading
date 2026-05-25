"""
Tests for auto_trade.py signal parsing utilities.

No network calls or broker connections needed — pure logic.
"""

import sys
import os
import csv
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auto_trade import _parse_delta_target, _normalize_signal, load_signals


class TestParseDeltaTarget(unittest.TestCase):

    def test_en_dash_range(self):
        self.assertAlmostEqual(_parse_delta_target("0.10–0.18"), 0.14, places=4)

    def test_hyphen_range(self):
        self.assertAlmostEqual(_parse_delta_target("0.08-0.13"), 0.105, places=4)

    def test_single_value(self):
        self.assertAlmostEqual(_parse_delta_target("0.15"), 0.15, places=4)

    def test_invalid_fallback(self):
        self.assertEqual(_parse_delta_target("N/A"), 0.15)

    def test_empty_string_fallback(self):
        self.assertEqual(_parse_delta_target(""), 0.15)

    def test_em_dash_range(self):
        self.assertAlmostEqual(_parse_delta_target("0.10—0.20"), 0.15, places=4)

    def test_whitespace_tolerance(self):
        self.assertAlmostEqual(_parse_delta_target("  0.10 - 0.18  "), 0.14, places=4)


class TestNormalizeSignal(unittest.TestCase):

    def _row(self, **kwargs):
        defaults = {
            "Ticker": "AAPL",
            "Signal_Strength": "75",
            "Delta_Target": "0.10–0.18",
            "Tier": "TIER1",
            "Expiry_Date": "2026-06-20",
            "Expiry_DTE": "26",
            "VIX_Regime": "normal",
            "Cluster_Risk": "False",
        }
        defaults.update(kwargs)
        return defaults

    def test_basic_normalization(self):
        result = _normalize_signal(self._row())
        self.assertEqual(result["ticker"], "AAPL")
        self.assertEqual(result["signal_strength"], 75)
        self.assertAlmostEqual(result["target_delta"], 0.14, places=4)
        self.assertEqual(result["spread_type"], "put_credit")
        self.assertEqual(result["tier"], "TIER1")

    def test_uppercase_ticker(self):
        result = _normalize_signal(self._row(Ticker="nvda"))
        self.assertEqual(result["ticker"], "NVDA")

    def test_signal_strength_float_string(self):
        result = _normalize_signal(self._row(Signal_Strength="80.5"))
        self.assertEqual(result["signal_strength"], 80)

    def test_invalid_signal_strength_defaults_zero(self):
        result = _normalize_signal(self._row(Signal_Strength=""))
        self.assertEqual(result["signal_strength"], 0)

    def test_cluster_risk_true(self):
        result = _normalize_signal(self._row(Cluster_Risk="True"))
        self.assertTrue(result["cluster_risk"])

    def test_cluster_risk_false(self):
        result = _normalize_signal(self._row(Cluster_Risk="false"))
        self.assertFalse(result["cluster_risk"])

    def test_symbol_key_alias(self):
        row = {k.lower(): v for k, v in self._row(Ticker="TSLA").items()}
        row.pop("ticker")
        row["symbol"] = "TSLA"
        result = _normalize_signal(row)
        self.assertEqual(result["ticker"], "TSLA")


class TestLoadSignals(unittest.TestCase):

    def _write_csv(self, rows, path: Path):
        if not rows:
            path.write_text("")
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def test_load_valid_signals(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "signals.csv"
            self._write_csv([
                {"Ticker": "AAPL", "Signal_Strength": "80", "Delta_Target": "0.10-0.18",
                 "Tier": "TIER1", "Expiry_Date": "2026-06-20", "Expiry_DTE": "26",
                 "VIX_Regime": "normal", "Cluster_Risk": "False"},
                {"Ticker": "NVDA", "Signal_Strength": "70", "Delta_Target": "0.08-0.13",
                 "Tier": "TIER1", "Expiry_Date": "2026-06-20", "Expiry_DTE": "26",
                 "VIX_Regime": "normal", "Cluster_Risk": "False"},
            ], p)
            signals = load_signals(p)
        self.assertEqual(len(signals), 2)
        self.assertEqual(signals[0]["ticker"], "AAPL")
        self.assertEqual(signals[1]["ticker"], "NVDA")

    def test_missing_file_returns_empty(self):
        signals = load_signals(Path("/tmp/nonexistent_signals_abc123.csv"))
        self.assertEqual(signals, [])

    def test_blank_ticker_rows_filtered(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "signals.csv"
            self._write_csv([
                {"Ticker": "", "Signal_Strength": "80", "Delta_Target": "0.15",
                 "Tier": "TIER1", "Expiry_Date": "2026-06-20", "Expiry_DTE": "26",
                 "VIX_Regime": "normal", "Cluster_Risk": "False"},
                {"Ticker": "TSLA", "Signal_Strength": "65", "Delta_Target": "0.15",
                 "Tier": "TIER1", "Expiry_Date": "2026-06-20", "Expiry_DTE": "26",
                 "VIX_Regime": "normal", "Cluster_Risk": "False"},
            ], p)
            signals = load_signals(p)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["ticker"], "TSLA")


if __name__ == "__main__":
    unittest.main()
