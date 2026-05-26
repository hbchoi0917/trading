"""
Covered call alert screener — Fidelity positions, manual execution.

Scans CC_TICKERS daily and sends a Telegram/Gmail alert when conditions
are right to sell a covered call. No orders are placed automatically.

Entry conditions (opposite of put credit spreads):
  - Green day (close > prior close)
  - RSI 58–78  (mildly overbought; above 78 = momentum may continue)
  - BB position > 0.65  (near upper band)
  - Above SMA-200  (long-term uptrend intact)
  - No earnings within 7 days

Alert includes 3 suggested OTM call strikes (3% / 5% / 7% OTM)
with bid/ask so you can enter the order manually in Fidelity.

Usage:
    python covered_call_screener.py          # run once
    python covered_call_screener.py --test   # send a test notification
"""

import argparse
import logging
import math
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import yfinance as yf

from notifications import notify
from options_premium_screener import (
    _append_ta,
    get_earnings_date,
    get_vix,
    get_vix_regime,
    RSI_PERIOD,
    BB_PERIOD,
    ATR_PERIOD,
    EARNINGS_ENTRY_BUFFER_BEFORE,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Covered call positions (Fidelity) ────────────────────────────────────────

CC_TICKERS = ['VOO', 'QQQM', 'EWY', 'GOOGL', 'SCHD', 'DRAM']

# ── Entry thresholds ──────────────────────────────────────────────────────────

CC_RSI_MIN        = 58     # below this: not overbought enough
CC_RSI_MAX        = 78     # above this: strong momentum — don't cap upside yet
CC_BB_MIN         = 0.65   # BB position (0=lower, 1=upper band)
CC_EARNINGS_DAYS  = 7      # skip if earnings within this many days
CC_DTE_MIN        = 21
CC_DTE_MAX        = 35
CC_OTM_PCTS       = [0.03, 0.05, 0.07]   # suggested strike levels above spot
CC_NO_ASSIGN_TICKERS  = {'VOO', 'QQQM'}   # long-term core ETFs; premium only, assignment undesirable
CC_NO_ASSIGN_MIN_OTM  = 0.05      # minimum OTM for no-assign tickers (skip 3% strikes)


# ── Option chain helpers ──────────────────────────────────────────────────────

def _find_cc_expiry(ticker: str) -> tuple[str, int] | None:
    """Return (expiry_str, dte) for the nearest monthly in [DTE_MIN, DTE_MAX]."""
    today = datetime.today().date()
    window_start = today + timedelta(days=CC_DTE_MIN)
    window_end   = today + timedelta(days=CC_DTE_MAX)
    try:
        raw = yf.Ticker(ticker).options
        available = sorted([
            datetime.strptime(e, '%Y-%m-%d').date()
            for e in raw
            if window_start <= datetime.strptime(e, '%Y-%m-%d').date() <= window_end
        ])
    except Exception:
        return None
    if not available:
        return None

    # Prefer 3rd-Friday monthly; fall back to nearest weekly
    def _is_monthly(d):
        first = datetime(d.year, d.month, 1).date()
        first_fri = first + timedelta(days=(4 - first.weekday()) % 7)
        return d == first_fri + timedelta(weeks=2)

    for exp in available:
        if _is_monthly(exp):
            return str(exp), (exp - today).days
    exp = available[0]
    return str(exp), (exp - today).days


def _get_call_strikes(ticker: str, expiry_str: str, spot: float) -> list[dict]:
    """
    Return up to 3 OTM call strikes near CC_OTM_PCTS targets.
    Each entry: {strike, bid, ask, mid, iv, otm_pct}
    """
    try:
        chain = yf.Ticker(ticker).option_chain(expiry_str).calls
        if chain is None or chain.empty:
            return []
        chain = chain[chain['strike'] > spot].copy()
        chain['otm_pct'] = (chain['strike'] - spot) / spot
        if ticker in CC_NO_ASSIGN_TICKERS:
            chain = chain[chain['otm_pct'] >= CC_NO_ASSIGN_MIN_OTM]

        results = []
        seen_strikes = set()
        for target_pct in CC_OTM_PCTS:
            if chain.empty:
                break
            idx = (chain['otm_pct'] - target_pct).abs().idxmin()
            row = chain.loc[idx]
            strike = float(row['strike'])
            if strike in seen_strikes:
                continue
            seen_strikes.add(strike)

            bid = float(row['bid']) if not pd.isna(row['bid']) else 0.0
            ask = float(row['ask']) if not pd.isna(row['ask']) else 0.0
            iv  = float(row['impliedVolatility']) if not pd.isna(row['impliedVolatility']) else None
            mid = round((bid + ask) / 2, 2) if ask > 0 else None

            if ask <= 0:
                continue
            results.append({
                'strike':  strike,
                'bid':     round(bid, 2),
                'ask':     round(ask, 2),
                'mid':     mid,
                'iv':      round(iv * 100, 1) if iv else None,
                'otm_pct': round(float(row['otm_pct']) * 100, 1),
            })
        return results
    except Exception as e:
        logger.warning(f"[CC] {ticker}: option chain fetch failed — {e}")
        return []


# ── Screener ──────────────────────────────────────────────────────────────────

def screen_covered_calls(vix=None) -> dict[str, dict]:
    """
    Screen CC_TICKERS for covered call entry conditions.
    Returns a dict of qualifying tickers → signal data.
    """
    vix        = vix or get_vix()
    regime     = get_vix_regime(vix)
    today      = datetime.today().date()
    results    = {}

    logger.info("=" * 60)
    logger.info("COVERED CALL SCREENER")
    logger.info(f"Tickers : {', '.join(CC_TICKERS)}")
    logger.info(f"VIX     : {vix}  Regime: {regime}")
    logger.info("=" * 60)

    for ticker in CC_TICKERS:
        try:
            # ── Earnings guard ────────────────────────────────────────────────
            earnings_date = get_earnings_date(ticker)
            if earnings_date is not None:
                days_to_earnings = (earnings_date - today).days
                if 0 <= days_to_earnings <= CC_EARNINGS_DAYS:
                    logger.info(
                        f"[CC] {ticker}: earnings in {days_to_earnings}d ({earnings_date}) "
                        f"— skipping (buffer={CC_EARNINGS_DAYS}d)"
                    )
                    continue

            # ── Price data ────────────────────────────────────────────────────
            data = yf.download(ticker, period='1y', interval='1d', progress=False, group_by=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(0)
            data.columns = [c.capitalize() for c in data.columns]
            if data.empty or 'Close' not in data.columns or len(data) < 200:
                logger.warning(f"[CC] {ticker}: insufficient data")
                continue

            data['SMA_200'] = data['Close'].rolling(200).mean()
            _append_ta(data, rsi_length=RSI_PERIOD, atr_length=ATR_PERIOD)
            rsi_col = f'RSI_{RSI_PERIOD}'
            data['BB_middle']   = data['Close'].rolling(BB_PERIOD).mean()
            data['BB_std']      = data['Close'].rolling(BB_PERIOD).std()
            data['BB_upper']    = data['BB_middle'] + data['BB_std'] * 2
            data['BB_lower']    = data['BB_middle'] - data['BB_std'] * 2
            data['BB_position'] = (
                (data['Close'] - data['BB_lower']) /
                (data['BB_upper'] - data['BB_lower'])
            )
            if rsi_col not in data.columns:
                logger.warning(f"[CC] {ticker}: TA indicators missing")
                continue

            latest_close  = float(data['Close'].iloc[-1])
            prior_close   = float(data['Close'].iloc[-2])
            current_rsi   = float(data[rsi_col].iloc[-1])
            latest_bb_pos = float(data['BB_position'].iloc[-1])
            latest_sma200 = float(data['SMA_200'].iloc[-1])
            price_chg_pct = (latest_close - prior_close) / prior_close * 100

            # ── Entry filters ─────────────────────────────────────────────────
            is_green_day    = latest_close > prior_close
            is_overbought   = CC_RSI_MIN <= current_rsi <= CC_RSI_MAX
            is_near_upper   = latest_bb_pos > CC_BB_MIN
            is_uptrend      = latest_close > latest_sma200

            logger.info(
                f"[CC] {ticker}: green={is_green_day} ({price_chg_pct:+.1f}%) | "
                f"RSI={current_rsi:.1f} ({CC_RSI_MIN}–{CC_RSI_MAX}) | "
                f"BB={latest_bb_pos:.2f} (>{CC_BB_MIN}) | "
                f"SMA200={is_uptrend}"
            )

            if not (is_green_day and is_overbought and is_near_upper and is_uptrend):
                logger.info(f"[CC] {ticker}: conditions not met — skip")
                continue

            # ── Option chain ──────────────────────────────────────────────────
            expiry_info = _find_cc_expiry(ticker)
            if expiry_info is None:
                logger.info(f"[CC] {ticker}: no expiry in {CC_DTE_MIN}–{CC_DTE_MAX} DTE window")
                continue
            expiry_str, dte = expiry_info

            strikes = _get_call_strikes(ticker, expiry_str, latest_close)
            if not strikes:
                logger.info(f"[CC] {ticker}: no usable call strikes found")
                continue

            results[ticker] = {
                'ticker':        ticker,
                'price':         round(latest_close, 2),
                'price_chg_pct': round(price_chg_pct, 2),
                'rsi':           round(current_rsi, 1),
                'bb_position':   round(latest_bb_pos, 2),
                'sma200':        round(latest_sma200, 2),
                'expiry':        expiry_str,
                'dte':           dte,
                'earnings_date': str(earnings_date) if earnings_date else 'N/A',
                'strikes':       strikes,
                'vix':           vix,
                'vix_regime':    regime,
            }
            logger.info(
                f"✓ [CC] {ticker}: RSI {current_rsi:.1f} | BB {latest_bb_pos:.2f} | "
                f"+{price_chg_pct:.1f}% | expiry {expiry_str} ({dte} DTE) | "
                f"{len(strikes)} strikes suggested"
            )

        except Exception as e:
            logger.error(f"[CC] {ticker}: unexpected error — {e}")

    logger.info(f"CC screener done — {len(results)} signal(s): {list(results.keys())}")
    return results


# ── Notification ──────────────────────────────────────────────────────────────

def _format_strikes(strikes: list[dict]) -> str:
    lines = []
    for s in strikes:
        iv_str = f"  IV {s['iv']}%" if s['iv'] else ""
        lines.append(
            f"  ${s['strike']:.1f}C (+{s['otm_pct']}% OTM) "
            f"bid ${s['bid']:.2f} / ask ${s['ask']:.2f} "
            f"mid ${s['mid']:.2f}{iv_str}"
        )
    return "\n".join(lines)


def send_cc_alerts(results: dict[str, dict]) -> None:
    if not results:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M ET")
    blocks = []
    for data in results.values():
        strike_lines = _format_strikes(data['strikes'])
        block = (
            f"<b>📈 COVERED CALL — {data['ticker']}</b>\n"
            f"Price   : ${data['price']:.2f}  ({data['price_chg_pct']:+.1f}% today)\n"
            f"RSI     : {data['rsi']}  |  BB pos: {data['bb_position']:.2f}\n"
            f"Expiry  : {data['expiry']}  ({data['dte']} DTE)\n"
            f"Earnings: {data['earnings_date']}\n"
            f"VIX     : {data['vix']}  ({data['vix_regime']})\n"
            f"\nSuggested strikes (sell 1 call in Fidelity):\n"
            f"{strike_lines}"
        )
        blocks.append(block)

    full_message = "\n\n─────────────────────\n\n".join(blocks)
    subject = f"📈 CC Alert: {', '.join(results.keys())} — {timestamp}"
    notify(subject, full_message)
    logger.info(f"CC alert sent for: {list(results.keys())}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_covered_call_screener() -> dict[str, dict]:
    results = screen_covered_calls()
    if results:
        send_cc_alerts(results)
    else:
        logger.info("No covered call signals today.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Covered call alert screener")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Send a test notification without screening",
    )
    args = parser.parse_args()

    if args.test:
        notify(
            "📈 CC Screener — Test",
            "Covered call alert module is working correctly.\nThis is a test message.",
        )
        print("Test notification sent.")
    else:
        run_covered_call_screener()
