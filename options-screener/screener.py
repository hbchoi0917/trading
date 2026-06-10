#!/usr/bin/env python3
"""
Put Credit Spread Screener — Demo / Educational Baseline
=========================================================

⚠️  This is a simplified starting point, not a production-ready system.
    Customization is required before live trading:

    1. Watchlist     — replace the example tickers with names you've researched
                       and sized appropriately for your account.
    2. Thresholds    — RSI, BB, IV Rank cutoffs shown here are illustrative.
                       Back-test and tune them against your own trade history.
    3. Delta target  — adjust DELTA_TARGET to match your risk tolerance and
                       desired probability of profit.
    4. Broker API    — this script outputs a CSV of signals only.
                       Automated order placement requires integration with a
                       broker API (e.g. Tastytrade, IBKR, TD Ameritrade).
    5. Cloud server  — for fully unattended daily execution, deploy to a cloud
                       server (e.g. AWS EC2) and schedule via cron (Linux) or
                       Task Scheduler (Windows). The server must run in the
                       US/Eastern timezone to align with market hours.

Filters applied (all must pass):
  1. Price > SMA-200         (long-term uptrend)
  2. RSI (14) oversold       (hourly bars, daily fallback; VIX-regime adjusted)
  3. Bollinger Band position (price near lower band)
  4. ATR% > minimum         (adequate premium-generating volatility)
  5. Volume > 50-day avg    (confirm selling pressure, not just drift)
  6. IV dual-pass           (Pass 1: IV Rank >= threshold; Pass 2: IV/HV >= 1.0)
  7. No earnings blackout   (configurable window around announcement)

Methodology notes (see README.md for rationale):
  - Today's daily bar is patched with live intraday data before screening,
    so signals reflect current prices rather than yesterday's close.
  - RSI is computed on hourly bars (falls back to daily if intraday data
    is unavailable) — daily RSI lags badly on large intraday moves.
  - Broad-market index tickers (INDEX_TICKERS) skip the RSI/BB momentum
    gates: for premium sellers on cash-settled indices, IV Rank determines
    the edge; only trend confirmation + the IV dual-pass are required.

Output: signals_YYYYMMDD.csv + console table

Usage:
    pip install yfinance pandas
    python screener.py
"""

import math
import logging
from datetime import date, datetime, timedelta

import pandas as pd
import yfinance as yf

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(f"screener_{datetime.now():%Y%m%d}.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Watchlist ─────────────────────────────────────────────────────────────────
# Customize this list to your preferences.
# Stick to liquid names with tight option spreads (high open interest).
WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "AMZN",
    "META", "SPY", "IWM", "GS", "JPM",
]

_ETF_TICKERS = frozenset({"SPY", "QQQ", "IWM", "VOO", "GLD", "TLT"})

# Broad-market index-class tickers skip the RSI/BB momentum gates.
# RSI is a directional-trader tool; for premium sellers on cash-settled
# indices, IV Rank determines whether there is edge — only trend confirmation
# plus the IV dual-pass filter are required. (True European-style indices
# like SPX are the production use case; index ETFs stand in for the demo.)
INDEX_TICKERS = frozenset({"SPY", "IWM"})

# ── Screening parameters ──────────────────────────────────────────────────────
RSI_PERIOD  = 14
RSI_INTERVAL = "1h"  # RSI timeframe — hourly catches intraday oversold that daily misses
BB_PERIOD   = 20
ATR_PERIOD  = 14
SMA_PERIOD  = 200

DTE_MIN     = 21    # minimum days to expiry
DTE_MAX     = 45    # maximum days to expiry

ATR_MIN_PCT = 1.0   # skip if ATR% is below this — premium too thin
IV_RANK_MIN = 25    # IV dual-pass, Pass 1: skip if IV Rank below this
IV_HV_MIN   = 1.0   # IV dual-pass, Pass 2: skip if IV/HV below this

EARNINGS_BUFFER_DAYS = 3  # illustrative — tune the blackout window to your strategy

# Target delta range for the short put leg.
# Lower delta = further OTM = lower premium but higher probability of profit.
DELTA_TARGET = (0.15, 0.25)

# ── VIX regime thresholds ─────────────────────────────────────────────────────
# Screening thresholds shift with the volatility regime:
#   LOW      — premium thin; require deep oversold before entering
#   NORMAL   — standard thresholds
#   ELEVATED — fat premium; slightly relaxed, but still disciplined
#   HIGH     — tail risk elevated; tighten thresholds to avoid breakdown traps
VIX_REGIMES = [
    {"name": "LOW",      "vix_max": 16,  "rsi": 30, "bb": 0.30},
    {"name": "NORMAL",   "vix_max": 25,  "rsi": 35, "bb": 0.40},
    {"name": "ELEVATED", "vix_max": 35,  "rsi": 38, "bb": 0.45},
    {"name": "HIGH",     "vix_max": 999, "rsi": 32, "bb": 0.35},
]

# ── FOMC decision days ────────────────────────────────────────────────────────
# Rate decisions drop at 2 PM ET — close to the typical 3:30 PM entry window.
# Warn on these days; post-announcement reversals are common.
# Update each November: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
FOMC_DAYS = {
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29),
    date(2026, 6, 17), date(2026, 7, 29), date(2026, 9, 16),
    date(2026, 10, 28), date(2026, 12, 9),
}


# ── Technical indicators ──────────────────────────────────────────────────────
try:
    import importlib.util as _ilu
    _USE_PTA = _ilu.find_spec("pandas_ta") is not None
    if _USE_PTA:
        import pandas_ta as _pta  # noqa: F401
except Exception:
    _USE_PTA = False


def _wilder_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """RSI with Wilder's exponential smoothing — pure pandas, any bar interval."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    alpha = 1 / period
    ag = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    al = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    rs = ag / al.replace(0, float("inf"))
    return 100 - (100 / (1 + rs))


def _append_ta(df: pd.DataFrame) -> None:
    """Append RSI, ATR, MACD columns in-place. Falls back to pure pandas if pandas_ta is unavailable."""
    if _USE_PTA:
        df.ta.rsi(length=RSI_PERIOD, append=True)
        df.ta.atr(length=ATR_PERIOD, append=True)
        df.ta.macd(append=True)
        return

    df[f"RSI_{RSI_PERIOD}"] = _wilder_rsi(df["Close"])

    # ATR — Wilder's smoothing
    pc = df["Close"].shift(1)
    tr = pd.concat(
        [df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()],
        axis=1,
    ).max(axis=1)
    df[f"ATR_{ATR_PERIOD}"] = tr.ewm(
        alpha=1 / ATR_PERIOD, min_periods=ATR_PERIOD, adjust=False
    ).mean()

    # MACD (12/26/9 EMA)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    df["MACD_12_26_9"]  = macd
    df["MACDs_12_26_9"] = sig
    df["MACDh_12_26_9"] = macd - sig


def _safe_mid(row) -> float | None:
    try:
        bid = float(row.get("bid") or 0)
        ask = float(row.get("ask") or 0)
    except (TypeError, ValueError):
        return None
    if math.isnan(bid) or math.isnan(ask) or bid < 0 or ask <= 0 or ask < bid:
        return None
    return (bid + ask) / 2.0


def _normalize_yf(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)
    df.columns = [str(c).capitalize() for c in df.columns]
    return df


# ── Intraday data ─────────────────────────────────────────────────────────────
def get_hourly_rsi(ticker: str) -> float | None:
    """
    Latest RSI computed on hourly bars.

    On large down days, daily RSI can read neutral (~50) while hourly RSI
    correctly shows oversold — daily reflects yesterday's close, hourly
    reflects what the market is doing right now. Returns None if intraday
    data is unavailable; callers fall back to daily RSI.
    """
    try:
        h = _normalize_yf(yf.download(
            ticker, period="30d", interval=RSI_INTERVAL,
            progress=False, group_by=False,
        ))
        if h.empty or len(h) < RSI_PERIOD + 1:
            return None
        rsi = _wilder_rsi(h["Close"])
        val = float(rsi.iloc[-1])
        return val if not math.isnan(val) else None
    except Exception as e:
        log.warning(f"[hourly RSI] {ticker}: {e}")
        return None


def patch_intraday_bar(df: pd.DataFrame, ticker: str) -> None:
    """
    Update today's daily OHLCV bar in-place with live intraday data, so
    indicators reflect current prices rather than yesterday's close.
    Best-effort: any failure leaves the daily frame untouched.
    """
    try:
        intra = _normalize_yf(yf.download(
            ticker, period="1d", interval="5m",
            progress=False, group_by=False,
        ))
        if intra.empty:
            return
        bar = {
            "Open":   float(intra["Open"].iloc[0]),
            "High":   float(intra["High"].max()),
            "Low":    float(intra["Low"].min()),
            "Close":  float(intra["Close"].iloc[-1]),
            "Volume": float(intra["Volume"].sum()),
        }
        today = pd.Timestamp(datetime.today().date())
        last_day = pd.Timestamp(df.index[-1]).normalize()
        if last_day == today:
            for col, val in bar.items():
                df.loc[df.index[-1], col] = val
        else:
            df.loc[today] = bar
    except Exception as e:
        log.warning(f"[intraday patch] {ticker}: {e}")


# ── VIX ───────────────────────────────────────────────────────────────────────
def get_vix() -> float | None:
    try:
        d = yf.download("^VIX", period="5d", interval="1d", progress=False, group_by=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.droplevel(0)
        d.columns = [c.capitalize() for c in d.columns]
        return round(float(d["Close"].iloc[-1]), 2)
    except Exception as e:
        log.warning(f"VIX fetch failed: {e}")
        return None


def get_vix_regime(vix: float | None) -> dict:
    if vix is None or not isinstance(vix, (int, float)) or math.isnan(vix):
        return next(r for r in VIX_REGIMES if r["name"] == "NORMAL")
    for regime in VIX_REGIMES:
        if vix < regime["vix_max"]:
            return regime
    return VIX_REGIMES[-1]


# ── IV Rank + IV/HV ratio ─────────────────────────────────────────────────────
def compute_iv_rank(ticker: str) -> dict:
    """
    Estimate IV Rank and IV/HV ratio for a ticker.

    IV Rank (0–100): where current implied vol sits within its 52-week range.
    IV/HV ratio: implied vol relative to 30-day realized vol (historical volatility).
      > 1.0 means options are priced above recent realized moves — desirable for sellers.

    IV is approximated from ATM option mid-prices using Brenner-Subrahmanyam:
        IV ≈ (option_mid / spot) × sqrt(2π / T)

    Returns None values on data issues — callers treat None as fail-open
    (a data outage does not block a valid entry signal).
    """
    empty = {"iv_rank": None, "iv_hv_ratio": None, "hv_30": None, "skipped_reason": None}
    try:
        t = yf.Ticker(ticker)
        today = datetime.today().date()

        # Nearest expiry with at least 7 DTE
        raw_expiries = t.options or []
        valid = sorted([
            datetime.strptime(e, "%Y-%m-%d").date() for e in raw_expiries
            if (datetime.strptime(e, "%Y-%m-%d").date() - today).days >= 7
        ])
        if not valid:
            return {**empty, "skipped_reason": "no expiry >= 7 DTE"}
        near_exp = valid[0]
        dte = (near_exp - today).days

        # Spot price
        hist = yf.download(ticker, period="2d", interval="1d", progress=False, group_by=False)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.droplevel(0)
        hist.columns = [c.capitalize() for c in hist.columns]
        if hist.empty:
            return {**empty, "skipped_reason": "no price data"}
        spot = float(hist["Close"].iloc[-1])

        # ATM option mid-price
        chain = t.option_chain(str(near_exp))
        calls, puts = chain.calls, chain.puts
        if calls.empty or puts.empty:
            return {**empty, "skipped_reason": "empty option chain"}
        atm_strike = calls["strike"].iloc[(calls["strike"] - spot).abs().argsort().iloc[0]]
        call_row = calls[calls["strike"] == atm_strike].iloc[0]
        put_row = puts.iloc[(puts["strike"] - atm_strike).abs().argsort().iloc[0]]
        mids = [m for m in (_safe_mid(call_row), _safe_mid(put_row)) if m is not None]
        if not mids:
            return {**empty, "skipped_reason": "invalid bid/ask"}
        avg_mid = sum(mids) / len(mids)

        # Brenner-Subrahmanyam ATM IV approximation
        iv_cur = round((avg_mid / spot) * math.sqrt(2 * math.pi / (dte / 365.0)) * 100, 1)
        if math.isnan(iv_cur) or iv_cur <= 0:
            return {**empty, "skipped_reason": "invalid IV"}

        # 30-day realized vol (HV) and 52-week rvol series (IV history proxy)
        h1y = yf.download(ticker, period="1y", interval="1d", progress=False, group_by=False)
        if isinstance(h1y.columns, pd.MultiIndex):
            h1y.columns = h1y.columns.droplevel(0)
        h1y.columns = [c.capitalize() for c in h1y.columns]
        if len(h1y) < 31:
            return {**empty, "skipped_reason": "insufficient price history"}
        log_ret = h1y["Close"].pct_change().apply(
            lambda x: math.log(1 + x) if pd.notna(x) and x > -1 else 0.0
        )
        rvol = (log_ret.rolling(30).std() * math.sqrt(252) * 100).dropna()
        if rvol.empty:
            return {**empty, "skipped_reason": "rvol empty"}

        hv_30 = round(float(rvol.iloc[-1]), 1)
        iv_hv = round(iv_cur / hv_30, 2) if hv_30 > 0 else None
        lo, hi = float(rvol.min()), float(rvol.max())
        if hi <= lo:
            return {**empty, "hv_30": hv_30, "iv_hv_ratio": iv_hv, "skipped_reason": "zero IV range"}
        iv_rank = round(max(0.0, min(100.0, (iv_cur - lo) / (hi - lo) * 100)), 1)

        return {"iv_rank": iv_rank, "iv_hv_ratio": iv_hv, "hv_30": hv_30, "skipped_reason": None}

    except Exception as e:
        log.warning(f"[IV] {ticker}: {e}")
        return {**empty, "skipped_reason": str(e)}


# ── Earnings blackout ─────────────────────────────────────────────────────────
def get_earnings_date(ticker: str) -> date | None:
    if ticker in _ETF_TICKERS:
        return None
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is None or cal.empty:
            return None
        col = "Earnings Date"
        if col in cal.columns:
            val = cal[col].iloc[0]
        elif col in cal.index:
            val = cal.loc[col].iloc[0]
        else:
            return None
        return pd.Timestamp(val).date() if val and not pd.isna(val) else None
    except Exception:
        return None


def in_earnings_blackout(earnings_date: date | None, buffer_days: int = EARNINGS_BUFFER_DAYS) -> bool:
    if earnings_date is None:
        return False
    today = datetime.today().date()
    return (earnings_date - timedelta(days=buffer_days)) <= today <= earnings_date


# ── Expiry selection ──────────────────────────────────────────────────────────
def get_target_expiry(ticker: str, earnings_date: date | None = None):
    """Return (expiry_date, dte) within DTE_MIN–DTE_MAX, skipping earnings conflicts."""
    today = datetime.today().date()
    start, end = today + timedelta(days=DTE_MIN), today + timedelta(days=DTE_MAX)
    try:
        raw = yf.Ticker(ticker).options or []
        avail = sorted([
            datetime.strptime(e, "%Y-%m-%d").date() for e in raw
            if start <= datetime.strptime(e, "%Y-%m-%d").date() <= end
        ])
    except Exception:
        return None
    for exp in avail:
        if earnings_date and abs((exp - earnings_date).days) <= 5:
            continue
        return exp, (exp - today).days
    return None


# ── Per-ticker screening ──────────────────────────────────────────────────────
def screen_ticker(ticker: str, regime: dict) -> dict | None:
    rsi_thr  = regime["rsi"]
    bb_thr   = regime["bb"]
    is_index = ticker in INDEX_TICKERS
    try:
        df = _normalize_yf(yf.download(ticker, period="1y", interval="1d", progress=False, group_by=False))
        if df.empty or len(df) < SMA_PERIOD:
            log.warning(f"{ticker}: insufficient data")
            return None

        # Patch today's bar with live intraday data — signals should reflect
        # current prices, not yesterday's close.
        patch_intraday_bar(df, ticker)

        earnings = get_earnings_date(ticker)
        if in_earnings_blackout(earnings):
            log.info(f"{ticker}: earnings blackout ({earnings}), skipping")
            return None

        # Indicators
        df[f"SMA_{SMA_PERIOD}"] = df["Close"].rolling(SMA_PERIOD).mean()
        df["AVG_VOL_50"]        = df["Volume"].rolling(50).mean()
        _append_ta(df)
        df["BB_mid"] = df["Close"].rolling(BB_PERIOD).mean()
        df["BB_std"] = df["Close"].rolling(BB_PERIOD).std()
        df["BB_pos"] = (df["Close"] - (df["BB_mid"] - df["BB_std"] * 2)) / (df["BB_std"] * 4)

        rsi_col = f"RSI_{RSI_PERIOD}"
        atr_col = f"ATR_{ATR_PERIOD}"
        if not all(c in df.columns for c in [rsi_col, atr_col]):
            log.warning(f"{ticker}: TA columns missing")
            return None

        close   = float(df["Close"].iloc[-1])
        sma200  = float(df[f"SMA_{SMA_PERIOD}"].iloc[-1])
        bb_pos  = float(df["BB_pos"].iloc[-1])
        atr_pct = float(df[atr_col].iloc[-1]) / close * 100
        volume  = float(df["Volume"].iloc[-1])
        avg_vol = float(df["AVG_VOL_50"].iloc[-1])

        # RSI on hourly bars; fall back to daily if intraday data unavailable
        hourly_rsi = get_hourly_rsi(ticker)
        if hourly_rsi is not None:
            rsi, rsi_interval = hourly_rsi, RSI_INTERVAL
        else:
            rsi, rsi_interval = float(df[rsi_col].iloc[-1]), "1d"

        # Filter checks
        failures = []
        if close <= sma200:
            failures.append(f"price {close:.2f} <= SMA200 {sma200:.2f}")
        if is_index:
            # Index-class: trend + IV dual-pass only. RSI/BB momentum gates
            # are directional-trader tools — for premium sellers on indices,
            # IV Rank determines whether there is edge to capture.
            log.info(f"{ticker}: index-class — RSI/BB gates skipped")
        else:
            if rsi >= rsi_thr:
                failures.append(f"RSI({rsi_interval}) {rsi:.1f} >= {rsi_thr}")
            if bb_pos >= bb_thr:
                failures.append(f"BB {bb_pos:.2f} >= {bb_thr}")
        if atr_pct < ATR_MIN_PCT:
            failures.append(f"ATR% {atr_pct:.2f} < {ATR_MIN_PCT}")
        if volume < avg_vol:
            failures.append("vol below 50d avg")
        if failures:
            log.info(f"{ticker}: no signal — {'; '.join(failures)}")
            return None

        # IV dual-pass filter (fail-open: None values do not suppress the signal)
        # Pass 1 — is IV elevated vs. its own 52-week history? (IV Rank)
        # Pass 2 — is the market paying above recent realized vol? (IV/HV)
        iv = compute_iv_rank(ticker)
        if iv["iv_rank"] is not None and iv["iv_rank"] < IV_RANK_MIN:
            log.info(f"{ticker}: IV Rank {iv['iv_rank']} < {IV_RANK_MIN} — premium cheap, skip")
            return None
        if iv["iv_hv_ratio"] is not None and iv["iv_hv_ratio"] < IV_HV_MIN:
            log.info(f"{ticker}: IV/HV {iv['iv_hv_ratio']} < {IV_HV_MIN} — skip")
            return None

        expiry = get_target_expiry(ticker, earnings)

        return {
            "Ticker":        ticker,
            "Price":         round(close, 2),
            "RSI":           round(rsi, 1),
            "RSI_Interval":  rsi_interval,
            "RSI_Threshold": rsi_thr if not is_index else "N/A (index)",
            "BB_Position":   round(bb_pos, 2),
            "BB_Threshold":  bb_thr,
            "ATR_%":         round(atr_pct, 2),
            "SMA_200":       round(sma200, 2),
            "VIX_Regime":    regime["name"],
            "IV_Rank":       iv["iv_rank"],
            "IV_HV_Ratio":   iv["iv_hv_ratio"],
            "HV_30":         iv["hv_30"],
            "Delta_Target":  f"{DELTA_TARGET[0]}–{DELTA_TARGET[1]}",
            "Expiry_Date":   str(expiry[0]) if expiry else "N/A",
            "Expiry_DTE":    expiry[1] if expiry else None,
            "Earnings_Date": str(earnings) if earnings else "N/A",
            "Scan_Date":     datetime.now().strftime("%Y-%m-%d"),
        }

    except Exception as e:
        log.error(f"{ticker}: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────
def run_screener():
    log.info("=" * 60)
    log.info("PUT CREDIT SPREAD SCREENER")
    log.info("=" * 60)

    today = datetime.today().date()
    if today in FOMC_DAYS:
        log.warning("⚠️  FOMC decision day — review signals carefully before trading.")

    vix    = get_vix()
    regime = get_vix_regime(vix)
    log.info(
        f"VIX: {vix} → Regime: {regime['name']} | "
        f"RSI threshold: {regime['rsi']} | BB threshold: {regime['bb']}"
    )
    log.info("=" * 60)

    signals = []
    for ticker in WATCHLIST:
        log.info(f"--- {ticker} ---")
        result = screen_ticker(ticker, regime)
        if result:
            signals.append(result)
            log.info(
                f"✓ {ticker}: RSI {result['RSI']} | BB {result['BB_Position']} | "
                f"IV Rank {result['IV_Rank']} | Expiry {result['Expiry_Date']} (DTE {result['Expiry_DTE']})"
            )

    log.info("=" * 60)
    if signals:
        df = pd.DataFrame(signals).set_index("Ticker")
        output = f"signals_{datetime.now():%Y%m%d}.csv"
        df.to_csv(output)
        log.info(f"{len(signals)} signal(s) → {output}")
        print("\n" + df.to_string())
    else:
        log.info("No signals today.")

    return signals


if __name__ == "__main__":
    run_screener()
