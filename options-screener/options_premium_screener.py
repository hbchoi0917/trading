import math
import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timedelta

try:
    import pandas_ta as ta
    if ta is None:
        raise ImportError("pandas_ta imported as None")
except ImportError:
    raise ImportError(
        "pandas_ta is required. Install it with: pip install pandas_ta"
    )

# ============ LOGGING SETUP ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'screener_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============ CURATED TICKER LISTS ============
# SPX (^GSPC) handled separately via screen_spx()
TIER1_CORE = [
    'COST',   # Costco
    'NVDA',   # NVIDIA
    'IWM',    # Russell 2000 ETF
    'GOOGL',  # Alphabet Class A
]

TIER2_WATCHLIST = [
    'MSFT',
    'AAPL',
    'AMZN',
    'META',
    'AVGO',
    'CRWD',
    'PLTR',
    'AMD',
    'MU',
    'TSLA',
    'QQQM',
    'CLS',    # pending review
    'STX',    # pending review
]

# Removed: SPY, QQQ, VOO (too large), NFLX, ORCL, AMAT, ANET, ARM (low conviction)
# Removed: IONQ, RGTI, MARA, OKLO, MP, QLD, HIMS (high volatility)

# ============ SCREENING PARAMETERS ============
RSI_PERIOD = 14
RSI_THRESHOLD = 35          # Base threshold — overridden by VIX regime at runtime
BB_PERIOD = 20
ATR_PERIOD = 14

# SPX-specific
SPX_TICKER        = '^GSPC'
SPX_RSI_THRESHOLD = 30      # Base threshold — overridden by VIX regime at runtime
SPX_GAP_DOWN_PCT  = -1.0

# Delta targets by tier
TIER1_DELTA_MIN = 0.10
TIER1_DELTA_MAX = 0.18
TIER2_DELTA_MIN = 0.08
TIER2_DELTA_MAX = 0.13

# DTE window for expiry selection
DTE_MIN = 28   # ~4 weeks
DTE_MAX = 45

# VIX regime thresholds
VIX_LOW    = 15
VIX_NORMAL = 20
VIX_HIGH   = 30

# Tier 2 volatility guard
TIER2_ATR_MAX = 5.0

# ============ DYNAMIC VIX-ADJUSTED THRESHOLDS ============
# LOW  (<15) : premium thin — require deep oversold for entry
# NORMAL (15-20) : standard thresholds
# ELEVATED (20-30) : fat premium — slightly relaxed, more cushion
# HIGH (>30) : tail risk elevated — TIGHTEN thresholds, demand stronger signals
#              Even though premium is rich, gap-down / black-swan risk rises sharply.
#              Require deeper oversold and tighter BB position to confirm real
#              mean-reversion rather than a trending breakdown.
#
# Regime:      LOW (<15)    NORMAL (15-20)  ELEVATED (20-30)  HIGH (>30)
# RSI:         28           35              38                 30   <- tightened
# BB pos:      0.25         0.40            0.45               0.30 <- tightened
# SPX RSI:     25           30              33                 28   <- tightened

VIX_ADJUSTED_PARAMS = {
    'LOW': {
        'rsi_threshold':     28,    # Only enter on deep oversold — premium is thin
        'bb_threshold':      0.25,  # Require price very near lower band
        'spx_rsi_threshold': 25,
    },
    'NORMAL': {
        'rsi_threshold':     35,    # Standard thresholds
        'bb_threshold':      0.40,
        'spx_rsi_threshold': 30,
    },
    'ELEVATED': {
        'rsi_threshold':     38,    # Slightly relaxed — premium is fat, more cushion
        'bb_threshold':      0.45,
        'spx_rsi_threshold': 33,
    },
    'HIGH': {
        'rsi_threshold':     30,    # TIGHTENED: tail risk high, demand deeper oversold
        'bb_threshold':      0.30,  # TIGHTENED: require price near lower band for conviction
        'spx_rsi_threshold': 28,    # TIGHTENED: SPX gap-down must be more severe
    },
}

# Safe fallback used when VIX is unavailable or regime key is missing
_FALLBACK_PARAMS = VIX_ADJUSTED_PARAMS['NORMAL']


def get_vix_regime(vix):
    """
    Return regime string key for VIX_ADJUSTED_PARAMS lookup.
    Returns 'NORMAL' for None or any non-numeric VIX value so callers
    always get a valid key even when VIX fetch fails.
    """
    try:
        if vix is None or not isinstance(vix, (int, float)) or math.isnan(vix):
            return 'NORMAL'
        if vix < VIX_LOW:
            return 'LOW'
        elif vix < VIX_NORMAL:
            return 'NORMAL'
        elif vix < VIX_HIGH:
            return 'ELEVATED'
        else:
            return 'HIGH'
    except Exception:
        return 'NORMAL'


def get_adjusted_params(vix):
    """
    Return (params_dict, regime_str) adjusted for current VIX regime.
    Falls back to NORMAL params on any lookup failure — never raises.
    """
    try:
        regime = get_vix_regime(vix)
        params = VIX_ADJUSTED_PARAMS.get(regime, _FALLBACK_PARAMS)
        logger.info(
            f"[VIX-PARAMS] Regime={regime} | RSI threshold={params['rsi_threshold']} | "
            f"BB threshold={params['bb_threshold']} | SPX RSI threshold={params['spx_rsi_threshold']}"
        )
        return params, regime
    except Exception as e:
        logger.warning(f"[VIX-PARAMS] Failed to resolve regime params ({e}). Using NORMAL fallback.")
        return _FALLBACK_PARAMS, 'NORMAL'


# ============ IV RANK + IV/HV RATIO ============
# Two complementary filters — used together as a dual-pass premium quality check:
#
#   Pass 1 — IV Rank (52-week context):
#     IV Rank = (current IV - 52w low) / (52w high - 52w low) * 100
#     < IV_RANK_MIN (25): premium historically cheap — skip entry
#     Answers: "Is IV elevated vs its own history?"
#
#   Pass 2 — IV/HV Ratio (current relative value):
#     IV/HV = implied vol / 30-day realized vol
#     < IV_HV_MIN (1.0): market is not pricing in a vol premium over recent moves
#     Answers: "Is the market paying up for options right now?"
#
#   Why both? IV Rank can miss cases where IV is high vs history but realized vol
#   has caught up (e.g., post-CPI spike). IV/HV ratio catches that directly.
#   Together they confirm: high premium AND option buyers are overpaying vs recent moves.
#
# FAIL-SAFE POLICY (both filters):
#   Any data quality issue returns None for that metric.
#   None = fail-open: callers do NOT suppress the signal — a data outage never
#   silently blocks a valid entry. Only a confirmed bad value triggers suppression.

IV_RANK_MIN = 25    # IV Rank below this: skip (premium historically cheap)
IV_HV_MIN   = 1.0  # IV/HV ratio below this: skip (options not priced above realized vol)


def _safe_mid(row):
    """
    Compute option mid-price from a chain row.
    Returns None if bid or ask is NaN, zero, negative, or inverted.
    """
    bid = row.get('bid') if isinstance(row, dict) else getattr(row, 'bid', None)
    ask = row.get('ask') if isinstance(row, dict) else getattr(row, 'ask', None)
    try:
        bid = float(bid)
        ask = float(ask)
    except (TypeError, ValueError):
        return None
    if math.isnan(bid) or math.isnan(ask) or bid < 0 or ask <= 0 or ask < bid:
        return None
    return (bid + ask) / 2.0


def compute_iv_rank(ticker):
    """
    Compute IV Rank, IV Percentile, and IV/HV Ratio for a ticker.

    IV current is estimated from ATM call/put mid-prices using the
    Brenner-Subrahmanyam approximation:
        IV ≈ (option_mid / spot) × sqrt(2π / T)

    HV_30 = 30-day rolling realized vol (annualized, from log returns).
    IV/HV ratio = iv_current / hv_30 — values > 1.0 mean options are
    priced above recent realized volatility (desirable for premium sellers).

    52-week IV history is proxied via the same rolling rvol series since
    yfinance does not expose historical implied vol snapshots.

    Returns
    -------
    dict with keys:
        iv_current     : float or None
        iv_rank        : float in [0, 100] or None
        iv_pct         : float in [0, 100] or None
        iv_52w_high    : float or None
        iv_52w_low     : float or None
        hv_30          : float or None  (30-day realized vol, annualized %)
        iv_hv_ratio    : float or None  (iv_current / hv_30)
        skipped_reason : str or None    (None = success)

    NEVER raises. iv_rank=None or iv_hv_ratio=None means data unavailable;
    callers must treat both as fail-open (do not suppress the signal).
    """
    _empty = {
        'iv_current': None, 'iv_rank': None, 'iv_pct': None,
        'iv_52w_high': None, 'iv_52w_low': None,
        'hv_30': None, 'iv_hv_ratio': None,
        'skipped_reason': None,
    }

    try:
        t = yf.Ticker(ticker)
        today = datetime.today().date()

        # --- 1. Get nearest expiry with 7+ DTE ---
        raw_expiries = t.options
        if not raw_expiries:
            return {**_empty, 'skipped_reason': 'no options chain available'}

        valid_expiries = []
        for e in raw_expiries:
            try:
                exp_date = datetime.strptime(e, '%Y-%m-%d').date()
                if (exp_date - today).days >= 7:
                    valid_expiries.append(exp_date)
            except (ValueError, TypeError):
                continue
        valid_expiries.sort()

        if not valid_expiries:
            return {**_empty, 'skipped_reason': 'no expiry >= 7 DTE'}

        near_exp = valid_expiries[0]
        dte = (near_exp - today).days

        # --- 2. Fetch spot price ---
        hist = yf.download(ticker, period='2d', interval='1d', progress=False, group_by=False)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.droplevel(0)
        hist.columns = [c.capitalize() for c in hist.columns]
        if hist.empty or 'Close' not in hist.columns or len(hist) < 1:
            return {**_empty, 'skipped_reason': 'no price data'}
        spot = float(hist['Close'].iloc[-1])
        if math.isnan(spot) or spot <= 0:
            return {**_empty, 'skipped_reason': f'invalid spot price: {spot}'}

        # --- 3. Fetch option chain and find ATM strike ---
        try:
            chain = t.option_chain(str(near_exp))
        except Exception as ce:
            return {**_empty, 'skipped_reason': f'option_chain fetch failed: {ce}'}

        calls = chain.calls
        puts  = chain.puts

        if calls is None or calls.empty:
            return {**_empty, 'skipped_reason': 'empty calls chain'}
        if puts is None or puts.empty:
            return {**_empty, 'skipped_reason': 'empty puts chain'}

        call_strikes = calls['strike'].dropna()
        if call_strikes.empty:
            return {**_empty, 'skipped_reason': 'no valid call strikes'}
        atm_strike = call_strikes.iloc[(call_strikes - spot).abs().argsort().iloc[0]]

        atm_call_rows = calls[calls['strike'] == atm_strike]
        if atm_call_rows.empty:
            return {**_empty, 'skipped_reason': f'ATM call row missing for strike {atm_strike}'}
        atm_call = atm_call_rows.iloc[0]

        put_strikes = puts['strike'].dropna()
        if put_strikes.empty:
            return {**_empty, 'skipped_reason': 'no valid put strikes'}
        atm_put_strike = put_strikes.iloc[(put_strikes - atm_strike).abs().argsort().iloc[0]]
        atm_put_rows = puts[puts['strike'] == atm_put_strike]
        if atm_put_rows.empty:
            return {**_empty, 'skipped_reason': f'ATM put row missing for strike {atm_put_strike}'}
        atm_put = atm_put_rows.iloc[0]

        # --- 4. Mid-price with NaN/zero guard ---
        call_mid = _safe_mid(atm_call)
        put_mid  = _safe_mid(atm_put)

        if call_mid is None and put_mid is None:
            return {**_empty, 'skipped_reason': 'both ATM call and put have invalid bid/ask'}

        valid_mids = [m for m in [call_mid, put_mid] if m is not None]
        avg_mid = sum(valid_mids) / len(valid_mids)

        if avg_mid <= 0 or dte <= 0:
            return {**_empty, 'skipped_reason': f'non-positive mid ({avg_mid}) or DTE ({dte})'}

        # --- 5. Brenner-Subrahmanyam ATM IV approximation ---
        t_years = dte / 365.0
        iv_current = (avg_mid / spot) * math.sqrt(2.0 * math.pi / t_years)
        iv_current = round(iv_current * 100.0, 1)  # express as %
        if math.isnan(iv_current) or iv_current <= 0:
            return {**_empty, 'skipped_reason': f'computed IV is invalid: {iv_current}'}

        # --- 6. 52-week realized vol (IV history proxy) + HV_30 ---
        hist_1y = yf.download(ticker, period='1y', interval='1d', progress=False, group_by=False)
        if isinstance(hist_1y.columns, pd.MultiIndex):
            hist_1y.columns = hist_1y.columns.droplevel(0)
        hist_1y.columns = [c.capitalize() for c in hist_1y.columns]
        if hist_1y.empty or 'Close' not in hist_1y.columns or len(hist_1y) < 31:
            return {**_empty, 'skipped_reason': 'insufficient 1y price history for rvol'}

        pct = hist_1y['Close'].pct_change()
        log_ret = pct.apply(
            lambda x: math.log(1.0 + x) if (not pd.isna(x) and x > -1.0) else 0.0
        )
        hist_1y['rvol_30'] = log_ret.rolling(30).std() * math.sqrt(252) * 100.0

        rvol_series = hist_1y['rvol_30'].dropna()
        if rvol_series.empty:
            return {**_empty, 'skipped_reason': 'rvol_30 series is all NaN after dropna'}

        # HV_30: most recent 30-day realized vol value
        hv_30_raw = rvol_series.iloc[-1]
        hv_30 = round(float(hv_30_raw), 1) if not pd.isna(hv_30_raw) else None

        # IV/HV ratio — fail-open if hv_30 is zero or None
        if hv_30 is not None and hv_30 > 0:
            iv_hv_ratio = round(iv_current / hv_30, 2)
        else:
            iv_hv_ratio = None

        # --- 7. IV Rank + Percentile ---
        iv_52w_high_raw = rvol_series.max()
        iv_52w_low_raw  = rvol_series.min()

        if pd.isna(iv_52w_high_raw) or pd.isna(iv_52w_low_raw):
            return {**_empty, 'hv_30': hv_30, 'iv_hv_ratio': iv_hv_ratio,
                    'skipped_reason': 'rvol_30 min/max are NaN — insufficient data'}

        iv_52w_high = round(float(iv_52w_high_raw), 1)
        iv_52w_low  = round(float(iv_52w_low_raw), 1)

        iv_range = iv_52w_high - iv_52w_low
        if iv_range <= 0 or math.isnan(iv_range):
            return {**_empty, 'hv_30': hv_30, 'iv_hv_ratio': iv_hv_ratio,
                    'skipped_reason': f'zero or NaN IV range ({iv_range}) over 52w'}

        iv_rank = round((iv_current - iv_52w_low) / iv_range * 100.0, 1)
        iv_rank = max(0.0, min(100.0, iv_rank))

        iv_pct = round(float((rvol_series < iv_current).mean() * 100.0), 1)

        return {
            'iv_current':     iv_current,
            'iv_rank':        iv_rank,
            'iv_pct':         iv_pct,
            'iv_52w_high':    iv_52w_high,
            'iv_52w_low':     iv_52w_low,
            'hv_30':          hv_30,
            'iv_hv_ratio':    iv_hv_ratio,
            'skipped_reason': None,
        }

    except Exception as e:
        logger.warning(f"[IV-RANK] {ticker}: unexpected error — {e}")
        return {**_empty, 'skipped_reason': f'unexpected error: {e}'}


def _apply_iv_filters(ticker, iv_data, label):
    """
    Apply dual IV filter: IV Rank (Pass 1) then IV/HV ratio (Pass 2).
    Returns True if the signal should be SUPPRESSED, False if it should pass.
    Logs the reason. Both filters are fail-open on None values.

    Parameters
    ----------
    ticker   : str
    iv_data  : dict returned by compute_iv_rank()
    label    : str  log prefix e.g. '[TIER1_CORE]' or '[SPX]'

    Returns
    -------
    suppress : bool
    """
    iv_rank    = iv_data.get('iv_rank')
    iv_hv_ratio = iv_data.get('iv_hv_ratio')
    skip_reason = iv_data.get('skipped_reason')

    # Both None: data fully unavailable — fail-open, log once
    if iv_rank is None and iv_hv_ratio is None:
        logger.warning(
            f"{label} {ticker}: IV data unavailable ({skip_reason}) — "
            f"proceeding without IV/HV filters (fail-open)."
        )
        return False

    # Pass 1: IV Rank
    if iv_rank is not None:
        if iv_rank < IV_RANK_MIN:
            logger.info(
                f"{label} {ticker}: IV Rank {iv_rank} < {IV_RANK_MIN} — "
                f"premium historically cheap. Signal suppressed."
            )
            return True
    else:
        logger.warning(f"{label} {ticker}: IV Rank unavailable — skipping Pass 1 (fail-open).")

    # Pass 2: IV/HV ratio
    if iv_hv_ratio is not None:
        if iv_hv_ratio < IV_HV_MIN:
            logger.info(
                f"{label} {ticker}: IV/HV ratio {iv_hv_ratio} < {IV_HV_MIN} — "
                f"options not priced above realized vol. Signal suppressed."
            )
            return True
    else:
        logger.warning(f"{label} {ticker}: IV/HV ratio unavailable — skipping Pass 2 (fail-open).")

    return False


# ============ CLUSTER / CONCENTRATION GUARD ============
# If too many tickers signal on the same day, they're likely responding to
# the same macro event — not independent trades. Warn but do not auto-skip.

CLUSTER_WARN_THRESHOLD = 5
_CLUSTER_DISPLAY_MAX   = 10


def check_cluster_risk(all_results):
    """
    Count total signals and flag concentration risk.
    Never raises — safe for any input including empty or None.
    """
    try:
        if not all_results:
            return {
                'signal_count': 0,
                'cluster_risk': False,
                'cluster_note': '✓ Cluster check passed: 0 signals.',
                'tickers': [],
            }
        count   = len(all_results)
        tickers = list(all_results.keys())
        cluster_risk = count >= CLUSTER_WARN_THRESHOLD

        display_tickers = tickers[:_CLUSTER_DISPLAY_MAX]
        suffix = f' ... +{count - _CLUSTER_DISPLAY_MAX} more' if count > _CLUSTER_DISPLAY_MAX else ''

        note = (
            f"⚠️  CONCENTRATION RISK: {count} tickers triggered simultaneously "
            f"({', '.join(display_tickers)}{suffix}). "
            f"These likely share macro exposure — treat as correlated, not independent trades. "
            f"Consider sizing down or selecting the top 2-3 highest-conviction names only."
            if cluster_risk else
            f"✓ Cluster check passed: {count} signal(s) ({', '.join(display_tickers)}{suffix}) — "
            f"concentration risk low."
        )
        return {
            'signal_count': count,
            'cluster_risk': cluster_risk,
            'cluster_note': note,
            'tickers':      tickers,
        }
    except Exception as e:
        logger.warning(f"[CLUSTER] check_cluster_risk failed unexpectedly: {e}")
        return {
            'signal_count': len(all_results) if all_results else 0,
            'cluster_risk': False,
            'cluster_note': f'Cluster check error: {e}',
            'tickers':      [],
        }


# ============ POSITION MANAGEMENT CONSTANTS ============
EARLY_CLOSE_PROFIT_PCT  = 0.80
SPREAD_WIDTH            = 10
BASE_DTE_ACTION         = 4

EARNINGS_ENTRY_BUFFER_BEFORE = 5
EARNINGS_ENTRY_BUFFER_AFTER  = 1

T2_ROLLOVER_DTE        = 7
MAX_ROLLOVER_DEBIT_PCT = 0.50
T2_EMERGENCY_CLOSE_DTE = 7


# ============ POSITION EVALUATOR (Tier 2 runtime check) ============
def evaluate_tier2_position(ticker, current_price, short_put_strike, expiry_date,
                            entry_credit=None):
    long_put_strike = short_put_strike - SPREAD_WIDTH
    today = datetime.today().date()
    dte   = (expiry_date - today).days

    max_debit = round(entry_credit * MAX_ROLLOVER_DEBIT_PCT, 2) if entry_credit else None
    rollover_note = (
        f'1st: net credit roll (lower strike). '
        f'2nd: same-strike debit roll '
        f'(max debit ${max_debit}/share = ${max_debit*100:.0f}/contract). '
        f'Fallback: EMERGENCY_CLOSE.'
    ) if max_debit else (
        f'1st: net credit roll (lower strike). '
        f'2nd: same-strike debit <= {int(MAX_ROLLOVER_DEBIT_PCT*100)}% of entry credit. '
        f'Fallback: EMERGENCY_CLOSE.'
    )

    if dte <= T2_EMERGENCY_CLOSE_DTE and current_price <= long_put_strike:
        return {
            'ticker': ticker, 'action': 'EMERGENCY_CLOSE', 'stage': 2,
            'dte': dte, 'current_price': current_price,
            'short_put_strike': short_put_strike, 'long_put_strike': long_put_strike,
            'max_rollover_debit': max_debit,
            'reason': (
                f'DEEP ITM: price ${current_price} <= long put ${long_put_strike} '
                f'with DTE {dte}. Near max loss — close immediately.'
            ),
        }

    if dte <= T2_ROLLOVER_DTE and current_price < short_put_strike:
        return {
            'ticker': ticker, 'action': 'ROLLOVER', 'stage': 1,
            'dte': dte, 'current_price': current_price,
            'short_put_strike': short_put_strike, 'long_put_strike': long_put_strike,
            'max_rollover_debit': max_debit, 'rollover_priority': rollover_note,
            'reason': (
                f'SHORT PUT ITM: price ${current_price} < short put ${short_put_strike} '
                f'with DTE {dte}. Attempt rollover per priority order.'
            ),
        }

    if dte <= BASE_DTE_ACTION:
        return {
            'ticker': ticker, 'action': 'ROUTINE_REVIEW', 'stage': 0,
            'dte': dte, 'current_price': current_price,
            'short_put_strike': short_put_strike, 'long_put_strike': long_put_strike,
            'max_rollover_debit': max_debit,
            'reason': f'DTE {dte} <= {BASE_DTE_ACTION}: routine close/rollover review.',
        }

    return {
        'ticker': ticker, 'action': 'HOLD', 'stage': 0, 'dte': dte,
        'reason': f'No action needed. DTE {dte}, price ${current_price} above strikes.',
    }


# ============ VIX FETCH ============
def get_vix():
    try:
        vix_data = yf.download('^VIX', period='5d', interval='1d', progress=False, group_by=False)
        if isinstance(vix_data.columns, pd.MultiIndex):
            vix_data.columns = vix_data.columns.droplevel(0)
        vix_data.columns = [c.capitalize() for c in vix_data.columns]
        vix = round(float(vix_data['Close'].iloc[-1]), 2)
        if vix < VIX_LOW:
            regime = 'LOW (premiums thin — be selective)'
        elif vix < VIX_NORMAL:
            regime = 'NORMAL'
        elif vix < VIX_HIGH:
            regime = 'ELEVATED ✔ (good premium environment)'
        else:
            regime = 'HIGH ⚠️ (tail risk elevated — thresholds tightened)'
        logger.info(f"[VIX] Current: {vix} — Regime: {regime}")
        return vix
    except Exception as e:
        logger.warning(f"[VIX] Could not fetch VIX: {e}")
        return None


# ============ EARNINGS DATE FETCH ============
def get_earnings_date(ticker):
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        if cal is None or cal.empty:
            return None
        if 'Earnings Date' in cal.columns:
            date_val = cal['Earnings Date'].iloc[0]
        elif 'Earnings Date' in cal.index:
            date_val = cal.loc['Earnings Date'].iloc[0]
        else:
            return None
        if pd.isna(date_val):
            return None
        return pd.Timestamp(date_val).date()
    except Exception:
        return None


def is_earnings_blackout(earnings_date):
    if earnings_date is None:
        return False
    today = datetime.today().date()
    blackout_start = earnings_date - timedelta(days=EARNINGS_ENTRY_BUFFER_BEFORE)
    blackout_end   = earnings_date + timedelta(days=EARNINGS_ENTRY_BUFFER_AFTER)
    return blackout_start <= today <= blackout_end


# ============ EXPIRY SELECTION ============
def get_monthly_expiries(start_date, end_date):
    monthlies = []
    year, month = start_date.year, start_date.month
    while True:
        first_day    = datetime(year, month, 1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(weeks=2)
        exp_date     = third_friday.date()
        if exp_date > end_date:
            break
        if exp_date >= start_date:
            monthlies.append(exp_date)
        month += 1
        if month > 12:
            month = 1
            year += 1
    return monthlies


def get_target_expiry(ticker, earnings_date=None):
    today        = datetime.today().date()
    window_start = today + timedelta(days=DTE_MIN)
    window_end   = today + timedelta(days=DTE_MAX)
    try:
        t = yf.Ticker(ticker)
        raw_expiries = t.options
        available = sorted([
            datetime.strptime(e, '%Y-%m-%d').date()
            for e in raw_expiries
            if window_start <= datetime.strptime(e, '%Y-%m-%d').date() <= window_end
        ])
    except Exception:
        available = []
    if not available:
        return None
    monthlies = get_monthly_expiries(window_start, window_end)

    def is_earnings_conflict(exp):
        if earnings_date is None:
            return False
        return abs((exp - earnings_date).days) <= 5

    for exp in available:
        if exp in monthlies and not is_earnings_conflict(exp):
            return exp, (exp - today).days, True, (earnings_date is not None)
    for exp in available:
        if not is_earnings_conflict(exp):
            return exp, (exp - today).days, False, (earnings_date is not None)
    return None


# ============ HELPER FUNCTIONS ============
def get_support_level(stock_data, lookback=20):
    recent_low          = stock_data['Low'].tail(lookback).min()
    current_price       = stock_data['Close'].iloc[-1]
    distance_to_support = ((current_price - recent_low) / current_price) * 100
    return recent_low, distance_to_support


def calculate_signal_strength(rsi, bb_pos, vol_surge, atr_pct):
    score = 0
    if rsi < 25:      score += 30
    elif rsi < 30:    score += 25
    elif rsi < 35:    score += 20
    else:             score += 10
    if bb_pos < 0.15:   score += 30
    elif bb_pos < 0.25: score += 25
    elif bb_pos < 0.35: score += 20
    else:               score += 10
    if vol_surge > 2.0:   score += 25
    elif vol_surge > 1.5: score += 20
    elif vol_surge > 1.2: score += 15
    else:                 score += 10
    if atr_pct > 3.0:   score += 15
    elif atr_pct > 2.0: score += 12
    elif atr_pct > 1.5: score += 10
    else:               score += 5
    return score


# ============ SPX SCREENING ============
def screen_spx(vix, adjusted_params):
    logger.info("\n>>> Screening SPX (^GSPC) — European-style, Put Spread Specialist <<<")
    results = {}
    spx_rsi_threshold = adjusted_params['spx_rsi_threshold']
    try:
        stock_data = yf.download(SPX_TICKER, period='1y', interval='1d', progress=False, group_by=False)
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.droplevel(0)
        stock_data.columns = [c.capitalize() for c in stock_data.columns]
        if stock_data.empty or len(stock_data) < 200:
            logger.warning("[SPX] Insufficient data.")
            return results

        stock_data['SMA_200']    = stock_data['Close'].rolling(200).mean()
        stock_data['AVG_VOL_50'] = stock_data['Volume'].rolling(50).mean()
        stock_data.ta.rsi(length=RSI_PERIOD, append=True)
        rsi_col = f'RSI_{RSI_PERIOD}'
        stock_data['BB_middle']   = stock_data['Close'].rolling(BB_PERIOD).mean()
        stock_data['BB_std']      = stock_data['Close'].rolling(BB_PERIOD).std()
        stock_data['BB_upper']    = stock_data['BB_middle'] + stock_data['BB_std'] * 2
        stock_data['BB_lower']    = stock_data['BB_middle'] - stock_data['BB_std'] * 2
        stock_data['BB_position'] = (stock_data['Close'] - stock_data['BB_lower']) / (stock_data['BB_upper'] - stock_data['BB_lower'])
        stock_data.ta.atr(length=ATR_PERIOD, append=True)
        atr_col = f'ATR_{ATR_PERIOD}'
        stock_data.ta.macd(append=True)
        if not all(c in stock_data.columns for c in [rsi_col, atr_col, 'MACDh_12_26_9']):
            logger.warning("[SPX] TA indicators missing.")
            return results

        latest_close      = float(stock_data['Close'].iloc[-1])
        prior_close       = float(stock_data['Close'].iloc[-2])
        latest_open       = float(stock_data['Open'].iloc[-1])
        latest_sma_200    = float(stock_data['SMA_200'].iloc[-1])
        latest_volume     = float(stock_data['Volume'].iloc[-1])
        latest_avg_vol_50 = float(stock_data['AVG_VOL_50'].iloc[-1])
        current_rsi       = float(stock_data[rsi_col].iloc[-1])
        latest_bb_pos     = float(stock_data['BB_position'].iloc[-1])
        latest_bb_lower   = float(stock_data['BB_lower'].iloc[-1])
        latest_bb_upper   = float(stock_data['BB_upper'].iloc[-1])
        latest_atr        = float(stock_data[atr_col].iloc[-1])
        macd_histogram    = float(stock_data['MACDh_12_26_9'].iloc[-1])
        atr_pct            = (latest_atr / latest_close) * 100
        volume_surge_ratio = latest_volume / latest_avg_vol_50
        support_price, pct_above_support = get_support_level(stock_data)

        gap_down_pct = ((latest_open - prior_close) / prior_close) * 100
        is_gap_down  = gap_down_pct <= SPX_GAP_DOWN_PCT
        is_rsi_low   = current_rsi < spx_rsi_threshold
        is_uptrend   = latest_close > latest_sma_200
        logger.info(
            f"[SPX] Gap: {gap_down_pct:.2f}% | RSI: {current_rsi:.1f} "
            f"(threshold: {spx_rsi_threshold}) | SMA200: {is_uptrend} | VIX: {vix}"
        )

        if is_gap_down and is_rsi_low and is_uptrend:
            iv_data  = compute_iv_rank(SPX_TICKER)
            suppress = _apply_iv_filters('SPX', iv_data, '[SPX]')
            if suppress:
                return results

            signal_strength = calculate_signal_strength(current_rsi, latest_bb_pos, volume_surge_ratio, atr_pct)
            expiry_info = get_target_expiry(SPX_TICKER)
            expiry_date = str(expiry_info[0]) if expiry_info else 'N/A'
            expiry_dte  = expiry_info[1] if expiry_info else None
            is_monthly  = expiry_info[2] if expiry_info else None
            results['SPX'] = {
                'Tier': 'TIER1_CORE', 'Signal_Strength': signal_strength,
                'RSI': round(current_rsi, 2), 'Price': round(latest_close, 2),
                'Prior_Close': round(prior_close, 2), 'Open': round(latest_open, 2),
                'Gap_Down_%': round(gap_down_pct, 2), 'SMA_200': round(latest_sma_200, 2),
                'BB_Position': round(latest_bb_pos, 2), 'BB_Lower': round(latest_bb_lower, 2),
                'BB_Upper': round(latest_bb_upper, 2), 'ATR_%': round(atr_pct, 2),
                'Vol_Surge': round(volume_surge_ratio, 2), 'Support': round(support_price, 2),
                'Distance_to_Support_%': round(pct_above_support, 1),
                'MACD_Histogram': round(macd_histogram, 3), 'VIX': vix,
                'VIX_Regime': get_vix_regime(vix),
                'RSI_Threshold_Used': spx_rsi_threshold,
                'IV_Rank':      iv_data.get('iv_rank'),
                'IV_Pct':       iv_data.get('iv_pct'),
                'IV_52w_High':  iv_data.get('iv_52w_high'),
                'IV_52w_Low':   iv_data.get('iv_52w_low'),
                'HV_30':        iv_data.get('hv_30'),
                'IV_HV_Ratio':  iv_data.get('iv_hv_ratio'),
                'IV_Skip_Reason': iv_data.get('skipped_reason'),
                'Delta_Target': f'{TIER1_DELTA_MIN}–{TIER1_DELTA_MAX}',
                'Expiry_Date': expiry_date, 'Expiry_DTE': expiry_dte, 'Is_Monthly': is_monthly,
                'Earnings_Avoided': 'N/A (index)', 'Earnings_Blackout': False,
                'Position_Mgmt': f'Routine review at DTE<={BASE_DTE_ACTION} only',
                'Note': 'European-style. No early assignment. CBOE SPX options only.',
            }
            logger.info(
                f"✓ [SPX] Gap {gap_down_pct:.2f}% | RSI {current_rsi:.1f} | "
                f"IV Rank {iv_data.get('iv_rank')} | IV/HV {iv_data.get('iv_hv_ratio')} | "
                f"Signal: {signal_strength}/100 | Expiry: {expiry_date} (DTE {expiry_dte})"
            )
        else:
            logger.info("[SPX] Conditions not met. No signal.")
    except Exception as e:
        logger.error(f"[SPX] Error: {e}")
    return results


# ============ GENERAL TIER SCREENING ============
def screen_tickers(tickers, tier_label, vix, adjusted_params):
    is_tier2     = (tier_label == 'TIER2_WATCH')
    delta_target = (
        f'{TIER2_DELTA_MIN}–{TIER2_DELTA_MAX}' if is_tier2
        else f'{TIER1_DELTA_MIN}–{TIER1_DELTA_MAX}'
    )
    rsi_threshold = adjusted_params['rsi_threshold']
    bb_threshold  = adjusted_params['bb_threshold']
    results = {}
    error_count = 0
    successful_count = 0

    for ticker in tickers:
        try:
            stock_data = yf.download(ticker, period='1y', interval='1d', progress=False, group_by=False)
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = stock_data.columns.droplevel(0)
            stock_data.columns = [c.capitalize() for c in stock_data.columns]
            if 'Close' not in stock_data.columns or stock_data.empty or len(stock_data) < 200:
                logger.warning(f"[{tier_label}] Insufficient data for {ticker}.")
                error_count += 1
                continue

            earnings_date = get_earnings_date(ticker)
            if is_earnings_blackout(earnings_date):
                logger.info(
                    f"[{tier_label}] {ticker}: EARNINGS BLACKOUT — earnings {earnings_date}, "
                    f"within ±{EARNINGS_ENTRY_BUFFER_BEFORE}/{EARNINGS_ENTRY_BUFFER_AFTER}d. Skipping."
                )
                successful_count += 1
                continue

            stock_data['SMA_200']    = stock_data['Close'].rolling(200).mean()
            stock_data['AVG_VOL_50'] = stock_data['Volume'].rolling(50).mean()
            stock_data.ta.rsi(length=RSI_PERIOD, append=True)
            rsi_col = f'RSI_{RSI_PERIOD}'
            stock_data['BB_middle']   = stock_data['Close'].rolling(BB_PERIOD).mean()
            stock_data['BB_std']      = stock_data['Close'].rolling(BB_PERIOD).std()
            stock_data['BB_upper']    = stock_data['BB_middle'] + stock_data['BB_std'] * 2
            stock_data['BB_lower']    = stock_data['BB_middle'] - stock_data['BB_std'] * 2
            stock_data['BB_position'] = (
                (stock_data['Close'] - stock_data['BB_lower']) /
                (stock_data['BB_upper'] - stock_data['BB_lower'])
            )
            stock_data.ta.atr(length=ATR_PERIOD, append=True)
            atr_col = f'ATR_{ATR_PERIOD}'
            stock_data.ta.macd(append=True)
            if not all(c in stock_data.columns for c in [rsi_col, atr_col, 'MACDh_12_26_9']):
                logger.warning(f"[{tier_label}] TA indicators missing for {ticker}.")
                error_count += 1
                continue

            latest_close      = float(stock_data['Close'].iloc[-1])
            prior_close       = float(stock_data['Close'].iloc[-2])
            latest_sma_200    = float(stock_data['SMA_200'].iloc[-1])
            latest_volume     = float(stock_data['Volume'].iloc[-1])
            latest_avg_vol_50 = float(stock_data['AVG_VOL_50'].iloc[-1])
            current_rsi       = float(stock_data[rsi_col].iloc[-1])
            latest_bb_pos     = float(stock_data['BB_position'].iloc[-1])
            latest_bb_lower   = float(stock_data['BB_lower'].iloc[-1])
            latest_bb_upper   = float(stock_data['BB_upper'].iloc[-1])
            latest_atr        = float(stock_data[atr_col].iloc[-1])
            macd_histogram    = float(stock_data['MACDh_12_26_9'].iloc[-1])
            atr_pct            = (latest_atr / latest_close) * 100
            volume_surge_ratio = latest_volume / latest_avg_vol_50
            support_price, pct_above_support = get_support_level(stock_data)

            is_red_day       = latest_close < prior_close
            is_oversold      = current_rsi < rsi_threshold
            is_uptrend_long  = latest_close > latest_sma_200
            is_liquid        = latest_volume > latest_avg_vol_50
            is_near_lower_bb = latest_bb_pos < bb_threshold
            is_adequate_vol  = atr_pct > 1.0
            is_volume_surge  = volume_surge_ratio > 1.2

            if is_tier2 and atr_pct > TIER2_ATR_MAX:
                logger.info(
                    f"[{tier_label}] {ticker}: ATR% {atr_pct:.2f}% > {TIER2_ATR_MAX}% "
                    f"— too volatile, skipping."
                )
                successful_count += 1
                continue

            if (is_oversold and is_uptrend_long and is_liquid and
                    is_near_lower_bb and is_adequate_vol and is_volume_surge):

                iv_data  = compute_iv_rank(ticker)
                suppress = _apply_iv_filters(ticker, iv_data, f'[{tier_label}]')
                if suppress:
                    successful_count += 1
                    continue

                signal_strength = calculate_signal_strength(current_rsi, latest_bb_pos, volume_surge_ratio, atr_pct)
                expiry_info     = get_target_expiry(ticker, earnings_date)
                expiry_date_str = str(expiry_info[0]) if expiry_info else 'N/A (earnings conflict)'
                expiry_dte      = expiry_info[1] if expiry_info else None
                is_monthly      = expiry_info[2] if expiry_info else None
                earn_avoided    = str(earnings_date) if expiry_info and expiry_info[3] else 'N/A'
                t2_mgmt_note = (
                    f'Stage1(DTE<={T2_ROLLOVER_DTE}+price<short_put): '
                    f'1st net credit roll, 2nd debit<={int(MAX_ROLLOVER_DEBIT_PCT*100)}% of credit, fallback close | '
                    f'Stage2(DTE<={T2_EMERGENCY_CLOSE_DTE}+price<=long_put): emergency close'
                ) if is_tier2 else f'Routine review at DTE<={BASE_DTE_ACTION} only'

                results[ticker] = {
                    'Tier': tier_label, 'Signal_Strength': signal_strength,
                    'RSI': round(current_rsi, 2), 'Price': round(latest_close, 2),
                    'Red_Day': is_red_day, 'SMA_200': round(latest_sma_200, 2),
                    'BB_Position': round(latest_bb_pos, 2), 'BB_Lower': round(latest_bb_lower, 2),
                    'BB_Upper': round(latest_bb_upper, 2), 'ATR_%': round(atr_pct, 2),
                    'Vol_Surge': round(volume_surge_ratio, 2), 'Support': round(support_price, 2),
                    'Distance_to_Support_%': round(pct_above_support, 1),
                    'MACD_Histogram': round(macd_histogram, 3), 'VIX': vix,
                    'VIX_Regime': get_vix_regime(vix),
                    'RSI_Threshold_Used': rsi_threshold,
                    'BB_Threshold_Used':  bb_threshold,
                    'IV_Rank':      iv_data.get('iv_rank'),
                    'IV_Pct':       iv_data.get('iv_pct'),
                    'IV_52w_High':  iv_data.get('iv_52w_high'),
                    'IV_52w_Low':   iv_data.get('iv_52w_low'),
                    'HV_30':        iv_data.get('hv_30'),
                    'IV_HV_Ratio':  iv_data.get('iv_hv_ratio'),
                    'IV_Skip_Reason': iv_data.get('skipped_reason'),
                    'Delta_Target': delta_target, 'Expiry_Date': expiry_date_str,
                    'Expiry_DTE': expiry_dte, 'Is_Monthly': is_monthly,
                    'Earnings_Avoided': earn_avoided, 'Earnings_Blackout': False,
                    'Position_Mgmt': t2_mgmt_note,
                }
                logger.info(
                    f"✓ [{tier_label}] {ticker}: RSI {current_rsi:.1f} (thr={rsi_threshold}) | "
                    f"BB {latest_bb_pos:.2f} (thr={bb_threshold}) | ATR% {atr_pct:.2f} | "
                    f"IV Rank {iv_data.get('iv_rank')} | IV/HV {iv_data.get('iv_hv_ratio')} | "
                    f"Signal: {signal_strength}/100 | Expiry: {expiry_date_str} (DTE {expiry_dte})"
                )
            successful_count += 1
        except Exception as e:
            logger.error(f"[{tier_label}] Error processing {ticker}: {e}")
            error_count += 1

    logger.info(f"[{tier_label}] Done — {successful_count} analyzed, {len(results)} signals, {error_count} errors")
    return results


# ============ MAIN RUNNER ============
def run_screener():
    logger.info("=" * 70)
    logger.info("OPTIONS PREMIUM SCREENER — WINNING STOCKS PRIORITY MODE")
    logger.info("=" * 70)

    vix = get_vix()
    adjusted_params, regime = get_adjusted_params(vix)

    logger.info(f"VIX Regime                     : {regime}")
    logger.info(f"RSI Threshold (general, adj.)  : {adjusted_params['rsi_threshold']} (base: {RSI_THRESHOLD})")
    logger.info(f"RSI Threshold (SPX, adj.)      : {adjusted_params['spx_rsi_threshold']} (base: {SPX_RSI_THRESHOLD}) + gap-down >= {SPX_GAP_DOWN_PCT}%")
    logger.info(f"BB Threshold (adj.)            : {adjusted_params['bb_threshold']} (base: 0.40)")
    logger.info(f"IV Rank minimum (Pass 1)       : {IV_RANK_MIN} (below = premium historically cheap)")
    logger.info(f"IV/HV minimum (Pass 2)         : {IV_HV_MIN} (below = options not priced above realized vol)")
    logger.info(f"IV filter policy               : fail-open (None = proceed without filter)")
    logger.info(f"Cluster warn threshold         : {CLUSTER_WARN_THRESHOLD} simultaneous signals")
    logger.info(f"Delta — Tier 1                 : {TIER1_DELTA_MIN}–{TIER1_DELTA_MAX}")
    logger.info(f"Delta — Tier 2                 : {TIER2_DELTA_MIN}–{TIER2_DELTA_MAX} (ATR% guard <= {TIER2_ATR_MAX}%)")
    logger.info(f"DTE window                     : {DTE_MIN}–{DTE_MAX} days (monthly preferred)")
    logger.info(f"Early close profit target      : {int(EARLY_CLOSE_PROFIT_PCT*100)}%")
    logger.info(f"Base DTE action (all tiers)    : DTE <= {BASE_DTE_ACTION}")
    logger.info(f"Earnings blackout              : -{EARNINGS_ENTRY_BUFFER_BEFORE}d / +{EARNINGS_ENTRY_BUFFER_AFTER}d")
    logger.info(f"Tier 2 Stage 1 rollover        : DTE<={T2_ROLLOVER_DTE} + price<short_put")
    logger.info(f"  -> 1st: net credit roll (lower strike)")
    logger.info(f"  -> 2nd: same-strike debit <= {int(MAX_ROLLOVER_DEBIT_PCT*100)}% of entry credit")
    logger.info(f"  -> fallback: EMERGENCY_CLOSE")
    logger.info(f"Tier 2 Stage 2 emergency close : DTE<={T2_EMERGENCY_CLOSE_DTE} + price<=long_put")
    logger.info(f"Tier 1                         : SPX + {', '.join(TIER1_CORE)}")
    logger.info(f"Tier 2                         : {', '.join(TIER2_WATCHLIST)}")
    logger.info("=" * 70)

    all_results = {}
    spx_result    = screen_spx(vix, adjusted_params)
    all_results.update(spx_result)
    logger.info("\n>>> Screening TIER 1 — Core Winning Stocks <<<")
    tier1_results = screen_tickers(TIER1_CORE, tier_label="TIER1_CORE", vix=vix, adjusted_params=adjusted_params)
    all_results.update(tier1_results)
    logger.info("\n>>> Screening TIER 2 — Watchlist <<<")
    tier2_results = screen_tickers(TIER2_WATCHLIST, tier_label="TIER2_WATCH", vix=vix, adjusted_params=adjusted_params)
    all_results.update(tier2_results)

    # ============ CLUSTER / CONCENTRATION GUARD ============
    cluster_info = check_cluster_risk(all_results)
    logger.info("=" * 70)
    logger.info(cluster_info['cluster_note'])
    logger.info("=" * 70)
    logger.info(
        f"Total signals : {len(all_results)}  "
        f"(SPX: {len(spx_result)}, T1: {len(tier1_results)}, T2: {len(tier2_results)})"
    )

    if all_results:
        results_df = pd.DataFrame.from_dict(all_results, orient='index')
        results_df.index.name = 'Ticker'
        tier_order = {'TIER1_CORE': 0, 'TIER2_WATCH': 1}
        results_df['_tier_rank'] = results_df['Tier'].map(tier_order)
        results_df['_spx_first'] = (results_df.index == 'SPX').astype(int) * -1
        results_df = results_df.sort_values(
            ['_tier_rank', '_spx_first', 'Signal_Strength'], ascending=[True, True, False]
        ).drop(columns=['_tier_rank', '_spx_first'])
        results_df['Scan_Date']    = datetime.now().strftime('%Y-%m-%d')
        results_df['Scan_Time']    = datetime.now().strftime('%H:%M:%S')
        results_df['Cluster_Risk'] = cluster_info['cluster_risk']
        output_file = f'signals_{datetime.now().strftime("%Y%m%d")}.csv'
        results_df.to_csv(output_file)
        logger.info(f"Results saved → {output_file}")
        display_cols = [
            'Tier', 'Signal_Strength', 'RSI', 'RSI_Threshold_Used',
            'Price', 'ATR_%', 'VIX', 'VIX_Regime',
            'IV_Rank', 'IV_Pct', 'HV_30', 'IV_HV_Ratio',
            'Delta_Target', 'Expiry_Date', 'Expiry_DTE',
            'Is_Monthly', 'Earnings_Avoided', 'Cluster_Risk', 'Position_Mgmt',
        ]
        if 'Gap_Down_%' in results_df.columns:
            display_cols.insert(4, 'Gap_Down_%')
        if 'Red_Day' in results_df.columns:
            display_cols.insert(4, 'Red_Day')
        print(results_df[[c for c in display_cols if c in results_df.columns]])
    else:
        logger.info("No stocks found meeting the criteria today.")
    logger.info("\nScreening complete. Check log file for full details.")
    return all_results


if __name__ == "__main__":
    run_screener()
