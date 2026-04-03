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
RSI_THRESHOLD = 35
BB_PERIOD = 20
ATR_PERIOD = 14

# SPX-specific
SPX_TICKER        = '^GSPC'
SPX_RSI_THRESHOLD = 30
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

# ============ POSITION MANAGEMENT CONSTANTS ============
# --- Universal (all tiers) ---
EARLY_CLOSE_PROFIT_PCT  = 0.80   # Close when premium decayed >= 80%
SPREAD_WIDTH            = 10     # Fixed $10 put spread width
BASE_DTE_ACTION         = 4      # DTE <= 4: routine close/rollover review

# --- Earnings entry blackout (all tiers) ---
EARNINGS_ENTRY_BUFFER_BEFORE = 5  # days before earnings: block entry
EARNINGS_ENTRY_BUFFER_AFTER  = 1  # days after earnings: wait for stabilization

# --- Tier 2 only: two-stage emergency management ---
# Stage 1 - Rollover:
#   DTE <= T2_ROLLOVER_DTE AND current_price < short_put_strike
#   Priority 1: roll to lower strike for net credit
#   Priority 2: roll same strike, debit <= entry_credit * MAX_ROLLOVER_DEBIT_PCT
#   If neither viable: escalate to EMERGENCY_CLOSE
T2_ROLLOVER_DTE       = 7
MAX_ROLLOVER_DEBIT_PCT = 0.50  # Max debit = 50% of original entry credit
                                # e.g. entry credit $1.00 -> max debit $0.50 ($50/contract)
                                # Rollover priority:
                                #   1st: net credit roll (lower strike) <- ideal
                                #   2nd: same-strike debit roll <= $0.50 <- acceptable
                                #   fallback: EMERGENCY_CLOSE

# Stage 2 - Emergency close:
#   DTE <= T2_EMERGENCY_CLOSE_DTE AND current_price <= long_put_strike (deep ITM)
T2_EMERGENCY_CLOSE_DTE = 7

# Note on Tier 1:
#   No emergency triggers. 1+ year trading history: zero early assignments
#   or reserved capital loss on Tier 1. Only BASE_DTE_ACTION applies.


# ============ POSITION EVALUATOR (Tier 2 runtime check) ============
def evaluate_tier2_position(ticker, current_price, short_put_strike, expiry_date,
                            entry_credit=None):
    """
    Evaluate an open Tier 2 put spread position for emergency action.
    Call this daily for each open Tier 2 position.

    Parameters
    ----------
    ticker           : str
    current_price    : float  current market price of underlying
    short_put_strike : float  higher strike (credit leg, e.g. 200.0)
    expiry_date      : date
    entry_credit     : float  optional — original premium received per share
                              used to calculate max rollover debit allowance

    Returns
    -------
    dict: action in ['HOLD', 'ROLLOVER', 'EMERGENCY_CLOSE', 'ROUTINE_REVIEW']
    """
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

    # Stage 2: deep ITM
    if dte <= T2_EMERGENCY_CLOSE_DTE and current_price <= long_put_strike:
        return {
            'ticker':            ticker,
            'action':            'EMERGENCY_CLOSE',
            'stage':             2,
            'dte':               dte,
            'current_price':     current_price,
            'short_put_strike':  short_put_strike,
            'long_put_strike':   long_put_strike,
            'max_rollover_debit': max_debit,
            'reason': (
                f'DEEP ITM: price ${current_price} <= long put ${long_put_strike} '
                f'with DTE {dte}. Near max loss — close immediately.'
            ),
        }

    # Stage 1: short put ITM -> rollover
    if dte <= T2_ROLLOVER_DTE and current_price < short_put_strike:
        return {
            'ticker':            ticker,
            'action':            'ROLLOVER',
            'stage':             1,
            'dte':               dte,
            'current_price':     current_price,
            'short_put_strike':  short_put_strike,
            'long_put_strike':   long_put_strike,
            'max_rollover_debit': max_debit,
            'rollover_priority': rollover_note,
            'reason': (
                f'SHORT PUT ITM: price ${current_price} < short put ${short_put_strike} '
                f'with DTE {dte}. Attempt rollover per priority order.'
            ),
        }

    # Routine review
    if dte <= BASE_DTE_ACTION:
        return {
            'ticker':            ticker,
            'action':            'ROUTINE_REVIEW',
            'stage':             0,
            'dte':               dte,
            'current_price':     current_price,
            'short_put_strike':  short_put_strike,
            'long_put_strike':   long_put_strike,
            'max_rollover_debit': max_debit,
            'reason': f'DTE {dte} <= {BASE_DTE_ACTION}: routine close/rollover review.',
        }

    return {
        'ticker':  ticker,
        'action':  'HOLD',
        'stage':   0,
        'dte':     dte,
        'reason':  f'No action needed. DTE {dte}, price ${current_price} above strikes.',
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
            regime = 'HIGH ⚠️ (great premium but tail risk elevated)'
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
def screen_spx(vix):
    logger.info("\n>>> Screening SPX (^GSPC) — European-style, Put Spread Specialist <<<")
    results = {}
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
        is_rsi_low   = current_rsi < SPX_RSI_THRESHOLD
        is_uptrend   = latest_close > latest_sma_200
        logger.info(f"[SPX] Gap: {gap_down_pct:.2f}% | RSI: {current_rsi:.1f} | SMA200: {is_uptrend} | VIX: {vix}")

        if is_gap_down and is_rsi_low and is_uptrend:
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
                'Delta_Target': f'{TIER1_DELTA_MIN}–{TIER1_DELTA_MAX}',
                'Expiry_Date': expiry_date, 'Expiry_DTE': expiry_dte, 'Is_Monthly': is_monthly,
                'Earnings_Avoided': 'N/A (index)', 'Earnings_Blackout': False,
                'Position_Mgmt': f'Routine review at DTE<={BASE_DTE_ACTION} only',
                'Note': 'European-style. No early assignment. CBOE SPX options only.',
            }
            logger.info(f"✓ [SPX] Gap {gap_down_pct:.2f}% | RSI {current_rsi:.1f} | Signal: {signal_strength}/100 | Expiry: {expiry_date} (DTE {expiry_dte}, monthly={is_monthly})")
        else:
            logger.info("[SPX] Conditions not met. No signal.")
    except Exception as e:
        logger.error(f"[SPX] Error: {e}")
    return results


# ============ GENERAL TIER SCREENING ============
def screen_tickers(tickers, tier_label, vix):
    is_tier2     = (tier_label == 'TIER2_WATCH')
    delta_target = (
        f'{TIER2_DELTA_MIN}–{TIER2_DELTA_MAX}' if is_tier2
        else f'{TIER1_DELTA_MIN}–{TIER1_DELTA_MAX}'
    )
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
                logger.info(f"[{tier_label}] {ticker}: EARNINGS BLACKOUT — earnings {earnings_date}, within ±{EARNINGS_ENTRY_BUFFER_BEFORE}/{EARNINGS_ENTRY_BUFFER_AFTER}d. Skipping.")
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
            stock_data['BB_position'] = (stock_data['Close'] - stock_data['BB_lower']) / (stock_data['BB_upper'] - stock_data['BB_lower'])
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
            is_oversold      = current_rsi < RSI_THRESHOLD
            is_uptrend_long  = latest_close > latest_sma_200
            is_liquid        = latest_volume > latest_avg_vol_50
            is_near_lower_bb = latest_bb_pos < 0.4
            is_adequate_vol  = atr_pct > 1.0
            is_volume_surge  = volume_surge_ratio > 1.2

            if is_tier2 and atr_pct > TIER2_ATR_MAX:
                logger.info(f"[{tier_label}] {ticker}: ATR% {atr_pct:.2f}% > {TIER2_ATR_MAX}% — too volatile, skipping.")
                successful_count += 1
                continue

            if (is_oversold and is_uptrend_long and is_liquid and
                    is_near_lower_bb and is_adequate_vol and is_volume_surge):
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
                    'Delta_Target': delta_target, 'Expiry_Date': expiry_date_str,
                    'Expiry_DTE': expiry_dte, 'Is_Monthly': is_monthly,
                    'Earnings_Avoided': earn_avoided, 'Earnings_Blackout': False,
                    'Position_Mgmt': t2_mgmt_note,
                }
                logger.info(
                    f"✓ [{tier_label}] {ticker}: RSI {current_rsi:.1f} | BB {latest_bb_pos:.2f} | "
                    f"ATR% {atr_pct:.2f} | Signal: {signal_strength}/100 | "
                    f"Expiry: {expiry_date_str} (DTE {expiry_dte}) | Delta: {delta_target}"
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
    logger.info(f"RSI Threshold (general)        : {RSI_THRESHOLD}")
    logger.info(f"RSI Threshold (SPX)            : {SPX_RSI_THRESHOLD} + gap-down >= {SPX_GAP_DOWN_PCT}%")
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

    vix = get_vix()
    all_results = {}
    spx_result    = screen_spx(vix)
    all_results.update(spx_result)
    logger.info("\n>>> Screening TIER 1 — Core Winning Stocks <<<")
    tier1_results = screen_tickers(TIER1_CORE, tier_label="TIER1_CORE", vix=vix)
    all_results.update(tier1_results)
    logger.info("\n>>> Screening TIER 2 — Watchlist <<<")
    tier2_results = screen_tickers(TIER2_WATCHLIST, tier_label="TIER2_WATCH", vix=vix)
    all_results.update(tier2_results)

    logger.info("=" * 70)
    logger.info(f"Total signals : {len(all_results)}  (SPX: {len(spx_result)}, T1: {len(tier1_results)}, T2: {len(tier2_results)})")

    if all_results:
        results_df = pd.DataFrame.from_dict(all_results, orient='index')
        results_df.index.name = 'Ticker'
        tier_order = {'TIER1_CORE': 0, 'TIER2_WATCH': 1}
        results_df['_tier_rank'] = results_df['Tier'].map(tier_order)
        results_df['_spx_first'] = (results_df.index == 'SPX').astype(int) * -1
        results_df = results_df.sort_values(
            ['_tier_rank', '_spx_first', 'Signal_Strength'], ascending=[True, True, False]
        ).drop(columns=['_tier_rank', '_spx_first'])
        results_df['Scan_Date'] = datetime.now().strftime('%Y-%m-%d')
        results_df['Scan_Time'] = datetime.now().strftime('%H:%M:%S')
        output_file = f'signals_{datetime.now().strftime("%Y%m%d")}.csv'
        results_df.to_csv(output_file)
        logger.info(f"Results saved → {output_file}")
        display_cols = ['Tier', 'Signal_Strength', 'RSI', 'Price', 'ATR_%',
                        'VIX', 'Delta_Target', 'Expiry_Date', 'Expiry_DTE',
                        'Is_Monthly', 'Earnings_Avoided', 'Position_Mgmt']
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
