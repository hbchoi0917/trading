import yfinance as yf
import pandas as pd
import logging
from datetime import datetime

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
# Tier 1: Core winning stocks — screened first, highest conviction
# SPX (^GSPC) is handled separately via screen_spx() with its own logic
TIER1_CORE = [
    'COST',   # Costco — steady, strong premiums
    'NVDA',   # NVIDIA — high IV, great premium
    'IWM',    # Russell 2000 ETF — liquid, range-bound friendly
    'GOOGL',  # Alphabet Class A
]

# Tier 2: Watchlist — screened after Tier 1, monitor for re-entry signals
# CLS and STX: kept pending further performance review
TIER2_WATCHLIST = [
    'MSFT',   # Microsoft
    'AAPL',   # Apple
    'AMZN',   # Amazon
    'META',   # Meta
    'AVGO',   # Broadcom
    'CRWD',   # CrowdStrike
    'PLTR',   # Palantir
    'AMD',    # AMD
    'MU',     # Micron
    'TSLA',   # Tesla
    'QQQM',   # Invesco Nasdaq 100 ETF (smaller shares)
    'CLS',    # Celestica — watchlist pending review
    'STX',    # Seagate — watchlist pending review
]

# Removed tickers:
# SPY, QQQ, VOO       — too large/liquid, spread premiums too thin
# NFLX, ORCL, AMAT, ANET, ARM — low conviction, removed from watchlist
# IONQ, RGTI, MARA, OKLO, MP, QLD, HIMS — high volatility / low premium quality

# ============ SCREENING PARAMETERS ============
RSI_PERIOD = 14
RSI_THRESHOLD = 35        # General oversold threshold for Tier 1 / Tier 2
BB_PERIOD = 20
ATR_PERIOD = 14

# SPX-specific thresholds (stricter — European-style, no early assignment)
SPX_TICKER = '^GSPC'
SPX_RSI_THRESHOLD = 30    # Tighter RSI for SPX put spread entry
SPX_GAP_DOWN_PCT = -1.0   # Minimum gap-down % at open vs prior close

# ============ HELPER FUNCTIONS ============
def get_support_level(stock_data, lookback=20):
    """Find recent support level (lowest low in lookback period)"""
    recent_low = stock_data['Low'].tail(lookback).min()
    current_price = stock_data['Close'].iloc[-1]
    distance_to_support = ((current_price - recent_low) / current_price) * 100
    return recent_low, distance_to_support

def calculate_signal_strength(rsi, bb_pos, vol_surge, atr_pct):
    """Calculate signal strength score (0-100)"""
    score = 0

    # RSI scoring (30 points max)
    if rsi < 25:
        score += 30
    elif rsi < 30:
        score += 25
    elif rsi < 35:
        score += 20
    else:
        score += 10

    # Bollinger Band position (30 points max)
    if bb_pos < 0.15:
        score += 30
    elif bb_pos < 0.25:
        score += 25
    elif bb_pos < 0.35:
        score += 20
    else:
        score += 10

    # Volume surge (25 points max)
    if vol_surge > 2.0:
        score += 25
    elif vol_surge > 1.5:
        score += 20
    elif vol_surge > 1.2:
        score += 15
    else:
        score += 10

    # ATR / Volatility (15 points max)
    if atr_pct > 3.0:
        score += 15
    elif atr_pct > 2.0:
        score += 12
    elif atr_pct > 1.5:
        score += 10
    else:
        score += 5

    return score


def screen_spx():
    """
    SPX-specific screening logic.
    Entry signal: gap-down >= -1% at open vs prior close AND RSI < 30.
    SPX options (CBOE) are European-style: no early assignment risk.
    Returns a dict with SPX signal data if conditions are met, else empty dict.
    """
    logger.info("\n>>> Screening SPX (^GSPC) — European-style, Put Spread Specialist <<<")
    results = {}

    try:
        stock_data = yf.download(SPX_TICKER, period='1y', interval='1d', progress=False, group_by=False)

        # Normalize column names
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.droplevel(0)
        stock_data.columns = [c.capitalize() for c in stock_data.columns]

        if stock_data.empty or len(stock_data) < 200:
            logger.warning("[SPX] Insufficient data. Skipping.")
            return results

        # ---- Indicators ----
        stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()
        stock_data['AVG_VOL_50'] = stock_data['Volume'].rolling(window=50).mean()

        stock_data.ta.rsi(length=RSI_PERIOD, append=True)
        rsi_col = f'RSI_{RSI_PERIOD}'

        stock_data['BB_middle'] = stock_data['Close'].rolling(BB_PERIOD).mean()
        stock_data['BB_std'] = stock_data['Close'].rolling(BB_PERIOD).std()
        stock_data['BB_upper'] = stock_data['BB_middle'] + (stock_data['BB_std'] * 2)
        stock_data['BB_lower'] = stock_data['BB_middle'] - (stock_data['BB_std'] * 2)
        stock_data['BB_position'] = (
            (stock_data['Close'] - stock_data['BB_lower']) /
            (stock_data['BB_upper'] - stock_data['BB_lower'])
        )

        stock_data.ta.atr(length=ATR_PERIOD, append=True)
        atr_col = f'ATR_{ATR_PERIOD}'

        stock_data.ta.macd(append=True)

        required_cols = [rsi_col, atr_col, 'MACDh_12_26_9']
        if not all(col in stock_data.columns for col in required_cols):
            logger.warning("[SPX] TA indicators missing. Skipping.")
            return results

        # ---- Latest Values ----
        latest_close      = stock_data['Close'].iloc[-1]
        prior_close       = stock_data['Close'].iloc[-2]
        latest_open       = stock_data['Open'].iloc[-1]
        latest_sma_200    = stock_data['SMA_200'].iloc[-1]
        latest_volume     = stock_data['Volume'].iloc[-1]
        latest_avg_vol_50 = stock_data['AVG_VOL_50'].iloc[-1]
        current_rsi       = stock_data[rsi_col].iloc[-1]
        latest_bb_pos     = stock_data['BB_position'].iloc[-1]
        latest_bb_lower   = stock_data['BB_lower'].iloc[-1]
        latest_bb_upper   = stock_data['BB_upper'].iloc[-1]
        latest_atr        = stock_data[atr_col].iloc[-1]
        macd_histogram    = stock_data['MACDh_12_26_9'].iloc[-1]

        atr_pct            = (latest_atr / latest_close) * 100
        volume_surge_ratio = latest_volume / latest_avg_vol_50
        support_price, pct_above_support = get_support_level(stock_data)

        # ---- SPX-Specific Entry Conditions ----
        # Gap-down: today's open vs prior day's close
        gap_down_pct = ((latest_open - prior_close) / prior_close) * 100
        is_gap_down  = gap_down_pct <= SPX_GAP_DOWN_PCT   # e.g. <= -1.0%
        is_rsi_low   = current_rsi < SPX_RSI_THRESHOLD    # RSI < 30
        is_uptrend   = latest_close > latest_sma_200

        logger.info(
            f"[SPX] Gap-down: {gap_down_pct:.2f}% (threshold: {SPX_GAP_DOWN_PCT}%) | "
            f"RSI: {current_rsi:.1f} (threshold: {SPX_RSI_THRESHOLD}) | "
            f"Above SMA200: {is_uptrend}"
        )

        if is_gap_down and is_rsi_low and is_uptrend:
            signal_strength = calculate_signal_strength(
                current_rsi, latest_bb_pos, volume_surge_ratio, atr_pct
            )

            results['SPX'] = {
                'Tier':                   'TIER1_CORE',
                'Signal_Strength':        signal_strength,
                'RSI':                    round(current_rsi, 2),
                'Price':                  round(latest_close, 2),
                'Prior_Close':            round(prior_close, 2),
                'Open':                   round(latest_open, 2),
                'Gap_Down_%':             round(gap_down_pct, 2),
                'SMA_200':                round(latest_sma_200, 2),
                'BB_Position':            round(latest_bb_pos, 2),
                'BB_Lower':               round(latest_bb_lower, 2),
                'BB_Upper':               round(latest_bb_upper, 2),
                'ATR_%':                  round(atr_pct, 2),
                'Vol_Surge':              round(volume_surge_ratio, 2),
                'Support':                round(support_price, 2),
                'Distance_to_Support_%':  round(pct_above_support, 1),
                'MACD_Histogram':         round(macd_histogram, 3),
                'Note':                   'European-style. No early assignment. CBOE SPX options only.',
            }

            logger.info(
                f"✓ [SPX] Gap {gap_down_pct:.2f}% | RSI {current_rsi:.1f} | "
                f"Signal: {signal_strength}/100 | **PUT SPREAD SIGNAL**"
            )
        else:
            logger.info("[SPX] Conditions not met. No signal.")

    except Exception as e:
        logger.error(f"[SPX] Error: {e}")

    return results


def screen_tickers(tickers, tier_label):
    """
    Run the screening logic on a list of tickers.
    Returns a dict of {ticker: metrics} for tickers that pass all filters.
    """
    results = {}
    error_count = 0
    successful_count = 0

    for ticker in tickers:
        try:
            stock_data = yf.download(ticker, period='1y', interval='1d', progress=False, group_by=False)

            # Normalize column names
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = stock_data.columns.droplevel(0)
            stock_data.columns = [c.capitalize() for c in stock_data.columns]

            if 'Close' not in stock_data.columns or stock_data.empty or len(stock_data) < 200:
                logger.warning(f"[{tier_label}] Insufficient data for {ticker}. Skipping.")
                error_count += 1
                continue

            # ---- Indicators ----
            stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()
            stock_data['AVG_VOL_50'] = stock_data['Volume'].rolling(window=50).mean()

            stock_data.ta.rsi(length=RSI_PERIOD, append=True)
            rsi_col = f'RSI_{RSI_PERIOD}'

            stock_data['BB_middle'] = stock_data['Close'].rolling(BB_PERIOD).mean()
            stock_data['BB_std'] = stock_data['Close'].rolling(BB_PERIOD).std()
            stock_data['BB_upper'] = stock_data['BB_middle'] + (stock_data['BB_std'] * 2)
            stock_data['BB_lower'] = stock_data['BB_middle'] - (stock_data['BB_std'] * 2)
            stock_data['BB_position'] = (
                (stock_data['Close'] - stock_data['BB_lower']) /
                (stock_data['BB_upper'] - stock_data['BB_lower'])
            )

            stock_data.ta.atr(length=ATR_PERIOD, append=True)
            atr_col = f'ATR_{ATR_PERIOD}'

            stock_data.ta.macd(append=True)

            required_cols = [rsi_col, atr_col, 'MACDh_12_26_9']
            if not all(col in stock_data.columns for col in required_cols):
                logger.warning(f"[{tier_label}] TA indicators missing for {ticker}. Skipping.")
                error_count += 1
                continue

            # ---- Extract Latest Values ----
            latest_close      = stock_data['Close'].iloc[-1]
            latest_sma_200    = stock_data['SMA_200'].iloc[-1]
            latest_volume     = stock_data['Volume'].iloc[-1]
            latest_avg_vol_50 = stock_data['AVG_VOL_50'].iloc[-1]
            current_rsi       = stock_data[rsi_col].iloc[-1]
            latest_bb_pos     = stock_data['BB_position'].iloc[-1]
            latest_bb_lower   = stock_data['BB_lower'].iloc[-1]
            latest_bb_upper   = stock_data['BB_upper'].iloc[-1]
            latest_atr        = stock_data[atr_col].iloc[-1]
            macd_histogram    = stock_data['MACDh_12_26_9'].iloc[-1]

            atr_pct            = (latest_atr / latest_close) * 100
            volume_surge_ratio = latest_volume / latest_avg_vol_50
            support_price, pct_above_support = get_support_level(stock_data)

            # ---- Screening Filters ----
            is_oversold      = current_rsi < RSI_THRESHOLD
            is_uptrend_long  = latest_close > latest_sma_200
            is_liquid        = latest_volume > latest_avg_vol_50
            is_near_lower_bb = latest_bb_pos < 0.4
            is_adequate_vol  = atr_pct > 1.0
            is_volume_surge  = volume_surge_ratio > 1.2

            if (is_oversold and is_uptrend_long and is_liquid and
                    is_near_lower_bb and is_adequate_vol and is_volume_surge):

                signal_strength = calculate_signal_strength(
                    current_rsi, latest_bb_pos, volume_surge_ratio, atr_pct
                )

                results[ticker] = {
                    'Tier':                   tier_label,
                    'Signal_Strength':        signal_strength,
                    'RSI':                    round(current_rsi, 2),
                    'Price':                  round(latest_close, 2),
                    'SMA_200':                round(latest_sma_200, 2),
                    'BB_Position':            round(latest_bb_pos, 2),
                    'BB_Lower':               round(latest_bb_lower, 2),
                    'BB_Upper':               round(latest_bb_upper, 2),
                    'ATR_%':                  round(atr_pct, 2),
                    'Vol_Surge':              round(volume_surge_ratio, 2),
                    'Support':                round(support_price, 2),
                    'Distance_to_Support_%':  round(pct_above_support, 1),
                    'MACD_Histogram':         round(macd_histogram, 3),
                }

                logger.info(
                    f"✓ [{tier_label}] {ticker}: RSI {current_rsi:.1f} | "
                    f"BB {latest_bb_pos:.2f} | Signal: {signal_strength}/100 | **SIGNAL FOUND**"
                )

            successful_count += 1

        except Exception as e:
            logger.error(f"[{tier_label}] Error processing {ticker}: {e}")
            error_count += 1

    logger.info(
        f"[{tier_label}] Done — {successful_count} analyzed, "
        f"{len(results)} signals, {error_count} errors"
    )
    return results


def run_screener():
    logger.info("=" * 70)
    logger.info("OPTIONS PREMIUM SCREENER — WINNING STOCKS PRIORITY MODE")
    logger.info("=" * 70)
    logger.info(f"RSI Threshold (general) : {RSI_THRESHOLD}")
    logger.info(f"RSI Threshold (SPX)     : {SPX_RSI_THRESHOLD} + gap-down >= {SPX_GAP_DOWN_PCT}%")
    logger.info(f"Tier 1 (Core)           : SPX + {', '.join(TIER1_CORE)}")
    logger.info(f"Tier 2 (Watch)          : {', '.join(TIER2_WATCHLIST)}")
    logger.info("=" * 70)

    all_results = {}

    # 0. SPX first — specialist logic
    spx_result = screen_spx()
    all_results.update(spx_result)

    # 1. Tier 1 core stocks
    logger.info("\n>>> Screening TIER 1 — Core Winning Stocks <<<")
    tier1_results = screen_tickers(TIER1_CORE, tier_label="TIER1_CORE")
    all_results.update(tier1_results)

    # 2. Tier 2 watchlist
    logger.info("\n>>> Screening TIER 2 — Watchlist <<<")
    tier2_results = screen_tickers(TIER2_WATCHLIST, tier_label="TIER2_WATCH")
    all_results.update(tier2_results)

    # ---- Summary ----
    logger.info("=" * 70)
    logger.info("SCREENING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total signals found : {len(all_results)}")
    logger.info(f"  SPX signals       : {len(spx_result)}")
    logger.info(f"  Tier 1 signals    : {len(tier1_results)}")
    logger.info(f"  Tier 2 signals    : {len(tier2_results)}")

    if all_results:
        results_df = pd.DataFrame.from_dict(all_results, orient='index')
        results_df.index.name = 'Ticker'

        # Sort: SPX first (Tier1), then Tier1 by signal strength, then Tier2
        tier_order = {'TIER1_CORE': 0, 'TIER2_WATCH': 1}
        results_df['_tier_rank'] = results_df['Tier'].map(tier_order)
        # SPX always floats to top within Tier 1
        results_df['_spx_first'] = (results_df.index == 'SPX').astype(int) * -1
        results_df = results_df.sort_values(
            ['_tier_rank', '_spx_first', 'Signal_Strength'],
            ascending=[True, True, False]
        ).drop(columns=['_tier_rank', '_spx_first'])

        results_df['Scan_Date'] = datetime.now().strftime('%Y-%m-%d')
        results_df['Scan_Time'] = datetime.now().strftime('%H:%M:%S')

        output_file = f'signals_{datetime.now().strftime("%Y%m%d")}.csv'
        results_df.to_csv(output_file)

        logger.info(f"\nResults saved to: {output_file}")
        logger.info(f"  Avg Signal Strength : {results_df['Signal_Strength'].mean():.1f}/100")
        logger.info(f"  Avg RSI             : {results_df['RSI'].mean():.2f}")
        logger.info(f"  Price range         : ${results_df['Price'].min():.2f} - ${results_df['Price'].max():.2f}")
        logger.info(f"  Avg ATR             : {results_df['ATR_%'].mean():.2f}%")
        logger.info(f"  Avg Volume Surge    : {results_df['Vol_Surge'].mean():.2f}x")

        logger.info("\n[TOP SIGNALS — SPX FIRST, THEN TIER 1, BY SIGNAL STRENGTH]")
        display_cols = ['Tier', 'Signal_Strength', 'RSI', 'Price', 'BB_Position', 'ATR_%', 'Vol_Surge']
        # Add Gap_Down_% for SPX row if present
        if 'Gap_Down_%' in results_df.columns:
            display_cols.insert(4, 'Gap_Down_%')
        print(results_df[display_cols])

    else:
        logger.info("No stocks found meeting the criteria today.")

    logger.info("\nScreening complete. Check log file for full details.")
    return all_results


if __name__ == "__main__":
    run_screener()
