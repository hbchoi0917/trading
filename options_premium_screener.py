import yfinance as yf
import pandas_ta as ta
import pandas as pd
import logging
from datetime import datetime

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

# ============ S&P 500 TICKER RETRIEVAL ============
def get_sp500_tickers():
    """Scrapes the current list of S&P 500 tickers from Wikipedia, or uses a robust fallback."""
    FALLBACK_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA','CLS','VOO','QQQM','QLD','IWM','AMD','STX','MARA','HIMS','ANET','ARM','LRCX','MP','OKLO','ORCL','AMAT','IONQ','RGTI','CRWD','NFLX','META','AVGO','MU','PLTR'] 

    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', 
                              header=0, 
                              storage_options={'User-Agent': 'Mozilla/5.0'})
        
        sp500_df = tables[0]
        tickers = sp500_df['Symbol'].tolist()
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        
        logger.info(f"Successfully pulled {len(tickers)} S&P 500 tickers from Wikipedia")
        return tickers

    except Exception as e:
        logger.warning(f"Error pulling S&P 500 list from Wikipedia: {e}. Using fallback list ({len(FALLBACK_TICKERS)} tickers)")
        return FALLBACK_TICKERS

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
    
    # ATR/Volatility (15 points max)
    if atr_pct > 3.0:
        score += 15
    elif atr_pct > 2.0:
        score += 12
    elif atr_pct > 1.5:
        score += 10
    else:
        score += 5
    
    return score

# ============ SCREENING PARAMETERS ============
TICKERS = get_sp500_tickers()
RSI_PERIOD = 14
RSI_THRESHOLD = 35
BB_PERIOD = 20
ATR_PERIOD = 14

oversold_stocks = {}
error_count = 0
successful_count = 0

logger.info("=" * 70)
logger.info("ENHANCED OPTIONS PREMIUM SCREENER")
logger.info("=" * 70)
logger.info(f"Criteria: RSI < {RSI_THRESHOLD}, Price > SMA 200, Active Volume")
logger.info(f"Enhanced Filters: Bollinger Bands, ATR, Volume Surge, Support Levels")
logger.info("=" * 70)

# ============ MAIN SCREENING LOOP ============
for ticker in TICKERS:
    try:
        # Download 1 year of data
        stock_data = yf.download(ticker, period='1y', interval='1d', progress=False, group_by=False)

        # Fix column names
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.droplevel(0)
        stock_data.columns = [c.capitalize() for c in stock_data.columns]
        
        # Data validation
        if 'Close' not in stock_data.columns or stock_data.empty or len(stock_data) < 200:
            logger.warning(f"Insufficient data for {ticker}. Skipping.")
            error_count += 1
            continue
        
        # ============ CALCULATE INDICATORS ============
        # Moving Averages
        stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()
        stock_data['AVG_VOL_50'] = stock_data['Volume'].rolling(window=50).mean()
        
        # RSI
        stock_data.ta.rsi(length=RSI_PERIOD, append=True)
        rsi_column_name = f'RSI_{RSI_PERIOD}'
        
        # Bollinger Bands
        stock_data['BB_middle'] = stock_data['Close'].rolling(BB_PERIOD).mean()
        stock_data['BB_std'] = stock_data['Close'].rolling(BB_PERIOD).std()
        stock_data['BB_upper'] = stock_data['BB_middle'] + (stock_data['BB_std'] * 2)
        stock_data['BB_lower'] = stock_data['BB_middle'] - (stock_data['BB_std'] * 2)
        stock_data['BB_position'] = (stock_data['Close'] - stock_data['BB_lower']) / (stock_data['BB_upper'] - stock_data['BB_lower'])
        
        # ATR (Average True Range)
        stock_data.ta.atr(length=ATR_PERIOD, append=True)
        atr_column_name = f'ATR_{ATR_PERIOD}'
        
        # MACD
        stock_data.ta.macd(append=True)
        
        # ============ EXTRACT LATEST VALUES ============
        latest_close = stock_data['Close'].iloc[-1]
        latest_sma_200 = stock_data['SMA_200'].iloc[-1]
        latest_volume = stock_data['Volume'].iloc[-1]
        latest_avg_vol_50 = stock_data['AVG_VOL_50'].iloc[-1]
        current_rsi = stock_data[rsi_column_name].iloc[-1]
        latest_bb_position = stock_data['BB_position'].iloc[-1]
        latest_bb_lower = stock_data['BB_lower'].iloc[-1]
        latest_bb_upper = stock_data['BB_upper'].iloc[-1]
        latest_atr = stock_data[atr_column_name].iloc[-1]
        
        # MACD values
        macd_histogram = stock_data['MACDh_12_26_9'].iloc[-1]
        
        # Calculated metrics
        atr_pct = (latest_atr / latest_close) * 100
        volume_surge_ratio = latest_volume / latest_avg_vol_50
        support_price, pct_above_support = get_support_level(stock_data)
        
        # ============ SCREENING FILTERS ============
        is_oversold = current_rsi < RSI_THRESHOLD
        is_uptrend_long = latest_close > latest_sma_200
        is_liquid = latest_volume > latest_avg_vol_50
        is_near_lower_bb = latest_bb_position < 0.3  # In lower 30% of BB range
        is_adequate_volatility = atr_pct > 1.5  # At least 1.5% daily range
        is_volume_surge = volume_surge_ratio > 1.2  # 20% above average volume
        is_near_support = pct_above_support < 8  # Within 8% of recent support
        
        # ENHANCED FINAL CRITERIA
        if (is_oversold and 
            is_uptrend_long and 
            is_liquid and 
            is_near_lower_bb and 
            is_adequate_volatility and
            is_volume_surge):
            
            # Calculate signal strength
            signal_strength = calculate_signal_strength(
                current_rsi, 
                latest_bb_position, 
                volume_surge_ratio, 
                atr_pct
            )
            
            oversold_stocks[ticker] = {
                'Signal_Strength': signal_strength,
                'RSI': round(current_rsi, 2),
                'Price': round(latest_close, 2),
                'SMA_200': round(latest_sma_200, 2),
                'BB_Position': round(latest_bb_position, 2),
                'BB_Lower': round(latest_bb_lower, 2),
                'BB_Upper': round(latest_bb_upper, 2),
                'ATR_%': round(atr_pct, 2),
                'Vol_Surge': round(volume_surge_ratio, 2),
                'Support': round(support_price, 2),
                'Distance_to_Support_%': round(pct_above_support, 1),
                'MACD_Histogram': round(macd_histogram, 3)
            }
            
            logger.info(f"✓ {ticker}: RSI {current_rsi:.1f} | BB {latest_bb_position:.2f} | Signal: {signal_strength}/100 | **SIGNAL FOUND**")
        
        successful_count += 1

    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}")
        error_count += 1

# ============ FINAL SUMMARY ============
logger.info("=" * 70)
logger.info("SCREENING STATISTICS")
logger.info("=" * 70)
logger.info(f"Total tickers analyzed: {len(TICKERS)}")
logger.info(f"Successful analyses: {successful_count}")
logger.info(f"Errors encountered: {error_count}")
logger.info(f"Signals generated: {len(oversold_stocks)}")
if len(TICKERS) > 0:
    logger.info(f"Signal rate: {len(oversold_stocks)/len(TICKERS)*100:.2f}%")

if oversold_stocks:
    results_df = pd.DataFrame.from_dict(oversold_stocks, orient='index')
    results_df.index.name = 'Ticker'
    
    # Sort by signal strength (highest first)
    results_df = results_df.sort_values('Signal_Strength', ascending=False)
    
    # Add timestamp
    results_df['Scan_Date'] = datetime.now().strftime('%Y-%m-%d')
    results_df['Scan_Time'] = datetime.now().strftime('%H:%M:%S')
    
    # Save to CSV with timestamp
    output_file = f'signals_{datetime.now().strftime("%Y%m%d")}.csv'
    results_df.to_csv(output_file)
    
    logger.info("=" * 70)
    logger.info(f"Found {len(oversold_stocks)} stocks meeting ALL criteria")
    logger.info(f"Results saved to: {output_file}")
    logger.info("=" * 70)
    
    # Display statistics
    logger.info("\nSIGNAL STATISTICS:")
    logger.info(f"  Average Signal Strength: {results_df['Signal_Strength'].mean():.1f}/100")
    logger.info(f"  Average RSI: {results_df['RSI'].mean():.2f}")
    logger.info(f"  Median Price: ${results_df['Price'].median():.2f}")
    logger.info(f"  Price range: ${results_df['Price'].min():.2f} - ${results_df['Price'].max():.2f}")
    logger.info(f"  Average ATR: {results_df['ATR_%'].mean():.2f}%")
    logger.info(f"  Average Volume Surge: {results_df['Vol_Surge'].mean():.2f}x")
    
    logger.info("\n[TOP 10 SIGNALS BY STRENGTH]")
    print(results_df[['Signal_Strength', 'RSI', 'Price', 'BB_Position', 'ATR_%', 'Vol_Surge']].head(10))

else:
    logger.info("=" * 70)
    logger.info("No stocks found meeting the enhanced criteria today")
    logger.info("=" * 70)

logger.info("\nScreening complete. Check log file for details.")
