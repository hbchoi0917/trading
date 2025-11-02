import yfinance as yf
import pandas_ta as ta
import pandas as pd

# 1. Define the list of stock tickers to check: In the first place, tried listing the individual tickers interested but changed to S&P 500 stocks for more candidates selection
# FAV_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA','CLS','VOO','QQQM','QLD','IWM','AMD','STX','MARA','HIMS','ANET','ARM','LRCX','MP','OKLO','ORCL','AMAT','IONQ','RGTI','CRWD','NFLX','META','AVGO','MU','PLTR'] 

def get_sp500_tickers():
    """Scrapes the current list of S&P 500 tickers from Wikipedia, or uses a robust fallback."""
    # Robust fallback list below (my win list of put options)
    FALLBACK_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA','CLS','VOO','QQQM','QLD','IWM','AMD','STX','MARA','HIMS','ANET','ARM','LRCX','MP','OKLO','ORCL','AMAT','IONQ','RGTI','CRWD','NFLX','META','AVGO','MU','PLTR'] 

    try:
        # Pass a User-Agent to mimic a browser
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', 
                              header=0, 
                              storage_options={'User-Agent': 'Mozilla/5.0'} 
                             )
        
        sp500_df = tables[0]
        tickers = sp500_df['Symbol'].tolist()
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        
        print(f"✅ Successfully pulled {len(tickers)} S&P 500 tickers from Wikipedia.")
        return tickers

    except Exception as e:
        print(f"⚠️ Error pulling S&P 500 list from Wikipedia: {e}. Using a fallback list ({len(FALLBACK_TICKERS)} tickers).")
        return FALLBACK_TICKERS


# 2. Define the screening parameters to filter 
TICKERS = get_sp500_tickers()
RSI_PERIOD = 14
RSI_THRESHOLD = 35 

oversold_stocks = {}

print("\n--- Starting Options Premium Screener (Simple & Solid) ---")
print(f"Criteria: RSI < {RSI_THRESHOLD}, Price > SMA 200, AND Active Volume")
print("------------------------------------------------------------------")

for ticker in TICKERS:
    try:
        # Optimized: Download data once (1 year)
        stock_data = yf.download(ticker, period='1y', interval='1d', progress=False, group_by=False) 

        # Fix: Flatten multi index columns & capitalize
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.droplevel(0)
        
        stock_data.columns = [c.capitalize() for c in stock_data.columns]
        
        # Ensure we have enough data (at least 200 days for the SMA) and a Close column
        if 'Close' not in stock_data.columns or stock_data.empty or len(stock_data) < 200:
            print(f"⚠️ Data error: Insufficient data for {ticker}. Skipping.")
            continue
        
        # 3. Calculate SMA, RSI, and Average Volume
        stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()
        stock_data['AVG_VOL_50'] = stock_data['Volume'].rolling(window=50).mean() 
        stock_data.ta.rsi(length=RSI_PERIOD, append=True)
        rsi_column_name = f'RSI_{RSI_PERIOD}'
        
        # 4. Extract latest values
        latest_close = stock_data['Close'].iloc[-1]
        latest_sma_200 = stock_data['SMA_200'].iloc[-1]
        latest_volume = stock_data['Volume'].iloc[-1] 
        latest_avg_vol_50 = stock_data['AVG_VOL_50'].iloc[-1] 
        current_rsi = stock_data[rsi_column_name].iloc[-1].round(2)
        
        # 5. Check the combined conditions
        is_oversold = current_rsi < RSI_THRESHOLD
        is_uptrend_long = latest_close > latest_sma_200
        is_liquid = latest_volume > latest_avg_vol_50 

        # FINAL SCREENING CRITERIA: RSI (Oversold) AND Bullish Trend AND Liquidity
        if is_oversold and is_uptrend_long and is_uptrend_short and is_liquid:
            oversold_stocks[ticker] = {
                'RSI': current_rsi,
                'Price': latest_close.round(2),
                'SMA_200': latest_sma_200.round(2),
                'Volume': latest_volume, 
                'Avg_Vol_50': latest_avg_vol_50.round(0)
            }
            print(f"✅ {ticker}: RSI {current_rsi} | **SIGNAL FOUND**")
        else:
            print(f"   {ticker}: RSI is {current_rsi} (Did not meet all criteria)")

    except Exception as e:
        print(f"❌ An unexpected error occurred while processing {ticker}: {e}")

# FINAL SUMMARY AND CSV SAVING
print("\n--- Summary of Stocks Meeting ALL Premium Collection Criteria ---")

if oversold_stocks:
    results_df = pd.DataFrame.from_dict(oversold_stocks, orient='index')
    results_df.index.name = 'Ticker'
    output_file = 'options_premium_signals.csv'
    
    results_df.to_csv(output_file)
    
    print(f"\n**Found {len(oversold_stocks)} stocks meeting the criteria:**")
    print(f"Results saved to **{output_file}**.")
    
    print("\n[Preview of Results]")
    print(results_df)

else:
    print("No stocks found meeting the combined premium criteria today.")
    
print("---------------------------------------")
