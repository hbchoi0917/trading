# Options trading screener

Project Overview
: This is an optimized, automated Python script designed to screen the S&P 500 universe for high-probability, low-risk candidates suitable for a Vertical Put Spread (Credit Put Spread) options strategy.

Technical Filters (The Logic)
: This screener uses a multi-factor technical analysis approach to isolate stocks that are briefly oversold but remain in a strong, confirmed bullish trend. This minimizes downside risk for premium collection strategies.

    RSI(14) < 35	
        Oversold Entry: Identifies short-term price weakness and signals a likely short-term bottoming or bounce.
    Price > SMA(50) & SMA(200)	
        Trend Confirmation: Ensures the stock remains in a strong long-term (200-day) uptrend.
    Volume > AVG_VOL(50)
        Liquidity Check: Ensures the stock is highly liquid, which is essential for options trading to guarantee tight bid-ask spreads and efficient order execution.

Files in this Repository
    options_premium_screener.py: The main, fully optimized Python script containing all data acquisition, analysis, filtering logic, and output routines.
    options_premium_signals.csv: Example output file (ignored by Git) showing the final filtered signals for immediate trading review.
    .gitignore:	Standard file ensuring clean commits by excluding temporary data outputs and environment files.



How to Run the Screener

1.Clone the Repository:
git clone https://github.com/YourUsername/RepoName.git
cd RepoName

2.Install Dependencies:
py -m pip install pandas yfinance pandas-ta html5lib

3.Execute the Script:
py options_premium_screener.py

4.Review Results: The final output will be saved to options_premium_signals.csv in the project directory.
