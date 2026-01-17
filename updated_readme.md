# Enhanced Options Premium Screener

**Automated multi-factor technical analysis pipeline for S&P 500 options trading**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-2.0+-green.svg)](https://pandas.pydata.org/)

## 🎯 Project Overview

Production-ready Python screener analyzing **500+ S&P 500 stocks daily** using 7+ technical indicators to identify high-probability candidates for Vertical Put Spread (Credit Put Spread) options strategies.

### Key Features
- **Multi-factor technical analysis**: RSI, Bollinger Bands, ATR, MACD, Volume Analysis
- **Smart signal scoring**: 0-100 signal strength algorithm
- **Automated data quality**: Error handling and logging
- **Visual analytics**: Comprehensive charts and statistics
- **Production-grade**: Timestamped outputs, configurable parameters

## 📊 Technical Indicators

| Indicator | Purpose | Threshold |
|-----------|---------|-----------|
| **RSI (14)** | Oversold detection | < 35 |
| **Bollinger Bands** | Price extremes | Lower 30% of range |
| **SMA (200)** | Long-term trend | Price > SMA |
| **ATR %** | Volatility measurement | > 1.5% |
| **Volume Surge** | Liquidity confirmation | > 1.2x average |
| **Support Level** | Risk assessment | Within 8% |
| **MACD** | Momentum confirmation | Histogram analysis |

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/trading.git
cd trading
pip install -r requirements.txt
```

### Run Screener
```bash
python options_premium_screener.py
```

### Generate Analytics
```bash
python analyze_signals.py
```

## 📈 Sample Output

```
SCREENING STATISTICS
=====================
Total tickers analyzed: 503
Successful analyses: 498
Signals generated: 23
Signal rate: 4.6%

SIGNAL STATISTICS:
  Average Signal Strength: 67.3/100
  Average RSI: 31.4
  Median Price: $145.67
  Average ATR: 2.3%
  Average Volume Surge: 1.8x
```

**Top Signals (Example)**:
| Ticker | Signal Strength | RSI | Price | BB Position | ATR % |
|--------|----------------|-----|-------|-------------|-------|
| AAPL | 89 | 28.3 | $182.45 | 0.18 | 2.8% |
| MSFT | 85 | 29.1 | $405.23 | 0.22 | 2.4% |
| NVDA | 82 | 31.2 | $875.30 | 0.25 | 3.2% |

## 🏗️ Architecture

```
┌─────────────────┐
│  Wikipedia API  │  ← S&P 500 ticker list
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  yfinance API   │  ← Historical price data (1 year)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Technical Analysis Engine       │
│  • RSI, BB, SMA, ATR, MACD      │
│  • Volume analysis              │
│  • Support level detection      │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Multi-Filter Screening Logic   │
│  • 7 concurrent conditions      │
│  • Signal strength scoring      │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Output Layer                   │
│  • CSV: signals_YYYYMMDD.csv   │
│  • LOG: screener_YYYYMMDD.log  │
│  • PNG: signal_analysis.png    │
└─────────────────────────────────┘
```

## 🔍 Screening Logic

### Primary Filters (All Must Pass)
```python
1. RSI < 35                      # Oversold condition
2. Price > SMA(200)              # Bullish long-term trend
3. Volume > AVG_VOL(50)          # Liquidity confirmation
4. BB_Position < 0.3             # Near lower Bollinger Band
5. ATR% > 1.5                    # Adequate volatility
6. Volume_Surge > 1.2x           # Above-average activity
```

### Signal Strength Algorithm
```
Score Components (0-100):
• RSI extremity       → 0-30 points
• BB position         → 0-30 points
• Volume surge        → 0-25 points
• ATR/Volatility      → 0-15 points
```

## 📁 Project Structure

```
trading/
├── options_premium_screener.py   # Main screening engine
├── analyze_signals.py            # Visualization script
├── requirements.txt              # Dependencies
├── signals_YYYYMMDD.csv         # Daily outputs
├── screener_YYYYMMDD.log        # Execution logs
├── signal_analysis_YYYYMMDD.png # Analytics charts
└── README.md
```

## 💡 Business Strategy

**Use Case**: Identify stocks temporarily oversold but in strong uptrends for selling put credit spreads.

**Risk Management**:
- Long-term uptrend (SMA 200) reduces downside risk
- Bollinger Band position confirms mean reversion opportunity
- Volume surge indicates institutional interest
- Support level proximity provides safety margin

**Typical Trade Setup** (example):
```
Stock: AAPL at $182 (RSI: 28, BB Position: 0.18)
Strategy: Sell $175/$170 Put Credit Spread
Premium: ~$0.80 ($80 per contract)
Max Risk: $4.20 ($420 per contract)
Risk/Reward: ~5:1
```

## 📊 Visualization Examples

The `analyze_signals.py` script generates:
1. **Signal Strength Rankings** - Top 10 opportunities
2. **RSI Distribution** - Oversold intensity analysis
3. **Bollinger Band Positions** - Mean reversion setup
4. **ATR Analysis** - Volatility assessment
5. **Volume Surge Correlation** - Liquidity quality
6. **Price vs SMA** - Trend confirmation
7. **Support Distance** - Risk proximity

## 🎓 Technical Skills Demonstrated

- **Data Engineering**: ETL pipeline (Wikipedia → yfinance → pandas)
- **Statistical Analysis**: Multi-factor technical indicators
- **Error Handling**: Robust exception management and logging
- **Code Quality**: Modular functions, type safety, documentation
- **Data Visualization**: matplotlib/seaborn analytics
- **Version Control**: Git workflow, proper .gitignore

## 🚀 Future Enhancements

- [ ] Implied Volatility (IV) integration
- [ ] Backtesting framework with historical performance
- [ ] Email/Slack alerts for new signals
- [ ] SQLite database for historical tracking
- [ ] Web dashboard (Streamlit/Flask)
- [ ] Machine learning signal optimization

## 📝 Dependencies

- **pandas**: Data manipulation and analysis
- **yfinance**: Market data acquisition
- **pandas-ta**: Technical analysis indicators
- **matplotlib/seaborn**: Data visualization
- **lxml/html5lib**: Web scraping utilities

## 🤝 Contributing

This is a portfolio project, but feedback and suggestions are welcome!

## 📄 License

MIT License - See LICENSE file for details

## 👤 Author

**Hanbit Choi**
- LinkedIn: [linkedin.com/in/hanbitchoi0917](https://www.linkedin.com/in/hanbitchoi0917/)
- GitHub: [@hbchoi0917](https://github.com/hbchoi0917)

---

**⚠️ Disclaimer**: This tool is for educational and research purposes only. Not financial advice. Options trading involves substantial risk.
