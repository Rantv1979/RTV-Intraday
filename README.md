# RTV-Intraday
NSE_Stocks_Intraday
# Rantv Intraday Trading Terminal Pro

## Overview
A comprehensive Streamlit-based intraday trading terminal for Indian stock markets (NSE) with real-time market data, technical analysis, signal generation, ML-enhanced predictions, and Kite Connect broker integration.

## Current State
- **Status**: Production Ready
- **Last Updated**: December 16, 2025
- **Framework**: Streamlit + Python 3.11

## Key Features
1. **Real-time Market Data**: Fetches live prices from Yahoo Finance for Nifty 50, Nifty 100, and Midcap 150 stocks
2. **Technical Indicators**: EMA, VWAP, RSI, MACD, Bollinger Bands, ADX, ATR, Stochastic
3. **Trading Signals**: Multi-strategy signal generation with buy/sell recommendations
4. **Market Mood Gauges**: Circular visual gauges for Nifty 50 and Bank Nifty sentiment
5. **Paper Trading**: Simulated trading with P&L tracking
6. **Trading Journal**: PostgreSQL database for trade history and token persistence
7. **Risk Management**: Position sizing, stop-loss, and target calculations
8. **Kite Connect**: Secure OAuth broker integration for live order execution
9. **ML Signal Enhancement**: RandomForest model for signal quality prediction
10. **Backtesting Engine**: Strategy validation against historical data
11. **Alert System**: Real-time notifications for high-confidence signals
12. **Portfolio Analytics**: Comprehensive P&L analysis and risk metrics

## Recent Changes (December 16, 2025)
1. **Enhanced ML Model**: Upgraded MLSignalEnhancer with sklearn RandomForestClassifier
   - Trains on historical trade outcomes
   - Persists model with joblib to `data/signal_quality_model.pkl`
   - 11 feature inputs including RSI, MACD, volume, EMA alignment
   
2. **Backtesting Engine**: Full strategy backtesting implementation
   - Historical data download via yfinance
   - Trade simulation with stop-loss/target execution
   - Performance metrics: win rate, Sharpe ratio, max drawdown
   - Interactive UI with equity curve visualization
   
3. **Alert Manager**: Notification system for trading alerts
   - High-confidence signal alerts (>85% confidence)
   - P&L alerts for significant gains/losses
   - Risk warning alerts
   - Sidebar panel with alert history
   
4. **Portfolio Analytics Tab**: New dashboard for portfolio tracking
   - Real-time P&L (realized and unrealized)
   - Risk metrics (exposure, diversification, cash reserve)
   - Strategy performance breakdown with charts
   - Open position details

5. **Security Improvements**: 
   - API credentials loaded from environment variables (KITE_API_KEY, KITE_API_SECRET)
   - PostgreSQL token persistence with DATABASE_URL
   - Session-based token storage

## Project Structure
```
/
├── app.py                 # Main Streamlit application (5000+ lines)
├── .streamlit/
│   └── config.toml        # Streamlit configuration
├── data/
│   ├── trading_journal.db # SQLite database for trades
│   ├── signal_quality_model.pkl  # ML model
│   └── signal_scaler.pkl  # Feature scaler
└── attached_assets/       # Original source files
```

## Running the Application
```bash
streamlit run app.py --server.port 5000
```

## Stock Universes
- **NIFTY 50**: Top 50 large-cap stocks
- **NIFTY 100**: Top 100 stocks (includes Nifty 50)
- **MIDCAP 150**: Mid-cap stocks with high intraday potential
- **All Stocks**: Combined universe (~200 unique stocks)

## Trading Strategies
### Standard Strategies
- EMA + VWAP Confluence
- RSI Mean Reversion
- Bollinger Band Reversion/Rejection
- MACD Momentum
- Support/Resistance Breakout
- Trend Reversal

### High Accuracy Strategies
- Multi-Confirmation Ultra
- Enhanced EMA-VWAP
- Volume Weighted Breakout
- RSI Divergence
- MACD Trend Momentum

## Signal Quality Filters
- Minimum Confidence: 70% (configurable)
- Minimum Score: 6 (configurable)
- Risk-Reward Ratio: >= 2.5:1
- Volume Confirmation: >= 1.3x average
- ADX Trend Filter: >= 25

## Dependencies
- streamlit, pandas, numpy
- yfinance, plotly
- streamlit-autorefresh
- sqlalchemy, joblib
- kiteconnect
- pytz, requests
- scikit-learn, psycopg2-binary

## Configuration
- Capital: ₹20,00,000
- Trade Allocation: 15%
- Max Daily Trades: 10
- Auto-refresh: Every 2 minutes

## Market Hours
- Market Open: 9:15 AM IST
- Peak Hours: 9:30 AM - 2:30 PM IST
- Market Close: 3:30 PM IST
- Auto-Close: 3:10 PM IST

## Kite Connect Integration
- API Key and Secret loaded from environment variables
- Secure OAuth authentication flow with token persistence
- Historical data and live order execution
- Token storage in PostgreSQL for session persistence

## Environment Variables Required
- `DATABASE_URL`: PostgreSQL connection string (auto-provided by Replit)
- `KITE_API_KEY`: Kite Connect API key (optional, for live trading)
- `KITE_API_SECRET`: Kite Connect API secret (optional, for live trading)

## Tabs Overview
1. **Dashboard**: Account summary and strategy performance
2. **Signals**: Live trading signals with confidence scores
3. **Paper Trading**: Simulated trading interface
4. **Trade History**: Historical trades and journal
5. **RSI Extreme**: Oversold/overbought scanner
6. **Backtest**: Strategy validation against historical data
7. **Strategies**: Strategy configuration and details
8. **High Accuracy Scanner**: Premium signal detection
9. **Kite Live Charts**: Broker chart integration
10. **Portfolio Analytics**: P&L analysis and risk metrics
