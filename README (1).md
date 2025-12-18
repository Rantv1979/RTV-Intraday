Rantv Intraday Trading Terminal Pro
Overview
A comprehensive Streamlit-based intraday trading terminal for Indian stock markets (NSE) with real-time market data, technical analysis, signal generation, ML-enhanced predictions, and Kite Connect broker integration.

Current State
Status: Production Ready
Framework: Streamlit + Python 3.11
Port: 5000
Key Features
Real-time Market Data from Yahoo Finance for Nifty 50, Nifty 100, and Midcap 150 stocks
Technical Indicators: EMA, VWAP, RSI, MACD, Bollinger Bands, ADX, ATR, Stochastic
Trading Signals with multi-strategy signal generation and buy/sell recommendations
Market Mood Gauges for Nifty 50 and Bank Nifty sentiment
Paper Trading with simulated trading and P&L tracking
Trading Journal with PostgreSQL database for trade history
Risk Management with position sizing, stop-loss, and target calculations
Kite Connect integration for live order execution (requires API keys)
ML Signal Enhancement using RandomForest model
Backtesting Engine for strategy validation
Alert System for real-time notifications
Portfolio Analytics with comprehensive P&L analysis
Algo Trading Engine - Automated order execution with scheduling and risk management
Project Structure
/
├── app.py                    # Main entry point
├── RANTV_FINAL_OPTIMIZED.py  # Full application code (5200+ lines)
├── algo_engine.py            # Algo Trading Engine module
├── .streamlit/
│   └── config.toml           # Streamlit configuration (port 5000)
├── data/                     # Data directory for models and databases
└── requirements.txt          # Python dependencies

Running the Application
The app runs via Streamlit on port 5000:

streamlit run app.py

Environment Variables (Optional)
DATABASE_URL: PostgreSQL connection string (auto-provided by Replit)
KITE_API_KEY: Kite Connect API key (for live trading)
KITE_API_SECRET: Kite Connect API secret (for live trading)
Stock Universes
NIFTY 50: Top 50 large-cap stocks
NIFTY 100: Top 100 stocks
MIDCAP 150: Mid-cap stocks with high intraday potential
All Stocks: Combined universe (~200 unique stocks)
Tabs Overview
Dashboard - Account summary and strategy performance
Signals - Live trading signals with confidence scores
Paper Trading - Simulated trading interface
Trade History - Historical trades and journal
RSI Extreme - Oversold/overbought scanner
Backtest - Strategy validation against historical data
Strategies - Strategy configuration and details
High Accuracy Scanner - Premium signal detection
Kite Live Charts - Broker chart integration
Portfolio Analytics - P&L analysis and risk metrics
Algo Trading - Automated trading control panel
Algo Trading Module
The algo trading engine provides automated order execution with:

Background Scheduler: Scans for signals every 60 seconds during market hours
Risk Controls: Position limits, daily loss limits, emergency stop
Order Management: Automatic placement and tracking of orders
Paper/Live Trading: Switch between paper trading and live execution
Environment Variables for Algo Trading
ALGO_TRADING_ENABLED: Set to "true" to enable (default: false)
ALGO_MAX_POSITIONS: Maximum concurrent positions (default: 5)
ALGO_MAX_DAILY_LOSS: Maximum daily loss limit in INR (default: 50000)
ALGO_MIN_CONFIDENCE: Minimum signal confidence threshold (default: 0.80)
Market Hours (IST)
Market Open: 9:15 AM
Peak Hours: 9:30 AM - 2:30 PM
Market Close: 3:30 PM
Recent Changes
2025-12-18: Added Algo Trading Engine with automated order execution
2025-12-18: Removed hardcoded API credentials - all sensitive data now via environment variables
2025-12-18: Added risk management controls (kill switch, position limits, daily loss limits)
