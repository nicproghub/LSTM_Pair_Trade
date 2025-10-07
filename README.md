# LSTM Pair Trading Backtest Pipeline

## Overview
This project implements a **complete statistical-arbitrage pipeline** for **LSTM-based pair trading** on cointegrated futures spreads.  
It integrates econometric testing, feature engineering, deep learning forecasting, signal generation, and backtesting within a modular Python framework.  
The system is designed for research and production-oriented evaluation of **mean-reversion trading strategies** across multi-asset markets such as equity indices, commodities, and FX.


## Key Features

- **Cointegration & Hedge Ratio Estimation:**  
  Conducts Engle–Granger tests and dynamically estimates hedge ratios using rolling-window OLS.

- **Feature Engineering:**  
  Generates advanced indicators including z-scores, volatility ratios, momentum metrics, and technical signals (RSI, MACD, Bollinger Bands).

- **LSTM Forecasting Model:**  
  Trains a multi-layer **LSTM neural network** to forecast spread behavior.  
  Incorporates early stopping, learning-rate scheduling, and walk-forward validation for robust predictive performance.

- **Signal Generation Logic:**  
  Derives mean-reversion entry/exit signals based on predicted vs. observed z-scores with confidence thresholds.  
  Includes position limits, stop-loss rules, and execution lags.

- **Backtesting Engine:**  
  Simulates realistic trading environments with transaction-cost modeling and detailed P&L attribution.  
  Reports Sharpe, Sortino, Calmar ratios, max drawdown, and profit factor.

- **Visualization & Reporting:**  
  Provides cumulative returns, drawdown plots, and trade-level analytics with Matplotlib and Seaborn.

## Core Workflow

1. **Data Input:** Load or generate two correlated price series.  
2. **Cointegration Testing:** Validate long-term equilibrium relationships.  
3. **Feature Engineering:** Compute statistical and technical predictors.  
4. **Model Training:** Train LSTM with configurable lookback and forecast horizons.  
5. **Signal Generation:** Generate mean-reversion trades using predictive confidence.  
6. **Backtesting:** Evaluate performance and visualize strategy diagnostics.

- **Visualization:** Matplotlib, Seaborn  
- **Utilities:** TimeSeriesSplit, EarlyStopping, ReduceLROnPlateau
