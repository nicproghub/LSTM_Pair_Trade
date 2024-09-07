import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class CointegratedSpreadsPipeline:
    """Complete pipeline for statistical arbitrage on cointegrated spreads"""
    
    def __init__(self, lookback=60, forecast_horizon=5, confidence_threshold=0.95):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.confidence_threshold = confidence_threshold
        self.scaler = StandardScaler()
        self.model = None
        self.hedge_ratio_window = 20
        self.transaction_cost = 0.0002  # 2 bps
        self.position_limit = 3  # Max z-score for position entry
        
    def test_cointegration(self, series1, series2, significance=0.05):
        """Test for cointegration between two price series"""
        score, pvalue, _ = coint(series1, series2)
        return pvalue < significance, pvalue
    
    def calculate_hedge_ratio(self, series1, series2, method='ols'):
        """Calculate dynamic hedge ratio using OLS or Kalman filter"""
        if method == 'ols':
            model = sm.OLS(series1, sm.add_constant(series2))
            result = model.fit()
            return result.params[1]
        elif method == 'rolling':
            # Rolling window hedge ratio
            ratios = []
            for i in range(self.hedge_ratio_window, len(series1)):
                window_s1 = series1[i-self.hedge_ratio_window:i]
                window_s2 = series2[i-self.hedge_ratio_window:i]
                model = sm.OLS(window_s1, sm.add_constant(window_s2))
                result = model.fit()
                ratios.append(result.params[1])
            return np.array(ratios)
    
    def create_spread(self, series1, series2, hedge_ratio):
        """Create spread using hedge ratio"""
        return series1 - hedge_ratio * series2
    
    def calculate_zscore(self, spread, window=20):
        """Calculate rolling z-score of spread"""
        spread_mean = spread.rolling(window=window).mean()
        spread_std = spread.rolling(window=window).std()
        zscore = (spread - spread_mean) / spread_std
        return zscore
    
    def engineer_features(self, data):
        """Engineer features for LSTM model"""
        features = pd.DataFrame(index=data.index)
        
        # Price data
        features['spread'] = data['spread']
        features['zscore'] = self.calculate_zscore(data['spread'])
        
        # Technical indicators
        features['rsi'] = self.calculate_rsi(data['spread'])
        features['bb_upper'], features['bb_lower'] = self.bollinger_bands(data['spread'])
        features['macd'], features['signal'] = self.calculate_macd(data['spread'])
        
        # Volatility features
        features['volatility'] = data['spread'].rolling(20).std()
        features['volatility_ratio'] = features['volatility'] / features['volatility'].rolling(60).mean()
        features['atr'] = self.calculate_atr(data)
        
        # Mean reversion indicators
        features['half_life'] = self.calculate_half_life(data['spread'])
        features['hurst'] = self.calculate_hurst_exponent(data['spread'])
        
        # Momentum features
        features['momentum_5'] = data['spread'].pct_change(5)
        features['momentum_10'] = data['spread'].pct_change(10)
        features['momentum_20'] = data['spread'].pct_change(20)
        
        # Volume features if available
        if 'volume1' in data.columns:
            features['volume_ratio'] = data['volume1'] / data['volume2']
            features['volume_ma'] = (data['volume1'] + data['volume2']).rolling(20).mean()
        
        return features.dropna()
    
    def calculate_rsi(self, series, period=14):
        """Calculate RSI indicator"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def bollinger_bands(self, series, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band
    
    def calculate_macd(self, series, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        ema_fast = series.ewm(span=fast).mean()
        ema_slow = series.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line, signal_line
    
    def calculate_atr(self, data, period=14):
        """Calculate Average True Range"""
        if 'high' in data.columns and 'low' in data.columns and 'close' in data.columns:
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(period).mean()
        else:
            # Approximate ATR using spread volatility
            atr = data['spread'].rolling(period).std() * np.sqrt(period)
        return atr
    
    def calculate_half_life(self, spread, window=60):
        """Calculate half-life of mean reversion"""
        half_lives = []
        for i in range(window, len(spread)):
            y = spread[i-window:i].values
            x = spread[i-window:i].shift(1).dropna().values
            y = y[1:]
            
            model = sm.OLS(y - x, x)
            result = model.fit()
            half_life = -np.log(2) / result.params[0] if result.params[0] < 0 else np.nan
            half_lives.append(half_life)
        
        return pd.Series(half_lives, index=spread.index[window:])
    
    def calculate_hurst_exponent(self, series, max_lag=20):
        """Calculate Hurst exponent for mean reversion detection"""
        lags = range(2, max_lag)
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    
    def prepare_lstm_data(self, features, target):
        """Prepare data for LSTM model"""
        X, y = [], []
        for i in range(self.lookback, len(features) - self.forecast_horizon):
            X.append(features[i-self.lookback:i])
            y.append(target[i+self.forecast_horizon])
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, n_features):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(self.lookback, n_features)),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(25, return_sequences=False),
            Dropout(0.2),
            Dense(10, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def generate_signals(self, predictions, features):
        """Generate trading signals with confidence thresholds
        Signals are generated based on current bar but will be executed on next bar
        """
        signals = pd.DataFrame(index=features.index)
        
        # Store current market conditions
        signals['current_zscore'] = features['zscore']
        signals['predicted_zscore'] = predictions
        
        # Calculate confidence for each bar
        confidence = self.calculate_signal_confidence(features)
        signals['confidence'] = confidence
        
        # Initialize position tracking
        signals['position'] = 0.0
        signals['signal_reason'] = ''
        
        # Track current position for proper exit logic
        current_position = 0
        
        for i in range(len(signals)):
            idx = signals.index[i]
            z_score = signals['current_zscore'].iloc[i]
            pred_z = signals['predicted_zscore'].iloc[i]
            conf = signals['confidence'].iloc[i]
            
            # Determine signal for NEXT bar execution
            if current_position == 0:  # No position
                # Entry signals (executed next bar)
                if z_score < -2 and pred_z > z_score and conf > self.confidence_threshold:
                    signals.loc[idx, 'position'] = 1
                    signals.loc[idx, 'signal_reason'] = 'long_entry'
                    current_position = 1
                elif z_score > 2 and pred_z < z_score and conf > self.confidence_threshold:
                    signals.loc[idx, 'position'] = -1
                    signals.loc[idx, 'signal_reason'] = 'short_entry'
                    current_position = -1
                    
            elif current_position == 1:  # Long position
                # Exit conditions for long (executed next bar)
                if z_score > -0.5:  # Mean reversion achieved
                    signals.loc[idx, 'position'] = 0
                    signals.loc[idx, 'signal_reason'] = 'long_exit_target'
                    current_position = 0
                elif z_score < -self.position_limit:  # Stop loss
                    signals.loc[idx, 'position'] = 0
                    signals.loc[idx, 'signal_reason'] = 'long_exit_stop'
                    current_position = 0
                else:
                    signals.loc[idx, 'position'] = current_position
                    
            elif current_position == -1:  # Short position
                # Exit conditions for short (executed next bar)
                if z_score < 0.5:  # Mean reversion achieved
                    signals.loc[idx, 'position'] = 0
                    signals.loc[idx, 'signal_reason'] = 'short_exit_target'
                    current_position = 0
                elif z_score > self.position_limit:  # Stop loss
                    signals.loc[idx, 'position'] = 0
                    signals.loc[idx, 'signal_reason'] = 'short_exit_stop'
                    current_position = 0
                else:
                    signals.loc[idx, 'position'] = current_position
        
        # Add signal statistics
        signals['signal_change'] = signals['position'].diff().fillna(0)
        signals['is_entry'] = signals['signal_change'].abs() > 0
        signals['is_exit'] = (signals['position'] == 0) & (signals['signal_change'] != 0)
        
        return signals
    
    def calculate_signal_confidence(self, features):
        """Calculate confidence score for signals"""
        confidence = pd.Series(index=features.index, data=0.5)
        
        # Mean reversion confidence (lower Hurst = higher confidence)
        if 'hurst' in features.columns:
            hurst_conf = 1 - features['hurst'].clip(0, 1)
            confidence += hurst_conf * 0.2
        
        # Half-life confidence (shorter half-life = higher confidence)
        if 'half_life' in features.columns:
            hl_conf = 1 / (1 + features['half_life'] / 10)
            confidence += hl_conf * 0.2
        
        # Volatility regime confidence
        if 'volatility_ratio' in features.columns:
            vol_conf = 1 / (1 + np.abs(features['volatility_ratio'] - 1))
            confidence += vol_conf * 0.1
        
        return confidence.clip(0, 1)
    
    def backtest(self, data, signals, initial_capital=100000):
        """Backtest strategy with transaction costs and position limits
        IMPORTANT: Signals generated at time t are executed at time t+1 prices
        """
        results = pd.DataFrame(index=signals.index)
        
        # Store raw signals (generated today)
        results['signal_today'] = signals['position']
        
        # Actual positions are taken next bar after signal generation
        # shift(1) means we use yesterday's signal for today's position
        results['position'] = signals['position'].shift(1).fillna(0)
        
        # Calculate spread returns (today's return)
        results['spread_returns'] = data['spread'].pct_change()
        
        # Strategy returns: position from yesterday's signal * today's return
        # This ensures no look-ahead bias
        results['strategy_returns'] = results['position'] * results['spread_returns']
        
        # Detect trades (position changes)
        # Trade happens when position changes, cost applied at execution bar
        results['position_change'] = results['position'].diff().fillna(0)
        results['trades'] = results['position_change'].abs()
        
        # Transaction costs applied when position changes
        # Cost is based on the spread return magnitude for more realistic modeling
        results['transaction_costs'] = results['trades'] * self.transaction_cost
        
        # Net returns after transaction costs
        results['net_returns'] = results['strategy_returns'] - results['transaction_costs']
        
        # Calculate cumulative returns
        results['cumulative_returns'] = (1 + results['net_returns']).cumprod()
        results['cumulative_strategy'] = initial_capital * results['cumulative_returns']
        
        # Calculate drawdown
        results['running_max'] = results['cumulative_strategy'].cummax()
        results['drawdown'] = (results['cumulative_strategy'] - results['running_max']) / results['running_max']
        
        # Add trade statistics
        results['trade_pnl'] = 0
        position_held = 0
        entry_price = 0
        
        for i in range(1, len(results)):
            if results['position_change'].iloc[i] != 0:
                if position_held != 0:  # Closing position
                    exit_price = data['spread'].iloc[i]
                    if position_held > 0:  # Was long
                        results.loc[results.index[i], 'trade_pnl'] = (exit_price - entry_price) / entry_price
                    else:  # Was short
                        results.loc[results.index[i], 'trade_pnl'] = (entry_price - exit_price) / entry_price
                
                if results['position'].iloc[i] != 0:  # Opening new position
                    entry_price = data['spread'].iloc[i]
                    position_held = results['position'].iloc[i]
                else:
                    position_held = 0
        
        return results
    
    def calculate_performance_metrics(self, results):
        """Calculate performance metrics ensuring no look-ahead bias"""
        metrics = {}
        
        # Filter out the first row since position is NaN
        clean_results = results[1:].copy()
        
        # Returns
        total_return = clean_results['cumulative_returns'].iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(clean_results)) - 1
        
        # Risk metrics (using net returns which already account for execution lag)
        daily_returns = clean_results['net_returns'].dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            downside_returns = daily_returns[daily_returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                sortino_ratio = np.sqrt(252) * daily_returns.mean() / downside_returns.std()
            else:
                sortino_ratio = 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
        
        # Drawdown
        max_drawdown = clean_results['drawdown'].min()
        
        # Trade statistics (counting actual executed trades)
        n_trades = (clean_results['position_change'] != 0).sum()
        n_round_trips = n_trades // 2
        
        # Win rate based on executed positions
        positive_returns = clean_results[clean_results['strategy_returns'] > 0]
        negative_returns = clean_results[clean_results['strategy_returns'] < 0]
        if len(positive_returns) + len(negative_returns) > 0:
            win_rate = len(positive_returns) / (len(positive_returns) + len(negative_returns))
        else:
            win_rate = 0
        
        # Calculate profit factor
        gross_profit = clean_results[clean_results['net_returns'] > 0]['net_returns'].sum()
        gross_loss = abs(clean_results[clean_results['net_returns'] < 0]['net_returns'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Average trade PnL
        trade_pnls = clean_results[clean_results['trade_pnl'] != 0]['trade_pnl']
        avg_trade_pnl = trade_pnls.mean() if len(trade_pnls) > 0 else 0
        
        metrics = {
            'Total Return': f"{total_return:.2%}",
            'Annualized Return': f"{annualized_return:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Sortino Ratio': f"{sortino_ratio:.2f}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Calmar Ratio': f"{calmar_ratio:.2f}",
            'Profit Factor': f"{profit_factor:.2f}",
            'Number of Trades': int(n_trades),
            'Round Trips': int(n_round_trips),
            'Win Rate': f"{win_rate:.2%}",
            'Avg Trade PnL': f"{avg_trade_pnl:.2%}"
        }
        
        return metrics
    
    def run_pipeline(self, price1, price2, train_size=0.7, validation_size=0.15):
        """Run complete pipeline"""
        print("="*50)
        print("LSTM Statistical Arbitrage Pipeline")
        print("="*50)
        
        # 1. Test cointegration
        print("\n1. Testing Cointegration...")
        is_cointegrated, p_value = self.test_cointegration(price1, price2)
        print(f"   Cointegration p-value: {p_value:.4f}")
        print(f"   Cointegrated: {is_cointegrated}")
        
        if not is_cointegrated:
            print("   Warning: Series may not be cointegrated!")
        
        # 2. Calculate dynamic hedge ratio
        print("\n2. Calculating Dynamic Hedge Ratio...")
        hedge_ratios = self.calculate_hedge_ratio(price1, price2, method='rolling')
        
        # 3. Create spread
        print("\n3. Creating Spread...")
        data = pd.DataFrame({
            'price1': price1[self.hedge_ratio_window:],
            'price2': price2[self.hedge_ratio_window:],
            'hedge_ratio': hedge_ratios
        })
        data['spread'] = self.create_spread(data['price1'], data['price2'], data['hedge_ratio'])
        
        # 4. Engineer features
        print("\n4. Engineering Features...")
        features = self.engineer_features(data)
        print(f"   Features created: {features.shape[1]}")
        
        # 5. Prepare data splits
        print("\n5. Preparing Data Splits...")
        n_samples = len(features)
        train_end = int(n_samples * train_size)
        val_end = int(n_samples * (train_size + validation_size))
        
        train_features = features[:train_end]
        val_features = features[train_end:val_end]
        test_features = features[val_end:]
        
        print(f"   Train: {len(train_features)} samples")
        print(f"   Validation: {len(val_features)} samples")
        print(f"   Test: {len(test_features)} samples")
        
        # 6. Prepare LSTM data
        print("\n6. Preparing LSTM Data...")
        feature_cols = features.columns.tolist()
        X_train, y_train = self.prepare_lstm_data(
            self.scaler.fit_transform(train_features), 
            train_features['zscore'].values
        )
        X_val, y_val = self.prepare_lstm_data(
            self.scaler.transform(val_features),
            val_features['zscore'].values
        )
        X_test, y_test = self.prepare_lstm_data(
            self.scaler.transform(test_features),
            test_features['zscore'].values
        )
        
        # 7. Train LSTM model
        print("\n7. Training LSTM Model...")
        self.model = self.build_lstm_model(len(feature_cols))
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        print(f"   Training completed. Final loss: {history.history['loss'][-1]:.4f}")
        
        # 8. Generate predictions and signals
        print("\n8. Generating Trading Signals...")
        test_predictions = self.model.predict(X_test, verbose=0).flatten()
        test_features_aligned = test_features.iloc[self.lookback:-self.forecast_horizon].copy()
        test_features_aligned['predictions'] = test_predictions
        
        signals = self.generate_signals(test_predictions, test_features_aligned)
        
        # 9. Backtest strategy
        print("\n9. Backtesting Strategy...")
        test_data = data.loc[test_features_aligned.index]
        results = self.backtest(test_data, signals)
        
        # 10. Calculate performance metrics
        print("\n10. Performance Metrics:")
        print("-" * 30)
        metrics = self.calculate_performance_metrics(results)
        for key, value in metrics.items():
            print(f"   {key}: {value}")
        
        # 11. Visualize results
        self.plot_results(results, test_features_aligned)
        
        return {
            'features': features,
            'model': self.model,
            'signals': signals,
            'results': results,
            'metrics': metrics,
            'history': history
        }
    
    def plot_results(self, results, features):
        """Plot backtest results with proper signal/execution timing visualization"""
        fig, axes = plt.subplots(5, 1, figsize=(14, 12))
        
        # Spread and z-score
        ax1 = axes[0]
        ax1.plot(features.index, features['zscore'], label='Z-Score', alpha=0.7)
        ax1.axhline(y=2, color='r', linestyle='--', alpha=0.5, label='Entry Threshold')
        ax1.axhline(y=-2, color='r', linestyle='--', alpha=0.5)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_ylabel('Z-Score')
        ax1.set_title('Spread Z-Score with Entry/Exit Thresholds')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Signals vs Positions (showing lag)
        ax2 = axes[1]
        ax2.plot(results.index, results['signal_today'], label='Signal Generated', 
                alpha=0.5, linestyle='--', color='blue')
        ax2.plot(results.index, results['position'], label='Position Taken (Next Bar)', 
                drawstyle='steps-post', color='green', linewidth=2)
        ax2.set_ylabel('Position')
        ax2.set_title('Trading Signals vs Actual Positions (Showing Execution Lag)')
        ax2.set_ylim([-1.5, 1.5])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Cumulative returns
        ax3 = axes[2]
        ax3.plot(results.index, (results['cumulative_returns'] - 1) * 100, 
                label='Strategy Returns', linewidth=2, color='darkgreen')
        ax3.set_ylabel('Returns (%)')
        ax3.set_title('Cumulative Returns (After Transaction Costs)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Drawdown
        ax4 = axes[3]
        ax4.fill_between(results.index, results['drawdown'] * 100, 0, 
                        alpha=0.3, color='red', label='Drawdown')
        ax4.set_ylabel('Drawdown (%)')
        ax4.set_title('Strategy Drawdown')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Trade PnL Distribution
        ax5 = axes[4]
        trade_pnls = results[results['trade_pnl'] != 0]['trade_pnl'] * 100
        if len(trade_pnls) > 0:
            ax5.hist(trade_pnls, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax5.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax5.axvline(x=trade_pnls.mean(), color='green', linestyle='-', 
                       label=f'Avg: {trade_pnls.mean():.2f}%')
        ax5.set_xlabel('Trade PnL (%)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Trade PnL Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# Example usage and testing
def generate_synthetic_data(n_samples=2000):
    """Generate synthetic cointegrated price series for testing"""
    np.random.seed(42)
    
    # Generate base random walk
    returns1 = np.random.normal(0.0001, 0.02, n_samples)
    price1 = 100 * np.exp(np.cumsum(returns1))
    
    # Generate cointegrated series
    spread_mean_reverting = np.cumsum(np.random.normal(0, 0.01, n_samples))
    spread_mean_reverting = spread_mean_reverting - 0.01 * np.cumsum(spread_mean_reverting)
    
    price2 = price1 * 0.8 + spread_mean_reverting + 80
    
    return pd.Series(price1), pd.Series(price2)

def main():
    """Main execution function"""
    # Generate or load your data
    print("Generating synthetic cointegrated data...")
    price1, price2 = generate_synthetic_data(2000)
    
    # Initialize pipeline
    pipeline = CointegratedSpreadsPipeline(
        lookback=60,
        forecast_horizon=5,
        confidence_threshold=0.95
    )
    
    # Run complete pipeline
    results = pipeline.run_pipeline(price1, price2, train_size=0.6, validation_size=0.2)
    
    print("\n" + "="*50)
    print("Pipeline Execution Complete!")
    print("="*50)
    
    return pipeline, results

if __name__ == "__main__":
    pipeline, results = main()