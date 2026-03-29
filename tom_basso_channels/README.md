# Tom Basso All-Weather Trader Strategy

Implementation of the channel breakout strategy from Tom Basso's book "The All-Weather Trader" using three indicators:
- **Donchian Channels** - Highest high / lowest low breakout
- **Keltner Channels** - EMA ± ATR multiplier
- **Bollinger Bands** - SMA ± standard deviation

## Strategy Logic

### Entry (Buy Signal)
Buy at next day's open if **ANY** of the three indicators signals a buy:
- Close > Donchian upper band (highest high of lookback period)
- Close > Keltner upper band (EMA + 2x ATR)
- Close > Bollinger upper band (SMA + 2x StdDev)

### Exit (Sell Signal)
Exit at next day's open if **ANY** indicator reverses and shows a sell:
- Close < Donchian lower band (lowest low of lookback period)
- Close < Keltner lower band (EMA - 2x ATR)
- Close < Bollinger lower band (SMA - 2x StdDev)

## Files

| File | Description |
|------|-------------|
| `strategy.py` | Core strategy with vectorized signal generation and `run_universe_backtest_vectorized()` |
| `profit_factor_analysis.py` | Profit factor vs lookback sensitivity analysis |
| `backtest_full.py` | Full backtest with Long/Short/Combined modes and comprehensive metrics |
| `merge_iv_data.py` | Merge IV data with stock history (optional) |

## Key Functions

- **`get_combined_signal(close, high, low, lookback)`** - Vectorized combined signal from all 3 indicators
- **`run_universe_backtest_vectorized(df, lookback)`** - Fast backtest across all symbols, returns DataFrame with `strategy_return` column
- **`calculate_profit_factor_from_mean_returns(df)`** - Calculates PF from mean returns across symbols by date (log returns)

## Usage

### Run Profit Factor Analysis
```bash
cd /Users/jo/Desktop/backtests
source venv/bin/activate
python tom_basso_channels/profit_factor_analysis.py
```

This generates:
- `profit_factor_vs_lookback.png` - Visualization of profit factor across lookback values
- `lookback_analysis_results.csv` - Raw data for further analysis

### Using the Strategy in Your Code
```python
from tom_basso_channels.strategy import get_combined_signal, run_universe_backtest

# Run backtest with 20-day lookback
metrics = run_universe_backtest(lookback=20)
print(f"Profit Factor: {metrics['profit_factor']:.4f}")

# Or get signals for a single stock DataFrame
signal = get_combined_signal(ohlc_df, lookback=20)
```

### Merge IV Data (Optional)
```bash
python tom_basso_channels/merge_iv_data.py
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback` | 20 | Period for all three indicators |
| `atr_mult` | 2.0 | ATR multiplier for Keltner Channels |
| `std_mult` | 2.0 | Standard deviation multiplier for Bollinger Bands |

## Data Requirements

Uses `mean_reversion/us_stock_history_10y.csv` with columns:
- `Date`, `Symbol`, `Open`, `High`, `Low`, `Close`, `Volume`

## Latest Results (Feb 2026)

Full universe backtest (4341 symbols):

| Lookback | Profit Factor | Sharpe Ratio | Total Log Return |
|----------|---------------|--------------|------------------|
| 10 | 0.934 | -0.289 | -0.405 |
| 20 | 0.946 | -0.223 | -0.300 |
| 50 | 0.977 | -0.091 | -0.119 |
| 60 | 0.999 | -0.002 | -0.003 |
| 80 | 1.037 | 0.147 | 0.198 |
| 100 | 1.063 | 0.250 | 0.338 |

**Key Finding:** Profit factor increases with longer lookbacks. Strategy shows positive edge at lookbacks >= 60 days.

## Full Backtest Results (200d Lookback, Top 100 Stocks)

Backtest period: 2015-11-23 to 2025-11-20 (10 years)

| Metric | LONG | SHORT | COMBINED |
|--------|------|-------|----------|
| Total Return | 124.39% | -42.77% | 28.42% |
| CAGR | 8.44% | -5.44% | 2.54% |
| Sharpe Ratio | 0.84 | -0.39 | 0.18 |
| Profit Factor | 1.17 | 0.91 | 1.04 |
| Max Drawdown | -17.79% | -59.65% | -33.80% |
| Win Rate | 57.39% | 46.07% | 53.59% |
| Avg Days Held | 183.3 | 134.8 | 159.4 |
| N Trades | 575 | 558 | 1133 |

**Key Findings:**
- **Long-only** is the clear winner with 8.44% CAGR and 0.84 Sharpe
- **Short-only** underperforms significantly in a bull market
- **Combined** dilutes the long-only edge

### Run Full Backtest
```bash
python tom_basso_channels/backtest_full.py
```

## Portfolio Backtest Results (Proper Simulation)

Backtest with realistic constraints:
- **Starting Capital:** $100,000
- **Position Size:** 5% of equity
- **Commission:** $0.005/share (min $1)
- **Long Filter:** Stock > 150d SMA
- **Ranking:** By dollar volume
- **Period:** 2015-11-23 to 2025-11-21 (10 years)

### Results

| Metric | LONG | SHORT | COMBINED |
|--------|------|-------|----------|
| Final Equity | $1,092,716 | $10,545 | $230,072 |
| Total Return | 992.72% | -89.45% | 130.07% |
| CAGR | **27.03%** | -20.15% | 8.69% |
| Sharpe Ratio | **1.07** | 0.21 | 0.43 |
| Profit Factor | **5.01** | 0.39 | 2.90 |
| Max Drawdown | **-30.08%** | -113.79% | -72.07% |
| Win Rate | 42.67% | 18.31% | 40.59% |
| Avg Days Held | 249.7 | 204.3 | 254.2 |
| N Trades | 225 | 568 | 340 |

### Short Strategy Experiments

Tested multiple short approaches - all failed:

1. **200d Channel Breakdown:** -89.45% (trend-following shorts fail in bull market)
2. **3d Breakout + 5d EMA Exit:** -99.88% (short-term breakouts fail)
3. **RSI(2) > 90 Mean Reversion:** -99.93% (overbought stocks keep going higher)

**Conclusion:** Short strategies are not viable in this 10-year bull market period. **Long-only is recommended.**

### Run Portfolio Backtest
```bash
python tom_basso_channels/backtest_portfolio.py
```

## Long-Only Strategy (Final Optimized Version)

We created a dedicated long-only backtest with SPY benchmark comparison and trade visualization.

**Performance (2015-2025):**
- **Final Equity:** $946,669 (Start: $100k)
- **Total Return:** 846.67% (CAGR: 25.20%)
- **Max Drawdown:** -35.27%
- **Sharpe Ratio:** > 1.0
- **Total Trades:** 209

### Outputs
- `backtest_long_only_result.png`: Equity curve vs SPY, Drawdown, and Open Positions.
- `backtest_long_only_trades.csv`: Complete list of trades with indicator values.
- `trade_examples/`: Folder containing 30 random trade plots showing entry/exit points and indicators.

### Run Long-Only Backtest
```bash
python tom_basso_channels/backtest_long_only.py
```

### Generate Trade Examples
```bash
python tom_basso_channels/plot_trade_examples.py
```

## Methodology

- **Signal Generation:** Vectorized per-symbol using pandas groupby
- **Returns:** Log returns, shifted for next-day execution
- **Aggregation:** Mean strategy return across all symbols per date
- **Profit Factor:** Sum of positive mean returns / Sum of absolute negative mean returns
