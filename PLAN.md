# Backtests Project Plan

## Completed Tasks

### LTlowvol (Long Trend Low Volatility)
- [x] Created `LTlowvol/` folder
- [x] Implemented `backtest.py` with strategy logic:
  - Filter: Avg Dollar Vol > $100M, HV between 10-40%
  - Setup: SPY > 200 SMA, Close > 200 SMA
  - Ranking: Lowest 4-day RSI
  - Entry: Market on Open
  - Exits: 1.5x ATR(40) Stop Loss, 20% Trailing Stop
  - Sizing: 2% risk, 10% max size
- [x] Fixed yfinance data download issues (v1.1.0 MultiIndex)
- [x] Verified backtest results (Sharpe: 0.92, Total Ret: 222%)

### short_rsi_thrust Backtest (2024)
- [x] Created `short_rsi_thrust/` folder
- [x] Implemented `b_short_rsi_thrust.py` with full strategy logic:
  - Filter: Price >= $5, 20-day Avg Dollar Volume > $25M, ATR(10) >= 3% of close
  - Setup: RSI(3) > 90, last 2 days closed higher
  - Ranking: Highest ADX(7)
  - Entry: Limit order 4% above previous close
  - Stop-Loss: 3x ATR(10) above entry
  - Profit Taking: Exit at MOC if profit >= 4%, or after 2 days
  - Position Sizing: 2% risk, 10% max size, max 10 positions
- [x] Tested with sample data (10 symbols)
- [x] Updated README.md with strategy documentation

### LT_highIV (High IV + Channel Breakout - Tom Basso Style)
- [x] Created `LT_highIV/` folder
- [x] Implemented `backtest.py` - High Volatility Momentum Breakout:
  - HV > 40%, 20-day High Breakout, ROC Ranking
  - **Results: 11.18% CAGR, 0.47 Sharpe**
- [x] Implemented `highiv_channel_long.py` - Tom Basso style channel breakout:
  - Channels: Donchian (20d), Keltner (EMA20 ± 2x ATR), Bollinger (SMA20 ± 2σ)
  - Entry: ANY channel breakout + High IV ranking
  - Exit: ANY channel exit (close < middle) OR 2x ATR stop
  - **Results: 5.06% CAGR, 0.48 Sharpe, -22.52% Max DD** (936 trades)
- [x] Implemented `highiv_channel_short.py` - Channel breakdown for shorts:
  - Same channels, short on breakdown
  - Results: -5.85% CAGR (struggled in bull market)
- [x] A/B Testing: IV ranking adds +1.31% CAGR vs without IV
  - IV=ON: 5.06% CAGR, 0.48 Sharpe
  - IV=OFF: 3.75% CAGR, 0.36 Sharpe
- [x] Updated README.md and LT_highIV/README.md
- [x] Deprecated legacy RSI-based strategies (highiv_long.py, highiv_short.py)

### tom_basso_channels (Tom Basso All-Weather Trader)
- [x] Created `tom_basso_channels/` folder
- [x] Implemented `strategy.py` with vectorized combined signal function:
  - Donchian Channels: Highest high / lowest low breakout
  - Keltner Channels: EMA ± 2x ATR
  - Bollinger Bands: SMA ± 2x StdDev
  - Entry: Buy if ANY indicator signals breakout
  - Exit: Sell if ANY indicator reverses
- [x] Refactored to vectorized signal generation (Feb 2026):
  - `run_universe_backtest_vectorized()` for fast multi-symbol backtesting
  - `calculate_profit_factor_from_mean_returns()` using mean returns across symbols by date
  - All calculations use log returns
- [x] Implemented `profit_factor_analysis.py` for lookback sensitivity analysis
- [x] Implemented `merge_iv_data.py` for IV data integration
- [x] **Results (4341 symbols, full universe):**
  - Best Lookback: 100 days (PF=1.063, Sharpe=0.250)
  - Profit factor increases with longer lookbacks (0.93 @ 10d → 1.06 @ 100d)
  - Strategy shows positive edge at lookbacks >= 60 days
- [x] Generated `profit_factor_vs_lookback.png` visualization
- [x] Implemented `backtest_full.py` with Long/Short/Combined modes (Feb 2026):
  - 200-day lookback, Top 100 most traded stocks filter
  - **LONG: 8.44% CAGR, 0.84 Sharpe, -17.79% Max DD**
  - SHORT: -5.44% CAGR (underperforms in bull market)
  - COMBINED: 2.54% CAGR (diluted by shorts)
- [x] Implemented `backtest_portfolio.py` - proper day-by-day simulation:
  - $100k capital, 5% position size, $0.005/share commission
  - 150d SMA filter for longs, volume ranking
  - **LONG: 27.03% CAGR, 1.07 Sharpe, -30% Max DD, 5.01 PF**
  - SHORT: -89.45% total return (lost almost all capital)
  - COMBINED: 130% total return (shorts drag down)
- [x] Tested alternative short strategies - all failed:
  - 3d breakout + 5d EMA exit: -99.88%
  - RSI(2) > 90 mean reversion: -99.93%
  - **Conclusion: Long-only recommended**
- [x] Created clean Long-Only implementation `backtest_long_only.py`:
  - Benchmarked against SPY
  - Visualizations: Equity, Drawdown, Open Positions
  - Generated trade log `backtest_long_only_trades.csv`
  - Created trade visualizer `plot_trade_examples.py` (30 random examples)
  - **Final Result: 25.2% CAGR, -35% Max DD, 846% Return vs SPY**

## Pending Tasks

### tom_basso_channels
- [x] ~~Test long-only vs long/short versions~~ (done - long-only wins)
- [x] ~~Combine with position sizing and risk management~~ (done - 5% sizing)
- [ ] Add SPY trend filter to reduce drawdowns during bear markets
- [ ] Add IV ranking to prioritize high-IV stocks
- [ ] Test with different lookback periods (100d, 150d)
- [ ] Consider adding trailing stop for risk management

### LT_highIV
- [ ] Try different channel periods (10d, 50d) for optimization
- [ ] Test with HV instead of IV for full 10-year coverage
- [ ] Combine LONG channel with LTlowvol for portfolio diversification

### short_rsi_thrust
- [ ] Run backtest with full stock universe (requires ArcticDB or full CSV data)
- [ ] Analyze results and optimize parameters if needed
- [ ] Add commission/slippage modeling

### General
- [ ] Complete intraday breakout strategy on QQQ
- [ ] Add more mean reversion strategies from `mean_reversion/` folder
- [ ] Populate `neurotrader/mcpt/` with profit-factor stability tooling (currently empty)
