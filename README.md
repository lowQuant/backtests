This is a directory for backtests.

There is a utils folder that contains some useful functions for downloading data.

All backtests should be in a separate folder.

## Backtests

### short_rsi_thrust
A mean-reversion short strategy targeting overbought stocks with strong momentum.

**Strategy Rules:**
- **Filter:** Price >= $5, 20-day Avg Dollar Volume > $25M, ATR(10) >= 3% of close
- **Setup:** RSI(3) > 90, last 2 days closed higher than previous day
- **Ranking:** Highest ADX(7)
- **Entry:** Sell short at 4% above previous close (limit order)
- **Stop-Loss:** 3x ATR(10) above entry price (placed day after entry)
- **Profit Taking:** Exit at MOC if profit >= 4%, or exit after 2 days
- **Position Sizing:** 2% risk, 10% max position size, max 10 positions

Run: `python short_rsi_thrust/b_short_rsi_thrust.py`

### LTlowvol (Long Trend Low Volatility)
A trend-following long strategy focusing on low volatility stocks in an uptrend.

**Strategy Rules:**
- **Filter:** Avg Daily Dollar Volume > $100M (50d), Historic Volatility between 10% and 40%
- **Setup:** S&P 500 Close > 200d SMA, Stock Close > 200d SMA
- **Ranking:** Lowest 4-day RSI (pullback entry)
- **Entry:** Next day Market on Open
- **Stop-Loss:** 1.5x ATR(40) below execution price
- **Profit Protection:** Trailing stop of 20%
- **Position Sizing:** 2% risk, 10% maximum position size

Run: `python LTlowvol/backtest.py`

### LT_highIV (High IV + Channel Breakout - Tom Basso Style)
Combines High IV ranking with classic channel breakouts (Donchian, Keltner, Bollinger).

**Channel Breakout LONG (`highiv_channel_long.py`):** ⭐ RECOMMENDED
- **Channels:** Donchian (20d), Keltner (EMA20 ± 2x ATR), Bollinger (SMA20 ± 2σ)
- **Filter:** Avg Dollar Vol > $100M, IV >= 30%, Price >= $5
- **Setup:** SPY > 200 SMA, Stock > 200 SMA, ANY channel breakout
- **Ranking:** Highest IV
- **Exit:** ANY channel gives exit (close < middle) OR 2x ATR stop
- **Results:** **5.06% CAGR, 0.48 Sharpe, -22.52% Max DD** (936 trades)
- **A/B Test:** IV ranking adds +1.31% CAGR vs without IV

**Channel Breakdown SHORT (`highiv_channel_short.py`):**
- Same channels, but short on breakdown (close < lower band)
- Results: -5.85% CAGR (struggled in bull market)

**Momentum Breakout (`backtest.py`):**
- Simplified strategy targeting stocks with HV > 40% breaking out to 20-day highs.
- **Results:** 11.18% CAGR, 0.47 Sharpe. Captures massive winners (Avg Win 155%).

Run:
```bash
python LT_highIV/highiv_channel_long.py
python LT_highIV/highiv_channel_short.py
python LT_highIV/backtest.py
```

<<<<<<< /Users/jo/Desktop/backtests/README.md
### tom_basso_channels (Tom Basso All-Weather Trader)
Channel breakout strategy using three indicators from "The All-Weather Trader":
- **Donchian Channels** - Highest high / lowest low breakout
- **Keltner Channels** - EMA ± 2x ATR
- **Bollinger Bands** - SMA ± 2x StdDev

**Strategy Rules:**
- **Entry:** Buy next day if ANY channel signals breakout (close > upper band)
- **Exit:** Close position if ANY channel reverses (close < lower band)
- **Combined Signal:** Uses OR logic for entry/exit across all 3 indicators

**Results (Top 500 stocks by volume):**
- Best Lookback: 10 days (PF=1.004, Sharpe=0.018)
- Strategy is near break-even without additional filters

Run:
```bash
python tom_basso_channels/profit_factor_analysis.py
=======
### tom_basso_arena
Vectorized Tom Basso *All-Weather Trader* channel-breakout backtest using a single combined signal from:
- Donchian Channels
- Keltner Channels
- Bollinger Bands

Run:
```bash
python tom_basso_arena/backtest.py --lookback 20 --data-path path/to/us_stock_history_10y.csv
python tom_basso_arena/profit_factor_analysis.py --min-lookback 10 --max-lookback 100 --step 5 --data-path path/to/us_stock_history_10y.csv
>>>>>>> /Users/Jo/.windsurf/worktrees/backtests/backtests-326155b7/README.md
```

### Other Strategies
- Intraday breakout strategy on QQQ (in progress)

### Environment Setup
It is recommended to use a virtual environment to avoid dependency conflicts.

```bash
# Create virtual environment (using python 3.10+)
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Data Downloading
The utility script `utils/ib_intraday_downloader.py` can be used to download historical data from Interactive Brokers.

**Usage:**
```bash
# Ensure venv is active or use full path
./venv/bin/python3 -m utils.ib_intraday_downloader --symbol SYMBOL [--start START_DATE] [--end END_DATE] [--interval INTERVAL]
```

**Example:**
Download QQQ 1-minute data from 2020-01-01 to today:
```bash
./venv/bin/python3 -m utils.ib_intraday_downloader --symbol QQQ --start 2020-01-01 --end today
```

The data will be saved as a Parquet file in the `data/` directory, named `SYMBOL_START_END.parquet`.