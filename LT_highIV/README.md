# LT_highIV - High Implied Volatility Strategies

Strategies combining High IV ranking with classic trend-following channel breakouts (Tom Basso "All-Weather Trader" style using Donchian, Keltner, and Bollinger channels).

## Data Requirements

- **Price Data:** `mean_reversion/us_stock_history_10y.csv`
- **IV Data:** `volatility_research/data/iv_full.parquet` (covers 2019-02 to 2024-11)

---

## 1. Channel Breakout LONG (`highiv_channel_long.py`) ⭐ RECOMMENDED

**Concept:** Tom Basso style - if ANY channel gives breakout signal → go long, ranked by highest IV

### Channel Indicators
- **Donchian:** 20-day high/low breakout
- **Keltner:** EMA(20) ± 2x ATR
- **Bollinger:** SMA(20) ± 2 std dev

### Strategy Rules

- **Filter:** Avg Dollar Vol > $100M, IV >= 30%, Price >= $5
- **Setup:** SPY > 200 SMA, Stock > 200 SMA, ANY channel breakout
- **Ranking:** Highest IV
- **Entry:** Next day Market on Open
- **Exit:** ANY channel gives exit (close below middle) OR stop loss (2x ATR)
- **Position Sizing:** 2% risk, 10% max

### A/B Test Results (IV adds value!)

| Metric | IV=ON | IV=OFF | Difference |
|--------|-------|--------|------------|
| CAGR | **5.06%** | 3.75% | +1.31% |
| Sharpe | **0.48** | 0.36 | +0.12 |
| Max DD | **-22.52%** | -25.94% | +3.42% better |
| Trades | 936 | 1948 | More selective |

### Run
```bash
conda activate yfinance
python LT_highIV/highiv_channel_long.py

# A/B test without IV: set USE_IV_RANKING = False in script
```

---

## 2. Channel Breakdown SHORT (`highiv_channel_short.py`)

**Concept:** Short on channel breakdown (reverse of LONG), ranked by highest IV

### Strategy Rules

- **Filter:** Avg Dollar Vol > $50M, IV >= 40%, Price >= $10
- **Setup:** Stock < 200 SMA (downtrend), ANY channel breakdown
- **Ranking:** Highest IV
- **Exit:** ANY channel gives exit (close above middle), stop loss (2.5x ATR), or 15-day time stop
- **Position Sizing:** 1.5% risk, 6% max

### Results

| Metric | Value |
|--------|-------|
| CAGR | -5.85% |
| Sharpe | -0.53 |
| Max DD | -57.27% |

**Note:** Short strategies struggled in the predominantly bullish 2014-2024 period.

### Run
```bash
conda activate yfinance
python LT_highIV/highiv_channel_short.py
```

---

## 3. High Volatility Momentum Breakout (`backtest.py`)

**Concept:** A simplified momentum breakout strategy targeting high volatility stocks (>40% HV) hitting 20-day highs.

### Strategy Rules

- **Filter:** Avg Dollar Vol > $100M, Historic Volatility (HV) > 40%, Price > $5
- **Setup:** SPY > 200 SMA, Stock Close > 100 SMA
- **Entry:** Stock Close > Highest Close of last 20 days (20-day Breakout)
- **Ranking:** Highest 20-day Rate of Change (ROC)
- **Exit:**
  - **Stop Loss:** 3.0x ATR(20) below entry
  - **Trailing Stop:** 25% Trailing Stop
- **Position Sizing:** 1% risk per trade, 5% max position size

### Results (10-Year Backtest)

| Metric | Value |
|--------|-------|
| CAGR | **11.18%** |
| Sharpe | **0.47** |
| Max DD | -42.99% |
| Win Rate | 45.45% |
| Profit Factor | 1.07 |
| Avg Win | 155.07% |

### Run
```bash
python LT_highIV/backtest.py
```

---

## Legacy Strategies (Deprecated)

The original `highiv_long.py` and `highiv_short.py` used RSI pullback/reversal entries instead of channel breakouts. These performed poorly and are superseded by the channel-based strategies above.

---

## Summary

| Strategy | CAGR | Sharpe | Max DD | Trades |
|----------|------|--------|--------|--------|
| **highiv_channel_long (IV=ON)** | **5.06%** | **0.48** | **-22.52%** | 936 |
| highiv_channel_long (IV=OFF) | 3.75% | 0.36 | -25.94% | 1948 |
| highiv_channel_short | -5.85% | -0.53 | -57.27% | 935 |

**Conclusion:** IV ranking adds ~1.3% CAGR and improves risk-adjusted returns. The LONG channel strategy is recommended for production use.
