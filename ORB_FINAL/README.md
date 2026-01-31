# Opening Range Breakout (ORB) Strategy - Final Version

## Overview

This is an **intraday breakout strategy** trading the 5-minute Opening Range Breakout on **TQQQ** (3x Leveraged QQQ ETF). The strategy takes one trade per day, entering after the opening range is established and exiting via stop loss or Market-On-Close (MOC).

**Core Concept**: Trade in the direction of the initial momentum established in the first 5 minutes of the trading day, with regime-based filtering using ADX (trend strength) and VIX (market volatility).

---

## Strategy Logic

### 1. Opening Range Definition

- **Timeframe**: First 5 one-minute bars of the trading session (09:30 - 09:35 ET)
- **Range High**: Maximum high of bars 0-4
- **Range Low**: Minimum low of bars 0-4

### 2. Entry Signal

**Direction Determination** (at 09:35 close):
```
side = sign(Close[bar 4] - Open[bar 0])
```
- If the close of the 5th minute > open of the 1st minute → **Long**
- If the close of the 5th minute < open of the 1st minute → **Short**
- If equal → **No trade**

**Entry Price**: Open of bar 5 (09:36 ET) - the first available price after signal confirmation.

### 3. Stop Loss Calculation

The stop is placed at the opposite extreme of the opening range:

| Direction | Stop Level | Stop Distance (%) |
|-----------|------------|-------------------|
| Long | Opening Range Low | `abs(OR_Low / Entry - 1)` |
| Short | Opening Range High | `abs(OR_High / Entry - 1)` |

**Note**: The stop distance is calculated relative to the **entry price**, not the range boundary itself.

### 4. Profit Target

- **Default**: `target_R = inf` (No profit target)
- When no target is set, the trade exits at MOC or stop loss, whichever comes first
- Optional: Can set a target as a multiple of risk (e.g., `target_R = 2` for 2:1 reward-to-risk)

### 5. Exit Logic

Exits are evaluated each minute after entry:

1. **Stop Loss Hit**: If Low ≤ Stop Price (Long) or High ≥ Stop Price (Short)
2. **Target Hit**: If High > Target Price (Long) or Low < Target Price (Short)
3. **Same-Bar Conflict**: If both stop and target are hit on the same bar, the one hit on an earlier bar wins. If same bar, stop takes priority.
4. **Market-On-Close (MOC)**: If neither stop nor target is hit by end of day, exit at the last bar's close

**Fill Price Logic**:
- Stop fills: `min(Stop Price, Open of Exit Bar)` for longs, `max(Stop Price, Open of Exit Bar)` for shorts
- Target fills: `max(Target Price, Open of Exit Bar)` for longs, `min(Target Price, Open of Exit Bar)` for shorts
- This handles gap-through scenarios conservatively

---

## Filters (Regime V2)

### ADX Filter (Trend Strength)

- **Indicator**: 14-period ADX calculated on 1-minute bars
- **Sampling Time**: Index 3 of the day (09:34 bar close, ~4 minutes after open)
- **Threshold**: ADX > 20 required to take any trade
- **Rationale**: Filters out choppy, range-bound days where breakouts are more likely to fail

### VIX Filter (Volatility Regime)

- **Data Source**: Daily VIX Open price
- **Logic**:
  - **VIX ≤ 20**: Both Long and Short trades allowed
  - **VIX > 20**: Only Short trades allowed (Long signals ignored)
- **Rationale**: In high-volatility regimes, long breakouts tend to reverse; short breakouts (panic selling) are more reliable

### Filter Summary Table

| ADX | VIX | Long Allowed | Short Allowed |
|-----|-----|--------------|---------------|
| ≤ 20 | Any | ❌ | ❌ |
| > 20 | ≤ 20 | ✅ | ✅ |
| > 20 | > 20 | ❌ | ✅ |

---

## Position Sizing

### Risk-Based Sizing

Each trade risks a fixed percentage of current equity:

```python
risk = 0.01  # 1% of AUM per trade
risk_amt = AUM * risk
shares_risk = risk_amt / (entry * stop_pct)
```

### Leverage Constraint

Maximum leverage caps the position size:

```python
max_Lev = 4  # Maximum 4x leverage
shares_lev = (max_Lev * AUM) / entry
```

### Final Position

```python
shares = floor(min(shares_risk, shares_lev))
```

The position is the **minimum** of risk-based and leverage-constrained sizes, rounded down to whole shares.

---

## Transaction Costs

- **Commission**: $0.0005 per share (round-trip: 2x per trade)
- **Slippage**: Not explicitly modeled (see Limitations)

---

## Parameters Summary

| Parameter | Value | Description |
|-----------|-------|-------------|
| `orb_m` | 5 | Opening range duration (minutes) |
| `target_R` | ∞ | Profit target as R-multiple (inf = no target) |
| `risk` | 0.01 | Risk per trade (1% of AUM) |
| `max_Lev` | 4 | Maximum leverage |
| `AUM_0` | $25,000 | Starting capital |
| `commission` | 0.0005 | Per-share commission |
| `adx_threshold` | 20 | Minimum ADX to trade |
| `vix_threshold` | 20 | VIX level for regime switch |

---

## Lookahead Bias Analysis

### ✅ No Lookahead Bias Detected

After careful review, the strategy is **free of lookahead bias**:

| Component | Analysis | Verdict |
|-----------|----------|---------|
| **ADX Calculation** | Uses `ewm()` which is backward-looking. ADX at bar t only uses data from bars 0 to t. | ✅ Clean |
| **VIX Data** | Uses `VIX_Open` which is known at market open (09:30), before any trade decisions | ✅ Clean |
| **ATR Calculation** | Explicitly shifted: `ATR = ATR.shift(1)` uses prior day's value | ✅ Clean |
| **Direction Signal** | Based on bars 0-4, known at 09:35 close | ✅ Clean |
| **Entry Price** | Uses Open of bar 5 (09:36), first available price after signal | ✅ Clean |
| **Stop Price** | Calculated from OR High/Low (bars 0-4), known before entry | ✅ Clean |
| **Position Sizing** | Uses `AUM[t-1]` (previous day's capital) | ✅ Clean |

---

## Live Trading Considerations & Potential Issues

### ⚠️ Execution Assumptions

1. **Entry at Open**: Strategy assumes entry at the exact open of bar 5 (09:36). In live trading:
   - Market orders may get slight slippage
   - Limit orders may not fill
   - **Recommendation**: Use market orders with acceptable slippage tolerance

2. **Stop Execution**: Strategy assumes fills at stop price or open (if gap-through):
   - Actual fills may be worse in fast markets
   - Consider using stop-limit orders with reasonable buffers

3. **MOC Orders**: End-of-day exit assumes perfect MOC execution:
   - MOC orders typically have cutoff times (e.g., 3:45 PM for NYSE)
   - May need to exit slightly before close in practice

### ⚠️ Data & Timing

1. **1-Minute Bar Alignment**: Live data feed bars must align with backtest bars:
   - Bars should be 09:30-09:31, 09:31-09:32, etc.
   - Different brokers may have different bar conventions

2. **ADX Calculation Window**: ADX uses rolling calculations starting from prior days:
   - Need sufficient history loaded at market open
   - Ensure ADX is calculated with the same methodology as backtest

3. **VIX Data Source**: Must use the same VIX index opening price:
   - CBOE VIX opens at 09:30 ET
   - Some feeds may have delays

### ⚠️ Risk Considerations

1. **Leverage on 3x ETF**: Trading TQQQ with 4x leverage = 12x effective QQQ exposure:
   - Extreme drawdowns possible in market crashes
   - Consider reducing `max_Lev` for live trading

2. **Single-Day Trades**: All positions are closed daily:
   - No overnight risk
   - But susceptible to intraday volatility

3. **Concentrated Bets**: One trade per day, fully sized:
   - High variance between days
   - Extended losing streaks possible

### ⚠️ Same-Bar Exit Ambiguity

When both stop and target could be hit within the same 1-minute bar:
- Backtest assumes stop was hit (conservative for longs, aggressive for shorts)
- In reality, the sequence within the bar is unknown
- **Impact**: Minimal with 1-minute resolution, but exists

---

## File Structure

```
ORB_FINAL/
├── orb_final.py              # Main strategy (Regime V2) - THIS FILE
├── orb_backtest.py           # Core backtest engine & data prep
├── orb_backtest_adx.py       # ADX calculation & ADX-only strategy
├── orb_backtest_combined.py  # Combined ADX+VIX filter (VIX≤20 only)
├── vix_analysis.py           # VIX data loading & regression analysis
├── mcpt_analysis.py          # Monte Carlo Permutation Test
├── mcpt_validation.py        # MCPT validation scripts
├── param_analysis/           # Parameter sensitivity analysis
├── TQQQ_intraday_2020-2026.parquet  # Price data
├── VIX_daily.csv             # VIX data
└── *.png                     # Output charts
```

---

## Running the Strategy

```bash
cd /Users/jo/Desktop/backtests/ORB_FINAL
source ../venv/bin/activate
python orb_final.py
```

**Output**: Comparison of three strategy variants:
1. Combined Filter (ADX>20 & VIX≤20 only)
2. Regime V2 (ADX>20, shorts allowed in high VIX)
3. ADX Only (no VIX filter)

---

## Statistical Validation

The strategy has been validated using Monte Carlo Permutation Tests (MCPT):
- **Regime V2**: P-Value = 0.0227 (Significant at 5% level)
- Tests confirm the edge is not due to random chance or curve-fitting

See `mcpt_analysis.py` and `mcpt_validation.py` for details.

---

## Changelog

- **V1 (Combined)**: ADX>20 AND VIX≤20 required for any trade
- **V2 (Regime)**: ADX>20 required; VIX regime determines direction allowance
  - VIX≤20: Long & Short
  - VIX>20: Short only

---

## Disclaimer

This strategy is for research and educational purposes. Past performance does not guarantee future results. Always paper trade before committing real capital.
