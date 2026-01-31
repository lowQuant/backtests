# Short Parabolic Spike Strategy (Japan)

## Executive Summary
This strategy is a mean-reversion short-selling system designed for the Japanese equity market. It identifies stocks that have experienced a rapid, parabolic price increase (spike) and enters short positions anticipating a pullback. The strategy utilizes RSI and short-term returns to define the "spike" condition and uses EMA crossover or time-based exits to close positions.

## Strategy Details

### Universe
- **Market:** Japanese Equities.
- **Data Source:** `jp_stock_history_10y.csv`.
- **Filters:**
    - Minimum Price: 500 JPY (to avoid penny stocks).
    - Minimum Volume: 200,000 shares (to ensure liquidity).

### Entry Logic
A short position is initiated when all the following conditions are met at the Close of Day T:
1.  **Parabolic Move:** The 3-Day Return is greater than **15%** (`SPIKE_THRESHOLD = 0.15`).
2.  **Overbought Condition:** The 4-Day RSI (Wilder's) is greater than **90** (`RSI_THRESHOLD = 90`).
3.  **Ranking:** Candidates are ranked by **ADX(14)** in descending order (preferring strongest recent trends).

**Execution:**
- **Signal Generation:** End of Day T.
- **Entry Execution:** Market Order at the **Open of Day T+1**.

### Exit Logic
Positions are monitored daily. The exit is triggered by the **first** condition met:
1.  **Target/Trend Reversion (Primary):**
    - **Condition:** Current Price < **5-Day EMA** (`EMA_EXIT_WINDOW = 5`).
    - **Execution (Recommended):** **Market-On-Close (MOC)** on the day the condition is met.
    - *Operational Note:* Traders should monitor positions ~5-10 minutes before the market close. If `Price < EMA(5)`, execute a Market-On-Close (or limit at current price) order to exit.
2.  **Time Stop:**
    - **Condition:** Position held for **10 trading days**.
    - **Execution:** Market-On-Close on Day 10.
3.  **Stop Loss:**
    - Standard Strategy: **No Stop Loss**.

### Risk Management & Sizing
- **Starting Capital:** 10,000,000 JPY.
- **Position Sizing:** Dynamic Lot Sizing (Japan Market Standard).
    - **Lot Size:** 100 shares.
    - **Target Allocation:** **2%** of current equity.
    - **Maximum Allocation:** **3%** of current equity.
    - **Sizing Logic:** 
        1. Calculate `Ideal_Lots = Round(Target_Equity_2% / Cost_Per_Lot)`.
        2. **Constraint 1 (Hard Cap):** If `Ideal_Lots * Cost_Per_Lot` > `Max_Equity_3%`, reduce lot count until it fits.
        3. **Constraint 2 (Minimum):** If the cost of even **1 Lot** exceeds `Max_Equity_3%`, the trade is **SKIPPED** entirely.
- **Max Positions:** 50 concurrent positions.
- **Commissions:** 0.05% (5 bps) per trade, with a minimum of 100 JPY.

## Performance & Recommendations

### Execution Timing (MOC vs. Next Open)
Backtests revealed a significant performance disparity based on exit timing:
- **Market-On-Close (Recommended):** ~32% CAGR. Captures the intraday weakness when the mean reversion occurs.
- **Next Open:** ~23% CAGR. Delaying the exit to the next morning gives back significant profits, suggesting the market often "bounces" or stabilizes overnight after closing below the EMA.

**Recommendation:** strictly adhere to the **Market-On-Close** protocol. Automated alerts should be set to trigger near the session close if the price is trading below the 5-day EMA.

### Code Implementation Reference
- **File:** `b_short_spike_jp.py` (MOC Version - Primary)
- **File:** `b_short_spike_jp_next_open.py` (Next Open Version - For comparison/Robustness check)
- **File:** `plot_trades.py` (Visualization tool for verification)
