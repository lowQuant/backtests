# Opening Range Breakout (ORB) Strategy

## Overview
This strategy implements a 5-minute Opening Range Breakout (ORB) system. It sets a trade entry based on the high/low of the first 5 minutes of the trading session.

## Strategy Logic
1.  **Opening Range**: Define the high and low of the first `orb_m` minutes (default: 5 minutes).
2.  **Direction**:
    *   **Long**: If the close of the OR candle is higher than the day's open.
    *   **Short**: If the close of the OR candle is lower than the day's open.
3.  **Entry**:
    *   **Long**: Enter at the open of the next candle (6th minute).
    *   **Short**: Enter at the open of the next candle (6th minute).
4.  **Risk Management**:
    *   **Stop Loss (Long)**: Set at the low of the Opening Range.
    *   **Stop Loss (Short)**: Set at the high of the Opening Range.
    *   **Position Sizing**: Risk `risk` % (default 1%) of equity per trade, capped at `max_Lev` (default 4x) leverage.
5.  **Exit**:
    *   **Target**: Optional profit target (default: infinite/no target).
    *   **Stop Loss**: If price hits the stop level.
    *   **Market On Close (MOC)**: If no stop/target is hit by end of day.

## Performance Results (2020-2026)

### QQQ
-   **Total Return**: +280.83%
-   **CAGR**: 25.09%
-   **Sharpe Ratio**: 0.85
-   **Max Drawdown**: -30.50%

### TQQQ (3x Leveraged)
-   **Total Return**: +404.31%
-   **CAGR**: 31.07%
-   **Sharpe Ratio**: 0.85
-   **Max Drawdown**: -35.00%

## Files
-   `orb_backtest.py`: Main script to load local parquet data, calculate indicators (ATR), and run the backtest logic.
-   `equity_QQQ.png`: Equity curve for QQQ.
-   `equity_TQQQ.png`: Equity curve for TQQQ.
