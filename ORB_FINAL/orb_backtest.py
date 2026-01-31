import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import math
import time
from datetime import datetime, timedelta
import os

# ================================
# Data Processing Functions
# ================================

def load_data(filepath):
    print(f"Loading {filepath}...")
    df = pd.read_parquet(filepath)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Sort index just in case
    df = df.sort_index()
    
    # Ensure columns are lower case
    df.columns = [c.lower() for c in df.columns]
    
    # Add 'day' column for grouping
    df['day'] = df.index.normalize()
    
    return df

def calculate_daily_stats(df):
    """
    Resample intraday data to daily to calculate ATR and dOpen.
    Assumes df is consistent (adjusted or not).
    """
    print("Calculating daily stats and ATR...")
    
    # Resample to Daily
    # Aggregation rules
    daily_agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    daily = df.resample('D').agg(daily_agg).dropna()
    
    # Rename for ATR func
    daily = daily.rename(columns={'open': 'dOpen', 'high': 'dHigh', 'low': 'dLow', 'close': 'dClose'})
    
    # Calculate ATR
    period = 14
    daily['prev_close'] = daily['dClose'].shift(1)
    daily['HC'] = abs(daily['dHigh'] - daily['prev_close'])
    daily['LC'] = abs(daily['prev_close'] - daily['dLow'])
    daily['HL'] = abs(daily['dHigh'] - daily['dLow'])
    daily['TR'] = daily[['HC', 'LC', 'HL']].max(axis=1)
    daily['ATR'] = daily['TR'].rolling(window=period, min_periods=1).mean()
    daily['ATR'] = daily['ATR'].shift(1) # ATR is based on PRIOR days
    
    # Drop temp columns
    daily.drop(['prev_close', 'HC', 'LC', 'HL', 'TR'], axis=1, inplace=True)
    
    # We only need dOpen and ATR mapped back to intraday
    # Note: dOpen here is the Open of the day. 
    # In the original code, dOpen was used for split adjustment ratio. 
    # Here we assume data is internally consistent.
    
    return daily[['dOpen', 'ATR']]

def prepare_backtest_data(filepath):
    # 1. Load Intraday
    df = load_data(filepath)
    
    # 2. Calc Daily Stats
    daily = calculate_daily_stats(df)
    
    # 3. Merge
    # df has 'day' column (datetime). daily index is datetime (normalized).
    
    # Reset index of df to preserve time
    # Force index name to None to ensure reset_index creates 'index' column, 
    # or just rename whatever comes out.
    df.index.name = 'caldt' 
    df = df.reset_index()
    # Now df has 'caldt' column
    
    # Ensure 'day' in daily is a column
    daily.index.name = 'day'
    daily = daily.reset_index()
    
    # Merge on 'day'
    # Ensure types match
    df['day'] = pd.to_datetime(df['day'])
    daily['day'] = pd.to_datetime(daily['day'])
    
    merged = pd.merge(df, daily, on='day', how='left')
    
    # Filter out days with no ATR (first 14 days)
    merged = merged.dropna(subset=['ATR'])
    
    return merged

# ================================
# Performance Analysis
# ================================

def summary_statistics(dailyReturns, pnl_r=None):
    """Calculate performance metrics and return a summary table."""
    riskFreeRate = 0
    tradingDays = 252
    dailyReturns = np.array(dailyReturns)
    dailyReturns = dailyReturns[~np.isnan(dailyReturns)]
    
    if len(dailyReturns) == 0:
        return pd.DataFrame({'Metric': [], 'Value': []})
        
    totalReturn = np.prod(1 + dailyReturns) - 1
    numYears = len(dailyReturns) / tradingDays
    CAGR = (1 + totalReturn)**(1/numYears) - 1
    volatility = np.std(dailyReturns, ddof=0) * np.sqrt(tradingDays)
    
    std = np.std(dailyReturns, ddof=0)
    if std == 0:
        sharpeRatio = 0
    else:
        sharpeRatio = (np.mean(dailyReturns) - riskFreeRate/tradingDays) / std * np.sqrt(tradingDays)
    
    nav = np.cumprod(1 + dailyReturns)
    peak = np.maximum.accumulate(nav)
    drawdown = (nav - peak) / peak
    MDD = np.min(drawdown)
    
    metrics = ["Total Return (%)", "CAGR (%)", "Volatility (%)", "Sharpe Ratio", "Max Drawdown (%)"]
    values = [totalReturn*100, CAGR*100, volatility*100, sharpeRatio, MDD*100]
    
    # Trade Statistics (if pnl_r provided)
    if pnl_r is not None:
        # Filter for actual trades (non-zero PnL)
        trades = pnl_r[pnl_r != 0]
        if len(trades) > 0:
            win_rate = (trades > 0).mean()
            avg_win = trades[trades > 0].mean() if np.any(trades > 0) else 0
            avg_loss = abs(trades[trades < 0].mean()) if np.any(trades < 0) else 0
            
            gross_win = trades[trades > 0].sum()
            gross_loss = abs(trades[trades < 0].sum())
            profit_factor = gross_win / gross_loss if gross_loss != 0 else 999.0
            
            metrics.extend(["Win Rate (%)", "Profit Factor", "Avg Win (R)", "Avg Loss (R)", "Trade Count"])
            values.extend([win_rate*100, profit_factor, avg_win, avg_loss, len(trades)])
            
    formatted_values = []
    for m, v in zip(metrics, values):
        if "Count" in m:
             formatted_values.append(f"{int(v)}")
        elif "(%)" in m:
             formatted_values.append(f"{v:.2f}")
        elif "(R)" in m:
             formatted_values.append(f"{v:.2f}")
        elif "Sharpe" in m:
             formatted_values.append(f"{v:.4f}")
        else:
             formatted_values.append(f"{v:.4f}")

    performance_table = pd.DataFrame({'Metric': metrics, 'Value': formatted_values})
    return performance_table

def monthly_performance_table(returns, dates):
    """Create a table of monthly returns."""
    # Ensure returns is a numpy array or list to ignore index alignment
    if isinstance(returns, pd.Series):
        returns = returns.values
        
    returns_series = pd.Series(returns, index=pd.DatetimeIndex(dates))
    returns_series = returns_series[~np.isnan(returns_series)]
    
    if returns_series.empty:
        return pd.DataFrame()
        
    df = pd.DataFrame({'return': returns_series,
                       'year': returns_series.index.year,
                       'month': returns_series.index.month})
    monthly_returns = df.groupby(['year', 'month'])['return'].apply(lambda x: np.prod(1 + x) - 1).reset_index()
    pivot_table = monthly_returns.pivot(index='year', columns='month', values='return')
    pivot_table['Year Total'] = pivot_table.apply(lambda row: np.prod(1 + row.dropna()) - 1
                                                   if not row.dropna().empty else np.nan, axis=1)
    formatted_table = pivot_table.apply(lambda col: col.map(lambda x: f"{x*100:.2f}%" if not pd.isna(x) else ""))
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    formatted_table = formatted_table.rename(columns=month_names)
    return formatted_table

# ================================
# Backtest Logic
# ================================

def backtest(days, p, orb_m, target_R, risk, max_Lev, AUM_0, commission):
    """Perform an optimized backtest for the ORB strategy."""
    start_time = time.time()
    str_df = pd.DataFrame()
    str_df['Date'] = days
    str_df['AUM'] = np.nan
    str_df.loc[0, 'AUM'] = AUM_0
    str_df['pnl_R'] = np.nan
    
    # Convert minutes to number of rows (assuming 1-min bars)
    # The user logic assumes 'orb_m' corresponds to row index 'orb_m'. 
    # If data is 1-minute bars, orb_m=5 means index 5 (6th bar? 0..4 is 5 bars).
    # "OHLC[:or_candles]" -> 0 to orb_m-1. 
    # If orb_m=5, it takes 0,1,2,3,4 (5 bars).
    # entry is at 'or_candles' (index 5, which is the 6th bar).
    # This implies we trade AFTER the range is complete.
    
    or_candles = orb_m 
    
    # Group by day for speed
    # p['day'] is datetime
    day_groups = dict(tuple(p.groupby('day')))

    for t in range(1, len(days)):
        current_day = days[t]
        
        if current_day not in day_groups:
            str_df.loc[t, 'pnl_R'] = 0
            str_df.loc[t, 'AUM'] = str_df.loc[t-1, 'AUM']
            continue

        day_data = day_groups[current_day].reset_index(drop=True)
        
        if len(day_data) <= or_candles:
            str_df.loc[t, 'pnl_R'] = 0
            str_df.loc[t, 'AUM'] = str_df.loc[t-1, 'AUM']
            continue

        # Extract numpy array for speed: open, high, low, close
        OHLC = day_data[['open', 'high', 'low', 'close']].values
        
        # Split Adjustment
        # If data is consistent, split_adj should be 1. 
        # But we calculate it just in case dOpen (daily) differs from OHLC[0,0] (intraday first open) significantly due to data source diffs.
        # Here we derived dOpen from the same data, so it should be OHLC[0,0].
        # split_adj = OHLC[0, 0] / day_data['dOpen'].iloc[0] 
        # Actually, split_adj is 1.0 for us.
        split_adj = 1.0
        
        # Determine Side
        # Range Candle Close (index or_candles-1) vs Day Open (index 0)
        # "np.sign(OHLC[or_candles-1, 3] - OHLC[0, 0])"
        # 3 is Close, 0 is Open.
        # So if Close of 5th min > Open of 1st min -> Long
        side = np.sign(OHLC[or_candles-1, 3] - OHLC[0, 0])
        
        # Entry Price = Open of the NEXT bar (index or_candles)
        entry = OHLC[or_candles, 0] if len(OHLC) > or_candles else np.nan

        if side == 1:
            # Stop = (Entry - Min(Low during OR)) / Entry
            # "abs(np.min(OHLC[:or_candles, 2]) / entry - 1)"
            # Note: The user logic calculates stop % distance based on OR Low relative to ENTRY price.
            stop = abs(np.min(OHLC[:or_candles, 2]) / entry - 1)
        elif side == -1:
            # Stop = (Max(High during OR) - Entry) / Entry
            stop = abs(np.max(OHLC[:or_candles, 1]) / entry - 1)
        else:
            stop = np.nan

        if side == 0 or math.isnan(stop) or math.isnan(entry):
            str_df.loc[t, 'pnl_R'] = 0
            str_df.loc[t, 'AUM'] = str_df.loc[t-1, 'AUM']
            continue

        if entry == 0 or stop == 0:
            shares = 0
        else:
            # Position Sizing
            risk_amt = str_df.loc[t-1, 'AUM'] * risk
            # Shares = Risk / (Entry * StopPct)
            # Entry * StopPct = Distance in dollars
            
            shares_risk = risk_amt / (entry * stop)
            shares_lev = (max_Lev * str_df.loc[t-1, 'AUM']) / entry
            
            shares = math.floor(min(shares_risk, shares_lev))

        if shares == 0:
            str_df.loc[t, 'pnl_R'] = 0
            str_df.loc[t, 'AUM'] = str_df.loc[t-1, 'AUM']
            continue

        # Post Entry Data
        OHLC_post_entry = OHLC[or_candles:, :]

        PnL_T = 0
        
        if side == 1:  # Long trade
            stop_price = entry * (1 - stop)
            target_price = entry * (1 + target_R * stop) if np.isfinite(target_R) else float('inf')
            
            # Check hits
            # 2 is Low, 1 is High, 3 is Close
            stop_hits = OHLC_post_entry[:, 2] <= stop_price
            target_hits = OHLC_post_entry[:, 1] > target_price

            if np.any(stop_hits) and np.any(target_hits):
                idx_stop = np.argmax(stop_hits)
                idx_target = np.argmax(target_hits)
                if idx_target < idx_stop:
                    # Hit target first
                    # Exit at Target Price? Or High? Usually Target Price.
                    # User logic: "max(target_price, OHLC_post_entry[idx_target, 0]) - entry"
                    # Wait, OHLC_post_entry[idx_target, 0] is OPEN of that bar.
                    # It assumes gap up?
                    PnL_T = max(target_price, OHLC_post_entry[idx_target, 0]) - entry
                else:
                    # Hit stop first
                    PnL_T = min(stop_price, OHLC_post_entry[idx_stop, 0]) - entry
            elif np.any(stop_hits):
                idx_stop = np.argmax(stop_hits)
                PnL_T = min(stop_price, OHLC_post_entry[idx_stop, 0]) - entry
            elif np.any(target_hits):
                idx_target = np.argmax(target_hits)
                PnL_T = max(target_price, OHLC_post_entry[idx_target, 0]) - entry
            else:
                # MOC
                PnL_T = OHLC_post_entry[-1, 3] - entry
                
        elif side == -1:  # Short trade
            stop_price = entry * (1 + stop)
            target_price = entry * (1 - target_R * stop) if np.isfinite(target_R) else 0
            
            stop_hits = OHLC_post_entry[:, 1] >= stop_price
            target_hits = OHLC_post_entry[:, 2] < target_price

            if np.any(stop_hits) and np.any(target_hits):
                idx_stop = np.argmax(stop_hits)
                idx_target = np.argmax(target_hits)
                if idx_target < idx_stop:
                    PnL_T = entry - min(target_price, OHLC_post_entry[idx_target, 0])
                else:
                    PnL_T = entry - max(stop_price, OHLC_post_entry[idx_stop, 0])
            elif np.any(stop_hits):
                idx_stop = np.argmax(stop_hits)
                PnL_T = entry - max(stop_price, OHLC_post_entry[idx_stop, 0])
            elif np.any(target_hits):
                idx_target = np.argmax(target_hits)
                PnL_T = entry - min(target_price, OHLC_post_entry[idx_target, 0])
            else:
                PnL_T = entry - OHLC_post_entry[-1, 3]

        str_df.loc[t, 'AUM'] = str_df.loc[t-1, 'AUM'] + shares * PnL_T - shares * commission * 2
        
        # Avoid division by zero
        if str_df.loc[t-1, 'AUM'] > 0:
            str_df.loc[t, 'pnl_R'] = (str_df.loc[t, 'AUM'] - str_df.loc[t-1, 'AUM']) / (risk * str_df.loc[t-1, 'AUM'])
        else:
            str_df.loc[t, 'pnl_R'] = 0

    end_time = time.time()
    print(f"******** Optimized Backtest Completed in {round(end_time - start_time, 2)} seconds! ********")
    print(f"Starting AUM: ${AUM_0:,.2f}")
    print(f"Final AUM: ${str_df['AUM'].iloc[-1]:,.2f}")
    print(f"Total Return: {(str_df['AUM'].iloc[-1]/AUM_0 - 1)*100:.4f}%")
    return str_df

def plot_equity_curve(str_df, AUM_0, orb_m, target_R, ticker, output_dir):
    """Plot the equity curve with weekly resampling and highlight out-of-sample period."""
    fig, ax = plt.subplots(figsize=(12, 7))
    df_plot = str_df.copy()
    if 'Date' in df_plot.columns:
        df_plot = df_plot.set_index('Date')
    try:
        weekly_data = df_plot['AUM'].resample('W').last().dropna()
    except Exception as e:
        print("Resampling failed, using original data.", e)
        weekly_data = df_plot['AUM'].dropna()

    p1, = ax.plot(weekly_data.index, weekly_data.values, 'r-', linewidth=2, label='Equity')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=90)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.grid(True, linestyle=':')

    min_val = weekly_data.min() if not weekly_data.empty else AUM_0
    max_val = weekly_data.max() if not weekly_data.empty else AUM_0
    ax.set_ylim([0.9 * min_val, 1.25 * max_val])

    target_str = f"Target {target_R}R" if np.isfinite(target_R) else "No Target"
    ax.set_title(f"{orb_m}m-ORB - Stop @ OR High/Low - {target_str}\nFull Period - Ticker = {ticker}", fontsize=12)

    ax.set_yscale('log')
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=None))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/equity_{ticker}.png")
    print(f"Saved equity curve to {output_dir}/equity_{ticker}.png")
    plt.close()

def run_analysis(ticker, filepath):
    print(f"\nProcessing {ticker}...")
    
    # Parameters
    orb_m       = 5             # Opening Range (minutes)
    target_R    = float('inf')  # Profit target (use inf for no target)
    commission  = 0.0005        # Commission per share
    risk        = 0.01          # Equity risk per trade (1% of AUM)
    max_Lev     = 4             # Maximum leverage
    AUM_0       = 25000         # Starting capital
    
    # Load & Prep Data
    try:
        p = prepare_backtest_data(filepath)
    except Exception as e:
        print(f"Failed to prepare data for {ticker}: {e}")
        return

    # Days list
    days = sorted(p['day'].unique())
    
    # Backtest
    str_df = backtest(days, p, orb_m, target_R, risk, max_Lev, AUM_0, commission)
    
    # Calculate daily returns for statistics
    str_df['daily_return'] = str_df['AUM'].pct_change()
    
    # Summary Statistics
    print(f"\n--- Performance Summary: {ticker} ---")
    stats = summary_statistics(str_df['daily_return'])
    print(stats.to_string(index=False))
    
    # Monthly Table
    print(f"\n--- Monthly Returns: {ticker} ---")
    monthly = monthly_performance_table(str_df['daily_return'], str_df['Date'])
    print(monthly.to_string())
    
    # Plot
    plot_equity_curve(str_df, AUM_0, orb_m, target_R, ticker, '.')

if __name__ == "__main__":
    # Use data files in the same directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # TQQQ
    tqqq_path = os.path.join(script_dir, 'TQQQ_intraday_2020-2026.parquet')
    if os.path.exists(tqqq_path):
        run_analysis('TQQQ', tqqq_path)
    else:
        print(f"Data file not found: {tqqq_path}")
