import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import sys
import os
import math

# Add paths
sys.path.append(os.path.join(os.getcwd(), 'orb_strategy'))

try:
    from orb_backtest import prepare_backtest_data, summary_statistics, monthly_performance_table
except ImportError:
    from orb_strategy.orb_backtest import prepare_backtest_data, summary_statistics, monthly_performance_table

def calculate_adx_series(high, low, close, period=14):
    """
    Calculate ADX on numpy arrays for speed/series.
    """
    # Using pandas is safer for ewm
    df = pd.DataFrame({'high': high, 'low': low, 'close': close})
    df['prev_close'] = df['close'].shift(1)
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    
    df['TR'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            np.abs(df['high'] - df['prev_close']),
            np.abs(df['low'] - df['prev_close'])
        )
    )
    
    df['plus_dm'] = np.where(
        (df['high'] - df['prev_high']) > (df['prev_low'] - df['low']),
        np.maximum(df['high'] - df['prev_high'], 0),
        0
    )
    
    df['minus_dm'] = np.where(
        (df['prev_low'] - df['low']) > (df['high'] - df['prev_high']),
        np.maximum(df['prev_low'] - df['low'], 0),
        0
    )
    
    # Smooth
    alpha = 1 / period
    df['tr_smooth'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()
    df['plus_dm_smooth'] = df['plus_dm'].ewm(alpha=alpha, adjust=False).mean()
    df['minus_dm_smooth'] = df['minus_dm'].ewm(alpha=alpha, adjust=False).mean()
    
    # DI
    df['plus_di'] = 100 * (df['plus_dm_smooth'] / df['tr_smooth'])
    df['minus_di'] = 100 * (df['minus_dm_smooth'] / df['tr_smooth'])
    
    # DX
    sum_di = df['plus_di'] + df['minus_di']
    diff_di = np.abs(df['plus_di'] - df['minus_di'])
    df['dx'] = 100 * (diff_di / sum_di)
    
    # ADX
    df['adx'] = df['dx'].ewm(alpha=alpha, adjust=False).mean()
    
    return df['adx'].values

def backtest_with_adx(days, p, orb_m, target_R, risk, max_Lev, AUM_0, commission, adx_threshold=20):
    str_df = pd.DataFrame()
    str_df['Date'] = days
    str_df['AUM'] = np.nan
    str_df.loc[0, 'AUM'] = AUM_0
    str_df['pnl_R'] = np.nan
    
    or_candles = orb_m 
    
    day_groups = dict(tuple(p.groupby('day')))
    
    trade_count = 0
    filtered_count = 0
    
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

        # Check ADX Filter
        # User requested: "last 10 minutes of previous day and first 4 minutes of current day"
        # This corresponds to sampling ADX at index 3 (4th minute, i.e., 09:34 close).
        if len(day_data) > 3:
            current_adx = day_data.loc[3, 'ADX_14']
        else:
            current_adx = 0 # Default low if missing data
            
        # Apply Filter
        if current_adx <= adx_threshold:
            filtered_count += 1
            str_df.loc[t, 'pnl_R'] = 0
            str_df.loc[t, 'AUM'] = str_df.loc[t-1, 'AUM']
            continue
            
        # --- Standard ORB Logic ---
        OHLC = day_data[['open', 'high', 'low', 'close']].values
        
        side = np.sign(OHLC[or_candles-1, 3] - OHLC[0, 0])
        entry = OHLC[or_candles, 0] if len(OHLC) > or_candles else np.nan

        if side == 1:
            stop = abs(np.min(OHLC[:or_candles, 2]) / entry - 1)
        elif side == -1:
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
            risk_amt = str_df.loc[t-1, 'AUM'] * risk
            shares_risk = risk_amt / (entry * stop)
            shares_lev = (max_Lev * str_df.loc[t-1, 'AUM']) / entry
            shares = math.floor(min(shares_risk, shares_lev))

        if shares == 0:
            str_df.loc[t, 'pnl_R'] = 0
            str_df.loc[t, 'AUM'] = str_df.loc[t-1, 'AUM']
            continue

        trade_count += 1
        OHLC_post_entry = OHLC[or_candles:, :]
        PnL_T = 0
        
        if side == 1:  # Long
            stop_price = entry * (1 - stop)
            target_price = entry * (1 + target_R * stop) if np.isfinite(target_R) else float('inf')
            
            stop_hits = OHLC_post_entry[:, 2] <= stop_price
            target_hits = OHLC_post_entry[:, 1] > target_price

            if np.any(stop_hits) and np.any(target_hits):
                idx_stop = np.argmax(stop_hits)
                idx_target = np.argmax(target_hits)
                if idx_target < idx_stop:
                    PnL_T = max(target_price, OHLC_post_entry[idx_target, 0]) - entry
                else:
                    PnL_T = min(stop_price, OHLC_post_entry[idx_stop, 0]) - entry
            elif np.any(stop_hits):
                idx_stop = np.argmax(stop_hits)
                PnL_T = min(stop_price, OHLC_post_entry[idx_stop, 0]) - entry
            elif np.any(target_hits):
                idx_target = np.argmax(target_hits)
                PnL_T = max(target_price, OHLC_post_entry[idx_target, 0]) - entry
            else:
                PnL_T = OHLC_post_entry[-1, 3] - entry # MOC
                
        elif side == -1:  # Short
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
        
        if str_df.loc[t-1, 'AUM'] > 0:
            str_df.loc[t, 'pnl_R'] = (str_df.loc[t, 'AUM'] - str_df.loc[t-1, 'AUM']) / (risk * str_df.loc[t-1, 'AUM'])
        else:
            str_df.loc[t, 'pnl_R'] = 0

    print(f"Trades Taken: {trade_count}, Trades Filtered (ADX <= {adx_threshold}): {filtered_count}")
    return str_df

def run_comparison():
    ticker = 'TQQQ'
    filepath = 'data/TQQQ_intraday_2020-2026.parquet'
    
    print(f"Loading data for {ticker}...")
    df = prepare_backtest_data(filepath)
    df = df.sort_values('caldt').reset_index(drop=True)
    
    print("Calculating Intraday ADX(14)...")
    df['ADX_14'] = calculate_adx_series(df['high'], df['low'], df['close'], period=14)
    
    days = sorted(df['day'].unique())
    
    # Params
    orb_m = 5
    target_R = float('inf')
    risk = 0.01
    max_Lev = 4
    AUM_0 = 25000
    commission = 0.0005
    
    print("\n--- Running Baseline (No Filter) ---")
    res_base = backtest_with_adx(days, df, orb_m, target_R, risk, max_Lev, AUM_0, commission, adx_threshold=-1)
    
    print("\n--- Running ADX Filtered (ADX > 20) ---")
    res_adx = backtest_with_adx(days, df, orb_m, target_R, risk, max_Lev, AUM_0, commission, adx_threshold=20)
    
    # Compare
    res_base['daily_return'] = res_base['AUM'].pct_change()
    res_adx['daily_return'] = res_adx['AUM'].pct_change()
    
    stats_base = summary_statistics(res_base['daily_return'])
    stats_adx = summary_statistics(res_adx['daily_return'])
    
    print(f"\n--- Baseline Performance ({ticker}) ---")
    print(stats_base.to_string(index=False))
    
    print(f"\n--- ADX Filtered Performance ({ticker}) ---")
    print(stats_adx.to_string(index=False))
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(res_base['Date'], res_base['AUM'], label='Baseline (No Filter)', alpha=0.7)
    plt.plot(res_adx['Date'], res_adx['AUM'], label='ADX > 20 Filter', linewidth=2)
    plt.yscale('log')
    plt.title(f'5-min ORB Strategy: Baseline vs ADX Filter\nTicker: {ticker}')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig('orb_strategy/comparison_adx.png')
    print("Saved comparison chart to orb_strategy/comparison_adx.png")

if __name__ == "__main__":
    run_comparison()
