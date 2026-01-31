import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Local imports
from orb_backtest import prepare_backtest_data, summary_statistics
from orb_backtest_adx import calculate_adx_series, backtest_with_adx

def get_5min_adx_map(df_1min, period=14):
    """
    Resamples 1-min data to 5-min, calculates ADX(14), 
    and returns a map of {Date -> ADX_of_first_5min_bar}.
    """
    # Resample to 5 min
    # We need a datetime index
    df_5m = df_1min.set_index('caldt').resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    # Calculate ADX
    df_5m['ADX_5m'] = calculate_adx_series(df_5m['high'], df_5m['low'], df_5m['close'], period=period)
    
    # We want the ADX of the first bar of the day (09:30)
    # The 09:30 bar covers 09:30:00 to 09:34:59.
    # Its timestamp in pandas resample (default label='left') is 09:30:00.
    
    # Filter for 09:30 bars
    # Assuming US Eastern time, usually aligned.
    # We'll just take the first bar of each unique date.
    
    df_5m['date_only'] = df_5m.index.date
    
    # Group by date and take the first ADX
    # Note: We need to ensure it's actually the 09:30 bar.
    # Some days might start late? We'll assume the first available 5m bar is the one we want.
    
    first_adx = df_5m.groupby('date_only')['ADX_5m'].first()
    
    # Convert keys to datetime64[ns] normalized to match 'day' column in backtest
    adx_map = {pd.Timestamp(d): val for d, val in first_adx.items()}
    
    return adx_map

def backtest_with_external_adx_map(days, p, orb_m, target_R, risk, max_Lev, AUM_0, commission, adx_map, adx_threshold=20):
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
        
        # Convert numpy.datetime64 to pandas.Timestamp for map lookup
        current_ts = pd.Timestamp(current_day)
        
        if current_ts not in day_groups:
            str_df.loc[t, 'pnl_R'] = 0
            str_df.loc[t, 'AUM'] = str_df.loc[t-1, 'AUM']
            continue

        day_data = day_groups[current_ts].reset_index(drop=True)
        
        if len(day_data) <= or_candles:
            str_df.loc[t, 'pnl_R'] = 0
            str_df.loc[t, 'AUM'] = str_df.loc[t-1, 'AUM']
            continue

        # Check ADX Filter from Map
        current_adx = adx_map.get(current_ts, 0)
        
        # Debug print for first few days
        if t < 5:
            print(f"Day: {current_ts}, ADX from map: {current_adx}")
            
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
                PnL_T = OHLC_post_entry[-1, 3] - entry 
                
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

    return str_df

def run_comparison():
    ticker = 'TQQQ'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, 'TQQQ_intraday_2020-2026.parquet')
    
    print(f"Loading data for {ticker}...")
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return

    df = prepare_backtest_data(filepath)
    df = df.sort_values('caldt').reset_index(drop=True)
    
    days = sorted(df['day'].unique())
    
    # Params
    orb_m = 5
    target_R = float('inf')
    risk = 0.01
    max_Lev = 4
    AUM_0 = 25000
    commission = 0.0005
    
    # Debug Date Types
    print(f"Type of days elements: {type(days[0])}")
    
    # --- Variant A: 1-min ADX (Current) ---
    print("\n--- Running Variant A: 1-min ADX (Current) ---")
    print("Calculating Intraday ADX(14) on 1-min bars...")
    df['ADX_14'] = calculate_adx_series(df['high'], df['low'], df['close'], period=14)
    
    print("Extracting 1-min ADX values at 09:34...")
    adx_map_1min = {}
    day_groups = df.groupby('day')
    for d, group in day_groups:
        # d is typically Timestamp
        ts = pd.Timestamp(d)
        if len(group) > 3:
            adx_map_1min[ts] = group.iloc[3]['ADX_14']
        else:
            adx_map_1min[ts] = 0
            
    print(f"Sample ADX 1min key: {list(adx_map_1min.keys())[0]}, type: {type(list(adx_map_1min.keys())[0])}")
    print(f"Sample ADX 1min val: {list(adx_map_1min.values())[0]}")
            
    res_1min = backtest_with_external_adx_map(days, df, orb_m, target_R, risk, max_Lev, AUM_0, commission, adx_map_1min, adx_threshold=20)
    
    # --- Variant B: 5-min ADX ---
    print("\n--- Running Variant B: 5-min ADX ---")
    print("Resampling to 5-min and calculating ADX(14)...")
    adx_map_5min = get_5min_adx_map(df, period=14)
    
    print(f"Sample ADX 5min key: {list(adx_map_5min.keys())[0]}, type: {type(list(adx_map_5min.keys())[0])}")
    print(f"Sample ADX 5min val: {list(adx_map_5min.values())[0]}")
    
    res_5min = backtest_with_external_adx_map(days, df, orb_m, target_R, risk, max_Lev, AUM_0, commission, adx_map_5min, adx_threshold=20)
    
    # Compare
    res_1min['daily_return'] = res_1min['AUM'].pct_change()
    res_5min['daily_return'] = res_5min['AUM'].pct_change()
    
    stats_1min = summary_statistics(res_1min['daily_return'])
    stats_5min = summary_statistics(res_5min['daily_return'])
    
    print(f"\n--- 1-min ADX Performance ({ticker}) ---")
    print(stats_1min.to_string(index=False))
    
    print(f"\n--- 5-min ADX Performance ({ticker}) ---")
    print(stats_5min.to_string(index=False))
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(res_1min['Date'], res_1min['AUM'], label='1-min ADX (Current)', alpha=0.7)
    plt.plot(res_5min['Date'], res_5min['AUM'], label='5-min ADX', linewidth=2, color='orange')
    plt.yscale('log')
    plt.title(f'5-min ORB Strategy: 1-min ADX vs 5-min ADX\nTicker: {ticker} | ADX>20')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    output_plot = os.path.join(script_dir, 'comparison_adx_timeframe.png')
    plt.savefig(output_plot)
    print(f"Saved comparison chart to {output_plot}")

if __name__ == "__main__":
    import math # ensure math is available
    run_comparison()
