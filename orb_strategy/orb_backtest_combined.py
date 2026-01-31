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
    from orb_backtest import prepare_backtest_data, summary_statistics
    from vix_analysis import load_vix_data
    from orb_backtest_adx import calculate_adx_series
except ImportError:
    from orb_strategy.orb_backtest import prepare_backtest_data, summary_statistics
    from orb_strategy.vix_analysis import load_vix_data
    from orb_strategy.orb_backtest_adx import calculate_adx_series

def backtest_combined(days, p, orb_m, target_R, risk, max_Lev, AUM_0, commission, 
                     vix_map, vix_threshold=20, adx_threshold=20):
    str_df = pd.DataFrame()
    str_df['Date'] = days
    str_df['AUM'] = np.nan
    str_df.loc[0, 'AUM'] = AUM_0
    str_df['pnl_R'] = np.nan
    
    or_candles = orb_m 
    
    day_groups = dict(tuple(p.groupby('day')))
    
    trade_count = 0
    filtered_count_vix = 0
    filtered_count_adx = 0
    filtered_count_both = 0
    
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

        # ---------------------------
        # FILTERS
        # ---------------------------
        is_filtered = False
        
        # 1. ADX Filter
        # Sample at index 3 (4th minute)
        if len(day_data) > 3:
            current_adx = day_data.loc[3, 'ADX_14']
        else:
            current_adx = 0 
            
        if current_adx <= adx_threshold:
            filtered_count_adx += 1
            is_filtered = True
            
        # 2. VIX Filter
        current_vix = vix_map.get(current_day)
        if current_vix is None or np.isnan(current_vix):
             current_vix = 999 
        
        if current_vix > vix_threshold:
            filtered_count_vix += 1
            if is_filtered:
                filtered_count_both += 1
            is_filtered = True
            
        if is_filtered:
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

    print(f"Trades Taken: {trade_count}")
    print(f"Filter Hits: ADX<=20 ({filtered_count_adx}), VIX>20 ({filtered_count_vix}), Overlap ({filtered_count_both})")
    return str_df

def run_combined_analysis():
    ticker = 'TQQQ'
    filepath = 'data/TQQQ_intraday_2020-2026.parquet'
    vix_path = 'data/VIX_daily.csv'
    
    print(f"Loading data for {ticker}...")
    df = prepare_backtest_data(filepath)
    df = df.sort_values('caldt').reset_index(drop=True)
    
    # Calculate ADX
    print("Calculating Intraday ADX(14)...")
    df['ADX_14'] = calculate_adx_series(df['high'], df['low'], df['close'], period=14)
    
    # Load VIX
    print("Loading VIX Data...")
    vix_df = load_vix_data(vix_path)
    vix_map = vix_df.set_index('Date')['VIX_Open'].to_dict()
    
    days = sorted(df['day'].unique())
    
    # Params
    orb_m = 5
    target_R = float('inf')
    risk = 0.01
    max_Lev = 4
    AUM_0 = 25000
    commission = 0.0005
    
    print("\n--- Running Combined Filter (ADX > 20 AND VIX <= 20) ---")
    res_combined = backtest_combined(days, df, orb_m, target_R, risk, max_Lev, AUM_0, commission, 
                                    vix_map, vix_threshold=20, adx_threshold=20)
    
    print("\n--- Running Baseline (No Filter) ---")
    res_base = backtest_combined(days, df, orb_m, target_R, risk, max_Lev, AUM_0, commission, 
                                vix_map, vix_threshold=999, adx_threshold=-1) # Effectively no filter
    
    # Compare
    res_base['daily_return'] = res_base['AUM'].pct_change()
    res_combined['daily_return'] = res_combined['AUM'].pct_change()
    
    stats_base = summary_statistics(res_base['daily_return'], res_base['pnl_R'])
    stats_combined = summary_statistics(res_combined['daily_return'], res_combined['pnl_R'])
    
    print(f"\n--- Baseline Performance ({ticker}) ---")
    print(stats_base.to_string(index=False))
    
    print(f"\n--- Combined Filter (ADX>20 & VIX<=20) Performance ({ticker}) ---")
    print(stats_combined.to_string(index=False))
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(res_base['Date'], res_base['AUM'], label='Baseline', alpha=0.6, color='gray')
    plt.plot(res_combined['Date'], res_combined['AUM'], label='Combined Filter', linewidth=2, color='green')
    plt.yscale('log')
    plt.title(f'5-min ORB Strategy: Baseline vs Combined Filter\nTicker: {ticker} | ADX>20 & VIX<=20')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig('orb_strategy/comparison_combined.png')
    print("Saved comparison chart to orb_strategy/comparison_combined.png")

if __name__ == "__main__":
    run_combined_analysis()
