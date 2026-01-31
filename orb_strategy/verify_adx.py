import pandas as pd
import numpy as np
import sys
import os

# Add paths
sys.path.append(os.path.join(os.getcwd(), 'orb_strategy'))

try:
    from orb_backtest import prepare_backtest_data, load_data
    from adx_analysis import calculate_adx
except ImportError:
    from orb_strategy.orb_backtest import prepare_backtest_data, load_data
    from orb_strategy.adx_analysis import calculate_adx

def verify_alignment():
    ticker = 'TQQQ'
    filepath = 'data/TQQQ_intraday_2020-2026.parquet'
    
    print(f"Loading data for {ticker}...")
    df_intra = load_data(filepath)
    
    # 1. Calculate Daily ADX
    daily_agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }
    daily = df_intra.resample('D').agg(daily_agg).dropna()
    adx_df = calculate_adx(daily, period=14)
    daily['ADX'] = adx_df['adx']
    daily['Prev_ADX'] = daily['ADX'].shift(1)
    
    print("\n--- Sample Daily Data (Last 5 days) ---")
    print(daily[['close', 'ADX', 'Prev_ADX']].tail(5))
    
    # 2. Simulate Trade Usage
    # Let's pick a random date where we know a trade might happen
    test_date = daily.index[-1]
    prev_date = daily.index[-2]
    
    print(f"\n--- Verification for Trade on {test_date.date()} ---")
    print(f"Trade happens intraday on {test_date.date()}.")
    print(f"We need ADX value computed from data strictly BEFORE {test_date.date()}.")
    print(f"This is the ADX value calculated at Close of {prev_date.date()}.")
    
    adx_used = daily.loc[test_date, 'Prev_ADX']
    adx_actual_yesterday = daily.loc[prev_date, 'ADX']
    
    print(f"ADX from Yesterday ({prev_date.date()}): {adx_actual_yesterday:.4f}")
    print(f"ADX assigned to Today's Trade ({test_date.date()}): {adx_used:.4f}")
    
    if abs(adx_used - adx_actual_yesterday) < 1e-9:
        print("SUCCESS: The ADX used for the trade MATCHES yesterday's closing ADX.")
    else:
        print("FAILURE: Mismatch in ADX alignment.")

if __name__ == "__main__":
    verify_alignment()
