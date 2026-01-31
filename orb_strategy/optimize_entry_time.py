import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the directory to path to allow imports if running from root
sys.path.append(os.path.join(os.getcwd(), 'orb_strategy'))

try:
    from orb_backtest import prepare_backtest_data, backtest
except ImportError:
    # Try importing as package
    from orb_strategy.orb_backtest import prepare_backtest_data, backtest

def calculate_profit_factor(str_df):
    """Calculate Profit Factor from AUM changes (assuming 1 trade per day)."""
    # Daily PnL
    str_df['daily_pnl'] = str_df['AUM'].diff()
    
    # Filter for days with trades (pnl != 0)
    # Note: Commission might make a flat trade slightly negative, but essentially:
    # Gross Profit = Sum of positive PnL
    # Gross Loss = Abs(Sum of negative PnL)
    
    profits = str_df[str_df['daily_pnl'] > 0]['daily_pnl'].sum()
    losses = abs(str_df[str_df['daily_pnl'] < 0]['daily_pnl'].sum())
    
    if losses == 0:
        return float('inf') if profits > 0 else 0
        
    return profits / losses

def optimize_entry_time():
    ticker = 'TQQQ'
    filepath = 'data/TQQQ_intraday_2020-2026.parquet'
    
    print(f"Loading data for {ticker}...")
    try:
        p = prepare_backtest_data(filepath)
    except Exception as e:
        print(f"Failed to prepare data: {e}")
        return

    days = sorted(p['day'].unique())
    
    # Parameters (Fixed)
    target_R = float('inf')
    commission = 0.0005
    risk = 0.01
    max_Lev = 4
    AUM_0 = 25000
    
    results = []
    
    print("Running optimization for entry times (1-100 min)...")
    
    for m in range(1, 101):
        if m % 10 == 0:
            print(f"Processing m={m}...")
            
        str_df = backtest(days, p, m, target_R, risk, max_Lev, AUM_0, commission)
        pf = calculate_profit_factor(str_df)
        
        results.append({
            'minute': m,
            'profit_factor': pf
        })
    
    # Convert to DataFrame
    res_df = pd.DataFrame(results)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(res_df['minute'], res_df['profit_factor'], marker='o', markersize=3, linestyle='-', color='b')
    plt.title(f'Profit Factor vs Entry Time (Opening Range Duration)\nTicker: {ticker}')
    plt.xlabel('Opening Range (Minutes)')
    plt.ylabel('Profit Factor')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.minorticks_on()
    
    # Highlight max
    max_row = res_df.loc[res_df['profit_factor'].idxmax()]
    plt.plot(max_row['minute'], max_row['profit_factor'], 'ro', label=f"Max: {max_row['minute']}m (PF: {max_row['profit_factor']:.2f})")
    plt.legend()
    
    output_path = 'orb_strategy/profit_factor_vs_entry_time.png'
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    
    # Print top 5
    print("\nTop 5 Entry Times by Profit Factor:")
    print(res_df.sort_values('profit_factor', ascending=False).head(5).to_string(index=False))

if __name__ == "__main__":
    optimize_entry_time()
