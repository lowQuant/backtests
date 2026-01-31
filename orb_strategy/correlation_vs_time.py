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
    from orb_strategy.orb_backtest import prepare_backtest_data, backtest

def analyze_correlation_vs_baseline():
    ticker = 'TQQQ'
    filepath = 'data/TQQQ_intraday_2020-2026.parquet'
    
    print(f"Loading data for {ticker}...")
    try:
        p = prepare_backtest_data(filepath)
    except Exception as e:
        print(f"Failed to prepare data: {e}")
        return

    days = sorted(p['day'].unique())
    
    # Parameters
    baseline_m = 5
    target_R = float('inf')
    commission = 0.0005
    risk = 0.01
    max_Lev = 4
    AUM_0 = 25000
    
    print(f"Running Baseline ({baseline_m}m ORB)...")
    df_baseline = backtest(days, p, baseline_m, target_R, risk, max_Lev, AUM_0, commission)
    baseline_returns = df_baseline['AUM'].pct_change().fillna(0)
    
    results = []
    
    print("Running optimization for correlations (1-100 min)...")
    
    for m in range(1, 101):
        if m == baseline_m:
            corr = 1.0
        else:
            if m % 10 == 0:
                print(f"Processing m={m}...")
                
            str_df = backtest(days, p, m, target_R, risk, max_Lev, AUM_0, commission)
            returns = str_df['AUM'].pct_change().fillna(0)
            
            # Calculate Correlation with Baseline
            corr = returns.corr(baseline_returns)
        
        results.append({
            'minute': m,
            'correlation': corr
        })
    
    # Convert to DataFrame
    res_df = pd.DataFrame(results)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(res_df['minute'], res_df['correlation'], marker='o', markersize=3, linestyle='-', color='purple')
    plt.title(f'Correlation of Returns vs {baseline_m}-min Baseline\nTicker: {ticker}')
    plt.xlabel('Opening Range (Minutes)')
    plt.ylabel(f'Correlation with {baseline_m}m ORB')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.minorticks_on()
    
    # Highlight specific points
    # Min correlation
    min_row = res_df.loc[res_df['correlation'].idxmin()]
    plt.plot(min_row['minute'], min_row['correlation'], 'ro', label=f"Min: {min_row['minute']}m (Corr: {min_row['correlation']:.2f})")
    
    # Max correlation (excluding self)
    temp_df = res_df[res_df['minute'] != baseline_m]
    if not temp_df.empty:
        max_row = temp_df.loc[temp_df['correlation'].idxmax()]
        plt.plot(max_row['minute'], max_row['correlation'], 'go', label=f"Max: {max_row['minute']}m (Corr: {max_row['correlation']:.2f})")

    plt.legend()
    
    output_path = 'orb_strategy/correlation_vs_5min.png'
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")
    
    # Print interesting stats
    print("\nLowest Correlations with 5-min Baseline:")
    print(res_df.sort_values('correlation').head(5).to_string(index=False))

if __name__ == "__main__":
    analyze_correlation_vs_baseline()
