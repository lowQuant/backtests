import pandas as pd
import numpy as np
import sys
import os

# Add the directory to path to allow imports if running from root
sys.path.append(os.path.join(os.getcwd(), 'orb_strategy'))

try:
    from orb_backtest import prepare_backtest_data, backtest, summary_statistics
except ImportError:
    from orb_strategy.orb_backtest import prepare_backtest_data, backtest, summary_statistics

def compare_variants():
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
    target_R = float('inf')
    commission = 0.0005
    risk = 0.01
    max_Lev = 4
    AUM_0 = 25000
    
    variants = [3, 5, 14]
    results = {}
    returns_df = pd.DataFrame()
    
    print("\nRunning Backtests...")
    for m in variants:
        print(f"  Running {m}-minute ORB...")
        str_df = backtest(days, p, m, target_R, risk, max_Lev, AUM_0, commission)
        
        # Calculate Daily Returns
        daily_ret = str_df['AUM'].pct_change()
        returns_df[f'{m}m_ORB'] = daily_ret
        
        # Calculate Stats
        stats = summary_statistics(daily_ret)
        
        # Store key metrics
        final_aum = str_df['AUM'].iloc[-1]
        total_ret = (final_aum / AUM_0 - 1) * 100
        
        # Extract specific metrics from the stats DataFrame
        cagr = float(stats.loc[stats['Metric'] == 'CAGR (%)', 'Value'].values[0])
        sharpe = float(stats.loc[stats['Metric'] == 'Sharpe Ratio', 'Value'].values[0])
        mdd = float(stats.loc[stats['Metric'] == 'Max Drawdown (%)', 'Value'].values[0])
        
        results[m] = {
            'Total Return (%)': total_ret,
            'CAGR (%)': cagr,
            'Sharpe Ratio': sharpe,
            'Max Drawdown (%)': mdd
        }

    # Display Performance Comparison
    print(f"\n--- Performance Comparison (TQQQ) ---")
    comp_df = pd.DataFrame(results).T
    comp_df.index.name = 'Variant'
    print(comp_df.to_string(float_format="%.2f"))
    
    # Correlation Analysis
    print(f"\n--- Correlation Matrix of Daily Returns ---")
    corr_matrix = returns_df.corr()
    print(corr_matrix.to_string(float_format="%.4f"))

if __name__ == "__main__":
    compare_variants()
