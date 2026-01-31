import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import sys
import os

# Add the directory to path to allow imports if running from root
sys.path.append(os.path.join(os.getcwd(), 'orb_strategy'))

try:
    from orb_backtest import prepare_backtest_data, backtest, summary_statistics, monthly_performance_table
except ImportError:
    from orb_strategy.orb_backtest import prepare_backtest_data, backtest, summary_statistics, monthly_performance_table

def calculate_profit_factor_from_aum(aum_series):
    daily_pnl = aum_series.diff()
    profits = daily_pnl[daily_pnl > 0].sum()
    losses = abs(daily_pnl[daily_pnl < 0].sum())
    if losses == 0:
        return float('inf') if profits > 0 else 0
    return profits / losses

def plot_comparison(aum_comb, aum_3m, aum_14m, dates, ticker):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create DF for easier plotting
    df = pd.DataFrame({
        'Combined': aum_comb,
        '3m_ORB': aum_3m,
        '14m_ORB': aum_14m
    }, index=pd.DatetimeIndex(dates))
    
    # Resample to weekly for cleaner plot if dense
    # df = df.resample('W').last().dropna()
    
    ax.plot(df.index, df['Combined'], 'k-', linewidth=2.5, label='Combined (50/50)')
    ax.plot(df.index, df['3m_ORB'], 'r--', linewidth=1, alpha=0.7, label='3m ORB')
    ax.plot(df.index, df['14m_ORB'], 'b--', linewidth=1, alpha=0.7, label='14m ORB')
    
    ax.set_title(f'Equity Curve Comparison: Combined vs Individual Variants\nTicker: {ticker} (Daily Rebalanced)', fontsize=14)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend()
    
    output_path = 'orb_strategy/combined_equity.png'
    plt.savefig(output_path)
    print(f"Comparison plot saved to {output_path}")

def run_combined_strategy():
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
    
    print("Running 3-minute ORB...")
    df_3m = backtest(days, p, 3, target_R, risk, max_Lev, AUM_0, commission)
    
    print("Running 14-minute ORB...")
    df_14m = backtest(days, p, 14, target_R, risk, max_Lev, AUM_0, commission)
    
    # Calculate Daily Returns
    ret_3m = df_3m['AUM'].pct_change().fillna(0)
    ret_14m = df_14m['AUM'].pct_change().fillna(0)
    
    # Combined Return (50/50 Daily Rebalanced)
    ret_comb = 0.5 * ret_3m + 0.5 * ret_14m
    
    # Reconstruct AUM
    aum_comb_series = [AUM_0]
    for r in ret_comb.iloc[1:]:
        aum_comb_series.append(aum_comb_series[-1] * (1 + r))
        
    df_comb = pd.DataFrame({
        'Date': df_3m['Date'],
        'AUM': aum_comb_series,
        'daily_return': ret_comb
    })
    
    # --- Statistics ---
    print(f"\n--- Performance Summary: Combined (50/50) ---")
    stats = summary_statistics(df_comb['daily_return'])
    
    # Add Profit Factor to Stats
    pf = calculate_profit_factor_from_aum(df_comb['AUM'])
    
    # Print custom table
    print(stats.to_string(index=False))
    print(f"Profit Factor   {pf:.4f}")
    
    # Monthly Table
    print(f"\n--- Monthly Returns: Combined ---")
    monthly = monthly_performance_table(df_comb['daily_return'], df_comb['Date'])
    print(monthly.to_string())
    
    # Plot
    plot_comparison(df_comb['AUM'], df_3m['AUM'], df_14m['AUM'], df_comb['Date'], ticker)

if __name__ == "__main__":
    run_combined_strategy()
