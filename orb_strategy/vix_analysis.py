import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add paths
sys.path.append(os.path.join(os.getcwd(), 'orb_strategy'))

try:
    from orb_backtest import prepare_backtest_data, backtest, load_data
except ImportError:
    from orb_strategy.orb_backtest import prepare_backtest_data, backtest, load_data

def load_vix_data(filepath):
    """
    Load VIX daily data.
    Format has 3 header lines.
    """
    print(f"Loading VIX data from {filepath}...")
    # Skip first 3 lines, use manual column names based on inspection
    df = pd.read_csv(filepath, skiprows=3, names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'])
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # We only need Date and Open
    return df[['Date', 'Open']].rename(columns={'Open': 'VIX_Open'})

def run_vix_analysis():
    ticker = 'TQQQ'
    tqqq_path = 'data/TQQQ_intraday_2020-2026.parquet'
    vix_path = 'data/VIX_daily.csv'
    
    # 1. Load VIX
    vix_df = load_vix_data(vix_path)
    
    # 2. Run ORB Backtest
    print(f"Loading data for {ticker}...")
    p = prepare_backtest_data(tqqq_path)
    
    days = sorted(p['day'].unique())
    
    # Standard Params
    orb_m = 5
    target_R = float('inf')
    risk = 0.01
    max_Lev = 4
    AUM_0 = 25000
    commission = 0.0005
    
    print("Running ORB Backtest...")
    results = backtest(days, p, orb_m, target_R, risk, max_Lev, AUM_0, commission)
    
    trades = results[results['pnl_R'] != 0].copy()
    print(f"Total Trades: {len(trades)}")
    
    # 3. Merge with VIX
    trades['Date'] = pd.to_datetime(trades['Date'])
    
    # Merge on Date
    # VIX Open on Day T is known at Market Open of Day T.
    # So we map VIX_Open of Date T to Trade on Date T.
    
    trades = pd.merge(trades, vix_df, on='Date', how='left')
    
    # Drop missing VIX (if any)
    missing_count = trades['VIX_Open'].isna().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} trades have no VIX data.")
        
    trades = trades.dropna(subset=['VIX_Open', 'pnl_R'])
    print(f"Trades with valid VIX: {len(trades)}")
    
    # 4. Regression
    X = trades['VIX_Open'].values
    y = trades['pnl_R'].values
    
    slope, intercept = np.polyfit(X, y, 1)
    corr_coef = np.corrcoef(X, y)[0, 1]
    
    print("\n--- Regression Results (PnL_R vs VIX Open) ---")
    print(f"Slope: {slope:.6f}")
    print(f"Intercept: {intercept:.6f}")
    print(f"Correlation (r): {corr_coef:.6f}")
    print(f"R-squared: {corr_coef**2:.6f}")
    
    # 5. Bucket Analysis
    # Bins: <15, 15-20, 20-25, 25-30, 30-35, >35
    bins = [0, 15, 20, 25, 30, 35, 100]
    trades['VIX_Bin'] = pd.cut(trades['VIX_Open'], bins=bins)
    
    grouped = trades.groupby('VIX_Bin', observed=True)['pnl_R'].agg(['count', 'mean', lambda x: (x > 0).mean()])
    grouped.columns = ['Count', 'Avg PnL(R)', 'Win Rate']
    
    print("\n--- Performance by VIX Open Bucket ---")
    print(grouped)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5, label='Trades')
    
    x_range = np.linspace(X.min(), X.max(), 100)
    y_pred = slope * x_range + intercept
    plt.plot(x_range, y_pred, color='red', label=f'Fit: y={slope:.4f}x + {intercept:.4f}')
    
    plt.title(f'Trade Return (R) vs VIX Open\nTicker: {ticker}, 5min ORB')
    plt.xlabel('VIX Open')
    plt.ylabel('Trade PnL (R)')
    plt.legend()
    plt.grid(True)
    plt.savefig('orb_strategy/vix_regression.png')
    print("Saved plot to orb_strategy/vix_regression.png")

if __name__ == "__main__":
    run_vix_analysis()
