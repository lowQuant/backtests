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

def calculate_adx_series(high, low, close, period=14):
    """
    Calculate ADX on numpy arrays for speed/series.
    """
    n = len(close)
    
    # TR, +DM, -DM
    # prev_close = np.roll(close, 1)
    # prev_close[0] = close[0] # handle first
    
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

def run_adx_analysis():
    ticker = 'TQQQ'
    filepath = 'data/TQQQ_intraday_2020-2026.parquet'
    
    print(f"Loading data for {ticker}...")
    
    # 1. Load Intraday Data
    # prepare_backtest_data calculates Daily Stats but returns Intraday DF
    # We need the full intraday DF with 'day' column
    df = prepare_backtest_data(filepath)
    # Ensure sorted
    df = df.sort_values('caldt').reset_index(drop=True)
    
    print("Calculating Intraday ADX(14)...")
    # This runs on ~500k rows, fast enough
    df['ADX_14'] = calculate_adx_series(df['high'], df['low'], df['close'], period=14)
    
    # 2. Extract ADX at the 4th minute (Index 3) for each day
    # User Requirement: "last 10 minutes of previous day and first 4 minutes of current day"
    # The ADX at Index 3 (4th bar of the day) incorporates exactly this history 
    # (assuming the series is continuous).
    
    # Identify the 4th bar of each day
    # We can use groupby 'day' and take nth(3)
    
    # We want a map: Day -> ADX_at_minute_4
    print("Sampling ADX at 4th minute of each day...")
    
    # Group and pick 4th row (iloc[3])
    # Some days might be short?
    def get_4th_adx(g):
        if len(g) > 3:
            return g.iloc[3]['ADX_14']
        return np.nan
        
    day_adx = df.groupby('day').apply(get_4th_adx)
    
    # 3. Run ORB Backtest
    print("Running ORB Backtest...")
    # Pass existing df (p) to backtest if compatible?
    # prepare_backtest_data returns 'merged'. 
    # We already have 'df' which is 'merged'.
    
    days = sorted(df['day'].unique())
    
    # Standard Params
    orb_m = 5
    target_R = float('inf')
    risk = 0.01
    max_Lev = 4
    AUM_0 = 25000
    commission = 0.0005
    
    results = backtest(days, df, orb_m, target_R, risk, max_Lev, AUM_0, commission)
    
    trades = results[results['pnl_R'] != 0].copy()
    print(f"Total Trades: {len(trades)}")
    
    # 4. Merge ADX
    trades['Date'] = pd.to_datetime(trades['Date'])
    # day_adx index is datetime 'day'. trades['Date'] is the day.
    trades['Intraday_ADX'] = trades['Date'].map(day_adx)
    
    trades = trades.dropna(subset=['Intraday_ADX', 'pnl_R'])
    print(f"Trades with valid ADX: {len(trades)}")
    
    # 5. Regression
    X = trades['Intraday_ADX'].values
    y = trades['pnl_R'].values
    
    slope, intercept = np.polyfit(X, y, 1)
    corr_coef = np.corrcoef(X, y)[0, 1]
    
    print("\n--- Regression Results (PnL_R vs Intraday ADX @ Min 4) ---")
    print(f"Slope: {slope:.6f}")
    print(f"Intercept: {intercept:.6f}")
    print(f"Correlation (r): {corr_coef:.6f}")
    print(f"R-squared: {corr_coef**2:.6f}")
    
    # 6. Bucket Analysis
    # Bins: 0-20, 20-30, 30-40, 40+
    trades['ADX_Bin'] = pd.cut(trades['Intraday_ADX'], bins=[0, 20, 25, 30, 40, 100])
    grouped = trades.groupby('ADX_Bin', observed=True)['pnl_R'].agg(['count', 'mean', lambda x: (x > 0).mean()])
    grouped.columns = ['Count', 'Avg PnL(R)', 'Win Rate']
    
    print("\n--- Performance by Intraday ADX Bucket ---")
    print(grouped)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5, label='Trades')
    
    x_range = np.linspace(X.min(), X.max(), 100)
    y_pred = slope * x_range + intercept
    plt.plot(x_range, y_pred, color='red', label=f'Fit: y={slope:.4f}x + {intercept:.4f}')
    
    plt.title(f'Trade Return (R) vs Intraday ADX (Min 4)\nTicker: {ticker}, 5min ORB')
    plt.xlabel('Intraday ADX (14) - at 4th Minute')
    plt.ylabel('Trade PnL (R)')
    plt.legend()
    plt.grid(True)
    plt.savefig('orb_strategy/adx_intraday_regression.png')
    print("Saved plot to orb_strategy/adx_intraday_regression.png")

if __name__ == "__main__":
    run_adx_analysis()

