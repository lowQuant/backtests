import pandas as pd
import numpy as np

def load_data(filepath):
    print(f"Loading {filepath}...")
    df = pd.read_parquet(filepath)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # Normalize columns
    df.columns = [c.lower() for c in df.columns]
    return df

def calculate_correlation():
    # Load Data
    qqq = load_data('data/QQQ_20200102_20251230.parquet')
    tqqq = load_data('data/TQQQ_intraday_2020-2026.parquet')
    
    # Resample to Daily Returns for standard correlation
    # Using Close prices
    qqq_daily = qqq['close'].resample('D').last().dropna()
    tqqq_daily = tqqq['close'].resample('D').last().dropna()
    
    # Align Data
    combined = pd.DataFrame({
        'QQQ': qqq_daily,
        'TQQQ': tqqq_daily
    }).dropna()
    
    # Calculate Returns
    returns = combined.pct_change().dropna()
    
    # Calculate Correlation
    corr = returns.corr().iloc[0, 1]
    
    print(f"\n--- Correlation Analysis (Daily Returns) ---")
    print(f"Data Points: {len(returns)}")
    print(f"Date Range: {returns.index.min().date()} to {returns.index.max().date()}")
    print(f"Correlation (QQQ vs TQQQ): {corr:.6f}")
    
    # Intraday Correlation (1-minute)
    # This might be memory intensive but let's try a sample or alignment
    print("\nCalculating Intraday (1-min) Correlation...")
    # Resample to 1T to handle missing bars if any, or just inner join
    qqq_intra = qqq['close']
    tqqq_intra = tqqq['close']
    
    intra_combined = pd.DataFrame({'QQQ': qqq_intra, 'TQQQ': tqqq_intra}).dropna()
    intra_returns = intra_combined.pct_change().dropna()
    
    intra_corr = intra_returns.corr().iloc[0, 1]
    print(f"Intraday (1-min) Correlation: {intra_corr:.6f}")

if __name__ == "__main__":
    calculate_correlation()
