"""
Merge IV data from DoltHub with US stock history.
Caches the result as us_stocks_with_iv.csv to avoid repeated API calls.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent
STOCK_DATA_PATH = DATA_DIR / "mean_reversion" / "us_stock_history_10y.csv"
IV_DATA_PATH = DATA_DIR / "volatility_research" / "data" / "iv_full.parquet"
OUTPUT_PATH = DATA_DIR / "tom_basso_channels" / "us_stocks_with_iv.csv"


def load_stock_data() -> pd.DataFrame:
    """Load US stock history data."""
    print(f"Loading stock data from {STOCK_DATA_PATH}...")
    df = pd.read_csv(STOCK_DATA_PATH, parse_dates=['Date'])
    print(f"  Loaded {len(df):,} rows, {df['Symbol'].nunique()} symbols")
    return df


def load_iv_data() -> pd.DataFrame:
    """Load IV data from parquet file."""
    if not IV_DATA_PATH.exists():
        print(f"IV data not found at {IV_DATA_PATH}")
        print("Please run volatility_research/fetch_full_iv.py first to download IV data.")
        return None
    
    print(f"Loading IV data from {IV_DATA_PATH}...")
    df = pd.read_parquet(IV_DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    print(f"  Loaded {len(df):,} rows, {df['act_symbol'].nunique()} symbols")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    return df


def merge_data(stock_df: pd.DataFrame, iv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge stock data with IV data.
    Performs a left join on Date and Symbol.
    """
    print("Merging stock and IV data...")
    
    # Rename IV columns for merge
    iv_df = iv_df.rename(columns={
        'date': 'Date',
        'act_symbol': 'Symbol',
        'iv_current': 'IV'
    })
    
    # Merge
    merged = pd.merge(
        stock_df,
        iv_df[['Date', 'Symbol', 'IV']],
        on=['Date', 'Symbol'],
        how='left'
    )
    
    # Fill missing IV with forward fill per symbol
    merged = merged.sort_values(['Symbol', 'Date'])
    merged['IV'] = merged.groupby('Symbol')['IV'].ffill()
    
    iv_coverage = merged['IV'].notna().mean() * 100
    print(f"  Merged: {len(merged):,} rows")
    print(f"  IV coverage: {iv_coverage:.1f}%")
    
    return merged


def main():
    # Check if cached file exists
    if OUTPUT_PATH.exists():
        print(f"Cached file exists: {OUTPUT_PATH}")
        user_input = input("Regenerate? (y/n): ").strip().lower()
        if user_input != 'y':
            print("Using existing cached file.")
            return
    
    # Load data
    stock_df = load_stock_data()
    iv_df = load_iv_data()
    
    if iv_df is None:
        print("Cannot proceed without IV data.")
        return
    
    # Merge
    merged_df = merge_data(stock_df, iv_df)
    
    # Save
    print(f"Saving to {OUTPUT_PATH}...")
    merged_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Done! File size: {OUTPUT_PATH.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
