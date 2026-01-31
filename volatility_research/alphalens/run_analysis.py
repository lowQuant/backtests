import sys
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# Add parent directory to path to import iv_analysis
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from volatility_research import iv_analysis
from volatility_research.alphalens import alphalens_lite

# Config
CACHE_FILE = Path("volatility_research/iv_cache.csv")
OUTPUT_DIR = Path("volatility_research/alphalens")
OUTPUT_DIR.mkdir(exist_ok=True)

def get_iv_data():
    if CACHE_FILE.exists():
        print(f"Loading IV data from cache: {CACHE_FILE}")
        df = pd.read_csv(CACHE_FILE, parse_dates=['date'])
        return df
    
    print("Cache not found. Fetching data via iv_analysis logic...")
    dates = iv_analysis.load_dates()
    
    # Filter for monthly dates similar to iv_analysis.py
    # Replicating logic to ensure consistency
    unique_months = sorted(list(set([d[:7] for d in dates])))
    selected_dates = []
    for m in unique_months:
        for d in dates:
            if d.startswith(m):
                selected_dates.append(d)
                break
                
    print(f"Fetching IV for {len(selected_dates)} dates...")
    iv_dfs = []
    for d in tqdm(selected_dates):
        df = iv_analysis.fetch_iv_data(d)
        if not df.empty:
            iv_dfs.append(df)
        time.sleep(0.1)
        
    if not iv_dfs:
        return pd.DataFrame()
        
    full_iv = pd.concat(iv_dfs)
    full_iv.to_csv(CACHE_FILE, index=False)
    print(f"Saved {len(full_iv)} records to cache.")
    return full_iv

def main():
    # 1. Load Factor (IV)
    iv_df = get_iv_data()
    if iv_df.empty:
        print("No IV data available.")
        return

    # Clean Factor Data
    # We need MultiIndex (date, asset)
    iv_df['date'] = pd.to_datetime(iv_df['date'])
    iv_df['iv_current'] = pd.to_numeric(iv_df['iv_current'], errors='coerce')
    iv_df = iv_df.dropna(subset=['iv_current'])
    
    # Rename cols to match expectation
    iv_df = iv_df.rename(columns={'act_symbol': 'asset', 'iv_current': 'factor'})
    iv_df = iv_df.set_index(['date', 'asset'])
    factor_data = iv_df['factor']
    
    # 2. Load Prices
    price_df = iv_analysis.load_price_data()
    if price_df is None:
        print("Price data missing.")
        return
        
    # Price DF needs to be Index=Date, Columns=Asset for alphalens_lite
    # Current format: MultiIndex (Date, Symbol) with 'Close' column
    print("Reshaping price data...")
    price_df = price_df.reset_index()
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    prices_wide = price_df.pivot(index='Date', columns='Symbol', values='Close')
    
    # 3. Run Alphalens Logic
    print("Running Alphalens analysis (Forward Returns & Quantiles)...")
    # User requested 1D and 5D
    periods = (1, 5)
    merged_data = alphalens_lite.get_clean_factor_and_forward_returns(
        factor_data, 
        prices_wide, 
        periods=periods, 
        quantiles=5
    )
    
    print(f"Merged Data: {len(merged_data)} rows.")
    
    # 4. Generate Outputs
    # A. Table with spread (Q5 - Q1)
    mean_ret, spread = alphalens_lite.create_performance_tearsheet(merged_data, factor_name="IV")
    
    # Save stats to file
    stats_file = OUTPUT_DIR / "iv_stats.txt"
    with open(stats_file, "w") as f:
        f.write("MEAN PERIOD RETURN BY IV QUANTILE (bps)\n")
        f.write((mean_ret * 10000).round(2).to_string())
        f.write("\n\nSPREAD (Q5 - Q1) (bps)\n")
        f.write((spread * 10000).round(2).to_string())
    print(f"Stats saved to {stats_file}")
    
    # B. Cumulative Performance Charts
    # 1D Forward Return Cumulative
    alphalens_lite.plot_cumulative_returns(
        merged_data, 
        period='1D', 
        output_path=OUTPUT_DIR / "cumulative_return_1d_quintiles.png"
    )
    
    # 5D Forward Return Cumulative
    alphalens_lite.plot_cumulative_returns(
        merged_data, 
        period='5D', 
        output_path=OUTPUT_DIR / "cumulative_return_5d_quintiles.png"
    )

if __name__ == "__main__":
    main()
