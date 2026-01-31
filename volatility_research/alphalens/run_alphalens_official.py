import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import alphalens

# Add parent directory to path to import iv_analysis
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from volatility_research import iv_analysis

# Config
CACHE_FILE = Path("volatility_research/iv_cache.csv")
OUTPUT_DIR = Path("volatility_research/alphalens")
OUTPUT_DIR.mkdir(exist_ok=True)

def get_iv_data():
    if CACHE_FILE.exists():
        print(f"Loading IV data from cache: {CACHE_FILE}")
        df = pd.read_csv(CACHE_FILE, parse_dates=['date'])
        return df
    print("Error: IV Cache file not found. Please run the previous script to generate cache.")
    return pd.DataFrame()

def main():
    # 1. Load Factor (IV)
    iv_df = get_iv_data()
    if iv_df.empty: return

    # Clean Factor Data
    iv_df['date'] = pd.to_datetime(iv_df['date'])
    iv_df['iv_current'] = pd.to_numeric(iv_df['iv_current'], errors='coerce')
    iv_df = iv_df.dropna(subset=['iv_current'])
    
    # Format for Alphalens: MultiIndex (date, asset), Series name="factor"
    # Ensure timezone-naive to match prices usually (or handle timezone matching)
    iv_df['date'] = iv_df['date'].dt.tz_localize(None)
    
    # 2. Load Prices
    price_df = iv_analysis.load_price_data()
    if price_df is None: return
    
    # Price DF: Index=Date, Columns=Asset
    # Current format: MultiIndex (Date, Symbol) with 'Close' column
    print("Reshaping price data...")
    price_df = price_df.reset_index()
    price_df['Date'] = pd.to_datetime(price_df['Date']).dt.tz_localize(None)
    prices_wide = price_df.pivot(index='Date', columns='Symbol', values='Close')
    
    # Fix Frequency: Reindex to ensure valid Business Day frequency
    # Alphalens relies on inferring frequency for shifts
    print("Ensuring Business Day frequency for prices...")
    if not prices_wide.empty:
        full_idx = pd.date_range(start=prices_wide.index.min(), end=prices_wide.index.max(), freq='B')
        prices_wide = prices_wide.reindex(full_idx, method='ffill')
        print(f"Price Index Frequency: {prices_wide.index.freq}")

    # Align Factor Dates to Price Index
    # If factor date is not in prices, move it to the nearest valid trading day (forward or backward)
    # Or simply filter. Let's try to map to nearest valid trading day using searchsorted.
    print("Aligning factor dates to price index...")
    valid_dates = prices_wide.index
    
    # Function to find nearest date
    def get_nearest_date(d):
        if d in valid_dates: return d
        loc = valid_dates.searchsorted(d)
        if loc < len(valid_dates):
            return valid_dates[loc]
        return valid_dates[-1]

    # Map dates
    # This might take a moment if many unique dates
    unique_factor_dates = iv_df['date'].unique()
    date_map = {d: get_nearest_date(d) for d in unique_factor_dates}
    
    iv_df['date'] = iv_df['date'].map(date_map)
    
    # Re-set index after alignment
    iv_df = iv_df.set_index(['date', 'act_symbol'])
    iv_df = iv_df.sort_index()
    factor_data = iv_df['iv_current']
    factor_data.name = 'IV'
    
    # Remove duplicates if mapping caused collisions (same asset, same mapped date)
    # taking the last one or mean
    if factor_data.index.duplicated().any():
        print("Removing duplicate entries after date alignment...")
        factor_data = factor_data.groupby(level=[0, 1]).mean()

    # 3. Ingest and Format Data using Alphalens
    print("Running Alphalens Ingestion...")
    
    # Alphalens expects prices index to cover the factor index. 
    # Our Factor is monthly, Prices are daily. 
    # Alphalens will handle forward returns calculation.
    
    try:
        clean_factor_data = alphalens.utils.get_clean_factor_and_forward_returns(
            factor=factor_data,
            prices=prices_wide,
            quantiles=5,
            periods=(1, 5, 20),
            filter_zscore=None  # Disable zscore filtering if not desired
        )
    except Exception as e:
        print(f"Alphalens Ingestion Error: {e}")
        # Debugging: Print intersection
        common_dates = factor_data.index.levels[0].intersection(prices_wide.index)
        print(f"Common Dates: {len(common_dates)}")
        return

    print(f"Clean Data: {len(clean_factor_data)} rows.")

    # 4. Create Tear Sheet
    print("Generating Tear Sheet...")
    
    # We want to save the plots instead of showing them interactively
    # Alphalens plots are matplotlib based.
    
    # Returns Analysis
    mean_quant_ret, std_quant_daily = alphalens.performance.mean_return_by_quantile(clean_factor_data)
    
    print("\nMEAN RETURN BY QUANTILE (bps):")
    print((mean_quant_ret * 10000).to_string())
    
    # Spread
    mean_ret_spread, std_spread = alphalens.performance.compute_mean_returns_spread(mean_quant_ret, 5, 1, std_quant_daily)
    print("\nSPREAD (Q5 - Q1) (bps):")
    print((mean_ret_spread * 10000).to_string())
    
    # Plotting Cumulative Returns
    # Alphalens creates a big figure. We can extract specific plots.
    
    # Cumulative Returns Plot
    plt.figure(figsize=(12, 8))
    alphalens.plotting.plot_cumulative_returns_by_quantile(mean_quant_ret, period='5D')
    plt.title("Cumulative Return by IV Quantile (5D Period)")
    plt.savefig(OUTPUT_DIR / "alphalens_cumulative_5d.png")
    print(f"Saved 5D cumulative plot to {OUTPUT_DIR / 'alphalens_cumulative_5d.png'}")
    
    plt.figure(figsize=(12, 8))
    alphalens.plotting.plot_cumulative_returns_by_quantile(mean_quant_ret, period='1D')
    plt.title("Cumulative Return by IV Quantile (1D Period)")
    plt.savefig(OUTPUT_DIR / "alphalens_cumulative_1d.png")
    print(f"Saved 1D cumulative plot to {OUTPUT_DIR / 'alphalens_cumulative_1d.png'}")
    
    # Save full text report
    with open(OUTPUT_DIR / "alphalens_stats_official.txt", "w") as f:
        f.write("MEAN RETURN BY QUANTILE (bps):\n")
        f.write((mean_quant_ret * 10000).to_string())
        f.write("\n\nSPREAD (Q5 - Q1) (bps):\n")
        f.write((mean_ret_spread * 10000).to_string())

if __name__ == "__main__":
    main()
