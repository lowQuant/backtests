import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import volatility_research
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from volatility_research.alphalens import alphalens_lite

# Config
IV_DATA_PATH = Path("volatility_research/data/iv_full.parquet")
PRICE_DATA_PATH = Path("mean_reversion/us_stock_history_10y.csv")
OUTPUT_DIR = Path("volatility_research/alphalens")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data():
    print("Loading IV Data...")
    iv_df = pd.read_parquet(IV_DATA_PATH)
    
    # IV Data Cleaning
    iv_df['date'] = pd.to_datetime(iv_df['date']).dt.tz_localize(None)
    iv_df = iv_df.rename(columns={'act_symbol': 'asset', 'iv_current': 'factor'})
    iv_df = iv_df.set_index(['date', 'asset']).sort_index()
    
    print(f"IV Data: {len(iv_df)} rows, {iv_df.index.get_level_values('asset').nunique()} assets.")
    
    print("Loading Price Data...")
    # Price Data has MultiIndex (Date, Symbol)
    price_df_raw = pd.read_csv(PRICE_DATA_PATH, index_col=[0, 1], parse_dates=[0])
    
    # Pivot to (Date, Asset)
    print("Pivoting Price Data...")
    prices = price_df_raw['Close'].unstack()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    
    return iv_df['factor'], prices

def run_analysis():
    factor, prices = load_data()
    
    # Run Custom Alphalens Logic
    print("Running Analysis (Custom Implementation)...")
    
    # Use alphalens_lite to get merged data
    merged_data = alphalens_lite.get_clean_factor_and_forward_returns(
        factor, 
        prices, 
        periods=(1, 5, 10), 
        quantiles=5
    )
    
    print(f"Analyzable Data: {len(merged_data)} rows.")

    # Generate Stats
    mean_ret, spread = alphalens_lite.create_performance_tearsheet(merged_data, factor_name="IV")
    
    # Save Stats
    with open(OUTPUT_DIR / "full_iv_stats.txt", "w") as f:
        f.write("MEAN RETURN BY QUANTILE (bps):\n")
        f.write((mean_ret * 10000).to_string())
        f.write("\n\nSPREAD (Q5 - Q1) (bps):\n")
        f.write((spread * 10000).to_string())

    # Plot Cumulative
    for p in ['1D', '5D', '10D']:
        alphalens_lite.plot_cumulative_returns(
            merged_data, 
            period=p, 
            output_path=OUTPUT_DIR / f"full_iv_cumulative_{p}.png"
        )

if __name__ == "__main__":
    run_analysis()
