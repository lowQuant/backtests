import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

text_intro = """# Implied Volatility (IV) Factor Analysis using Alphalens

This notebook analyzes the predictive power of Implied Volatility (IV) on future stock returns.
We use **Alphalens** to generate a full tear sheet of performance metrics.

## Setup
Ensure you have `alphalens-reloaded` installed:
```bash
pip install alphalens-reloaded
```
"""

code_imports = """import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import alphalens

# Add project root to path to allow importing local modules if needed
sys.path.append(os.path.abspath('../../'))
"""

code_load_data = """# 1. Load Cached IV Data
# We assume the cache file 'volatility_research/iv_cache.csv' exists from previous runs.
# If not, you may need to run the data fetching scripts.

iv_cache_path = '../iv_cache.csv'
if os.path.exists(iv_cache_path):
    print(f"Loading IV data from {iv_cache_path}")
    iv_df = pd.read_csv(iv_cache_path, parse_dates=['date'])
else:
    print("IV Cache not found! Please ensure data is fetched.")
    iv_df = pd.DataFrame()

# Clean IV Data
iv_df['date'] = pd.to_datetime(iv_df['date']).dt.tz_localize(None)
iv_df['iv_current'] = pd.to_numeric(iv_df['iv_current'], errors='coerce')
iv_df = iv_df.dropna(subset=['iv_current'])
iv_df = iv_df.rename(columns={'act_symbol': 'asset', 'iv_current': 'factor'})

# Set Index for Factor
# Alphalens expects a MultiIndex (date, asset)
factor_df = iv_df.set_index(['date', 'asset']).sort_index()
factor = factor_df['factor']

print(f"Loaded {len(factor)} factor records.")
factor.head()
"""

code_load_prices = """# 2. Load Price Data
# We use the provided US Stock History CSV
price_path = '../../mean_reversion/us_stock_history_10y.csv'

print(f"Loading prices from {price_path}...")
price_df_raw = pd.read_csv(price_path, index_col=[0, 1], parse_dates=[0])

# Reshape to (Date, Asset) with Close prices
prices = price_df_raw['Close'].unstack()
prices.index = pd.to_datetime(prices.index).dt.tz_localize(None)

# Ensure Business Day Frequency
# Alphalens requires a recognized frequency to compute forward returns accurately.
prices = prices.asfreq('B', method='ffill')

print(f"Prices Shape: {prices.shape}")
print(f"Price Frequency: {prices.index.freq}")
prices.head()
"""

code_align = """# 3. Align Factor Dates
# Since our IV data might be sparse or on non-business days, we map them to the nearest valid trading day in our price data.

valid_dates = prices.index
unique_factor_dates = factor.index.get_level_values('date').unique()

def get_nearest_date(d):
    if d in valid_dates: return d
    loc = valid_dates.searchsorted(d)
    if loc < len(valid_dates):
        return valid_dates[loc]
    return valid_dates[-1]

date_map = {d: get_nearest_date(d) for d in unique_factor_dates}

# Apply mapping
# We reset index, map, and set index back
factor_reset = factor.reset_index()
factor_reset['date'] = factor_reset['date'].map(date_map)

# Handle collisions (if multiple factor dates map to same trading day) by taking the mean
factor_aligned = factor_reset.groupby(['date', 'asset'])['factor'].mean()

print(f"Aligned Factor Records: {len(factor_aligned)}")
"""

code_run_alphalens = """# 4. Run Alphalens
# We generate the clean factor data and forward returns.

from alphalens.utils import get_clean_factor_and_forward_returns
from alphalens.tears import create_full_tear_sheet

# Define Quantiles and Periods (1D, 5D, 20D)
quantiles = 5
periods = (1, 5, 20)

try:
    factor_data = get_clean_factor_and_forward_returns(
        factor=factor_aligned,
        prices=prices,
        quantiles=quantiles,
        periods=periods,
        filter_zscore=None 
    )
    print("Factor data created successfully!")
except Exception as e:
    print(f"Error creating factor data: {e}")
"""

code_tearsheet = """# 5. Create Full Tear Sheet
if 'factor_data' in locals():
    create_full_tear_sheet(factor_data)
"""

nb.cells = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_code_cell(code_load_data),
    nbf.v4.new_code_cell(code_load_prices),
    nbf.v4.new_code_cell(code_align),
    nbf.v4.new_code_cell(code_run_alphalens),
    nbf.v4.new_code_cell(code_tearsheet)
]

output_path = Path("volatility_research/alphalens/iv_analysis_notebook.ipynb")
with open(output_path, 'w') as f:
    nbf.write(nb, f)

print(f"Notebook created at {output_path}")
