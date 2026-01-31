import json
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Configuration
DATES_FILE = "volatility_research/dates.json"
DATA_PATH = Path("mean_reversion/us_stock_history_10y.csv")
OUTPUT_DIR = Path("volatility_research")
OUTPUT_DIR.mkdir(exist_ok=True)

DOLT_API_URL = "https://www.dolthub.com/api/v1alpha1/dolthub/options/master"

def load_dates():
    with open(DATES_FILE, 'r') as f:
        data = json.load(f)
    dates = [row['date'] for row in data['rows']]
    return sorted(dates)

def fetch_iv_data(date):
    """Fetch IV data for a specific date from DoltHub."""
    query = f"SELECT date, act_symbol, iv_current FROM volatility_history WHERE date = '{date}'"
    params = {'q': query}
    try:
        response = requests.get(DOLT_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        if 'rows' in data:
            return pd.DataFrame(data['rows'])
        else:
            print(f"No rows for {date}: {data}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching {date}: {e}")
        return pd.DataFrame()

def load_price_data():
    if not DATA_PATH.exists():
        print(f"Price data not found at {DATA_PATH}")
        return None
    
    print("Loading price history...")
    df = pd.read_csv(DATA_PATH, index_col=[0, 1], parse_dates=[0])
    df = df.sort_index()
    return df

def main():
    # 1. Select Dates (Monthly)
    dates = load_dates()
    dates_s = pd.to_datetime(dates)
    
    # Resample to get first available date per month
    df_dates = pd.DataFrame({'date': dates}, index=dates_s)
    monthly_dates = df_dates.resample('ME').first().dropna()['date'].tolist() # 'ME' is month end, but we take first avail which is close?
    # Actually resample 'MS' (Month Start) and take nearest?
    # Simpler: Group by Year-Month and take first.
    
    unique_months = sorted(list(set([d[:7] for d in dates])))
    selected_dates = []
    for m in unique_months:
        # Find first date in this month
        for d in dates:
            if d.startswith(m):
                selected_dates.append(d)
                break
                
    print(f"Selected {len(selected_dates)} monthly dates for analysis.")
    
    # 2. Fetch IV Data
    iv_dfs = []
    for d in tqdm(selected_dates, desc="Fetching IV Data"):
        df = fetch_iv_data(d)
        if not df.empty:
            iv_dfs.append(df)
        time.sleep(0.5) # Be nice to API
        
    if not iv_dfs:
        print("No IV data fetched.")
        return

    full_iv_df = pd.concat(iv_dfs)
    full_iv_df['date'] = pd.to_datetime(full_iv_df['date'])
    full_iv_df['iv_current'] = pd.to_numeric(full_iv_df['iv_current'], errors='coerce')
    full_iv_df = full_iv_df.dropna(subset=['iv_current'])
    
    print(f"Fetched {len(full_iv_df)} IV records.")
    
    # 3. Load Price Data & Compute Forward Returns
    price_df = load_price_data()
    if price_df is None: return

    # Ensure price_df index is sorted
    price_df = price_df.sort_index()
    
    # Helper to get forward return
    # We need to map (Date, Symbol) -> Forward Returns
    # Pre-calculating forward returns on the whole price DF might be memory intensive but faster.
    # Let's do it on the whole DF.
    
    print("Calculating forward returns...")
    # Reset index to make operations easier
    price_df_reset = price_df.reset_index()
    price_df_reset['Date'] = pd.to_datetime(price_df_reset['Date']) # Ensure datetime
    
    # Pivot to close prices: Index=Date, Columns=Symbol
    close_prices = price_df_reset.pivot(index='Date', columns='Symbol', values='Close')
    
    # Compute returns
    # 1d, 5d, 20d
    ret_1d = close_prices.pct_change(1).shift(-1) # Forward 1D
    ret_5d = close_prices.pct_change(5).shift(-5)
    ret_20d = close_prices.pct_change(20).shift(-20)
    
    # Stack back to fit merge
    # This might be slow.
    # Alternative: Look up per row in full_iv_df.
    # Since we have ~50 dates * ~3000 stocks = 150k rows, lookups are fine.
    
    records = []
    
    print("Merging IV with Returns...")
    for idx, row in tqdm(full_iv_df.iterrows(), total=len(full_iv_df)):
        d = row['date']
        sym = row['act_symbol']
        iv = row['iv_current']
        
        if d not in close_prices.index:
            # Find closest date? Or just skip?
            # DoltHub dates might differ slightly from trading days?
            # Check nearest forward date
            try:
                loc = close_prices.index.get_indexer([d], method='bfill')[0]
                if loc == -1: continue
                d_trade = close_prices.index[loc]
            except:
                continue
        else:
            d_trade = d
            
        if sym in close_prices.columns:
            r1 = ret_1d.loc[d_trade, sym]
            r5 = ret_5d.loc[d_trade, sym]
            r20 = ret_20d.loc[d_trade, sym]
            
            records.append({
                'Date': d,
                'Symbol': sym,
                'IV': iv,
                'Ret_1D': r1,
                'Ret_5D': r5,
                'Ret_20D': r20
            })
            
    results_df = pd.DataFrame(records).dropna()
    print(f"Analyzable records: {len(results_df)}")
    
    # 4. Quintile Analysis (Time Series)
    # Create Quintiles by Date
    results_df['Quintile'] = results_df.groupby('Date')['IV'].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
    )
    
    # Group by Date and Quintile to get mean return per period
    quintile_returns = results_df.groupby(['Date', 'Quintile'])['Ret_5D'].mean().unstack()
    
    # Cumulative Performance (Compounding the 5-day returns)
    # Note: These are monthly samples of 5-day returns.
    cum_returns = (1 + quintile_returns).cumprod()
    
    # Plot Cumulative Performance (Bloomberg Style)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#00ff00', '#00ccff', '#ffff00', '#ff9900', '#ff0000'] # Neon-ish
    for i in range(5):
        ax.plot(cum_returns.index, cum_returns[i], label=f'Quintile {i} (High IV)' if i==4 else f'Quintile {i}', color=colors[i], linewidth=2)
        
    ax.set_title("Cumulative Performance of 5-Day Forward Returns by IV Quintile", fontsize=16, color='white', fontweight='bold')
    ax.set_ylabel("Cumulative Return (Factor)", color='white')
    ax.set_xlabel("Date", color='white')
    ax.legend(facecolor='black', edgecolor='white')
    ax.grid(True, color='#333333', linestyle='--')
    
    # Save
    plt.savefig(OUTPUT_DIR / "iv_quintile_cumulative_bloomberg.png", facecolor='black')
    print(f"Cumulative plot saved to {OUTPUT_DIR / 'iv_quintile_cumulative_bloomberg.png'}")
    
    # Calculate Spread (Q4 - Q0)
    # Annualized Spread? 
    # We have ~50 periods of 5 days.
    # Simple Spread: Mean(Q4) - Mean(Q0)
    q_means = quintile_returns.mean()
    spread_per_trade = q_means[4] - q_means[0]
    
    # Annualizing: (1 + spread)^(252/5) - 1 ? Or just spread * (252/5)?
    # Let's use simple scaling for the spread concept:
    ann_spread = spread_per_trade * (252/5) 
    
    print("\n" + "="*40)
    print("FACTOR ANALYSIS (5-Day Horizon)")
    print("="*40)
    print(f"Mean Return Q0 (Low IV): {q_means[0]*100:.2f}% per trade")
    print(f"Mean Return Q4 (High IV): {q_means[4]*100:.2f}% per trade")
    print(f"Spread per Trade: {spread_per_trade*100:.2f}%")
    print(f"Annualized Spread (approx): {ann_spread*100:.2f}%")
    
    # 5. Decile Bar Charts (Bloomberg Style) - 1D, 5D, 20D
    results_df['Decile'] = results_df.groupby('Date')['IV'].transform(
        lambda x: pd.qcut(x, 10, labels=False, duplicates='drop')
    )
    decile_summary = results_df.groupby('Decile')[['Ret_1D', 'Ret_5D', 'Ret_20D']].mean()
    
    def plot_decile_bar(data, col_name, title, filename, bar_color='#00ccff', highlight_color='#ff00cc'):
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(data.index, data[col_name] * 100, color=bar_color, edgecolor='white')
        
        # Highlight top decile
        bars[9].set_color(highlight_color) 
        
        ax.set_title(title, fontsize=16, color='white', fontweight='bold')
        ax.set_ylabel("Mean Return (%)", color='white')
        ax.set_xlabel("IV Decile (0=Lowest, 9=Highest)", color='white')
        ax.set_xticks(range(10))
        ax.grid(axis='y', color='#333333', linestyle='--')
        
        # Add labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom', color='white', fontsize=10)

        plt.savefig(OUTPUT_DIR / filename, facecolor='black')
        print(f"Chart saved to {OUTPUT_DIR / filename}")
        plt.close()

    # Generate Plots
    plot_decile_bar(decile_summary, 'Ret_1D', "Mean 1-Day Forward Return by IV Decile", "iv_decile_1d_bloomberg.png", bar_color='#00ff99')
    plot_decile_bar(decile_summary, 'Ret_5D', "Mean 5-Day Forward Return by IV Decile", "iv_decile_5d_bloomberg.png", bar_color='#00ccff')
    plot_decile_bar(decile_summary, 'Ret_20D', "Mean 20-Day Forward Return by IV Decile", "iv_decile_20d_bloomberg.png", bar_color='#bd00ff')

if __name__ == "__main__":
    main()
