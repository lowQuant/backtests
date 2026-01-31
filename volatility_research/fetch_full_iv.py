import requests
import pandas as pd
import time
from pathlib import Path
from tqdm import tqdm

DOLT_API_URL = "https://www.dolthub.com/api/v1alpha1/dolthub/options/master"
OUTPUT_FILE = Path("volatility_research/data/iv_full.parquet")

def fetch_dates():
    query = "SELECT DISTINCT date FROM volatility_history ORDER BY date"
    try:
        response = requests.get(DOLT_API_URL, params={'q': query})
        response.raise_for_status()
        data = response.json()
        if 'rows' in data:
            return [row['date'] for row in data['rows']]
        return []
    except Exception as e:
        print(f"Error fetching dates: {e}")
        return []

def fetch_date_data(date):
    """Fetch all rows for a specific date using pagination."""
    all_rows = []
    offset = 0
    limit = 1000 # Apparent API limit
    
    while True:
        query = f"SELECT date, act_symbol, iv_current FROM volatility_history WHERE date = '{date}' LIMIT {limit} OFFSET {offset}"
        try:
            response = requests.get(DOLT_API_URL, params={'q': query})
            # Check for rate limit
            if response.status_code == 429:
                print("Rate limit hit, sleeping...")
                time.sleep(5)
                continue
                
            response.raise_for_status()
            data = response.json()
            rows = data.get('rows', [])
            
            if not rows:
                break
                
            all_rows.extend(rows)
            
            if len(rows) < limit:
                break
                
            offset += limit
            # time.sleep(0.05) # Small delay to be polite
            
        except Exception as e:
            print(f"Error fetching {date} offset {offset}: {e}")
            break
            
    return pd.DataFrame(all_rows)

def main():
    if OUTPUT_FILE.exists():
        print(f"File {OUTPUT_FILE} already exists. Overwriting to ensure full data...")
    
    print("Fetching unique dates...")
    dates = fetch_dates()
    print(f"Found {len(dates)} dates with data.")
    
    all_dfs = []
    
    for d in tqdm(dates, desc="Fetching Daily Data"):
        df = fetch_date_data(d)
        if not df.empty:
            all_dfs.append(df)
            
    if not all_dfs:
        print("No data fetched.")
        return

    full_df = pd.concat(all_dfs)
    full_df['date'] = pd.to_datetime(full_df['date'])
    full_df['iv_current'] = pd.to_numeric(full_df['iv_current'], errors='coerce')
    full_df = full_df.dropna()
    
    print(f"Total rows fetched: {len(full_df)}")
    
    # Save
    full_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()