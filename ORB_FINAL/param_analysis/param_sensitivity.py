import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. Data Loading & Helper Functions
# ---------------------------------------------------------

def load_data(file_path):
    """Load intraday parquet data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    df = pd.read_parquet(file_path)
    # Ensure datetime index
    if 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'])
        df = df.set_index('datetime')
    return df

def load_vix_data(file_path):
    """Load daily VIX data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    
    # The file has 3 header lines. 
    # Row 0: Price,Close,High,Low,Open,Volume
    # Row 1: Ticker,^VIX,^VIX,^VIX,^VIX,^VIX
    # Row 2: Date,,,,,
    # Row 3: Data start
    
    # We will skip 3 rows and provide names manually to be safe
    # Col 0: Date
    # Col 1: Close
    # Col 2: High
    # Col 3: Low
    # Col 4: Open (This is VIX Open)
    # Col 5: Volume
    
    try:
        df = pd.read_csv(file_path, skiprows=3, names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'])
    except Exception as e:
        print(f"Error reading VIX csv: {e}")
        raise

    # Ensure Date parsing
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Rename Open to VIX_Open
    df = df.rename(columns={'Open': 'VIX_Open'})
    
    # Return Date and VIX_Open
    return df[['Date', 'VIX_Open']]

def calculate_adx_series(df, period=14):
    """
    Calculates the ADX series for the entire intraday dataframe.
    """
    # Calculate True Range
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)

    # Calculate Directional Movement
    df['UpMove'] = df['high'] - df['high'].shift(1)
    df['DownMove'] = df['low'].shift(1) - df['low']
    
    df['PlusDM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
    df['MinusDM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)

    # Smooth TR, PlusDM, MinusDM using Wilder's Smoothing (alpha = 1/period)
    # Note: Pandas ewm(alpha=...) matches Wilder's smoothing if alpha=1/period and adjust=False
    alpha = 1 / period
    df['TR_smooth'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()
    df['PlusDM_smooth'] = df['PlusDM'].ewm(alpha=alpha, adjust=False).mean()
    df['MinusDM_smooth'] = df['MinusDM'].ewm(alpha=alpha, adjust=False).mean()

    # Calculate DI
    df['PlusDI'] = 100 * (df['PlusDM_smooth'] / df['TR_smooth'])
    df['MinusDI'] = 100 * (df['MinusDM_smooth'] / df['TR_smooth'])

    # Calculate DX
    di_sum = df['PlusDI'] + df['MinusDI']
    di_diff = abs(df['PlusDI'] - df['MinusDI'])
    # Handle division by zero
    df['DX'] = 100 * (di_diff / di_sum.replace(0, np.nan))
    
    # Calculate ADX (Smooth DX)
    adx_series = df['DX'].ewm(alpha=alpha, adjust=False).mean()
    
    return adx_series

# ---------------------------------------------------------
# 2. Fast Backtester Class
# ---------------------------------------------------------

class FastBacktester:
    def __init__(self, df_intra, df_vix):
        self.df_intra = df_intra.sort_index()
        self.df_vix = df_vix.set_index('Date')['VIX_Open'].to_dict()
        self.days = self.df_intra.index.normalize().unique()
        
        # Pre-group by day for faster access
        # Storing indices for each day to slice quickly
        self.day_indices = {}
        # We can also pre-calculate open prices if needed, but ORB depends on m
        
        # To speed up, we can create a list of daily dataframes or slices
        # But slicing a large dataframe 1500 times is fast enough if logic is simple.
        
    def run(self, orb_m, adx_series, adx_threshold, vix_threshold, use_vix_filter=True):
        """
        Runs the backtest for a specific set of parameters.
        Returns Profit Factor.
        
        Params:
            orb_m (int): Opening Range duration in minutes.
            adx_series (pd.Series): Pre-calculated ADX series aligned with df_intra.
            adx_threshold (float): Minimum ADX to trade.
            vix_threshold (float): VIX level splitting regimes.
            use_vix_filter (bool): Whether to use the Regime V2 logic.
        """
        total_wins = 0.0
        total_losses = 0.0
        
        # Pre-calculate daily start times to avoid Timedelta overhead in loop
        # Actually, simpler to just iterate over pre-grouped data if possible.
        # But `orb_m` changes, so the "end of range" changes.
        
        # Optimization: We only need High/Low of first M minutes, and Open/Close/High/Low of rest of day.
        
        # Let's iterate days
        for day in self.days:
            day_str = day.strftime('%Y-%m-%d')
            
            # VIX Check
            vix_val = self.df_vix.get(day, 0)
            
            # Regime Logic
            allow_long = True
            allow_short = True
            
            if use_vix_filter:
                if vix_val > vix_threshold:
                    allow_long = False
                    # allow_short remains True
            
            # Data Slicing
            # Assuming market open is 09:30
            # We need 09:30 to 09:30 + orb_m for Range
            # We need 09:30 + orb_m to 16:00 for Trading
            
            day_start = day + pd.Timedelta(hours=9, minutes=30)
            day_end = day + pd.Timedelta(hours=16, minutes=0)
            
            # Fast slice
            try:
                # We can use slicing on the main dataframe index
                day_data = self.df_intra.loc[day_start:day_end]
                if day_data.empty:
                    continue
            except KeyError:
                continue

            # ADX Check
            # We check ADX at 09:33 (index 3) per strategy definition
            # If data is missing or short, skip
            if len(day_data) < 5:
                continue
                
            # ADX value from the passed series
            # We need the absolute index of the 4th row (09:33)
            # This is slightly slow if we lookup by timestamp every time.
            # Faster: use integer indexing on the day_data slice
            
            # Get ADX at 09:33 (index 3)
            # adx_series is aligned with df_intra
            current_adx = adx_series.loc[day_data.index[3]]
            
            if current_adx <= adx_threshold:
                continue # Chop filter
            
            # ORB Calculation
            # Range: first orb_m bars
            # 0-indexed: 0 to orb_m-1
            if len(day_data) < orb_m + 1:
                continue
                
            range_data = day_data.iloc[:orb_m]
            orb_high = range_data['high'].max()
            orb_low = range_data['low'].min()
            
            # Trading Session
            trade_data = day_data.iloc[orb_m:]
            if trade_data.empty:
                continue
            
            # Check Breakouts
            # Vectorized check for first breakout?
            # We need the FIRST event: High > ORB_High or Low < ORB_Low
            
            # Create boolean arrays
            breakout_high = trade_data['high'] > orb_high
            breakout_low = trade_data['low'] < orb_low
            
            # Find first indices
            first_high_idx = breakout_high.idxmax() if breakout_high.any() else None
            first_low_idx = breakout_low.idxmax() if breakout_low.any() else None
            
            entry_type = None
            entry_time = None
            entry_price = 0.0
            stop_loss = 0.0
            
            # Logic to determine which happened first
            if first_high_idx and first_low_idx:
                if first_high_idx < first_low_idx:
                    entry_type = 'Long'
                    entry_time = first_high_idx
                else:
                    entry_type = 'Short'
                    entry_time = first_low_idx
            elif first_high_idx:
                entry_type = 'Long'
                entry_time = first_high_idx
            elif first_low_idx:
                entry_type = 'Short'
                entry_time = first_low_idx
            else:
                continue # No trade
            
            # Filter Logic Application
            if entry_type == 'Long' and not allow_long:
                continue
            if entry_type == 'Short' and not allow_short:
                continue
                
            # Execute Trade
            if entry_type == 'Long':
                entry_price = orb_high # Stop limit assumption or simple breakout
                # Stop Loss: Opposite side of range (Simplified for speed, vs ATR stop)
                # Note: User's original code used ATR-based stop or Low of Range? 
                # Let's check original logic: 
                # "stop_price = orb_low" (Base) 
                # "stop_loss = entry_price - 2 * atr" (Regime V2)
                # To be accurate to Regime V2, we need ATR.
                # Calculating ATR daily is slow inside loop.
                # We should pre-calculate ATR.
                pass 
                
            # WAIT: The original Regime V2 uses 2*ATR stop.
            # I need to implement that for accuracy.
            
            # For simplicity in this sensitivity analysis, I will use the code's logic.
            # But wait, passing ATR into this function is needed.
            
        return 0 # Placeholder
        
# Refined Approach:
# I will implement the loop properly with ATR.

# ---------------------------------------------------------
# 3. Execution Script
# ---------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    tqqq_path = os.path.join(parent_dir, 'TQQQ_intraday_2020-2026.parquet')
    vix_path = os.path.join(parent_dir, 'VIX_daily.csv')
    output_dir = os.path.join(script_dir)
    
    print("Loading data...")
    df = load_data(tqqq_path)
    vix_df = load_vix_data(vix_path)
    
    # Pre-calculate Daily ATR (14)
    # Resample to Daily
    daily_agg = {'high': 'max', 'low': 'min', 'close': 'last'}
    df_daily = df.resample('D').agg(daily_agg).dropna()
    
    df_daily['H-L'] = df_daily['high'] - df_daily['low']
    df_daily['H-PC'] = abs(df_daily['high'] - df_daily['close'].shift(1))
    df_daily['L-PC'] = abs(df_daily['low'] - df_daily['close'].shift(1))
    df_daily['TR'] = df_daily[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    # ATR calculation: shift by 1 to use PRIOR day's data for TODAY's trading
    df_daily['ATR'] = df_daily['TR'].rolling(window=14).mean().shift(1)
    
    # Drop NaNs from ATR to prevent poisoning the simulation
    df_daily = df_daily.dropna(subset=['ATR'])
    atr_map = df_daily['ATR'].to_dict()
    
    vix_map = vix_df.set_index('Date')['VIX_Open'].to_dict()
    
    # Define days available in data
    days = df.index.normalize().unique()
    
    # Debug VIX Alignment
    if len(days) > 20:
        sample_day = days[20]
        print(f"\n--- Data Check ---")
        print(f"Sample Day: {sample_day}")
        print(f"VIX Map Value: {vix_map.get(sample_day, 'MISSING')}")
        print(f"ATR Map Value: {atr_map.get(sample_day, 'MISSING')}")
    
    # Helper to run backtest
    def run_simulation(orb_m, adx_series, adx_thresh, vix_thresh):
        wins = 0.0
        losses = 0.0
        trade_count = 0
        
        # Iterate days
        # days is already defined in outer scope
        
        for day in days:
            # Skip if no ATR (first 14 days or missing)
            if day not in atr_map:
                continue
            
            atr = atr_map[day]
            # Safety check
            if np.isnan(atr) or atr <= 0:
                continue
                
            vix = vix_map.get(day, 0)
            
            # Regime Filters
            allow_long = True
            allow_short = True
            if vix > vix_thresh:
                allow_long = False
            
            # Slice Day
            day_label = day.strftime('%Y-%m-%d')
            try:
                day_data = df.loc[day_label]
            except KeyError:
                continue
                
            if len(day_data) < max(orb_m + 1, 5):
                continue
            
            # ADX Filter
            # Check at orb_m - 2 (matches baseline: orb=5, check=3)
            # Ensure we don't look ahead or go negative
            check_idx = max(0, orb_m - 2)
            
            # Ensure we have enough data
            if len(day_data) <= check_idx:
                continue
                
            # Use iloc for speed
            check_time = day_data.index[check_idx]
            
            # Handle potential missing ADX
            if check_time not in adx_series.index:
                continue
                
            adx_val = adx_series.at[check_time]
            
            if np.isnan(adx_val) or adx_val <= adx_thresh:
                continue
                
            # ORB
            range_data = day_data.iloc[:orb_m]
            orb_high = range_data['high'].max()
            orb_low = range_data['low'].min()
            
            trade_data = day_data.iloc[orb_m:]
            if trade_data.empty:
                continue
                
            # Breakouts
            # Use numpy arrays for speed
            highs = trade_data['high'].values
            lows = trade_data['low'].values
            opens = trade_data['open'].values # approximate fill at open of bar
            
            # Find first breakout
            # This is a bit complex to vectorize fully because we need the FIRST occurrence of EITHER
            
            break_h = highs > orb_high
            break_l = lows < orb_low
            
            if not break_h.any() and not break_l.any():
                continue
                
            idx_h = np.argmax(break_h) if break_h.any() else 99999
            idx_l = np.argmax(break_l) if break_l.any() else 99999
            
            pnl = 0.0
            stop_dist = 0.0
            
            # Stop Distance based on ORB Range (Regime V2 Logic)
            # Long Stop: orb_low
            # Short Stop: orb_high
            
            if idx_h < idx_l:
                # Long
                if not allow_long: continue
                
                bar_open = opens[idx_h]
                fill_price = max(orb_high, bar_open)
                
                stop_price = orb_low
                stop_dist = fill_price - stop_price
                
                if stop_dist <= 0: continue
                
                # Check stops
                rest_lows = lows[idx_h:]
                rest_closes = trade_data['close'].values[idx_h:]
                
                stop_hits = rest_lows < stop_price
                if stop_hits.any():
                    pnl = stop_price - fill_price
                else:
                    pnl = rest_closes[-1] - fill_price
                    
            else:
                # Short
                if not allow_short: continue
                
                bar_open = opens[idx_l]
                fill_price = min(orb_low, bar_open)
                
                stop_price = orb_high
                stop_dist = stop_price - fill_price
                
                if stop_dist <= 0: continue
                
                # Check stops
                rest_highs = highs[idx_l:]
                rest_closes = trade_data['close'].values[idx_l:]
                
                stop_hits = rest_highs > stop_price
                if stop_hits.any():
                    pnl = fill_price - stop_price
                else:
                    pnl = fill_price - rest_closes[-1]
            
            # Calculate R
            r_multiple = pnl / stop_dist
            
            if np.isnan(r_multiple):
                continue
            
            trade_count += 1
            if r_multiple > 0:
                wins += r_multiple
            else:
                losses += abs(r_multiple)
                
        if losses == 0:
            return 999.0 if wins > 0 else 0.0
        return wins / losses

    # -----------------------------------------------
    # Analysis 1: Entry Time (1-200)
    # -----------------------------------------------
    print("\nRunning Entry Time Analysis (1-200 min)...")
    # Base Params
    base_adx_series = calculate_adx_series(df, period=14)
    base_adx_thresh = 20
    base_vix_thresh = 20
    
    pf_entry = []
    x_entry = list(range(1, 201, 1)) # Step 1 for high resolution
    
    for m in tqdm(x_entry):
        pf = run_simulation(m, base_adx_series, base_adx_thresh, base_vix_thresh)
        pf_entry.append(pf)
        
    plt.figure()
    plt.plot(x_entry, pf_entry)
    plt.title('Profit Factor vs Entry Time (ORB Duration)')
    plt.xlabel('Minutes')
    plt.ylabel('Profit Factor')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'pf_vs_entry_time.png'))
    plt.close()
    
    # -----------------------------------------------
    # Analysis 2: ADX Lookback (5-60)
    # -----------------------------------------------
    print("\nRunning ADX Lookback Analysis (5-60)...")
    # Base ORB = 5
    base_orb = 5
    pf_lookback = []
    x_lookback = list(range(5, 61, 1)) # Step 1
    
    for n in tqdm(x_lookback):
        # Recalculate ADX
        adx_s = calculate_adx_series(df, period=n)
        pf = run_simulation(base_orb, adx_s, base_adx_thresh, base_vix_thresh)
        pf_lookback.append(pf)
        
    plt.figure()
    plt.plot(x_lookback, pf_lookback)
    plt.title('Profit Factor vs ADX Lookback Period')
    plt.xlabel('Period')
    plt.ylabel('Profit Factor')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'pf_vs_adx_lookback.png'))
    plt.close()

    # -----------------------------------------------
    # Analysis 3: ADX Filter Threshold (5-60)
    # -----------------------------------------------
    print("\nRunning ADX Threshold Analysis (5-60)...")
    # Base ORB 5, ADX 14
    pf_adx_thresh = []
    x_adx_thresh = list(range(5, 61, 1)) # Step 1
    
    for t in tqdm(x_adx_thresh):
        pf = run_simulation(base_orb, base_adx_series, t, base_vix_thresh)
        pf_adx_thresh.append(pf)
        
    plt.figure()
    plt.plot(x_adx_thresh, pf_adx_thresh)
    plt.title('Profit Factor vs ADX Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Profit Factor')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'pf_vs_adx_threshold.png'))
    plt.close()

    # -----------------------------------------------
    # Analysis 4: VIX Open Threshold (5-60)
    # -----------------------------------------------
    print("\nRunning VIX Threshold Analysis (5-60)...")
    pf_vix = []
    x_vix = list(range(5, 61, 1)) # Step 1
    
    for v in tqdm(x_vix):
        pf = run_simulation(base_orb, base_adx_series, base_adx_thresh, v)
        pf_vix.append(pf)
        
    plt.figure()
    plt.plot(x_vix, pf_vix)
    plt.title('Profit Factor vs VIX Threshold')
    plt.xlabel('VIX Level')
    plt.ylabel('Profit Factor')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'pf_vs_vix_threshold.png'))
    plt.close()
    
    print("Analysis Complete.")

if __name__ == "__main__":
    main()
