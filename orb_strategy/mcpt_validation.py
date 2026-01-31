import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm

# Add paths
sys.path.append(os.path.join(os.getcwd(), 'orb_strategy'))
sys.path.append(os.path.join(os.getcwd(), 'neurotrader/mcpt'))

try:
    from orb_backtest import load_data
    from bar_permute import get_permutation
    from vix_analysis import load_vix_data
except ImportError:
    # Fallback if run directly
    sys.path.append(os.path.join(os.getcwd(), '../neurotrader/mcpt'))
    from orb_strategy.orb_backtest import load_data
    from orb_strategy.vix_analysis import load_vix_data
    from neurotrader.mcpt.bar_permute import get_permutation

def clean_ohlc(df):
    """
    Ensure dataframe has exactly one column for open, high, low, close.
    Handles duplicate column names by taking the first one.
    """
    data = {}
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            series = df[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0] # Take first column
            data[col] = series
    return pd.DataFrame(data, index=df.index)

def calculate_adx_vectorized(high, low, close, period=14):
    """
    Vectorized ADX calculation for 1D numpy arrays.
    Returns ADX array.
    """
    # Ensure inputs are 1D numpy arrays
    high = np.asarray(high).flatten()
    low = np.asarray(low).flatten()
    close = np.asarray(close).flatten()
    
    n = len(close)
    
    # TR
    # prev_close = np.roll(close, 1) # Shift 1
    # prev_close[0] = close[0]
    
    # For speed in simulation, we can use simple pandas apply or just loop if N is small (days=1500)
    # But 10,000 runs requires max speed.
    # Pandas is reasonably fast for 1500 rows.
    
    df = pd.DataFrame({'high': high, 'low': low, 'close': close})
    df['prev_close'] = df['close'].shift(1)
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    
    tr = np.maximum(df['high'] - df['low'], 
           np.maximum(np.abs(df['high'] - df['prev_close']), 
                      np.abs(df['low'] - df['prev_close'])))
    
    plus_dm = np.where((df['high'] - df['prev_high']) > (df['prev_low'] - df['low']),
                       np.maximum(df['high'] - df['prev_high'], 0), 0)
    
    minus_dm = np.where((df['prev_low'] - df['low']) > (df['high'] - df['prev_high']),
                        np.maximum(df['prev_low'] - df['low'], 0), 0)
    
    # Smoothing (EWM alpha = 1/period)
    alpha = 1 / period
    
    tr_s = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean()
    
    plus_di = 100 * (plus_dm_s / tr_s)
    minus_di = 100 * (minus_dm_s / tr_s)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    
    return adx.values

def calculate_atr_vectorized(high, low, close, period=14):
    """
    Vectorized ATR calculation.
    """
    # Ensure inputs are 1D numpy arrays
    high = np.asarray(high).flatten()
    low = np.asarray(low).flatten()
    close = np.asarray(close).flatten()
    
    df = pd.DataFrame({'high': high, 'low': low, 'close': close})
    df['prev_close'] = df['close'].shift(1)
    
    tr = np.maximum(df['high'] - df['low'], 
           np.maximum(np.abs(df['high'] - df['prev_close']), 
                      np.abs(df['low'] - df['prev_close'])))
    
    # ATR is SMA or EMA? Standard is usually Wilder (alpha=1/n)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr.values

def fast_daily_backtest(ohlc_df, vix_arr, commission=0.0005):
    """
    Approximates the ORB strategy using Daily Bars with Causal Logic.
    Entry: Open +/- 0.2 * ATR
    Exit: Close
    """
    # Force column uniqueness just in case
    ohlc_df = ohlc_df.loc[:, ~ohlc_df.columns.duplicated()]
    
    high = ohlc_df['high'].values
    low = ohlc_df['low'].values
    close = ohlc_df['close'].values
    open_p = ohlc_df['open'].values
    
    # Enforce 1D
    if high.ndim > 1: high = high[:, 0]
    if low.ndim > 1: low = low[:, 0]
    if close.ndim > 1: close = close[:, 0]
    if open_p.ndim > 1: open_p = open_p[:, 0]
    
    # 1. Indicators
    adx = calculate_adx_vectorized(high, low, close)
    atr = calculate_atr_vectorized(high, low, close)
    
    # Shift indicators (Yesterday's values used for Today's Setup)
    prev_adx = np.roll(adx, 1); prev_adx[0] = 0
    prev_atr = np.roll(atr, 1); prev_atr[0] = 0
    
    # 2. Causal Trade Logic
    # Trigger Distance: 0.2 * ATR (Proxy for 5-min Range)
    trig_dist = 0.2 * prev_atr
    
    long_entry = open_p + trig_dist
    short_entry = open_p - trig_dist
    
    # Check Triggers
    long_filled = high >= long_entry
    short_filled = low <= short_entry
    
    # Calculate PnL (Price Points)
    # Long: Close - Entry
    long_pnl = np.where(long_filled, close - long_entry, 0.0)
    
    # Short: Entry - Close
    short_pnl = np.where(short_filled, short_entry - close, 0.0)
    
    # 3. Apply Filters (Regime V2)
    # Global: ADX > 20
    adx_filter = prev_adx > 20
    
    # VIX Filter
    # VIX <= 20: Long + Short
    # VIX > 20: Short Only
    vix_low = vix_arr <= 20
    vix_high = vix_arr > 20
    
    # Combined PnL
    # Initialize
    total_pnl = np.zeros_like(close)
    
    # VIX <= 20: Add Both (subject to ADX)
    # If ADX good AND VIX Low: Take Longs + Shorts
    mask_low = adx_filter & vix_low
    total_pnl += np.where(mask_low, long_pnl + short_pnl, 0)
    
    # VIX > 20: Add Shorts Only (subject to ADX)
    # If ADX good AND VIX High: Take Shorts Only
    mask_high = adx_filter & vix_high
    total_pnl += np.where(mask_high, short_pnl, 0)
    
    # Note: If both Long and Short trigger on the same day (Whipsaw),
    # long_pnl + short_pnl effectively creates a loss equal to the spread (2*trig_dist)
    # plus the net move. This is a fair proxy for a "Stop and Reverse" or "Failed Breakout".
    
    # Profit Factor
    # We sum PnL. To get PF, we need Gross Win / Gross Loss.
    wins = total_pnl[total_pnl > 0].sum()
    losses = -total_pnl[total_pnl < 0].sum()
    
    if losses == 0:
        return 0.0 if wins == 0 else 999.0
        
    return wins / losses

def fast_get_permutation(ohlc_df, vix_arr=None):
    """
    Vectorized version of get_permutation using cumsum.
    Reconstructs log prices.
    If vix_arr is provided, permutes it in tandem with Intraday components (perm1).
    """
    # Convert to log prices
    log_open = np.log(ohlc_df['open'].values)
    log_high = np.log(ohlc_df['high'].values)
    log_low = np.log(ohlc_df['low'].values)
    log_close = np.log(ohlc_df['close'].values)
    
    n = len(ohlc_df)
    
    # Relatives
    prev_close = np.roll(log_close, 1)
    r_o = log_open - prev_close
    r_o[0] = 0 
               
    r_h = log_high - log_open
    r_l = log_low - log_open
    r_c = log_close - log_open
    
    # Shuffle
    # We shuffle indices 1 to N-1 (leave 0 fixed to anchor)
    inds = np.arange(1, n)
    
    perm1 = np.random.permutation(inds) # Intraday shuffle
    perm2 = np.random.permutation(inds) # Gap shuffle
    
    # Create new relative arrays
    new_r_h = r_h.copy()
    new_r_l = r_l.copy()
    new_r_c = r_c.copy()
    new_r_o = r_o.copy()
    
    new_r_h[1:] = r_h[perm1]
    new_r_l[1:] = r_l[perm1]
    new_r_c[1:] = r_c[perm1]
    new_r_o[1:] = r_o[perm2]
    
    # Permute VIX if provided
    new_vix = None
    if vix_arr is not None:
        new_vix = vix_arr.copy()
        new_vix[1:] = vix_arr[perm1]
    
    # Reconstruct
    total_change = new_r_o + new_r_c
    
    new_log_close = np.zeros(n)
    new_log_close[0] = log_close[0]
    new_log_close[1:] = log_close[0] + np.cumsum(total_change[1:])
    
    new_log_open = new_log_close - new_r_c
    new_log_high = new_log_open + new_r_h
    new_log_low = new_log_open + new_r_l
    
    # Exp to get prices
    df_new = pd.DataFrame({
        'open': np.exp(new_log_open),
        'high': np.exp(new_log_high),
        'low': np.exp(new_log_low),
        'close': np.exp(new_log_close)
    }, index=ohlc_df.index)
    
    return df_new, new_vix

def fast_daily_backtest(ohlc_df, vix_arr, use_adx=True, use_vix=True):
    """
    Approximates the ORB strategy using Daily Bars with Causal Logic.
    Entry: Open +/- 0.2 * ATR
    Exit: Close
    Flags: use_adx, use_vix
    """
    # Force column uniqueness just in case
    ohlc_df = ohlc_df.loc[:, ~ohlc_df.columns.duplicated()]
    
    high = ohlc_df['high'].values
    low = ohlc_df['low'].values
    close = ohlc_df['close'].values
    open_p = ohlc_df['open'].values
    
    # Enforce 1D
    if high.ndim > 1: high = high[:, 0]
    if low.ndim > 1: low = low[:, 0]
    if close.ndim > 1: close = close[:, 0]
    if open_p.ndim > 1: open_p = open_p[:, 0]
    
    # 1. Indicators
    adx = calculate_adx_vectorized(high, low, close)
    atr = calculate_atr_vectorized(high, low, close)
    
    # Shift indicators (Yesterday's values used for Today's Setup)
    prev_adx = np.roll(adx, 1); prev_adx[0] = 0
    prev_atr = np.roll(atr, 1); prev_atr[0] = 0
    
    # 2. Causal Trade Logic
    trig_dist = 0.2 * prev_atr
    
    long_entry = open_p + trig_dist
    short_entry = open_p - trig_dist
    
    # Check Triggers
    long_filled = high >= long_entry
    short_filled = low <= short_entry
    
    # Calculate PnL (Price Points)
    long_pnl = np.where(long_filled, close - long_entry, 0.0)
    short_pnl = np.where(short_filled, short_entry - close, 0.0)
    
    # 3. Apply Filters
    
    # ADX Filter
    if use_adx:
        adx_mask = prev_adx > 20
    else:
        adx_mask = np.ones_like(close, dtype=bool)
        
    # VIX Filter Logic
    # If use_vix=True: VIX <= 20 (Long/Short), VIX > 20 (Short Only)
    # If use_vix=False: Always Allow Both (subject to ADX)
    
    if use_vix:
        vix_low = vix_arr <= 20
        vix_high = vix_arr > 20
        
        # Combined PnL with VIX Logic
        total_pnl = np.zeros_like(close)
        
        # Low VIX: Long + Short
        mask_low = adx_mask & vix_low
        total_pnl += np.where(mask_low, long_pnl + short_pnl, 0)
        
        # High VIX: Short Only
        mask_high = adx_mask & vix_high
        total_pnl += np.where(mask_high, short_pnl, 0)
        
    else:
        # No VIX Filter: Long + Short always allowed (subject to ADX)
        total_pnl = np.zeros_like(close)
        total_pnl += np.where(adx_mask, long_pnl + short_pnl, 0)

    # Profit Factor
    wins = total_pnl[total_pnl > 0].sum()
    losses = -total_pnl[total_pnl < 0].sum()
    
    if losses == 0:
        return 0.0 if wins == 0 else 999.0
        
    return wins / losses

def run_mcpt_validation():
    # Load Real Data
    tqqq_path = 'data/TQQQ_intraday_2020-2026.parquet'
    vix_path = 'data/VIX_daily.csv'
    
    print("Loading Data...")
    df_intra = load_data(tqqq_path)
    
    # Clean duplicates immediately
    df_intra = clean_ohlc(df_intra)
    
    # Resample to Daily for Permutation
    daily_agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    df_daily = df_intra.resample('D').agg(daily_agg).dropna()
    
    print(f"Data Loaded: {len(df_daily)} daily bars.")
    
    vix_df = load_vix_data(vix_path)
    vix_series = vix_df.set_index('Date')['VIX_Open']
    vix_aligned = vix_series.reindex(df_daily.index).ffill().fillna(0).values
    
    # Scenarios to test
    scenarios = [
        {'name': 'No Filters', 'adx': False, 'vix': False},
        {'name': 'VIX Only', 'adx': False, 'vix': True},
        {'name': 'ADX Only', 'adx': True, 'vix': False},
        {'name': 'Regime V2 (ADX+VIX)', 'adx': True, 'vix': True}
    ]
    
    results = {}
    
    print("\n--- Calculating Real Profit Factors ---")
    for sc in scenarios:
        pf = fast_daily_backtest(df_daily, vix_aligned, use_adx=sc['adx'], use_vix=sc['vix'])
        sc['real_pf'] = pf
        results[sc['name']] = []
        print(f"{sc['name']}: {pf:.4f}")
    
    n_sims = 100000 
    
    print(f"\n--- Running {n_sims} Permutations (Vectorized) ---")
    
    for i in tqdm(range(n_sims)):
        # Permute TQQQ and VIX in tandem
        perm_df, perm_vix = fast_get_permutation(df_daily, vix_arr=vix_aligned)
        
        for sc in scenarios:
            pf = fast_daily_backtest(perm_df, perm_vix, use_adx=sc['adx'], use_vix=sc['vix'])
            results[sc['name']].append(pf)
            
    # Analysis & Plotting
    plt.figure(figsize=(12, 10))
    
    for i, sc in enumerate(scenarios):
        name = sc['name']
        real_pf = sc['real_pf']
        perm_pfs = np.array(results[name])
        
        p_value = (perm_pfs >= real_pf).mean()
        
        print(f"\n{name} Analysis:")
        print(f"  Real PF: {real_pf:.4f}")
        print(f"  Mean Perm PF: {perm_pfs.mean():.4f}")
        print(f"  P-Value: {p_value:.5f}")
        
        # Subplot 2x2
        plt.subplot(2, 2, i+1)
        plt.hist(perm_pfs, bins=50, alpha=0.7, color=f'C{i}', density=True)
        plt.axvline(real_pf, color='red', linestyle='--', linewidth=2, label=f'Real PF={real_pf:.2f}')
        plt.title(f"{name}\nP-Val: {p_value:.5f}")
        plt.xlabel("Profit Factor")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig('orb_strategy/mcpt_scenarios_comparison.png')
    print("\nSaved chart to orb_strategy/mcpt_scenarios_comparison.png")

if __name__ == "__main__":
    run_mcpt_validation()
