import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import traceback
from tqdm import tqdm

# Add paths
sys.path.append(os.path.join(os.getcwd(), 'orb_strategy'))
sys.path.append(os.path.join(os.getcwd(), 'neurotrader/mcpt'))

try:
    from orb_backtest import load_data
    from vix_analysis import load_vix_data
    from mcpt_validation import calculate_adx_vectorized
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '../neurotrader/mcpt'))
    from orb_strategy.orb_backtest import load_data
    from orb_strategy.vix_analysis import load_vix_data
    from orb_strategy.mcpt_validation import calculate_adx_vectorized

def extract_daily_features(df):
    """
    Extracts relevant intraday features for each day to enable fast vector backtesting.
    """
    print("Extracting daily features...", flush=True)
    df['date'] = df.index.date
    
    def get_day_stats(g):
        if len(g) < 6:
            return pd.Series([np.nan]*12)
            
        open_0 = g['open'].iloc[0]
        close_4 = g['close'].iloc[4] # 5th bar close
        entry_price = g['open'].iloc[5] # 6th bar open
        
        # OR High/Low (First 5 mins)
        or_low = g['low'].iloc[:5].min()
        or_high = g['high'].iloc[:5].max()
        
        # Post Entry Data
        post_low = g['low'].iloc[5:].min()
        post_high = g['high'].iloc[5:].max()
        moc = g['close'].iloc[-1]
        
        # Day Stats for ADX Reconstruction
        day_high = g['high'].max()
        day_low = g['low'].min()
        day_close = g['close'].iloc[-1]
        
        # Breakout Direction
        long_triggered = post_high > or_high
        short_triggered = post_low < or_low
        
        # R-Multiples
        # Long
        stop_dist_l = entry_price - or_low
        if stop_dist_l <= 0: stop_dist_l = entry_price * 0.001
        long_stopped = post_low <= or_low
        long_r = -1.0 if long_stopped else (moc - entry_price) / stop_dist_l
        if not long_triggered: long_r = 0.0
        
        # Short
        stop_dist_s = or_high - entry_price
        if stop_dist_s <= 0: stop_dist_s = entry_price * 0.001
        short_stopped = post_high >= or_high
        short_r = -1.0 if short_stopped else (entry_price - moc) / stop_dist_s
        if not short_triggered: short_r = 0.0
        
        # Direction Logic for Simulation (Bias)
        # 1 = Long, -1 = Short
        direction = np.sign(close_4 - open_0)
        if direction == 0: direction = 1
        
        # Final R for this day based on Direction Bias
        final_r = long_r if direction == 1 else short_r
        
        return pd.Series({
            'Open': open_0,
            'High_Rel': day_high / open_0,
            'Low_Rel': day_low / open_0,
            'Close_Rel': day_close / open_0,
            'Direction': direction,
            'Long_R': long_r,
            'Short_R': short_r,
            'Final_R': final_r
        })

    daily_feats = df.groupby('date').apply(get_day_stats)
    daily_feats.index = pd.to_datetime(daily_feats.index)
    daily_feats = daily_feats.dropna()
    
    # Calculate Gaps
    daily_agg = df.resample('D').agg({'open':'first', 'high':'max', 'low':'min', 'close':'last'}).dropna()
    daily_agg = daily_agg.loc[daily_feats.index] # Align
    
    log_open = np.log(daily_agg['open'].values)
    log_high = np.log(daily_agg['high'].values)
    log_low = np.log(daily_agg['low'].values)
    log_close = np.log(daily_agg['close'].values)
    
    daily_feats['r_o'] = log_open - np.roll(log_close, 1) # Gap
    daily_feats['r_o'].iloc[0] = 0
    daily_feats['r_h'] = log_high - log_open
    daily_feats['r_l'] = log_low - log_open
    daily_feats['r_c'] = log_close - log_open
    
    return daily_feats

def get_signal(adx_series, vix_series, adx_threshold=20, vix_threshold=20, use_adx=True, use_vix=True):
    """
    Vectorized Signal Generation.
    """
    # ADX Filter
    if use_adx:
        adx_ok = adx_series > adx_threshold
    else:
        adx_ok = np.ones_like(adx_series, dtype=bool)
        
    # VIX Filter
    if use_vix:
        vix_low = vix_series <= vix_threshold
        vix_high = vix_series > vix_threshold
        
        # Logic: 
        # VIX <= 20: Long & Short
        # VIX > 20: Short Only
        allow_long = vix_low & adx_ok
        allow_short = (vix_low | vix_high) & adx_ok 
    else:
        allow_long = adx_ok
        allow_short = adx_ok
        
    return allow_long, allow_short

def run_mcpt_v2():
    # 1. Load Data
    tqqq_path = 'data/TQQQ_intraday_2020-2026.parquet'
    vix_path = 'data/VIX_daily.csv'
    
    print("Loading Data...", flush=True)
    if not os.path.exists(tqqq_path):
        print(f"Error: {tqqq_path} not found.", flush=True)
        return
        
    df_intra = load_data(tqqq_path)
    
    # 2. Extract Features
    df_feats = extract_daily_features(df_intra)
    print(f"Features extracted: {len(df_feats)} days", flush=True)
    
    # 3. Load VIX
    print("Loading VIX Data...", flush=True)
    vix_df = load_vix_data(vix_path)
    vix_series = vix_df.set_index('Date')['VIX_Open']
    vix_aligned = vix_series.reindex(df_feats.index).ffill().fillna(0).values
    df_feats['VIX'] = vix_aligned
    
    # 4. Scenarios
    scenarios = [
        {'name': 'ORB Strategy (Base)', 'adx': False, 'vix': False, 'type': 'base'},
        {'name': 'ORB Strategy (Custom)', 'adx': True, 'vix': True, 'type': 'custom'}
    ]
    
    # 5. Real ADX & Performance
    # Reconstruct OHLC for ADX calc
    # LogClose[t] = LogClose[0] + CumSum(Gap + IntradayChange)
    total_change = df_feats['r_o'].values + df_feats['r_c'].values
    log_price = np.zeros(len(df_feats))
    log_price[0] = np.log(100)
    log_price[1:] = log_price[0] + np.cumsum(total_change[1:])
    
    rec_open = np.exp(log_price - df_feats['r_c'].values)
    rec_high = np.exp(np.log(rec_open) + df_feats['r_h'].values)
    rec_low = np.exp(np.log(rec_open) + df_feats['r_l'].values)
    rec_close = np.exp(log_price)
    
    real_adx = calculate_adx_vectorized(rec_high, rec_low, rec_close)
    prev_adx = np.roll(real_adx, 1); prev_adx[0] = 0
    
    print("\n--- Real Profit Factors ---", flush=True)
    results = {s['name']: [] for s in scenarios}
    stats_output = {}
    
    for sc in scenarios:
        allow_l, allow_s = get_signal(prev_adx, df_feats['VIX'].values, use_adx=sc['adx'], use_vix=sc['vix'])
        
        # PnL
        pnl = np.where((df_feats['Direction'] == 1) & allow_l, df_feats['Final_R'],
                       np.where((df_feats['Direction'] == -1) & allow_s, df_feats['Final_R'], 0))
        
        wins = pnl[pnl > 0].sum()
        losses = -pnl[pnl < 0].sum()
        pf = wins / losses if losses != 0 else 0
        sc['real_pf'] = pf
        print(f"{sc['name']}: {pf:.4f}", flush=True)
        
    # 6. Permutations
    n_sims = 10000
    print(f"\n--- Running {n_sims} Permutations ---", flush=True)
    print("Mode: Base -> Shuffled Signal vs Outcomes", flush=True)
    print("Mode: Custom -> Fixed VIX vs Permuted Price/ADX", flush=True)
    
    # Arrays to shuffle
    r_h = df_feats['r_h'].values
    r_l = df_feats['r_l'].values
    r_c = df_feats['r_c'].values
    r_o = df_feats['r_o'].values # Gap
    
    # Outcomes for Direction Shuffling
    long_r_arr = df_feats['Long_R'].values
    short_r_arr = df_feats['Short_R'].values
    
    # VIX is FIXED (Historical Sequence)
    vix_fixed = df_feats['VIX'].values
    
    direction_arr = df_feats['Direction'].values
    
    n = len(df_feats)
    inds = np.arange(1, n) # Keep 0 fixed
    
    for i in tqdm(range(n_sims)):
        # Shuffle 1: Intraday Shapes + Outcomes (Price/Trade Sequence)
        perm1 = np.random.permutation(inds)
        
        # Shuffle 2: Gaps (Independent Sequence)
        perm2 = np.random.permutation(inds)
        
        # Construct Permuted Arrays
        p_r_h = r_h.copy(); p_r_h[1:] = r_h[perm1]
        p_r_l = r_l.copy(); p_r_l[1:] = r_l[perm1]
        p_r_c = r_c.copy(); p_r_c[1:] = r_c[perm1]
        p_r_o = r_o.copy(); p_r_o[1:] = r_o[perm2]
        
        # Outcomes aligned with Price (perm1)
        p_long_r = long_r_arr.copy(); p_long_r[1:] = long_r_arr[perm1]
        p_short_r = short_r_arr.copy(); p_short_r[1:] = short_r_arr[perm1]
        
        # Directions aligned with Price (for Custom Strategy - we trust the signal, test the filters)
        p_dir_aligned = direction_arr.copy(); p_dir_aligned[1:] = direction_arr[perm1]
        
        # Shuffled Direction (for Base Strategy - we test the signal quality against random direction)
        # We shuffle the direction_arr independently to maintain the Long/Short ratio bias of the strategy
        perm_dir = np.random.permutation(inds)
        p_dir_random = direction_arr.copy(); p_dir_random[1:] = direction_arr[perm_dir]
        
        # VIX is FIXED
        p_vix = vix_fixed
        
        # Reconstruct Prices for ADX (Needed for Custom)
        total_chg = p_r_o + p_r_c
        p_log_close = np.zeros(n)
        p_log_close[0] = np.log(100)
        p_log_close[1:] = p_log_close[0] + np.cumsum(total_chg[1:])
        
        # Derive OHLC
        p_log_open = p_log_close - p_r_c
        p_rec_open = np.exp(p_log_open)
        p_rec_high = np.exp(p_log_open + p_r_h)
        p_rec_low = np.exp(p_log_open + p_r_l)
        p_rec_close = np.exp(p_log_close)
        
        # Recalculate ADX
        p_adx = calculate_adx_vectorized(p_rec_high, p_rec_low, p_rec_close)
        p_prev_adx = np.roll(p_adx, 1); p_prev_adx[0] = 0
        
        for sc in scenarios:
            if sc['type'] == 'base':
                # Base Strategy: Test Signal Quality (Random Direction)
                pnl = np.where(p_dir_random == 1, p_long_r,
                               np.where(p_dir_random == -1, p_short_r, 0))
            else:
                # Custom Strategy: Test Filter Quality (Fixed VIX vs Permuted Price)
                allow_l, allow_s = get_signal(p_prev_adx, p_vix, use_adx=sc['adx'], use_vix=sc['vix'])
                pnl = np.where((p_dir_aligned == 1) & allow_l, p_long_r,
                               np.where((p_dir_aligned == -1) & allow_s, p_short_r, 0))
            
            wins = pnl[pnl > 0].sum()
            losses = -pnl[pnl < 0].sum()
            pf = wins / losses if losses != 0 else 0
            results[sc['name']].append(pf)
            
    print(f"\n--- Analysis Completed ---", flush=True)
    
    plt.figure(figsize=(14, 6))
    for i, sc in enumerate(scenarios):
        name = sc['name']
        real_pf = sc['real_pf']
        perm_pfs = np.array(results[name])
        
        if not np.isfinite(perm_pfs).all():
            perm_pfs = np.nan_to_num(perm_pfs, nan=0.0, posinf=999.0, neginf=-999.0)
            
        p_value = (perm_pfs >= real_pf).mean()
        
        print(f"\n{name} Stats:", flush=True)
        print(f"  Real PF: {real_pf:.4f}", flush=True)
        print(f"  Perm Mean: {perm_pfs.mean():.4f}", flush=True)
        print(f"  Perm Std: {perm_pfs.std():.4f}", flush=True)
        print(f"  P-Value: {p_value:.5f}", flush=True)
        
        stats_output[name] = {
            'real_pf': real_pf,
            'perm_mean': perm_pfs.mean(),
            'perm_std': perm_pfs.std(),
            'p_value': p_value
        }
        
        plt.subplot(1, 2, i+1)
        plt.hist(perm_pfs, bins=50, alpha=0.7, color=f'C{i}', density=True)
        plt.axvline(real_pf, color='red', linestyle='--', linewidth=2, label=f'Real PF={real_pf:.2f}')
        plt.title(f"{name}\nP-Val: {p_value:.5f}")
        plt.xlabel("Profit Factor")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig('orb_strategy/mcpt_v2_results.png')
    print("\nSaved to orb_strategy/mcpt_v2_results.png", flush=True)

    # Save stats to file
    with open('orb_strategy/mcpt_v2_stats.txt', 'w') as f:
        for name, stats in stats_output.items():
            f.write(f"{name} Stats:\n")
            f.write(f"  Real PF: {stats['real_pf']:.4f}\n")
            f.write(f"  Perm Mean: {stats['perm_mean']:.4f}\n")
            f.write(f"  Perm Std: {stats['perm_std']:.4f}\n")
            f.write(f"  P-Value: {stats['p_value']:.5f}\n\n")

if __name__ == "__main__":
    print("Script started...", flush=True)
    try:
        run_mcpt_v2()
    except Exception as e:
        traceback.print_exc()
        print(f"Error: {e}", flush=True)